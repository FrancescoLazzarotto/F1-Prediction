"""Assemble the full feature matrix for a race event."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from f1predict import cache as C
from f1predict.data import fastf1_source as F1
from f1predict.data import jolpica_source as JLP
from f1predict.data import weather_source as WX
from f1predict.features.form import build_form_matrix
from f1predict.features.practice import extract_fp_features
from f1predict.features.qualifying import extract_quali_features

log = logging.getLogger(__name__)

# Columns the race model expects (in this order)
RACE_FEATURE_COLS = [
    "grid_pos",
    "quali_gap_to_pole_s",
    "fp_long_run_pace_s",
    "fp_gap_to_best_s",
    "form_avg_pos",
    "form_avg_pts",
    "form_dnf_rate",
    "circuit_avg_pos",
    "circuit_n_starts",
    "team_avg_pos",
    "weather_temp",
    "weather_wind",
    "weather_rain_prob",
    "reached_q3",
]

# Columns the quali model expects
QUALI_FEATURE_COLS = [
    "fp_best_lap_s",
    "fp_gap_to_best_s",
    "fp_long_run_pace_s",
    "form_avg_pos",
    "form_avg_pts",
    "team_avg_pos",
]


def _impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values with column median (then 0 as last resort)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            median = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(median) else median)
    return df


def build_prediction_matrix(
    cfg: dict,
    year: int,
    round_num: int,
    season_results: pd.DataFrame,
    circuit_info: dict,
    use_actual_quali: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the race feature matrix for prediction.

    Returns:
        (meta_df, feature_df) where meta_df has driver/team info and
        feature_df contains only the RACE_FEATURE_COLS.
    """
    C.setup_fastf1_cache(cfg)

    # ── Entry list ────────────────────────────────────────────────────────────
    entry_list = F1.get_entry_list(year, round_num)
    if entry_list.empty:
        log.warning("Empty entry list for %d R%d", year, round_num)
        return pd.DataFrame(), pd.DataFrame()

    # ── FP features ──────────────────────────────────────────────────────────
    fp_feat = cache_or_compute_fp(cfg, year, round_num)

    # ── Qualifying features ───────────────────────────────────────────────────
    if use_actual_quali:
        try:
            quali_raw = JLP.get_quali_results(year, round_num)
            quali_feat = extract_quali_features(quali_raw) if not quali_raw.empty else pd.DataFrame()
        except Exception as exc:
            log.warning("Could not fetch quali results: %s", exc)
            quali_feat = pd.DataFrame()
    else:
        quali_feat = pd.DataFrame()

    # ── Form features ─────────────────────────────────────────────────────────
    form_feat = build_form_matrix(
        season_results, entry_list, year, round_num,
        circuit_info["circuit_id"], cfg["form"]["recent_races"],
    )

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = _get_weather(cfg, year, round_num, circuit_info)

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = entry_list[["driver_code", "driver_id", "driver_name", "team"]].copy()

    if not fp_feat.empty:
        df = df.merge(fp_feat[["driver_code", "fp_best_lap_s", "fp_gap_to_best_s",
                                "fp_long_run_pace_s"] +
                               [c for c in fp_feat.columns if c.startswith("fp_s")]],
                      on="driver_code", how="left")

    if not quali_feat.empty:
        df = df.merge(
            quali_feat[["driver_code", "grid_pos", "quali_time_s",
                        "quali_gap_to_pole_s", "reached_q2", "reached_q3"]],
            on="driver_code", how="left",
        )

    if not form_feat.empty:
        df = df.merge(
            form_feat[["driver_code", "form_avg_pos", "form_avg_pts",
                       "form_dnf_rate", "circuit_avg_pos", "circuit_n_starts",
                       "team_avg_pos"]],
            on="driver_code", how="left",
        )

    df["weather_temp"] = weather["temperature"]
    df["weather_wind"] = weather["wind_speed"]
    df["weather_rain_prob"] = weather["rain_prob"]

    meta = df[["driver_code", "driver_name", "team"]].copy()
    features = _impute(df, RACE_FEATURE_COLS)[RACE_FEATURE_COLS]
    return meta, features


def build_quali_prediction_matrix(
    cfg: dict,
    year: int,
    round_num: int,
    season_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble feature matrix for qualifying prediction (from FP data).
    Returns (meta_df, feature_df with QUALI_FEATURE_COLS).
    """
    C.setup_fastf1_cache(cfg)

    entry_list = F1.get_entry_list(year, round_num)
    if entry_list.empty:
        return pd.DataFrame(), pd.DataFrame()

    fp_feat = cache_or_compute_fp(cfg, year, round_num)
    form_feat = build_form_matrix(
        season_results, entry_list, year, round_num, "", cfg["form"]["recent_races"]
    )

    df = entry_list[["driver_code", "driver_id", "driver_name", "team"]].copy()
    if not fp_feat.empty:
        df = df.merge(fp_feat, on="driver_code", how="left")
    if not form_feat.empty:
        df = df.merge(
            form_feat[["driver_code", "form_avg_pos", "form_avg_pts", "team_avg_pos"]],
            on="driver_code", how="left",
        )

    meta = df[["driver_code", "driver_name", "team"]].copy()
    features = _impute(df, QUALI_FEATURE_COLS)[QUALI_FEATURE_COLS]
    return meta, features


def build_training_row(
    cfg: dict,
    year: int,
    round_num: int,
    season_results: pd.DataFrame,
    circuit_info: dict,
) -> pd.DataFrame | None:
    """
    Build a training DataFrame for one race (one row per driver).
    Includes all features + targets (race_pos, quali_pos).
    Returns None if essential data is missing.
    """
    C.setup_fastf1_cache(cfg)

    # Check cache first
    cached = C.load_features(cfg, year, round_num, "training")
    if cached is not None:
        return cached

    try:
        # Targets from Jolpica
        race_results = JLP.get_race_results(year, round_num)
        quali_results = JLP.get_quali_results(year, round_num)
        if race_results.empty:
            return None

        # FP features
        fp_feat = cache_or_compute_fp(cfg, year, round_num)

        # Qualifying features as input
        quali_feat = extract_quali_features(quali_results) if not quali_results.empty else pd.DataFrame()

        # Form features
        entry_list = race_results[["driver_id", "driver_code", "driver_name", "team"]].drop_duplicates()
        form_feat = build_form_matrix(
            season_results, entry_list, year, round_num,
            circuit_info["circuit_id"], cfg["form"]["recent_races"],
        )

        # Weather
        weather = _get_weather(cfg, year, round_num, circuit_info)

        # Assemble
        df = race_results[["driver_code", "driver_id", "driver_name", "team",
                            "grid_pos", "race_pos", "constructor_id"]].copy()

        if not fp_feat.empty:
            df = df.merge(fp_feat[["driver_code", "fp_best_lap_s", "fp_gap_to_best_s",
                                   "fp_long_run_pace_s"]], on="driver_code", how="left")

        if not quali_feat.empty:
            df = df.merge(
                quali_feat[["driver_code", "quali_time_s", "quali_gap_to_pole_s",
                            "reached_q2", "reached_q3"]],
                on="driver_code", how="left",
            )

        if not form_feat.empty:
            df = df.merge(
                form_feat[["driver_code", "form_avg_pos", "form_avg_pts",
                           "form_dnf_rate", "circuit_avg_pos", "circuit_n_starts", "team_avg_pos"]],
                on="driver_code", how="left",
            )

        df["weather_temp"] = weather["temperature"]
        df["weather_wind"] = weather["wind_speed"]
        df["weather_rain_prob"] = weather["rain_prob"]
        df["year"] = year
        df["round"] = round_num

        C.save_features(cfg, year, round_num, "training", df)
        return df

    except Exception as exc:
        log.warning("Could not build training row for %d R%d: %s", year, round_num, exc)
        return None


def cache_or_compute_fp(cfg: dict, year: int, round_num: int) -> pd.DataFrame:
    """Load FP features from cache, or compute and cache them."""
    cached = C.load_features(cfg, year, round_num, "fp")
    if cached is not None:
        return cached

    laps = F1.get_fp_laps(year, round_num)
    if laps is None:
        return pd.DataFrame()

    feat = extract_fp_features(laps, cfg["training"]["min_laps_long_run"])
    if not feat.empty:
        C.save_features(cfg, year, round_num, "fp", feat)
    return feat


def _get_weather(cfg: dict, year: int, round_num: int, circuit_info: dict) -> dict:
    """Get weather from cache or fetch and cache."""
    cached = C.load_weather(cfg, year, round_num)
    if cached is not None:
        return cached

    try:
        race_date_str = circuit_info.get("race_date", "")
        if race_date_str:
            race_dt = pd.to_datetime(race_date_str, utc=True).to_pydatetime()
        else:
            race_dt = datetime.utcnow()

        data = WX.get_weather_for_race(circuit_info["lat"], circuit_info["lon"], race_dt)
        C.save_weather(cfg, year, round_num, data)
        return data
    except Exception as exc:
        log.warning("Weather fetch error: %s", exc)
        return {"temperature": 20.0, "humidity": 60.0, "wind_speed": 3.0,
                "cloud_cover": 30.0, "rain_prob": 0.1}
