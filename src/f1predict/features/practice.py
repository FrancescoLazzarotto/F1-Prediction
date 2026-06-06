"""Extract per-driver features from Free Practice sessions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MIN_LONG_RUN_LAPS = 5   # minimum consecutive laps to consider a stint a "long run"


def extract_fp_features(laps: pd.DataFrame, min_laps: int = _MIN_LONG_RUN_LAPS) -> pd.DataFrame:
    """
    Given a FastF1 Laps DataFrame (from FP1/FP2/FP3), return a per-driver
    feature DataFrame with columns:
        driver_code, fp_best_lap_s, fp_gap_to_best_s, fp_long_run_pace_s,
        fp_s1_best_s, fp_s2_best_s, fp_s3_best_s.

    Returns an empty DataFrame if input is empty or malformed.
    """
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    laps = laps.copy()

    # Convert timedelta columns to seconds
    for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        if col in laps.columns:
            laps[f"_{col}_s"] = laps[col].dt.total_seconds()

    if "_LapTime_s" not in laps.columns:
        return pd.DataFrame()

    # ── Best lap per driver ──────────────────────────────────────────────────
    best_laps = (
        laps.groupby("Driver")["_LapTime_s"]
        .min()
        .rename("fp_best_lap_s")
        .reset_index()
        .rename(columns={"Driver": "driver_code"})
    )
    fastest = best_laps["fp_best_lap_s"].min()
    best_laps["fp_gap_to_best_s"] = best_laps["fp_best_lap_s"] - fastest

    # ── Long-run pace ────────────────────────────────────────────────────────
    # Filter to clean laps: not pit-in/pit-out, IsAccurate where available
    clean = laps.copy()
    if "PitOutLap" in clean.columns:
        clean = clean[~clean["PitOutLap"].fillna(False)]
    if "PitInLap" in clean.columns:
        clean = clean[~clean["PitInLap"].fillna(False)]
    if "IsAccurate" in clean.columns:
        clean = clean[clean["IsAccurate"].fillna(True)]
    # Only clear-track laps where possible
    if "TrackStatus" in clean.columns:
        clean = clean[clean["TrackStatus"].astype(str) == "1"]

    long_run_rows = []
    if "Stint" in clean.columns:
        for (driver, stint), group in clean.groupby(["Driver", "Stint"]):
            if len(group) >= min_laps:
                median_s = group["_LapTime_s"].median()
                long_run_rows.append({"driver_code": driver, "fp_long_run_pace_s": median_s})

    if long_run_rows:
        long_run_df = pd.DataFrame(long_run_rows)
        long_run_df = (
            long_run_df.groupby("driver_code")["fp_long_run_pace_s"]
            .min()
            .reset_index()
        )
    else:
        # Fallback: use best 5-lap average × 1.02 as proxy
        fallback = (
            clean.groupby("Driver")["_LapTime_s"]
            .apply(lambda x: x.nsmallest(5).mean() * 1.02 if len(x) >= 5 else x.mean() * 1.02)
            .rename("fp_long_run_pace_s")
            .reset_index()
            .rename(columns={"Driver": "driver_code"})
        )
        long_run_df = fallback

    # ── Best sector times per driver ─────────────────────────────────────────
    sector_agg: dict[str, pd.Series] = {}
    for i, col in enumerate(("_Sector1Time_s", "_Sector2Time_s", "_Sector3Time_s"), start=1):
        if col in laps.columns:
            sector_agg[f"fp_s{i}_best_s"] = laps.groupby("Driver")[col].min()

    if sector_agg:
        sector_df = pd.DataFrame(sector_agg).reset_index().rename(columns={"Driver": "driver_code"})
    else:
        sector_df = pd.DataFrame(columns=["driver_code"])

    # ── Merge all ────────────────────────────────────────────────────────────
    result = best_laps.merge(long_run_df, on="driver_code", how="left")
    if not sector_df.empty and len(sector_df.columns) > 1:
        result = result.merge(sector_df, on="driver_code", how="left")

    # Fill any remaining NaN long-run paces with best lap × 1.02
    if "fp_long_run_pace_s" in result.columns:
        mask = result["fp_long_run_pace_s"].isna()
        result.loc[mask, "fp_long_run_pace_s"] = result.loc[mask, "fp_best_lap_s"] * 1.02

    return result
