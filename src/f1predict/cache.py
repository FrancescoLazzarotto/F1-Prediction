"""Unified cache layer: FastF1 session cache + parquet feature store + model store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

log = logging.getLogger(__name__)


def ensure_dirs(cfg: dict) -> None:
    """Create all cache directories if they don't exist."""
    for key in ("fastf1_dir", "features_dir", "models_dir"):
        Path(cfg["cache"][key]).mkdir(parents=True, exist_ok=True)


def setup_fastf1_cache(cfg: dict) -> None:
    """Enable FastF1's built-in session cache."""
    import fastf1

    cache_dir = cfg["cache"]["fastf1_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    log.debug("FastF1 cache: %s", cache_dir)


# ── Feature parquet cache ─────────────────────────────────────────────────────

def _feature_path(cfg: dict, year: int, round_num: int, kind: str) -> Path:
    return Path(cfg["cache"]["features_dir"]) / f"{year}_{round_num:02d}_{kind}.parquet"


def save_features(cfg: dict, year: int, round_num: int, kind: str, df: pd.DataFrame) -> None:
    path = _feature_path(cfg, year, round_num, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.debug("Cached features: %s", path.name)


def load_features(cfg: dict, year: int, round_num: int, kind: str) -> pd.DataFrame | None:
    path = _feature_path(cfg, year, round_num, kind)
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Weather JSON cache ────────────────────────────────────────────────────────

def _weather_path(cfg: dict, year: int, round_num: int) -> Path:
    return Path(cfg["cache"]["features_dir"]) / f"{year}_{round_num:02d}_weather.json"


def save_weather(cfg: dict, year: int, round_num: int, data: dict) -> None:
    path = _weather_path(cfg, year, round_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_weather(cfg: dict, year: int, round_num: int) -> dict | None:
    path = _weather_path(cfg, year, round_num)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Season results JSON cache (Jolpica bulk fetch) ────────────────────────────

def _season_path(cfg: dict, year: int) -> Path:
    return Path(cfg["cache"]["features_dir"]) / f"season_{year}_results.parquet"


def save_season_results(cfg: dict, year: int, df: pd.DataFrame) -> None:
    path = _season_path(cfg, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_season_results(cfg: dict, year: int) -> pd.DataFrame | None:
    path = _season_path(cfg, year)
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Model joblib cache ────────────────────────────────────────────────────────

def save_model(cfg: dict, name: str, obj: object) -> None:
    path = Path(cfg["cache"]["models_dir"]) / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    log.debug("Saved model: %s", path.name)


def load_model(cfg: dict, name: str) -> object | None:
    path = Path(cfg["cache"]["models_dir"]) / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def models_exist(cfg: dict) -> bool:
    race = Path(cfg["cache"]["models_dir"]) / "race_model.joblib"
    quali = Path(cfg["cache"]["models_dir"]) / "quali_model.joblib"
    return race.exists() and quali.exists()
