"""Unified cache: FastF1 sessions, parquet frames, JSON blobs, model bundles.

Everything the project caches goes through here, so there is exactly one place
that knows about paths, TTLs and atomic writes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from f1predict.config import Config

log = logging.getLogger(__name__)

RACE_MODEL = "race_model"
QUALI_MODEL = "quali_model"
DNF_MODEL = "dnf_model"


def ensure_dirs(cfg: Config) -> None:
    """Create every cache directory."""
    for path in cfg.cache.paths():
        path.mkdir(parents=True, exist_ok=True)


_fastf1_cache_enabled: str | None = None


def setup_fastf1_cache(cfg: Config) -> None:
    """Point FastF1 at our cache directory, at most once per process.

    ``enable_cache`` re-scans the directory on every call, which is slow when
    the cache holds thousands of session pickles.
    """
    global _fastf1_cache_enabled
    cache_dir = cfg.cache.fastf1_dir
    if _fastf1_cache_enabled == cache_dir:
        return

    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    _fastf1_cache_enabled = cache_dir
    log.debug("FastF1 cache enabled at %s", cache_dir)


# ── Parquet frames ────────────────────────────────────────────────────────────

def _frame_path(cfg: Config, key: str) -> Path:
    return Path(cfg.cache.features_dir) / f"{_safe(key)}.parquet"


def save_frame(cfg: Config, key: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame, atomically so readers never see a partial file."""
    path = _frame_path(cfg, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    except (OSError, ValueError) as exc:
        log.warning("Could not cache frame %s: %s", key, exc)
        tmp.unlink(missing_ok=True)


def load_frame(
    cfg: Config,
    key: str,
    ttl_s: float | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Read a cached DataFrame, or ``None`` when absent, stale or out of date.

    Args:
        required_columns: Columns the caller needs. A cached frame missing any
            of them was written by an older schema, so it is discarded and
            refetched instead of being handed back half-populated.
    """
    path = _frame_path(cfg, key)
    if not path.exists():
        return None
    if ttl_s is not None and (time.time() - path.stat().st_mtime) > ttl_s:
        return None
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        log.warning("Discarding unreadable cache file %s: %s", path.name, exc)
        path.unlink(missing_ok=True)
        return None

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            log.info(
                "Cached %s predates the current schema (missing %s); refetching.",
                path.name, ", ".join(missing),
            )
            path.unlink(missing_ok=True)
            return None
    return df


def race_key(year: int, round_num: int, kind: str) -> str:
    """Canonical cache key for a per-race artefact.

    The schema version is part of the name so that changing the feature set
    invalidates every cached frame automatically, instead of silently mixing
    old and new columns in the next training run.
    """
    from f1predict.features.schema import SCHEMA_VERSION

    return f"{year}_{round_num:02d}_{kind}_v{SCHEMA_VERSION}"


def save_features(cfg: Config, year: int, round_num: int, kind: str, df: pd.DataFrame) -> None:
    save_frame(cfg, race_key(year, round_num, kind), df)


def load_features(
    cfg: Config, year: int, round_num: int, kind: str, ttl_s: float | None = None
) -> pd.DataFrame | None:
    return load_frame(cfg, race_key(year, round_num, kind), ttl_s=ttl_s)


# ── JSON blobs ────────────────────────────────────────────────────────────────

def _json_path(cfg: Config, key: str) -> Path:
    return Path(cfg.cache.features_dir) / f"{_safe(key)}.json"


def save_json(cfg: Config, key: str, data: Any) -> None:
    path = _json_path(cfg, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str)
        tmp.replace(path)
    except OSError as exc:
        log.warning("Could not cache JSON %s: %s", key, exc)
        tmp.unlink(missing_ok=True)


def load_json(cfg: Config, key: str, ttl_s: float | None = None) -> Any | None:
    path = _json_path(cfg, key)
    if not path.exists():
        return None
    if ttl_s is not None and (time.time() - path.stat().st_mtime) > ttl_s:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def save_weather(cfg: Config, year: int, round_num: int, data: dict) -> None:
    save_json(cfg, race_key(year, round_num, "weather"), data)


def load_weather(
    cfg: Config, year: int, round_num: int, ttl_s: float | None = None
) -> dict | None:
    return load_json(cfg, race_key(year, round_num, "weather"), ttl_s=ttl_s)


# ── Models ────────────────────────────────────────────────────────────────────

def _model_path(cfg: Config, name: str) -> Path:
    return Path(cfg.cache.models_dir) / f"{_safe(name)}.joblib"


def save_model(cfg: Config, name: str, obj: object) -> None:
    path = _model_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)
    log.debug("Saved model %s", path.name)


def load_model(cfg: Config, name: str) -> Any | None:
    """Load a model bundle, returning ``None`` if missing or unreadable.

    An unpickling failure means the artefact was written by an incompatible
    version, so the caller should retrain rather than crash.
    """
    path = _model_path(cfg, name)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        log.warning("Could not load %s (%s); it will be retrained.", path.name, exc)
        return None


def models_exist(cfg: Config) -> bool:
    """True when the two models required for a race prediction are present."""
    return all(_model_path(cfg, n).exists() for n in (RACE_MODEL, QUALI_MODEL))


def model_mtime(cfg: Config, name: str) -> float | None:
    path = _model_path(cfg, name)
    return path.stat().st_mtime if path.exists() else None


def clear(cfg: Config, what: str = "all") -> list[str]:
    """Delete cached artefacts. Returns the human-readable names removed.

    ``what`` is one of ``features``, ``models``, ``http``, ``fastf1`` or ``all``.
    """
    targets = {
        "features": Path(cfg.cache.features_dir),
        "models": Path(cfg.cache.models_dir),
        "http": Path(cfg.cache.http_dir),
        "fastf1": Path(cfg.cache.fastf1_dir),
    }
    if what != "all":
        if what not in targets:
            raise ValueError(f"Unknown cache section {what!r}; expected one of {sorted(targets)}")
        targets = {what: targets[what]}

    removed = []
    for name, directory in targets.items():
        if not directory.exists():
            continue
        count = 0
        for path in directory.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                count += 1
        removed.append(f"{name} ({count} files)")
    return removed


def usage(cfg: Config) -> dict[str, int]:
    """Bytes on disk per cache section."""
    out: dict[str, int] = {}
    for name, directory in (
        ("fastf1", Path(cfg.cache.fastf1_dir)),
        ("features", Path(cfg.cache.features_dir)),
        ("models", Path(cfg.cache.models_dir)),
        ("http", Path(cfg.cache.http_dir)),
    ):
        out[name] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) \
            if directory.exists() else 0
    return out


def _safe(key: str) -> str:
    """Make a cache key safe for every filesystem we run on."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
