"""Single façade over every historical-data source.

Callers ask this module for results, qualifying, schedules and standings; it
decides between the Jolpica API and the bundled CSV dump, and keeps a parquet
copy of each season so a re-train never re-downloads a completed year.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from f1predict import cache as C
from f1predict.config import Config, get_config
from f1predict.constants import is_dnf
from f1predict.data import jolpica, offline
from f1predict.data.http import NetworkError
from f1predict.data.jolpica import _QUALI_COLUMNS, _RESULT_COLUMNS

log = logging.getLogger(__name__)


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute columns derived from raw fields, every time data is read.

    ``dnf`` and ``classified`` are functions of ``status``. Persisting them
    would freeze whatever classification rule was in force when the parquet was
    written, so a later fix to :func:`is_dnf` would silently not apply to any
    cached season. Deriving on read keeps the cache holding only raw facts.
    """
    if df is None or df.empty or "status" not in df.columns:
        return df
    df = df.copy()
    df["dnf"] = df["status"].map(is_dnf)
    df["classified"] = ~df["dnf"]
    return df


class DataUnavailableError(RuntimeError):
    """Raised when neither the API nor the offline dump can serve a request."""


def _current_year() -> int:
    return datetime.now(tz=timezone.utc).year


def _offline_allowed(cfg: Config, year: int) -> bool:
    return cfg.data.offline_fallback and offline.is_available() and year in offline.covered_seasons()


def season_results(year: int, cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """All race results for ``year``, from parquet cache, API, or CSV dump.

    Completed seasons are cached to parquet permanently; the running season is
    re-fetched once its cache entry goes stale.
    """
    cfg = cfg or get_config()
    is_current = year >= _current_year()
    ttl = cfg.cache.live_ttl_s if is_current else None

    if not refresh:
        cached = C.load_frame(
            cfg, f"season_{year}_results", ttl_s=ttl, required_columns=_RESULT_COLUMNS
        )
        if cached is not None:
            return _with_derived(cached)

    try:
        df = jolpica.get_season_results(year, cfg)
        if not df.empty:
            C.save_frame(cfg, f"season_{year}_results", df)
            return _with_derived(df)
        log.info("API returned no results for %d", year)
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Jolpica season fetch failed for %d (%s)", year, exc)

    if _offline_allowed(cfg, year):
        log.info("Falling back to the offline dataset for %d", year)
        df = offline.get_season_results(year)
        if not df.empty:
            C.save_frame(cfg, f"season_{year}_results", df)
            return _with_derived(df)

    # A stale cache entry beats no data at all.
    stale = C.load_frame(
        cfg, f"season_{year}_results", ttl_s=None, required_columns=_RESULT_COLUMNS
    )
    if stale is not None:
        log.warning("Serving stale cached results for %d", year)
        return _with_derived(stale)

    return pd.DataFrame()


def season_quali(year: int, cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """All qualifying classifications for ``year``."""
    cfg = cfg or get_config()
    is_current = year >= _current_year()
    ttl = cfg.cache.live_ttl_s if is_current else None

    if not refresh:
        cached = C.load_frame(
            cfg, f"season_{year}_quali", ttl_s=ttl, required_columns=_QUALI_COLUMNS
        )
        if cached is not None:
            return cached

    try:
        df = jolpica.get_season_quali(year, cfg)
        if not df.empty:
            C.save_frame(cfg, f"season_{year}_quali", df)
            return df
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Jolpica qualifying fetch failed for %d (%s)", year, exc)

    stale = C.load_frame(
        cfg, f"season_{year}_quali", ttl_s=None, required_columns=_QUALI_COLUMNS
    )
    return stale if stale is not None else pd.DataFrame()


def history_for(year: int, seasons_back: int = 2, cfg: Config | None = None) -> pd.DataFrame:
    """Results for ``year`` plus the preceding seasons, for form features.

    Form windows roll across a season boundary, so predicting round 1 of a year
    still needs the tail of the previous one.
    """
    cfg = cfg or get_config()
    frames = []
    for y in range(year - seasons_back, year + 1):
        df = season_results(y, cfg)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["year", "round"], ignore_index=True)


def quali_history_for(year: int, seasons_back: int = 2, cfg: Config | None = None) -> pd.DataFrame:
    """Qualifying results for ``year`` and the preceding seasons."""
    cfg = cfg or get_config()
    frames = []
    for y in range(year - seasons_back, year + 1):
        df = season_quali(y, cfg)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["year", "round"], ignore_index=True)


def race_results(year: int, round_num: int, cfg: Config | None = None) -> pd.DataFrame:
    """Results for a single round, preferring the cached season frame."""
    cfg = cfg or get_config()
    season = season_results(year, cfg)
    if not season.empty and "round" in season.columns:
        subset = season[season["round"] == round_num]
        if not subset.empty:
            return subset.reset_index(drop=True)

    try:
        return _with_derived(jolpica.get_race_results(year, round_num, cfg))
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Race result fetch failed for %d R%d (%s)", year, round_num, exc)

    if _offline_allowed(cfg, year):
        return _with_derived(offline.get_race_results(year, round_num))
    return pd.DataFrame()


def quali_results(year: int, round_num: int, cfg: Config | None = None) -> pd.DataFrame:
    """Qualifying classification for a single round."""
    cfg = cfg or get_config()
    season = season_quali(year, cfg)
    if not season.empty and "round" in season.columns:
        subset = season[season["round"] == round_num]
        if not subset.empty:
            return subset.reset_index(drop=True)

    try:
        return jolpica.get_quali_results(year, round_num, cfg)
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Qualifying fetch failed for %d R%d (%s)", year, round_num, exc)

    if _offline_allowed(cfg, year):
        return offline.get_quali_results(year, round_num)
    return pd.DataFrame()


def circuit_info(year: int, round_num: int, cfg: Config | None = None) -> dict:
    """Circuit metadata for a round, with a neutral fallback dict."""
    cfg = cfg or get_config()
    try:
        return jolpica.get_circuit_info(year, round_num, cfg)
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Circuit info fetch failed for %d R%d (%s)", year, round_num, exc)

    if _offline_allowed(cfg, year):
        try:
            return offline.get_circuit_info(year, round_num)
        except ValueError:
            pass

    return {
        "name": "Unknown circuit", "circuit_id": "", "lat": 0.0, "lon": 0.0,
        "locality": "", "country": "", "race_name": "", "race_date": "",
        "round": round_num, "year": year,
    }


def season_schedule(year: int, cfg: Config | None = None) -> pd.DataFrame:
    """Calendar for a season."""
    cfg = cfg or get_config()
    try:
        df = jolpica.get_season_schedule(year, cfg)
        if not df.empty:
            return df
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Schedule fetch failed for %d (%s)", year, exc)

    if _offline_allowed(cfg, year):
        return offline.get_season_schedule(year)
    return pd.DataFrame()


def driver_standings(
    year: int, round_num: int | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """Drivers' championship table, computed locally if the API is down."""
    cfg = cfg or get_config()
    try:
        df = jolpica.get_driver_standings(year, round_num, cfg)
        if not df.empty:
            return df
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Driver standings fetch failed for %d (%s)", year, exc)
    return _standings_from_results(season_results(year, cfg), round_num, by="driver")


def constructor_standings(
    year: int, round_num: int | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """Constructors' championship table, computed locally if the API is down."""
    cfg = cfg or get_config()
    try:
        df = jolpica.get_constructor_standings(year, round_num, cfg)
        if not df.empty:
            return df
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Constructor standings fetch failed for %d (%s)", year, exc)
    return _standings_from_results(season_results(year, cfg), round_num, by="constructor")


def _standings_from_results(
    results: pd.DataFrame, round_num: int | None, by: str
) -> pd.DataFrame:
    """Rebuild a championship table by summing points from race results."""
    if results.empty:
        return pd.DataFrame()

    df = results if round_num is None else results[results["round"] <= round_num]
    if df.empty:
        return pd.DataFrame()

    if by == "driver":
        keys = ["driver_id", "driver_code", "driver_name", "team", "constructor_id"]
    else:
        keys = ["constructor_id", "team"]

    # Keep the most recent identity for each key (drivers change teams mid-season).
    latest = df.sort_values(["year", "round"]).groupby(keys[0]).last()

    agg = (
        df.groupby(keys[0])
        .agg(points=("points", "sum"), wins=("race_pos", lambda s: int((s == 1).sum())))
        .reset_index()
    )
    for col in keys[1:]:
        agg[col] = agg[keys[0]].map(latest[col])

    agg = agg.sort_values(["points", "wins"], ascending=False).reset_index(drop=True)
    agg.insert(0, "position", range(1, len(agg) + 1))
    return agg
