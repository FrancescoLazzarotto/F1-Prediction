"""Jolpica-F1 API source — the maintained Ergast successor. Free, no auth.

Docs: https://api.jolpi.ca/ergast/f1/

Every function returns a tidy DataFrame with snake_case columns, so nothing
downstream ever touches the nested Ergast JSON shape.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd

from f1predict.config import Config, get_config
from f1predict.constants import is_dnf
from f1predict.data.http import HttpJsonClient, NetworkError

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100

#: Ergast caps a single response at 100 rows, so a full season needs paging.
_MAX_PAGES = 40


@lru_cache(maxsize=4)
def _client_for(http_dir: str, rate_limit: float, timeout_s: int, retries: int) -> HttpJsonClient:
    return HttpJsonClient(
        BASE_URL, cache_dir=http_dir, rate_limit=rate_limit,
        timeout_s=timeout_s, max_retries=retries,
    )


def _client(cfg: Config | None = None) -> HttpJsonClient:
    cfg = cfg or get_config()
    return _client_for(
        cfg.cache.http_dir, cfg.data.jolpica_rate_limit,
        cfg.data.request_timeout_s, cfg.data.max_retries,
    )


def _ttl_for_season(year: int, cfg: Config | None = None) -> float | None:
    """Completed seasons are immutable, so cache them forever."""
    cfg = cfg or get_config()
    return None if year < datetime.now(tz=timezone.utc).year else float(cfg.cache.live_ttl_s)


def _paginate(path: str, year: int, cfg: Config | None = None) -> list[dict]:
    """Fetch every page of a season-scoped endpoint and return the race list."""
    cfg = cfg or get_config()
    client = _client(cfg)
    ttl = _ttl_for_season(year, cfg)
    races: list[dict] = []

    for page in range(_MAX_PAGES):
        offset = page * PAGE_SIZE
        payload = client.get(path, params={"limit": PAGE_SIZE, "offset": offset}, ttl_s=ttl)
        mr = payload.get("MRData", {})
        races.extend(mr.get("RaceTable", {}).get("Races", []))
        total = int(mr.get("total", 0))
        if offset + PAGE_SIZE >= total:
            break
    else:
        log.warning("Hit the %d-page cap paginating %s", _MAX_PAGES, path)

    return races


# ── Circuits ──────────────────────────────────────────────────────────────────

def get_circuit_info(year: int, round_num: int, cfg: Config | None = None) -> dict:
    """Return ``{name, circuit_id, lat, lon, locality, country, race_date}``."""
    cfg = cfg or get_config()
    payload = _client(cfg).get(f"{year}/{round_num}.json", ttl_s=_ttl_for_season(year, cfg))
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        raise ValueError(f"No race data for {year} round {round_num}")

    race = races[0]
    circuit = race["Circuit"]
    location = circuit.get("Location", {})
    return {
        "name": circuit.get("circuitName", ""),
        "circuit_id": circuit.get("circuitId", ""),
        "lat": _to_float(location.get("lat"), 0.0),
        "lon": _to_float(location.get("long"), 0.0),
        "locality": location.get("locality", ""),
        "country": location.get("country", ""),
        "race_name": race.get("raceName", ""),
        "race_date": _race_datetime(race),
        "round": int(race.get("round", round_num)),
        "year": year,
    }


def _race_datetime(race: dict) -> str:
    """Combine the separate Ergast date and time fields into one ISO string."""
    date = race.get("date", "")
    if not date:
        return ""
    time_part = race.get("time", "")
    return f"{date}T{time_part}" if time_part else date


# ── Results ───────────────────────────────────────────────────────────────────

_RESULT_COLUMNS = [
    "year", "round", "circuit_id", "race_date", "driver_id", "driver_code",
    "driver_name", "team", "constructor_id", "grid_pos", "race_pos", "points",
    "status", "laps", "dnf", "classified",
]


def get_race_results(year: int, round_num: int, cfg: Config | None = None) -> pd.DataFrame:
    """Per-driver results for a single round."""
    cfg = cfg or get_config()
    payload = _client(cfg).get(
        f"{year}/{round_num}/results.json",
        params={"limit": PAGE_SIZE},
        ttl_s=_ttl_for_season(year, cfg),
    )
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return _empty(_RESULT_COLUMNS)
    return _parse_race(races[0], year)


def get_season_results(year: int, cfg: Config | None = None) -> pd.DataFrame:
    """Every race result of a season, paginated and concatenated."""
    races = _paginate(f"{year}/results.json", year, cfg)
    frames = [_parse_race(race, year) for race in races]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty(_RESULT_COLUMNS)
    return _concat(frames, _RESULT_COLUMNS)


def _parse_race(race: dict, year: int) -> pd.DataFrame:
    circuit_id = race.get("Circuit", {}).get("circuitId", "")
    round_num = int(race.get("round", 0))
    race_date = _race_datetime(race)

    rows = []
    for result in race.get("Results", []):
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        status = result.get("status", "")
        driver_id = driver.get("driverId", "")
        rows.append({
            "year": year,
            "round": round_num,
            "circuit_id": circuit_id,
            "race_date": race_date,
            "driver_id": driver_id,
            "driver_code": driver.get("code") or driver_id[:3].upper(),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name", ""),
            "constructor_id": constructor.get("constructorId", ""),
            "grid_pos": _to_int(result.get("grid"), 20),
            "race_pos": _finishing_position(result),
            "points": _to_float(result.get("points"), 0.0),
            "status": status,
            "laps": _to_int(result.get("laps"), 0),
            "dnf": is_dnf(status),
            "classified": not is_dnf(status),
        })
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS) if rows else _empty(_RESULT_COLUMNS)


def _finishing_position(result: dict) -> int:
    """Classification order, preferring the DNF-aware ``positionOrder``.

    ``position`` is blank or non-numeric for retirements; ``positionOrder``
    always ranks the full field, which is what a ranking target needs.
    """
    for key in ("positionOrder", "position"):
        value = _to_int(result.get(key), None)
        if value is not None and value > 0:
            return value
    return 20


# ── Qualifying ────────────────────────────────────────────────────────────────

_QUALI_COLUMNS = [
    "year", "round", "driver_id", "driver_code", "driver_name", "team",
    "constructor_id", "quali_pos", "q1_s", "q2_s", "q3_s", "best_quali_s",
    "quali_gap_to_pole_s", "quali_gap_to_pole_pct", "reached_q2", "reached_q3",
]


def get_quali_results(year: int, round_num: int, cfg: Config | None = None) -> pd.DataFrame:
    """Qualifying classification for one round, with derived gap columns."""
    cfg = cfg or get_config()
    payload = _client(cfg).get(
        f"{year}/{round_num}/qualifying.json",
        params={"limit": PAGE_SIZE},
        ttl_s=_ttl_for_season(year, cfg),
    )
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return _empty(_QUALI_COLUMNS)
    return _parse_quali(races[0], year)


def get_season_quali(year: int, cfg: Config | None = None) -> pd.DataFrame:
    """Every qualifying session of a season."""
    races = _paginate(f"{year}/qualifying.json", year, cfg)
    frames = [_parse_quali(race, year) for race in races]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty(_QUALI_COLUMNS)
    return _concat(frames, _QUALI_COLUMNS)


def _parse_quali(race: dict, year: int) -> pd.DataFrame:
    round_num = int(race.get("round", 0))
    rows = []
    for result in race.get("QualifyingResults", []):
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        driver_id = driver.get("driverId", "")
        rows.append({
            "year": year,
            "round": round_num,
            "driver_id": driver_id,
            "driver_code": driver.get("code") or driver_id[:3].upper(),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name", ""),
            "constructor_id": constructor.get("constructorId", ""),
            "quali_pos": _to_int(result.get("position"), 20),
            "q1_s": parse_lap_time(result.get("Q1")),
            "q2_s": parse_lap_time(result.get("Q2")),
            "q3_s": parse_lap_time(result.get("Q3")),
        })

    if not rows:
        return _empty(_QUALI_COLUMNS)

    df = pd.DataFrame(rows)
    return add_quali_derived_columns(df)


def add_quali_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add best-time, pole-gap and session-reached columns to a quali frame."""
    df = df.copy()
    # Parsed times arrive as object dtype when a session was cancelled and every
    # entry is None; force float so the row-wise bfill below stays numeric.
    for col in ("q1_s", "q2_s", "q3_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A driver's representative time is their fastest run in the last segment
    # they took part in, so prefer Q3, then Q2, then Q1.
    df["best_quali_s"] = df[["q3_s", "q2_s", "q1_s"]].bfill(axis=1).iloc[:, 0]

    pole = df["best_quali_s"].min(skipna=True)
    if pd.isna(pole) or pole <= 0:
        df["quali_gap_to_pole_s"] = pd.NA
        df["quali_gap_to_pole_pct"] = pd.NA
    else:
        df["quali_gap_to_pole_s"] = df["best_quali_s"] - pole
        df["quali_gap_to_pole_pct"] = (df["best_quali_s"] / pole - 1.0) * 100.0

    df["reached_q2"] = df["q2_s"].notna().astype(int)
    df["reached_q3"] = df["q3_s"].notna().astype(int)
    return df.reindex(columns=_QUALI_COLUMNS)


def parse_lap_time(value: str | None) -> float | None:
    """Convert ``'1:15.123'`` or ``'75.123'`` to seconds; ``None`` when absent."""
    if not value or not str(value).strip():
        return None
    try:
        parts = str(value).strip().split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None


# ── Schedule ──────────────────────────────────────────────────────────────────

_SCHEDULE_COLUMNS = [
    "year", "round", "race_name", "circuit_id", "circuit_name", "locality",
    "country", "lat", "lon", "race_date",
]


def get_season_schedule(year: int, cfg: Config | None = None) -> pd.DataFrame:
    """Calendar for a season: one row per round."""
    cfg = cfg or get_config()
    payload = _client(cfg).get(
        f"{year}.json", params={"limit": PAGE_SIZE}, ttl_s=_ttl_for_season(year, cfg)
    )
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        rows.append({
            "year": year,
            "round": int(race.get("round", 0)),
            "race_name": race.get("raceName", ""),
            "circuit_id": circuit.get("circuitId", ""),
            "circuit_name": circuit.get("circuitName", ""),
            "locality": location.get("locality", ""),
            "country": location.get("country", ""),
            "lat": _to_float(location.get("lat"), 0.0),
            "lon": _to_float(location.get("long"), 0.0),
            "race_date": _race_datetime(race),
        })
    return pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS) if rows else _empty(_SCHEDULE_COLUMNS)


# ── Standings ─────────────────────────────────────────────────────────────────

_DRIVER_STANDINGS_COLUMNS = [
    "position", "driver_id", "driver_code", "driver_name", "team",
    "constructor_id", "points", "wins",
]
_CONSTRUCTOR_STANDINGS_COLUMNS = ["position", "constructor_id", "team", "points", "wins"]


def get_driver_standings(
    year: int, round_num: int | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """Drivers' championship standings, optionally as of a given round."""
    cfg = cfg or get_config()
    path = f"{year}/{round_num}/driverStandings.json" if round_num else f"{year}/driverStandings.json"
    payload = _client(cfg).get(path, params={"limit": PAGE_SIZE}, ttl_s=_ttl_for_season(year, cfg))
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return _empty(_DRIVER_STANDINGS_COLUMNS)

    rows = []
    for entry in lists[0].get("DriverStandings", []):
        driver = entry.get("Driver", {})
        constructors = entry.get("Constructors", [{}])
        driver_id = driver.get("driverId", "")
        rows.append({
            "position": _to_int(entry.get("position"), 99),
            "driver_id": driver_id,
            "driver_code": driver.get("code") or driver_id[:3].upper(),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructors[-1].get("name", ""),
            "constructor_id": constructors[-1].get("constructorId", ""),
            "points": _to_float(entry.get("points"), 0.0),
            "wins": _to_int(entry.get("wins"), 0),
        })
    return pd.DataFrame(rows, columns=_DRIVER_STANDINGS_COLUMNS)


def get_constructor_standings(
    year: int, round_num: int | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """Constructors' championship standings, optionally as of a given round."""
    cfg = cfg or get_config()
    path = (
        f"{year}/{round_num}/constructorStandings.json"
        if round_num else f"{year}/constructorStandings.json"
    )
    payload = _client(cfg).get(path, params={"limit": PAGE_SIZE}, ttl_s=_ttl_for_season(year, cfg))
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return _empty(_CONSTRUCTOR_STANDINGS_COLUMNS)

    rows = []
    for entry in lists[0].get("ConstructorStandings", []):
        constructor = entry.get("Constructor", {})
        rows.append({
            "position": _to_int(entry.get("position"), 99),
            "constructor_id": constructor.get("constructorId", ""),
            "team": constructor.get("name", ""),
            "points": _to_float(entry.get("points"), 0.0),
            "wins": _to_int(entry.get("wins"), 0),
        })
    return pd.DataFrame(rows, columns=_CONSTRUCTOR_STANDINGS_COLUMNS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concatenate per-race frames, dropping all-NA columns first.

    A cancelled session yields a frame whose optional columns are entirely NA;
    pandas warns about inferring dtypes from those, so exclude them and let the
    reindex restore the full schema.
    """
    cleaned = [f.dropna(axis=1, how="all") for f in frames]
    cleaned = [f for f in cleaned if not f.empty]
    if not cleaned:
        return _empty(columns)
    return pd.concat(cleaned, ignore_index=True).reindex(columns=columns)


def _to_int(value, default):
    """Parse an Ergast string field to int, tolerating None/''/non-numeric."""
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _to_float(value, default):
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


__all__ = [
    "NetworkError",
    "add_quali_derived_columns",
    "get_circuit_info",
    "get_constructor_standings",
    "get_driver_standings",
    "get_quali_results",
    "get_race_results",
    "get_season_quali",
    "get_season_results",
    "get_season_schedule",
    "parse_lap_time",
]
