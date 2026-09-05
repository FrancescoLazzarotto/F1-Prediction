"""Offline data source backed by the bundled Ergast CSV dump in ``Dataset/``.

The dump covers 1950-2023 and is used as a fallback when Jolpica is unreachable,
so training and backtesting still work on a plane. It emits exactly the same
column schema as :mod:`f1predict.data.jolpica`, so callers cannot tell them apart.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from f1predict.config import REPO_ROOT
from f1predict.constants import is_dnf

log = logging.getLogger(__name__)

DATASET_DIR = REPO_ROOT / "Dataset"

#: Ergast writes SQL NULLs as a literal backslash-N.
_NA_VALUES = ["\\N", "\\\\N", ""]


def is_available() -> bool:
    """True when the CSV dump is present and readable."""
    return (DATASET_DIR / "results.csv").exists() and (DATASET_DIR / "races.csv").exists()


@lru_cache(maxsize=16)
def _table(name: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Offline dataset table missing: {path}")
    # The dump is UTF-8; without saying so, Windows reads it as cp1252 and
    # mangles every accented driver name.
    return pd.read_csv(
        path, na_values=_NA_VALUES, keep_default_na=True,
        low_memory=False, encoding="utf-8",
    )


@lru_cache(maxsize=1)
def covered_seasons() -> tuple[int, ...]:
    """Seasons for which the dump holds results."""
    if not is_available():
        return ()
    races = _table("races")
    return tuple(sorted(races["year"].dropna().astype(int).unique()))


@lru_cache(maxsize=1)
def _joined_results() -> pd.DataFrame:
    """Results joined to races, drivers, constructors and statuses.

    Computed once and memoised: the join is ~26k rows and would otherwise be
    repeated for every season and every backtest.
    """
    results = _table("results")
    races = _table("races")[["raceId", "year", "round", "circuitId", "name", "date", "time"]]
    drivers = _table("drivers")[["driverId", "driverRef", "code", "forename", "surname"]]
    constructors = _table("constructors")[["constructorId", "constructorRef", "name"]]
    circuits = _table("circuits")[["circuitId", "circuitRef", "name", "location", "country",
                                   "lat", "lng"]]
    statuses = _table("status")

    df = (
        results
        .merge(races, on="raceId", how="left", suffixes=("", "_race"))
        .merge(drivers, on="driverId", how="left")
        .merge(constructors, on="constructorId", how="left", suffixes=("", "_constructor"))
        .merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit"))
        .merge(statuses, on="statusId", how="left")
    )

    df["driver_name"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
    df["race_date"] = df["date"].fillna("").astype(str)
    return df


def _to_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Project the joined table onto the canonical result schema."""
    from f1predict.data.jolpica import _RESULT_COLUMNS

    out = pd.DataFrame({
        "year": df["year"].astype("Int64"),
        "round": df["round"].astype("Int64"),
        "circuit_id": df["circuitRef"].fillna(""),
        "race_date": df["race_date"],
        "driver_id": df["driverRef"].fillna(""),
        "driver_code": df["code"].fillna(df["driverRef"].str[:3].str.upper()),
        "driver_name": df["driver_name"].str.strip(),
        "team": df["name_constructor"].fillna(""),
        "constructor_id": df["constructorRef"].fillna(""),
        "grid_pos": pd.to_numeric(df["grid"], errors="coerce").fillna(20).astype(int),
        "race_pos": pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20).astype(int),
        "points": pd.to_numeric(df["points"], errors="coerce").fillna(0.0),
        "status": df["status"].fillna(""),
        "laps": pd.to_numeric(df["laps"], errors="coerce").fillna(0).astype(int),
    })
    out["dnf"] = out["status"].map(is_dnf)
    out["classified"] = ~out["dnf"]
    return out.reindex(columns=_RESULT_COLUMNS).reset_index(drop=True)


def get_season_results(year: int) -> pd.DataFrame:
    """Every race result of a season, from the CSV dump."""
    df = _joined_results()
    season = df[df["year"] == year]
    if season.empty:
        return _to_result_frame(season)
    return _to_result_frame(season).sort_values(["round", "race_pos"]).reset_index(drop=True)


def get_race_results(year: int, round_num: int) -> pd.DataFrame:
    """Results for one round, from the CSV dump."""
    df = _joined_results()
    race = df[(df["year"] == year) & (df["round"] == round_num)]
    return _to_result_frame(race).sort_values("race_pos").reset_index(drop=True)


def get_quali_results(year: int, round_num: int) -> pd.DataFrame:
    """Qualifying classification for one round, from the CSV dump."""
    from f1predict.data.jolpica import _QUALI_COLUMNS, add_quali_derived_columns, parse_lap_time

    quali = _table("qualifying")
    races = _table("races")[["raceId", "year", "round"]]
    drivers = _table("drivers")[["driverId", "driverRef", "code", "forename", "surname"]]
    constructors = _table("constructors")[["constructorId", "constructorRef", "name"]]

    df = (
        quali.merge(races, on="raceId", how="inner")
        .merge(drivers, on="driverId", how="left")
        .merge(constructors, on="constructorId", how="left")
    )
    df = df[(df["year"] == year) & (df["round"] == round_num)]
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _QUALI_COLUMNS})

    out = pd.DataFrame({
        "year": year,
        "round": round_num,
        "driver_id": df["driverRef"].fillna(""),
        "driver_code": df["code"].fillna(df["driverRef"].str[:3].str.upper()),
        "driver_name": (df["forename"].fillna("") + " " + df["surname"].fillna("")).str.strip(),
        "team": df["name"].fillna(""),
        "constructor_id": df["constructorRef"].fillna(""),
        "quali_pos": pd.to_numeric(df["position"], errors="coerce").fillna(20).astype(int),
        "q1_s": df["q1"].map(parse_lap_time),
        "q2_s": df["q2"].map(parse_lap_time),
        "q3_s": df["q3"].map(parse_lap_time),
    }).reset_index(drop=True)

    return add_quali_derived_columns(out).sort_values("quali_pos").reset_index(drop=True)


def get_circuit_info(year: int, round_num: int) -> dict:
    """Circuit metadata for one round, from the CSV dump."""
    df = _joined_results()
    race = df[(df["year"] == year) & (df["round"] == round_num)]
    if race.empty:
        raise ValueError(f"Offline dataset has no {year} round {round_num}")

    row = race.iloc[0]
    return {
        "name": str(row.get("name_circuit", "") or ""),
        "circuit_id": str(row.get("circuitRef", "") or ""),
        "lat": float(row.get("lat", 0.0) or 0.0),
        "lon": float(row.get("lng", 0.0) or 0.0),
        "locality": str(row.get("location", "") or ""),
        "country": str(row.get("country", "") or ""),
        "race_name": str(row.get("name_race", row.get("name", "")) or ""),
        "race_date": str(row.get("race_date", "") or ""),
        "round": int(round_num),
        "year": int(year),
    }


def get_season_schedule(year: int) -> pd.DataFrame:
    """Calendar for a season, from the CSV dump."""
    from f1predict.data.jolpica import _SCHEDULE_COLUMNS

    races = _table("races")
    circuits = _table("circuits")
    df = races[races["year"] == year].merge(circuits, on="circuitId", how="left",
                                            suffixes=("_race", "_circuit"))
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _SCHEDULE_COLUMNS})

    return pd.DataFrame({
        "year": year,
        "round": df["round"].astype(int),
        "race_name": df["name_race"].fillna(""),
        "circuit_id": df["circuitRef"].fillna(""),
        "circuit_name": df["name_circuit"].fillna(""),
        "locality": df["location"].fillna(""),
        "country": df["country"].fillna(""),
        "lat": pd.to_numeric(df["lat"], errors="coerce").fillna(0.0),
        "lon": pd.to_numeric(df["lng"], errors="coerce").fillna(0.0),
        "race_date": df["date"].fillna("").astype(str),
    }).sort_values("round").reset_index(drop=True)
