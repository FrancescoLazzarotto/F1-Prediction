"""FastF1 data source: sessions, laps, results, driver info."""

from __future__ import annotations

import logging
import warnings

import fastf1
import pandas as pd

log = logging.getLogger(__name__)

# Suppress verbose fastf1 INFO logs
logging.getLogger("fastf1").setLevel(logging.WARNING)


def load_session(year: int, round_num: int, session_name: str) -> fastf1.core.Session | None:
    """
    Load a FastF1 session with laps but without full telemetry.
    Returns None on failure (network error, session not found, etc.).
    """
    try:
        session = fastf1.get_session(year, round_num, session_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
        return session
    except Exception as exc:
        log.warning("Could not load %d R%d %s: %s", year, round_num, session_name, exc)
        return None


def get_fp_laps(year: int, round_num: int) -> pd.DataFrame | None:
    """
    Load FP laps (tries FP2, then FP3, then FP1 in order of representativeness).
    Returns a DataFrame with columns: Driver, LapTime, Stint, Compound,
    TyreLife, IsAccurate, PitOutLap, PitInLap, Sector1Time, Sector2Time,
    Sector3Time, TrackStatus.
    """
    for session_name in ("FP2", "FP3", "FP1"):
        session = load_session(year, round_num, session_name)
        if session is None:
            continue
        laps = session.laps
        if laps is None or len(laps) == 0:
            continue
        log.info("Using %s for FP features (%d R%d)", session_name, year, round_num)
        return laps.copy()
    return None


def get_quali_laps(year: int, round_num: int) -> fastf1.core.Session | None:
    """Load the Qualifying session (used to verify quali predictions)."""
    return load_session(year, round_num, "Q")


def get_race_session(year: int, round_num: int) -> fastf1.core.Session | None:
    """Load the Race session."""
    return load_session(year, round_num, "R")


def get_race_results(year: int, round_num: int) -> pd.DataFrame | None:
    """
    Get race results from FastF1 with columns:
    driver_code, driver_name, team, grid_pos, race_pos, points, status.
    """
    session = get_race_session(year, round_num)
    if session is None:
        return None
    results = session.results
    if results is None or len(results) == 0:
        return None

    rows = []
    for _, row in results.iterrows():
        rows.append({
            "driver_code": row.get("Abbreviation", "???"),
            "driver_name": str(row.get("FullName", row.get("BroadcastName", ""))),
            "team": str(row.get("TeamName", "")),
            "grid_pos": int(row.get("GridPosition", 20) or 20),
            "race_pos": int(row.get("Position", 20) or 20),
            "points": float(row.get("Points", 0) or 0),
            "status": str(row.get("Status", "")),
            "driver_id": str(row.get("DriverId", row.get("Abbreviation", ""))).lower(),
        })
    return pd.DataFrame(rows)


def get_entry_list(year: int, round_num: int) -> pd.DataFrame:
    """
    Return the list of drivers entered for this event.
    Columns: driver_code, driver_name, team, driver_number.
    Falls back to qualifying or practice sessions if race not available.
    """
    for sess_name in ("R", "Q", "FP2", "FP1"):
        session = load_session(year, round_num, sess_name)
        if session is None or session.results is None or len(session.results) == 0:
            continue
        results = session.results
        rows = []
        for _, row in results.iterrows():
            rows.append({
                "driver_code": row.get("Abbreviation", "???"),
                "driver_name": str(row.get("FullName", row.get("BroadcastName", ""))),
                "team": str(row.get("TeamName", "")),
                "driver_number": str(row.get("DriverNumber", "")),
                "driver_id": str(row.get("DriverId", row.get("Abbreviation", ""))).lower(),
            })
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["driver_code", "driver_name", "team", "driver_number", "driver_id"])
