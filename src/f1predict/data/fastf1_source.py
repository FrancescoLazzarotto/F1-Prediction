"""FastF1 data source: practice laps, entry lists, session results.

FastF1 talks to the live timing archive, which is slow and occasionally
unavailable, so everything here degrades to ``None``/empty rather than raising.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

# FastF1 logs one INFO line per cached file, which drowns out our own output.
logging.getLogger("fastf1").setLevel(logging.WARNING)

#: Practice sessions in descending order of how representative they are of
#: qualifying pace: FP2 carries the low-fuel runs on a conventional weekend,
#: FP3 is the final dry-run before quali, FP1 is usually programme work.
PRACTICE_PREFERENCE: tuple[str, ...] = ("FP2", "FP3", "FP1")

#: On a sprint weekend there is a single practice session before parc fermé.
SPRINT_PRACTICE_PREFERENCE: tuple[str, ...] = ("FP1",)

ENTRY_COLUMNS = ["driver_code", "driver_id", "driver_name", "team", "driver_number"]


@dataclass(frozen=True, slots=True)
class PracticeData:
    """Laps from the practice session we chose, plus which one it was."""

    session_name: str
    laps: pd.DataFrame


def load_session(year: int, round_num: int, session_name: str, *, laps: bool = True):
    """Load one session, returning ``None`` on any failure."""
    try:
        import fastf1

        session = fastf1.get_session(year, round_num, session_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.load(laps=laps, telemetry=False, weather=False, messages=False)
        return session
    except Exception as exc:
        log.debug("Could not load %d R%d %s: %s", year, round_num, session_name, exc)
        return None


def get_practice_laps(
    year: int, round_num: int, sprint_weekend: bool = False
) -> PracticeData | None:
    """Laps from the most representative practice session that has data."""
    preference = SPRINT_PRACTICE_PREFERENCE if sprint_weekend else PRACTICE_PREFERENCE
    for name in preference:
        session = load_session(year, round_num, name)
        if session is None:
            continue
        laps = getattr(session, "laps", None)
        if laps is None or len(laps) == 0:
            continue
        log.info("Using %s for practice features (%d R%d)", name, year, round_num)
        return PracticeData(session_name=name, laps=laps.copy())
    return None


def get_entry_list(year: int, round_num: int) -> pd.DataFrame:
    """Drivers entered for an event.

    Tries sessions from latest to earliest so that a completed weekend uses the
    race classification and an upcoming one still finds the practice entry list.
    """
    for session_name in ("R", "Q", "SQ", "FP3", "FP2", "FP1"):
        session = load_session(year, round_num, session_name, laps=False)
        results = getattr(session, "results", None) if session is not None else None
        if results is None or len(results) == 0:
            continue
        return _results_to_entries(results)

    return pd.DataFrame({c: pd.Series(dtype="object") for c in ENTRY_COLUMNS})


def _results_to_entries(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        abbreviation = str(row.get("Abbreviation", "") or "")
        driver_id = str(row.get("DriverId", "") or abbreviation).lower()
        full_name = str(row.get("FullName", "") or row.get("BroadcastName", "") or abbreviation)
        rows.append({
            "driver_code": abbreviation,
            "driver_id": driver_id,
            "driver_name": full_name,
            "team": str(row.get("TeamName", "") or ""),
            "driver_number": str(row.get("DriverNumber", "") or ""),
        })
    df = pd.DataFrame(rows, columns=ENTRY_COLUMNS)
    return df[df["driver_code"].astype(bool)].drop_duplicates("driver_code").reset_index(drop=True)


def get_session_results(year: int, round_num: int, session_name: str) -> pd.DataFrame:
    """Classification for a session as a tidy frame, or empty on failure."""
    session = load_session(year, round_num, session_name, laps=False)
    results = getattr(session, "results", None) if session is not None else None
    if results is None or len(results) == 0:
        return pd.DataFrame()

    rows = []
    for _, row in results.iterrows():
        rows.append({
            "driver_code": str(row.get("Abbreviation", "") or ""),
            "driver_name": str(row.get("FullName", "") or ""),
            "team": str(row.get("TeamName", "") or ""),
            "position": _safe_int(row.get("Position")),
            "grid_pos": _safe_int(row.get("GridPosition")),
            "points": _safe_float(row.get("Points")),
            "status": str(row.get("Status", "") or ""),
        })
    return pd.DataFrame(rows)


def get_session_weather(year: int, round_num: int, session_name: str = "R") -> dict | None:
    """Measured trackside weather, averaged over a session.

    Only available once a session has run; used to ground-truth the forecast
    when backtesting.
    """
    session = load_session(year, round_num, session_name, laps=False)
    if session is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.load(laps=False, telemetry=False, weather=True, messages=False)
        weather = session.weather_data
    except Exception:
        return None

    if weather is None or len(weather) == 0:
        return None

    return {
        "temperature": float(weather["AirTemp"].mean()),
        "track_temp": float(weather["TrackTemp"].mean()),
        "humidity": float(weather["Humidity"].mean()),
        "wind_speed": float(weather["WindSpeed"].mean()),
        "rain_prob": float(weather["Rainfall"].mean()),
        "cloud_cover": 50.0,
        "source": "measured",
    }


def _safe_int(value, default: int | None = None) -> int | None:
    """Parse a FastF1 numeric cell, which is NaN for non-starters.

    ``int(value or default)`` is wrong here: NaN is truthy, so it survives the
    ``or`` and then blows up inside ``int()``.
    """
    if value is None or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
