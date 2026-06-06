"""Schedule resolution: year/round, GP name, or 'next race'."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import fastf1
import pandas as pd

log = logging.getLogger(__name__)


def get_schedule(year: int) -> pd.DataFrame:
    """Return the full event schedule for a season as a DataFrame."""
    return fastf1.get_event_schedule(year, include_testing=False)


def get_next_event() -> tuple[int, int, dict]:
    """
    Find the next upcoming race event.
    Returns (year, round_num, event_info_dict).
    """
    now = datetime.now(tz=timezone.utc)
    year = now.year

    for y in (year, year + 1):
        try:
            schedule = get_schedule(y)
        except Exception:
            continue

        # Use the race session date (Session5DateUtc) or fall back to EventDate
        date_col = "Session5DateUtc" if "Session5DateUtc" in schedule.columns else "EventDate"
        schedule = schedule.copy()
        schedule["_race_dt"] = pd.to_datetime(schedule[date_col], utc=True, errors="coerce")
        future = schedule[schedule["_race_dt"] > now].sort_values("_race_dt")
        if len(future) > 0:
            row = future.iloc[0]
            return y, int(row["RoundNumber"]), _event_dict(row)

    raise RuntimeError("Could not determine next race from FastF1 schedule.")


def resolve_event(
    year: int | None,
    round_num: int | None,
    gp: str | None,
    next_race: bool = False,
) -> tuple[int, int, dict]:
    """
    Resolve the (year, round, info) tuple from whichever args are provided.
    Priority: next_race > (year+round) > (year+gp_name).
    """
    if next_race:
        return get_next_event()

    if year is None:
        year = datetime.now().year

    if round_num is not None:
        schedule = get_schedule(year)
        row = schedule[schedule["RoundNumber"] == round_num]
        if row.empty:
            raise ValueError(f"Round {round_num} not found in {year} schedule.")
        return year, round_num, _event_dict(row.iloc[0])

    if gp is not None:
        schedule = get_schedule(year)
        # Case-insensitive substring match on EventName
        mask = schedule["EventName"].str.lower().str.contains(gp.lower(), na=False)
        matches = schedule[mask]
        if matches.empty:
            # Try OfficialEventName
            mask2 = schedule["OfficialEventName"].str.lower().str.contains(gp.lower(), na=False)
            matches = schedule[mask2]
        if matches.empty:
            raise ValueError(f"GP '{gp}' not found in {year} schedule.")
        row = matches.iloc[0]
        return year, int(row["RoundNumber"]), _event_dict(row)

    raise ValueError("Provide --round, --gp, or --next.")


def _event_dict(row: pd.Series) -> dict:
    """Convert a schedule row to a plain dict with the fields we need."""
    d = {
        "name": row.get("EventName", ""),
        "official_name": row.get("OfficialEventName", ""),
        "round": int(row.get("RoundNumber", 0)),
        "location": row.get("Location", ""),
        "country": row.get("Country", ""),
        "circuit_short": row.get("CircuitShortName", ""),
        "format": row.get("EventFormat", "conventional"),
    }
    # Race date
    for col in ("Session5DateUtc", "Session5Date", "EventDate"):
        if col in row.index and pd.notna(row[col]):
            d["race_date"] = str(row[col])
            break
    return d


def available_sessions(year: int, round_num: int) -> dict[str, bool]:
    """
    Check which sessions have data available (i.e., have already occurred).
    Returns a dict like {"FP1": True, "FP2": True, "FP3": False, "Q": False, "R": False}.
    """
    now = datetime.now(tz=timezone.utc)
    try:
        schedule = get_schedule(year)
        row = schedule[schedule["RoundNumber"] == round_num]
        if row.empty:
            return {}
        row = row.iloc[0]
    except Exception:
        return {}

    result = {}
    session_map = {
        "FP1": ("Session1", "Session1DateUtc", "Session1Date"),
        "FP2": ("Session2", "Session2DateUtc", "Session2Date"),
        "FP3": ("Session3", "Session3DateUtc", "Session3Date"),
        "Q": ("Session4", "Session4DateUtc", "Session4Date"),
        "R": ("Session5", "Session5DateUtc", "Session5Date"),
    }
    for session_key, (name_col, utc_col, local_col) in session_map.items():
        session_name = row.get(name_col, "")
        # Skip if session doesn't exist (e.g., no FP3 on sprint weekends)
        if not session_name or str(session_name).strip() in ("", "nan"):
            result[session_key] = False
            continue
        date = None
        for col in (utc_col, local_col):
            if col in row.index and pd.notna(row.get(col)):
                date = pd.to_datetime(row[col], utc=True, errors="coerce")
                break
        result[session_key] = bool(date is not None and date < now)

    return result
