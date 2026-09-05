"""Event resolution: next race, year+round, GP name, session availability."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

#: Session key -> the FastF1 schedule column group that describes it.
_SESSION_SLOTS: dict[str, int] = {"FP1": 1, "FP2": 2, "FP3": 3, "Q": 4, "R": 5}

#: On a sprint weekend FastF1 fills the slots differently; these are the names
#: we treat as each logical session regardless of which numbered slot holds them.
_SESSION_ALIASES: dict[str, tuple[str, ...]] = {
    "FP1": ("Practice 1",),
    "FP2": ("Practice 2",),
    "FP3": ("Practice 3",),
    "SQ": ("Sprint Qualifying", "Sprint Shootout"),
    "S": ("Sprint",),
    "Q": ("Qualifying",),
    "R": ("Race",),
}


@dataclass(frozen=True, slots=True)
class Event:
    """One Grand Prix weekend."""

    year: int
    round: int
    name: str = ""
    official_name: str = ""
    location: str = ""
    country: str = ""
    circuit_short: str = ""
    event_format: str = "conventional"
    race_date: datetime | None = None
    sessions: dict[str, datetime] = field(default_factory=dict)

    @property
    def is_sprint(self) -> bool:
        return "sprint" in str(self.event_format).lower()

    @property
    def label(self) -> str:
        return f"{self.name} {self.year} (R{self.round})"

    def time_until(self, now: datetime | None = None) -> timedelta | None:
        """Time left before lights out, or ``None`` when the date is unknown."""
        if self.race_date is None:
            return None
        now = now or datetime.now(tz=timezone.utc)
        return self.race_date - now

    def to_dict(self) -> dict:
        return {
            "year": self.year, "round": self.round, "name": self.name,
            "official_name": self.official_name, "location": self.location,
            "country": self.country, "circuit_short": self.circuit_short,
            "format": self.event_format,
            "race_date": self.race_date.isoformat() if self.race_date else "",
        }


@lru_cache(maxsize=8)
def get_schedule(year: int) -> pd.DataFrame:
    """Full event schedule for a season, memoised per process."""
    import fastf1

    return fastf1.get_event_schedule(year, include_testing=False)


def _utc(value) -> datetime | None:
    """Coerce a schedule cell to a timezone-aware UTC datetime."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _session_times(row: pd.Series) -> dict[str, datetime]:
    """Map logical session keys to their UTC start times.

    FastF1 numbers the five weekend slots but names them differently on sprint
    weekends, so match on the *name* and fall back to the slot number.
    """
    times: dict[str, datetime] = {}
    for slot in range(1, 6):
        name = str(row.get(f"Session{slot}", "") or "").strip()
        when = _utc(row.get(f"Session{slot}DateUtc")) or _utc(row.get(f"Session{slot}Date"))
        if when is None or not name:
            continue
        for key, aliases in _SESSION_ALIASES.items():
            if name in aliases:
                times[key] = when
                break
        else:
            times[f"S{slot}"] = when
    return times


def _event_from_row(row: pd.Series, year: int) -> Event:
    sessions = _session_times(row)
    race_date = (
        sessions.get("R")
        or _utc(row.get("Session5DateUtc"))
        or _utc(row.get("EventDate"))
    )
    return Event(
        year=year,
        round=int(row.get("RoundNumber", 0) or 0),
        name=str(row.get("EventName", "") or ""),
        official_name=str(row.get("OfficialEventName", "") or ""),
        location=str(row.get("Location", "") or ""),
        country=str(row.get("Country", "") or ""),
        circuit_short=str(row.get("CircuitShortName", "") or ""),
        event_format=str(row.get("EventFormat", "conventional") or "conventional"),
        race_date=race_date,
        sessions=sessions,
    )


def season_events(year: int) -> list[Event]:
    """Every round of a season as :class:`Event` objects."""
    try:
        schedule = get_schedule(year)
    except Exception as exc:
        log.warning("Could not load the %d schedule: %s", year, exc)
        return []
    events = [_event_from_row(row, year) for _, row in schedule.iterrows()]
    return [e for e in events if e.round > 0]


def get_next_event(now: datetime | None = None) -> Event:
    """The next race that has not yet started.

    Searches the current season first, then rolls into the next one during the
    winter break.
    """
    now = now or datetime.now(tz=timezone.utc)
    for year in (now.year, now.year + 1):
        upcoming = [
            e for e in season_events(year)
            if e.race_date is not None and e.race_date > now
        ]
        if upcoming:
            return min(upcoming, key=lambda e: e.race_date)
    raise LookupError("Could not determine the next race from the FastF1 schedule.")


def get_last_event(now: datetime | None = None) -> Event:
    """The most recently completed race."""
    now = now or datetime.now(tz=timezone.utc)
    for year in (now.year, now.year - 1):
        past = [e for e in season_events(year) if e.race_date is not None and e.race_date <= now]
        if past:
            return max(past, key=lambda e: e.race_date)
    raise LookupError("Could not determine the last race from the FastF1 schedule.")


def resolve_event(
    year: int | None = None,
    round_num: int | None = None,
    gp: str | None = None,
    next_race: bool = False,
    last_race: bool = False,
) -> Event:
    """Resolve whichever selector the caller supplied into a single event.

    Precedence: ``next_race`` > ``last_race`` > ``round_num`` > ``gp``.
    """
    if next_race:
        return get_next_event()
    if last_race:
        return get_last_event()

    year = year or datetime.now(tz=timezone.utc).year
    events = season_events(year)
    if not events:
        raise LookupError(f"No schedule available for {year}.")

    if round_num is not None:
        for event in events:
            if event.round == round_num:
                return event
        raise ValueError(f"Round {round_num} is not in the {year} calendar.")

    if gp:
        needle = gp.strip().lower()
        haystacks = ("name", "official_name", "location", "country", "circuit_short")
        # Prefer an exact name match before falling back to a substring hit, so
        # "Monaco" does not resolve to "Monaco Historic" style near-duplicates.
        for attr in haystacks:
            for event in events:
                if getattr(event, attr).lower() == needle:
                    return event
        for event in events:
            if any(needle in getattr(event, attr).lower() for attr in haystacks):
                return event
        raise ValueError(f"No {year} event matches {gp!r}.")

    raise ValueError("Specify a round, a GP name, --next or --last.")


def available_sessions(
    year: int, round_num: int, now: datetime | None = None
) -> dict[str, bool]:
    """Which sessions of a weekend have already run.

    A session counts as available once its scheduled start is at least an hour
    in the past, since timing data only lands after the session is under way.
    """
    now = now or datetime.now(tz=timezone.utc)
    grace = timedelta(hours=1)

    for event in season_events(year):
        if event.round != round_num:
            continue
        result = dict.fromkeys(_SESSION_SLOTS, False)
        for key, when in event.sessions.items():
            result[key] = (when + grace) <= now
        return result

    return dict.fromkeys(_SESSION_SLOTS, False)


def event_status(year: int, round_num: int, now: datetime | None = None) -> str:
    """One of ``upcoming``, ``practice``, ``qualifying``, ``race_weekend``, ``completed``."""
    sessions = available_sessions(year, round_num, now)
    if sessions.get("R"):
        return "completed"
    if sessions.get("Q"):
        return "race_weekend"
    if sessions.get("FP2") or sessions.get("FP3"):
        return "qualifying"
    if sessions.get("FP1"):
        return "practice"
    return "upcoming"
