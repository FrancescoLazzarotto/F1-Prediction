"""Domain constants: points systems, DNF classification, team identity, tyres.

Everything here is static F1 domain knowledge with no external dependencies,
so it can be imported from anywhere without triggering network or heavy imports.
"""

from __future__ import annotations

from typing import Final

# ── Points ────────────────────────────────────────────────────────────────────

#: Championship points by finishing position (2010-present regulations).
RACE_POINTS: Final[dict[int, int]] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}

#: Sprint race points (2022-present).
SPRINT_POINTS: Final[dict[int, int]] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

#: Bonus point for fastest lap, awarded only inside the top 10 (through 2024).
FASTEST_LAP_POINT: Final[float] = 1.0


def points_for(position: int, sprint: bool = False) -> float:
    """Championship points awarded for a finishing position."""
    table = SPRINT_POINTS if sprint else RACE_POINTS
    return float(table.get(int(position), 0))


# ── Race classification ───────────────────────────────────────────────────────

#: Statuses meaning the car was classified at the finish.
#:
#: The two data sources spell this differently, and getting it wrong is
#: expensive: Jolpica normalises Ergast's "+1 Lap" to "Lapped", so treating an
#: unrecognised status as a retirement would misclassify roughly a third of
#: every field as a DNF.
CLASSIFIED_STATUSES: Final[frozenset[str]] = frozenset({
    "Finished",   # both sources
    "Lapped",     # Jolpica's spelling for a car one or more laps down
})

#: Ergast's spelling for the same thing: "+1 Lap", "+2 Laps", …
LAPPED_PREFIX: Final[str] = "+"

#: Retirements that are not mechanical or driving failures. Kept for callers
#: that want to separate "the car broke" from "the entry was withdrawn".
NON_MECHANICAL_DNF: Final[frozenset[str]] = frozenset({
    "Disqualified", "Excluded", "Did not qualify", "Did not prequalify",
    "Not classified", "Withdrew", "Did not start", "Injury", "Illness",
})


def is_dnf(status: str | None) -> bool:
    """True when a result status means the car did not see the flag.

    A finish is either an outright classification ("Finished"), or a car one or
    more laps down — which Jolpica writes as "Lapped" and Ergast as "+1 Lap".
    Everything else is a retirement, a disqualification or a non-start.
    """
    if not status:
        return False
    s = str(status).strip()
    return not (s in CLASSIFIED_STATUSES or s.startswith(LAPPED_PREFIX))


# ── Team identity ─────────────────────────────────────────────────────────────

#: Canonical hex colour per constructor, keyed by lowercase substring of the
#: team name. Matched with "first substring that appears in the name wins",
#: which survives sponsor renames (e.g. "Oracle Red Bull Racing").
TEAM_COLORS: Final[dict[str, str]] = {
    "red bull": "#3671C6",
    "rb ": "#6692FF",
    "racing bulls": "#6692FF",
    "alphatauri": "#2B4562",
    "toro rosso": "#2B4562",
    "ferrari": "#E8002D",
    "mercedes": "#27F4D2",
    "mclaren": "#FF8000",
    "aston martin": "#229971",
    "alpine": "#00A1E8",
    "renault": "#FFF500",
    "williams": "#64C4FF",
    "haas": "#B6BABD",
    "kick sauber": "#52E252",
    "sauber": "#52E252",
    "alfa romeo": "#C92D4B",
    "audi": "#00505C",
    "cadillac": "#B3995D",
    "racing point": "#F596C8",
    "force india": "#F596C8",
}

DEFAULT_TEAM_COLOR: Final[str] = "#8A8F98"


def team_color(team_name: str | None) -> str:
    """Best-effort constructor colour for charts and tables."""
    if not team_name:
        return DEFAULT_TEAM_COLOR
    name = f" {str(team_name).lower()} "
    for key, color in TEAM_COLORS.items():
        if key in name:
            return color
    return DEFAULT_TEAM_COLOR


# ── Tyres ─────────────────────────────────────────────────────────────────────

#: Approximate pace offset (seconds/lap) of each compound versus SOFT, used to
#: normalise long-run stints that were set on different rubber.
COMPOUND_OFFSET_S: Final[dict[str, float]] = {
    "SOFT": 0.0,
    "MEDIUM": 0.45,
    "HARD": 0.95,
    "INTERMEDIATE": 6.0,
    "WET": 12.0,
    "UNKNOWN": 0.45,
}

PODIUM_COLORS: Final[tuple[str, str, str]] = ("#FFD700", "#C0C0C0", "#CD7F32")
