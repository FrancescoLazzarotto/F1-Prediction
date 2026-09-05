"""Feature contract shared by the builder, the models and the explainer.

Every feature is **circuit-relative**: gaps are expressed as a percentage of the
reference lap and positions as a fraction of the field. Raw lap times in seconds
are deliberately excluded — a 71 s lap at the Red Bull Ring and a 71 s lap at Spa
mean opposite things, so feeding absolute times to a cross-circuit model teaches
it the calendar rather than the pace.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bump whenever the feature set *or the way a feature is computed* changes.
#: Cached models embed it and refuse to serve predictions built from a
#: different contract, and cached feature frames are keyed by it.
#:
#: 6 - "Lapped" recognised as a finish, not a retirement
#: 5 - team form aggregated per race weekend, closing a teammate leak
#: 4 - split the teammate delta into practice- and qualifying-derived columns
#: 3 - circuit-relative features throughout
SCHEMA_VERSION = 6


@dataclass(frozen=True, slots=True)
class Feature:
    """One model input: its name, a human explanation, and a neutral fallback."""

    name: str
    label: str
    #: Value used when the feature genuinely cannot be computed for a driver.
    #: Chosen to be "average driver" rather than 0, which would be off-scale.
    neutral: float
    #: True when a *lower* value means a *better* expected result. Used by the
    #: explainer to phrase a contribution as a strength or a weakness.
    lower_is_better: bool = True


def _f(name: str, label: str, neutral: float, lower_is_better: bool = True) -> Feature:
    return Feature(name, label, neutral, lower_is_better)


# ── Race model ────────────────────────────────────────────────────────────────

RACE_FEATURES: tuple[Feature, ...] = (
    _f("grid_pos", "Starting grid position", 10.0),
    _f("grid_pos_pct", "Grid position, share of field", 0.5),
    _f("quali_gap_to_pole_pct", "Qualifying gap to pole (%)", 1.0),
    _f("reached_q3", "Reached Q3", 0.0, lower_is_better=False),
    _f("fp_pace_gap_pct", "Long-run pace gap in practice (%)", 1.0),
    _f("fp_best_gap_pct", "Best practice lap gap (%)", 1.0),
    _f("fp_rank_pct", "Practice rank, share of field", 0.5),
    _f("form_avg_pos", "Average finish, recent races", 10.0),
    _f("form_avg_pts", "Average points, recent races", 4.0, lower_is_better=False),
    _f("form_pos_gain", "Places gained vs grid, recent races", 0.0, lower_is_better=False),
    _f("form_trend", "Form trend (negative = improving)", 0.0),
    _f("form_dnf_rate", "Retirement rate, long window", 0.09),
    _f("circuit_avg_pos", "Average finish at this circuit", 10.0),
    _f("circuit_n_starts", "Starts at this circuit", 0.0, lower_is_better=False),
    _f("team_avg_pos", "Team average finish", 10.0),
    _f("team_avg_pts", "Team average points", 4.0, lower_is_better=False),
    _f("team_dnf_rate", "Team retirement rate", 0.09),
    _f("teammate_quali_delta_pct", "Qualifying gap to teammate (%)", 0.0),
    _f("weather_temp", "Air temperature (C)", 22.0, lower_is_better=False),
    _f("weather_rain_prob", "Rain probability", 0.1),
    _f("weather_wind", "Wind speed (m/s)", 3.0),
    _f("season_progress", "Season progress", 0.5, lower_is_better=False),
)

# ── Qualifying model ──────────────────────────────────────────────────────────

QUALI_FEATURES: tuple[Feature, ...] = (
    _f("fp_best_gap_pct", "Best practice lap gap (%)", 1.0),
    _f("fp_pace_gap_pct", "Long-run pace gap in practice (%)", 1.0),
    _f("fp_rank_pct", "Practice rank, share of field", 0.5),
    _f("fp_theory_gap_pct", "Theoretical-best-lap gap (%)", 1.0),
    _f("fp_n_laps", "Practice laps completed", 15.0, lower_is_better=False),
    _f("form_avg_pos", "Average finish, recent races", 10.0),
    _f("form_avg_quali_pos", "Average qualifying position, recent races", 10.0),
    _f("form_avg_pts", "Average points, recent races", 4.0, lower_is_better=False),
    _f("team_avg_pos", "Team average finish", 10.0),
    _f("team_avg_quali_pos", "Team average qualifying position", 10.0),
    # Practice-derived, never qualifying-derived: this model predicts the
    # qualifying session, so a feature built from its result would leak.
    _f("teammate_fp_delta_pct", "Practice gap to teammate (%)", 0.0),
)

# ── Retirement model ──────────────────────────────────────────────────────────

DNF_FEATURES: tuple[Feature, ...] = (
    _f("form_dnf_rate", "Retirement rate, long window", 0.09),
    _f("team_dnf_rate", "Team retirement rate", 0.09),
    _f("grid_pos_pct", "Grid position, share of field", 0.5),
    _f("weather_rain_prob", "Rain probability", 0.1),
    _f("circuit_n_starts", "Starts at this circuit", 0.0, lower_is_better=False),
    _f("season_progress", "Season progress", 0.5, lower_is_better=False),
)


def names(features: tuple[Feature, ...]) -> list[str]:
    """Column names, in the exact order the model expects them."""
    return [f.name for f in features]


def neutrals(features: tuple[Feature, ...]) -> dict[str, float]:
    """Fallback value per feature, used when a driver has no data at all."""
    return {f.name: f.neutral for f in features}


def by_name(name: str) -> Feature | None:
    """Look up a :class:`Feature` across every model's schema."""
    for group in (RACE_FEATURES, QUALI_FEATURES, DNF_FEATURES):
        for feat in group:
            if feat.name == name:
                return feat
    return None


RACE_FEATURE_COLS = names(RACE_FEATURES)
QUALI_FEATURE_COLS = names(QUALI_FEATURES)
DNF_FEATURE_COLS = names(DNF_FEATURES)

#: Union of every column the builder must produce.
ALL_FEATURE_COLS = sorted(set(RACE_FEATURE_COLS) | set(QUALI_FEATURE_COLS) | set(DNF_FEATURE_COLS))
