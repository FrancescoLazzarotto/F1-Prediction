"""Season-long Monte Carlo: who actually wins the championship?

Standings answer "who is ahead"; this answers "who is going to win", which is a
different question once you account for how many races are left, how big the
gap is, and how reliable each car has been.

Every remaining round is simulated as an independent race using each driver's
current pace estimate, points are accumulated across the season, and the title
probability is the share of simulated seasons that driver finished on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1predict.config import SimulationConfig
from f1predict.constants import RACE_POINTS, SPRINT_POINTS


@dataclass(slots=True)
class ChampionshipOutlook:
    """Simulated end-of-season outcomes for drivers or constructors."""

    table: pd.DataFrame
    n_simulations: int
    races_remaining: int
    sprints_remaining: int
    #: (n_simulations, n_entrants) final points totals.
    final_points: np.ndarray
    labels: list[str]

    @property
    def is_decided(self) -> bool:
        """True when one entrant wins every simulated season."""
        return bool((self.table["p_title"] >= 0.9995).any())

    def points_percentiles(self, low: int = 5, high: int = 95) -> pd.DataFrame:
        return pd.DataFrame({
            "label": self.labels,
            f"p{low}": np.percentile(self.final_points, low, axis=0),
            "median": np.median(self.final_points, axis=0),
            f"p{high}": np.percentile(self.final_points, high, axis=0),
        })


def simulate_championship(
    standings: pd.DataFrame,
    pace: pd.Series | None = None,
    races_remaining: int = 0,
    sprints_remaining: int = 0,
    dnf_rates: pd.Series | None = None,
    active_keys: set[str] | None = None,
    cfg: SimulationConfig | None = None,
    label_column: str = "driver_name",
    key_column: str = "driver_id",
    seed: int | None = None,
) -> ChampionshipOutlook:
    """Project the championship forward over the remaining rounds.

    Args:
        standings: Current table with ``points`` plus ``key_column`` and
            ``label_column``.
        pace: Pace score per key; lower is better. Defaults to deriving pace
            from the current standings order, which assumes the season so far is
            the best available guide to the rest of it.
        races_remaining: Full Grands Prix still to run.
        sprints_remaining: Sprint races still to run.
        dnf_rates: Per-key retirement probability.
        active_keys: Entrants still racing. Anyone outside this set keeps the
            points they have already scored but cannot add more — without it, a
            driver who left mid-season keeps accumulating imaginary points.
    """
    cfg = cfg or SimulationConfig()

    if standings is None or standings.empty:
        raise ValueError("Championship simulation needs a non-empty standings table.")

    table = standings.reset_index(drop=True).copy()
    keys = table[key_column].astype(str).tolist()
    labels = table.get(label_column, table[key_column]).astype(str).tolist()
    current = pd.to_numeric(table["points"], errors="coerce").fillna(0.0).to_numpy()

    n_entrants = len(table)
    n_sims = max(int(cfg.n_simulations), 1)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    scores = _pace_scores(pace, keys, n_entrants)
    retirement = _dnf_vector(dnf_rates, keys, cfg.base_dnf_rate)
    active = (
        np.array([k in active_keys for k in keys]) if active_keys is not None
        else np.ones(n_entrants, dtype=bool)
    )

    totals = np.tile(current, (n_sims, 1))
    if races_remaining > 0:
        totals += _accumulate(
            n_sims, n_entrants, races_remaining, scores, retirement, RACE_POINTS,
            active, cfg, rng,
        )
    if sprints_remaining > 0:
        totals += _accumulate(
            n_sims, n_entrants, sprints_remaining, scores, retirement, SPRINT_POINTS,
            active, cfg, rng,
        )

    result = _summarise(table, labels, totals, current, label_column, key_column)

    return ChampionshipOutlook(
        table=result, n_simulations=n_sims, races_remaining=races_remaining,
        sprints_remaining=sprints_remaining, final_points=totals, labels=labels,
    )


def _accumulate(
    n_sims: int,
    n_entrants: int,
    n_races: int,
    scores: np.ndarray,
    retirement: np.ndarray,
    points_table: dict[int, int],
    active: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Points gained across ``n_races`` simulated rounds.

    Races are simulated one at a time and summed, rather than as a single big
    array, to keep peak memory at (n_sims x n_entrants) instead of multiplying
    by the number of rounds.
    """
    lookup = np.zeros(n_entrants + 2, dtype="float64")
    for position, value in points_table.items():
        if position <= n_entrants:
            lookup[position] = value

    depth = np.clip(
        (pd.Series(scores).rank(method="first").to_numpy() - 1.0) / max(n_entrants - 1, 1),
        0.0, 1.0,
    )
    sigma = cfg.position_noise_std * (1.0 + cfg.backmarker_noise_scale * depth)

    # Entrants who are no longer racing sort to the back of every simulated
    # grid and score nothing, so they neither gain points nor displace anyone.
    inactive_penalty = np.where(active, 0.0, 1e6)

    gained = np.zeros((n_sims, n_entrants), dtype="float64")
    for _ in range(n_races):
        noisy = scores[None, :] + rng.normal(0.0, 1.0, (n_sims, n_entrants)) * sigma[None, :]
        retired = rng.random((n_sims, n_entrants)) < retirement[None, :]
        noisy = noisy + retired * 1_000.0 + inactive_penalty[None, :]

        order = np.argsort(noisy, axis=1, kind="stable")
        positions = np.empty_like(order)
        np.put_along_axis(
            positions, order,
            np.broadcast_to(np.arange(1, n_entrants + 1), order.shape), axis=1,
        )
        gained += lookup[positions] * ~retired * active[None, :]

    return gained


def _summarise(
    table: pd.DataFrame,
    labels: list[str],
    totals: np.ndarray,
    current: np.ndarray,
    label_column: str,
    key_column: str,
) -> pd.DataFrame:
    """Title, top-3 and points statistics per entrant."""
    # Rank descending by final points; ties go to the higher current total,
    # a reasonable stand-in for the countback rules.
    order = np.lexsort((-current[None, :].repeat(len(totals), 0), -totals), axis=1)
    final_rank = np.empty_like(order)
    np.put_along_axis(
        final_rank, order,
        np.broadcast_to(np.arange(1, totals.shape[1] + 1), order.shape), axis=1,
    )

    out = pd.DataFrame({
        key_column: table[key_column].to_numpy(),
        label_column: labels,
        "current_points": current,
        "p_title": (final_rank == 1).mean(axis=0),
        "p_top3": (final_rank <= 3).mean(axis=0),
        "expected_points": totals.mean(axis=0),
        "points_p10": np.percentile(totals, 10, axis=0),
        "points_p90": np.percentile(totals, 90, axis=0),
    })

    for extra in ("team", "constructor_id", "driver_code", "wins"):
        if extra in table.columns and extra not in out.columns:
            out[extra] = table[extra].to_numpy()

    return out.sort_values("p_title", ascending=False).reset_index(drop=True)


def _pace_scores(pace: pd.Series | None, keys: list[str], n: int) -> np.ndarray:
    """Pace score per entrant, falling back to the standings order."""
    if pace is not None and len(pace) > 0:
        mapped = pd.Series(keys).map(pace)
        if mapped.notna().any():
            # An entrant with no pace estimate sits at the field's median.
            return mapped.fillna(mapped.median()).to_numpy(dtype="float64")
    return np.arange(1, n + 1, dtype="float64")


def _dnf_vector(dnf_rates: pd.Series | None, keys: list[str], base: float) -> np.ndarray:
    if dnf_rates is not None and len(dnf_rates) > 0:
        mapped = pd.Series(keys).map(dnf_rates)
        return np.clip(mapped.fillna(base).to_numpy(dtype="float64"), 0.0, 0.6)
    return np.full(len(keys), base)


def remaining_rounds(
    schedule: pd.DataFrame, completed_round: int
) -> tuple[int, int]:
    """Count the races and sprints still to run after ``completed_round``.

    Sprint counting needs the FastF1 calendar, which the Ergast schedule frame
    does not carry, so the sprint total is reported as 0 when unknown.
    """
    if schedule is None or schedule.empty or "round" not in schedule.columns:
        return 0, 0
    remaining = schedule[pd.to_numeric(schedule["round"], errors="coerce") > completed_round]
    n_races = len(remaining)
    n_sprints = int(remaining.get("is_sprint", pd.Series(dtype=bool)).sum()) \
        if "is_sprint" in remaining.columns else 0
    return n_races, n_sprints
