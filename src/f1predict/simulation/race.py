"""Monte Carlo race simulation on top of the model's pace scores.

A point prediction says who is fastest; it does not say how likely they are to
win. This module turns scores into probabilities by replaying the race many
thousands of times with two sources of randomness the model cannot see:

* **pace scatter** — a start, a safety car, a bad pit stop. Wider for the
  midfield than the front row, and wider still in the wet.
* **retirement** — sampled per driver from the reliability model, with the
  retiring car classified behind everyone who finished.

The whole thing is one vectorised NumPy expression, so 20 000 simulations of a
20-car field costs a few milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1predict.config import SimulationConfig
from f1predict.constants import points_for

#: Score penalty that pushes a retirement behind every finisher without
#: disturbing the relative order of the cars that did finish.
_DNF_PENALTY = 1_000.0


@dataclass(slots=True)
class RaceSimulation:
    """Outcome distribution for one simulated race."""

    #: (n_simulations, n_drivers) integer finishing positions.
    positions: np.ndarray
    #: (n_simulations, n_drivers) boolean retirement mask.
    retirements: np.ndarray
    driver_codes: list[str]
    n_simulations: int

    def probabilities(self) -> pd.DataFrame:
        """Per-driver outcome probabilities and expected results."""
        pos = self.positions
        finished = ~self.retirements

        return pd.DataFrame({
            "driver_code": self.driver_codes,
            "p_win": (pos == 1).mean(axis=0),
            "p_podium": (pos <= 3).mean(axis=0),
            "p_top5": (pos <= 5).mean(axis=0),
            "p_top10": (pos <= 10).mean(axis=0),
            "p_points": ((pos <= 10) & finished).mean(axis=0),
            "p_dnf": self.retirements.mean(axis=0),
            "expected_pos": pos.mean(axis=0),
            "expected_points": self._expected_points(),
            "pos_p10": np.percentile(pos, 10, axis=0),
            "pos_p90": np.percentile(pos, 90, axis=0),
        })

    def _expected_points(self) -> np.ndarray:
        """Mean championship points across every simulated race."""
        lookup = np.zeros(self.positions.max() + 2, dtype="float64")
        for position in range(1, len(lookup)):
            lookup[position] = points_for(position)
        awarded = lookup[self.positions] * ~self.retirements
        return awarded.mean(axis=0)

    def position_distribution(self, max_position: int = 20) -> pd.DataFrame:
        """P(finishing in position k) per driver — a heat-map-shaped frame."""
        counts = np.zeros((len(self.driver_codes), max_position), dtype="float64")
        for slot in range(1, max_position + 1):
            counts[:, slot - 1] = (self.positions == slot).mean(axis=0)
        return pd.DataFrame(
            counts, index=self.driver_codes, columns=range(1, max_position + 1)
        )

    def head_to_head(self, a: str, b: str) -> float:
        """Probability that ``a`` finishes ahead of ``b``."""
        try:
            i, j = self.driver_codes.index(a), self.driver_codes.index(b)
        except ValueError as exc:
            raise KeyError(f"Unknown driver code in head-to-head: {exc}") from exc
        return float((self.positions[:, i] < self.positions[:, j]).mean())


def simulate_race(
    scores: np.ndarray,
    driver_codes: list[str],
    dnf_probabilities: np.ndarray | None = None,
    grid_positions: np.ndarray | None = None,
    rain_probability: float = 0.0,
    cfg: SimulationConfig | None = None,
    seed: int | None = None,
) -> RaceSimulation:
    """Replay a race many times and record where everyone finished.

    Args:
        scores: Model pace score per driver; lower is better.
        driver_codes: Driver labels, aligned with ``scores``.
        dnf_probabilities: Per-driver retirement probability. Defaults to the
            configured base rate.
        grid_positions: Starting positions, used to widen the spread down the
            field. Defaults to the order implied by ``scores``.
        rain_probability: Scales the spread; a wet race is far less predictable.
    """
    cfg = cfg or SimulationConfig()
    scores = np.asarray(scores, dtype="float64")
    n_drivers = len(scores)
    if n_drivers == 0:
        raise ValueError("Cannot simulate a race with no drivers.")

    n_sims = max(int(cfg.n_simulations), 1)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    sigma = _per_driver_sigma(scores, grid_positions, rain_probability, cfg, n_drivers)

    # One draw per driver per simulation, then rank within each simulation.
    noisy = scores[None, :] + rng.normal(0.0, 1.0, size=(n_sims, n_drivers)) * sigma[None, :]

    if dnf_probabilities is None:
        dnf_probabilities = np.full(n_drivers, cfg.base_dnf_rate)
    dnf_probabilities = np.clip(np.asarray(dnf_probabilities, dtype="float64"), 0.0, 1.0)
    retirements = rng.random((n_sims, n_drivers)) < dnf_probabilities[None, :]

    # Retirements sort behind the finishers but keep their relative order, so a
    # fast car that retires is still classified ahead of a slow one that did.
    noisy = noisy + retirements * _DNF_PENALTY

    # Double argsort turns scores into 1-based ranks along each row.
    order = np.argsort(noisy, axis=1, kind="stable")
    positions = np.empty_like(order)
    np.put_along_axis(
        positions, order,
        np.broadcast_to(np.arange(1, n_drivers + 1), order.shape), axis=1,
    )

    return RaceSimulation(
        positions=positions, retirements=retirements,
        driver_codes=list(driver_codes), n_simulations=n_sims,
    )


def _per_driver_sigma(
    scores: np.ndarray,
    grid_positions: np.ndarray | None,
    rain_probability: float,
    cfg: SimulationConfig,
    n_drivers: int,
) -> np.ndarray:
    """Spread per driver: wider down the grid, wider in the rain.

    A front-row car has a narrow band of plausible outcomes; a midfield car can
    plausibly finish anywhere from fourth to out of the points, so a single
    global standard deviation would misprice both ends of the grid.
    """
    if grid_positions is None:
        grid_positions = pd.Series(scores).rank(method="first").to_numpy(dtype="float64")
    grid = np.asarray(grid_positions, dtype="float64")

    # 0 at the front of the grid, 1 at the back.
    depth = np.clip((grid - 1.0) / max(n_drivers - 1, 1), 0.0, 1.0)
    sigma = cfg.position_noise_std * (1.0 + cfg.backmarker_noise_scale * depth)

    wet = 1.0 + (cfg.wet_noise_multiplier - 1.0) * float(np.clip(rain_probability, 0.0, 1.0))
    return sigma * wet


def summarise(
    meta: pd.DataFrame,
    scores: np.ndarray,
    simulation: RaceSimulation,
) -> pd.DataFrame:
    """Join simulation probabilities onto driver metadata, ordered by pace."""
    probabilities = simulation.probabilities()

    out = meta.reset_index(drop=True).copy()
    out["score"] = scores
    out = out.merge(probabilities, on="driver_code", how="left")
    out["predicted_pos"] = pd.Series(scores).rank(method="first").astype(int)
    return out.sort_values("predicted_pos").reset_index(drop=True)
