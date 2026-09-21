"""Monte Carlo simulation of a qualifying session from practice pace.

The qualifying model produces one score per driver; ranking those scores gives
an order but says nothing about how safe each slot is. Pole by three tenths and
pole by three thousandths are the same prediction and very different bets.

This module replays the session many thousands of times with the randomness the
model cannot see — the lap each driver actually puts together when it counts —
and reports the share of runs they spent in each grid slot.

Two things set a driver's spread apart from the field's:

* **measured model error** — the spread is solved for, so that the simulated
  mean position error matches the qualifying model's own cross-validated MAE.
  Being wrong by the amount the model is historically wrong by is the only
  defensible default.
* **pack density** — a driver sitting in the middle of a tight practice pack
  can lose four places to a hundredth; one with clear air either side cannot.
  Rain widens everything.

Unlike the race there is no retirement model here. A driver failing to set a
time is a different and far rarer event than a mechanical retirement over race
distance, and modelling it would need failure data this project does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1predict.config import SimulationConfig

#: Knockout cut-offs in the current format: Q1 drops to 15 cars, Q2 to 10.
_Q3_CUT = 10
_Q2_CUT = 15

#: Bounds on the pack-density multiplier. A driver buried in a tight pack is
#: less sure of their slot, but never so much that the simulation stops
#: resembling the order the model actually predicted.
_DENSITY_BOUNDS = (0.65, 1.8)

#: Probe size, seed, bracket and step count for the spread calibration below.
#: The seed is fixed rather than taken from the config so that the calibration
#: is a property of the model's error and the field size alone — two runs with
#: different simulation seeds get the same spread, only different draws from it.
_PROBE_SIMS = 1_500
_PROBE_SEED = 20_260
_SIGMA_BRACKET = (0.0, 10.0)
_BISECTION_STEPS = 18


@dataclass(slots=True)
class QualiSimulation:
    """Outcome distribution for one simulated qualifying session."""

    #: (n_simulations, n_drivers) integer grid slots.
    positions: np.ndarray
    driver_codes: list[str]
    n_simulations: int
    #: Per-driver spread actually used, in model-score units. Kept so the UI can
    #: say *why* one driver's distribution is flatter than another's.
    sigma: np.ndarray

    def probabilities(self) -> pd.DataFrame:
        """Per-driver session outcome probabilities."""
        pos = self.positions
        n_drivers = len(self.driver_codes)
        distribution = self.position_distribution(max_position=n_drivers)

        return pd.DataFrame({
            "driver_code": self.driver_codes,
            "p_pole": (pos == 1).mean(axis=0),
            "p_front_row": (pos <= 2).mean(axis=0),
            "p_top3": (pos <= 3).mean(axis=0),
            "p_q3": (pos <= min(_Q3_CUT, n_drivers)).mean(axis=0),
            "p_q2": (pos <= min(_Q2_CUT, n_drivers)).mean(axis=0),
            "expected_quali_pos": pos.mean(axis=0),
            "most_likely_quali_pos": distribution.to_numpy().argmax(axis=1) + 1,
            "quali_pos_p10": np.percentile(pos, 10, axis=0),
            "quali_pos_p90": np.percentile(pos, 90, axis=0),
            "quali_sigma": self.sigma,
        })

    def position_distribution(self, max_position: int = 20) -> pd.DataFrame:
        """P(qualifying in position k) per driver — a heat-map-shaped frame."""
        counts = np.zeros((len(self.driver_codes), max_position), dtype="float64")
        for slot in range(1, max_position + 1):
            counts[:, slot - 1] = (self.positions == slot).mean(axis=0)
        return pd.DataFrame(
            counts, index=self.driver_codes, columns=range(1, max_position + 1)
        )

    def driver_odds(self, driver_code: str) -> pd.DataFrame:
        """Every grid slot one driver could take, with a cumulative column.

        This is the per-driver view the rest of the app renders: one row per
        position, the share of simulated sessions that ended there, and the
        running total, so "fifth or better" can be read straight off.
        """
        column = self.positions[:, self._index(driver_code)]
        n_drivers = len(self.driver_codes)
        probability = np.array(
            [(column == slot).mean() for slot in range(1, n_drivers + 1)]
        )
        return pd.DataFrame({
            "position": np.arange(1, n_drivers + 1),
            "probability": probability,
            "cumulative": probability.cumsum(),
        })

    def interval(self, driver_code: str, mass: float = 0.8) -> tuple[int, int]:
        """Shortest run of grid slots holding ``mass`` of a driver's outcomes.

        Preferred over a P10-P90 pair because the distribution is discrete and
        often lopsided: "P3 to P6, 80% of the time" is a claim someone can act
        on, whereas symmetric percentiles straddle slots the driver rarely
        takes.
        """
        odds = self.driver_odds(driver_code)["probability"].to_numpy()
        n_slots = len(odds)
        best, best_width = (1, n_slots), n_slots

        for low in range(n_slots):
            total = 0.0
            for high in range(low, n_slots):
                total += odds[high]
                if total >= mass:
                    if high - low < best_width:
                        best, best_width = (low + 1, high + 1), high - low
                    break
        return best

    def head_to_head(self, a: str, b: str) -> float:
        """Probability that ``a`` out-qualifies ``b``."""
        return float((self.positions[:, self._index(a)] < self.positions[:, self._index(b)]).mean())

    def _index(self, driver_code: str) -> int:
        try:
            return self.driver_codes.index(driver_code)
        except ValueError as exc:
            raise KeyError(
                f"{driver_code} did not take part in this simulated session."
            ) from exc


def simulate_qualifying(
    scores: np.ndarray,
    driver_codes: list[str],
    pace_gaps: np.ndarray | None = None,
    model_position_error: float = float("nan"),
    rain_probability: float = 0.0,
    cfg: SimulationConfig | None = None,
    seed: int | None = None,
) -> QualiSimulation:
    """Replay a qualifying session many times and record every grid slot.

    Args:
        scores: Qualifying model score per driver; lower is better.
        driver_codes: Driver labels, aligned with ``scores``.
        pace_gaps: Practice gap per driver (``fp_best_gap_pct``), used to widen
            the spread for drivers hemmed in by close rivals. Omit for a field
            spread evenly.
        model_position_error: The qualifying model's cross-validated MAE, in
            positions. The spread is calibrated to reproduce it.
        rain_probability: Scales the spread; a wet session is a lottery.
    """
    cfg = cfg or SimulationConfig()
    scores = np.asarray(scores, dtype="float64")
    n_drivers = len(scores)
    if n_drivers == 0:
        raise ValueError("Cannot simulate a qualifying session with no drivers.")

    n_sims = max(int(cfg.n_simulations), 1)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    sigma = _per_driver_sigma(
        scores, pace_gaps, model_position_error, rain_probability, cfg
    )
    noisy = scores[None, :] + rng.normal(0.0, 1.0, size=(n_sims, n_drivers)) * sigma[None, :]

    return QualiSimulation(
        positions=_rank_rows(noisy),
        driver_codes=list(driver_codes),
        n_simulations=n_sims,
        sigma=sigma,
    )


def summarise_quali(
    meta: pd.DataFrame,
    scores: np.ndarray,
    simulation: QualiSimulation,
) -> pd.DataFrame:
    """Join qualifying probabilities onto driver metadata, ordered by pace.

    ``predicted_quali_pos`` stays the deterministic rank of the model score and
    never the simulated average: that column becomes the race model's grid when
    qualifying has not run, and the hand-off has to be reproducible.
    """
    out = meta.reset_index(drop=True).copy()
    out["quali_score"] = scores
    out["predicted_quali_pos"] = pd.Series(scores).rank(method="first").astype(int)
    out = out.merge(simulation.probabilities(), on="driver_code", how="left")
    return out.sort_values("predicted_quali_pos").reset_index(drop=True)


# ── Spread ────────────────────────────────────────────────────────────────────

def _per_driver_sigma(
    scores: np.ndarray,
    pace_gaps: np.ndarray | None,
    model_position_error: float,
    rain_probability: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Spread per driver: a shape across the field, scaled to the model's error.

    Splitting the two matters. The shape says who is *relatively* less certain;
    the scale is then solved so that the field as a whole is wrong by exactly as
    much as the model has been measured to be. Folding the two together would
    let a grid full of tightly packed cars quietly inflate every probability.
    """
    n_drivers = len(scores)
    # 0-based predicted position per driver, via the usual double argsort.
    rank = np.argsort(np.argsort(scores, kind="stable"), kind="stable")

    shape = _spread_shape(rank, pace_gaps, cfg, n_drivers)
    by_rank = np.empty(n_drivers, dtype="float64")
    by_rank[rank] = shape

    base = _calibrated_scale(by_rank, model_position_error, cfg, n_drivers)
    wet = 1.0 + (cfg.wet_noise_multiplier - 1.0) * float(np.clip(rain_probability, 0.0, 1.0))
    return base * shape * wet


def _spread_shape(
    rank: np.ndarray, pace_gaps: np.ndarray | None, cfg: SimulationConfig, n_drivers: int
) -> np.ndarray:
    """Relative spread per driver, normalised so the field averages one.

    Two effects a flat field would misprice. A pole contender's slot is far
    safer than a midfielder's — reusing the race simulation's front-to-back
    ramp, because the reason is the same one. And a car buried in a tight pack
    is less sure of its slot than one with clear air either side.
    """
    depth = rank.astype("float64") / max(n_drivers - 1, 1)
    shape = (1.0 + cfg.backmarker_noise_scale * depth) * _pack_density(pace_gaps, n_drivers)
    mean = float(shape.mean())
    return shape / mean if mean > 0 else np.ones(n_drivers)


def _calibrated_scale(
    shape: np.ndarray, target_mae: float, cfg: SimulationConfig, n_drivers: int
) -> float:
    """Scale whose simulated position error matches ``target_mae``.

    The qualifying model reports a cross-validated mean absolute error in
    positions, which is the most honest statement available about how wrong it
    usually is. Solving for the spread that reproduces it ties the published
    probabilities to measured accuracy rather than to a hand-picked constant —
    retrain on more data and the distributions tighten by themselves.

    ``shape`` arrives ordered by predicted position, so the probe field is the
    one actually being simulated rather than an evenly spread stand-in.
    """
    if not np.isfinite(target_mae) or target_mae <= 0.0 or n_drivers < 2:
        return float(cfg.quali_position_noise_std)

    ladder = np.arange(1.0, n_drivers + 1)[None, :]
    # One noise draw, reused at every step. With common random numbers the
    # measured error is monotone in the scale, which is what makes a bisection
    # over an otherwise stochastic objective behave.
    noise = (
        np.random.default_rng(_PROBE_SEED).normal(size=(_PROBE_SIMS, n_drivers))
        * shape[None, :]
    )

    def error_at(scale: float) -> float:
        return float(np.abs(_rank_rows(ladder + noise * scale) - ladder).mean())

    low, high = _SIGMA_BRACKET
    # A short field cannot be wrong by more than a few places however noisy it
    # gets, so an unreachable target saturates instead of running away.
    if error_at(high) <= target_mae:
        return high

    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if error_at(mid) < target_mae:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _pack_density(pace_gaps: np.ndarray | None, n_drivers: int) -> np.ndarray:
    """Per-driver multiplier from how crowded the order is around them.

    Grid slots are decided by hundredths in the midfield and by tenths at the
    front, so the same lap-time error costs a very different number of places
    depending on who is parked either side. Drivers packed tighter than the
    field's typical spacing get a wider spread, and drivers in clear air a
    narrower one.
    """
    if pace_gaps is None:
        return np.ones(n_drivers)

    pace = pd.to_numeric(pd.Series(pace_gaps), errors="coerce").to_numpy(dtype="float64")
    if len(pace) != n_drivers or not np.isfinite(pace).all():
        return np.ones(n_drivers)

    order = np.argsort(pace, kind="stable")
    step = np.diff(pace[order])
    if len(step) == 0 or not np.any(step > 0):
        return np.ones(n_drivers)

    # Mean distance to the neighbours either side. The two cars at the ends of
    # the order have only one neighbour, which is counted twice rather than
    # treated as clear air.
    neighbour = 0.5 * (
        np.concatenate([step[:1], step]) + np.concatenate([step, step[-1:]])
    )
    typical = float(np.median(neighbour))
    if typical <= 0.0:
        return np.ones(n_drivers)

    factor = np.empty(n_drivers, dtype="float64")
    factor[order] = np.clip(typical / np.maximum(neighbour, 1e-9), *_DENSITY_BOUNDS)
    return factor


def _rank_rows(values: np.ndarray) -> np.ndarray:
    """1-based ranks along each row — the double argsort, vectorised."""
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(np.arange(1, values.shape[1] + 1), order.shape), axis=1,
    )
    return ranks
