"""Ranking metrics for evaluating a predicted classification against reality.

Mean absolute error on raw positions is easy to read but a poor guide on its
own: a model that predicts the midfield perfectly and the podium randomly can
still score well. Spearman and top-k accuracy are reported alongside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(slots=True)
class RankingMetrics:
    """How close a predicted order was to the real one."""

    n: int = 0
    mae: float = float("nan")
    spearman: float = float("nan")
    top1: float = float("nan")
    top3: float = float("nan")
    top5: float = float("nan")
    top10: float = float("nan")
    exact: float = float("nan")
    within_1: float = float("nan")
    within_3: float = float("nan")
    podium_order: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Spearman {self.spearman:+.3f} · MAE {self.mae:.2f} pos · "
            f"winner {self.top1:.0%} · podium {self.top3:.0%} · "
            f"within 3 {self.within_3:.0%}"
        )


def rank_series(values, ascending: bool = True) -> np.ndarray:
    """Dense 1-based ranks with ties broken by original order."""
    series = pd.Series(np.asarray(values, dtype="float64"))
    return series.rank(method="first", ascending=ascending).to_numpy()


def evaluate_ranking(predicted, actual) -> RankingMetrics:
    """Compare a predicted finishing order with the actual one.

    Both inputs are positions (1 = best) for the same drivers, in the same row
    order. Rows where either side is missing are dropped.
    """
    pred = pd.to_numeric(pd.Series(np.asarray(predicted, dtype="float64")), errors="coerce")
    act = pd.to_numeric(pd.Series(np.asarray(actual, dtype="float64")), errors="coerce")

    mask = pred.notna() & act.notna()
    pred, act = pred[mask].to_numpy(), act[mask].to_numpy()
    n = len(pred)
    if n < 2:
        return RankingMetrics(n=n)

    diff = np.abs(pred - act)
    rho = spearmanr(pred, act).statistic if n >= 3 else float("nan")

    return RankingMetrics(
        n=n,
        mae=float(diff.mean()),
        spearman=float(rho) if rho == rho else float("nan"),
        top1=_set_overlap(pred, act, 1),
        top3=_set_overlap(pred, act, 3),
        top5=_set_overlap(pred, act, 5),
        top10=_set_overlap(pred, act, 10),
        exact=float((diff == 0).mean()),
        within_1=float((diff <= 1).mean()),
        within_3=float((diff <= 3).mean()),
        podium_order=_exact_prefix(pred, act, 3),
    )


def _set_overlap(pred: np.ndarray, act: np.ndarray, k: int) -> float:
    """Share of the actual top-k that the prediction also placed in its top-k."""
    k = min(k, len(pred))
    if k <= 0:
        return float("nan")
    predicted_top = set(np.argsort(pred, kind="stable")[:k])
    actual_top = set(np.argsort(act, kind="stable")[:k])
    return len(predicted_top & actual_top) / k


def _exact_prefix(pred: np.ndarray, act: np.ndarray, k: int) -> float:
    """Share of the first ``k`` places predicted in exactly the right order."""
    k = min(k, len(pred))
    if k <= 0:
        return float("nan")
    predicted_order = np.argsort(pred, kind="stable")[:k]
    actual_order = np.argsort(act, kind="stable")[:k]
    return float((predicted_order == actual_order).mean())


def aggregate(metrics: list[RankingMetrics]) -> RankingMetrics:
    """Average a list of per-race metrics, weighting each race equally."""
    usable = [m for m in metrics if m.n >= 2]
    if not usable:
        return RankingMetrics()

    def mean(attr: str) -> float:
        values = [getattr(m, attr) for m in usable]
        values = [v for v in values if v == v]  # drop NaN
        return float(np.mean(values)) if values else float("nan")

    return RankingMetrics(
        n=sum(m.n for m in usable),
        mae=mean("mae"), spearman=mean("spearman"),
        top1=mean("top1"), top3=mean("top3"), top5=mean("top5"), top10=mean("top10"),
        exact=mean("exact"), within_1=mean("within_1"), within_3=mean("within_3"),
        podium_order=mean("podium_order"),
    )


def brier_score(probabilities, outcomes) -> float:
    """Mean squared error of probabilistic forecasts — lower is better.

    Applied to the Monte Carlo win/podium probabilities, this is the honest test
    of whether the simulated spread is calibrated rather than merely ordered.
    """
    p = np.asarray(probabilities, dtype="float64")
    y = np.asarray(outcomes, dtype="float64")
    if p.size == 0 or p.size != y.size:
        return float("nan")
    return float(np.mean((p - y) ** 2))
