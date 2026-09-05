"""The trainable models: a ranking regressor and a retirement classifier.

Both wrap a scikit-learn pipeline and carry their own feature contract, so a
model loaded from disk knows exactly which columns it needs and in what order —
a mismatch raises instead of silently predicting from misaligned data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from f1predict.features.schema import Feature, names
from f1predict.models.base import (
    SCALE_SENSITIVE_KINDS,
    ModelMetadata,
    TrainingReport,
    build_pipeline,
    make_classifier,
    make_regressor,
)
from f1predict.models.metrics import evaluate_ranking

log = logging.getLogger(__name__)


class FeatureMismatchError(ValueError):
    """Raised when input columns do not satisfy a model's feature contract."""


class NotTrainedError(RuntimeError):
    """Raised when predict is called before fit."""


class RankingPredictor:
    """Predicts a continuous score per driver; lower means a better result.

    The absolute score is not meaningful — only the order is. Training targets
    are finishing positions, so the score lands roughly on the position scale,
    which makes it readable without ever being used as a literal position.
    """

    def __init__(
        self,
        features: tuple[Feature, ...],
        kind: str = "hist_gradient_boosting",
        params: dict | None = None,
        metadata: ModelMetadata | None = None,
    ):
        self.features = features
        self.kind = kind
        self.params = params or {}
        self.metadata = metadata or ModelMetadata(kind=kind)
        self._pipeline = build_pipeline(
            make_regressor(kind, self.params), scale=kind in SCALE_SENSITIVE_KINDS
        )
        self._fitted = False
        self.report = TrainingReport()

    @property
    def feature_names(self) -> list[str]:
        return names(self.features)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
        cv_folds: int = 5,
        min_samples_for_cv: int = 60,
    ) -> TrainingReport:
        """Fit the pipeline and measure it with race-grouped cross-validation.

        Args:
            groups: Race identifier per row. Folds are split on it so no race
                appears in both training and validation — without that, two
                drivers from the same race leak each other's context.
        """
        X_sel = self._select(X)
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")

        valid = ~np.isnan(y_arr)
        X_sel, y_arr = X_sel[valid], y_arr[valid]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype="float64")[valid]
        groups_arr = (
            pd.Series(groups).to_numpy()[valid] if groups is not None else None
        )

        if len(y_arr) < 10:
            raise ValueError(f"Not enough training rows: {len(y_arr)}")

        report = TrainingReport(n_samples=len(y_arr), n_features=X_sel.shape[1])

        if groups_arr is not None and len(y_arr) >= min_samples_for_cv:
            self._cross_validate(X_sel, y_arr, groups_arr, sample_weight, cv_folds, report)
        else:
            report.notes.append("Cross-validation skipped: too few samples.")

        self._pipeline.fit(X_sel, y_arr, **self._weight_kwargs(sample_weight))
        self._fitted = True

        report.train_mae = float(mean_absolute_error(y_arr, self._pipeline.predict(X_sel)))
        report.feature_importance = self._importances(X_sel, y_arr)
        self.report = report

        log.info("%s trained: %s", type(self).__name__, report.summary())
        return report

    def _cross_validate(self, X, y, groups, sample_weight, cv_folds, report) -> None:
        """Race-grouped CV, scoring each held-out race as a ranking problem."""
        n_groups = len(np.unique(groups))
        folds = min(cv_folds, n_groups)
        if folds < 2:
            report.notes.append("Cross-validation skipped: fewer than 2 race groups.")
            return

        splitter = GroupKFold(n_splits=folds)
        maes, per_race = [], []

        for train_idx, test_idx in splitter.split(X, y, groups):
            pipeline = build_pipeline(
                make_regressor(self.kind, self.params),
                scale=self.kind in SCALE_SENSITIVE_KINDS,
            )
            weights = sample_weight[train_idx] if sample_weight is not None else None
            pipeline.fit(X.iloc[train_idx], y[train_idx], **self._weight_kwargs(weights))
            predicted = pipeline.predict(X.iloc[test_idx])

            maes.append(mean_absolute_error(y[test_idx], predicted))

            # Score each held-out race separately: ranking quality is a
            # within-race property, and pooling races would compare drivers
            # who never raced each other.
            fold_groups = groups[test_idx]
            for race in np.unique(fold_groups):
                mask = fold_groups == race
                if mask.sum() >= 3:
                    per_race.append(evaluate_ranking(predicted[mask], y[test_idx][mask]))

        report.cv_folds = folds
        report.cv_mae = float(np.mean(maes)) if maes else float("nan")
        if per_race:
            spearmans = [m.spearman for m in per_race if m.spearman == m.spearman]
            report.cv_spearman = float(np.mean(spearmans)) if spearmans else float("nan")
            report.cv_top3 = float(np.mean([m.top3 for m in per_race]))

    def _weight_kwargs(self, sample_weight) -> dict:
        return {"model__sample_weight": sample_weight} if sample_weight is not None else {}

    def _importances(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Permutation importance, normalised to sum to 1.

        Permutation works for every estimator we support, including
        HistGradientBoosting, which exposes no ``feature_importances_``.
        """
        try:
            result = permutation_importance(
                self._pipeline, X, y, n_repeats=5, random_state=42,
                scoring="neg_mean_absolute_error", n_jobs=1,
            )
        except Exception as exc:
            log.debug("Permutation importance failed: %s", exc)
            return {}

        values = np.clip(result.importances_mean, 0, None)
        total = values.sum()
        if total <= 0:
            return {}
        return {name: float(v / total) for name, v in zip(X.columns, values, strict=True)}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotTrainedError(
                f"{type(self).__name__} is not trained. Run `f1predict train` first."
            )
        return np.asarray(self._pipeline.predict(self._select(X)), dtype="float64")

    def rank(self, X: pd.DataFrame) -> np.ndarray:
        """1-based predicted positions."""
        scores = self.predict(X)
        return pd.Series(scores).rank(method="first").to_numpy(dtype=int)

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project the input onto the exact feature contract, in order."""
        required = self.feature_names
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise FeatureMismatchError(
                f"{type(self).__name__} needs column(s) {missing}, which the input "
                f"does not provide. Retrain after a schema change."
            )
        return X[required].astype("float64")


class DnfPredictor:
    """Estimates each driver's probability of not finishing.

    Kept separate from the ranking model on purpose: retirement is a different
    process from pace, and mixing the two teaches the ranking model that fast
    cars sometimes finish last, which is noise rather than signal.
    """

    #: Probabilities outside this range are almost certainly overfitting; a
    #: modern F1 car retires from perhaps 3-30% of races, never 0% or 90%.
    PROBABILITY_BOUNDS = (0.01, 0.45)

    def __init__(
        self,
        features: tuple[Feature, ...],
        kind: str = "gradient_boosting_classifier",
        params: dict | None = None,
        metadata: ModelMetadata | None = None,
    ):
        self.features = features
        self.kind = kind
        self.params = params or {}
        self.metadata = metadata or ModelMetadata(kind=kind)
        self._pipeline = build_pipeline(make_classifier(kind, self.params), scale=False)
        self._fitted = False
        self._base_rate = 0.09
        self.report = TrainingReport()

    @property
    def feature_names(self) -> list[str]:
        return names(self.features)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
        cv_folds: int = 5,
    ) -> TrainingReport:
        """Fit on a boolean retirement target and score it with grouped CV."""
        X_sel = X[[c for c in self.feature_names if c in X.columns]].astype("float64")
        y_arr = pd.Series(y).astype("boolean").fillna(False).to_numpy(dtype=bool)

        self._base_rate = float(y_arr.mean()) if len(y_arr) else 0.09
        report = TrainingReport(
            n_samples=len(y_arr), n_features=X_sel.shape[1],
            is_classifier=True, positive_rate=self._base_rate,
        )

        # A single-class target (nobody retired) makes the classifier useless
        # and its predict_proba shape invalid, so fall back to the base rate.
        if len(np.unique(y_arr)) < 2 or len(y_arr) < 30:
            report.notes.append(
                f"Retirement model skipped; using base rate {self._base_rate:.1%}."
            )
            self.report = report
            return report

        report.cv_auc = self._cross_val_auc(X_sel, y_arr, groups, cv_folds)
        self._pipeline.fit(X_sel, y_arr)
        self._fitted = True
        self.report = report
        log.info("DnfPredictor trained: %s", report.summary())
        return report

    def _cross_val_auc(self, X, y, groups, cv_folds: int) -> float:
        """Race-grouped ROC AUC.

        Retirement is rare and unevenly distributed, so accuracy would be
        flattered by always predicting "finished"; AUC measures whether the
        model actually ranks the risky entries above the safe ones.
        """
        if groups is None:
            return float("nan")

        groups_arr = pd.Series(groups).to_numpy()
        folds = min(cv_folds, len(np.unique(groups_arr)))
        if folds < 2:
            return float("nan")

        scores = []
        for train_idx, test_idx in GroupKFold(n_splits=folds).split(X, y, groups_arr):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            pipeline = build_pipeline(make_classifier(self.kind, self.params), scale=False)
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(roc_auc_score(y[test_idx], proba))

        return float(np.mean(scores)) if scores else float("nan")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retirement probability per row, clipped to a plausible range."""
        if not self._fitted:
            return np.full(len(X), self._base_rate)

        available = [c for c in self.feature_names if c in X.columns]
        if not available:
            return np.full(len(X), self._base_rate)

        try:
            proba = self._pipeline.predict_proba(X[available].astype("float64"))[:, 1]
        except Exception as exc:
            log.warning("Retirement model failed (%s); using base rate.", exc)
            return np.full(len(X), self._base_rate)

        return np.clip(proba, *self.PROBABILITY_BOUNDS)

    @property
    def base_rate(self) -> float:
        return self._base_rate
