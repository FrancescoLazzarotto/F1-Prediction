"""Estimator factory and the metadata bundle every trained model carries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Every estimator the config may name, with defaults tuned for ~2000 rows of
#: driver-race data — small, wide and noisy.
_REGRESSORS: dict[str, tuple[type, dict[str, Any]]] = {
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor,
        {"max_iter": 400, "learning_rate": 0.06, "max_depth": 5,
         "min_samples_leaf": 12, "l2_regularization": 0.5, "random_state": 42},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor,
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
         "subsample": 0.85, "min_samples_leaf": 8, "random_state": 42},
    ),
    "random_forest": (
        RandomForestRegressor,
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4,
         "n_jobs": -1, "random_state": 42},
    ),
    "ridge": (Ridge, {"alpha": 1.0, "random_state": None}),
}

_CLASSIFIERS: dict[str, tuple[type, dict[str, Any]]] = {
    "gradient_boosting_classifier": (
        GradientBoostingClassifier,
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3,
         "subsample": 0.9, "random_state": 42},
    ),
}


class UnknownEstimatorError(ValueError):
    """Raised when the config names an estimator that does not exist."""


def make_regressor(kind: str, params: dict[str, Any] | None = None):
    if kind not in _REGRESSORS:
        raise UnknownEstimatorError(
            f"Unknown regressor {kind!r}. Available: {sorted(_REGRESSORS)}"
        )
    cls, defaults = _REGRESSORS[kind]
    merged = {**defaults, **(params or {})}
    # Ridge has no random_state; drop keys the estimator will not accept.
    valid = cls().get_params()
    return cls(**{k: v for k, v in merged.items() if k in valid})


def make_classifier(kind: str, params: dict[str, Any] | None = None):
    if kind not in _CLASSIFIERS:
        raise UnknownEstimatorError(
            f"Unknown classifier {kind!r}. Available: {sorted(_CLASSIFIERS)}"
        )
    cls, defaults = _CLASSIFIERS[kind]
    merged = {**defaults, **(params or {})}
    valid = cls().get_params()
    return cls(**{k: v for k, v in merged.items() if k in valid})


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    """Wrap an estimator with imputation and, optionally, scaling.

    Imputation lives inside the pipeline so the fill values are learned on the
    training fold only — computing them over the whole frame would leak the
    validation set's distribution into training.
    """
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


#: Tree ensembles are scale-invariant, so scaling only costs time.
SCALE_SENSITIVE_KINDS = frozenset({"ridge"})


@dataclass(slots=True)
class TrainingReport:
    """What happened during a fit, kept alongside the model for auditing."""

    n_samples: int = 0
    n_features: int = 0
    train_mae: float = float("nan")
    cv_mae: float = float("nan")
    cv_spearman: float = float("nan")
    cv_top3: float = float("nan")
    cv_folds: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Set for classifiers, where MAE and rank correlation are meaningless.
    is_classifier: bool = False
    cv_auc: float = float("nan")
    positive_rate: float = float("nan")

    def summary(self) -> str:
        if self.is_classifier:
            auc = f"AUC {self.cv_auc:.3f}" if self.cv_auc == self.cv_auc else "AUC n/a"
            return (
                f"{self.n_samples} samples, {auc}, "
                f"base rate {self.positive_rate:.1%}"
            )
        if self.cv_folds:
            return (
                f"{self.n_samples} samples, CV MAE {self.cv_mae:.2f} pos, "
                f"Spearman {self.cv_spearman:.3f}, top-3 {self.cv_top3:.0%}"
            )
        return f"{self.n_samples} samples, train MAE {self.train_mae:.2f} pos (no CV)"

    def top_features(self, n: int = 8) -> list[tuple[str, float]]:
        return sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:n]


@dataclass(slots=True)
class ModelMetadata:
    """Provenance stamped onto every saved model."""

    signature: str = ""
    schema_version: int = 0
    kind: str = ""
    seasons: list[int] = field(default_factory=list)
    trained_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    )
    package_version: str = ""

    def is_compatible_with(self, signature: str) -> bool:
        """Whether this artefact was built from the caller's feature contract."""
        return self.signature == signature
