"""Qualifying predictor: Free Practice features → predicted quali position."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from f1predict.models.base import BasePredictor

log = logging.getLogger(__name__)

FEATURES = [
    "fp_best_lap_s",
    "fp_gap_to_best_s",
    "fp_long_run_pace_s",
    "form_avg_pos",
    "form_avg_pts",
    "team_avg_pos",
]

TARGET = "quali_pos"


class QualiPredictor(BasePredictor):
    """
    Predicts qualifying position (1–20) from Free Practice features.
    Lower predicted value → better grid position.
    """

    def __init__(self, model_params: dict | None = None):
        params = model_params or {
            "n_estimators": 200, "learning_rate": 0.05,
            "max_depth": 4, "subsample": 0.8,
            "min_samples_leaf": 3, "random_state": 42,
        }
        self._gbr = GradientBoostingRegressor(**params)
        self._scaler = StandardScaler()
        self._fitted = False
        self._fitted_cols: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QualiPredictor":
        cols = [c for c in FEATURES if c in X.columns]
        X_sel = X[cols].fillna(X[cols].median()).fillna(0)
        X_scaled = self._scaler.fit_transform(X_sel)
        self._gbr.fit(X_scaled, y)
        self._fitted = True
        self._fitted_cols = cols
        y_pred = self._gbr.predict(X_scaled)
        log.info("QualiPredictor trained on %d samples, MAE=%.2f positions",
                 len(y), mean_absolute_error(y, y_pred))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("QualiPredictor not trained. Run `f1predict train` first.")
        X_sel = pd.DataFrame(index=X.index)
        for col in self._fitted_cols:
            X_sel[col] = X[col].fillna(0) if col in X.columns else 0.0
        X_scaled = self._scaler.transform(X_sel)
        return pd.Series(self._gbr.predict(X_scaled), index=X.index)
