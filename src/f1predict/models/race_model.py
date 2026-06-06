"""Race predictor: quali + form + weather → race position + Monte Carlo probabilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from f1predict.models.base import BasePredictor

log = logging.getLogger(__name__)

FEATURES = [
    "grid_pos",
    "quali_gap_to_pole_s",
    "fp_long_run_pace_s",
    "fp_gap_to_best_s",
    "form_avg_pos",
    "form_avg_pts",
    "form_dnf_rate",
    "circuit_avg_pos",
    "circuit_n_starts",
    "team_avg_pos",
    "weather_temp",
    "weather_wind",
    "weather_rain_prob",
    "reached_q3",
]

TARGET = "race_pos"


class RacePredictor(BasePredictor):
    """
    Predicts race finishing position (1–20, lower = better) from quali + form + weather features.
    Uses Monte Carlo simulation to estimate podium / win probabilities.
    """

    def __init__(self, model_params: dict | None = None, n_simulations: int = 2000,
                 noise_std: float = 3.0):
        params = model_params or {
            "n_estimators": 200, "learning_rate": 0.05,
            "max_depth": 4, "subsample": 0.8,
            "min_samples_leaf": 3, "random_state": 42,
        }
        self._gbr = GradientBoostingRegressor(**params)
        self._scaler = StandardScaler()
        self._n_sim = n_simulations
        self._noise_std = noise_std
        self._fitted = False
        self._fitted_cols: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RacePredictor":
        cols = [c for c in FEATURES if c in X.columns]
        X_sel = X[cols].fillna(X[cols].median()).fillna(0)
        X_scaled = self._scaler.fit_transform(X_sel)
        self._gbr.fit(X_scaled, y)
        self._fitted = True
        self._fitted_cols = cols
        y_pred = self._gbr.predict(X_scaled)
        log.info("RacePredictor trained on %d samples, MAE=%.2f positions",
                 len(y), mean_absolute_error(y, y_pred))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("RacePredictor not trained. Run `f1predict train` first.")
        # Use exactly the columns the scaler was fitted on; fill missing ones with 0
        X_sel = pd.DataFrame(index=X.index)
        for col in self._fitted_cols:
            X_sel[col] = X[col].fillna(0) if col in X.columns else 0.0
        X_scaled = self._scaler.transform(X_sel)
        return pd.Series(self._gbr.predict(X_scaled), index=X.index)

    def predict_with_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with predicted ranking + Monte Carlo probabilities.
        Columns: predicted_pos, p_win, p_podium, p_top5, p_top10.
        """
        raw = self.predict(X).values
        n = len(raw)
        rng = np.random.default_rng(seed=42)

        # Monte Carlo: add Gaussian noise and rank n_sim times
        wins = np.zeros(n)
        podiums = np.zeros(n)
        top5 = np.zeros(n)
        top10 = np.zeros(n)

        for _ in range(self._n_sim):
            noisy = raw + rng.normal(0, self._noise_std, size=n)
            ranks = noisy.argsort().argsort() + 1  # 1-indexed rank
            wins += (ranks == 1).astype(float)
            podiums += (ranks <= 3).astype(float)
            top5 += (ranks <= 5).astype(float)
            top10 += (ranks <= 10).astype(float)

        result = pd.DataFrame({
            "predicted_pos": pd.Series(raw, index=X.index).rank(method="first").astype(int),
            "p_win": wins / self._n_sim,
            "p_podium": podiums / self._n_sim,
            "p_top5": top5 / self._n_sim,
            "p_top10": top10 / self._n_sim,
        }, index=X.index)
        return result
