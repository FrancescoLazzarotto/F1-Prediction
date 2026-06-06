"""Abstract base class for all predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BasePredictor(ABC):
    """Minimal interface: fit on historical data, predict on new data."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        """Train the model. Returns self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return raw numeric predictions (lower = better for ranking)."""

    def predict_ranking(self, X: pd.DataFrame) -> pd.Series:
        """Return integer ranks (1 = best) derived from predict()."""
        raw = self.predict(X)
        return raw.rank(method="first").astype(int)

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of feature column names this model expects."""
