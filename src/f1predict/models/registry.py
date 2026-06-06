"""Model registry: instantiate models by name from config."""

from __future__ import annotations

from f1predict.models.quali_model import QualiPredictor
from f1predict.models.race_model import RacePredictor


def get_race_model(cfg: dict) -> RacePredictor:
    model_name = cfg["models"].get("default", "gradient_boosting")
    params = cfg["models"].get(model_name, {})
    mc = cfg.get("monte_carlo", {})
    return RacePredictor(
        model_params=params or None,
        n_simulations=mc.get("n_simulations", 2000),
        noise_std=mc.get("position_noise_std", 3.0),
    )


def get_quali_model(cfg: dict) -> QualiPredictor:
    model_name = cfg["models"].get("default", "gradient_boosting")
    params = cfg["models"].get(model_name, {})
    return QualiPredictor(model_params=params or None)
