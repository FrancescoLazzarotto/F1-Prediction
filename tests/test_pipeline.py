"""Smoke tests for the prediction pipeline using mocked data sources."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def cfg(tmp_path):
    """Minimal config pointing cache dirs to a temp directory."""
    return {
        "cache": {
            "fastf1_dir": str(tmp_path / "fastf1"),
            "features_dir": str(tmp_path / "features"),
            "models_dir": str(tmp_path / "models"),
        },
        "training": {
            "seasons": [2024],
            "min_laps_long_run": 3,
        },
        "models": {
            "default": "gradient_boosting",
            "gradient_boosting": {
                "n_estimators": 10, "learning_rate": 0.1,
                "max_depth": 2, "subsample": 0.8,
                "min_samples_leaf": 1, "random_state": 42,
            },
        },
        "monte_carlo": {"n_simulations": 50, "position_noise_std": 2.0},
        "form": {"recent_races": 3},
        "features": {
            "use_weather": True, "use_tyre": True,
            "use_long_run": True, "use_circuit_history": True,
        },
    }


def _fake_season_results(year: int) -> pd.DataFrame:
    """Minimal season results for 2 rounds, 5 drivers."""
    drivers = ["verstappen", "norris", "leclerc", "hamilton", "russell"]
    codes = ["VER", "NOR", "LEC", "HAM", "RUS"]
    teams = ["red_bull", "mclaren", "ferrari", "mercedes", "mercedes"]
    constructors = ["red_bull", "mclaren", "ferrari", "mercedes", "mercedes"]
    rows = []
    for r in range(1, 3):
        for i, did in enumerate(drivers):
            rows.append({
                "year": year, "round": r,
                "circuit_id": "bahrain" if r == 1 else "jeddah",
                "driver_id": did, "driver_code": codes[i],
                "driver_name": did.title(),
                "team": teams[i], "constructor_id": constructors[i],
                "grid_pos": i + 1, "race_pos": i + 1,
                "points": [25, 18, 15, 12, 10][i],
                "status": "Finished", "laps": 57,
            })
    return pd.DataFrame(rows)


def _fake_circuit_info():
    return {"name": "Bahrain GP", "circuit_id": "bahrain",
            "lat": 26.03, "lon": 50.51, "locality": "Sakhir",
            "country": "Bahrain", "race_date": "2024-03-02T15:00:00"}


def _fake_entry_list():
    return pd.DataFrame({
        "driver_code": ["VER", "NOR", "LEC", "HAM", "RUS"],
        "driver_id": ["verstappen", "norris", "leclerc", "hamilton", "russell"],
        "driver_name": ["Max Verstappen", "Lando Norris", "Charles Leclerc",
                        "Lewis Hamilton", "George Russell"],
        "team": ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Mercedes"],
        "driver_number": ["1", "4", "16", "44", "63"],
    })


def _fake_quali_results():
    return pd.DataFrame({
        "driver_id": ["verstappen", "norris", "leclerc", "hamilton", "russell"],
        "driver_code": ["VER", "NOR", "LEC", "HAM", "RUS"],
        "driver_name": ["Max Verstappen", "Lando Norris", "Charles Leclerc",
                        "Lewis Hamilton", "George Russell"],
        "team": ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Mercedes"],
        "quali_pos": [1, 2, 3, 4, 5],
        "q1_s": [74.5, 74.8, 75.0, 75.2, 75.3],
        "q2_s": [74.2, 74.5, 74.7, 74.9, 75.0],
        "q3_s": [74.0, 74.1, 74.3, 74.5, 74.6],
        "best_quali_s": [74.0, 74.1, 74.3, 74.5, 74.6],
        "quali_gap_to_pole_s": [0.0, 0.1, 0.3, 0.5, 0.6],
    })


def test_pipeline_train_and_predict(cfg):
    """Smoke test: train on fake data and run a prediction."""
    import f1predict.cache as C
    import f1predict.data.jolpica_source as JLP
    import f1predict.data.fastf1_source as F1
    import f1predict.data.weather_source as WX
    from f1predict.pipeline import F1Pipeline
    from f1predict.data import schedule as S

    season_data = _fake_season_results(2024)

    with (
        patch.object(JLP, "get_season_results", return_value=season_data),
        patch.object(JLP, "get_circuit_info", return_value=_fake_circuit_info()),
        patch.object(JLP, "get_race_results", return_value=season_data[season_data["round"] == 1]),
        patch.object(JLP, "get_quali_results", return_value=_fake_quali_results()),
        patch.object(F1, "get_fp_laps", return_value=None),
        patch.object(F1, "get_entry_list", return_value=_fake_entry_list()),
        patch.object(WX, "get_weather_for_race",
                     return_value={"temperature": 28.0, "humidity": 40.0,
                                   "wind_speed": 3.0, "cloud_cover": 10.0, "rain_prob": 0.0}),
        patch.object(S, "available_sessions", return_value={"FP1": True, "FP2": True,
                                                              "FP3": True, "Q": True, "R": False}),
    ):
        pipe = F1Pipeline(cfg)
        pipe.train([2024], force_retrain=True)

        result = pipe.predict_race(2024, 1)

    assert "race" in result
    race_df = result["race"]
    assert len(race_df) == 5
    assert "predicted_pos" in race_df.columns
    assert "p_win" in race_df.columns
    assert "p_podium" in race_df.columns
    # Probabilities should be in [0, 1]
    assert (race_df["p_win"] >= 0).all() and (race_df["p_win"] <= 1).all()
    # Win probabilities should sum to ~1
    assert race_df["p_win"].sum() == pytest.approx(1.0, abs=0.05)


def test_pipeline_backtest(cfg):
    """Smoke test for backtest mode."""
    import f1predict.data.jolpica_source as JLP
    import f1predict.data.fastf1_source as F1
    import f1predict.data.weather_source as WX
    from f1predict.pipeline import F1Pipeline
    from f1predict.data import schedule as S

    season_data = _fake_season_results(2024)
    actual = season_data[season_data["round"] == 2][["driver_code", "race_pos"]].copy()
    actual.rename(columns={"race_pos": "race_pos"}, inplace=True)

    with (
        patch.object(JLP, "get_season_results", return_value=season_data),
        patch.object(JLP, "get_circuit_info", return_value=_fake_circuit_info()),
        patch.object(JLP, "get_race_results", return_value=season_data[season_data["round"] == 2]),
        patch.object(JLP, "get_quali_results", return_value=_fake_quali_results()),
        patch.object(F1, "get_fp_laps", return_value=None),
        patch.object(F1, "get_entry_list", return_value=_fake_entry_list()),
        patch.object(WX, "get_weather_for_race",
                     return_value={"temperature": 28.0, "humidity": 40.0,
                                   "wind_speed": 3.0, "cloud_cover": 10.0, "rain_prob": 0.0}),
        patch.object(S, "available_sessions", return_value={"FP1": True, "Q": True, "R": True}),
    ):
        pipe = F1Pipeline(cfg)
        pipe.train([2024], force_retrain=True)
        metrics = pipe.backtest(2024, 2)

    assert "spearman_rho" in metrics
    assert "top3_accuracy" in metrics
    assert "mae_positions" in metrics
    assert -1.0 <= metrics["spearman_rho"] <= 1.0
    assert 0.0 <= metrics["top3_accuracy"] <= 1.0
