"""Tests for feature extraction modules."""

import pandas as pd
import numpy as np
import pytest
from datetime import timedelta


def _make_laps(n_drivers=5, laps_per_driver=20) -> pd.DataFrame:
    """Create a minimal fake FastF1 Laps DataFrame."""
    rows = []
    base_time = timedelta(seconds=90)
    for d_idx in range(n_drivers):
        driver = f"D{d_idx:02d}"
        for lap in range(1, laps_per_driver + 1):
            jitter = timedelta(seconds=np.random.uniform(-0.5, 1.0))
            rows.append({
                "Driver": driver,
                "LapTime": base_time + timedelta(seconds=d_idx * 0.3) + jitter,
                "Sector1Time": timedelta(seconds=30 + d_idx * 0.1),
                "Sector2Time": timedelta(seconds=30 + d_idx * 0.1),
                "Sector3Time": timedelta(seconds=30 + d_idx * 0.1),
                "Stint": 1 + lap // 8,
                "IsAccurate": True,
                "PitOutLap": lap == 1,
                "PitInLap": False,
                "TrackStatus": "1",
            })
    return pd.DataFrame(rows)


def test_extract_fp_features_basic():
    from f1predict.features.practice import extract_fp_features

    laps = _make_laps(n_drivers=5, laps_per_driver=20)
    result = extract_fp_features(laps, min_laps=3)

    assert not result.empty
    assert "driver_code" in result.columns
    assert "fp_best_lap_s" in result.columns
    assert "fp_gap_to_best_s" in result.columns
    assert "fp_long_run_pace_s" in result.columns
    # Best gap should be 0 for the fastest driver
    assert result["fp_gap_to_best_s"].min() == pytest.approx(0.0, abs=1e-6)
    # All gap values >= 0
    assert (result["fp_gap_to_best_s"] >= 0).all()


def test_extract_fp_features_empty():
    from f1predict.features.practice import extract_fp_features

    result = extract_fp_features(pd.DataFrame(), min_laps=5)
    assert result.empty


def test_extract_quali_features():
    from f1predict.features.qualifying import extract_quali_features

    quali_data = pd.DataFrame({
        "driver_id": ["verstappen", "norris", "leclerc"],
        "driver_code": ["VER", "NOR", "LEC"],
        "driver_name": ["Max Verstappen", "Lando Norris", "Charles Leclerc"],
        "team": ["Red Bull", "McLaren", "Ferrari"],
        "quali_pos": [1, 2, 3],
        "q1_s": [74.5, 74.8, 75.0],
        "q2_s": [74.2, 74.5, 74.7],
        "q3_s": [74.0, 74.1, 74.3],
        "best_quali_s": [74.0, 74.1, 74.3],
        "quali_gap_to_pole_s": [0.0, 0.1, 0.3],
    })

    result = extract_quali_features(quali_data)
    assert not result.empty
    assert "grid_pos" in result.columns
    assert "quali_gap_to_pole_s" in result.columns
    assert result.loc[result["driver_code"] == "VER", "grid_pos"].iloc[0] == 1


def test_form_features_defaults():
    from f1predict.features.form import compute_form_features

    empty_df = pd.DataFrame(columns=["year", "round", "circuit_id", "driver_id",
                                     "race_pos", "points", "status", "constructor_id"])
    result = compute_form_features(empty_df, "verstappen", 2024, 5, "bahrain")

    assert "form_avg_pos" in result
    assert result["circuit_n_starts"] == 0


def test_form_features_with_data():
    from f1predict.features.form import compute_form_features

    results = pd.DataFrame({
        "year": [2024] * 4,
        "round": [1, 2, 3, 4],
        "circuit_id": ["bahrain", "jeddah", "australia", "bahrain"],
        "driver_id": ["verstappen"] * 4,
        "race_pos": [1, 2, 1, 1],
        "points": [25, 18, 25, 25],
        "status": ["Finished"] * 4,
        "constructor_id": ["red_bull"] * 4,
    })

    # Ask for form before round 4 (so rounds 1,2,3 should be included; round 4 excluded)
    form = compute_form_features(results, "verstappen", 2024, 4, "bahrain")
    assert form["form_avg_pos"] == pytest.approx((1 + 2 + 1) / 3, rel=0.01)
    # Circuit history at bahrain: round 1 only (round 4 is excluded as it IS the target)
    assert form["circuit_n_starts"] == 1
