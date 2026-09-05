"""Shared fixtures: a temp-cache config and synthetic season data.

Every test runs against an isolated cache directory and fabricated results, so
the suite never touches the network or the developer's real cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1predict.config import (
    Config,
    FormConfig,
    ModelConfig,
    ModelsConfig,
    SimulationConfig,
    TrainingConfig,
    replace_cache_root,
)

DRIVERS = [
    ("verstappen", "VER", "Max Verstappen", "Red Bull", "red_bull"),
    ("perez", "PER", "Sergio Perez", "Red Bull", "red_bull"),
    ("norris", "NOR", "Lando Norris", "McLaren", "mclaren"),
    ("piastri", "PIA", "Oscar Piastri", "McLaren", "mclaren"),
    ("leclerc", "LEC", "Charles Leclerc", "Ferrari", "ferrari"),
    ("sainz", "SAI", "Carlos Sainz", "Ferrari", "ferrari"),
    ("hamilton", "HAM", "Lewis Hamilton", "Mercedes", "mercedes"),
    ("russell", "RUS", "George Russell", "Mercedes", "mercedes"),
    ("alonso", "ALO", "Fernando Alonso", "Aston Martin", "aston_martin"),
    ("stroll", "STR", "Lance Stroll", "Aston Martin", "aston_martin"),
]

CIRCUITS = ["bahrain", "jeddah", "albert_park", "suzuka", "shanghai", "miami"]


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A fast, fully isolated configuration."""
    base = Config(
        training=TrainingConfig(
            seasons=[2024], min_laps_long_run=3, cv_folds=3, min_samples_for_cv=40,
        ),
        models=ModelsConfig(
            race=ModelConfig("hist_gradient_boosting", {"max_iter": 30, "max_depth": 3}),
            quali=ModelConfig("hist_gradient_boosting", {"max_iter": 30, "max_depth": 3}),
            dnf=ModelConfig("gradient_boosting_classifier", {"n_estimators": 20}),
        ),
        simulation=SimulationConfig(n_simulations=500, seed=7),
        form=FormConfig(recent_races=3, reliability_races=5, trend_races=2),
    )
    return replace_cache_root(base, tmp_path / "cache")


@pytest.fixture
def season_results() -> pd.DataFrame:
    """Two synthetic seasons where car speed drives the result, plus noise."""
    return make_season_results([2023, 2024], n_rounds=len(CIRCUITS))


def make_season_results(years: list[int], n_rounds: int = 6) -> pd.DataFrame:
    """Deterministic results in which faster teams generally finish higher."""
    rng = np.random.default_rng(11)
    rows = []
    for year in years:
        for round_num in range(1, n_rounds + 1):
            circuit = CIRCUITS[(round_num - 1) % len(CIRCUITS)]
            # Base strength follows the driver order, jittered per race.
            strength = np.arange(len(DRIVERS), dtype=float) + rng.normal(0, 1.2, len(DRIVERS))
            order = np.argsort(strength)

            for finish, idx in enumerate(order, start=1):
                driver_id, code, name, team, constructor = DRIVERS[idx]
                retired = bool(rng.random() < 0.08)
                rows.append({
                    "year": year,
                    "round": round_num,
                    "circuit_id": circuit,
                    "race_date": f"{year}-{(round_num % 12) + 1:02d}-15T14:00:00Z",
                    "driver_id": driver_id,
                    "driver_code": code,
                    "driver_name": name,
                    "team": team,
                    "constructor_id": constructor,
                    "grid_pos": int(np.clip(finish + rng.integers(-2, 3), 1, len(DRIVERS))),
                    "race_pos": finish,
                    "points": float([25, 18, 15, 12, 10, 8, 6, 4, 2, 1][finish - 1])
                    if finish <= 10 and not retired else 0.0,
                    "status": "Engine" if retired else "Finished",
                    "laps": 50 if not retired else 20,
                    "dnf": retired,
                    "classified": not retired,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def quali_history(season_results) -> pd.DataFrame:
    """Qualifying results aligned with the synthetic race results."""
    return make_quali(season_results)


def make_quali(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, round_num), group in results.groupby(["year", "round"]):
        group = group.sort_values("grid_pos")
        pole_time = 78.0
        for position, (_, row) in enumerate(group.iterrows(), start=1):
            best = pole_time + (position - 1) * 0.18
            rows.append({
                "year": year, "round": round_num,
                "driver_id": row["driver_id"], "driver_code": row["driver_code"],
                "driver_name": row["driver_name"], "team": row["team"],
                "constructor_id": row["constructor_id"],
                "quali_pos": position,
                "q1_s": best + 0.6, "q2_s": best + 0.3 if position <= 15 else None,
                "q3_s": best if position <= 10 else None,
                "best_quali_s": best,
                "quali_gap_to_pole_s": best - pole_time,
                "quali_gap_to_pole_pct": (best / pole_time - 1) * 100,
                "reached_q2": int(position <= 15),
                "reached_q3": int(position <= 10),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def entry_list() -> pd.DataFrame:
    return pd.DataFrame([
        {"driver_code": code, "driver_id": did, "driver_name": name,
         "team": team, "constructor_id": constructor, "driver_number": str(i + 1)}
        for i, (did, code, name, team, constructor) in enumerate(DRIVERS)
    ])


@pytest.fixture
def practice_laps() -> pd.DataFrame:
    """A FastF1-shaped laps frame with clean stints and deliberate junk laps."""
    rows = []
    base = 80.0
    for idx, (_, code, *_rest) in enumerate(DRIVERS):
        pace = base + idx * 0.25
        for stint in (1, 2):
            for lap in range(1, 9):
                out_lap = lap == 1
                in_lap = lap == 8
                # One cool-down lap per stint, which must not pollute race pace.
                cooldown = lap == 5
                lap_time = pace + (12.0 if cooldown else 0.0) + (lap * 0.02)
                rows.append({
                    "Driver": code,
                    "LapNumber": float(lap + (stint - 1) * 8),
                    "LapTime": pd.Timedelta(seconds=lap_time),
                    "Sector1Time": pd.Timedelta(seconds=lap_time * 0.3),
                    "Sector2Time": pd.Timedelta(seconds=lap_time * 0.4),
                    "Sector3Time": pd.Timedelta(seconds=lap_time * 0.3),
                    "Stint": float(stint),
                    "Compound": "SOFT" if stint == 1 else "MEDIUM",
                    "TyreLife": float(lap),
                    "IsAccurate": True,
                    "PitOutTime": pd.Timedelta(seconds=1) if out_lap else pd.NaT,
                    "PitInTime": pd.Timedelta(seconds=1) if in_lap else pd.NaT,
                    "TrackStatus": "1",
                    "Deleted": False,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def weather() -> dict:
    return {
        "temperature": 24.0, "humidity": 50.0, "wind_speed": 2.5,
        "cloud_cover": 20.0, "rain_prob": 0.05, "source": "archive",
    }


@pytest.fixture
def fast_sim_cfg() -> SimulationConfig:
    return SimulationConfig(n_simulations=800, seed=3)
