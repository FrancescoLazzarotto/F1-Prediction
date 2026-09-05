"""End-to-end pipeline behaviour, with every network source mocked out."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from f1predict.data.schedule import Event
from f1predict.features.builder import GRID_ACTUAL, GRID_PREDICTED
from f1predict.pipeline import F1Pipeline
from tests.conftest import CIRCUITS, DRIVERS


def _circuit_info(year: int, round_num: int, cfg=None) -> dict:
    circuit = CIRCUITS[(round_num - 1) % len(CIRCUITS)]
    return {
        "name": f"{circuit.title()} Circuit", "circuit_id": circuit,
        "lat": 26.03, "lon": 50.51, "locality": "Somewhere", "country": "Nowhere",
        "race_name": f"{circuit.title()} Grand Prix",
        "race_date": f"{year}-0{(round_num % 9) + 1}-15T14:00:00Z",
        "round": round_num, "year": year,
    }


def _events(year: int) -> list[Event]:
    base = datetime(year, 3, 1, 14, 0, tzinfo=timezone.utc)
    return [
        Event(year=year, round=r, name=f"{CIRCUITS[(r - 1) % len(CIRCUITS)].title()} GP",
              location="Somewhere", race_date=base + timedelta(days=14 * (r - 1)))
        for r in range(1, len(CIRCUITS) + 1)
    ]


@pytest.fixture
def mocked(cfg, season_results, quali_history, entry_list, practice_laps, weather):
    """Patch every external source and yield the pipeline plus its fixtures."""
    from f1predict import pipeline as pipeline_module
    from f1predict.data import fastf1_source as F1
    from f1predict.data import repository as repo
    from f1predict.data import weather as WX
    from f1predict.features import builder

    class _Practice:
        session_name = "FP2"
        laps = practice_laps

    def _season_results(year, cfg=None, refresh=False):
        return season_results[season_results["year"] == year]

    def _season_quali(year, cfg=None, refresh=False):
        return quali_history[quali_history["year"] == year]

    def _history_for(year, seasons_back=2, cfg=None):
        return season_results[season_results["year"] <= year]

    def _quali_history_for(year, seasons_back=2, cfg=None):
        return quali_history[quali_history["year"] <= year]

    def _race_results(year, round_num, cfg=None):
        return season_results[
            (season_results["year"] == year) & (season_results["round"] == round_num)
        ].reset_index(drop=True)

    def _quali_results(year, round_num, cfg=None):
        return quali_history[
            (quali_history["year"] == year) & (quali_history["round"] == round_num)
        ].reset_index(drop=True)

    with ExitStack() as stack:
        for target, attr, value in [
            (repo, "season_results", _season_results),
            (repo, "season_quali", _season_quali),
            (repo, "history_for", _history_for),
            (repo, "quali_history_for", _quali_history_for),
            (repo, "race_results", _race_results),
            (repo, "quali_results", _quali_results),
            (repo, "circuit_info", _circuit_info),
            (builder, "repo", repo),
            (F1, "get_practice_laps", lambda *a, **k: _Practice()),
            (F1, "get_entry_list", lambda *a, **k: entry_list),
            (WX, "get_weather_for_race", lambda *a, **k: dict(weather)),
            (pipeline_module, "season_events", _events),
            (pipeline_module, "resolve_event", lambda **k: _events(k.get("year", 2024))[0]),
        ]:
            stack.enter_context(patch.object(target, attr, value))

        # available_sessions is imported by name into the pipeline module.
        stack.enter_context(patch.object(
            pipeline_module, "available_sessions",
            lambda y, r, now=None: {"FP1": True, "FP2": True, "FP3": True,
                                    "Q": True, "R": True},
        ))
        yield F1Pipeline(cfg)


@pytest.fixture
def trained(mocked):
    mocked.train([2024])
    return mocked


class TestTraining:
    def test_produces_all_three_models(self, mocked):
        reports = mocked.train([2024])
        assert set(reports) == {"race", "quali", "dnf"}
        assert reports["race"].n_samples > 0
        assert mocked.models_ready

    def test_race_model_excludes_retirements(self, mocked, season_results):
        """Retirements are handled by the DNF model, not the ranking model."""
        reports = mocked.train([2024])
        season = season_results[season_results["year"] == 2024]
        assert reports["race"].n_samples <= int((~season["dnf"]).sum())

    def test_models_persist_and_reload(self, mocked, cfg):
        mocked.train([2024])
        fresh = F1Pipeline(cfg)
        assert fresh.models_ready

    def test_progress_callback_receives_messages(self, mocked):
        messages: list[str] = []
        mocked.train([2024], progress=messages.append)
        assert messages
        assert any("Training" in m for m in messages)

    def test_recency_weighting_prefers_recent_races(self, season_results):
        from f1predict.pipeline import _recency_weights

        frame = season_results.rename(columns={}).copy()
        weights = _recency_weights(frame, half_life_races=3.0)
        newest = frame["year"] == frame["year"].max()
        oldest = frame["year"] == frame["year"].min()
        assert weights[newest].mean() > weights[oldest].mean()


class TestRacePrediction:
    def test_returns_a_row_per_entrant(self, trained, entry_list):
        prediction = trained.predict_race(2024, 3)
        assert len(prediction.table) == len(entry_list)
        assert prediction.table["predicted_pos"].tolist() == list(
            range(1, len(entry_list) + 1)
        )

    def test_includes_probability_columns(self, trained):
        table = trained.predict_race(2024, 3).table
        for column in ("p_win", "p_podium", "p_points", "p_dnf", "expected_points"):
            assert column in table.columns
        assert table["p_win"].sum() == pytest.approx(1.0, abs=1e-6)
        assert ((table["p_podium"] >= 0) & (table["p_podium"] <= 1)).all()

    def test_uses_the_real_grid_when_qualifying_has_run(self, trained):
        prediction = trained.predict_race(2024, 3)
        assert prediction.grid_source == GRID_ACTUAL
        assert prediction.confidence == "high"

    def test_predicts_the_grid_when_qualifying_has_not_run(self, trained):
        """The two-stage path: practice predicts qualifying, which sets the grid.

        This is the case the whole design exists for, so the predicted order
        must actually reach the race model rather than being discarded.
        """
        from f1predict import pipeline as pipeline_module

        with patch.object(
            pipeline_module, "available_sessions",
            lambda y, r, now=None: {"FP1": True, "FP2": True, "Q": False, "R": False},
        ):
            prediction = trained.predict_race(2024, 3)

        assert prediction.grid_source == GRID_PREDICTED
        assert prediction.confidence == "medium"
        assert not prediction.quali_table.empty

        # The grid the race model saw must match the predicted qualifying order.
        predicted_grid = dict(zip(
            prediction.quali_table["driver_code"],
            prediction.quali_table["predicted_quali_pos"],
            strict=True,
        ))
        used_grid = dict(zip(prediction.table["driver_code"], prediction.table["grid_pos"], strict=True))
        assert used_grid == pytest.approx(predicted_grid)

    def test_grid_delta_is_consistent(self, trained):
        table = trained.predict_race(2024, 3).table
        np.testing.assert_array_equal(
            table["grid_delta"].to_numpy(),
            (table["grid_pos"] - table["predicted_pos"]).to_numpy(),
        )

    def test_event_and_weather_context_is_populated(self, trained, weather):
        prediction = trained.predict_race(2024, 3)
        assert prediction.event["circuit_id"] == CIRCUITS[2]
        assert prediction.weather["temperature"] == weather["temperature"]
        assert prediction.practice_session == "FP2"

    def test_podium_is_the_top_three(self, trained):
        prediction = trained.predict_race(2024, 3)
        assert prediction.podium == prediction.table.head(3)["driver_name"].tolist()

    def test_simulation_is_attached(self, trained, entry_list):
        prediction = trained.predict_race(2024, 3)
        assert prediction.simulation is not None
        assert prediction.simulation.positions.shape[1] == len(entry_list)

    def test_empty_entry_list_raises_a_clear_error(self, trained):
        from f1predict.data import fastf1_source as F1

        with patch.object(F1, "get_entry_list", lambda *a, **k: pd.DataFrame()):
            with pytest.raises(RuntimeError, match="No data available"):
                trained.predict_race(2024, 3)


class TestQualiPrediction:
    def test_returns_a_full_predicted_order(self, trained, entry_list):
        result = trained.predict_quali(2024, 3)
        assert len(result) == len(entry_list)
        assert result["predicted_quali_pos"].tolist() == list(range(1, len(entry_list) + 1))

    def test_gap_column_starts_at_zero_for_pole(self, trained):
        result = trained.predict_quali(2024, 3)
        assert result["approx_gap_s"].iloc[0] == pytest.approx(0.0)
        assert result["approx_gap_s"].is_monotonic_increasing


class TestExplanation:
    def test_returns_ranked_contributions(self, trained):
        prediction = trained.predict_race(2024, 3)
        driver = prediction.table["driver_code"].iloc[0]
        contributions = trained.explain(prediction, driver, top_n=4)

        assert 0 < len(contributions) <= 4
        impacts = [c["impact"] for c in contributions]
        assert impacts == sorted(impacts, reverse=True)
        for item in contributions:
            assert item["direction"] in {"strength", "weakness"}
            assert item["label"]

    def test_unknown_driver_raises(self, trained):
        prediction = trained.predict_race(2024, 3)
        with pytest.raises(KeyError):
            trained.explain(prediction, "ZZZ")


class TestBacktest:
    def test_scores_against_the_real_result(self, trained):
        result = trained.backtest(2024, 4)
        assert result.metrics.n > 0
        assert -1.0 <= result.metrics.spearman <= 1.0
        assert result.metrics.mae >= 0
        assert {"actual_pos", "predicted_pos", "error"} <= set(result.table.columns)

    def test_flags_an_in_sample_race(self, trained):
        """Scoring a race the model trained on must be labelled as optimistic."""
        assert trained.backtest(2024, 4).in_sample is True

    def test_reports_brier_scores(self, trained):
        result = trained.backtest(2024, 4)
        assert 0.0 <= result.brier_win <= 1.0
        assert 0.0 <= result.brier_podium <= 1.0

    def test_a_race_without_results_raises(self, trained):
        from f1predict.data import repository as repo

        with patch.object(repo, "race_results", lambda *a, **k: pd.DataFrame()):
            with pytest.raises(RuntimeError, match="No results"):
                trained.backtest(2024, 4)

    def test_season_backtest_aggregates_every_round(self, trained):
        overall, per_race = trained.backtest_season(2024)
        assert len(per_race) == len(CIRCUITS)
        assert overall.n > 0
        assert set(per_race.columns) >= {"round", "spearman", "mae", "podium_hit"}


class TestChampionship:
    def test_projects_title_probabilities(self, trained, season_results):
        from f1predict.data import repository as repo

        standings = (
            season_results[season_results["year"] == 2024]
            .groupby(["driver_id", "driver_name", "team", "constructor_id"], as_index=False)
            .agg(points=("points", "sum"))
            .sort_values("points", ascending=False)
            .reset_index(drop=True)
        )
        standings.insert(0, "position", range(1, len(standings) + 1))
        standings["wins"] = 0

        with patch.object(repo, "driver_standings", lambda *a, **k: standings):
            outlook = trained.championship_outlook(2024)

        assert outlook.table["p_title"].sum() == pytest.approx(1.0, abs=1e-9)
        assert len(outlook.table) == len(DRIVERS)

    def test_no_standings_raises(self, trained):
        from f1predict.data import repository as repo

        with patch.object(repo, "driver_standings", lambda *a, **k: pd.DataFrame()):
            with pytest.raises(RuntimeError, match="No standings"):
                trained.championship_outlook(2024)


class TestDeterminism:
    def test_the_same_inputs_give_the_same_prediction(self, trained):
        first = trained.predict_race(2024, 3).table
        second = trained.predict_race(2024, 3).table
        pd.testing.assert_frame_equal(first, second)
