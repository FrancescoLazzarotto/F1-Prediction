"""Model training, the feature contract, metrics, and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predict import cache as C
from f1predict.features import schema
from f1predict.models import registry
from f1predict.models.base import UnknownEstimatorError, make_regressor
from f1predict.models.metrics import (
    aggregate,
    brier_score,
    evaluate_ranking,
)
from f1predict.models.predictor import (
    DnfPredictor,
    FeatureMismatchError,
    NotTrainedError,
    RankingPredictor,
)


def _training_frame(n_races: int = 25, n_drivers: int = 10) -> pd.DataFrame:
    """Synthetic driver-races where grid position mostly determines the result."""
    rng = np.random.default_rng(3)
    rows = []
    for race in range(n_races):
        for driver in range(n_drivers):
            grid = driver + 1
            noise = rng.normal(0, 1.5)
            row = {name: rng.normal(0, 1) for name in schema.ALL_FEATURE_COLS}
            row["grid_pos"] = float(grid)
            row["grid_pos_pct"] = grid / n_drivers
            row["form_avg_pos"] = grid + rng.normal(0, 0.5)
            row["race_id"] = f"r{race:02d}"
            row["race_pos"] = float(np.clip(grid + noise, 1, n_drivers))
            row["quali_pos_actual"] = float(grid)
            row["dnf"] = bool(rng.random() < 0.12)
            rows.append(row)
    return pd.DataFrame(rows)


class TestRankingPredictor:
    @pytest.fixture
    def trained(self):
        df = _training_frame()
        model = RankingPredictor(schema.RACE_FEATURES, kind="hist_gradient_boosting",
                                 params={"max_iter": 40, "max_depth": 3})
        report = model.fit(
            df[schema.RACE_FEATURE_COLS], df["race_pos"],
            groups=df["race_id"], cv_folds=3, min_samples_for_cv=40,
        )
        return model, report, df

    def test_learns_the_underlying_signal(self, trained):
        _model, report, _df = trained
        assert report.cv_folds == 3
        assert report.cv_spearman > 0.7
        assert report.cv_mae < 3.0

    def test_reports_feature_importance(self, trained):
        _model, report, _df = trained
        assert report.feature_importance
        assert sum(report.feature_importance.values()) == pytest.approx(1.0, abs=1e-6)
        # Grid position drives the synthetic target, so it must rank high.
        top = [name for name, _ in report.top_features(4)]
        assert any("grid" in name or "form_avg_pos" in name for name in top)

    def test_predicts_one_score_per_row(self, trained):
        model, _report, df = trained
        scores = model.predict(df[schema.RACE_FEATURE_COLS].head(10))
        assert scores.shape == (10,)
        assert np.isfinite(scores).all()

    def test_rank_returns_a_permutation(self, trained):
        model, _report, df = trained
        ranks = model.rank(df[schema.RACE_FEATURE_COLS].head(10))
        assert sorted(ranks) == list(range(1, 11))

    def test_predicting_before_fit_raises(self):
        model = RankingPredictor(schema.RACE_FEATURES)
        with pytest.raises(NotTrainedError):
            model.predict(pd.DataFrame({c: [0.0] for c in schema.RACE_FEATURE_COLS}))

    def test_missing_feature_column_raises(self, trained):
        model, _report, df = trained
        incomplete = df[schema.RACE_FEATURE_COLS].drop(columns=["grid_pos"])
        with pytest.raises(FeatureMismatchError, match="grid_pos"):
            model.predict(incomplete)

    def test_column_order_does_not_change_predictions(self, trained):
        """The model must project onto its own contract, not trust input order."""
        model, _report, df = trained
        X = df[schema.RACE_FEATURE_COLS].head(20)
        shuffled = X[list(reversed(schema.RACE_FEATURE_COLS))]
        np.testing.assert_allclose(model.predict(X), model.predict(shuffled))

    def test_sample_weights_are_accepted(self):
        df = _training_frame(n_races=12)
        model = RankingPredictor(schema.RACE_FEATURES, params={"max_iter": 20})
        weights = np.linspace(0.1, 1.0, len(df))
        report = model.fit(
            df[schema.RACE_FEATURE_COLS], df["race_pos"],
            groups=df["race_id"], sample_weight=weights,
            cv_folds=3, min_samples_for_cv=40,
        )
        assert report.n_samples == len(df)

    def test_rows_with_a_missing_target_are_dropped(self):
        df = _training_frame(n_races=12)
        df.loc[df.index[:20], "race_pos"] = np.nan
        model = RankingPredictor(schema.RACE_FEATURES, params={"max_iter": 20})
        report = model.fit(df[schema.RACE_FEATURE_COLS], df["race_pos"],
                           groups=df["race_id"])
        assert report.n_samples == len(df) - 20

    def test_too_few_rows_raises(self):
        df = _training_frame(n_races=1, n_drivers=5)
        model = RankingPredictor(schema.RACE_FEATURES)
        with pytest.raises(ValueError, match="Not enough training rows"):
            model.fit(df[schema.RACE_FEATURE_COLS], df["race_pos"])

    def test_unknown_estimator_raises(self):
        with pytest.raises(UnknownEstimatorError):
            make_regressor("magic_forest")

    @pytest.mark.parametrize("kind", ["hist_gradient_boosting", "gradient_boosting",
                                      "random_forest", "ridge"])
    def test_every_configured_estimator_trains(self, kind):
        df = _training_frame(n_races=12)
        model = RankingPredictor(schema.RACE_FEATURES, kind=kind)
        report = model.fit(df[schema.RACE_FEATURE_COLS], df["race_pos"],
                           groups=df["race_id"], cv_folds=2, min_samples_for_cv=40)
        assert report.n_samples > 0
        assert model.is_fitted


class TestDnfPredictor:
    def test_learns_and_bounds_probabilities(self):
        df = _training_frame(n_races=30)
        model = DnfPredictor(schema.DNF_FEATURES, params={"n_estimators": 30})
        report = model.fit(df[schema.DNF_FEATURE_COLS], df["dnf"],
                           groups=df["race_id"], cv_folds=3)

        assert report.is_classifier
        assert 0.0 < report.positive_rate < 0.5

        proba = model.predict_proba(df[schema.DNF_FEATURE_COLS])
        low, high = DnfPredictor.PROBABILITY_BOUNDS
        assert ((proba >= low) & (proba <= high)).all()

    def test_falls_back_to_the_base_rate_when_nobody_retired(self):
        df = _training_frame(n_races=10)
        df["dnf"] = False
        model = DnfPredictor(schema.DNF_FEATURES)
        report = model.fit(df[schema.DNF_FEATURE_COLS], df["dnf"])

        assert not model.is_fitted
        assert report.notes
        proba = model.predict_proba(df[schema.DNF_FEATURE_COLS])
        assert np.allclose(proba, 0.0)

    def test_untrained_model_still_returns_usable_probabilities(self):
        model = DnfPredictor(schema.DNF_FEATURES)
        X = pd.DataFrame({c: [0.0] * 5 for c in schema.DNF_FEATURE_COLS})
        proba = model.predict_proba(X)
        assert proba.shape == (5,)
        assert (proba > 0).all()


class TestMetrics:
    def test_perfect_prediction_scores_perfectly(self):
        actual = list(range(1, 11))
        metrics = evaluate_ranking(actual, actual)
        assert metrics.spearman == pytest.approx(1.0)
        assert metrics.mae == 0.0
        assert metrics.top1 == 1.0
        assert metrics.exact == 1.0
        assert metrics.podium_order == 1.0

    def test_reversed_prediction_is_maximally_wrong(self):
        actual = list(range(1, 11))
        metrics = evaluate_ranking(list(reversed(actual)), actual)
        assert metrics.spearman == pytest.approx(-1.0)
        assert metrics.top1 == 0.0

    def test_swapping_two_places_costs_a_little(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [2, 1, 3, 4, 5]
        metrics = evaluate_ranking(predicted, actual)
        assert metrics.mae == pytest.approx(0.4)
        assert metrics.top3 == 1.0  # same three drivers, different order
        assert metrics.within_1 == 1.0

    def test_missing_values_are_dropped(self):
        metrics = evaluate_ranking([1, 2, np.nan, 4], [1, 2, 3, 4])
        assert metrics.n == 3

    def test_too_few_rows_returns_empty_metrics(self):
        assert np.isnan(evaluate_ranking([1], [1]).spearman)

    def test_aggregate_averages_per_race(self):
        a = evaluate_ranking([1, 2, 3, 4], [1, 2, 3, 4])
        b = evaluate_ranking([4, 3, 2, 1], [1, 2, 3, 4])
        combined = aggregate([a, b])
        assert combined.spearman == pytest.approx(0.0)
        assert combined.n == 8

    def test_aggregate_of_nothing_is_empty(self):
        assert aggregate([]).n == 0

    def test_brier_score_rewards_calibration(self):
        confident_right = brier_score([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        confident_wrong = brier_score([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        assert confident_right == 0.0
        assert confident_wrong > confident_right


class TestRegistry:
    def test_models_carry_the_config_signature(self, cfg):
        model = registry.new_race_model(cfg, [2024])
        assert model.metadata.signature == cfg.feature_signature
        assert model.metadata.schema_version == schema.SCHEMA_VERSION
        assert model.metadata.seasons == [2024]

    def test_round_trips_through_disk(self, cfg):
        df = _training_frame(n_races=12)
        model = registry.new_race_model(cfg, [2024])
        model.fit(df[schema.RACE_FEATURE_COLS], df["race_pos"], groups=df["race_id"])

        C.ensure_dirs(cfg)
        C.save_model(cfg, C.RACE_MODEL, model)
        loaded = C.load_model(cfg, C.RACE_MODEL)

        assert loaded is not None
        np.testing.assert_allclose(
            model.predict(df[schema.RACE_FEATURE_COLS].head(5)),
            loaded.predict(df[schema.RACE_FEATURE_COLS].head(5)),
        )

    def test_a_stale_signature_is_rejected_on_load(self, cfg):
        df = _training_frame(n_races=12)
        model = registry.new_race_model(cfg, [2024])
        model.fit(df[schema.RACE_FEATURE_COLS], df["race_pos"], groups=df["race_id"])
        model.metadata.signature = "s1-ancient"

        C.ensure_dirs(cfg)
        C.save_model(cfg, C.RACE_MODEL, model)

        race, _quali, _dnf = registry.load_all(cfg)
        assert race is None, "a model from an older feature contract must be discarded"

    def test_missing_models_report_not_ready(self, cfg):
        C.ensure_dirs(cfg)
        assert registry.models_ready(cfg) is False
        card = registry.describe(cfg)
        assert card["models"][C.RACE_MODEL]["status"] == "missing or stale"
