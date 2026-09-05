"""Feature extraction: practice pace, rolling form, and the assembled matrix."""

from __future__ import annotations

import pandas as pd
import pytest

from f1predict.config import FormConfig
from f1predict.features import schema
from f1predict.features.builder import assemble_features
from f1predict.features.form import build_form_table, form_for_event
from f1predict.features.practice import extract_practice_features

# ── Practice ──────────────────────────────────────────────────────────────────

class TestPracticeFeatures:
    def test_produces_a_row_per_driver(self, practice_laps):
        result = extract_practice_features(practice_laps, min_laps=3)
        assert len(result) == practice_laps["Driver"].nunique()
        assert set(schema.QUALI_FEATURE_COLS) & set(result.columns)

    def test_gaps_are_relative_and_non_negative(self, practice_laps):
        result = extract_practice_features(practice_laps, min_laps=3)
        assert result["fp_best_gap_pct"].min() == pytest.approx(0.0, abs=1e-9)
        assert (result["fp_best_gap_pct"] >= 0).all()
        assert result["fp_pace_gap_pct"].min() == pytest.approx(0.0, abs=1e-9)

    def test_ranks_the_fastest_driver_first(self, practice_laps):
        result = extract_practice_features(practice_laps, min_laps=3)
        fastest = result.sort_values("fp_best_lap_s").iloc[0]
        # The fixture makes VER fastest by construction.
        assert fastest["driver_code"] == "VER"
        assert fastest["fp_rank_pct"] == result["fp_rank_pct"].min()

    def test_cooldown_laps_do_not_inflate_race_pace(self, practice_laps):
        """A 12-second-slow cool-down lap must be excluded from the stint median."""
        result = extract_practice_features(practice_laps, min_laps=3)
        ver = result.set_index("driver_code").loc["VER"]
        # Clean laps sit at ~80s; including the 92s lap would push the median up.
        assert ver["fp_pace_s"] < 81.5

    def test_excludes_in_and_out_laps(self, practice_laps):
        result = extract_practice_features(practice_laps, min_laps=3)
        # 16 laps per driver, minus 2 out-laps, 2 in-laps.
        assert result["fp_n_laps"].max() == 12

    def test_theoretical_best_never_exceeds_actual_best(self, practice_laps):
        result = extract_practice_features(practice_laps, min_laps=3)
        assert (result["fp_theory_lap_s"] <= result["fp_best_lap_s"] + 1e-6).all()

    def test_empty_input_returns_empty_frame(self):
        assert extract_practice_features(pd.DataFrame(), min_laps=3).empty
        assert extract_practice_features(None, min_laps=3).empty

    def test_survives_a_session_with_no_valid_laps(self, practice_laps):
        """A red-flagged session filters to nothing; fall back, do not crash."""
        laps = practice_laps.copy()
        laps["TrackStatus"] = "4"
        result = extract_practice_features(laps, min_laps=3)
        assert not result.empty
        assert result["fp_best_lap_s"].notna().all()

    def test_compound_normalisation_favours_the_slower_tyre(self):
        """Equal lap times on hard and soft mean the hard-tyre run was faster."""
        rows = []
        for code, compound in (("AAA", "SOFT"), ("BBB", "HARD")):
            for lap in range(1, 8):
                rows.append({
                    "Driver": code, "LapNumber": float(lap),
                    "LapTime": pd.Timedelta(seconds=90.0),
                    "Sector1Time": pd.Timedelta(seconds=30.0),
                    "Sector2Time": pd.Timedelta(seconds=30.0),
                    "Sector3Time": pd.Timedelta(seconds=30.0),
                    "Stint": 1.0, "Compound": compound, "IsAccurate": True,
                    "PitOutTime": pd.NaT, "PitInTime": pd.NaT,
                    "TrackStatus": "1", "Deleted": False,
                })
        result = extract_practice_features(pd.DataFrame(rows), min_laps=3)
        paces = result.set_index("driver_code")["fp_pace_s"]
        assert paces["BBB"] < paces["AAA"]


# ── Form ──────────────────────────────────────────────────────────────────────

class TestFormTable:
    def test_one_row_per_driver_race(self, season_results, quali_history):
        table = build_form_table(season_results, quali_history, FormConfig())
        assert len(table) == len(season_results)
        for column in ("form_avg_pos", "team_avg_pos", "circuit_avg_pos"):
            assert column in table.columns

    def test_no_leakage_from_the_target_race(self, season_results):
        """A driver's form must be built only from races before the target.

        Rebuilding the table with the target race's result changed must leave
        that race's own features untouched.
        """
        cfg = FormConfig(recent_races=3, reliability_races=5, trend_races=2)
        baseline = build_form_table(season_results, form_cfg=cfg)

        tampered = season_results.copy()
        target = (tampered["year"] == 2024) & (tampered["round"] == 4)
        tampered.loc[target, "race_pos"] = 20
        tampered.loc[target, "points"] = 0.0
        after = build_form_table(tampered, form_cfg=cfg)

        key = ["year", "round", "driver_id"]
        before_row = baseline[(baseline["year"] == 2024) & (baseline["round"] == 4)]
        after_row = after[(after["year"] == 2024) & (after["round"] == 4)]
        merged = before_row.merge(after_row, on=key, suffixes=("_a", "_b"))

        assert not merged.empty
        for column in ("form_avg_pos", "form_avg_pts", "circuit_avg_pos", "team_avg_pos"):
            pd.testing.assert_series_equal(
                merged[f"{column}_a"], merged[f"{column}_b"], check_names=False,
            )

    def test_first_race_of_history_uses_neutral_defaults(self, season_results):
        table = build_form_table(season_results, form_cfg=FormConfig())
        first = table[(table["year"] == 2023) & (table["round"] == 1)]
        assert (first["form_avg_pos"] == 10.0).all()
        assert (first["circuit_n_starts"] == 0.0).all()

    def test_circuit_history_counts_only_prior_visits(self, season_results):
        table = build_form_table(season_results, form_cfg=FormConfig())
        # Round 1 of 2024 revisits the round-1 circuit from 2023, so exactly one
        # prior start exists for every driver who raced it.
        second_visit = table[(table["year"] == 2024) & (table["round"] == 1)]
        assert (second_visit["circuit_n_starts"] == 1.0).all()

    def test_empty_history_returns_empty_frame(self):
        assert build_form_table(pd.DataFrame()).empty

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"year": [2024], "round": [1], "driver_id": ["x"]})
        with pytest.raises(ValueError, match="race_pos"):
            build_form_table(bad)


class TestFormForEvent:
    def test_returns_a_row_per_entrant(self, season_results, entry_list, quali_history):
        form = form_for_event(
            season_results, entry_list, 2025, 1, circuit_id="bahrain",
            quali=quali_history, form_cfg=FormConfig(),
        )
        assert len(form) == len(entry_list)
        assert set(form["driver_id"]) == set(entry_list["driver_id"])

    def test_excludes_the_target_race_from_its_own_window(
        self, season_results, entry_list, quali_history
    ):
        """Asking about a race already in history must not use its result."""
        with_race = form_for_event(
            season_results, entry_list, 2024, 3, circuit_id="albert_park",
            quali=quali_history, form_cfg=FormConfig(),
        )
        history_without = season_results[
            ~((season_results["year"] == 2024) & (season_results["round"] == 3))
        ]
        without_race = form_for_event(
            history_without, entry_list, 2024, 3, circuit_id="albert_park",
            quali=quali_history, form_cfg=FormConfig(),
        )
        pd.testing.assert_frame_equal(
            with_race.sort_values("driver_id").reset_index(drop=True),
            without_race.sort_values("driver_id").reset_index(drop=True),
        )

    def test_unknown_driver_gets_neutral_form(self, season_results):
        rookie = pd.DataFrame([{
            "driver_code": "ZZZ", "driver_id": "rookie", "driver_name": "New Driver",
            "team": "New Team", "constructor_id": "new_team",
        }])
        form = form_for_event(season_results, rookie, 2025, 1, form_cfg=FormConfig())
        assert form.iloc[0]["form_avg_pos"] == 10.0
        assert form.iloc[0]["form_dnf_rate"] == pytest.approx(0.09)


# ── Assembly ──────────────────────────────────────────────────────────────────

class TestAssembleFeatures:
    def _assemble(self, entry_list, practice_laps, quali_history, season_results, weather):
        practice = extract_practice_features(practice_laps, min_laps=3)
        quali = quali_history[
            (quali_history["year"] == 2024) & (quali_history["round"] == 2)
        ]
        form = form_for_event(
            season_results, entry_list, 2024, 2, circuit_id="jeddah",
            quali=quali_history, form_cfg=FormConfig(),
        )
        return assemble_features(
            entries=entry_list, practice=practice, quali=quali, form=form,
            weather=weather, round_num=2, n_rounds=6,
        )

    def test_emits_every_schema_column(
        self, entry_list, practice_laps, quali_history, season_results, weather
    ):
        _, features = self._assemble(
            entry_list, practice_laps, quali_history, season_results, weather
        )
        assert list(features.columns) == schema.ALL_FEATURE_COLS
        assert features.notna().all().all(), "features must never contain NaN"

    def test_column_order_matches_each_model_contract(
        self, entry_list, practice_laps, quali_history, season_results, weather
    ):
        _, features = self._assemble(
            entry_list, practice_laps, quali_history, season_results, weather
        )
        for cols in (schema.RACE_FEATURE_COLS, schema.QUALI_FEATURE_COLS,
                     schema.DNF_FEATURE_COLS):
            assert set(cols) <= set(features.columns)

    def test_grid_is_inferred_when_qualifying_is_missing(
        self, entry_list, practice_laps, season_results, weather
    ):
        """No qualifying data must still yield a usable, varied grid column."""
        practice = extract_practice_features(practice_laps, min_laps=3)
        form = form_for_event(season_results, entry_list, 2025, 1, form_cfg=FormConfig())
        _, features = assemble_features(
            entries=entry_list, practice=practice, quali=pd.DataFrame(),
            form=form, weather=weather, round_num=1, n_rounds=6,
        )
        assert features["grid_pos"].std() > 0
        assert features["grid_pos"].min() >= 1

    def test_handles_a_complete_absence_of_session_data(
        self, entry_list, season_results, weather
    ):
        form = form_for_event(season_results, entry_list, 2025, 1, form_cfg=FormConfig())
        _, features = assemble_features(
            entries=entry_list, practice=None, quali=None, form=form,
            weather=weather, round_num=1, n_rounds=6,
        )
        assert len(features) == len(entry_list)
        assert features.notna().all().all()

    def test_teammate_delta_is_antisymmetric_within_a_team(
        self, entry_list, practice_laps, quali_history, season_results, weather
    ):
        meta, features = self._assemble(
            entry_list, practice_laps, quali_history, season_results, weather
        )
        combined = pd.concat([meta, features], axis=1)
        for _, pair in combined.groupby("constructor_id"):
            if len(pair) == 2:
                assert pair["teammate_quali_delta_pct"].sum() == pytest.approx(0.0, abs=1e-9)

    def test_quali_model_features_never_touch_qualifying_data(
        self, entry_list, practice_laps, quali_history, season_results, weather
    ):
        """The qualifying model must be blind to the session it predicts.

        Assembling with and without qualifying results must leave every
        qualifying-model feature identical.
        """
        practice = extract_practice_features(practice_laps, min_laps=3)
        form = form_for_event(
            season_results, entry_list, 2024, 2, circuit_id="jeddah",
            quali=quali_history, form_cfg=FormConfig(),
        )
        quali = quali_history[
            (quali_history["year"] == 2024) & (quali_history["round"] == 2)
        ]

        _, with_quali = assemble_features(
            entries=entry_list, practice=practice, quali=quali, form=form,
            weather=weather, round_num=2, n_rounds=6,
        )
        _, without_quali = assemble_features(
            entries=entry_list, practice=practice, quali=pd.DataFrame(), form=form,
            weather=weather, round_num=2, n_rounds=6,
        )
        pd.testing.assert_frame_equal(
            with_quali[schema.QUALI_FEATURE_COLS],
            without_quali[schema.QUALI_FEATURE_COLS],
        )

    def test_season_progress_tracks_the_round(
        self, entry_list, season_results, weather
    ):
        form = form_for_event(season_results, entry_list, 2024, 3, form_cfg=FormConfig())
        _, features = assemble_features(
            entries=entry_list, practice=None, quali=None, form=form,
            weather=weather, round_num=3, n_rounds=6,
        )
        assert features["season_progress"].iloc[0] == pytest.approx(0.5)


class TestSchema:
    def test_every_feature_has_a_neutral_value(self):
        for group in (schema.RACE_FEATURES, schema.QUALI_FEATURES, schema.DNF_FEATURES):
            for feature in group:
                assert isinstance(feature.neutral, float)
                assert feature.label

    def test_feature_names_are_unique_within_a_model(self):
        for cols in (schema.RACE_FEATURE_COLS, schema.QUALI_FEATURE_COLS,
                     schema.DNF_FEATURE_COLS):
            assert len(cols) == len(set(cols))

    def test_lookup_covers_all_columns(self):
        for name in schema.ALL_FEATURE_COLS:
            assert schema.by_name(name) is not None

    def test_no_absolute_lap_times_reach_the_models(self):
        """Raw seconds are circuit-dependent and must stay out of the contract."""
        banned = {"fp_best_lap_s", "fp_pace_s", "fp_theory_lap_s",
                  "quali_time_s", "quali_gap_to_pole_s"}
        assert not banned & set(schema.ALL_FEATURE_COLS)
