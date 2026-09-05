"""Monte Carlo race and championship simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predict.config import SimulationConfig
from f1predict.constants import RACE_POINTS, is_dnf, points_for, team_color
from f1predict.simulation import simulate_championship, simulate_race, summarise

CODES = [f"D{i:02d}" for i in range(10)]
SCORES = np.arange(1.0, 11.0)


class TestRaceSimulation:
    def test_positions_form_a_valid_permutation_each_run(self, fast_sim_cfg):
        sim = simulate_race(SCORES, CODES, cfg=fast_sim_cfg)
        assert sim.positions.shape == (fast_sim_cfg.n_simulations, len(CODES))
        expected = np.arange(1, len(CODES) + 1)
        for row in sim.positions[:25]:
            assert np.array_equal(np.sort(row), expected)

    def test_win_probabilities_sum_to_one(self, fast_sim_cfg):
        probabilities = simulate_race(SCORES, CODES, cfg=fast_sim_cfg).probabilities()
        assert probabilities["p_win"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_are_ordered_by_pace(self, fast_sim_cfg):
        probabilities = simulate_race(
            SCORES, CODES, dnf_probabilities=np.zeros(len(CODES)), cfg=fast_sim_cfg
        ).probabilities()
        assert probabilities["p_win"].iloc[0] > probabilities["p_win"].iloc[-1]
        assert probabilities["p_podium"].is_monotonic_decreasing

    def test_nested_probabilities_are_consistent(self, fast_sim_cfg):
        p = simulate_race(SCORES, CODES, cfg=fast_sim_cfg).probabilities()
        assert (p["p_win"] <= p["p_podium"] + 1e-12).all()
        assert (p["p_podium"] <= p["p_top5"] + 1e-12).all()
        assert (p["p_top5"] <= p["p_top10"] + 1e-12).all()

    def test_retirement_rate_matches_the_input(self, fast_sim_cfg):
        rates = np.full(len(CODES), 0.25)
        sim = simulate_race(SCORES, CODES, dnf_probabilities=rates, cfg=fast_sim_cfg)
        assert sim.probabilities()["p_dnf"].mean() == pytest.approx(0.25, abs=0.05)

    def test_a_certain_retirement_never_wins(self, fast_sim_cfg):
        rates = np.zeros(len(CODES))
        rates[0] = 1.0  # the fastest car always breaks
        sim = simulate_race(SCORES, CODES, dnf_probabilities=rates, cfg=fast_sim_cfg)
        probabilities = sim.probabilities().set_index("driver_code")
        assert probabilities.loc["D00", "p_win"] == 0.0
        assert probabilities.loc["D00", "p_dnf"] == 1.0
        # The retired car still classifies behind every finisher.
        assert probabilities.loc["D00", "expected_pos"] == len(CODES)

    def test_rain_widens_the_outcome_spread(self, fast_sim_cfg):
        dry = simulate_race(SCORES, CODES, rain_probability=0.0, cfg=fast_sim_cfg)
        wet = simulate_race(SCORES, CODES, rain_probability=1.0, cfg=fast_sim_cfg)
        assert wet.probabilities()["p_win"].iloc[0] < dry.probabilities()["p_win"].iloc[0]

    def test_backmarkers_have_a_wider_spread_than_the_front_row(self, fast_sim_cfg):
        cfg = SimulationConfig(
            n_simulations=4000, seed=5, backmarker_noise_scale=1.5,
            base_dnf_rate=0.0,
        )
        sim = simulate_race(
            SCORES, CODES, grid_positions=SCORES,
            dnf_probabilities=np.zeros(len(CODES)), cfg=cfg,
        )
        spread = sim.positions.std(axis=0)
        assert spread[-1] > spread[0]

    def test_head_to_head_is_complementary(self, fast_sim_cfg):
        sim = simulate_race(SCORES, CODES, cfg=fast_sim_cfg)
        forward = sim.head_to_head("D00", "D05")
        backward = sim.head_to_head("D05", "D00")
        assert forward + backward == pytest.approx(1.0)
        assert forward > 0.5

    def test_head_to_head_rejects_unknown_drivers(self, fast_sim_cfg):
        sim = simulate_race(SCORES, CODES, cfg=fast_sim_cfg)
        with pytest.raises(KeyError):
            sim.head_to_head("D00", "NOPE")

    def test_position_distribution_rows_sum_to_one(self, fast_sim_cfg):
        sim = simulate_race(SCORES, CODES, cfg=fast_sim_cfg)
        distribution = sim.position_distribution(max_position=len(CODES))
        assert distribution.sum(axis=1).round(6).eq(1.0).all()

    def test_expected_points_respect_the_scoring_table(self, fast_sim_cfg):
        sim = simulate_race(
            SCORES, CODES, dnf_probabilities=np.zeros(len(CODES)), cfg=fast_sim_cfg
        )
        probabilities = sim.probabilities()
        assert probabilities["expected_points"].iloc[0] <= RACE_POINTS[1]
        assert probabilities["expected_points"].is_monotonic_decreasing

    def test_is_reproducible_for_a_fixed_seed(self, fast_sim_cfg):
        a = simulate_race(SCORES, CODES, cfg=fast_sim_cfg).probabilities()
        b = simulate_race(SCORES, CODES, cfg=fast_sim_cfg).probabilities()
        pd.testing.assert_frame_equal(a, b)

    def test_rejects_an_empty_field(self, fast_sim_cfg):
        with pytest.raises(ValueError):
            simulate_race(np.array([]), [], cfg=fast_sim_cfg)

    def test_summarise_orders_by_predicted_position(self, fast_sim_cfg):
        meta = pd.DataFrame({
            "driver_code": CODES, "driver_name": CODES, "team": ["T"] * len(CODES),
        })
        sim = simulate_race(SCORES, CODES, cfg=fast_sim_cfg)
        table = summarise(meta, SCORES, sim)
        assert table["predicted_pos"].tolist() == list(range(1, len(CODES) + 1))
        assert table["driver_code"].iloc[0] == "D00"


class TestChampionshipSimulation:
    @pytest.fixture
    def standings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "position": [1, 2, 3, 4],
            "driver_id": ["a", "b", "c", "d"],
            "driver_name": ["Alice", "Bob", "Cara", "Dan"],
            "team": ["T1", "T1", "T2", "T2"],
            "points": [200.0, 180.0, 90.0, 40.0],
        })

    def test_title_probabilities_sum_to_one(self, standings, fast_sim_cfg):
        outlook = simulate_championship(
            standings, races_remaining=5, cfg=fast_sim_cfg
        )
        assert outlook.table["p_title"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_no_races_left_means_the_leader_is_certain(self, standings, fast_sim_cfg):
        outlook = simulate_championship(standings, races_remaining=0, cfg=fast_sim_cfg)
        assert outlook.is_decided
        assert outlook.table.iloc[0]["driver_id"] == "a"
        assert outlook.table.iloc[0]["p_title"] == 1.0

    def test_points_never_decrease(self, standings, fast_sim_cfg):
        outlook = simulate_championship(standings, races_remaining=6, cfg=fast_sim_cfg)
        for _, row in outlook.table.iterrows():
            assert row["expected_points"] >= row["current_points"]
            assert row["points_p10"] >= row["current_points"]

    def test_inactive_entrants_stop_scoring(self, standings, fast_sim_cfg):
        """A driver who has left the grid keeps their points but gains none."""
        outlook = simulate_championship(
            standings, races_remaining=8, active_keys={"a", "b", "c"}, cfg=fast_sim_cfg
        )
        dan = outlook.table.set_index("driver_id").loc["d"]
        assert dan["expected_points"] == pytest.approx(40.0)
        assert dan["points_p10"] == dan["points_p90"] == 40.0
        assert dan["p_title"] == 0.0

    def test_more_races_left_gives_the_chaser_a_better_chance(
        self, standings, fast_sim_cfg
    ):
        pace = pd.Series({"a": 1.0, "b": 1.2, "c": 5.0, "d": 8.0})
        few = simulate_championship(
            standings, pace=pace, races_remaining=1, cfg=fast_sim_cfg
        ).table.set_index("driver_id")
        many = simulate_championship(
            standings, pace=pace, races_remaining=15, cfg=fast_sim_cfg
        ).table.set_index("driver_id")
        assert many.loc["b", "p_title"] > few.loc["b", "p_title"]

    def test_sprints_add_points(self, standings, fast_sim_cfg):
        without = simulate_championship(
            standings, races_remaining=4, cfg=fast_sim_cfg
        ).table["expected_points"].sum()
        with_sprints = simulate_championship(
            standings, races_remaining=4, sprints_remaining=3, cfg=fast_sim_cfg
        ).table["expected_points"].sum()
        assert with_sprints > without

    def test_constructor_mode_uses_team_keys(self, fast_sim_cfg):
        constructors = pd.DataFrame({
            "position": [1, 2],
            "constructor_id": ["t1", "t2"],
            "team": ["Team One", "Team Two"],
            "points": [400.0, 250.0],
        })
        outlook = simulate_championship(
            constructors, races_remaining=4, cfg=fast_sim_cfg,
            label_column="team", key_column="constructor_id",
        )
        assert "constructor_id" in outlook.table.columns
        assert outlook.table["p_title"].sum() == pytest.approx(1.0)

    def test_rejects_empty_standings(self, fast_sim_cfg):
        with pytest.raises(ValueError):
            simulate_championship(pd.DataFrame(), races_remaining=3, cfg=fast_sim_cfg)


class TestConstants:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            # Classified finishes. The two sources spell "a lap down"
            # differently: Jolpica says "Lapped", Ergast says "+1 Lap".
            ("Finished", False),
            ("Lapped", False),
            ("+1 Lap", False),
            ("+2 Laps", False),
            # Genuine retirements.
            ("Engine", True),
            ("Retired", True),
            ("Accident", True),
            ("Collision damage", True),
            ("Undertray", True),
            ("Disqualified", True),
            ("Did not start", True),
            ("Withdrew", True),
            # Absent status: nothing to conclude.
            ("", False),
            (None, False),
        ],
    )
    def test_dnf_classification(self, status, expected):
        assert is_dnf(status) is expected

    def test_lapped_cars_are_not_retirements(self):
        """Regression: Jolpica reports lapped finishers as "Lapped".

        Treating that as a retirement misclassified about a third of every
        field, which corrupted both the DNF features and the race model's
        training split.
        """
        assert is_dnf("Lapped") is False

    def test_points_table(self):
        assert points_for(1) == 25
        assert points_for(10) == 1
        assert points_for(11) == 0
        assert points_for(1, sprint=True) == 8

    @pytest.mark.parametrize(
        ("team", "expected"),
        [
            ("Oracle Red Bull Racing", "#3671C6"),
            ("Scuderia Ferrari", "#E8002D"),
            ("McLaren Formula 1 Team", "#FF8000"),
            ("Stake F1 Team Kick Sauber", "#52E252"),
        ],
    )
    def test_team_colours_survive_sponsor_names(self, team, expected):
        assert team_color(team) == expected

    def test_unknown_team_gets_a_fallback_colour(self):
        assert team_color("Some New Team").startswith("#")
        assert team_color(None).startswith("#")
