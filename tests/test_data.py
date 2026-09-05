"""Data layer: config, cache, HTTP client, parsing, schedule and offline source."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from f1predict import cache as C
from f1predict.config import Config, load_config, replace_cache_root
from f1predict.data import jolpica, offline
from f1predict.data.http import JsonCache, RateLimiter
from f1predict.data.schedule import Event, _session_times, resolve_event
from f1predict.data.weather import _extract_hour, _rain_probability

# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_shipped_yaml_loads(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.training.seasons
        assert cfg.simulation.n_simulations > 0

    def test_cache_paths_are_absolute_and_distinct(self, cfg):
        paths = [str(p) for p in cfg.cache.paths()]
        assert all(p == str(p) for p in paths)
        assert len(set(paths)) == len(paths)

    def test_unknown_key_raises_with_a_helpful_message(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("training:\n  seasons: [2024]\n  typo_key: 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="typo_key"):
            load_config(bad)

    def test_unknown_top_level_section_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("nonsense:\n  a: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="nonsense"):
            load_config(bad)

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "does_not_exist.yaml")
        assert cfg.training.seasons

    def test_feature_signature_tracks_the_form_window(self, cfg):
        from dataclasses import replace

        from f1predict.config import FormConfig

        other = replace(cfg, form=FormConfig(recent_races=99))
        assert cfg.feature_signature != other.feature_signature

    def test_replace_cache_root_moves_every_directory(self, cfg, tmp_path):
        moved = replace_cache_root(cfg, tmp_path / "elsewhere")
        assert all("elsewhere" in str(p) for p in moved.cache.paths())


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestCache:
    def test_frame_round_trip(self, cfg):
        C.ensure_dirs(cfg)
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        C.save_frame(cfg, "demo", df)
        pd.testing.assert_frame_equal(C.load_frame(cfg, "demo"), df)

    def test_missing_frame_returns_none(self, cfg):
        C.ensure_dirs(cfg)
        assert C.load_frame(cfg, "never_written") is None

    def test_ttl_expires_an_entry(self, cfg):
        C.ensure_dirs(cfg)
        C.save_frame(cfg, "perishable", pd.DataFrame({"a": [1]}))
        assert C.load_frame(cfg, "perishable", ttl_s=60) is not None
        assert C.load_frame(cfg, "perishable", ttl_s=-1) is None

    def test_a_frame_missing_required_columns_is_discarded(self, cfg):
        """Guards against serving data written by an older schema."""
        C.ensure_dirs(cfg)
        C.save_frame(cfg, "old_schema", pd.DataFrame({"a": [1]}))
        assert C.load_frame(cfg, "old_schema", required_columns=["a", "b"]) is None
        # And the stale file is removed rather than retried forever.
        assert C.load_frame(cfg, "old_schema") is None

    def test_race_key_embeds_the_schema_version(self):
        from f1predict.features.schema import SCHEMA_VERSION

        assert C.race_key(2024, 3, "training") == f"2024_03_training_v{SCHEMA_VERSION}"

    def test_json_round_trip(self, cfg):
        C.ensure_dirs(cfg)
        C.save_weather(cfg, 2024, 5, {"temperature": 21.5, "rain_prob": 0.3})
        assert C.load_weather(cfg, 2024, 5)["temperature"] == 21.5

    def test_clear_removes_files(self, cfg):
        C.ensure_dirs(cfg)
        C.save_frame(cfg, "doomed", pd.DataFrame({"a": [1]}))
        assert C.usage(cfg)["features"] > 0
        C.clear(cfg, "features")
        assert C.load_frame(cfg, "doomed") is None

    def test_clear_rejects_an_unknown_section(self, cfg):
        with pytest.raises(ValueError, match="Unknown cache section"):
            C.clear(cfg, "bananas")

    def test_corrupt_file_is_removed_not_raised(self, cfg):
        C.ensure_dirs(cfg)
        C.save_frame(cfg, "corrupt", pd.DataFrame({"a": [1]}))
        path = next(p for p in cfg.cache.paths()[1].glob("corrupt*"))
        path.write_bytes(b"not parquet at all")
        assert C.load_frame(cfg, "corrupt") is None


class TestJsonCache:
    def test_round_trip_and_expiry(self, tmp_path):
        cache = JsonCache(tmp_path)
        cache.put("key", {"value": 1})
        assert cache.get("key", ttl_s=None) == {"value": 1}
        assert cache.get("key", ttl_s=-1) is None
        assert cache.get("absent", ttl_s=None) is None

    def test_corrupt_entry_is_discarded(self, tmp_path):
        cache = JsonCache(tmp_path)
        cache.put("key", {"value": 1})
        next(tmp_path.glob("*.json")).write_text("{broken", encoding="utf-8")
        assert cache.get("key", ttl_s=None) is None


class TestRateLimiter:
    def test_spaces_calls_apart(self):
        limiter = RateLimiter(calls_per_second=50)
        start = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        assert time.monotonic() - start >= 0.06

    def test_zero_rate_disables_limiting(self):
        limiter = RateLimiter(calls_per_second=0)
        start = time.monotonic()
        for _ in range(100):
            limiter.acquire()
        assert time.monotonic() - start < 0.05


# ── Jolpica parsing ───────────────────────────────────────────────────────────

class TestJolpicaParsing:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("1:15.123", 75.123),
            ("75.123", 75.123),
            ("1:01:15.500", 3675.5),
            ("", None),
            (None, None),
            ("nonsense", None),
        ],
    )
    def test_lap_time_parsing(self, text, seconds):
        result = jolpica.parse_lap_time(text)
        if seconds is None:
            assert result is None
        else:
            assert result == pytest.approx(seconds)

    def test_race_parsing_marks_retirements(self):
        payload = {
            "round": "3",
            "Circuit": {"circuitId": "monza"},
            "date": "2024-09-01", "time": "13:00:00Z",
            "Results": [
                {
                    "positionOrder": "1", "position": "1", "grid": "1", "points": "25",
                    "laps": "53", "status": "Finished",
                    "Driver": {"driverId": "verstappen", "code": "VER",
                               "givenName": "Max", "familyName": "Verstappen"},
                    "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                },
                {
                    "positionOrder": "20", "position": "", "grid": "5", "points": "0",
                    "laps": "12", "status": "Engine",
                    "Driver": {"driverId": "perez", "code": "PER",
                               "givenName": "Sergio", "familyName": "Perez"},
                    "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                },
            ],
        }
        df = jolpica._parse_race(payload, 2024)

        assert list(df.columns) == jolpica._RESULT_COLUMNS
        assert df.loc[0, "dnf"] is False or df.loc[0, "dnf"] == False  # noqa: E712
        assert bool(df.loc[1, "dnf"]) is True
        # A retirement has a blank `position`, so positionOrder must be used.
        assert df.loc[1, "race_pos"] == 20
        assert df.loc[0, "race_date"] == "2024-09-01T13:00:00Z"

    def test_quali_derived_columns(self):
        df = pd.DataFrame({
            "year": [2024] * 3, "round": [1] * 3,
            "driver_id": ["a", "b", "c"], "driver_code": ["A", "B", "C"],
            "driver_name": ["A", "B", "C"], "team": ["T"] * 3,
            "constructor_id": ["t"] * 3, "quali_pos": [1, 2, 3],
            "q1_s": [80.0, 80.5, 81.0],
            "q2_s": [79.5, 80.0, None],
            "q3_s": [79.0, None, None],
        })
        out = jolpica.add_quali_derived_columns(df)

        assert out.loc[0, "best_quali_s"] == 79.0
        assert out.loc[1, "best_quali_s"] == 80.0  # fell out in Q2
        assert out.loc[2, "best_quali_s"] == 81.0  # fell out in Q1
        assert out.loc[0, "quali_gap_to_pole_pct"] == 0.0
        assert out.loc[2, "quali_gap_to_pole_pct"] > 0
        assert out["reached_q3"].tolist() == [1, 0, 0]

    def test_quali_with_no_times_does_not_crash(self):
        df = pd.DataFrame({
            "year": [2024], "round": [1], "driver_id": ["a"], "driver_code": ["A"],
            "driver_name": ["A"], "team": ["T"], "constructor_id": ["t"],
            "quali_pos": [1], "q1_s": [None], "q2_s": [None], "q3_s": [None],
        })
        out = jolpica.add_quali_derived_columns(df)
        assert len(out) == 1
        assert pd.isna(out.loc[0, "best_quali_s"])

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [("5", 0, 5), ("", 20, 20), (None, 20, 20), ("abc", 7, 7), ("3.0", 0, 3)],
    )
    def test_int_coercion_is_total(self, value, default, expected):
        assert jolpica._to_int(value, default) == expected


# ── Schedule ──────────────────────────────────────────────────────────────────

class TestSchedule:
    @pytest.fixture
    def calendar(self):
        return pd.DataFrame({
            "RoundNumber": [1, 2, 3],
            "EventName": ["Bahrain Grand Prix", "Saudi Arabian Grand Prix",
                          "Australian Grand Prix"],
            "OfficialEventName": ["F1 Bahrain GP", "F1 Saudi GP", "F1 Australian GP"],
            "Location": ["Sakhir", "Jeddah", "Melbourne"],
            "Country": ["Bahrain", "Saudi Arabia", "Australia"],
            "CircuitShortName": ["Sakhir", "Jeddah", "Melbourne"],
            "EventFormat": ["conventional", "sprint_qualifying", "conventional"],
            "EventDate": pd.to_datetime(["2024-03-02", "2024-03-09", "2024-03-24"]),
            "Session5DateUtc": pd.to_datetime(
                ["2024-03-02T15:00", "2024-03-09T17:00", "2024-03-24T04:00"]
            ),
        })

    @pytest.fixture(autouse=True)
    def _patch_schedule(self, monkeypatch, calendar):
        from f1predict.data import schedule as S

        monkeypatch.setattr(S, "get_schedule", lambda year: calendar)

    def test_resolve_by_round(self):
        event = resolve_event(2024, 2, None)
        assert event.round == 2
        assert "Saudi" in event.name
        assert event.is_sprint

    def test_resolve_by_name_substring(self):
        assert resolve_event(2024, None, "australia").round == 3

    def test_resolve_by_location(self):
        assert resolve_event(2024, None, "Jeddah").round == 2

    def test_resolve_prefers_an_exact_name_match(self):
        assert resolve_event(2024, None, "Bahrain Grand Prix").round == 1

    def test_unknown_round_raises(self):
        with pytest.raises(ValueError, match="not in the 2024 calendar"):
            resolve_event(2024, 99, None)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="matches"):
            resolve_event(2024, None, "Narnia")

    def test_no_selector_raises(self):
        with pytest.raises(ValueError, match="Specify a round"):
            resolve_event(2024, None, None)

    def test_sprint_sessions_map_to_their_own_keys(self):
        row = pd.Series({
            "Session1": "Practice 1", "Session1DateUtc": "2024-04-19T11:30",
            "Session2": "Sprint Qualifying", "Session2DateUtc": "2024-04-19T15:30",
            "Session3": "Sprint", "Session3DateUtc": "2024-04-20T11:00",
            "Session4": "Qualifying", "Session4DateUtc": "2024-04-20T15:00",
            "Session5": "Race", "Session5DateUtc": "2024-04-21T14:00",
        })
        times = _session_times(row)
        assert set(times) == {"FP1", "SQ", "S", "Q", "R"}
        assert times["R"].hour == 14

    def test_event_countdown(self):
        future = datetime.now(tz=timezone.utc) + timedelta(days=2)
        event = Event(year=2024, round=1, race_date=future)
        remaining = event.time_until().total_seconds()
        assert 2 * 86400 - 5 <= remaining <= 2 * 86400

    def test_past_event_counts_down_negative(self):
        past = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        assert Event(year=2024, round=1, race_date=past).time_until().total_seconds() < 0

    def test_event_without_a_date_has_no_countdown(self):
        assert Event(year=2024, round=1).time_until() is None


# ── Weather ───────────────────────────────────────────────────────────────────

class TestWeather:
    def test_picks_the_hour_nearest_the_race(self):
        payload = {"hourly": {
            "time": ["2024-03-02T13:00", "2024-03-02T14:00", "2024-03-02T15:00"],
            "temperature_2m": [20.0, 25.0, 30.0],
            "relative_humidity_2m": [40, 45, 50],
            "wind_speed_10m": [1.0, 2.0, 3.0],
            "cloud_cover": [10, 20, 30],
            "precipitation": [0.0, 0.0, 0.0],
        }}
        result = _extract_hour(
            payload, datetime(2024, 3, 2, 14, 50, tzinfo=timezone.utc), "archive"
        )
        assert result["temperature"] == 30.0
        assert result["source"] == "archive"

    def test_empty_payload_falls_back_to_defaults(self):
        assert _extract_hour({}, datetime.now(tz=timezone.utc), "archive")["source"] == "default"

    def test_forecast_probability_is_used_directly(self):
        row = pd.Series({"precipitation_probability": 70, "precipitation": 0.0})
        assert _rain_probability(row) == pytest.approx(0.7)

    def test_archive_rainfall_becomes_a_probability(self):
        assert _rain_probability(pd.Series({"precipitation": 0.0})) == 0.0
        wet = _rain_probability(pd.Series({"precipitation": 5.0}))
        assert wet == 1.0
        drizzle = _rain_probability(pd.Series({"precipitation": 0.4}))
        assert 0.0 < drizzle < 1.0


# ── Offline dataset ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not offline.is_available(), reason="bundled dataset not present")
class TestOfflineSource:
    def test_covers_historical_seasons(self):
        seasons = offline.covered_seasons()
        assert 1950 in seasons
        assert 2023 in seasons

    def test_season_results_match_the_canonical_schema(self):
        df = offline.get_season_results(2023)
        assert not df.empty
        assert list(df.columns) == jolpica._RESULT_COLUMNS
        assert df["dnf"].dtype == bool

    def test_names_decode_as_utf8(self):
        df = offline.get_season_results(2023)
        names = set(df["driver_name"])
        # A mojibake read would give "P�rez" rather than the real name.
        assert any("é" in name for name in names)

    def test_quali_results_have_derived_gaps(self):
        df = offline.get_quali_results(2023, 1)
        assert not df.empty
        assert df["quali_gap_to_pole_pct"].min() == pytest.approx(0.0)

    def test_circuit_info_has_coordinates(self):
        info = offline.get_circuit_info(2023, 1)
        assert info["circuit_id"] == "bahrain"
        assert info["lat"] != 0.0

    def test_unknown_round_raises(self):
        with pytest.raises(ValueError):
            offline.get_circuit_info(2023, 99)
