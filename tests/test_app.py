"""Smoke tests for the Streamlit app and the CLI surface.

These do not assert on content — they assert that every page renders and every
command is wired up, which is the class of breakage that unit tests miss.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_path() -> str:
    if str(ROOT / "app") not in sys.path:
        sys.path.insert(0, str(ROOT / "app"))
    return str(APP)


def _calendar() -> pd.DataFrame:
    """A three-round calendar straddling today, in FastF1's column shape."""
    now = datetime.now(tz=timezone.utc)
    dates = [now - timedelta(days=30), now + timedelta(days=7), now + timedelta(days=21)]
    return pd.DataFrame({
        "RoundNumber": [1, 2, 3],
        "EventName": ["Bahrain Grand Prix", "Monaco Grand Prix", "Italian Grand Prix"],
        "OfficialEventName": ["F1 Bahrain GP", "F1 Monaco GP", "F1 Italian GP"],
        "Location": ["Sakhir", "Monte Carlo", "Monza"],
        "Country": ["Bahrain", "Monaco", "Italy"],
        "CircuitShortName": ["Sakhir", "Monaco", "Monza"],
        "EventFormat": ["conventional"] * 3,
        "EventDate": [pd.Timestamp(d) for d in dates],
        "Session5": ["Race"] * 3,
        "Session5DateUtc": [pd.Timestamp(d) for d in dates],
    })


def _standings() -> pd.DataFrame:
    return pd.DataFrame({
        "position": [1, 2],
        "driver_id": ["verstappen", "norris"],
        "driver_code": ["VER", "NOR"],
        "driver_name": ["Max Verstappen", "Lando Norris"],
        "team": ["Red Bull", "McLaren"],
        "constructor_id": ["red_bull", "mclaren"],
        "points": [300.0, 250.0],
        "wins": [8, 5],
    })


@contextmanager
def _isolated_app(models_ready: bool):
    """Run the app with every external source stubbed out.

    The app reaches for the FastF1 calendar and the Jolpica standings on its
    very first render. Left unstubbed, this test would depend on two live
    services and on whether the developer happens to have trained models
    sitting in their cache — which is exactly how it passed locally and failed
    in CI.
    """
    pytest.importorskip("streamlit.testing.v1")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from f1predict.data import repository as repo
    from f1predict.data import schedule as schedule_module
    from f1predict.models import registry

    if str(ROOT / "app") not in sys.path:
        sys.path.insert(0, str(ROOT / "app"))

    # Streamlit memoises across AppTest instances, and schedule lookups are
    # lru_cached, so both have to be reset or the stubs never take effect.
    st.cache_data.clear()
    st.cache_resource.clear()
    schedule_module.get_schedule.cache_clear()

    with ExitStack() as stack:
        stack.enter_context(patch.object(schedule_module, "get_schedule", lambda year: _calendar()))
        stack.enter_context(patch.object(repo, "driver_standings", lambda *a, **k: _standings()))
        stack.enter_context(
            patch.object(repo, "constructor_standings", lambda *a, **k: pd.DataFrame())
        )
        stack.enter_context(patch.object(repo, "season_results", lambda *a, **k: pd.DataFrame()))
        stack.enter_context(patch.object(registry, "models_ready", lambda cfg: models_ready))

        app = AppTest.from_file(str(APP), default_timeout=120)
        app.run()
        yield app

    st.cache_data.clear()
    st.cache_resource.clear()
    schedule_module.get_schedule.cache_clear()


class TestWebAppFirstRun:
    """Before anything is trained, the app must explain itself rather than break."""

    def test_renders_without_exceptions(self):
        with _isolated_app(models_ready=False) as app:
            assert not app.exception, [str(e.value) for e in app.exception]

    def test_prompts_for_training_instead_of_showing_tabs(self):
        with _isolated_app(models_ready=False) as app:
            assert len(app.tabs) == 0
            assert app.warning, "the first run should say that no model exists yet"


class TestWebApp:
    """With models available, every tab must render."""

    def test_renders_without_exceptions(self):
        with _isolated_app(models_ready=True) as app:
            assert not app.exception, [str(e.value) for e in app.exception]

    def test_sidebar_offers_season_and_event_pickers(self):
        with _isolated_app(models_ready=True) as app:
            labels = [widget.label for widget in app.sidebar.selectbox]
            assert any("Season" in label for label in labels)
            assert any("Grand Prix" in label for label in labels)

    def test_has_a_predict_and_a_train_button(self):
        with _isolated_app(models_ready=True) as app:
            assert len(app.sidebar.button) >= 2

    def test_shows_every_tab(self):
        # Race, Qualifying, Championship, Accuracy, Calendar, Model.
        with _isolated_app(models_ready=True) as app:
            assert len(app.tabs) == 6

    def test_language_toggle_switches_strings(self):
        with _isolated_app(models_ready=True) as app:
            app.sidebar.radio[0].set_value("it").run()
            assert not app.exception, [str(e.value) for e in app.exception]
            labels = [widget.label for widget in app.sidebar.selectbox]
            assert any("Stagione" in label for label in labels)


class TestCliSurface:
    def test_every_command_is_registered(self):
        from typer.main import get_command

        from f1predict.cli import app

        commands = set(get_command(app).commands)
        assert commands == {
            "version", "predict", "quali", "train", "backtest",
            "championship", "schedule", "standings", "info",
            "clear-cache", "serve",
        }

    def test_help_renders_for_each_command(self):
        from typer.testing import CliRunner

        from f1predict.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        for command in ("predict", "quali", "train", "backtest", "championship",
                        "schedule", "standings", "info", "clear-cache", "serve"):
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, f"{command} --help failed"

    def test_version_command(self):
        from typer.testing import CliRunner

        from f1predict import __version__
        from f1predict.cli import app

        result = CliRunner().invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestPackaging:
    def test_theme_and_components_import_standalone(self, app_path):
        """The app modules must import without Streamlit page context."""
        import components  # noqa: F401
        import theme

        assert theme.CSS
        assert theme.medal_color(1) == theme.GOLD

    def test_i18n_covers_both_languages(self):
        from f1predict.i18n import LANGUAGES, STRINGS

        english = set(STRINGS["en"])
        italian = set(STRINGS["it"])
        assert english == italian, f"untranslated keys: {english ^ italian}"
        assert set(LANGUAGES) == {"en", "it"}

    def test_translation_falls_back_to_english(self):
        from f1predict.i18n import t

        assert t("app_title", "de") == t("app_title", "en")
        assert t("totally_unknown_key", "en") == "totally_unknown_key"

    def test_translation_formats_arguments(self):
        from f1predict.i18n import t

        assert "2024" in t("training_on", "en", "2024")
