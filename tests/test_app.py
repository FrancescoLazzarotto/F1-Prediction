"""Smoke tests for the Streamlit app and the CLI surface.

These do not assert on content — they assert that every page renders and every
command is wired up, which is the class of breakage that unit tests miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_path() -> str:
    if str(ROOT / "app") not in sys.path:
        sys.path.insert(0, str(ROOT / "app"))
    return str(APP)


class TestWebApp:
    """The app talks to the real cache, so it needs trained models to be useful.

    It must still render without raising when they are missing, which is the
    first-run experience.
    """

    @pytest.fixture(scope="class")
    def rendered(self):
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest

        if str(ROOT / "app") not in sys.path:
            sys.path.insert(0, str(ROOT / "app"))

        app = AppTest.from_file(str(APP), default_timeout=180)
        app.run()
        return app

    def test_renders_without_exceptions(self, rendered):
        assert not rendered.exception, [str(e.value) for e in rendered.exception]

    def test_sidebar_offers_season_and_event_pickers(self, rendered):
        labels = [widget.label for widget in rendered.sidebar.selectbox]
        assert any("Season" in label or "Stagione" in label for label in labels)

    def test_has_a_predict_and_a_train_button(self, rendered):
        labels = [button.label for button in rendered.sidebar.button]
        assert len(labels) >= 2

    def test_shows_every_tab(self, rendered):
        # Race, Qualifying, Championship, Accuracy, Calendar, Model.
        assert len(rendered.tabs) == 6

    def test_language_toggle_switches_strings(self, rendered):
        pytest.importorskip("streamlit.testing.v1")
        rendered.sidebar.radio[0].set_value("it").run()
        assert not rendered.exception
        labels = [widget.label for widget in rendered.sidebar.selectbox]
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
