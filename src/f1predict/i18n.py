"""Bilingual UI strings (English / Italian).

Keys missing from a language fall back to English, and a key missing everywhere
falls back to itself, so a new string never renders as a blank label.
"""

from __future__ import annotations

from typing import Final

LANGUAGES: Final[dict[str, str]] = {"en": "English", "it": "Italiano"}

STRINGS: Final[dict[str, dict[str, str]]] = {
    "en": {
        # Shell
        "app_title": "F1 Race Predictor",
        "app_tagline": "Two-stage forecasting from free, open data",
        "language": "Language",
        "season": "Season",
        "grand_prix": "Grand Prix",
        "predict": "Predict race",
        "train": "Train models",
        "refresh": "Refresh data",
        "settings": "Settings",
        "simulations": "Simulations",
        # Tabs
        "tab_race": "Race",
        "tab_quali": "Qualifying",
        "tab_championship": "Championship",
        "tab_backtest": "Accuracy",
        "tab_calendar": "Calendar",
        "tab_model": "Model",
        # Table headers
        "pos": "Pos",
        "driver": "Driver",
        "team": "Team",
        "grid": "Grid",
        "delta": "Δ",
        "p_win": "Win",
        "p_podium": "Podium",
        "p_points": "Points",
        "p_top5": "Top 5",
        "p_dnf": "DNF",
        "expected_points": "Exp. pts",
        "quali_pos": "Predicted quali",
        "gap": "Gap",
        # Race context
        "circuit": "Circuit",
        "race_date": "Race date",
        "weather": "Weather",
        "confidence": "Confidence",
        "grid_source": "Grid source",
        "grid_actual": "Actual qualifying result",
        "grid_predicted": "Predicted from practice",
        "grid_form": "Form only — no session data",
        "practice_used": "Practice session used",
        "rain_risk": "Rain risk",
        "temperature": "Temperature",
        "wind": "Wind",
        # Championship
        "title_odds": "Title probability",
        "races_left": "Rounds remaining",
        "current_points": "Points",
        "projected_points": "Projected",
        "drivers_title": "Drivers' championship",
        "constructors_title": "Constructors' championship",
        "title_decided": "The championship is mathematically settled.",
        # Accuracy
        "spearman": "Rank correlation",
        "mae": "Mean position error",
        "winner_hit": "Winner correct",
        "podium_hit": "Podium hit rate",
        "within_3": "Within 3 places",
        "run_backtest": "Run backtest",
        "backtest_season": "Backtest whole season",
        # Messages
        "loading": "Crunching the numbers…",
        "no_model": "No trained model yet. Click **Train models** to build one.",
        "training_on": "Training on seasons {}…",
        "done": "Done",
        "no_data": "No data available for this event yet.",
        "no_practice": "No practice data available for this event.",
        "select_prompt": "Pick a season and a Grand Prix, then hit Predict.",
        "why": "Why this prediction",
        "strength": "in their favour",
        "weakness": "against them",
        "field_median": "field median",
        "podium_chart": "Podium probability",
        "outcome_spread": "Where each driver could finish",
        "vs": "Head to head",
    },
    "it": {
        # Shell
        "app_title": "F1 Race Predictor",
        "app_tagline": "Previsioni a due stadi da dati liberi e aperti",
        "language": "Lingua",
        "season": "Stagione",
        "grand_prix": "Gran Premio",
        "predict": "Prevedi gara",
        "train": "Allena modelli",
        "refresh": "Aggiorna dati",
        "settings": "Impostazioni",
        "simulations": "Simulazioni",
        # Tabs
        "tab_race": "Gara",
        "tab_quali": "Qualifiche",
        "tab_championship": "Mondiale",
        "tab_backtest": "Accuratezza",
        "tab_calendar": "Calendario",
        "tab_model": "Modello",
        # Table headers
        "pos": "Pos",
        "driver": "Pilota",
        "team": "Team",
        "grid": "Griglia",
        "delta": "Δ",
        "p_win": "Vittoria",
        "p_podium": "Podio",
        "p_points": "Punti",
        "p_top5": "Top 5",
        "p_dnf": "Ritiro",
        "expected_points": "Punti att.",
        "quali_pos": "Qualifica prevista",
        "gap": "Distacco",
        # Race context
        "circuit": "Circuito",
        "race_date": "Data gara",
        "weather": "Meteo",
        "confidence": "Affidabilità",
        "grid_source": "Origine griglia",
        "grid_actual": "Qualifiche reali",
        "grid_predicted": "Prevista dalle libere",
        "grid_form": "Solo forma — nessun dato di sessione",
        "practice_used": "Sessione libere usata",
        "rain_risk": "Probabilità pioggia",
        "temperature": "Temperatura",
        "wind": "Vento",
        # Championship
        "title_odds": "Probabilità titolo",
        "races_left": "Gare rimanenti",
        "current_points": "Punti",
        "projected_points": "Previsti",
        "drivers_title": "Mondiale piloti",
        "constructors_title": "Mondiale costruttori",
        "title_decided": "Il campionato è matematicamente chiuso.",
        # Accuracy
        "spearman": "Correlazione di rango",
        "mae": "Errore medio di posizione",
        "winner_hit": "Vincitore corretto",
        "podium_hit": "Podio indovinato",
        "within_3": "Entro 3 posizioni",
        "run_backtest": "Esegui backtest",
        "backtest_season": "Backtest stagione intera",
        # Messages
        "loading": "Sto elaborando i dati…",
        "no_model": "Nessun modello allenato. Clicca **Allena modelli** per crearlo.",
        "training_on": "Alleno sulle stagioni {}…",
        "done": "Fatto",
        "no_data": "Nessun dato disponibile per questo evento.",
        "no_practice": "Nessun dato delle prove libere per questo evento.",
        "select_prompt": "Scegli stagione e Gran Premio, poi premi Prevedi.",
        "why": "Perché questa previsione",
        "strength": "a suo favore",
        "weakness": "a suo sfavore",
        "field_median": "mediana del gruppo",
        "podium_chart": "Probabilità di podio",
        "outcome_spread": "Dove può arrivare ogni pilota",
        "vs": "Testa a testa",
    },
}


def t(key: str, lang: str = "en", *args) -> str:
    """Translate ``key``, falling back to English and then to the key itself."""
    table = STRINGS.get(lang, STRINGS["en"])
    text = table.get(key) or STRINGS["en"].get(key, key)
    return text.format(*args) if args else text


def translator(lang: str):
    """Bind a language, so a caller can write ``_("season")``."""
    def _t(key: str, *args) -> str:
        return t(key, lang, *args)

    return _t


def grid_source_label(source: str, lang: str = "en") -> str:
    """Human-readable description of where the grid came from."""
    return t(
        {
            "actual_quali": "grid_actual",
            "predicted_quali": "grid_predicted",
            "form_only": "grid_form",
        }.get(source, "grid_form"),
        lang,
    )
