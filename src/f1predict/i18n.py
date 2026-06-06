"""Minimal bilingual string store (EN / IT) for the Streamlit UI."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title": "🏎️ F1 Race Predictor",
        "subtitle": "Real-time predictions powered by FastF1 · Jolpica · Open-Meteo",
        "select_year": "Season",
        "select_gp": "Grand Prix",
        "predict_btn": "Predict",
        "train_btn": "Train models",
        "tab_race": "Race",
        "tab_quali": "Qualifying",
        "col_pos": "Pos",
        "col_driver": "Driver",
        "col_team": "Team",
        "col_pred_pos": "Predicted",
        "col_p_win": "P(win)",
        "col_p_podium": "P(podium)",
        "col_p_top5": "P(top 5)",
        "col_quali_pos": "Quali pos",
        "col_quali_time": "Predicted time",
        "chart_title": "Podium probability",
        "loading": "Loading data and running predictions…",
        "no_model": "No trained model found. Run `f1predict train` first, or click 'Train models'.",
        "training": "Training models on seasons {}…",
        "done": "Done!",
        "backtest_header": "Backtest — {} Round {}",
        "spearman": "Spearman ρ",
        "top3_acc": "Top-3 accuracy",
        "circuit": "Circuit",
        "race_date": "Race date",
        "data_source": "Data: FP available — using Practice → Quali prediction | Quali available — using actual grid",
        "warning_no_fp": "FP data not available for this event.",
        "warning_no_quali": "Qualifying data not available yet — predicting from FP.",
    },
    "it": {
        "title": "🏎️ F1 Previsioni Gara",
        "subtitle": "Previsioni in tempo reale con FastF1 · Jolpica · Open-Meteo",
        "select_year": "Stagione",
        "select_gp": "Gran Premio",
        "predict_btn": "Prevedi",
        "train_btn": "Allena modelli",
        "tab_race": "Gara",
        "tab_quali": "Qualifiche",
        "col_pos": "Pos",
        "col_driver": "Pilota",
        "col_team": "Team",
        "col_pred_pos": "Previsto",
        "col_p_win": "P(vittoria)",
        "col_p_podium": "P(podio)",
        "col_p_top5": "P(top 5)",
        "col_quali_pos": "Pos. qualifiche",
        "col_quali_time": "Tempo previsto",
        "chart_title": "Probabilità podio",
        "loading": "Caricamento dati e calcolo previsioni…",
        "no_model": "Nessun modello trovato. Esegui `f1predict train` o clicca 'Allena modelli'.",
        "training": "Alleno i modelli sulle stagioni {}…",
        "done": "Fatto!",
        "backtest_header": "Backtest — {} Round {}",
        "spearman": "Spearman ρ",
        "top3_acc": "Accuratezza top-3",
        "circuit": "Circuito",
        "race_date": "Data gara",
        "data_source": "Dati: FP disponibili → previsione qualifiche | Qualifiche disponibili → griglia reale",
        "warning_no_fp": "Dati prove libere non disponibili per questo evento.",
        "warning_no_quali": "Qualifiche non ancora disponibili — previsione da prove libere.",
    },
}


def t(key: str, lang: str = "en", *args) -> str:
    s = STRINGS.get(lang, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))
    if args:
        return s.format(*args)
    return s
