# F1-Prediction

Modular, multi-race Formula 1 predictor powered entirely by **free, open APIs** — no API keys required.

## What it does

Two-stage prediction pipeline:

1. **Free Practice → Qualifying prediction** — extracts long-run pace, best laps, and sector times from FP sessions to forecast the qualifying order before the session happens.
2. **Qualifying → Race prediction** — combines grid position, driver and team form (last N races), circuit history, and weather forecast to predict the race finishing order and compute win/podium probabilities via Monte Carlo simulation.

## Data sources (all free, no keys)

| Source | What it provides |
|---|---|
| [FastF1 3.8+](https://docs.fastf1.dev/) | FP/Quali/Race sessions, lap times, telemetry, entry lists |
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) | Historical results, standings, circuit info (Ergast successor) |
| [Open-Meteo](https://open-meteo.com/) | Weather — historical archive + 16-day forecast, no key needed |

## Setup

```bash
pip install -e .
```

First run downloads session data into `.cache/` (takes several minutes). Subsequent runs are instant.

## Usage

### CLI

```bash
# Predict next upcoming race
f1predict predict --next

# Predict a specific race (full output: order + podium probabilities)
f1predict predict --year 2026 --round 10

# Predict by GP name
f1predict predict --year 2026 --gp "Monza"

# Predict qualifying order from FP data only
f1predict quali --year 2026 --gp Silverstone

# Train / retrain models on specific seasons
f1predict train 2023 2024 2025

# Evaluate accuracy on a past race
f1predict backtest 2024 3
```

### Web UI (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Opens a browser with season/GP dropdowns, race prediction table, and a podium probability chart. Language toggle (EN/IT) in the sidebar.

## Architecture

```
src/f1predict/
  config.py               — YAML + .env config loader
  cache.py                — FastF1 cache + parquet feature store + model store
  data/
    schedule.py           — Resolve "next race", year/round, GP name
    fastf1_source.py      — Load FP/Quali/Race sessions and lap data
    jolpica_source.py     — Historical results and circuit info
    weather_source.py     — Open-Meteo historical + forecast weather
  features/
    practice.py           — FP best lap, long-run pace, sector times
    qualifying.py         — Grid position, gap to pole
    form.py               — Rolling driver/team form, circuit history
    builder.py            — Assemble and cache the full feature matrix
  models/
    base.py               — Predictor interface
    quali_model.py        — GradientBoosting: FP features → quali position
    race_model.py         — GradientBoosting: grid + form + weather → race pos + MC proba
    registry.py           — Model instantiation from config
  pipeline.py             — Orchestration: train / predict_race / predict_quali / backtest
  i18n.py                 — EN/IT strings for the UI
  cli.py                  — Typer CLI (predict, quali, train, backtest)
app/
  streamlit_app.py        — Streamlit web UI
config/
  default.yaml            — Training seasons, model hyperparameters, cache paths
tests/                    — pytest unit tests
Dataset/                  — Legacy Ergast CSVs (kept as offline fallback)
```

## Running tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config/default.yaml` to change:
- `training.seasons` — which seasons to train on (default: 2023, 2024, 2025)
- `models.gradient_boosting.*` — model hyperparameters
- `monte_carlo.n_simulations` — number of simulations for probability estimates
- `form.recent_races` — rolling window for driver form (default: 5)
- `cache.*_dir` — cache directory locations (or set `F1PREDICT_CACHE_DIR` env var)
