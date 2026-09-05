<div align="center">

# 🏎️ F1 Predictor

**Two-stage Formula 1 race forecasting, built entirely on free and open data.**

Practice pace → predicted qualifying → predicted race → win probabilities.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-189%20passing-2ECC71)](tests/)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![API keys](https://img.shields.io/badge/API%20keys-none%20needed-E8002D)](#data-sources)

</div>

---

## What it does

Most F1 predictors stop at "who is quickest". This one answers the question
people actually ask — **how likely is each driver to win?** — and it does so
before qualifying has even happened.

The pipeline runs in two stages:

```
                 ┌──────────────────────────────────────────┐
  Free Practice  │  best lap · long-run stints · theoretical │
   lap timing ──▶│  best lap · gap to teammate              │──┐
                 └──────────────────────────────────────────┘  │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │  QUALIFYING  MODEL    │
                                                    │  → predicted grid     │
                                                    └───────────┬───────────┘
                                                                │
  Driver & team form ────────┐                                  │
  Circuit history ───────────┤                                  │
  Weather forecast ──────────┼──────────────────────────────────┤
  Real grid (once quali runs)┘                                  ▼
                                                    ┌───────────────────────┐
                                                    │     RACE  MODEL       │
                                                    │  + retirement model   │
                                                    └───────────┬───────────┘
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │  20 000 simulated     │
                                                    │  races → P(win),      │
                                                    │  P(podium), P(points) │
                                                    └───────────────────────┘
```

If qualifying has already run, the real grid is used and confidence is **high**.
If it hasn't, stage one's predicted order becomes the grid and confidence drops
to **medium**. The app always tells you which.

## Highlights

| | |
|---|---|
| 🎯 **Two-stage forecasting** | Predict the race days before qualifying, from practice pace alone |
| 🎲 **Real probabilities** | 20 000 Monte Carlo races per prediction, with heteroscedastic spread — the midfield is modelled as less predictable than the front row, and rain widens everything |
| 💥 **Retirement modelling** | A dedicated classifier estimates each car's DNF risk and feeds it into the simulation, so a fast-but-fragile car is priced accordingly |
| 🏆 **Championship projection** | Simulate every remaining round to get title probabilities and projected points ranges |
| 🔍 **Explainability** | Ask *why* a driver is rated where they are and get their standout factors versus the field |
| 📐 **Honest evaluation** | Race-grouped cross-validation, plus season backtests that flag when a race was in the training set |
| 🌐 **Bilingual UI** | Full English / Italian interface |
| ✈️ **Works offline** | Falls back to a bundled 1950-2023 Ergast dump when the network is unavailable |

## Quick start

```bash
git clone https://github.com/checcolazzarotto/F1-Prediction
cd F1-Prediction
pip install -e .

f1predict train              # first run: downloads practice data, ~5 min/season
f1predict predict --next     # predict the next race
f1predict serve              # or open the web UI
```

## The web UI

```bash
f1predict serve              # → http://localhost:8501
```

Six tabs, all driven by the same pipeline:

- **Race** — predicted classification, podium cards in team colours, win/podium
  probability bars, an expected-finish plot with P10–P90 whiskers, a full
  position-distribution heatmap, a grid→finish slope chart, per-driver
  explanations, and head-to-head odds between any two drivers.
- **Qualifying** — the predicted grid with the practice signals behind it.
- **Championship** — title probabilities and projected points for drivers and
  constructors.
- **Accuracy** — backtest one race or a whole season, with per-round scores.
- **Calendar** — the season schedule with weekend formats.
- **Model** — model card: estimator, training data, CV scores, feature
  importances, cache usage.

## CLI

```bash
# Predictions
f1predict predict --next                     # next upcoming race
f1predict predict --last                     # most recent race
f1predict predict -y 2025 -g monza           # by name
f1predict predict -y 2025 -r 16 --explain VER  # with an explanation
f1predict predict --next -o out.csv          # export CSV or JSON

# Qualifying only
f1predict quali --next

# Training
f1predict train                              # seasons from config
f1predict train 2023 2024 2025               # explicit seasons
f1predict train --refresh                    # rebuild cached features

# Evaluation
f1predict backtest 2024 21                   # one race
f1predict backtest 2024                      # whole season

# Season context
f1predict championship                       # drivers' title odds
f1predict championship --constructors
f1predict standings 2025
f1predict schedule 2025                      # with a countdown to the next race

# Housekeeping
f1predict info                               # model card + cache usage
f1predict clear-cache features
```

<details>
<summary><b>Sample output</b></summary>

```
────────────────── Italian Grand Prix 2026 · Round 13 ──────────────────
┌────────────────────────────────────────────────────────────────────┐
│ Circuit  Autodromo Nazionale di Monza  Date        2026-09-06 13:00 │
│ Grid     predicted from practice       Confidence  MEDIUM           │
│ Weather  31°C · rain 0% · wind 2 m/s   Practice    FP2              │
│                                                                     │
│ Predicted podium: George Russell · Kimi Antonelli · Lando Norris     │
└─────────────────────────────────────────────────────────────────────┘
 Pos  Grid    Δ  Driver             Team           Win  Podium  Points   DNF
   1     1    =  George Russell     Mercedes     28.7%   66.7%   91.4%  5.2%
   2     2    =  Kimi Antonelli     Mercedes     24.4%   60.6%   91.1%  5.4%
   3     3    =  Lando Norris       McLaren      18.9%   53.5%   91.0%  6.1%
   4     4    =  Oscar Piastri      McLaren      14.8%   46.9%   90.2%  6.4%
   5     5    =  Oliver Bearman     Haas F1 Team  3.2%   16.8%   84.1%  7.9%
```

</details>

## How it works

### Features

Every feature is **circuit-relative** — gaps as percentages, positions as
fractions of the field. Absolute lap times are deliberately excluded: 71 seconds
means opposite things at the Red Bull Ring and at Spa, so feeding raw seconds to
a cross-circuit model teaches it the calendar rather than the pace.

**From practice** — best lap gap, long-run race pace (stint medians with
cool-down laps stripped and tyre compounds normalised), theoretical best lap
from best sectors, laps completed, gap to teammate.

**From history** — rolling driver form (finishes, points, places gained, trend),
retirement rate over a longer window, per-circuit record, and team form
aggregated per race weekend.

**From context** — grid position and gap to pole, air temperature, wind, rain
probability, and how far the season has progressed.

### Models

| Model | Job | Target |
|---|---|---|
| `RankingPredictor` (race) | Order the field | Finishing position, **classified finishers only** |
| `RankingPredictor` (quali) | Order the grid | Qualifying position |
| `DnfPredictor` | Retirement risk | Did-not-finish, feeding the simulation |

Gradient boosting by default (`hist_gradient_boosting`), configurable to
`gradient_boosting`, `random_forest` or `ridge`. Imputation lives inside the
sklearn pipeline so fill values are learned per fold, and recent races carry
more weight through an exponential decay.

The race model trains only on cars that saw the flag. A retirement says nothing
about pace, and including it would teach the model that pole sitters finish
19th; retirements re-enter through the Monte Carlo instead.

### Honest evaluation

Cross-validation is **grouped by race**, so no race ever appears in both the
training and validation folds — without that, two drivers from the same event
leak each other's context and every score looks better than it is.

Typical out-of-sample performance, three seasons of training data:

| Model | Rank correlation (ρ) | Mean error | Podium hit rate |
|---|---|---|---|
| Race | **0.77** | 2.5 places | 68% |
| Qualifying | **0.68** | 3.4 places | 54% |
| Retirement | ROC AUC **0.61** | — | base rate 12.9% |

Retirement is the weakest of the three, and honestly so: whether a car breaks or
gets hit is close to a coin flip that the pre-race data barely constrains. It
still beats assuming every car finishes, which is what a single point prediction
implicitly does.

`f1predict backtest <year> <round>` warns you when the race you are scoring was
part of the training set, because those numbers flatter the model.

## Data sources

All free, all keyless.

| Source | Provides |
|---|---|
| [FastF1](https://docs.fastf1.dev/) | Practice/qualifying/race sessions, lap and sector times, entry lists, calendars |
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) | Historical results, qualifying, standings, circuits (the maintained Ergast successor) |
| [Open-Meteo](https://open-meteo.com/) | Weather — historical archive and 16-day forecast |
| `Dataset/` | Bundled Ergast dump, 1950-2023, used as an offline fallback |

Requests are rate-limited and disk-cached; completed seasons are cached
permanently, the running season expires after six hours.

## Project layout

```
src/f1predict/
├── config.py            Typed dataclass config; unknown YAML keys are an error
├── constants.py         Points systems, DNF rules, team colours, tyre offsets
├── cache.py             Parquet / JSON / model store, schema-aware and atomic
├── data/
│   ├── http.py          Rate-limited, disk-cached JSON client
│   ├── jolpica.py       Ergast-successor API
│   ├── fastf1_source.py Sessions, laps, entry lists
│   ├── weather.py       Open-Meteo archive + forecast
│   ├── offline.py       Bundled CSV fallback
│   ├── repository.py    One façade over all of the above
│   └── schedule.py      Event resolution, sprint-aware session availability
├── features/
│   ├── schema.py        The feature contract — single source of truth
│   ├── practice.py      Long-run pace, theoretical best lap
│   ├── form.py          Vectorised rolling form, leakage-free
│   └── builder.py       Assembly, shared by training and prediction
├── models/
│   ├── base.py          Estimator factory, training reports, metadata
│   ├── predictor.py     Ranking regressor + retirement classifier
│   ├── metrics.py       Spearman, top-k, Brier
│   └── registry.py      Persistence with schema-signature checks
├── simulation/
│   ├── race.py          Vectorised Monte Carlo
│   └── championship.py  Season-long title projection
├── pipeline.py          Orchestration — the only entry point the UIs use
├── reporting.py         Rich renderers
├── i18n.py              EN / IT strings
└── cli.py               Typer CLI
app/
├── streamlit_app.py     Web UI
├── components.py        Charts and layout blocks
└── theme.py             Palette, CSS, chart defaults
```

## Configuration

Edit [`config/default.yaml`](config/default.yaml). Every key maps onto a
dataclass, so a typo fails at load time with the list of valid keys instead of
being silently ignored.

```yaml
training:
  seasons: [2022, 2023, 2024, 2025]
  recency_half_life_races: 25.0   # older races count for less

simulation:
  n_simulations: 20000
  position_noise_std: 2.6         # baseline spread per driver
  backmarker_noise_scale: 0.9     # the midfield is less predictable
  wet_noise_multiplier: 1.8       # rain widens everything

form:
  recent_races: 5
  reliability_races: 12
```

Environment overrides: `F1PREDICT_CACHE_DIR`, `F1PREDICT_SIMULATIONS`.
Per-run overrides: `f1predict --cache-dir /tmp/x --config my.yaml <command>`.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 189 tests, no network access
ruff check .
```

The suite runs entirely on synthetic fixtures. Several tests exist specifically
to pin down correctness properties — data leakage, train/serve symmetry, and the
predicted grid actually reaching the race model. See [CLAUDE.md](CLAUDE.md) for
the invariants and the reasoning behind them.

## Caveats

- F1 is genuinely hard to predict. A rank correlation of ~0.7 means the model
  gets the broad shape right and will still be embarrassed by a safety car.
- Weather for an upcoming race is a forecast; for a past race the model sees
  what actually happened. Backtest numbers are mildly optimistic because of it.
- Practice pace is confounded by fuel loads and engine modes, which are not
  published. Long-run stint medians reduce this but cannot remove it.
- The bundled offline dataset stops at 2023, so offline mode cannot train on
  more recent seasons.

## License

MIT
