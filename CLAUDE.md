# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A two-stage Formula 1 predictor. Stage one turns Free Practice timing into a
predicted qualifying order; stage two turns a grid (real or predicted) plus
driver form, team form and weather into a predicted race result with Monte Carlo
outcome probabilities. Everything runs on free, keyless public data.

## Commands

```bash
pip install -e ".[dev]"      # install with dev tooling

pytest                       # full suite (~2 min; no network, everything mocked)
pytest tests/test_features.py -q          # one module
pytest -k "leak" -q                       # the leakage guarantees
ruff check .                 # lint (must be clean)
ruff check . --fix           # autofix

f1predict train 2023 2024 2025   # first run downloads practice data (slow)
f1predict predict --next         # predict the upcoming race
f1predict backtest 2024          # score a whole season
f1predict info                   # model card + cache usage
f1predict serve                  # launch the web UI
```

Training the first time downloads a FastF1 practice session per race and takes
several minutes per season. Everything is cached under `.cache/`, so subsequent
runs take seconds. `f1predict clear-cache features` forces a rebuild.

## Architecture

Data flows in one direction, and each layer only knows about the one below it:

```
cli.py / app/          presentation — Rich tables, Streamlit pages
    ↓
pipeline.py            orchestration — the only entry point the UIs touch
    ↓
features/  models/  simulation/
    ↓
data/repository.py     one façade over every source
    ↓
data/{jolpica,fastf1_source,weather,offline}.py
```

### The pieces that matter

- **`features/schema.py`** is the single source of truth for what the models
  consume. Adding, removing or renaming a feature means bumping
  `SCHEMA_VERSION`, which automatically invalidates cached feature parquet
  files *and* cached models. Never skip the bump.
- **`features/form.py`** computes rolling form for the whole history in one
  vectorised pass. Every window is shifted so a race is never in its own
  feature window.
- **`features/builder.py`** has one function, `assemble_features`, used by both
  training and prediction. That symmetry is the point — do not add a
  training-only or prediction-only branch to it.
- **`simulation/race.py`** converts model scores into probabilities with one
  vectorised NumPy expression. Do not reintroduce a Python loop over simulations.
- **`data/repository.py`** decides between the Jolpica API, the parquet cache
  and the bundled CSV dump. Call this, not the individual sources.

## Invariants

These are load-bearing. Several of them exist because the corresponding bug was
already shipped once.

1. **No leakage.** A feature for round *N* may only use data from rounds < *N*.
   Team form is aggregated per race weekend *before* the window is taken,
   because a constructor's two cars are adjacent rows and a naive `shift(1)`
   would let one driver see their teammate's result in the race being predicted.
   `tests/test_features.py::TestFormTable::test_no_leakage_from_the_target_race`
   guards this.
2. **The qualifying model never sees qualifying.** It predicts that session, so
   it uses `teammate_fp_delta_pct` (practice-derived), never
   `teammate_quali_delta_pct` (qualifying-derived). Guarded by
   `test_quali_model_features_never_touch_qualifying_data`.
3. **The predicted grid must reach the race model.** When qualifying has not
   run, stage one's output becomes stage two's `grid_pos`. Guarded by
   `test_predicts_the_grid_when_qualifying_has_not_run`.
4. **Features are circuit-relative.** Gaps are percentages, positions are
   fractions of the field. Absolute lap times in seconds must never enter the
   schema — a 71 s lap means opposite things at the Red Bull Ring and at Spa.
   Guarded by `test_no_absolute_lap_times_reach_the_models`.
5. **The race model trains on classified finishers only.** A retirement says
   nothing about pace. Retirements are modelled separately by `DnfPredictor`
   and reintroduced in the Monte Carlo.
6. **Cached artefacts carry their schema.** Models embed
   `cfg.feature_signature`; loading a mismatch returns `None` so the caller
   retrains. Cached frames are validated against `required_columns`.

## Conventions

- Type hints everywhere; `from __future__ import annotations` at the top.
- Comments explain *why*, never *what*. Most existing comments document a
  non-obvious decision or a bug that was fixed — preserve that.
- Data sources degrade to empty frames or `None`, never partial results.
  Catch the specific exception (`NetworkError`, `KeyError`, `ValueError`), not
  bare `Exception`, except where a comment justifies it.
- New user-facing strings go in `i18n.py` in **both** `en` and `it`.
  `test_i18n_covers_both_languages` fails otherwise.
- Rich/Streamlit rendering lives in `reporting.py` and `app/components.py`.
  Keep it out of `pipeline.py`.

## Testing

The suite never touches the network. `tests/conftest.py` provides synthetic
seasons, a FastF1-shaped practice frame and a temp-directory config. Pipeline
tests patch every source through `ExitStack`.

When adding a feature, add a test that would fail without it. When fixing a bug,
add the test that catches it first — several existing tests were written that
way and the comments say so.

## Gotchas

- **Windows encoding.** `cli.py` calls `_configure_stdio()` at import to force
  UTF-8; without it, `f1predict train > log.txt` dies on cp1252.
- **FastF1 logging.** Setting the parent logger's level is not enough; FastF1
  assigns explicit levels to each child. Use `_quiet_fastf1()`.
- **FastF1 cache.** Must be enabled before the first schedule lookup, or FastF1
  prints a warning and downloads into a temp directory.
- **Sprint weekends** have no FP2/FP3. `schedule.py` matches sessions by *name*,
  not slot number.
- **`pd.to_numeric(None)`** returns a scalar NaN, not a Series. Check
  `"col" in df.columns` rather than using `df.get("col")` for numeric coercion.
- **`DataFrame.attrs` does not survive parquet.** Persist metadata as a column.
- **Ergast blanks `position` for retirements**; use `positionOrder`.
- The `Dataset/` CSVs are the bundled offline fallback (1950-2023) and are read
  by `data/offline.py`. They are not dead weight — do not delete them.
