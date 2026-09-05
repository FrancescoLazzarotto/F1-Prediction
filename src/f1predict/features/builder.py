"""Assemble the feature matrix for one event, or for a whole training set.

Training and prediction both funnel through :func:`assemble_features`, so the
columns a model is fitted on are produced by the same code that produces the
columns it is asked to predict from. That symmetry is what keeps offline
metrics honest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1predict import cache as C
from f1predict.config import Config
from f1predict.data import fastf1_source as F1
from f1predict.data import repository as repo
from f1predict.data import weather as WX
from f1predict.features import schema
from f1predict.features.form import FORM_DEFAULTS, form_for_event
from f1predict.features.practice import extract_practice_features

log = logging.getLogger(__name__)

META_COLUMNS = ["driver_code", "driver_id", "driver_name", "team", "constructor_id"]

#: How the grid used for a prediction was obtained.
GRID_ACTUAL = "actual_quali"
GRID_PREDICTED = "predicted_quali"
GRID_FORM = "form_only"


@dataclass(slots=True)
class EventFeatures:
    """Everything a model needs for one race, plus the context to explain it."""

    meta: pd.DataFrame
    features: pd.DataFrame
    weather: dict = field(default_factory=dict)
    circuit: dict = field(default_factory=dict)
    grid_source: str = GRID_FORM
    practice_session: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.meta.empty or self.features.empty

    def frame(self) -> pd.DataFrame:
        """Meta and features side by side, for inspection and export."""
        return pd.concat(
            [self.meta.reset_index(drop=True), self.features.reset_index(drop=True)], axis=1
        )


# ── Public entry points ───────────────────────────────────────────────────────

def build_event_features(
    cfg: Config,
    year: int,
    round_num: int,
    *,
    history: pd.DataFrame,
    quali_history: pd.DataFrame | None = None,
    circuit_info: dict | None = None,
    entries: pd.DataFrame | None = None,
    use_actual_quali: bool = True,
    predicted_grid: pd.DataFrame | None = None,
    n_rounds: int = 24,
    sprint_weekend: bool = False,
) -> EventFeatures:
    """Build the feature matrix for a single event.

    Args:
        history: Multi-season race results used for form windows.
        quali_history: Multi-season qualifying results, for qualifying form.
        entries: Driver list. Defaults to the FastF1 entry list for the event.
        use_actual_quali: Use the real qualifying classification when it exists.
        predicted_grid: ``driver_code`` plus ``predicted_quali_pos``, used as the
            grid when qualifying has not happened yet.
        n_rounds: Rounds in the season, for the season-progress feature.
    """
    C.setup_fastf1_cache(cfg)
    circuit_info = circuit_info or repo.circuit_info(year, round_num, cfg)

    entry_list = entries if entries is not None else F1.get_entry_list(year, round_num)
    if entry_list is None or entry_list.empty:
        log.warning("No entry list available for %d R%d", year, round_num)
        return EventFeatures(meta=pd.DataFrame(), features=pd.DataFrame())

    entry_list = _with_constructor_ids(entry_list, history)

    practice = _cached_practice(cfg, year, round_num, sprint_weekend=sprint_weekend)
    quali = repo.quali_results(year, round_num, cfg) if use_actual_quali else pd.DataFrame()
    grid_source = GRID_ACTUAL if not quali.empty else GRID_FORM

    if quali.empty and predicted_grid is not None and not predicted_grid.empty:
        quali = _grid_from_prediction(predicted_grid)
        grid_source = GRID_PREDICTED

    form = form_for_event(
        history, entry_list, year, round_num,
        circuit_id=circuit_info.get("circuit_id", ""),
        quali=quali_history, form_cfg=cfg.form,
    )

    weather = _event_weather(cfg, year, round_num, circuit_info)

    meta, features = assemble_features(
        entries=entry_list, practice=practice.features if practice else None,
        quali=quali, form=form, weather=weather,
        round_num=round_num, n_rounds=n_rounds,
    )

    return EventFeatures(
        meta=meta, features=features, weather=weather, circuit=circuit_info,
        grid_source=grid_source,
        practice_session=practice.session_name if practice else None,
    )


def assemble_features(
    entries: pd.DataFrame,
    practice: pd.DataFrame | None,
    quali: pd.DataFrame | None,
    form: pd.DataFrame | None,
    weather: dict,
    round_num: int,
    n_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge every feature source into one aligned matrix.

    Pure and side-effect free: given the same inputs it always produces the same
    columns, in the order the models expect.
    """
    meta = entries.reindex(columns=META_COLUMNS).copy()
    for col in META_COLUMNS:
        meta[col] = meta[col].fillna("").astype(str)

    df = meta.copy()
    n_drivers = max(len(df), 1)

    if practice is not None and not practice.empty:
        # fp_compound and fp_session are labels, not numeric model inputs.
        labels = {"fp_compound", "fp_session"}
        keep = [c for c in practice.columns if c.startswith("fp_") and c not in labels]
        df = df.merge(practice[["driver_code", *keep]], on="driver_code", how="left")

    if quali is not None and not quali.empty:
        quali_cols = ["quali_pos", "quali_gap_to_pole_pct", "reached_q2", "reached_q3"]
        available = [c for c in quali_cols if c in quali.columns]
        key = "driver_id" if _joinable(df, quali, "driver_id") else "driver_code"
        df = df.merge(
            quali[[key, *available]].drop_duplicates(key), on=key, how="left"
        )
        if "quali_pos" in df.columns:
            df["grid_pos"] = pd.to_numeric(df["quali_pos"], errors="coerce")

    if form is not None and not form.empty:
        form_cols = [c for c in form.columns if c in FORM_DEFAULTS]
        df = df.merge(
            form[["driver_id", *form_cols]].drop_duplicates("driver_id"),
            on="driver_id", how="left",
        )

    df = _add_grid_features(df, n_drivers)
    df = _add_teammate_delta(df)

    df["weather_temp"] = weather.get("temperature", 22.0)
    df["weather_wind"] = weather.get("wind_speed", 3.0)
    df["weather_rain_prob"] = weather.get("rain_prob", 0.1)
    df["season_progress"] = round_num / max(n_rounds, 1)

    features = _finalise(df)
    return meta.reset_index(drop=True), features


def _add_grid_features(df: pd.DataFrame, n_drivers: int) -> pd.DataFrame:
    """Derive grid-position features, inferring the grid when qualifying is absent."""
    df = df.copy()
    # `df.get` on a missing column returns None, and pd.to_numeric(None) is a
    # bare NaN rather than a Series, so build the empty column explicitly.
    grid = (
        pd.to_numeric(df["grid_pos"], errors="coerce") if "grid_pos" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )

    if grid.isna().all():
        # No qualifying at all: rank the field by practice pace if we have it,
        # otherwise by recent form. Either beats leaving the model a constant.
        for column in ("fp_best_gap_pct", "form_avg_pos"):
            if column in df.columns and df[column].notna().any():
                grid = df[column].rank(method="first", na_option="bottom")
                break
        else:
            grid = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype="float64")

    df["grid_pos"] = grid
    df["grid_pos_pct"] = grid / max(n_drivers, 1)
    return df


def _add_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to the driver in the other car of the same team.

    Two separate columns, each locked to one source. Within-team comparisons
    strip out the car, which makes this one of the few features that isolates
    driver performance — but the qualifying model must use the practice-derived
    version, since the qualifying-derived one is the very thing it predicts.
    """
    df = df.copy()
    df["teammate_quali_delta_pct"] = _pairwise_delta(df, "quali_gap_to_pole_pct")
    df["teammate_fp_delta_pct"] = _pairwise_delta(df, "fp_best_gap_pct")
    return df


def _pairwise_delta(df: pd.DataFrame, column: str) -> pd.Series:
    """Each driver's value minus their teammate's, 0 where there is no pair."""
    if column not in df.columns or "constructor_id" not in df.columns:
        return pd.Series(0.0, index=df.index)

    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=df.index)

    team = df["constructor_id"].replace("", np.nan)
    grouped = values.groupby(team)
    # With two cars per team, "mean of the pair times two, minus me" is the
    # teammate's value; this also degrades sanely for one- or three-car groups.
    teammate = grouped.transform("mean") * 2 - values
    delta = values - teammate
    return delta.where(grouped.transform("count") > 1, 0.0).fillna(0.0)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Produce every schema column, in order, with neutral values for gaps."""
    neutrals = {}
    for group in (schema.RACE_FEATURES, schema.QUALI_FEATURES, schema.DNF_FEATURES):
        neutrals.update(schema.neutrals(group))

    out = pd.DataFrame(index=df.index)
    for column in schema.ALL_FEATURE_COLS:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            # Prefer the field's own median: a missing practice time is better
            # approximated by "what everyone else did today" than by a constant.
            median = series.median()
            fallback = neutrals[column] if pd.isna(median) else median
            out[column] = series.fillna(fallback)
        else:
            out[column] = neutrals[column]

    return out.astype("float64")


# ── Supporting pieces ─────────────────────────────────────────────────────────

def _joinable(left: pd.DataFrame, right: pd.DataFrame, key: str) -> bool:
    """True when ``key`` exists on both sides and actually overlaps."""
    if key not in left.columns or key not in right.columns:
        return False
    return bool(set(left[key].dropna()) & set(right[key].dropna()))


def _with_constructor_ids(entries: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable ``constructor_id`` to an entry list.

    FastF1 gives sponsor-laden team names that change year to year; the history
    carries the canonical Ergast id, so map through the driver where possible
    and fall back to a normalised team name.
    """
    entries = entries.copy()
    if "constructor_id" in entries.columns and entries["constructor_id"].astype(bool).all():
        return entries

    entries["constructor_id"] = ""
    if history is not None and not history.empty and "constructor_id" in history.columns:
        recent = history.sort_values(["year", "round"]).drop_duplicates("driver_id", keep="last")
        by_driver = dict(zip(recent["driver_id"], recent["constructor_id"], strict=False))
        entries["constructor_id"] = entries["driver_id"].map(by_driver).fillna("")

    missing = entries["constructor_id"] == ""
    if missing.any() and "team" in entries.columns:
        entries.loc[missing, "constructor_id"] = (
            entries.loc[missing, "team"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        )
    return entries


def _grid_from_prediction(predicted: pd.DataFrame) -> pd.DataFrame:
    """Turn a predicted qualifying order into a quali-shaped frame.

    The gap-to-pole column is synthesised from the predicted order: a real gap
    is unknowable before the session, and a plausible ramp keeps the feature on
    the same scale the model saw in training.
    """
    df = predicted.copy()
    if "predicted_quali_pos" not in df.columns:
        return pd.DataFrame()

    positions = pd.to_numeric(df["predicted_quali_pos"], errors="coerce")
    return pd.DataFrame({
        "driver_code": df.get("driver_code", ""),
        "driver_id": df.get("driver_id", ""),
        "quali_pos": positions,
        # Roughly 0.12% of lap time per grid slot, the typical modern spread.
        "quali_gap_to_pole_pct": (positions - 1).clip(lower=0) * 0.12,
        "reached_q2": (positions <= 15).astype(int),
        "reached_q3": (positions <= 10).astype(int),
    })


def _cached_practice(cfg: Config, year: int, round_num: int, sprint_weekend: bool = False):
    """Practice features, from the parquet cache or freshly extracted.

    During a live weekend more practice data lands after each session, so a
    cached entry for the current season expires; a completed weekend is final.
    """
    from dataclasses import dataclass as _dataclass

    @_dataclass(slots=True)
    class _Cached:
        session_name: str | None
        features: pd.DataFrame

    is_current = year >= datetime.now(tz=timezone.utc).year
    ttl = cfg.cache.live_ttl_s if is_current else None

    cached = C.load_features(cfg, year, round_num, "practice", ttl_s=ttl)
    if cached is not None and not cached.empty:
        # Parquet does not round-trip DataFrame.attrs, so the session name is
        # persisted as a column instead.
        session = (
            str(cached["fp_session"].iloc[0]) if "fp_session" in cached.columns else None
        )
        return _Cached(session_name=session, features=cached)

    practice = F1.get_practice_laps(year, round_num, sprint_weekend=sprint_weekend)
    if practice is None:
        return None

    features = extract_practice_features(practice.laps, cfg.training.min_laps_long_run)
    if features.empty:
        return None

    features = features.copy()
    features["fp_session"] = practice.session_name
    C.save_features(cfg, year, round_num, "practice", features)
    return _Cached(session_name=practice.session_name, features=features)


def _event_weather(cfg: Config, year: int, round_num: int, circuit_info: dict) -> dict:
    """Race-time weather, cached per event."""
    is_current = year >= datetime.now(tz=timezone.utc).year
    ttl = cfg.cache.live_ttl_s if is_current else None

    cached = C.load_weather(cfg, year, round_num, ttl_s=ttl)
    if cached is not None:
        return cached

    race_date = circuit_info.get("race_date", "")
    try:
        race_dt = (
            pd.to_datetime(race_date, utc=True, errors="coerce").to_pydatetime()
            if race_date else None
        )
    except (ValueError, TypeError):
        race_dt = None
    if race_dt is None or pd.isna(race_dt):
        race_dt = datetime.now(tz=timezone.utc)

    weather = WX.get_weather_for_race(
        circuit_info.get("lat", 0.0), circuit_info.get("lon", 0.0),
        race_dt, cache_dir=cfg.cache.http_dir,
    )
    C.save_weather(cfg, year, round_num, weather)
    return weather


# ── Training set ──────────────────────────────────────────────────────────────

TARGET_COLUMNS = ["race_pos", "quali_pos_actual", "dnf", "points", "grid_actual"]


def build_training_frame(
    cfg: Config,
    seasons: list[int],
    progress: Callable[[str], None] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Feature rows plus targets for every race in ``seasons``.

    The per-race work is cached to parquet, so re-training after adding one new
    race costs one race's worth of downloads rather than a whole season's.
    """
    def report(message: str) -> None:
        log.info(message)
        if progress:
            progress(message)

    frames: list[pd.DataFrame] = []
    for year in seasons:
        report(f"Loading {year} season data…")
        history = repo.history_for(year, seasons_back=2, cfg=cfg)
        quali_history = repo.quali_history_for(year, seasons_back=2, cfg=cfg)
        season = history[history["year"] == year] if not history.empty else pd.DataFrame()
        if season.empty:
            report(f"  {year}: no results available, skipping.")
            continue

        rounds = sorted(int(r) for r in season["round"].dropna().unique())
        n_rounds = max(rounds) if rounds else 24

        for round_num in rounds:
            key = C.race_key(year, round_num, "training")
            if not refresh:
                cached = C.load_frame(cfg, key)
                if cached is not None and not cached.empty:
                    frames.append(cached)
                    continue

            report(f"  Building {year} R{round_num:02d}…")
            row = _build_training_race(
                cfg, year, round_num, history, quali_history, season, n_rounds
            )
            if row is None or row.empty:
                continue
            C.save_frame(cfg, key, row)
            frames.append(row)

    if not frames:
        raise RuntimeError(
            "No training data could be assembled. Check network access, or run "
            "with --offline to use the bundled dataset."
        )

    train = pd.concat(frames, ignore_index=True)
    report(f"Training set: {len(train)} driver-races across {len(seasons)} season(s).")
    return train


def _build_training_race(
    cfg: Config,
    year: int,
    round_num: int,
    history: pd.DataFrame,
    quali_history: pd.DataFrame,
    season: pd.DataFrame,
    n_rounds: int,
) -> pd.DataFrame | None:
    """Feature + target rows for one historical race."""
    results = season[season["round"] == round_num]
    if results.empty:
        return None

    circuit = {
        "circuit_id": str(results["circuit_id"].iloc[0]),
        "race_date": str(results["race_date"].iloc[0]) if "race_date" in results else "",
        "lat": 0.0, "lon": 0.0,
    }
    # The results table has no coordinates; the circuit lookup does.
    info = repo.circuit_info(year, round_num, cfg)
    circuit["lat"], circuit["lon"] = info.get("lat", 0.0), info.get("lon", 0.0)
    if not circuit["race_date"]:
        circuit["race_date"] = info.get("race_date", "")

    entries = results[["driver_code", "driver_id", "driver_name", "team", "constructor_id"]]
    entries = entries.drop_duplicates("driver_id").reset_index(drop=True)

    try:
        event = build_event_features(
            cfg, year, round_num,
            history=history, quali_history=quali_history,
            circuit_info=circuit, entries=entries,
            use_actual_quali=True, n_rounds=n_rounds,
        )
    except Exception as exc:
        log.warning("Skipping %d R%d: %s", year, round_num, exc)
        return None

    if event.is_empty:
        return None

    frame = event.frame()
    targets = results.set_index("driver_id")
    frame["race_pos"] = frame["driver_id"].map(targets["race_pos"])
    frame["points"] = frame["driver_id"].map(targets["points"])
    frame["dnf"] = frame["driver_id"].map(targets["dnf"]).astype("boolean").fillna(False)
    frame["grid_actual"] = frame["driver_id"].map(targets["grid_pos"])

    quali = repo.quali_results(year, round_num, cfg)
    if not quali.empty:
        quali_map = quali.drop_duplicates("driver_id").set_index("driver_id")["quali_pos"]
        frame["quali_pos_actual"] = frame["driver_id"].map(quali_map)
    else:
        frame["quali_pos_actual"] = np.nan

    frame["year"] = year
    frame["round"] = round_num
    frame["race_id"] = f"{year}_{round_num:02d}"
    return frame.dropna(subset=["race_pos"])
