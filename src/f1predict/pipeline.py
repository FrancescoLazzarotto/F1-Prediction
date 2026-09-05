"""Orchestration: train, predict, backtest, explain, project the championship.

This is the only module the CLI and the web app talk to. Everything below it
(data sources, features, models, simulation) is reachable but not required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1predict import cache as C
from f1predict.config import Config, get_config
from f1predict.data import repository as repo
from f1predict.data.schedule import Event, available_sessions, resolve_event, season_events
from f1predict.features import schema
from f1predict.features.builder import (
    GRID_ACTUAL,
    GRID_FORM,
    GRID_PREDICTED,
    EventFeatures,
    build_event_features,
    build_training_frame,
)
from f1predict.models.metrics import RankingMetrics, aggregate, brier_score, evaluate_ranking
from f1predict.models.predictor import DnfPredictor, RankingPredictor
from f1predict.models.registry import (
    load_all,
    new_dnf_model,
    new_quali_model,
    new_race_model,
    save_all,
)
from f1predict.simulation import RaceSimulation, simulate_championship, simulate_race, summarise

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass(slots=True)
class RacePrediction:
    """A finished race forecast, ready for rendering or export."""

    table: pd.DataFrame
    quali_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    event: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)
    grid_source: str = GRID_FORM
    practice_session: str | None = None
    simulation: RaceSimulation | None = None
    features: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def confidence(self) -> str:
        """How much real session data went into this prediction."""
        return {
            GRID_ACTUAL: "high",
            GRID_PREDICTED: "medium",
            GRID_FORM: "low",
        }.get(self.grid_source, "low")

    @property
    def podium(self) -> list[str]:
        return self.table.head(3)["driver_name"].tolist()

    def to_records(self) -> list[dict]:
        return self.table.to_dict(orient="records")


@dataclass(slots=True)
class BacktestResult:
    """A prediction scored against what actually happened."""

    metrics: RankingMetrics
    table: pd.DataFrame
    event: dict = field(default_factory=dict)
    brier_win: float = float("nan")
    brier_podium: float = float("nan")
    #: True when this race was part of the model's training set, which flatters
    #: the score. The cross-validated figures in `f1predict info` are the
    #: honest out-of-sample measure.
    in_sample: bool = False


class F1Pipeline:
    """Loads or trains models once, then serves predictions from them."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        C.ensure_dirs(self.cfg)
        self._race: RankingPredictor | None = None
        self._quali: RankingPredictor | None = None
        self._dnf: DnfPredictor | None = None
        self._loaded = False

    # ── Models ────────────────────────────────────────────────────────────────

    def ensure_models(self, progress: Progress | None = None) -> None:
        """Load cached models, training them first if none are usable."""
        if self._loaded and self._race is not None:
            return

        self._race, self._quali, self._dnf = load_all(self.cfg)
        self._loaded = True
        if self._race is not None and self._quali is not None:
            log.info("Loaded cached models.")
            return

        _report(progress, "No usable cached models — training now.")
        self.train(self.cfg.training.seasons, progress=progress)

    @property
    def models_ready(self) -> bool:
        if not self._loaded:
            self._race, self._quali, self._dnf = load_all(self.cfg)
            self._loaded = True
        return self._race is not None and self._quali is not None

    def train(
        self,
        seasons: list[int] | None = None,
        progress: Progress | None = None,
        refresh: bool = False,
    ) -> dict[str, object]:
        """Train the race, qualifying and retirement models.

        Returns the three training reports, keyed by model name.
        """
        seasons = sorted(seasons or self.cfg.training.seasons)
        _report(progress, f"Assembling training data for {seasons}…")

        train_df = build_training_frame(self.cfg, seasons, progress=progress, refresh=refresh)
        weights = _recency_weights(train_df, self.cfg.training.recency_half_life_races)

        # ── Race model ────────────────────────────────────────────────────────
        # Fit on classified finishers only. A retirement says nothing about pace,
        # and training on it teaches the model that pole sitters finish 19th.
        finishers = train_df[~train_df["dnf"].astype("boolean").fillna(False)]
        _report(progress, f"Training race model on {len(finishers)} classified finishes…")
        race_model = new_race_model(self.cfg, seasons)
        race_report = race_model.fit(
            finishers[schema.RACE_FEATURE_COLS],
            finishers["race_pos"],
            groups=finishers["race_id"],
            sample_weight=weights[finishers.index],
            cv_folds=self.cfg.training.cv_folds,
            min_samples_for_cv=self.cfg.training.min_samples_for_cv,
        )
        _report(progress, f"  Race model: {race_report.summary()}")

        # ── Qualifying model ──────────────────────────────────────────────────
        quali_rows = train_df.dropna(subset=["quali_pos_actual"])
        quali_report = None
        quali_model = None
        if len(quali_rows) >= 30:
            _report(progress, f"Training qualifying model on {len(quali_rows)} sessions…")
            quali_model = new_quali_model(self.cfg, seasons)
            quali_report = quali_model.fit(
                quali_rows[schema.QUALI_FEATURE_COLS],
                quali_rows["quali_pos_actual"],
                groups=quali_rows["race_id"],
                sample_weight=weights[quali_rows.index],
                cv_folds=self.cfg.training.cv_folds,
                min_samples_for_cv=self.cfg.training.min_samples_for_cv,
            )
            _report(progress, f"  Qualifying model: {quali_report.summary()}")
        else:
            _report(progress, "  Not enough qualifying data to train that model.")

        # ── Retirement model ──────────────────────────────────────────────────
        _report(progress, "Training retirement model…")
        dnf_model = new_dnf_model(self.cfg, seasons)
        dnf_report = dnf_model.fit(
            train_df[schema.DNF_FEATURE_COLS], train_df["dnf"],
            groups=train_df["race_id"], cv_folds=self.cfg.training.cv_folds,
        )
        _report(progress, f"  Retirement model: {dnf_report.summary()}")

        save_all(self.cfg, race_model, quali_model, dnf_model)
        self._race, self._quali, self._dnf = race_model, quali_model, dnf_model
        self._loaded = True

        _report(progress, "Training complete.")
        return {"race": race_report, "quali": quali_report, "dnf": dnf_report}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_race(
        self,
        year: int,
        round_num: int,
        progress: Progress | None = None,
        force_grid_source: str | None = None,
    ) -> RacePrediction:
        """Forecast a race, using whatever session data exists at this moment.

        With qualifying done, the real grid is used. Before that, qualifying is
        predicted from practice and *that* order becomes the grid — the step the
        two-stage design exists for.
        """
        self.ensure_models(progress)

        event = _event_for(year, round_num)
        sessions = available_sessions(year, round_num)
        has_quali = sessions.get("Q", False) or sessions.get("SQ", False)
        use_actual = has_quali if force_grid_source is None else force_grid_source == GRID_ACTUAL

        _report(progress, f"Loading history for {year}…")
        history = repo.history_for(year, seasons_back=2, cfg=self.cfg)
        quali_history = repo.quali_history_for(year, seasons_back=2, cfg=self.cfg)
        circuit = repo.circuit_info(year, round_num, self.cfg)
        n_rounds = len(season_events(year)) or 24

        # Stage 1: qualifying from practice, whenever the real grid is unknown.
        quali_table = pd.DataFrame()
        predicted_grid = None
        if not use_actual:
            _report(progress, "Qualifying has not run — predicting the grid from practice…")
            quali_table = self._predict_quali(
                year, round_num, history, quali_history, circuit, n_rounds, event
            )
            if not quali_table.empty:
                predicted_grid = quali_table[["driver_code", "driver_id", "predicted_quali_pos"]]

        # Stage 2: the race itself.
        _report(progress, "Building race features…")
        features = build_event_features(
            self.cfg, year, round_num,
            history=history, quali_history=quali_history, circuit_info=circuit,
            use_actual_quali=use_actual, predicted_grid=predicted_grid,
            n_rounds=n_rounds, sprint_weekend=event.is_sprint if event else False,
        )
        if features.is_empty:
            raise RuntimeError(
                f"No data available for {year} round {round_num}. The entry list is "
                "empty — the weekend may not have started yet."
            )

        _report(progress, "Running the race model…")
        table, simulation = self._simulate(features)

        _report(progress, "Done.")
        return RacePrediction(
            table=table, quali_table=quali_table,
            event=_event_dict(event, circuit, year, round_num),
            weather=features.weather, grid_source=features.grid_source,
            practice_session=features.practice_session, simulation=simulation,
            features=features.features.set_axis(features.meta["driver_code"], axis=0),
        )

    def _simulate(self, features: EventFeatures) -> tuple[pd.DataFrame, RaceSimulation]:
        """Score the field, then turn scores into outcome probabilities."""
        scores = self._race.predict(features.features)

        dnf_probabilities = (
            self._dnf.predict_proba(features.features)
            if self._dnf is not None
            else np.full(len(scores), self.cfg.simulation.base_dnf_rate)
        )

        simulation = simulate_race(
            scores=scores,
            driver_codes=features.meta["driver_code"].tolist(),
            dnf_probabilities=dnf_probabilities,
            grid_positions=features.features["grid_pos"].to_numpy(),
            rain_probability=float(features.weather.get("rain_prob", 0.0)),
            cfg=self.cfg.simulation,
        )

        table = summarise(features.meta, scores, simulation)
        table["grid_pos"] = table["driver_code"].map(
            dict(zip(features.meta["driver_code"], features.features["grid_pos"], strict=True))
        ).astype(int)
        table["grid_delta"] = table["grid_pos"] - table["predicted_pos"]
        return table, simulation

    def predict_quali(
        self, year: int, round_num: int, progress: Progress | None = None
    ) -> pd.DataFrame:
        """Forecast the qualifying order from practice data."""
        self.ensure_models(progress)
        event = _event_for(year, round_num)
        history = repo.history_for(year, seasons_back=2, cfg=self.cfg)
        quali_history = repo.quali_history_for(year, seasons_back=2, cfg=self.cfg)
        circuit = repo.circuit_info(year, round_num, self.cfg)
        n_rounds = len(season_events(year)) or 24
        return self._predict_quali(
            year, round_num, history, quali_history, circuit, n_rounds, event
        )

    def _predict_quali(
        self, year: int, round_num: int, history: pd.DataFrame,
        quali_history: pd.DataFrame, circuit: dict, n_rounds: int, event: Event | None,
    ) -> pd.DataFrame:
        if self._quali is None:
            log.warning("No qualifying model available.")
            return pd.DataFrame()

        features = build_event_features(
            self.cfg, year, round_num,
            history=history, quali_history=quali_history, circuit_info=circuit,
            use_actual_quali=False, n_rounds=n_rounds,
            sprint_weekend=event.is_sprint if event else False,
        )
        if features.is_empty:
            return pd.DataFrame()

        scores = self._quali.predict(features.features)
        out = features.meta.reset_index(drop=True).copy()
        out["quali_score"] = scores
        out["predicted_quali_pos"] = pd.Series(scores).rank(method="first").astype(int)
        # Turn the score spread into an approximate lap-time gap, purely for
        # display: it is a monotone transform of the score, not a real time.
        spread = float(np.ptp(scores)) or 1.0
        out["approx_gap_s"] = (scores - scores.min()) / spread * 1.6
        for column in ("fp_best_gap_pct", "fp_pace_gap_pct", "fp_rank_pct"):
            if column in features.features.columns:
                out[column] = features.features[column].to_numpy()
        return out.sort_values("predicted_quali_pos").reset_index(drop=True)

    # ── Explanation ───────────────────────────────────────────────────────────

    def explain(self, prediction: RacePrediction, driver_code: str, top_n: int = 5) -> list[dict]:
        """Why the model rates one driver the way it does.

        Compares the driver's features against the field median and weights each
        gap by the model's permutation importance, which gives a readable
        "what stands out about this entry" list rather than a true attribution.
        """
        if self._race is None or prediction.features.empty:
            return []

        features = prediction.features
        if driver_code not in features.index:
            raise KeyError(f"{driver_code} is not in this prediction.")

        importance = getattr(self._race.report, "feature_importance", {}) or {}
        medians = features.median()
        std = features.std().replace(0, np.nan)
        row = features.loc[driver_code]

        contributions = []
        for name in self._race.feature_names:
            if name not in features.columns:
                continue
            value = float(row[name])
            deviation = (value - medians[name]) / std[name] if std[name] == std[name] else 0.0
            weight = importance.get(name, 0.0)
            if not weight or deviation != deviation:
                continue

            meta = schema.by_name(name)
            better = (deviation < 0) if (meta is None or meta.lower_is_better) else (deviation > 0)
            contributions.append({
                "feature": name,
                "label": meta.label if meta else name,
                "value": value,
                "field_median": float(medians[name]),
                "z_score": float(deviation),
                "importance": float(weight),
                "impact": float(abs(deviation) * weight),
                "direction": "strength" if better else "weakness",
            })

        contributions.sort(key=lambda c: -c["impact"])
        return contributions[:top_n]

    # ── Backtesting ───────────────────────────────────────────────────────────

    def backtest(self, year: int, round_num: int, progress: Progress | None = None) -> BacktestResult:
        """Score a prediction for a past race against the real classification."""
        self.ensure_models(progress)

        actual = repo.race_results(year, round_num, self.cfg)
        if actual.empty:
            raise RuntimeError(f"No results exist for {year} round {round_num}.")

        prediction = self.predict_race(
            year, round_num, progress=progress, force_grid_source=GRID_ACTUAL
        )

        merged = prediction.table.merge(
            actual[["driver_id", "race_pos", "status", "dnf"]].rename(
                columns={"race_pos": "actual_pos", "dnf": "actual_dnf"}
            ),
            on="driver_id", how="inner",
        )
        if len(merged) < 3:
            raise RuntimeError(
                f"Only {len(merged)} drivers matched between prediction and result."
            )

        metrics = evaluate_ranking(merged["predicted_pos"], merged["actual_pos"])
        merged["error"] = merged["predicted_pos"] - merged["actual_pos"]

        trained_seasons = getattr(getattr(self._race, "metadata", None), "seasons", []) or []
        return BacktestResult(
            metrics=metrics,
            table=merged.sort_values("actual_pos").reset_index(drop=True),
            event=prediction.event,
            in_sample=year in trained_seasons,
            brier_win=brier_score(merged["p_win"], (merged["actual_pos"] == 1).astype(float)),
            brier_podium=brier_score(
                merged["p_podium"], (merged["actual_pos"] <= 3).astype(float)
            ),
        )

    def backtest_season(
        self,
        year: int,
        rounds: list[int] | None = None,
        progress: Progress | None = None,
    ) -> tuple[RankingMetrics, pd.DataFrame]:
        """Backtest every completed round of a season and aggregate the scores."""
        self.ensure_models(progress)

        results = repo.season_results(year, self.cfg)
        if results.empty:
            raise RuntimeError(f"No results available for {year}.")

        candidates = rounds or sorted(int(r) for r in results["round"].dropna().unique())
        per_race, rows = [], []

        for round_num in candidates:
            try:
                outcome = self.backtest(year, round_num)
            except Exception as exc:
                _report(progress, f"  R{round_num:02d}: skipped ({exc})")
                continue

            per_race.append(outcome.metrics)
            rows.append({
                "round": round_num,
                "race": outcome.event.get("name", ""),
                "spearman": outcome.metrics.spearman,
                "mae": outcome.metrics.mae,
                "winner_hit": outcome.metrics.top1,
                "podium_hit": outcome.metrics.top3,
                "within_3": outcome.metrics.within_3,
                "brier_win": outcome.brier_win,
            })
            _report(progress, f"  R{round_num:02d}: {outcome.metrics.summary()}")

        if not per_race:
            raise RuntimeError(f"No rounds of {year} could be backtested.")

        return aggregate(per_race), pd.DataFrame(rows)

    # ── Championship ──────────────────────────────────────────────────────────

    def championship_outlook(
        self,
        year: int,
        progress: Progress | None = None,
        constructors: bool = False,
    ):
        """Simulate the rest of the season to get title probabilities."""
        _report(progress, f"Loading {year} standings…")
        standings = (
            repo.constructor_standings(year, cfg=self.cfg) if constructors
            else repo.driver_standings(year, cfg=self.cfg)
        )
        if standings.empty:
            raise RuntimeError(f"No standings available for {year}.")

        events = season_events(year)
        now = datetime.now(tz=timezone.utc)
        remaining = [e for e in events if e.race_date is not None and e.race_date > now]
        races_remaining = len(remaining)
        sprints_remaining = sum(1 for e in remaining if e.is_sprint)

        _report(progress, f"{races_remaining} rounds remain ({sprints_remaining} with a sprint).")

        results = repo.season_results(year, self.cfg)
        pace, dnf_rates = self._season_pace(results, standings, constructors)
        active = self._active_entrants(results, constructors)

        outlook = simulate_championship(
            standings=standings, pace=pace, races_remaining=races_remaining,
            sprints_remaining=sprints_remaining, dnf_rates=dnf_rates,
            active_keys=active, cfg=self.cfg.simulation,
            label_column="team" if constructors else "driver_name",
            key_column="constructor_id" if constructors else "driver_id",
        )
        _report(progress, "Done.")
        return outlook

    def _active_entrants(self, results: pd.DataFrame, constructors: bool) -> set[str] | None:
        """Who lined up for the most recent round.

        Reserve drivers and mid-season departures still appear in the standings
        with the points they scored, but must not keep scoring in the
        simulation of races they will not start.
        """
        key = "constructor_id" if constructors else "driver_id"
        if results.empty or key not in results.columns:
            return None

        last_round = results["round"].max()
        entrants = results.loc[results["round"] == last_round, key]
        return set(entrants.dropna().astype(str)) or None

    def _season_pace(
        self, results: pd.DataFrame, standings: pd.DataFrame, constructors: bool
    ) -> tuple[pd.Series | None, pd.Series | None]:
        """Recent average finishing position and retirement rate, per entrant.

        Recent form is a better guide to the rest of the season than the
        cumulative points table, which still reflects who was quick in March.
        """
        if results.empty:
            return None, None

        key = "constructor_id" if constructors else "driver_id"
        if key not in results.columns:
            return None, None

        window = self.cfg.form.recent_races * (2 if constructors else 1)
        recent_rounds = sorted(results["round"].dropna().unique())[-window:]
        recent = results[results["round"].isin(recent_rounds)]
        if recent.empty:
            recent = results

        pace = recent.groupby(key)["race_pos"].mean()
        dnf = recent.groupby(key)["dnf"].mean() if "dnf" in recent.columns else None

        # Entrants in the standings but not in recent results (a mid-season
        # replacement) fall back to the field median rather than dropping out.
        return pace, dnf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _report(progress: Progress | None, message: str) -> None:
    log.info(message)
    if progress:
        progress(message)


def _event_for(year: int, round_num: int) -> Event | None:
    try:
        return resolve_event(year=year, round_num=round_num)
    except (LookupError, ValueError):
        return None


def _event_dict(event: Event | None, circuit: dict, year: int, round_num: int) -> dict:
    out = {
        "year": year, "round": round_num,
        "name": circuit.get("race_name") or (event.name if event else ""),
        "circuit": circuit.get("name", ""),
        "circuit_id": circuit.get("circuit_id", ""),
        "locality": circuit.get("locality", ""),
        "country": circuit.get("country", ""),
        "race_date": circuit.get("race_date", ""),
        "lat": circuit.get("lat", 0.0), "lon": circuit.get("lon", 0.0),
    }
    if event is not None:
        out["name"] = out["name"] or event.name
        out["format"] = event.event_format
        out["is_sprint"] = event.is_sprint
        if event.race_date:
            out["race_date"] = event.race_date.isoformat()
    return out


def _recency_weights(train_df: pd.DataFrame, half_life_races: float) -> pd.Series:
    """Exponential decay so recent seasons dominate the fit.

    Regulations, drivers and cars all change; a 2022 race is evidence about
    2026, but weaker evidence than last month's.
    """
    if train_df.empty or half_life_races <= 0:
        return pd.Series(1.0, index=train_df.index)

    order = train_df[["year", "round"]].drop_duplicates().sort_values(["year", "round"])
    order["age"] = np.arange(len(order))[::-1]  # 0 = most recent race
    ages = train_df.merge(order, on=["year", "round"], how="left")["age"].to_numpy()

    weights = np.power(0.5, ages / half_life_races)
    return pd.Series(weights, index=train_df.index)
