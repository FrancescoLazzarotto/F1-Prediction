"""
Orchestrates the full prediction pipeline:
  data download → feature building → model training → prediction → backtest.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1predict import cache as C
from f1predict.config import get_config
from f1predict.data import jolpica_source as JLP
from f1predict.data.schedule import available_sessions, resolve_event
from f1predict.features.builder import (
    QUALI_FEATURE_COLS,
    RACE_FEATURE_COLS,
    build_prediction_matrix,
    build_quali_prediction_matrix,
    build_training_row,
    cache_or_compute_fp,
)
from f1predict.features.form import build_form_matrix
from f1predict.features.qualifying import extract_quali_features
from f1predict.models.registry import get_quali_model, get_race_model

log = logging.getLogger(__name__)


class F1Pipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get_config()
        C.ensure_dirs(self.cfg)
        self._race_model = None
        self._quali_model = None

    # ── Model loading / training ──────────────────────────────────────────────

    def _load_or_train(self, force: bool = False) -> None:
        if not force and C.models_exist(self.cfg):
            self._race_model = C.load_model(self.cfg, "race_model")
            self._quali_model = C.load_model(self.cfg, "quali_model")
            log.info("Loaded cached models.")
            return
        self.train(self.cfg["training"]["seasons"])

    def train(
        self,
        seasons: list[int],
        progress_cb: Callable[[str], None] | None = None,
        force_retrain: bool = False,
    ) -> None:
        """
        Train race and quali models on the given seasons.
        Uses cached per-race features where available.
        """
        def _log(msg: str):
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        _log(f"Loading season results for {seasons}…")
        season_dfs = []
        for y in seasons:
            cached = C.load_season_results(self.cfg, y)
            if cached is not None and not force_retrain:
                season_dfs.append(cached)
                _log(f"  Season {y}: loaded from cache.")
            else:
                try:
                    df = JLP.get_season_results(y)
                    if not df.empty:
                        C.save_season_results(self.cfg, y, df)
                        season_dfs.append(df)
                        _log(f"  Season {y}: fetched {len(df)} rows.")
                except Exception as exc:
                    _log(f"  Season {y}: fetch failed ({exc}).")

        if not season_dfs:
            raise RuntimeError("No season data available for training.")

        all_results = pd.concat(season_dfs, ignore_index=True)

        # ── Collect training rows ─────────────────────────────────────────────
        race_rows: list[pd.DataFrame] = []
        for y in seasons:
            season_df = all_results[all_results["year"] == y]
            rounds = sorted(season_df["round"].unique())
            for r in rounds:
                _log(f"  Building features for {y} R{r}…")
                try:
                    circuit_info = JLP.get_circuit_info(y, r)
                except Exception:
                    circuit_info = {"circuit_id": "", "lat": 0.0, "lon": 0.0, "race_date": ""}

                row_df = build_training_row(
                    self.cfg, y, r, all_results, circuit_info
                )
                if row_df is not None and len(row_df) > 0:
                    race_rows.append(row_df)

        if not race_rows:
            raise RuntimeError("No training rows could be assembled.")

        train_df = pd.concat(race_rows, ignore_index=True)
        _log(f"Training dataset: {len(train_df)} rows, {len(seasons)} seasons.")

        # ── Race model ────────────────────────────────────────────────────────
        race_feat_cols = [c for c in RACE_FEATURE_COLS if c in train_df.columns]
        valid_race = train_df.dropna(subset=["race_pos"])
        if len(valid_race) < 20:
            _log("WARNING: very few valid race training rows.")

        X_race = valid_race[race_feat_cols].fillna(valid_race[race_feat_cols].median()).fillna(0)
        y_race = valid_race["race_pos"].astype(float)
        self._race_model = get_race_model(self.cfg)
        self._race_model.fit(X_race, y_race)
        C.save_model(self.cfg, "race_model", self._race_model)

        # ── Quali model ───────────────────────────────────────────────────────
        quali_feat_cols = [c for c in QUALI_FEATURE_COLS if c in train_df.columns]
        valid_quali = train_df.dropna(subset=["grid_pos"])
        if len(valid_quali) >= 20:
            X_quali = valid_quali[quali_feat_cols].fillna(valid_quali[quali_feat_cols].median()).fillna(0)
            y_quali = valid_quali["grid_pos"].astype(float)
            self._quali_model = get_quali_model(self.cfg)
            self._quali_model.fit(X_quali, y_quali)
            C.save_model(self.cfg, "quali_model", self._quali_model)
        else:
            _log("WARNING: not enough qualifying data to train quali model.")

        _log("Training complete.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_race(
        self,
        year: int,
        round_num: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Predict race outcome for a given round.
        Auto-detects whether actual qualifying data is available.

        Returns:
            {
                "race": DataFrame with pos/driver/team/predicted_pos/p_win/p_podium/p_top5,
                "quali": DataFrame with predicted quali order (if FP-based),
                "event": dict with circuit/year/round/race_date,
                "data_mode": "actual_quali" | "predicted_quali" | "form_only",
            }
        """
        def _log(msg):
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        self._load_or_train()

        _log(f"Fetching circuit info for {year} R{round_num}…")
        try:
            circuit_info = JLP.get_circuit_info(year, round_num)
        except Exception as exc:
            _log(f"WARNING: circuit info unavailable ({exc}), using defaults.")
            circuit_info = {"circuit_id": "", "lat": 0.0, "lon": 0.0, "race_date": "", "name": "?"}

        _log("Loading historical season results for form features…")
        season_results = self._load_season_results_for(year)

        # Determine what data is available
        sessions = available_sessions(year, round_num)
        use_actual_quali = sessions.get("Q", False)
        data_mode = "actual_quali" if use_actual_quali else "form_only"

        # Predict qualifying from FP if available but quali hasn't happened
        quali_prediction_df = pd.DataFrame()
        if not use_actual_quali and sessions.get("FP2", False) or sessions.get("FP1", False):
            _log("Qualifying not yet done — predicting from FP data…")
            quali_prediction_df = self._predict_quali_internal(
                year, round_num, season_results, _log
            )
            data_mode = "predicted_quali"

        _log("Building race feature matrix…")
        meta, X_race = build_prediction_matrix(
            self.cfg, year, round_num, season_results, circuit_info,
            use_actual_quali=use_actual_quali,
        )

        if meta.empty or X_race.empty:
            raise RuntimeError(f"Could not build feature matrix for {year} R{round_num}.")

        _log("Running race model…")
        race_pred = self._race_model.predict_with_proba(X_race)
        result = pd.concat([meta.reset_index(drop=True), race_pred.reset_index(drop=True)], axis=1)
        result = result.sort_values("predicted_pos").reset_index(drop=True)
        result.index = result.index + 1

        _log("Done.")
        return {
            "race": result,
            "quali": quali_prediction_df,
            "event": circuit_info | {"year": year, "round": round_num},
            "data_mode": data_mode,
        }

    def predict_quali(
        self,
        year: int,
        round_num: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Predict qualifying order from FP features."""
        def _log(msg):
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        self._load_or_train()
        season_results = self._load_season_results_for(year)
        return self._predict_quali_internal(year, round_num, season_results, _log)

    def _predict_quali_internal(
        self, year: int, round_num: int, season_results: pd.DataFrame,
        log_fn: Callable[[str], None],
    ) -> pd.DataFrame:
        if self._quali_model is None:
            log_fn("Quali model not available.")
            return pd.DataFrame()

        meta, X_quali = build_quali_prediction_matrix(
            self.cfg, year, round_num, season_results
        )
        if meta.empty or X_quali.empty:
            return pd.DataFrame()

        raw = self._quali_model.predict(X_quali)
        ranking = raw.rank(method="first").astype(int)
        result = meta.copy().reset_index(drop=True)
        result["predicted_quali_pos"] = ranking.values
        result["quali_score"] = raw.values
        result = result.sort_values("predicted_quali_pos").reset_index(drop=True)
        result.index = result.index + 1
        return result

    # ── Backtest ──────────────────────────────────────────────────────────────

    def backtest(self, year: int, round_num: int) -> dict:
        """
        Predict for a past race and compare with actual results.
        Returns metrics: spearman_rho, top3_accuracy, mae_positions, prediction_df.
        """
        self._load_or_train()

        try:
            circuit_info = JLP.get_circuit_info(year, round_num)
        except Exception:
            circuit_info = {"circuit_id": "", "lat": 0.0, "lon": 0.0, "race_date": "", "name": "?"}

        season_results = self._load_season_results_for(year)

        # For backtesting: always use actual qualifying (it's in the past)
        meta, X_race = build_prediction_matrix(
            self.cfg, year, round_num, season_results, circuit_info,
            use_actual_quali=True,
        )
        if meta.empty or X_race.empty:
            raise RuntimeError(f"Cannot backtest {year} R{round_num}: no feature data.")

        race_pred = self._race_model.predict_with_proba(X_race)
        predicted = pd.concat([meta.reset_index(drop=True), race_pred.reset_index(drop=True)], axis=1)

        # Actual results
        actual = JLP.get_race_results(year, round_num)
        if actual.empty:
            raise RuntimeError(f"No actual results for {year} R{round_num}.")

        merged = predicted.merge(
            actual[["driver_code", "race_pos"]].rename(columns={"race_pos": "actual_pos"}),
            on="driver_code", how="inner",
        )

        if len(merged) < 5:
            raise RuntimeError("Not enough drivers matched for backtest.")

        rho, _ = spearmanr(merged["predicted_pos"], merged["actual_pos"])
        actual_top3 = set(merged.nsmallest(3, "actual_pos")["driver_code"])
        pred_top3 = set(merged.nsmallest(3, "predicted_pos")["driver_code"])
        top3_acc = len(actual_top3 & pred_top3) / 3.0
        mae = float((merged["predicted_pos"] - merged["actual_pos"]).abs().mean())

        merged = merged.sort_values("actual_pos").reset_index(drop=True)
        merged.index = merged.index + 1

        return {
            "spearman_rho": round(float(rho), 3),
            "top3_accuracy": round(top3_acc, 3),
            "mae_positions": round(mae, 2),
            "prediction_df": merged,
            "event": circuit_info | {"year": year, "round": round_num},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_season_results_for(self, year: int) -> pd.DataFrame:
        """Load multi-year results for form computation (year and year-1)."""
        dfs = []
        for y in (year - 1, year):
            cached = C.load_season_results(self.cfg, y)
            if cached is not None:
                dfs.append(cached)
            else:
                try:
                    df = JLP.get_season_results(y)
                    if not df.empty:
                        C.save_season_results(self.cfg, y, df)
                        dfs.append(df)
                except Exception as exc:
                    log.warning("Could not load season %d results: %s", y, exc)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
