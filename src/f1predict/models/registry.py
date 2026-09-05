"""Construct, persist and validate models against the current feature contract.

A joblib file on disk outlives the code that wrote it. Every bundle therefore
carries the feature signature it was trained under, and loading refuses a
mismatch instead of quietly predicting from misaligned columns.
"""

from __future__ import annotations

import logging

from f1predict import cache as C
from f1predict.config import Config
from f1predict.features import schema
from f1predict.models.base import ModelMetadata
from f1predict.models.predictor import DnfPredictor, RankingPredictor

log = logging.getLogger(__name__)


def _metadata(cfg: Config, kind: str, seasons: list[int] | None = None) -> ModelMetadata:
    from f1predict import __version__

    return ModelMetadata(
        signature=cfg.feature_signature,
        schema_version=schema.SCHEMA_VERSION,
        kind=kind,
        seasons=list(seasons or cfg.training.seasons),
        package_version=__version__,
    )


def new_race_model(cfg: Config, seasons: list[int] | None = None) -> RankingPredictor:
    model_cfg = cfg.models.race
    return RankingPredictor(
        features=schema.RACE_FEATURES, kind=model_cfg.kind, params=model_cfg.params,
        metadata=_metadata(cfg, model_cfg.kind, seasons),
    )


def new_quali_model(cfg: Config, seasons: list[int] | None = None) -> RankingPredictor:
    model_cfg = cfg.models.quali
    return RankingPredictor(
        features=schema.QUALI_FEATURES, kind=model_cfg.kind, params=model_cfg.params,
        metadata=_metadata(cfg, model_cfg.kind, seasons),
    )


def new_dnf_model(cfg: Config, seasons: list[int] | None = None) -> DnfPredictor:
    model_cfg = cfg.models.dnf
    return DnfPredictor(
        features=schema.DNF_FEATURES, kind=model_cfg.kind, params=model_cfg.params,
        metadata=_metadata(cfg, model_cfg.kind, seasons),
    )


def save_all(cfg: Config, race, quali, dnf) -> None:
    """Persist the three models that make up a full prediction."""
    for name, model in ((C.RACE_MODEL, race), (C.QUALI_MODEL, quali), (C.DNF_MODEL, dnf)):
        if model is not None:
            C.save_model(cfg, name, model)


def load_all(cfg: Config) -> tuple[RankingPredictor | None, RankingPredictor | None,
                                   DnfPredictor | None]:
    """Load the saved models, discarding any whose feature contract is stale."""
    race = _load_checked(cfg, C.RACE_MODEL)
    quali = _load_checked(cfg, C.QUALI_MODEL)
    dnf = _load_checked(cfg, C.DNF_MODEL)
    return race, quali, dnf


def _load_checked(cfg: Config, name: str):
    model = C.load_model(cfg, name)
    if model is None:
        return None

    metadata = getattr(model, "metadata", None)
    if metadata is None or not metadata.is_compatible_with(cfg.feature_signature):
        found = getattr(metadata, "signature", "unknown")
        log.warning(
            "Cached %s was built for feature signature %s but this config needs %s; "
            "it will be retrained.", name, found, cfg.feature_signature,
        )
        return None

    if not getattr(model, "is_fitted", False) and name != C.DNF_MODEL:
        log.warning("Cached %s is not fitted; it will be retrained.", name)
        return None

    return model


def models_ready(cfg: Config) -> bool:
    """True when usable race and qualifying models exist on disk."""
    race, quali, _ = load_all(cfg)
    return race is not None and quali is not None


def describe(cfg: Config) -> dict:
    """Model-card style summary of what is currently on disk."""
    race, quali, dnf = load_all(cfg)
    out: dict = {
        "feature_signature": cfg.feature_signature,
        "schema_version": schema.SCHEMA_VERSION,
        "models": {},
    }
    for name, model in ((C.RACE_MODEL, race), (C.QUALI_MODEL, quali), (C.DNF_MODEL, dnf)):
        if model is None:
            out["models"][name] = {"status": "missing or stale"}
            continue
        report = getattr(model, "report", None)
        metadata = getattr(model, "metadata", None)
        out["models"][name] = {
            "status": "ready",
            "kind": getattr(metadata, "kind", "?"),
            "trained_at": getattr(metadata, "trained_at", "?"),
            "seasons": getattr(metadata, "seasons", []),
            "n_features": len(getattr(model, "feature_names", [])),
            "n_samples": getattr(report, "n_samples", 0),
            "cv_mae": getattr(report, "cv_mae", float("nan")),
            "cv_spearman": getattr(report, "cv_spearman", float("nan")),
            "cv_top3": getattr(report, "cv_top3", float("nan")),
            "top_features": getattr(report, "top_features", lambda n=6: [])(6),
            "notes": getattr(report, "notes", []),
        }
    return out
