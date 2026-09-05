"""Typed configuration: YAML defaults, environment overrides, dotted access.

The config is a frozen dataclass tree rather than a bare dict so that typos
fail loudly at load time instead of silently producing ``None`` deep inside the
feature builder. :func:`get_config` memoises a single instance per process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml
from dotenv import load_dotenv

#: Package directory, and the repo root when running from a source checkout.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

T = TypeVar("T")


def _default_cache_root() -> Path:
    """Where caches live: repo-local for a checkout, user cache dir otherwise."""
    if (REPO_ROOT / "pyproject.toml").exists():
        return REPO_ROOT / ".cache"
    return Path.home() / ".cache" / "f1predict"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Filesystem locations for every cached artefact."""

    root: str = ""
    fastf1_dir: str = ""
    features_dir: str = ""
    models_dir: str = ""
    http_dir: str = ""
    #: Seconds a cached HTTP response for the current season stays fresh.
    live_ttl_s: int = 6 * 3600

    def resolved(self) -> CacheConfig:
        """Return a copy with every path absolute and mutually consistent."""
        root = Path(self.root) if self.root else _default_cache_root()
        if not root.is_absolute():
            root = REPO_ROOT / root

        def sub(value: str, name: str) -> str:
            p = Path(value) if value else root / name
            return str(p if p.is_absolute() else REPO_ROOT / p)

        return CacheConfig(
            root=str(root),
            fastf1_dir=sub(self.fastf1_dir, "fastf1"),
            features_dir=sub(self.features_dir, "features"),
            models_dir=sub(self.models_dir, "models"),
            http_dir=sub(self.http_dir, "http"),
            live_ttl_s=self.live_ttl_s,
        )

    def paths(self) -> list[Path]:
        return [
            Path(self.fastf1_dir), Path(self.features_dir),
            Path(self.models_dir), Path(self.http_dir),
        ]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seasons: list[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    #: Minimum consecutive green-flag laps for a stint to count as a long run.
    min_laps_long_run: int = 5
    #: Number of GroupKFold splits (grouped by race) for cross-validation.
    cv_folds: int = 5
    #: Skip cross-validation when there are fewer than this many samples.
    min_samples_for_cv: int = 60
    #: Half-life in races for sample weighting; recent races count for more.
    recency_half_life_races: float = 25.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    kind: str = "hist_gradient_boosting"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    race: ModelConfig = field(default_factory=ModelConfig)
    quali: ModelConfig = field(default_factory=ModelConfig)
    dnf: ModelConfig = field(
        default_factory=lambda: ModelConfig(kind="gradient_boosting_classifier")
    )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_simulations: int = 20_000
    #: Baseline spread (in model-score units) injected per driver per simulation.
    position_noise_std: float = 2.6
    #: Extra spread applied linearly from the front of the grid to the back.
    backmarker_noise_scale: float = 0.9
    #: Spread multiplier at 100% rain probability.
    wet_noise_multiplier: float = 1.8
    #: Fallback per-driver retirement probability when the DNF model is absent.
    base_dnf_rate: float = 0.09
    seed: int = 42


@dataclass(frozen=True, slots=True)
class FormConfig:
    recent_races: int = 5
    #: Longer window used for reliability / DNF-rate estimates.
    reliability_races: int = 12
    #: Window used for the "is this driver trending up" feature.
    trend_races: int = 3


@dataclass(frozen=True, slots=True)
class DataConfig:
    #: Max Jolpica requests per second (their published cap is 4/s, 500/hour).
    jolpica_rate_limit: float = 3.0
    request_timeout_s: int = 20
    max_retries: int = 4
    #: Fall back to the bundled Ergast CSV dump when the network is unavailable.
    offline_fallback: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    form: FormConfig = field(default_factory=FormConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def feature_signature(self) -> str:
        """Token identifying the feature/model contract of this config.

        Cached models embed it, so a config change invalidates stale artefacts
        instead of silently predicting from mismatched features.
        """
        from f1predict.features.schema import SCHEMA_VERSION

        return f"s{SCHEMA_VERSION}-f{self.form.recent_races}-r{self.form.reliability_races}"


def _build(cls: type[T], data: Any) -> T:
    """Recursively construct a dataclass tree from nested dicts.

    Unknown keys raise, which turns a config typo into an immediate, located
    error rather than a silently ignored setting.
    """
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section {cls.__name__}. "
            f"Valid keys: {sorted(known)}"
        )

    # `from __future__ import annotations` turns field.type into a string, so
    # resolve the real classes before recursing into nested sections.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        kwargs[name] = _build(ftype, value) if is_dataclass(ftype) else value
    return cls(**kwargs)


def default_config_path() -> Path:
    """Path to the YAML shipped with the project."""
    for candidate in (
        REPO_ROOT / "config" / "default.yaml",
        PACKAGE_DIR / "config" / "default.yaml",
    ):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "config" / "default.yaml"


def load_config(config_path: Path | str | None = None) -> Config:
    """Load YAML config, apply environment overrides, and resolve cache paths."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else default_config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = _build(Config, raw)

    if root := os.getenv("F1PREDICT_CACHE_DIR"):
        cfg = replace(cfg, cache=CacheConfig(root=root, live_ttl_s=cfg.cache.live_ttl_s))
    if sims := os.getenv("F1PREDICT_SIMULATIONS"):
        cfg = replace(cfg, simulation=replace(cfg.simulation, n_simulations=int(sims)))

    return replace(cfg, cache=cfg.cache.resolved())


def replace_cache_root(cfg: Config, root: str | Path) -> Config:
    """Return a copy of ``cfg`` with every cache directory under ``root``."""
    cache = CacheConfig(root=str(root), live_ttl_s=cfg.cache.live_ttl_s).resolved()
    return replace(cfg, cache=cache)


_config: Config | None = None


def get_config() -> Config:
    """Process-wide singleton config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: Config) -> None:
    """Override the singleton (used by tests and the Streamlit app)."""
    global _config
    _config = cfg
