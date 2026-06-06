"""Configuration loader: merges config/default.yaml with optional .env overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent  # repo root


def load_config(config_path: Path | None = None) -> dict:
    """Load and return the merged configuration dict."""
    load_dotenv(_ROOT / ".env", override=False)

    path = config_path or (_ROOT / "config" / "default.yaml")
    with open(path, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    # Allow env-var overrides for cache dirs
    if override := os.getenv("F1PREDICT_CACHE_DIR"):
        base = override.rstrip("/\\")
        cfg["cache"]["fastf1_dir"] = f"{base}/fastf1"
        cfg["cache"]["features_dir"] = f"{base}/features"
        cfg["cache"]["models_dir"] = f"{base}/models"

    # Resolve all cache dirs to absolute paths relative to repo root
    for key in ("fastf1_dir", "features_dir", "models_dir"):
        p = Path(cfg["cache"][key])
        cfg["cache"][key] = str(_ROOT / p if not p.is_absolute() else p)

    return cfg


# Singleton for use across the package
_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config
