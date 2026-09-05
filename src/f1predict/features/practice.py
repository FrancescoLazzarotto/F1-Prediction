"""Per-driver features extracted from a Free Practice session.

Practice timing is noisy: in-laps, out-laps, cool-down laps, traffic, red flags
and four different tyre compounds all land in the same table. The job here is to
separate the two signals that matter — one-lap pace and race pace — and express
both as a percentage gap to the session best so they compare across circuits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from f1predict.constants import COMPOUND_OFFSET_S

log = logging.getLogger(__name__)

#: Laps slower than this multiple of a stint's best are cool-down or traffic
#: laps, not representative running.
_STINT_OUTLIER_RATIO = 1.07

#: Ratio applied to a driver's best lap when no usable long run exists.
_LONG_RUN_FALLBACK_RATIO = 1.025

PRACTICE_COLUMNS = [
    "driver_code", "fp_best_lap_s", "fp_best_gap_pct", "fp_pace_s",
    "fp_pace_gap_pct", "fp_rank_pct", "fp_theory_lap_s", "fp_theory_gap_pct",
    "fp_n_laps", "fp_n_long_run_laps", "fp_compound",
]


def extract_practice_features(laps: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """Reduce a FastF1 laps table to one row per driver.

    Args:
        laps: A FastF1 ``Laps`` frame from any practice session.
        min_laps: Minimum clean laps in a stint for it to count as a long run.

    Returns:
        One row per driver with the columns in :data:`PRACTICE_COLUMNS`, or an
        empty frame when the input has no usable timing.
    """
    if laps is None or len(laps) == 0 or "Driver" not in laps.columns:
        return _empty()

    df = laps.copy()
    for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        df[f"{col}_s"] = (
            df[col].dt.total_seconds() if col in df.columns else np.nan
        )

    if df["LapTime_s"].notna().sum() == 0:
        return _empty()

    valid = _valid_laps(df)
    if valid.empty:
        # Every lap was filtered out (a wet or red-flagged session); fall back
        # to raw timed laps rather than returning nothing at all.
        valid = df[df["LapTime_s"].notna()]
    if valid.empty:
        return _empty()

    best = _best_laps(valid)
    pace = _long_run_pace(valid, min_laps)
    theory = _theoretical_best(valid)
    counts = (
        valid.groupby("Driver")["LapTime_s"].count().rename("fp_n_laps").reset_index()
        .rename(columns={"Driver": "driver_code"})
    )

    out = (
        best.merge(pace, on="driver_code", how="left")
        .merge(theory, on="driver_code", how="left")
        .merge(counts, on="driver_code", how="left")
    )

    # A driver with no clean long run still has one-lap pace to fall back on.
    missing = out["fp_pace_s"].isna()
    out.loc[missing, "fp_pace_s"] = out.loc[missing, "fp_best_lap_s"] * _LONG_RUN_FALLBACK_RATIO
    out["fp_n_long_run_laps"] = out["fp_n_long_run_laps"].fillna(0).astype(int)
    out["fp_n_laps"] = out["fp_n_laps"].fillna(0).astype(int)

    out["fp_best_gap_pct"] = _gap_pct(out["fp_best_lap_s"])
    out["fp_pace_gap_pct"] = _gap_pct(out["fp_pace_s"])
    out["fp_theory_gap_pct"] = _gap_pct(out["fp_theory_lap_s"])
    # Rank as a share of the field, so it means the same in a 20- and 22-car grid.
    out["fp_rank_pct"] = out["fp_best_lap_s"].rank(method="min") / max(len(out), 1)

    return out.reindex(columns=PRACTICE_COLUMNS).reset_index(drop=True)


def _valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only timed, accurate, green-flag laps that are not in/out laps."""
    mask = df["LapTime_s"].notna() & (df["LapTime_s"] > 0)

    if "PitOutTime" in df.columns:
        mask &= df["PitOutTime"].isna()
    if "PitInTime" in df.columns:
        mask &= df["PitInTime"].isna()
    if "IsAccurate" in df.columns:
        mask &= df["IsAccurate"].fillna(True).astype(bool)
    if "Deleted" in df.columns:
        # Object dtype with None entries, so coerce before the boolean cast.
        mask &= ~df["Deleted"].astype("boolean").fillna(False).astype(bool)
    if "TrackStatus" in df.columns:
        # "1" is all-clear; anything else is yellow, SC, VSC or red.
        mask &= df["TrackStatus"].astype(str).str.strip() == "1"

    return df[mask]


def _best_laps(valid: pd.DataFrame) -> pd.DataFrame:
    return (
        valid.groupby("Driver")["LapTime_s"].min()
        .rename("fp_best_lap_s").reset_index()
        .rename(columns={"Driver": "driver_code"})
    )


def _long_run_pace(valid: pd.DataFrame, min_laps: int) -> pd.DataFrame:
    """Best representative race-pace stint per driver, normalised for compound.

    A stint qualifies when it has ``min_laps`` clean laps left after dropping
    cool-down laps. Its pace is the median of those laps, shifted onto a common
    compound baseline so a hard-tyre run is not read as a slow car.
    """
    if "Stint" not in valid.columns:
        return pd.DataFrame(columns=["driver_code", "fp_pace_s",
                                     "fp_n_long_run_laps", "fp_compound"])

    records: list[dict] = []
    for (driver, _stint), group in valid.groupby(["Driver", "Stint"], dropna=True):
        times = group["LapTime_s"]
        if len(times) < min_laps:
            continue

        # Drop laps well off the stint's best: those are traffic or cool-downs.
        representative = times[times <= times.min() * _STINT_OUTLIER_RATIO]
        if len(representative) < min_laps:
            continue

        compound = str(group["Compound"].iloc[0]) if "Compound" in group.columns else "UNKNOWN"
        offset = COMPOUND_OFFSET_S.get(compound.upper(), COMPOUND_OFFSET_S["UNKNOWN"])
        records.append({
            "driver_code": driver,
            "fp_pace_s": float(representative.median()) - offset,
            "fp_n_long_run_laps": len(representative),
            "fp_compound": compound,
        })

    if not records:
        return pd.DataFrame(columns=["driver_code", "fp_pace_s",
                                     "fp_n_long_run_laps", "fp_compound"])

    stints = pd.DataFrame(records)
    best_idx = stints.groupby("driver_code")["fp_pace_s"].idxmin()
    return stints.loc[best_idx].reset_index(drop=True)


def _theoretical_best(valid: pd.DataFrame) -> pd.DataFrame:
    """Sum of each driver's best three sectors — their perfect lap.

    Rewards a driver who was fast in every sector but never strung them
    together, which is a better read on car pace than a single clean lap.
    """
    sector_cols = [f"Sector{i}Time_s" for i in (1, 2, 3)]
    if not all(col in valid.columns for col in sector_cols):
        return pd.DataFrame(columns=["driver_code", "fp_theory_lap_s"])

    bests = valid.groupby("Driver")[sector_cols].min()
    if bests.empty:
        return pd.DataFrame(columns=["driver_code", "fp_theory_lap_s"])

    theory = bests.sum(axis=1, min_count=3).rename("fp_theory_lap_s")
    return theory.reset_index().rename(columns={"Driver": "driver_code"})


def _gap_pct(series: pd.Series) -> pd.Series:
    """Percentage gap to the fastest value in the column."""
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    reference = series.min(skipna=True)
    if pd.isna(reference) or reference <= 0:
        return pd.Series(np.nan, index=series.index)
    return (series / reference - 1.0) * 100.0


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRACTICE_COLUMNS})
