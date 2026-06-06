"""Extract qualifying features from Jolpica results."""

from __future__ import annotations

import pandas as pd


def extract_quali_features(quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a Jolpica qualifying results DataFrame, return a feature DataFrame
    with columns:
        driver_code, driver_id, driver_name, team,
        grid_pos, quali_time_s, quali_gap_to_pole_s,
        reached_q2, reached_q3.

    Input columns expected: driver_code, driver_id, driver_name, team,
        quali_pos, q1_s, q2_s, q3_s, best_quali_s, quali_gap_to_pole_s.
    """
    if quali_df is None or quali_df.empty:
        return pd.DataFrame()

    df = quali_df.copy()

    # Rename for clarity
    df = df.rename(columns={
        "quali_pos": "grid_pos",
        "best_quali_s": "quali_time_s",
    })

    df["reached_q2"] = df["q2_s"].notna().astype(int)
    df["reached_q3"] = df["q3_s"].notna().astype(int)

    keep = [
        "driver_code", "driver_id", "driver_name", "team",
        "grid_pos", "quali_time_s", "quali_gap_to_pole_s",
        "reached_q2", "reached_q3",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
