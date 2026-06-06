"""Compute driver and team form features from historical season results."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DNF_STATUSES = {
    "Retired", "Accident", "Engine", "Gearbox", "Mechanical",
    "Collision", "Hydraulics", "Power Unit", "Brakes", "Electrical",
    "Suspension", "ERS", "Collision damage", "Damage", "Overheating",
    "Did not finish", "Excluded",
}


def compute_form_features(
    season_results: pd.DataFrame,
    driver_id: str,
    year: int,
    before_round: int,
    circuit_id: str,
    n_races: int = 5,
) -> dict:
    """
    Compute form features for a single driver before a given round.

    Args:
        season_results: Combined results DataFrame from one or more seasons.
                        Required columns: year, round, circuit_id, driver_id,
                        race_pos, points, status.
        driver_id: Jolpica driverId string.
        year: Target race year.
        before_round: Target race round (excluded from form window).
        circuit_id: Circuit identifier (for circuit-specific history).
        n_races: Rolling window size.

    Returns a dict with form feature values.
    """
    # Filter to this driver's results prior to the target race
    driver_df = season_results[
        (season_results["driver_id"] == driver_id) &
        ~(
            (season_results["year"] == year) &
            (season_results["round"] >= before_round)
        )
    ].sort_values(["year", "round"], ascending=True)

    if driver_df.empty:
        return _default_form()

    recent = driver_df.tail(n_races)
    dnf_mask = recent["status"].isin(_DNF_STATUSES)

    form: dict = {
        "form_avg_pos": float(recent["race_pos"].mean()),
        "form_avg_pts": float(recent["points"].mean()),
        "form_dnf_rate": float(dnf_mask.mean()),
        "form_n_races": int(len(recent)),
    }

    # ── Circuit-specific history ─────────────────────────────────────────────
    circuit_df = driver_df[driver_df["circuit_id"] == circuit_id]
    if not circuit_df.empty:
        form["circuit_avg_pos"] = float(circuit_df["race_pos"].mean())
        form["circuit_n_starts"] = int(len(circuit_df))
    else:
        form["circuit_avg_pos"] = form["form_avg_pos"]
        form["circuit_n_starts"] = 0

    return form


def compute_team_form(
    season_results: pd.DataFrame,
    constructor_id: str,
    year: int,
    before_round: int,
    n_races: int = 5,
) -> dict:
    """Return average team race position over the last n_races races."""
    team_df = season_results[
        (season_results["constructor_id"] == constructor_id) &
        ~(
            (season_results["year"] == year) &
            (season_results["round"] >= before_round)
        )
    ].sort_values(["year", "round"])

    if team_df.empty:
        return {"team_avg_pos": 10.0, "team_avg_pts": 5.0}

    recent = team_df.tail(n_races * 2)  # 2 drivers × n_races
    return {
        "team_avg_pos": float(recent["race_pos"].mean()),
        "team_avg_pts": float(recent["points"].mean()),
    }


def build_form_matrix(
    season_results: pd.DataFrame,
    drivers: pd.DataFrame,
    year: int,
    round_num: int,
    circuit_id: str,
    n_races: int = 5,
) -> pd.DataFrame:
    """
    Build the form feature DataFrame for all drivers in an entry list.

    Args:
        season_results: Multi-season results (columns: year, round, circuit_id,
                        driver_id, constructor_id, race_pos, points, status).
        drivers: Entry list DataFrame with columns: driver_code, driver_id, team.
        year, round_num, circuit_id: Target race identifiers.
        n_races: Rolling window.

    Returns a DataFrame with one row per driver and form feature columns.
    """
    if drivers.empty or season_results.empty:
        return pd.DataFrame()

    # Build constructor_id lookup from season_results for matching drivers
    cid_map: dict[str, str] = {}
    if "constructor_id" in season_results.columns:
        latest = season_results[season_results["driver_id"].isin(drivers["driver_id"].tolist())]
        if not latest.empty:
            cid_map = (
                latest.sort_values(["year", "round"])
                .groupby("driver_id")["constructor_id"]
                .last()
                .to_dict()
            )

    rows = []
    for _, driver_row in drivers.iterrows():
        did = driver_row.get("driver_id", "")
        form = compute_form_features(
            season_results, did, year, round_num, circuit_id, n_races
        )
        constructor_id = cid_map.get(did, "")
        if constructor_id:
            team_form = compute_team_form(season_results, constructor_id, year, round_num, n_races)
        else:
            team_form = {"team_avg_pos": 10.0, "team_avg_pts": 5.0}

        rows.append({
            "driver_code": driver_row.get("driver_code", ""),
            "driver_id": did,
            **form,
            **team_form,
        })

    return pd.DataFrame(rows)


def _default_form() -> dict:
    return {
        "form_avg_pos": 10.0,
        "form_avg_pts": 5.0,
        "form_dnf_rate": 0.1,
        "form_n_races": 0,
        "circuit_avg_pos": 10.0,
        "circuit_n_starts": 0,
    }
