"""Rolling driver and team form, computed for the whole history in one pass.

The central guarantee here is **no leakage**: every window is shifted so that a
driver's features for round 12 are built strictly from rounds 1-11. Training and
prediction share the same code path — an upcoming race is appended to the
history as a placeholder row and then read back out — so features cannot drift
between the two.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from f1predict.config import FormConfig

log = logging.getLogger(__name__)

FORM_COLUMNS = [
    "year", "round", "driver_id", "constructor_id",
    "form_avg_pos", "form_avg_pts", "form_pos_gain", "form_trend",
    "form_dnf_rate", "form_avg_quali_pos",
    "circuit_avg_pos", "circuit_n_starts",
    "team_avg_pos", "team_avg_pts", "team_dnf_rate", "team_avg_quali_pos",
]

#: Values used for a driver with no prior races at all (a rookie, or round 1 of
#: the earliest season we hold). Deliberately mid-field rather than zero.
FORM_DEFAULTS: dict[str, float] = {
    "form_avg_pos": 10.0,
    "form_avg_pts": 4.0,
    "form_pos_gain": 0.0,
    "form_trend": 0.0,
    "form_dnf_rate": 0.09,
    "form_avg_quali_pos": 10.0,
    "circuit_avg_pos": 10.0,
    "circuit_n_starts": 0.0,
    "team_avg_pos": 10.0,
    "team_avg_pts": 4.0,
    "team_dnf_rate": 0.09,
    "team_avg_quali_pos": 10.0,
}

_REQUIRED_COLUMNS = ("year", "round", "driver_id", "race_pos")


def build_form_table(
    results: pd.DataFrame,
    quali: pd.DataFrame | None = None,
    form_cfg: FormConfig | None = None,
) -> pd.DataFrame:
    """Form features for every (driver, race) in ``results``.

    Args:
        results: Multi-season race results. Needs at least ``year``, ``round``,
            ``driver_id`` and ``race_pos``; ``constructor_id``, ``grid_pos``,
            ``points``, ``circuit_id`` and ``dnf`` enrich the output when present.
        quali: Optional qualifying history, used for the qualifying-form columns.
        form_cfg: Window sizes. Defaults to :class:`FormConfig`.

    Returns:
        One row per (year, round, driver_id) whose values describe the driver's
        state *entering* that race.
    """
    cfg = form_cfg or FormConfig()

    if results is None or results.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in FORM_COLUMNS})
    missing = [c for c in _REQUIRED_COLUMNS if c not in results.columns]
    if missing:
        raise ValueError(f"Results frame is missing required column(s): {missing}")

    df = _normalise(results)
    df = _attach_quali(df, quali)

    df = _driver_form(df, cfg)
    df = _circuit_history(df)
    df = _team_form(df, cfg)

    # Fall back to sensible mid-field values only where a window was genuinely
    # empty, which is exactly the rookie / first-race case.
    for column, default in FORM_DEFAULTS.items():
        if column in df.columns:
            df[column] = df[column].fillna(default)

    return df.reindex(columns=FORM_COLUMNS).reset_index(drop=True)


def _normalise(results: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and derive the columns the windows need."""
    df = results.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["race_pos"] = pd.to_numeric(df["race_pos"], errors="coerce")
    df["points"] = pd.to_numeric(df.get("points"), errors="coerce")

    if "grid_pos" in df.columns:
        grid = pd.to_numeric(df["grid_pos"], errors="coerce")
        # A grid slot of 0 means a pit-lane start, which is not a real position.
        df["pos_gain"] = grid.where(grid > 0) - df["race_pos"]
    else:
        df["pos_gain"] = np.nan

    if "dnf" in df.columns:
        df["dnf_flag"] = df["dnf"].astype("boolean").fillna(False).astype(float)
    elif "status" in df.columns:
        from f1predict.constants import is_dnf

        df["dnf_flag"] = df["status"].map(is_dnf).astype(float)
    else:
        df["dnf_flag"] = np.nan

    if "constructor_id" not in df.columns:
        df["constructor_id"] = ""
    df["constructor_id"] = df["constructor_id"].fillna("").astype(str)
    df["driver_id"] = df["driver_id"].fillna("").astype(str)

    return df.sort_values(["year", "round"], kind="stable").reset_index(drop=True)


def _attach_quali(df: pd.DataFrame, quali: pd.DataFrame | None) -> pd.DataFrame:
    """Join each driver's qualifying position onto their race row."""
    if quali is None or quali.empty or "quali_pos" not in quali.columns:
        df["quali_pos"] = np.nan
        return df

    q = quali[["year", "round", "driver_id", "quali_pos"]].copy()
    q["year"] = pd.to_numeric(q["year"], errors="coerce").astype("Int64")
    q["round"] = pd.to_numeric(q["round"], errors="coerce").astype("Int64")
    q["quali_pos"] = pd.to_numeric(q["quali_pos"], errors="coerce")
    q["driver_id"] = q["driver_id"].fillna("").astype(str)
    q = q.drop_duplicates(["year", "round", "driver_id"])

    merged = df.merge(q, on=["year", "round", "driver_id"], how="left")
    return merged.sort_values(["year", "round"], kind="stable").reset_index(drop=True)


def _driver_form(df: pd.DataFrame, cfg: FormConfig) -> pd.DataFrame:
    """Rolling windows over a driver's own previous races."""
    by_driver = df.groupby("driver_id", sort=False)

    df["form_avg_pos"] = _rolling_mean(by_driver["race_pos"], cfg.recent_races)
    df["form_avg_pts"] = _rolling_mean(by_driver["points"], cfg.recent_races)
    df["form_pos_gain"] = _rolling_mean(by_driver["pos_gain"], cfg.recent_races)
    df["form_dnf_rate"] = _rolling_mean(by_driver["dnf_flag"], cfg.reliability_races)
    df["form_avg_quali_pos"] = _rolling_mean(by_driver["quali_pos"], cfg.recent_races)

    # Trend: recent short window versus the longer one. Negative means the
    # driver has been finishing higher up lately than their season average.
    short = _rolling_mean(by_driver["race_pos"], cfg.trend_races)
    df["form_trend"] = short - df["form_avg_pos"]
    return df


def _circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding record of how a driver has gone at this circuit before."""
    if "circuit_id" not in df.columns:
        df["circuit_avg_pos"] = np.nan
        df["circuit_n_starts"] = 0.0
        return df

    by_circuit = df.groupby(["driver_id", "circuit_id"], sort=False)
    df["circuit_avg_pos"] = by_circuit["race_pos"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["circuit_n_starts"] = by_circuit["race_pos"].transform(
        lambda s: s.shift(1).expanding().count()
    ).astype(float)
    return df


def _team_form(df: pd.DataFrame, cfg: FormConfig) -> pd.DataFrame:
    """Rolling team form, aggregated per race weekend before the window is taken.

    Shifting driver-level rows by one would leak: a constructor's two cars are
    adjacent rows in the same race, so the second driver's window would include
    their teammate's result *from the race being predicted*. Collapsing each
    constructor to one row per weekend first makes the shift skip the whole
    race, for both cars alike.
    """
    if "constructor_id" not in df.columns:
        for column in ("team_avg_pos", "team_avg_pts", "team_dnf_rate",
                       "team_avg_quali_pos"):
            df[column] = np.nan
        return df

    per_race = (
        df.groupby(["constructor_id", "year", "round"], as_index=False)
        .agg(
            _pos=("race_pos", "mean"),
            _pts=("points", "mean"),
            _dnf=("dnf_flag", "mean"),
            _quali=("quali_pos", "mean"),
        )
        .sort_values(["year", "round"], kind="stable")
        .reset_index(drop=True)
    )

    by_team = per_race.groupby("constructor_id", sort=False)
    per_race["team_avg_pos"] = _rolling_mean(by_team["_pos"], cfg.recent_races)
    per_race["team_avg_pts"] = _rolling_mean(by_team["_pts"], cfg.recent_races)
    per_race["team_dnf_rate"] = _rolling_mean(by_team["_dnf"], cfg.reliability_races)
    per_race["team_avg_quali_pos"] = _rolling_mean(by_team["_quali"], cfg.recent_races)

    columns = ["constructor_id", "year", "round", "team_avg_pos", "team_avg_pts",
               "team_dnf_rate", "team_avg_quali_pos"]
    return df.merge(per_race[columns], on=["constructor_id", "year", "round"], how="left")


def _rolling_mean(grouped, window: int) -> pd.Series:
    """Mean of the previous ``window`` rows within each group.

    The ``shift(1)`` is what keeps a row out of its own feature window.
    """
    return grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def form_for_event(
    results: pd.DataFrame,
    entries: pd.DataFrame,
    year: int,
    round_num: int,
    circuit_id: str = "",
    quali: pd.DataFrame | None = None,
    form_cfg: FormConfig | None = None,
) -> pd.DataFrame:
    """Form features for the drivers entered in one (possibly future) race.

    The event is appended to the history as a placeholder with no result, then
    the full table is rebuilt and that event's rows read back. This is slightly
    more work than a bespoke query, but it guarantees prediction-time features
    are computed by exactly the same code as training-time ones.
    """
    cfg = form_cfg or FormConfig()

    if entries is None or entries.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in FORM_COLUMNS})

    history = results.copy() if results is not None else pd.DataFrame()

    # Drop any existing rows for the target race: we want form *entering* it,
    # and a backtest must not see the result it is being scored against.
    if not history.empty and {"year", "round"} <= set(history.columns):
        history = history[~((history["year"] == year) & (history["round"] == round_num))]

    placeholder = pd.DataFrame({
        "year": year,
        "round": round_num,
        "circuit_id": circuit_id,
        "driver_id": entries["driver_id"].astype(str).to_numpy(),
        "constructor_id": entries.get(
            "constructor_id", pd.Series([""] * len(entries))
        ).astype(str).to_numpy(),
        "race_pos": np.nan,
        "grid_pos": np.nan,
        "points": np.nan,
        "dnf": pd.NA,
        "status": "",
    })

    combined = (
        pd.concat([history, placeholder], ignore_index=True)
        if not history.empty else placeholder
    )

    if quali is not None and not quali.empty:
        quali = quali[~((quali["year"] == year) & (quali["round"] == round_num))]

    table = build_form_table(combined, quali=quali, form_cfg=cfg)
    event = table[(table["year"] == year) & (table["round"] == round_num)]
    return event.reset_index(drop=True)
