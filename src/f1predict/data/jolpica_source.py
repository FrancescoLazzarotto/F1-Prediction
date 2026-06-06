"""Jolpica-F1 API source (Ergast successor, free, no auth).

Docs: https://api.jolpi.ca/ergast/f1/
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "f1predict/2.0 (github.com/f1predict)"


def _get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = 2 ** attempt
                log.warning("Rate-limited, waiting %ds…", wait)
                time.sleep(wait)
            else:
                raise
        except requests.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                raise RuntimeError(f"Jolpica request failed for {url}: {exc}") from exc
    raise RuntimeError(f"Jolpica request failed after {retries} retries: {url}")


# ── Circuit info ──────────────────────────────────────────────────────────────

def get_circuit_info(year: int, round_num: int) -> dict:
    """Return {name, circuit_id, lat, lon, locality, country} for a race."""
    data = _get(f"{year}/{round_num}.json")
    races = data["MRData"]["RaceTable"].get("Races", [])
    if not races:
        raise ValueError(f"No race data for {year} round {round_num}")
    circuit = races[0]["Circuit"]
    return {
        "name": circuit.get("circuitName", ""),
        "circuit_id": circuit.get("circuitId", ""),
        "lat": float(circuit["Location"]["lat"]),
        "lon": float(circuit["Location"]["long"]),
        "locality": circuit["Location"].get("locality", ""),
        "country": circuit["Location"].get("country", ""),
    }


# ── Race results ──────────────────────────────────────────────────────────────

def get_race_results(year: int, round_num: int) -> pd.DataFrame:
    """
    Return per-driver race results for one round.
    Columns: driver_id, driver_code, driver_name, team, constructor_id,
             grid_pos, race_pos, points, status, laps.
    """
    data = _get(f"{year}/{round_num}/results.json")
    races = data["MRData"]["RaceTable"].get("Races", [])
    if not races:
        return pd.DataFrame()
    return _parse_results(races[0]["Results"], year, round_num,
                          races[0]["Circuit"]["circuitId"])


def _safe_pos(r: dict) -> int:
    """
    Get numeric finishing position from a Jolpica result dict.
    Jolpica dropped `positionOrder`; uses `position` (string, may be non-numeric for DNFs).
    Falls back to a high number (20) for non-classified/DSQ.
    """
    for key in ("positionOrder", "position"):
        val = r.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return 20


def _parse_results(results: list, year: int, round_num: int, circuit_id: str) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r["Driver"]
        rows.append({
            "year": year,
            "round": round_num,
            "circuit_id": circuit_id,
            "driver_id": d["driverId"],
            "driver_code": d.get("code", d["driverId"][:3].upper()),
            "driver_name": f"{d['givenName']} {d['familyName']}",
            "team": r["Constructor"]["name"],
            "constructor_id": r["Constructor"]["constructorId"],
            "grid_pos": int(r.get("grid") or 20),
            "race_pos": _safe_pos(r),
            "points": float(r.get("points", 0) or 0),
            "status": r.get("status", ""),
            "laps": int(r.get("laps", 0) or 0),
        })
    return pd.DataFrame(rows)


def get_season_results(year: int) -> pd.DataFrame:
    """
    Fetch ALL race results for a full season, handling Jolpica's 100-row page cap.
    Paginates via offset until all results are retrieved.
    """
    page_size = 100
    dfs: list[pd.DataFrame] = []
    offset = 0

    while True:
        data = _get(f"{year}/results.json", params={"limit": page_size, "offset": offset})
        mr = data["MRData"]
        total = int(mr.get("total", 0))
        races = mr["RaceTable"].get("Races", [])

        for race in races:
            round_num = int(race["round"])
            circuit_id = race["Circuit"]["circuitId"]
            df = _parse_results(race["Results"], year, round_num, circuit_id)
            dfs.append(df)

        offset += page_size
        if offset >= total:
            break

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Qualifying results ────────────────────────────────────────────────────────

def get_quali_results(year: int, round_num: int) -> pd.DataFrame:
    """
    Return qualifying results.
    Columns: driver_id, driver_code, driver_name, team, quali_pos,
             q1_s, q2_s, q3_s, best_quali_s, quali_gap_to_pole_s.
    """
    data = _get(f"{year}/{round_num}/qualifying.json")
    races = data["MRData"]["RaceTable"].get("Races", [])
    if not races:
        return pd.DataFrame()
    return _parse_quali(races[0].get("QualifyingResults", []))


def _parse_quali(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r["Driver"]
        rows.append({
            "driver_id": d["driverId"],
            "driver_code": d.get("code", d["driverId"][:3].upper()),
            "driver_name": f"{d['givenName']} {d['familyName']}",
            "team": r["Constructor"]["name"],
            "quali_pos": int(r["position"]),
            "q1_s": _parse_lap_time(r.get("Q1", "")),
            "q2_s": _parse_lap_time(r.get("Q2", "")),
            "q3_s": _parse_lap_time(r.get("Q3", "")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["best_quali_s"] = df[["q3_s", "q2_s", "q1_s"]].bfill(axis=1).iloc[:, 0]
    pole = df["best_quali_s"].min()
    df["quali_gap_to_pole_s"] = df["best_quali_s"] - pole
    return df


def _parse_lap_time(t: str) -> float | None:
    """Convert '1:15.123' or '75.123' to seconds. Returns None if empty."""
    if not t or t.strip() == "":
        return None
    try:
        parts = t.strip().split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None
