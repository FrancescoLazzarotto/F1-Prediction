"""Open-Meteo weather source — completely free, no API key required.

Historical archive: https://archive-api.open-meteo.com/v1/archive
Forecast (up to 16 days): https://api.open-meteo.com/v1/forecast
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,cloud_cover"
_DEFAULTS = {
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed": 3.0,
    "cloud_cover": 30.0,
    "rain_prob": 0.1,
}


def get_weather_for_race(lat: float, lon: float, race_datetime: datetime) -> dict:
    """
    Return a weather dict for the given circuit (lat, lon) at race_datetime (UTC).
    Uses historical archive for past races, forecast API for future ones.
    Falls back gracefully to defaults on any error.
    """
    if race_datetime.tzinfo is None:
        race_datetime = race_datetime.replace(tzinfo=timezone.utc)

    now = datetime.now(tz=timezone.utc)
    is_past = race_datetime < now - timedelta(hours=6)

    try:
        if is_past:
            return _fetch_historical(lat, lon, race_datetime)
        else:
            return _fetch_forecast(lat, lon, race_datetime)
    except Exception as exc:
        log.warning("Weather fetch failed (%s), using defaults.", exc)
        return _DEFAULTS.copy()


def _fetch_historical(lat: float, lon: float, dt: datetime) -> dict:
    date_str = dt.strftime("%Y-%m-%d")
    r = requests.get(
        _ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
            "wind_speed_unit": "ms", "timezone": "UTC",
        },
        timeout=15,
    )
    r.raise_for_status()
    return _extract_hour(r.json(), dt)


def _fetch_forecast(lat: float, lon: float, dt: datetime) -> dict:
    r = requests.get(
        _FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon,
            "hourly": _HOURLY_VARS,
            "wind_speed_unit": "ms", "timezone": "UTC",
            "forecast_days": 16,
        },
        timeout=15,
    )
    r.raise_for_status()
    return _extract_hour(r.json(), dt)


def _extract_hour(data: dict, target_dt: datetime) -> dict:
    hourly = data.get("hourly", {})
    if "time" not in hourly:
        return _DEFAULTS.copy()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    target = pd.Timestamp(target_dt)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    idx = (df["time"] - target).abs().idxmin()
    row = df.iloc[idx]

    return {
        "temperature": float(row.get("temperature_2m", _DEFAULTS["temperature"]) or _DEFAULTS["temperature"]),
        "humidity": float(row.get("relative_humidity_2m", _DEFAULTS["humidity"]) or _DEFAULTS["humidity"]),
        "wind_speed": float(row.get("wind_speed_10m", _DEFAULTS["wind_speed"]) or _DEFAULTS["wind_speed"]),
        "cloud_cover": float(row.get("cloud_cover", _DEFAULTS["cloud_cover"]) or _DEFAULTS["cloud_cover"]),
        "rain_prob": float((row.get("precipitation_probability", _DEFAULTS["rain_prob"] * 100) or 0)) / 100.0,
    }
