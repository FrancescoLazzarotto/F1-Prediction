"""Open-Meteo weather: historical archive plus 16-day forecast. No API key.

Archive:  https://archive-api.open-meteo.com/v1/archive
Forecast: https://api.open-meteo.com/v1/forecast
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from f1predict.data.http import HttpJsonClient, NetworkError

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com"
FORECAST_URL = "https://api.open-meteo.com"

#: The archive lags real time by about five days, so anything more recent than
#: this has to come from the forecast endpoint even though it is in the past.
_ARCHIVE_LAG = timedelta(days=6)

#: Open-Meteo publishes forecasts up to 16 days out.
_FORECAST_HORIZON = timedelta(days=16)

_ARCHIVE_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation"
_FORECAST_VARS = (
    "temperature_2m,relative_humidity_2m,precipitation_probability,"
    "wind_speed_10m,cloud_cover,precipitation"
)

DEFAULT_WEATHER: dict[str, Any] = {
    "temperature": 22.0,
    "humidity": 55.0,
    "wind_speed": 3.0,
    "cloud_cover": 30.0,
    "rain_prob": 0.10,
    "source": "default",
}


def _client(base_url: str, cache_dir: str | None) -> HttpJsonClient:
    # Open-Meteo's free tier is generous; one request per second is polite.
    return HttpJsonClient(base_url, cache_dir=cache_dir, rate_limit=2.0, timeout_s=15,
                          max_retries=3)


def get_weather_for_race(
    lat: float,
    lon: float,
    race_datetime: datetime,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Weather at a circuit for the hour of the race.

    Picks the archive for past races and the forecast for upcoming ones, and
    falls back to seasonal-average defaults if neither is reachable.
    """
    if race_datetime.tzinfo is None:
        race_datetime = race_datetime.replace(tzinfo=timezone.utc)

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180) or (lat == 0.0 and lon == 0.0):
        log.debug("Implausible circuit coordinates (%s, %s); using defaults.", lat, lon)
        return dict(DEFAULT_WEATHER)

    now = datetime.now(tz=timezone.utc)
    try:
        if race_datetime < now - _ARCHIVE_LAG:
            return _fetch_archive(lat, lon, race_datetime, cache_dir)
        if race_datetime <= now + _FORECAST_HORIZON:
            return _fetch_forecast(lat, lon, race_datetime, cache_dir)
        log.debug("Race is beyond the forecast horizon; using defaults.")
    except (NetworkError, KeyError, ValueError) as exc:
        log.warning("Weather lookup failed (%s); using defaults.", exc)

    return dict(DEFAULT_WEATHER)


def _fetch_archive(lat: float, lon: float, dt: datetime, cache_dir: str | None) -> dict[str, Any]:
    date_str = dt.strftime("%Y-%m-%d")
    payload = _client(ARCHIVE_URL, cache_dir).get(
        "v1/archive",
        params={
            "latitude": round(lat, 3), "longitude": round(lon, 3),
            "start_date": date_str, "end_date": date_str,
            "hourly": _ARCHIVE_VARS,
            "wind_speed_unit": "ms", "timezone": "UTC",
        },
        ttl_s=None,  # historical weather never changes
    )
    return _extract_hour(payload, dt, source="archive")


def _fetch_forecast(lat: float, lon: float, dt: datetime, cache_dir: str | None) -> dict[str, Any]:
    payload = _client(FORECAST_URL, cache_dir).get(
        "v1/forecast",
        params={
            "latitude": round(lat, 3), "longitude": round(lon, 3),
            "hourly": _FORECAST_VARS,
            "wind_speed_unit": "ms", "timezone": "UTC",
            "past_days": 7, "forecast_days": 16,
        },
        ttl_s=3 * 3600,  # a forecast is worth re-fetching a few times a day
    )
    return _extract_hour(payload, dt, source="forecast")


def _extract_hour(payload: dict, target: datetime, source: str) -> dict[str, Any]:
    """Pick the hourly row closest to the race start."""
    hourly = payload.get("hourly") or {}
    if "time" not in hourly or not hourly["time"]:
        return dict(DEFAULT_WEATHER)

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        return dict(DEFAULT_WEATHER)

    target_ts = pd.Timestamp(target)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")

    # Positional lookup: idxmin returns a label, which only coincides with the
    # position while the index is a clean RangeIndex.
    position = int((df["time"] - target_ts).abs().to_numpy().argmin())
    row = df.iloc[position]

    return {
        "temperature": _num(row.get("temperature_2m"), DEFAULT_WEATHER["temperature"]),
        "humidity": _num(row.get("relative_humidity_2m"), DEFAULT_WEATHER["humidity"]),
        "wind_speed": _num(row.get("wind_speed_10m"), DEFAULT_WEATHER["wind_speed"]),
        "cloud_cover": _num(row.get("cloud_cover"), DEFAULT_WEATHER["cloud_cover"]),
        "rain_prob": _rain_probability(row),
        "source": source,
    }


def _rain_probability(row: pd.Series) -> float:
    """Rain likelihood in [0, 1].

    The forecast endpoint gives an explicit probability; the archive only has
    observed millimetres, so derive a probability from actual rainfall.
    """
    explicit = row.get("precipitation_probability")
    if explicit is not None and not pd.isna(explicit):
        return max(0.0, min(1.0, float(explicit) / 100.0))

    precipitation = row.get("precipitation")
    if precipitation is not None and not pd.isna(precipitation):
        mm = float(precipitation)
        if mm <= 0.0:
            return 0.0
        # Anything past ~2 mm/h is unambiguously a wet race.
        return min(1.0, 0.35 + mm / 3.0)

    return DEFAULT_WEATHER["rain_prob"]


def _num(value, default: float) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
