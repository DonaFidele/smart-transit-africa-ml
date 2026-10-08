"""
Accès à la météo réelle via Open-Meteo (gratuit, sans clé API).

- fetch_historical_rain : pluie horaire passée (entraînement), avec cache local.
- fetch_forecast        : prévision horaire (application, Cotonou).

Aucune dépendance externe : uniquement la bibliothèque standard + pandas.
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Seuil de "forte pluie" (mm/h) : classification météorologique usuelle (> 7,6 mm/h)
HEAVY_RAIN_MM = 7.6


def _get_json(url, params, timeout=30):
    full_url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full_url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_historical_rain(lat, lon, start, end, cache_path=None):
    """
    Pluie horaire (mm) entre deux dates incluses, à l'heure locale du lieu.
    Retourne un DataFrame [time, Rain_mm, Date, Hour].
    Si cache_path est fourni et couvre la période, aucun appel réseau n'est fait.
    """
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["time"])
        if (not cached.empty
                and cached["time"].min().date() <= start
                and cached["time"].max().date() >= end):
            return _finalize(cached, start, end)

    payload = _get_json(ARCHIVE_URL, {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "precipitation",
        "timezone": "auto",
    })
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "Rain_mm": hourly["precipitation"],
    })
    df["Rain_mm"] = df["Rain_mm"].fillna(0.0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
    return _finalize(df, start, end)


def _finalize(df, start, end):
    df = df.copy()
    df["Date"] = df["time"].dt.date
    df["Hour"] = df["time"].dt.hour
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    return df[["time", "Rain_mm", "Date", "Hour"]].reset_index(drop=True)


def fetch_forecast(lat, lon, hours=48):
    """
    Prévision horaire pour les prochaines heures (heure locale du lieu).
    Retourne un DataFrame [time, Rain_mm, Rain_prob].
    """
    payload = _get_json(FORECAST_URL, {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "hourly": "precipitation,precipitation_probability",
        "forecast_days": 3,
        "timezone": "auto",
    })
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "Rain_mm": hourly["precipitation"],
        "Rain_prob": hourly.get("precipitation_probability", [None] * len(hourly["time"])),
    })
    df["Rain_mm"] = df["Rain_mm"].fillna(0.0)

    # Heure locale du lieu = UTC + décalage fourni par l'API
    offset = int(payload.get("utc_offset_seconds", 0))
    now_local = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=offset)).replace(minute=0, second=0, microsecond=0)
    df = df[df["time"] >= pd.Timestamp(now_local)].head(hours)
    return df.reset_index(drop=True)