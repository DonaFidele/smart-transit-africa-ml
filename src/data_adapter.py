"""
Data adapter layer.

Everything downstream (training, forecasting, evaluation, app) consumes ONE canonical table:

    history : one row per (hour, zone) on a COMPLETE hourly grid
              columns: time, Zone_ID (int), Demand_Volume (activity / load, high = busy)
    zones   : one row per zone (index Zone_ID) with lat, lon, label, hub

To use another dataset, you do NOT touch the pipeline: you describe the dataset in `data_config.json`
(or write a new adapter function below). Without a config file, the defaults reproduce the original
behaviour exactly (event-level trips file clustered into zones).

Two adapters are provided:

  "events"      one row per event (trip / pickup) with timestamp + coordinates.
                Events are clustered into zones (KMeans) and counted per hour.
  "aggregated"  one row per (timestamp, site) with a measured value: vehicle counts, or speeds.
                Sites become zones; values are aggregated to hours.
                measure_kind "count": hourly sum.
                measure_kind "speed": hourly mean, converted to a congestion LOAD = max(free_flow_speed - speed, 0),
                so that "high = busy" holds for every dataset.

transfer_mode (how the Cotonou hubs relate to the zones):
  "profile_matching"  data come from elsewhere: each hub is matched to the zone with the most similar profile.
  "direct"            data come from Cotonou: each hub IS a zone (config key "zone_to_hub").
"""
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather import fetch_historical_rain  # noqa: E402

CONFIG_PATH = "data_config.json"

DEFAULT_CONFIG = {
    "adapter": "events",
    "path": "data/urban_mobility_raw_data.csv",
    "columns": {"time": "Date/Time", "lat": "Lat", "lon": "Lon"},
    "sample_n": 200000,
    "n_clusters": 15,
    "random_state": 42,
    "measure_kind": "count",
    "free_flow_speed": None,
    "missing": None,                 # "zero" | "interpolate" (aggregated adapter); default depends on measure_kind
    "zone_to_hub": {},
    "transfer_mode": "profile_matching",
    "weather_lat": None,
    "weather_lon": None,
    "weather_cache": "data/weather_cache.csv",
    "unit": None,                    # default depends on adapter / measure_kind
    "source_name": "Public mobility dataset (outside Cotonou)",
}


@dataclass
class CanonicalData:
    history: pd.DataFrame          # time, Zone_ID, Demand_Volume
    zones: pd.DataFrame            # index Zone_ID; lat, lon, label, hub
    zone_model: object             # fitted KMeans (events adapter) or None
    weather_lat: float
    weather_lon: float
    unit: str
    source_name: str
    transfer_mode: str
    measure_kind: str
    sparse_events: bool            # True when rows are counts of discrete events (empty slots are meaningful zeros)
    config: dict


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
def load_config(path=CONFIG_PATH):
    """Defaults, overridden by `data_config.json` if it exists."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        adapter = user.get("adapter", cfg["adapter"])
        for key, value in user.items():
            if key == "columns":
                base = cfg["columns"] if adapter == "events" else {}
                cfg["columns"] = {**base, **value}
            else:
                cfg[key] = value
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["adapter"] not in ("events", "aggregated"):
        raise SystemExit(f"data_config.json: unknown adapter '{cfg['adapter']}' (use 'events' or 'aggregated').")
    if cfg["transfer_mode"] not in ("profile_matching", "direct"):
        raise SystemExit("data_config.json: transfer_mode must be 'profile_matching' or 'direct'.")
    needed = ["time", "lat", "lon"] if cfg["adapter"] == "events" else ["time", "zone", "value"]
    missing = [k for k in needed if k not in cfg["columns"]]
    if missing:
        raise SystemExit(f"data_config.json: 'columns' must define {needed} (missing {missing}).")
    if cfg["adapter"] == "aggregated":
        if cfg["measure_kind"] not in ("count", "speed"):
            raise SystemExit("data_config.json: measure_kind must be 'count' or 'speed'.")
        if cfg["measure_kind"] == "speed" and not cfg.get("free_flow_speed"):
            raise SystemExit("data_config.json: 'free_flow_speed' is required when measure_kind is 'speed'.")
    if cfg["transfer_mode"] == "direct" and not cfg.get("zone_to_hub"):
        raise SystemExit("data_config.json: transfer_mode 'direct' requires 'zone_to_hub' "
                         "(maps each site label of your file to a Cotonou hub).")


def _default_unit(cfg):
    if cfg.get("unit"):
        return cfg["unit"]
    if cfg["adapter"] == "events":
        return "trips/h"
    return "km/h below free-flow" if cfg["measure_kind"] == "speed" else "vehicles/h"


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------
def _hourly_grid(long_df, zone_ids, fill):
    """long_df: time, Zone_ID, value  ->  complete hourly grid (time x zone) in long format."""
    first_day = long_df["time"].dt.normalize().min()
    last_day = long_df["time"].dt.normalize().max()
    full_index = pd.date_range(first_day, last_day + pd.Timedelta(hours=23), freq="h")
    wide = long_df.pivot_table(index="time", columns="Zone_ID", values="value", aggfunc="first")
    wide = wide.reindex(index=full_index, columns=zone_ids)
    n_missing = int(wide.isna().sum().sum())
    if fill == "zero":
        wide = wide.fillna(0)
    else:  # interpolate short gaps, then fall back to the zone median
        wide = wide.interpolate(limit=6, limit_direction="both").fillna(wide.median())
    wide.index.name = "time"
    history = wide.reset_index().melt(id_vars="time", var_name="Zone_ID", value_name="Demand_Volume")
    return history, n_missing


def load_events(cfg, zone_model=None):
    path = cfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Missing dataset file at '{path}'")
    cols = cfg["columns"]
    df = pd.read_csv(path, usecols=[cols["time"], cols["lat"], cols["lon"]])
    df = df.rename(columns={cols["time"]: "Date/Time", cols["lat"]: "Lat", cols["lon"]: "Lon"})
    df = df.sample(n=min(cfg["sample_n"], len(df)), random_state=cfg["random_state"])
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["time"] = df["Date/Time"].dt.floor("h")

    if zone_model is None:
        zone_model = KMeans(n_clusters=cfg["n_clusters"], random_state=cfg["random_state"], n_init=10)
        df["Zone_ID"] = zone_model.fit_predict(df[["Lat", "Lon"]])
    else:
        df["Zone_ID"] = zone_model.predict(df[["Lat", "Lon"]])
    zone_ids = list(range(zone_model.n_clusters))

    first_day = df["time"].dt.normalize().min()
    last_day = df["time"].dt.normalize().max()
    full_index = pd.date_range(first_day, last_day + pd.Timedelta(hours=23), freq="h")
    wide = (
        df.groupby(["time", "Zone_ID"]).size().unstack(fill_value=0)
        .reindex(index=full_index, columns=zone_ids, fill_value=0)
    )
    wide.index.name = "time"
    history = wide.reset_index().melt(id_vars="time", var_name="Zone_ID", value_name="Demand_Volume")

    zones = df.groupby("Zone_ID")[["Lat", "Lon"]].mean().rename(columns={"Lat": "lat", "Lon": "lon"})
    zones = zones.reindex(zone_ids)
    zones["label"] = [f"zone {z}" for z in zone_ids]
    zones["hub"] = np.nan

    w_lat = cfg["weather_lat"] if cfg["weather_lat"] is not None else round(float(df["Lat"].mean()), 3)
    w_lon = cfg["weather_lon"] if cfg["weather_lon"] is not None else round(float(df["Lon"].mean()), 3)
    return CanonicalData(history, zones, zone_model, float(w_lat), float(w_lon), _default_unit(cfg),
                         cfg["source_name"], cfg["transfer_mode"], "count", True, cfg)


def load_aggregated(cfg, zone_model=None):
    path = cfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Missing dataset file at '{path}'")
    cols = cfg["columns"]
    raw = pd.read_csv(path)
    for key in ("time", "zone", "value"):
        if cols[key] not in raw.columns:
            raise SystemExit(f"Column '{cols[key]}' (config 'columns.{key}') not found in {path}. "
                             f"Available: {list(raw.columns)}")

    time = pd.to_datetime(raw[cols["time"]], errors="coerce")
    value = pd.to_numeric(raw[cols["value"]], errors="coerce")
    ok = time.notna() & value.notna()
    if (~ok).any():
        print(f"Attention : {int((~ok).sum())} ligne(s) ignorée(s) (date ou valeur invalide).")
    d = pd.DataFrame({"time": time[ok].dt.floor("h"), "label": raw.loc[ok, cols["zone"]].astype(str),
                      "value": value[ok]})
    labels = sorted(d["label"].unique())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    d["Zone_ID"] = d["label"].map(label_to_id)

    kind = cfg["measure_kind"]
    grouped = d.groupby(["time", "Zone_ID"])["value"]
    long_df = (grouped.sum() if kind == "count" else grouped.mean()).reset_index()
    if kind == "speed":
        long_df["value"] = (float(cfg["free_flow_speed"]) - long_df["value"]).clip(lower=0)

    fill = cfg.get("missing") or ("zero" if kind == "count" else "interpolate")
    history, n_missing = _hourly_grid(long_df, list(range(len(labels))), fill)
    if n_missing:
        print(f"Attention : {n_missing} créneau(x) zone-heure sans mesure complétés ({fill}).")

    zones = pd.DataFrame({"label": labels}, index=pd.Index(range(len(labels)), name="Zone_ID"))
    if cols.get("lat") in raw.columns and cols.get("lon") in raw.columns:
        ll = raw.loc[ok].assign(label=d["label"], lat=pd.to_numeric(raw.loc[ok, cols["lat"]], errors="coerce"),
                                lon=pd.to_numeric(raw.loc[ok, cols["lon"]], errors="coerce"))
        ll = ll.groupby("label")[["lat", "lon"]].mean().reindex(labels)
        zones["lat"], zones["lon"] = ll["lat"].to_numpy(), ll["lon"].to_numpy()
    else:
        zones["lat"], zones["lon"] = np.nan, np.nan
    zones["hub"] = zones["label"].map(cfg.get("zone_to_hub", {}))

    if cfg["weather_lat"] is None or cfg["weather_lon"] is None:
        if zones["lat"].notna().any():
            w_lat, w_lon = round(float(zones["lat"].mean()), 3), round(float(zones["lon"].mean()), 3)
        else:
            raise SystemExit("data_config.json: set 'weather_lat' and 'weather_lon' (the file has no coordinates).")
    else:
        w_lat, w_lon = float(cfg["weather_lat"]), float(cfg["weather_lon"])
    return CanonicalData(history, zones, None, float(w_lat), float(w_lon), _default_unit(cfg),
                         cfg["source_name"], cfg["transfer_mode"], kind, False, cfg)


ADAPTERS = {"events": load_events, "aggregated": load_aggregated}


def load_canonical(cfg=None, zone_model=None):
    """Load the dataset described by the config into the canonical structure."""
    cfg = cfg or load_config()
    return ADAPTERS[cfg["adapter"]](cfg, zone_model)


# ----------------------------------------------------------------------------
# Weather
# ----------------------------------------------------------------------------
def attach_rain(data, cache_path=None):
    """
    Add real hourly rainfall (Open-Meteo) to the history. Returns (history_with_rain, weather_df).
    Needs internet once; results are cached in `cache_path`.
    """
    first = data.history["time"].min().date()
    last = data.history["time"].max().date()
    print(f"Météo réelle : lieu ({data.weather_lat}, {data.weather_lon}), du {first} au {last}")
    try:
        weather = fetch_historical_rain(data.weather_lat, data.weather_lon, first, last, cache_path=cache_path)
    except Exception as exc:
        raise SystemExit(
            f"Impossible de récupérer la météo ({exc}).\n"
            f"Vérifie ta connexion internet : le script en a besoin une première fois "
            f"(les données sont ensuite mises en cache dans {cache_path})."
        )
    history = data.history.merge(weather[["time", "Rain_mm"]], on="time", how="left")
    missing = int(history["Rain_mm"].isna().sum())
    if missing:
        print(f"Attention : {missing} lignes sans donnée météo (pluie fixée à 0).")
    history["Rain_mm"] = history["Rain_mm"].fillna(0.0)
    return history, weather