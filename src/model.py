"""
SmartTransit Africa - script d'entraînement (v4 : couche d'adaptation des données)

Le jeu de données est décrit dans `data_config.json` (facultatif). Sans ce fichier, le comportement
d'origine est conservé : un fichier d'événements (Date/Time, Lat, Lon) regroupé en 15 zones.

Deux modes de lien avec Cotonou (config "transfer_mode") :
  profile_matching : données d'ailleurs -> chaque site de Cotonou est associé à la zone au profil le plus proche.
  direct           : données de Cotonou -> chaque site EST une zone (config "zone_to_hub").

La pluie est la pluie RÉELLE (Open-Meteo) du lieu et des dates des données, variable continue (mm/h).

Lancer depuis la racine du projet :  python src/model.py

Sorties (dossier models/) :
  traffic_rf_model.pkl, kmeans_zones.pkl (None si pas de clustering), zone_profiles.csv,
  hub_zone_mapping.json, training_meta.json
"""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather import HEAVY_RAIN_MM  # noqa: E402
from data_adapter import attach_rain, load_canonical, load_config  # noqa: E402
from field_validation import resolve_site  # noqa: E402

COTONOU_HUBS = {
    "Échangeur de Godomey (West Gateway)": {
        "lat": 6.3811, "lon": 2.3522,
        "archetype": "Gateway / commuter flows",
        "target": {"demand": 0.6, "peak_share": 0.9, "night_share": 0.2},
        "why": "Flow dominated by home-work trips: high share of peak hours, little night activity.",
    },
    "Grand Marché de Dantokpa (Commercial Hub)": {
        "lat": 6.3708, "lon": 2.4344,
        "archetype": "Commercial hub",
        "target": {"demand": 1.0, "peak_share": 0.6, "night_share": 0.2},
        "why": "Highest demand in the network, activity concentrated in the daytime.",
    },
    "Carrefour Vèdoko (Central Junction)": {
        "lat": 6.3754, "lon": 2.3881,
        "archetype": "Central junction",
        "target": {"demand": 0.8, "peak_share": 0.8, "night_share": 0.5},
        "why": "High demand, marked peaks and sustained evening activity.",
    },
    "Carrefour Cadjehoun (Avenue Jean-Paul II)": {
        "lat": 6.3575, "lon": 2.3980,
        "archetype": "Mixed urban axis",
        "target": {"demand": 0.6, "peak_share": 0.6, "night_share": 0.6},
        "why": "Medium demand, balanced across the day with notable nightlife.",
    },
    "Zone Portuaire / Akpakpa (Logistics Hub)": {
        "lat": 6.3650, "lon": 2.4490,
        "archetype": "Logistics hub",
        "target": {"demand": 0.4, "peak_share": 0.3, "night_share": 0.1},
        "why": "Moderate demand, steady daytime activity without commuter peaks, very little at night.",
    },
}

PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4]
FEATURES = ["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]


def build_zone_profiles(history, zones):
    """Profil de chaque zone (volume, part des heures de pointe / de nuit / du week-end) + rangs."""
    h = history.assign(hour=history["time"].dt.hour, dow=history["time"].dt.dayofweek)
    total = h.groupby("Zone_ID")["Demand_Volume"].sum()
    safe = total.replace(0, np.nan)

    def share(mask):
        return h[mask].groupby("Zone_ID")["Demand_Volume"].sum().reindex(total.index, fill_value=0) / safe

    profiles = pd.DataFrame({
        "demand": total,
        "peak_share": share(h["hour"].isin(PEAK_HOURS)),
        "night_share": share(h["hour"].isin(NIGHT_HOURS)),
        "weekend_share": share(h["dow"] >= 5),
        "lat": zones["lat"].reindex(total.index),
        "lon": zones["lon"].reindex(total.index),
    })
    profiles.index.name = "Zone_ID"
    for col in ["demand", "peak_share", "night_share"]:
        profiles[f"{col}_rank"] = profiles[col].rank(pct=True)
    return profiles


def _hub_record(name, z, profiles, distance):
    hub = COTONOU_HUBS[name]
    return {
        "lat": hub["lat"],
        "lon": hub["lon"],
        "archetype": hub["archetype"],
        "criteria": hub["why"],
        "target": hub["target"],
        "zone_id": int(z),
        "distance": round(float(distance), 3),
        "zone_demand_rank": round(float(profiles.loc[z, "demand_rank"]), 2),
        "zone_peak_share": round(float(profiles.loc[z, "peak_share"]), 3),
        "zone_night_share": round(float(profiles.loc[z, "night_share"]), 3),
    }


def map_hubs_to_zones(profiles):
    """Mode profile_matching : affectation optimale un-pour-un par distance entre profils."""
    hub_names = list(COTONOU_HUBS.keys())
    zone_ids = list(profiles.index)
    cost = np.zeros((len(hub_names), len(zone_ids)))
    for i, name in enumerate(hub_names):
        t = COTONOU_HUBS[name]["target"]
        for j, z in enumerate(zone_ids):
            cost[i, j] = (
                (t["demand"] - profiles.loc[z, "demand_rank"]) ** 2
                + (t["peak_share"] - profiles.loc[z, "peak_share_rank"]) ** 2
                + (t["night_share"] - profiles.loc[z, "night_share_rank"]) ** 2
            ) ** 0.5
    rows, cols = linear_sum_assignment(cost)
    return {hub_names[i]: _hub_record(hub_names[i], zone_ids[j], profiles, cost[i, j]) for i, j in zip(rows, cols)}


def map_hubs_direct(profiles, zones):
    """Mode direct : chaque site de Cotonou est une zone mesurée sur place (config zone_to_hub)."""
    hub_names = list(COTONOU_HUBS.keys())
    hub_to_zone = {}
    for z, value in zones["hub"].items():
        if isinstance(value, str):
            resolved = resolve_site(value, hub_names)
            if resolved is None:
                raise SystemExit(f"zone_to_hub : '{value}' ne correspond à aucun site connu : {hub_names}")
            hub_to_zone[resolved] = z
    missing = [h for h in hub_names if h not in hub_to_zone]
    if missing:
        raise SystemExit(f"zone_to_hub : aucun site du fichier n'est associé à : {missing}")
    return {name: _hub_record(name, hub_to_zone[name], profiles, 0.0) for name in hub_names}


def main():
    cfg = load_config()
    data = load_canonical(cfg)
    history, weather = attach_rain(data, cache_path=cfg["weather_cache"])
    print(f"Source : {data.source_name} | adaptateur : {cfg['adapter']} | mode : {data.transfer_mode} | "
          f"mesure : {data.measure_kind} ({data.unit})")

    profiles = build_zone_profiles(history, data.zones)
    if data.transfer_mode == "direct":
        mapping = map_hubs_direct(profiles, data.zones)
    else:
        mapping = map_hubs_to_zones(profiles)

    # Table de travail : une ligne par créneau zone-heure
    agg = history.assign(
        Date=history["time"].dt.date,
        Day_of_Week=history["time"].dt.dayofweek,
        Hour=history["time"].dt.hour,
    )
    if data.sparse_events:  # événements discrets : seuls les créneaux non vides comptent (comportement d'origine)
        agg = agg[agg["Demand_Volume"] > 0]
    agg = (agg[["Date", "Day_of_Week", "Hour", "Zone_ID", "Demand_Volume", "Rain_mm"]]
           .sort_values(["Date", "Day_of_Week", "Hour", "Zone_ID"]).reset_index(drop=True))

    d_start, d_end = min(agg["Date"]), max(agg["Date"])
    rain_hours_share = float((weather["Rain_mm"] > 0.1).mean())
    heavy_hours = int((weather["Rain_mm"] >= HEAVY_RAIN_MM).sum())
    print(f"Heures avec pluie : {rain_hours_share:.0%} | heures de forte pluie (≥ {HEAVY_RAIN_MM} mm/h) : {heavy_hours}")

    # ---------------- Découpage chronologique ----------------
    dates = sorted(agg["Date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = agg[agg["Date"] < cut].copy()
    test = agg[agg["Date"] >= cut].copy()

    saturation_threshold = train["Demand_Volume"].quantile(0.75)
    train["Risk"] = (train["Demand_Volume"] > saturation_threshold).astype(int)
    test["Risk"] = (test["Demand_Volume"] > saturation_threshold).astype(int)

    X_train, y_train = train[FEATURES], train["Risk"]
    X_test, y_test = test[FEATURES], test["Risk"]

    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=14, random_state=42, class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)

    # Référence naïve : taux historique par (Zone, Heure, Jour de semaine)
    lookup = (
        train.groupby(["Zone_ID", "Hour", "Day_of_Week"])["Risk"].mean().rename("hist").reset_index()
    )
    merged = test.merge(lookup, on=["Zone_ID", "Hour", "Day_of_Week"], how="left")
    baseline_pred = (merged["hist"].fillna(0) > 0.5).astype(int)

    f1_model = float(f1_score(y_test, predictions))
    f1_base = float(f1_score(y_test, baseline_pred))

    # Effet moyen appris de la pluie : même test, pluie = 0 puis pluie forte
    probe0 = X_test.copy()
    probe0["Rain_mm"] = 0.0
    probe1 = X_test.copy()
    probe1["Rain_mm"] = HEAVY_RAIN_MM
    rain_effect = float(
        (rf_model.predict_proba(probe1)[:, 1] - rf_model.predict_proba(probe0)[:, 1]).mean() * 100
    )

    print(f"\nJours train : {train['Date'].nunique()} | jours test : {test['Date'].nunique()}")
    print(f"F1 modèle (test chronologique) : {f1_model:.2f}")
    print(f"F1 référence naïve             : {f1_base:.2f}")
    print(f"Effet moyen appris d'une forte pluie : {rain_effect:+.1f} points de risque")
    print("Importances :", dict(zip(FEATURES, np.round(rf_model.feature_importances_, 3))))
    print("\nClassification Report (modèle) :")
    print(classification_report(y_test, predictions))

    label = "zones locales" if data.transfer_mode == "direct" else "zones d'entraînement"
    print(f"\nAssociation sites de Cotonou -> {label} :")
    for name, m in mapping.items():
        print(f"  {name} -> zone {m['zone_id']} (distance de profil {m['distance']})")

    # ---------------- Export ----------------
    os.makedirs("models", exist_ok=True)
    with open("models/traffic_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/kmeans_zones.pkl", "wb") as f:
        pickle.dump(data.zone_model, f)
    profiles.round(4).to_csv("models/zone_profiles.csv")
    with open("models/hub_zone_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    meta = {
        "weather_source": "Open-Meteo (archive)",
        "weather_lat": data.weather_lat,
        "weather_lon": data.weather_lon,
        "date_start": str(d_start),
        "date_end": str(d_end),
        "heavy_rain_mm": HEAVY_RAIN_MM,
        "rain_hours_share": round(rain_hours_share, 3),
        "heavy_rain_hours": heavy_hours,
        "rain_effect_points": round(rain_effect, 2),
        "f1_model": round(f1_model, 3),
        "f1_baseline": round(f1_base, 3),
        "n_train_days": int(train["Date"].nunique()),
        "n_test_days": int(test["Date"].nunique()),
        "transfer_mode": data.transfer_mode,
        "source_name": data.source_name,
        "unit": data.unit,
        "measure_kind": data.measure_kind,
    }
    with open("models/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()