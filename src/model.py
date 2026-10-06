"""
SmartTransit Africa - script d'entraînement (v3 : météo réelle)

Démarche : transfert de méthode.
Le modèle est entraîné sur un jeu public de mobilité (hors Cotonou). La pluie n'est plus
simulée : on récupère la pluie horaire RÉELLE (Open-Meteo) pour les dates et le lieu du jeu
de données, et le modèle l'utilise comme variable continue (mm/h).

Lancer depuis la racine du projet :  python src/model.py

Sorties (dossier models/) :
  traffic_rf_model.pkl, kmeans_zones.pkl, zone_profiles.csv,
  hub_zone_mapping.json, training_meta.json
"""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather import fetch_historical_rain, HEAVY_RAIN_MM  # noqa: E402

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


def build_zone_profiles(df):
    profiles = df.groupby("Zone_ID").agg(
        demand=("Hour", "size"),
        peak_share=("Hour", lambda h: h.isin(PEAK_HOURS).mean()),
        night_share=("Hour", lambda h: h.isin(NIGHT_HOURS).mean()),
        weekend_share=("Day_of_Week", lambda d: (d >= 5).mean()),
        lat=("Lat", "mean"),
        lon=("Lon", "mean"),
    )
    for col in ["demand", "peak_share", "night_share"]:
        profiles[f"{col}_rank"] = profiles[col].rank(pct=True)
    return profiles


def map_hubs_to_zones(profiles):
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

    mapping = {}
    for i, j in zip(rows, cols):
        name = hub_names[i]
        z = zone_ids[j]
        hub = COTONOU_HUBS[name]
        mapping[name] = {
            "lat": hub["lat"],
            "lon": hub["lon"],
            "archetype": hub["archetype"],
            "criteria": hub["why"],
            "target": hub["target"],
            "zone_id": int(z),
            "distance": round(float(cost[i, j]), 3),
            "zone_demand_rank": round(float(profiles.loc[z, "demand_rank"]), 2),
            "zone_peak_share": round(float(profiles.loc[z, "peak_share"]), 3),
            "zone_night_share": round(float(profiles.loc[z, "night_share"]), 3),
        }
    return mapping


def main():
    data_path = "data/urban_mobility_raw_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Missing dataset file at '{data_path}'")

    df = pd.read_csv(data_path, usecols=["Date/Time", "Lat", "Lon"])
    df = df.sample(n=min(200000, len(df)), random_state=42)

    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["Date"] = df["Date/Time"].dt.date
    df["Hour"] = df["Date/Time"].dt.hour
    df["Day_of_Week"] = df["Date/Time"].dt.dayofweek

    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df["Zone_ID"] = kmeans.fit_predict(df[["Lat", "Lon"]])

    profiles = build_zone_profiles(df)
    mapping = map_hubs_to_zones(profiles)

    agg = (
        df.groupby(["Date", "Day_of_Week", "Hour", "Zone_ID"])
        .size()
        .reset_index(name="Demand_Volume")
    )

    # ---------------- Météo réelle (Open-Meteo) ----------------
    w_lat, w_lon = round(float(df["Lat"].mean()), 3), round(float(df["Lon"].mean()), 3)
    d_start, d_end = min(agg["Date"]), max(agg["Date"])
    print(f"Météo réelle : lieu ({w_lat}, {w_lon}), du {d_start} au {d_end}")
    try:
        weather = fetch_historical_rain(
            w_lat, w_lon, d_start, d_end, cache_path="data/weather_cache.csv"
        )
    except Exception as exc:
        raise SystemExit(
            f"Impossible de récupérer la météo ({exc}).\n"
            "Vérifie ta connexion internet : le script en a besoin une première fois "
            "(les données sont ensuite mises en cache dans data/weather_cache.csv)."
        )

    agg = agg.merge(weather[["Date", "Hour", "Rain_mm"]], on=["Date", "Hour"], how="left")
    missing = int(agg["Rain_mm"].isna().sum())
    if missing:
        print(f"Attention : {missing} lignes sans donnée météo (pluie fixée à 0).")
    agg["Rain_mm"] = agg["Rain_mm"].fillna(0.0)

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

    print("\nAssociation sites de Cotonou -> zones d'entraînement :")
    for name, m in mapping.items():
        print(f"  {name} -> zone {m['zone_id']} (distance de profil {m['distance']})")

    # ---------------- Export ----------------
    os.makedirs("models", exist_ok=True)
    with open("models/traffic_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/kmeans_zones.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    profiles.round(4).to_csv("models/zone_profiles.csv")
    with open("models/hub_zone_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    meta = {
        "weather_source": "Open-Meteo (archive)",
        "weather_lat": w_lat,
        "weather_lon": w_lon,
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
    }
    with open("models/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()