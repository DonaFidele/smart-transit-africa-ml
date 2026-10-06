"""
SmartTransit Africa - script d'entraînement (v2)

Démarche : transfert de méthode.
Le modèle est entraîné sur un jeu de données public de mobilité (type Uber / New York)
faute de données locales à Cotonou. Chaque site de Cotonou est ensuite associé à la zone
du jeu d'entraînement dont le PROFIL de demande lui ressemble le plus (et non plus à un
numéro de zone choisi à la main).

Sorties (dossier models/) :
  - traffic_rf_model.pkl     : classifieur Random Forest (Hour, Day_of_Week, Zone_ID)
  - kmeans_zones.pkl         : clustering spatial
  - zone_profiles.csv        : profil de demande de chaque zone
  - hub_zone_mapping.json    : association site de Cotonou -> zone + justification
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# --------------------------------------------------------------------------
# Sites de Cotonou et profil recherché (valeurs entre 0 et 1 = rang percentile
# parmi les 15 zones : demande totale, part des heures de pointe, part de nuit)
# --------------------------------------------------------------------------
COTONOU_HUBS = {
    "Échangeur de Godomey (West Gateway)": {
        "lat": 6.3811, "lon": 2.3522,
        "archetype": "Porte d'entrée / flux pendulaires",
        "target": {"demand": 0.6, "peak_share": 0.9, "night_share": 0.2},
        "why": "Flux dominé par les trajets domicile-travail : forte part d'heures de pointe, peu d'activité nocturne.",
    },
    "Grand Marché de Dantokpa (Commercial Hub)": {
        "lat": 6.3708, "lon": 2.4344,
        "archetype": "Pôle commercial",
        "target": {"demand": 1.0, "peak_share": 0.6, "night_share": 0.2},
        "why": "Plus forte demande du réseau, activité concentrée en journée.",
    },
    "Carrefour Vèdoko (Central Junction)": {
        "lat": 6.3754, "lon": 2.3881,
        "archetype": "Carrefour central",
        "target": {"demand": 0.8, "peak_share": 0.8, "night_share": 0.5},
        "why": "Forte demande, pointes marquées et activité soutenue en soirée.",
    },
    "Carrefour Cadjehoun (Avenue Jean-Paul II)": {
        "lat": 6.3575, "lon": 2.3980,
        "archetype": "Axe urbain mixte",
        "target": {"demand": 0.6, "peak_share": 0.6, "night_share": 0.6},
        "why": "Demande moyenne, répartition équilibrée sur la journée avec une vie nocturne notable.",
    },
    "Zone Portuaire / Akpakpa (Logistics Hub)": {
        "lat": 6.3650, "lon": 2.4490,
        "archetype": "Hub logistique",
        "target": {"demand": 0.4, "peak_share": 0.3, "night_share": 0.1},
        "why": "Demande modérée, activité diurne régulière sans pointes pendulaires, très peu de nuit.",
    },
}

PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4]
FEATURES = ["Hour", "Day_of_Week", "Zone_ID"]


def build_zone_profiles(df):
    """Profil de demande de chaque zone."""
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
    """Affectation optimale (1 site <-> 1 zone) par distance entre profils."""
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

    # Chargement et échantillonnage
    df = pd.read_csv(data_path, usecols=["Date/Time", "Lat", "Lon"])
    df = df.sample(n=min(200000, len(df)), random_state=42)

    # Variables temporelles
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["Date"] = df["Date/Time"].dt.date
    df["Hour"] = df["Date/Time"].dt.hour
    df["Day_of_Week"] = df["Date/Time"].dt.dayofweek

    # Clustering spatial
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df["Zone_ID"] = kmeans.fit_predict(df[["Lat", "Lon"]])

    # Profils de zones + association des sites de Cotonou
    profiles = build_zone_profiles(df)
    mapping = map_hubs_to_zones(profiles)

    # Cible : demande agrégée par date / heure / zone
    agg = (
        df.groupby(["Date", "Day_of_Week", "Hour", "Zone_ID"])
        .size()
        .reset_index(name="Demand_Volume")
    )

    # Découpage CHRONOLOGIQUE (80 % premiers jours = train, 20 % derniers = test)
    dates = sorted(agg["Date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = agg[agg["Date"] < cut].copy()
    test = agg[agg["Date"] >= cut].copy()

    # Seuil de saturation calculé sur le train uniquement (pas de fuite)
    saturation_threshold = train["Demand_Volume"].quantile(0.75)
    train["Risk"] = (train["Demand_Volume"] > saturation_threshold).astype(int)
    test["Risk"] = (test["Demand_Volume"] > saturation_threshold).astype(int)

    X_train, y_train = train[FEATURES], train["Risk"]
    X_test, y_test = test[FEATURES], test["Risk"]

    # Modèle
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

    print(f"Jours train : {train['Date'].nunique()} | jours test : {test['Date'].nunique()}")
    print(f"F1 modèle (test chronologique) : {f1_score(y_test, predictions):.2f}")
    print(f"F1 référence naïve             : {f1_score(y_test, baseline_pred):.2f}")
    print("\nClassification Report (modèle) :")
    print(classification_report(y_test, predictions))

    print("\nAssociation sites de Cotonou -> zones d'entraînement :")
    for name, m in mapping.items():
        print(f"  {name} -> zone {m['zone_id']} (distance de profil {m['distance']})")

    # Export
    os.makedirs("models", exist_ok=True)
    with open("models/traffic_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/kmeans_zones.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    profiles.round(4).to_csv("models/zone_profiles.csv")
    with open("models/hub_zone_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()