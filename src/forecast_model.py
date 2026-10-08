"""
SmartTransit Africa - modèle de PRÉVISION avec retards (horizons 1 h et 24 h)

À lancer APRÈS src/model.py (il réutilise models/kmeans_zones.pkl et le cache météo) :
    python src/forecast_model.py                     # cible absolue (par défaut)
    python src/forecast_model.py --target relative   # cible relative : pic au-dessus du niveau habituel de la zone

Cible absolue : « saturé » = demande dans les 25 % les plus hauts de tous les créneaux.
Cible relative : « pic » = rapport demande / niveau habituel de la zone à cette heure dans les 25 % les plus hauts.
Le niveau habituel est calculé sur la période d'ENTRAÎNEMENT uniquement (pas de fuite).

Sorties (models/) :
  forecast_rf_h1.pkl, forecast_rf_h24.pkl, forecast_history.csv, forecast_meta.json
  forecast_baseline.csv (cible relative seulement : niveau habituel par zone et heure)
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_adapter import attach_rain, load_canonical, load_config  # noqa: E402
from forecast_features import (  # noqa: E402
    FEATURES, FEATURES_REL, LAG_FEATURES, REL_LAG_FEATURES, build_features, build_relative_features,
)

HORIZONS = (1, 24)


def parse_target(argv):
    """Returns 'absolute' (default) or 'relative' from the command line (--target relative)."""
    if "--target" in argv:
        value = argv[argv.index("--target") + 1] if argv.index("--target") + 1 < len(argv) else ""
        if value not in ("absolute", "relative"):
            raise SystemExit("--target doit valoir 'absolute' ou 'relative'.")
        return value
    return "absolute"


def scores(y_true, y_pred):
    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
    }


def main():
    target = parse_target(sys.argv)
    cfg = load_config()

    # Événements : on réutilise EXACTEMENT les zones apprises par src/model.py (même clustering)
    zone_model = None
    if cfg["adapter"] == "events":
        if not os.path.exists("models/kmeans_zones.pkl"):
            raise SystemExit("models/kmeans_zones.pkl introuvable : lance d'abord `python src/model.py`.")
        with open("models/kmeans_zones.pkl", "rb") as f:
            zone_model = pickle.load(f)
        if zone_model is None:
            raise SystemExit("kmeans_zones.pkl est vide : relance `python src/model.py` avec la même config.")

    data = load_canonical(cfg, zone_model=zone_model)
    history, _ = attach_rain(data, cache_path=cfg["weather_cache"])
    print(f"Source : {data.source_name} | adaptateur : {cfg['adapter']} | mesure : {data.measure_kind} ({data.unit})")

    # Découpage chronologique : 80 % des premiers jours = entraînement
    dates = sorted(history["time"].dt.date.unique())
    cut_ts = pd.Timestamp(dates[int(len(dates) * 0.8)])
    train_hist = history[history["time"] < cut_ts]
    print(f"Cible : {target} | période : {dates[0]} -> {dates[-1]} | début du test : {cut_ts.date()}")

    os.makedirs("models", exist_ok=True)
    if target == "absolute":
        threshold = float(train_hist["Demand_Volume"].quantile(0.75))
        value, cols, lag_now, lag_yday = "Demand_Volume", FEATURES, "lag_h", "same_hour_yesterday"
        print(f"Seuil de saturation (75e percentile, grille complète) : {threshold:.1f} {data.unit}")
    else:
        usual = (train_hist.groupby(["Zone_ID", train_hist["time"].dt.hour.rename("Hour")])["Demand_Volume"]
                 .mean().rename("usual"))
        usual.reset_index().to_csv("models/forecast_baseline.csv", index=False)
        keys = pd.MultiIndex.from_arrays([train_hist["Zone_ID"], train_hist["time"].dt.hour])
        ratio_train = (train_hist["Demand_Volume"].to_numpy() + 1) / (usual.reindex(keys).to_numpy() + 1)
        threshold = float(np.nanquantile(ratio_train, 0.75))
        value, cols, lag_now, lag_yday = "ratio", FEATURES_REL, "rel_lag_h", "rel_same_hour_yesterday"
        print(f"Seuil de pic (75e percentile du rapport demande / niveau habituel) : {threshold:.2f}")

    meta = {
        "target": target,
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "cut_date": str(cut_ts.date()),
        "threshold": round(threshold, 3),
        "unit": data.unit,
        "source_name": data.source_name,
        "transfer_mode": data.transfer_mode,
        "features": cols,
        "horizons": {},
    }

    for h in HORIZONS:
        if target == "absolute":
            feats = build_features(history, h).dropna(subset=LAG_FEATURES)
        else:
            feats = build_relative_features(history, h, usual).dropna(subset=REL_LAG_FEATURES)
        feats["y"] = (feats[value] > threshold).astype(int)
        train = feats[feats["time"] < cut_ts]
        test = feats[feats["time"] >= cut_ts]

        rf = RandomForestClassifier(
            n_estimators=150, max_depth=14, random_state=42, class_weight="balanced", n_jobs=-1
        )
        rf.fit(train[cols], train["y"])
        pred_model = rf.predict(test[cols])

        # Références : persistance, même heure la veille, taux historique par (zone, heure, jour), toujours alerter
        pred_persist = (test[lag_now] > threshold).astype(int)
        pred_seasonal = (test[lag_yday] > threshold).astype(int)
        lookup = train.groupby(["Zone_ID", "Hour", "Day_of_Week"])["y"].mean().rename("hist").reset_index()
        merged = test.merge(lookup, on=["Zone_ID", "Hour", "Day_of_Week"], how="left")
        pred_static = (merged["hist"].fillna(0) > 0.5).astype(int)

        res = {
            "n_test": int(len(test)),
            "model": scores(test["y"], pred_model),
            "static": scores(test["y"], pred_static),
            "persistence": scores(test["y"], pred_persist),
            "same_hour_yesterday": scores(test["y"], pred_seasonal),
            "no_skill": scores(test["y"], np.ones(len(test), dtype=int)),
            "importances": {k: round(float(v), 3) for k, v in zip(cols, rf.feature_importances_)},
        }
        meta["horizons"][str(h)] = res

        print(f"\n=== Horizon {h} h  (test : {len(test)} lignes) ===")
        for key, label in [("model", "Modèle avec retards"), ("static", "Taux historique (sans retards)"),
                           ("persistence", "Persistance (t-h)"), ("same_hour_yesterday", "Même heure la veille"),
                           ("no_skill", "Toujours alerter (sans compétence)")]:
            r = res[key]
            print(f"  {label:36s} F1={r['f1']:.2f}  précision={r['precision']:.2f}  rappel={r['recall']:.2f}")
        print("  Importances :", res["importances"])

        with open(f"models/forecast_rf_h{h}.pkl", "wb") as f:
            pickle.dump(rf, f)

    history["is_test"] = (history["time"] >= cut_ts).astype(int)
    history.to_csv("models/forecast_history.csv", index=False)
    with open("models/forecast_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nFichiers écrits dans models/.")


if __name__ == "__main__":
    main()