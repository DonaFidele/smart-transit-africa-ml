"""
Variables de prévision (retards) partagées entre l'entraînement et l'application.

Pour un horizon h (en heures), on n'utilise QUE ce qui est connu h heures avant la cible :
  lag_h               demande observée à t-h
  lag_h1, lag_h2      demande à t-h-1 et t-h-2 (tendance récente)
  roll24              demande moyenne des 24 h se terminant à t-h
  same_hour_yesterday demande à la même heure la veille (t-24)
Variables connues pour l'heure cible : heure, jour de semaine, zone, pluie (prévue).
"""
import pandas as pd

BASE_FEATURES = ["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]
LAG_FEATURES = ["lag_h", "lag_h1", "lag_h2", "roll24", "same_hour_yesterday"]
FEATURES = BASE_FEATURES + LAG_FEATURES

# Relative version: demand is expressed as a ratio to the zone's usual level at that hour
REL_LAG_FEATURES = ["rel_lag_h", "rel_lag_h1", "rel_lag_h2", "rel_roll24", "rel_same_hour_yesterday"]
FEATURES_REL = BASE_FEATURES + REL_LAG_FEATURES


def build_features(history, horizon):
    """
    history : grille horaire COMPLÈTE avec colonnes time, Zone_ID, Demand_Volume, Rain_mm.
    Retourne une ligne par (time, zone) avec les variables ci-dessus.
    """
    frames = []
    for zone, g in history.sort_values("time").groupby("Zone_ID"):
        g = g.reset_index(drop=True)
        s = g["Demand_Volume"]
        frames.append(pd.DataFrame({
            "time": g["time"],
            "Zone_ID": zone,
            "Hour": g["time"].dt.hour,
            "Day_of_Week": g["time"].dt.dayofweek,
            "Rain_mm": g["Rain_mm"],
            "Demand_Volume": s,
            "lag_h": s.shift(horizon),
            "lag_h1": s.shift(horizon + 1),
            "lag_h2": s.shift(horizon + 2),
            "roll24": s.shift(horizon).rolling(24).mean(),
            "same_hour_yesterday": s.shift(24),
        }))
    return pd.concat(frames, ignore_index=True)


def build_relative_features(history, horizon, baseline):
    """
    Same idea as build_features, but demand is replaced by a RATIO to the zone's usual level:
        ratio = (demand + 1) / (usual demand of this zone at this hour + 1)
    `baseline` is a Series indexed by (Zone_ID, Hour) computed on TRAINING data only, so that
    nothing from the test period leaks into the features or the target.
    The target "surge" is then a deviation from the zone's own habit, not an absolute level.
    """
    h = history.copy()
    key = pd.MultiIndex.from_arrays([h["Zone_ID"], h["time"].dt.hour])
    usual = baseline.reindex(key).to_numpy()
    h["usual"] = pd.Series(usual, index=h.index).fillna(float(baseline.mean()))
    h["ratio"] = (h["Demand_Volume"] + 1) / (h["usual"] + 1)

    frames = []
    for zone, g in h.sort_values("time").groupby("Zone_ID"):
        g = g.reset_index(drop=True)
        r = g["ratio"]
        frames.append(pd.DataFrame({
            "time": g["time"],
            "Zone_ID": zone,
            "Hour": g["time"].dt.hour,
            "Day_of_Week": g["time"].dt.dayofweek,
            "Rain_mm": g["Rain_mm"],
            "Demand_Volume": g["Demand_Volume"],
            "ratio": r,
            "rel_lag_h": r.shift(horizon),
            "rel_lag_h1": r.shift(horizon + 1),
            "rel_lag_h2": r.shift(horizon + 2),
            "rel_roll24": r.shift(horizon).rolling(24).mean(),
            "rel_same_hour_yesterday": r.shift(24),
        }))
    return pd.concat(frames, ignore_index=True)