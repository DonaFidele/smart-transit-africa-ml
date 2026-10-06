"""
Validation locale à partir de comptages réalisés sur le terrain à Cotonou.

Deux questions :
  1. Le profil observé de chaque site (niveau de demande, pointes, soirée/nuit) ressemble-t-il
     à l'archétype supposé lors du transfert ?
  2. Les probabilités du modèle sont-elles cohérentes avec la saturation réellement observée ?
"""
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED_COLUMNS = [
    "date", "site", "heure_debut", "heure_fin",
    "motos", "voitures", "bus_minibus", "camions",
    "pluie_mm", "saturation_observee",
]
OPTIONAL_COLUMNS = ["observateur", "notes"]
VEHICLE_COLUMNS = ["motos", "voitures", "bus_minibus", "camions"]

PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4]
SITE_KEYWORDS = ["godomey", "dantokpa", "vedoko", "cadjehoun", "portuaire", "akpakpa"]


def _norm(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def resolve_site(value, hub_names):
    """Accepte le nom complet ou un mot-clé (ex. « Dantokpa », « vedoko »)."""
    v = _norm(value)
    for name in hub_names:
        if v == _norm(name):
            return name
    for key in SITE_KEYWORDS:
        if key in v:
            for name in hub_names:
                if key in _norm(name):
                    return name
    return None


def template_csv():
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS).to_csv(index=False)


def load_counts(raw, hubs):
    """Nettoie le tableau de comptages. Retourne (DataFrame ou None, liste de problèmes)."""
    problems = []
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, [f"Colonnes manquantes : {', '.join(missing)}"]

    hub_names = list(hubs.keys())
    df["site_resolved"] = df["site"].map(lambda v: resolve_site(v, hub_names))
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    start = pd.to_datetime(df["heure_debut"].astype(str).str.strip(), format="%H:%M", errors="coerce")
    end = pd.to_datetime(df["heure_fin"].astype(str).str.strip(), format="%H:%M", errors="coerce")
    df["duration_min"] = (end - start).dt.total_seconds() / 60
    df["hour"] = start.dt.hour

    for col in VEHICLE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pluie_mm"] = pd.to_numeric(df["pluie_mm"], errors="coerce").fillna(0.0)
    df["saturation"] = pd.to_numeric(df["saturation_observee"], errors="coerce")
    df.loc[~df["saturation"].isin([0, 1]), "saturation"] = np.nan

    bad = (
        df["site_resolved"].isna() | df["date_parsed"].isna() | start.isna() | end.isna()
        | (df["duration_min"] <= 0) | df[VEHICLE_COLUMNS].isna().all(axis=1)
    )
    if bad.any():
        lines = ", ".join(str(i + 2) for i in df.index[bad][:10])
        problems.append(
            f"{int(bad.sum())} ligne(s) ignorée(s) (site inconnu, date/heure invalide ou aucun comptage) "
            f"- lignes du fichier : {lines}{'…' if bad.sum() > 10 else ''}"
        )
    df = df[~bad].copy()
    if df.empty:
        return None, problems + ["Aucune ligne exploitable."]

    df[VEHICLE_COLUMNS] = df[VEHICLE_COLUMNS].fillna(0)
    df["total"] = df[VEHICLE_COLUMNS].sum(axis=1)
    df["flow_per_hour"] = df["total"] * 60 / df["duration_min"]
    df["site"] = df["site_resolved"]
    df["Day_of_Week"] = df["date_parsed"].dt.dayofweek
    df["hour"] = df["hour"].astype(int)
    df["zone_id"] = df["site"].map(lambda s: int(hubs[s]["zone_id"]))
    keep = ["site", "date_parsed", "Day_of_Week", "hour", "duration_min", "total",
            "flow_per_hour", "pluie_mm", "saturation", "zone_id"]
    return df[keep].rename(columns={"date_parsed": "date"}).reset_index(drop=True), problems


def observed_profiles(df):
    """Profil observé par site : débit moyen, indice de pointe, indice de soirée/nuit."""
    rows = []
    for site, g in df.groupby("site"):
        flow = g["flow_per_hour"]
        peak = g[g["hour"].isin(PEAK_HOURS)]["flow_per_hour"]
        night = g[g["hour"].isin(NIGHT_HOURS)]["flow_per_hour"]
        off = g[~g["hour"].isin(PEAK_HOURS + NIGHT_HOURS)]["flow_per_hour"]
        rows.append({
            "site": site,
            "n_fenetres": len(g),
            "debit_moyen": flow.mean(),
            "indice_pointe": peak.mean() / off.mean() if len(peak) and len(off) and off.mean() > 0 else np.nan,
            "indice_nuit": night.mean() / flow.mean() if len(night) and flow.mean() > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("site")


def verdict(gap):
    if pd.isna(gap):
        return "n/a"
    if gap < 0.25:
        return "cohérent"
    if gap < 0.40:
        return "partiel"
    return "divergent"


def compare_with_targets(prof, hubs):
    """Compare les rangs observés (entre sites) aux profils cibles de l'archétype."""
    if len(prof) < 3:
        return None, "Au moins 3 sites sont nécessaires pour comparer des rangs."
    if any("target" not in hubs[s] for s in prof.index):
        return None, "Profils cibles absents du fichier d'association : relance `python src/model.py`."

    out = prof.copy()
    out["rang_demande_obs"] = prof["debit_moyen"].rank(pct=True)
    out["rang_pointe_obs"] = prof["indice_pointe"].rank(pct=True)
    out["rang_nuit_obs"] = prof["indice_nuit"].rank(pct=True)
    out["rang_demande_cible"] = [hubs[s]["target"]["demand"] for s in out.index]
    out["rang_pointe_cible"] = [hubs[s]["target"]["peak_share"] for s in out.index]
    out["rang_nuit_cible"] = [hubs[s]["target"]["night_share"] for s in out.index]

    gaps = pd.DataFrame({
        "demande": (out["rang_demande_cible"] - out["rang_demande_obs"]).abs(),
        "pointe": (out["rang_pointe_cible"] - out["rang_pointe_obs"]).abs(),
        "nuit": (out["rang_nuit_cible"] - out["rang_nuit_obs"]).abs(),
    })
    out["ecart_moyen"] = gaps.mean(axis=1)
    out["verdict"] = out["ecart_moyen"].map(verdict)
    return out, None


def model_check(df, model, feature_columns, alert_threshold):
    """Compare les probabilités du modèle à la saturation observée sur le terrain."""
    X = pd.DataFrame({
        "Hour": df["hour"], "Day_of_Week": df["Day_of_Week"],
        "Zone_ID": df["zone_id"], "Rain_mm": df["pluie_mm"],
    })[feature_columns]
    out = df.copy()
    out["proba_modele"] = model.predict_proba(X)[:, 1] * 100

    summary = {"n": int(len(out)), "n_labelled": int(out["saturation"].notna().sum())}
    lab = out.dropna(subset=["saturation"])
    if len(lab):
        pred = (lab["proba_modele"] >= alert_threshold).astype(int)
        y = lab["saturation"].astype(int)
        summary["accuracy"] = float((pred == y).mean())
        summary["false_alarms"] = int(((pred == 1) & (y == 0)).sum())
        summary["missed"] = int(((pred == 0) & (y == 1)).sum())
        summary["n_saturated"] = int(y.sum())
        summary["auc"] = float(roc_auc_score(y, lab["proba_modele"])) if y.nunique() == 2 else None
    # Cohérence de la forme : débit observé vs probabilité du modèle, par site
    corr = {}
    for site, g in out.groupby("site"):
        if len(g) >= 4 and g["flow_per_hour"].nunique() > 1 and g["proba_modele"].nunique() > 1:
            corr[site] = float(g["flow_per_hour"].corr(g["proba_modele"], method="spearman"))
    summary["corr_by_site"] = corr
    return out, summary