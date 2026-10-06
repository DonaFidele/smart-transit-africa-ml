"""
Local validation from field counts collected in Cotonou.

Two questions:
  1. Does each site's observed profile (demand level, peaks, evening/night) match the assumed archetype?
  2. Are the model's probabilities consistent with the saturation actually observed?
"""
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED_COLUMNS = [
    "date", "site", "start_time", "end_time",
    "motorbikes", "cars", "minibuses_buses", "trucks",
    "rain_mm", "observed_saturation",
]
OPTIONAL_COLUMNS = ["observer", "notes"]
VEHICLE_COLUMNS = ["motorbikes", "cars", "minibuses_buses", "trucks"]

# Former French column names stay accepted
COLUMN_ALIASES = {
    "heure_debut": "start_time", "heure_fin": "end_time", "motos": "motorbikes",
    "voitures": "cars", "bus_minibus": "minibuses_buses", "camions": "trucks",
    "pluie_mm": "rain_mm", "saturation_observee": "observed_saturation", "observateur": "observer",
}

PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4]
SITE_KEYWORDS = ["godomey", "dantokpa", "vedoko", "cadjehoun", "portuaire", "akpakpa"]


def _norm(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def resolve_site(value, hub_names):
    """Accepts the full name or a keyword (e.g. 'Dantokpa', 'vedoko')."""
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
    """Clean the counts table. Returns (DataFrame or None, list of problems)."""
    problems = []
    df = raw.copy()
    df.columns = [COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"]

    hub_names = list(hubs.keys())
    df["site_resolved"] = df["site"].map(lambda v: resolve_site(v, hub_names))
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    start = pd.to_datetime(df["start_time"].astype(str).str.strip(), format="%H:%M", errors="coerce")
    end = pd.to_datetime(df["end_time"].astype(str).str.strip(), format="%H:%M", errors="coerce")
    df["duration_min"] = (end - start).dt.total_seconds() / 60
    df["hour"] = start.dt.hour

    for col in VEHICLE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0.0)
    df["saturation"] = pd.to_numeric(df["observed_saturation"], errors="coerce")
    df.loc[~df["saturation"].isin([0, 1]), "saturation"] = np.nan

    bad = (
        df["site_resolved"].isna() | df["date_parsed"].isna() | start.isna() | end.isna()
        | (df["duration_min"] <= 0) | df[VEHICLE_COLUMNS].isna().all(axis=1)
    )
    if bad.any():
        lines = ", ".join(str(i + 2) for i in df.index[bad][:10])
        problems.append(
            f"{int(bad.sum())} row(s) ignored (unknown site, invalid date/time or no counts) "
            f"- file lines: {lines}{'…' if bad.sum() > 10 else ''}"
        )
    df = df[~bad].copy()
    if df.empty:
        return None, problems + ["No usable rows."]

    df[VEHICLE_COLUMNS] = df[VEHICLE_COLUMNS].fillna(0)
    df["total"] = df[VEHICLE_COLUMNS].sum(axis=1)
    df["flow_per_hour"] = df["total"] * 60 / df["duration_min"]
    df["site"] = df["site_resolved"]
    df["Day_of_Week"] = df["date_parsed"].dt.dayofweek
    df["hour"] = df["hour"].astype(int)
    df["zone_id"] = df["site"].map(lambda s: int(hubs[s]["zone_id"]))
    keep = ["site", "date_parsed", "Day_of_Week", "hour", "duration_min", "total",
            "flow_per_hour", "rain_mm", "saturation", "zone_id"]
    return df[keep].rename(columns={"date_parsed": "date"}).reset_index(drop=True), problems


def observed_profiles(df):
    """Observed profile per site: mean flow, peak index, evening/night index."""
    rows = []
    for site, g in df.groupby("site"):
        flow = g["flow_per_hour"]
        peak = g[g["hour"].isin(PEAK_HOURS)]["flow_per_hour"]
        night = g[g["hour"].isin(NIGHT_HOURS)]["flow_per_hour"]
        off = g[~g["hour"].isin(PEAK_HOURS + NIGHT_HOURS)]["flow_per_hour"]
        rows.append({
            "site": site,
            "n_windows": len(g),
            "mean_flow": flow.mean(),
            "peak_index": peak.mean() / off.mean() if len(peak) and len(off) and off.mean() > 0 else np.nan,
            "night_index": night.mean() / flow.mean() if len(night) and flow.mean() > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("site")


def verdict(gap):
    if pd.isna(gap):
        return "n/a"
    if gap < 0.25:
        return "consistent"
    if gap < 0.40:
        return "partial"
    return "divergent"


def compare_with_targets(prof, hubs):
    """Compare observed ranks (between sites) with the archetype's target profile."""
    if len(prof) < 3:
        return None, "At least 3 sites are needed to compare ranks."
    if any("target" not in hubs[s] for s in prof.index):
        return None, "Target profiles missing from the mapping file: re-run `python src/model.py`."

    out = prof.copy()
    out["demand_rank_obs"] = prof["mean_flow"].rank(pct=True)
    out["peak_rank_obs"] = prof["peak_index"].rank(pct=True)
    out["night_rank_obs"] = prof["night_index"].rank(pct=True)
    out["demand_rank_target"] = [hubs[s]["target"]["demand"] for s in out.index]
    out["peak_rank_target"] = [hubs[s]["target"]["peak_share"] for s in out.index]
    out["night_rank_target"] = [hubs[s]["target"]["night_share"] for s in out.index]

    gaps = pd.DataFrame({
        "demand": (out["demand_rank_target"] - out["demand_rank_obs"]).abs(),
        "peak": (out["peak_rank_target"] - out["peak_rank_obs"]).abs(),
        "night": (out["night_rank_target"] - out["night_rank_obs"]).abs(),
    })
    out["mean_gap"] = gaps.mean(axis=1)
    out["verdict"] = out["mean_gap"].map(verdict)
    return out, None


def model_check(df, model, feature_columns, alert_threshold):
    """Compare the model's probabilities with the saturation observed in the field."""
    X = pd.DataFrame({
        "Hour": df["hour"], "Day_of_Week": df["Day_of_Week"],
        "Zone_ID": df["zone_id"], "Rain_mm": df["rain_mm"],
    })[feature_columns]
    out = df.copy()
    out["model_prob"] = model.predict_proba(X)[:, 1] * 100

    summary = {"n": int(len(out)), "n_labelled": int(out["saturation"].notna().sum())}
    lab = out.dropna(subset=["saturation"])
    if len(lab):
        pred = (lab["model_prob"] >= alert_threshold).astype(int)
        y = lab["saturation"].astype(int)
        summary["accuracy"] = float((pred == y).mean())
        summary["false_alarms"] = int(((pred == 1) & (y == 0)).sum())
        summary["missed"] = int(((pred == 0) & (y == 1)).sum())
        summary["n_saturated"] = int(y.sum())
        summary["auc"] = float(roc_auc_score(y, lab["model_prob"])) if y.nunique() == 2 else None
    corr = {}
    for site, g in out.groupby("site"):
        if len(g) >= 4 and g["flow_per_hour"].nunique() > 1 and g["model_prob"].nunique() > 1:
            corr[site] = float(g["flow_per_hour"].corr(g["model_prob"], method="spearman"))
    summary["corr_by_site"] = corr
    return out, summary