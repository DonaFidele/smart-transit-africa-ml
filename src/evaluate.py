"""
Statistical evaluation of the forecasting models, for TWO definitions of the target.

  absolute : "saturated"  = demand in the top 25% of all slots  (depends heavily on WHICH zone it is)
  relative : "surge"      = demand well above the zone's own usual level at that hour
                            (ratio to the zone-hour average, top 25% of ratios)

Why both? With the absolute target, knowing the zone is almost enough to guess right, so the model mostly
learns zone identity - and the Cotonou transfer then rests entirely on the hub -> zone matching.
The relative target removes the zone's habitual level, so the model must learn what makes a zone busier
than usual (weekday, weather, recent trend).

For each target and horizon (1 h, 24 h):
- Rolling-origin (expanding window) validation: several successive train/test periods.
- Model comparison against baselines, with 95% CIs by day-block bootstrap.
- Paired difference vs the "Historical rate" baseline (significant if the CI excludes 0).
- Calibration (Brier, ECE, reliability bins).
- Transfer-robustness test: F1 when the zone label is WRONG (zones randomly swapped).

Run from the project root, after src/forecast_model.py:
    python src/evaluate.py

Output: models/evaluation_results.json (also displayed in the app, Engineering tab).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_features import (  # noqa: E402
    BASE_FEATURES, FEATURES, FEATURES_REL, LAG_FEATURES, REL_LAG_FEATURES,
    build_features, build_relative_features,
)

HORIZONS = (1, 24)
TARGETS = ("absolute", "relative")
N_FOLDS = 4
MIN_TRAIN_FRAC = 0.5
N_BOOT = 2000
SEED = 42
BASELINE = "Historical rate"
NO_SKILL = "Always alert (no skill)"
RF_LAGS, RF_NOLAGS = "Random Forest (lags)", "Random Forest (no lags)"


# ----------------------------------------------------------------------------
# Building blocks (pure functions, unit-tested)
# ----------------------------------------------------------------------------
def make_folds(dates, n_folds=N_FOLDS, min_train_frac=MIN_TRAIN_FRAC):
    """Expanding-window folds over sorted dates: [(train_dates, test_dates), ...]."""
    n = len(dates)
    min_train = int(n * min_train_frac)
    block = max(1, (n - min_train) // n_folds)
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * block
        if train_end >= n:
            break
        test_end = n if k == n_folds - 1 else train_end + block
        folds.append((list(dates[:train_end]), list(dates[train_end:test_end])))
    return folds


def f1_from_counts(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def counts_by_day(df, all_dates):
    d = df.assign(
        tp=((df["y"] == 1) & (df["pred"] == 1)).astype(int),
        fp=((df["y"] == 0) & (df["pred"] == 1)).astype(int),
        fn=((df["y"] == 1) & (df["pred"] == 0)).astype(int),
    )
    return d.groupby("date")[["tp", "fp", "fn"]].sum().reindex(all_dates, fill_value=0)


def bootstrap_f1(counts, idx):
    """F1 for each bootstrap resample of days (idx: n_boot x n_days integer indices)."""
    tp, fp, fn = (counts[c].to_numpy() for c in ("tp", "fp", "fn"))
    return f1_from_counts(tp[idx].sum(axis=1), fp[idx].sum(axis=1), fn[idx].sum(axis=1))


def reliability(y, p, bins=10):
    """Reliability bins and expected calibration error."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3)})
        ece += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return rows, float(ece)


def zone_swap_f1(est, X, y, cols, rng, n_perm=5):
    """Mean F1 when zone labels are randomly permuted (simulates a wrong hub -> zone matching)."""
    zones = np.sort(X["Zone_ID"].unique())
    scores = []
    for _ in range(n_perm):
        mapping = dict(zip(zones, rng.permutation(zones)))
        Xs = X.copy()
        Xs["Zone_ID"] = X["Zone_ID"].map(mapping)
        scores.append(f1_score(y, est.predict_proba(Xs[cols])[:, 1] >= 0.5, zero_division=0))
    return float(np.mean(scores))


def make_models(cols_full, cols_base):
    rf = lambda: RandomForestClassifier(n_estimators=150, max_depth=14, random_state=SEED,  # noqa: E731
                                        class_weight="balanced", n_jobs=-1)
    logit = make_pipeline(
        ColumnTransformer([("zone", OneHotEncoder(handle_unknown="ignore"), ["Zone_ID"])],
                          remainder=StandardScaler()),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    calibrated = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=150, max_depth=14, random_state=SEED, n_jobs=-1),
        method="sigmoid", cv=TimeSeriesSplit(3),
    )
    return {
        RF_LAGS: (rf(), cols_full),
        "Random Forest (lags, calibrated)": (calibrated, cols_full),
        "Gradient Boosting (lags)": (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"), cols_full),
        "Logistic Regression (lags)": (logit, cols_full),
        RF_NOLAGS: (rf(), cols_base),
    }


def prepare_fold(history, horizon, target, train_dates, test_dates):
    """Features, target and baseline predictions for one fold (nothing from the test period is used to fit)."""
    if target == "absolute":
        feats = build_features(history, horizon).dropna(subset=LAG_FEATURES)
        value, lag_now, lag_yday, cols_full = "Demand_Volume", "lag_h", "same_hour_yesterday", FEATURES
    else:
        in_train = history["time"].dt.date.isin(train_dates)
        train_hist = history[in_train]
        baseline = (train_hist.groupby(["Zone_ID", train_hist["time"].dt.hour.rename("Hour")])["Demand_Volume"]
                    .mean())
        feats = build_relative_features(history, horizon, baseline).dropna(subset=REL_LAG_FEATURES)
        value, lag_now, lag_yday, cols_full = "ratio", "rel_lag_h", "rel_same_hour_yesterday", FEATURES_REL

    feats = feats.sort_values("time").reset_index(drop=True)  # chronological order (needed by TimeSeriesSplit)
    feats["date"] = feats["time"].dt.date
    train = feats[feats["date"].isin(train_dates)]
    test = feats[feats["date"].isin(test_dates)]
    thr = float(train[value].quantile(0.75))
    y_train = (train[value] > thr).astype(int)
    y_test = (test[value] > thr).astype(int)

    keys_zh = ["Zone_ID", "Hour", "Day_of_Week"]
    keys_h = ["Hour", "Day_of_Week"]
    t = train.assign(y=y_train)
    zone_rate = t.groupby(keys_zh)["y"].mean().rename("rate").reset_index()
    hour_rate = t.groupby(keys_h)["y"].mean().rename("rate").reset_index()
    p_zone = test[keys_zh].merge(zone_rate, on=keys_zh, how="left")["rate"].fillna(0).to_numpy()
    p_hour = test[keys_h].merge(hour_rate, on=keys_h, how="left")["rate"].fillna(0).to_numpy()

    baselines = {
        "Persistence (state at t-h)": (test[lag_now] > thr).to_numpy(),
        "Same hour yesterday": (test[lag_yday] > thr).to_numpy(),
        BASELINE: p_zone > 0.5,
        "Hour-of-week rate (no zone)": p_hour > 0.5,
        NO_SKILL: np.ones(len(test), dtype=bool),
    }
    return {"train": train, "test": test, "y_train": y_train, "y_test": y_test,
            "cols_full": cols_full, "cols_base": BASE_FEATURES, "baselines": baselines}


# ----------------------------------------------------------------------------
# Evaluation of one horizon and one target definition
# ----------------------------------------------------------------------------
def evaluate_horizon(history, horizon, target):
    dates = sorted(build_features(history, horizon).dropna(subset=LAG_FEATURES)["time"].dt.date.unique())
    folds = make_folds(dates)  # same folds and test rows for both targets -> comparable

    store, swap, imp = {}, {RF_LAGS: [], RF_NOLAGS: []}, {}
    rng_swap = np.random.default_rng(SEED)

    def keep(name, test, fold_id, y, pred, proba=None):
        store.setdefault(name, []).append(pd.DataFrame({
            "date": test["date"].to_numpy(), "fold": fold_id, "y": y.to_numpy(),
            "pred": np.asarray(pred).astype(int),
            "proba": np.nan if proba is None else proba,
        }))

    for fold_id, (train_dates, test_dates) in enumerate(folds):
        fold = prepare_fold(history, horizon, target, train_dates, test_dates)
        train, test, y_train, y_test = fold["train"], fold["test"], fold["y_train"], fold["y_test"]

        for name, (est, cols) in make_models(fold["cols_full"], fold["cols_base"]).items():
            est.fit(train[cols], y_train)
            proba = est.predict_proba(test[cols])[:, 1]
            keep(name, test, fold_id, y_test, proba >= 0.5, proba)
            if name in swap:
                swap[name].append((float(f1_score(y_test, proba >= 0.5, zero_division=0)),
                                   zone_swap_f1(est, test, y_test, cols, rng_swap)))
            if name == RF_LAGS:
                for c, v in zip(cols, est.feature_importances_):
                    imp[c] = imp.get(c, 0.0) + float(v) / len(folds)

        for name, pred in fold["baselines"].items():
            keep(name, test, fold_id, y_test, pred)

    results = {k: pd.concat(v, ignore_index=True) for k, v in store.items()}
    test_dates_all = sorted(results[BASELINE]["date"].unique())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(test_dates_all), size=(N_BOOT, len(test_dates_all)))
    base_boot = bootstrap_f1(counts_by_day(results[BASELINE], test_dates_all), idx)

    rows, calib = [], {}
    for name, df in results.items():
        counts = counts_by_day(df, test_dates_all)
        boot = bootstrap_f1(counts, idx)
        point = float(f1_from_counts(counts["tp"].sum(), counts["fp"].sum(), counts["fn"].sum()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        fold_f1 = [float(f1_from_counts(((g["y"] == 1) & (g["pred"] == 1)).sum(),
                                        ((g["y"] == 0) & (g["pred"] == 1)).sum(),
                                        ((g["y"] == 1) & (g["pred"] == 0)).sum()))
                   for _, g in df.groupby("fold")]
        row = {"model": name, "f1": round(point, 3), "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
               "f1_ci": f"{point:.2f} [{lo:.2f}, {hi:.2f}]",
               "fold_f1_mean": round(float(np.mean(fold_f1)), 3), "fold_f1_std": round(float(np.std(fold_f1)), 3),
               "fold_f1": f"{np.mean(fold_f1):.2f} ± {np.std(fold_f1):.2f}",
               "brier": None, "ece": None}
        row["auc"] = None
        if df["proba"].notna().all():
            if df["y"].nunique() == 2:
                row["auc"] = round(float(roc_auc_score(df["y"], df["proba"])), 3)
            row["brier"] = round(float(np.mean((df["proba"] - df["y"]) ** 2)), 4)
            bins, ece = reliability(df["y"], df["proba"])
            row["ece"] = round(ece, 4)
            calib[name] = bins
        if name != BASELINE:
            diff = boot - base_boot
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            row.update({"diff_vs_baseline": round(point - float(np.mean(base_boot)), 3),
                        "diff_ci": f"{point - np.mean(base_boot):+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]",
                        "significant": bool(d_lo > 0 or d_hi < 0)})
        else:
            row.update({"diff_vs_baseline": 0.0, "diff_ci": "reference", "significant": False})
        rows.append(row)

    rows.sort(key=lambda r: -r["f1"])
    zone_swap = {}
    for name, pairs in swap.items():
        true_f1, swapped_f1 = np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])
        zone_swap[name] = {"f1_true": round(float(true_f1), 3), "f1_swapped": round(float(swapped_f1), 3),
                           "drop": round(float(swapped_f1 - true_f1), 3)}
    return {
        "n_days": len(dates), "n_folds": len(folds), "n_test_days": len(test_dates_all),
        "n_test_rows": int(len(results[BASELINE])),
        "prevalence": round(float(results[BASELINE]["y"].mean()), 3),
        "models": rows, "calibration": calib,
        "importances": {k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])},
        "zone_swap": zone_swap,
    }


def print_block(horizon, target, res):
    print(f"\n=== Horizon {horizon} h | target: {target} | {res['n_folds']} rolling folds | "
          f"{res['n_test_days']} test days | {res['n_test_rows']} test rows | positives: {res['prevalence']:.0%} ===")
    table = pd.DataFrame(res["models"])[["model", "f1_ci", "fold_f1", "auc", "brier", "ece", "diff_ci", "significant"]]
    table.columns = ["Model", "F1 [95% CI]", "F1 per fold", "AUC", "Brier", "ECE", f"Δ F1 vs '{BASELINE}'", "Signif."]
    print(table.to_string(index=False))
    print("\nFeature importances (Random Forest with lags):", res["importances"])
    for name, s in res["zone_swap"].items():
        print(f"If the zone label is WRONG - {name}: F1 {s['f1_true']:.2f} -> {s['f1_swapped']:.2f} ({s['drop']:+.2f})")


def print_summary(out):
    rows = []
    for h, hblock in out["horizons"].items():
        for target, res in hblock["targets"].items():
            rf = next(r for r in res["models"] if r["model"] == RF_LAGS)
            no_skill = next(r for r in res["models"] if r["model"] == NO_SKILL)
            rows.append({
                "Horizon": f"{h} h", "Target": target, "Positives": f"{res['prevalence']:.0%}",
                "No-skill F1": f"{no_skill['f1']:.2f}",
                "RF (lags) F1 [95% CI]": rf["f1_ci"], "RF (lags) AUC": rf["auc"],
                "Δ vs historical rate": rf["diff_ci"],
                "Significant": rf["significant"],
                "Zone_ID importance": f"{res['importances'].get('Zone_ID', 0):.0%}",
                "F1 change if zone wrong (no-lags RF)": f"{res['zone_swap'][RF_NOLAGS]['drop']:+.2f}",
            })
    print("\n" + "=" * 100 + "\nSUMMARY: absolute vs relative target\n" + "=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nHow to read it:"
          "\n- F1 values are NOT comparable between targets (different difficulty); compare the GAIN over the baseline."
          "\n- 'No-skill F1' is what you get by always raising an alert: a model must beat it to be useful."
          "\n- AUC is comparable across targets: 0.5 = chance, 1.0 = perfect ranking of busy vs normal slots."
          "\n- 'Zone_ID importance': share of the model that relies on knowing the zone."
          "\n- 'F1 change if zone wrong': how much accuracy is lost if the hub is matched to the wrong training zone."
          "\n  A large drop means the Cotonou result depends on the matching; a small drop means it is robust to it.")


def main():
    path = "models/forecast_history.csv"
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run `python src/forecast_model.py` first.")
    history = pd.read_csv(path, parse_dates=["time"])

    out = {"n_boot": N_BOOT, "horizons": {}}
    for h in HORIZONS:
        out["horizons"][str(h)] = {"targets": {}}
        for target in TARGETS:
            res = evaluate_horizon(history, h, target)
            out["horizons"][str(h)]["targets"][target] = res
            print_block(h, target, res)

    print_summary(out)
    os.makedirs("models", exist_ok=True)
    with open("models/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSignif. = True when the 95% CI of the F1 difference vs the baseline excludes 0."
          "\nWith about a month of data, wide CIs are expected: report them as they are.")


if __name__ == "__main__":
    main()