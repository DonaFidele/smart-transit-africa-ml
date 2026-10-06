"""
Statistical evaluation of the forecasting models.

- Rolling-origin (expanding window) validation: several successive train/test periods.
- Model comparison: Random Forest, Gradient Boosting, Logistic Regression, Random Forest without lags,
  against three baselines (persistence, same hour yesterday, historical rate).
- 95% confidence intervals by day-block bootstrap (days are resampled, so the strong dependence
  between hours of the same day is respected).
- Paired difference against the "Historical rate" baseline with its own CI.
- Calibration: Brier score, expected calibration error (ECE), reliability bins.

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_features import BASE_FEATURES, FEATURES, LAG_FEATURES, build_features  # noqa: E402

HORIZONS = (1, 24)
N_FOLDS = 4
MIN_TRAIN_FRAC = 0.5
N_BOOT = 2000
SEED = 42
BASELINE = "Historical rate"


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


def make_models():
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
        "Random Forest (lags)": (rf(), FEATURES),
        "Random Forest (lags, calibrated)": (calibrated, FEATURES),
        "Gradient Boosting (lags)": (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"), FEATURES),
        "Logistic Regression (lags)": (logit, FEATURES),
        "Random Forest (no lags)": (rf(), BASE_FEATURES),
    }


# ----------------------------------------------------------------------------
# Evaluation of one horizon
# ----------------------------------------------------------------------------
def evaluate_horizon(history, horizon):
    feats = build_features(history, horizon).dropna(subset=LAG_FEATURES)
    feats = feats.sort_values("time").reset_index(drop=True)  # chronological order (needed by TimeSeriesSplit)
    feats["date"] = feats["time"].dt.date
    dates = sorted(feats["date"].unique())
    folds = make_folds(dates)

    store = {}  # model name -> list of DataFrames (date, fold, y, pred, proba)

    def keep(name, test, fold_id, y, pred, proba=None):
        store.setdefault(name, []).append(pd.DataFrame({
            "date": test["date"].to_numpy(), "fold": fold_id, "y": y.to_numpy(),
            "pred": np.asarray(pred).astype(int),
            "proba": np.nan if proba is None else proba,
        }))

    for fold_id, (train_dates, test_dates) in enumerate(folds):
        train = feats[feats["date"].isin(train_dates)]
        test = feats[feats["date"].isin(test_dates)]
        thr = float(train["Demand_Volume"].quantile(0.75))
        y_train = (train["Demand_Volume"] > thr).astype(int)
        y_test = (test["Demand_Volume"] > thr).astype(int)

        for name, (est, cols) in make_models().items():
            est.fit(train[cols], y_train)
            proba = est.predict_proba(test[cols])[:, 1]
            keep(name, test, fold_id, y_test, proba >= 0.5, proba)

        keep("Persistence (state at t-h)", test, fold_id, y_test, test["lag_h"] > thr)
        keep("Same hour yesterday", test, fold_id, y_test, test["same_hour_yesterday"] > thr)
        look = (train.assign(y=y_train).groupby(["Zone_ID", "Hour", "Day_of_Week"])["y"].mean()
                .rename("hist").reset_index())
        merged = test[["Zone_ID", "Hour", "Day_of_Week"]].merge(look, on=["Zone_ID", "Hour", "Day_of_Week"], how="left")
        keep(BASELINE, test, fold_id, y_test, (merged["hist"].fillna(0) > 0.5).to_numpy())

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
        fold_f1 = [float(f1_from_counts(*(((g["y"] == 1) & (g["pred"] == 1)).sum(),
                                          ((g["y"] == 0) & (g["pred"] == 1)).sum(),
                                          ((g["y"] == 1) & (g["pred"] == 0)).sum())))
                   for _, g in df.groupby("fold")]
        row = {"model": name, "f1": round(point, 3), "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
               "f1_ci": f"{point:.2f} [{lo:.2f}, {hi:.2f}]",
               "fold_f1_mean": round(float(np.mean(fold_f1)), 3), "fold_f1_std": round(float(np.std(fold_f1)), 3),
               "fold_f1": f"{np.mean(fold_f1):.2f} ± {np.std(fold_f1):.2f}",
               "brier": None, "ece": None}
        if df["proba"].notna().all():
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
    return {"n_days": len(dates), "n_folds": len(folds), "n_test_days": len(test_dates_all),
            "n_test_rows": int(len(results[BASELINE])), "models": rows, "calibration": calib}


def main():
    path = "models/forecast_history.csv"
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run `python src/forecast_model.py` first.")
    history = pd.read_csv(path, parse_dates=["time"])

    out = {"n_boot": N_BOOT, "horizons": {}}
    for h in HORIZONS:
        res = evaluate_horizon(history, h)
        out["horizons"][str(h)] = res
        print(f"\n=== Horizon {h} h | {res['n_folds']} rolling folds | {res['n_test_days']} test days "
              f"| {res['n_test_rows']} test rows ===")
        table = pd.DataFrame(res["models"])[["model", "f1_ci", "fold_f1", "brier", "ece", "diff_ci", "significant"]]
        table.columns = ["Model", "F1 [95% CI]", "F1 per fold", "Brier", "ECE", f"Δ F1 vs '{BASELINE}'", "Signif."]
        print(table.to_string(index=False))
        best_cal = min((r for r in res["models"] if r["ece"] is not None), key=lambda r: r["ece"])
        print(f"\nReliability of the best-calibrated model ({best_cal['model']}):")
        print(pd.DataFrame(res["calibration"][best_cal["model"]]).to_string(index=False))

    os.makedirs("models", exist_ok=True)
    with open("models/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nReading: 'Signif.' = True when the 95% CI of the F1 difference vs the baseline excludes 0."
          "\nECE close to 0 = predicted probabilities match observed frequencies."
          "\nWith about a month of data, wide CIs are expected: report them as they are.")


if __name__ == "__main__":
    main()