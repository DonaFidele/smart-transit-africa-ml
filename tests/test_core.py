"""
Automated tests. Run from the project root:   pytest -q

They use small synthetic data, so they need neither the dataset nor the trained models
(except the artifact checks, which are skipped when models/ is absent).
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evaluate  # noqa: E402
import field_validation as fv  # noqa: E402
import sensitivity  # noqa: E402
import weather  # noqa: E402
from forecast_features import FEATURES, LAG_FEATURES, build_features  # noqa: E402

HUBS = {
    "Échangeur de Godomey (West Gateway)": {"zone_id": 3, "target": {"demand": 0.6, "peak_share": 0.9, "night_share": 0.2}},
    "Grand Marché de Dantokpa (Commercial Hub)": {"zone_id": 1, "target": {"demand": 1.0, "peak_share": 0.6, "night_share": 0.2}},
    "Carrefour Vèdoko (Central Junction)": {"zone_id": 5, "target": {"demand": 0.8, "peak_share": 0.8, "night_share": 0.5}},
}


def synthetic_history(n_days=10, zones=(0, 1)):
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-01-01", periods=n_days * 24, freq="h")
    frames = []
    for z in zones:
        demand = rng.poisson(20 + 10 * np.sin(times.hour / 24 * 2 * np.pi) + 5 * z)
        frames.append(pd.DataFrame({"time": times, "Zone_ID": z, "Demand_Volume": demand, "Rain_mm": 0.0}))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Forecast features
# ----------------------------------------------------------------------------
def test_lag_features_use_only_the_past():
    history = synthetic_history()
    feats = build_features(history, horizon=3)
    z0 = feats[feats["Zone_ID"] == 0].reset_index(drop=True)
    assert z0["lag_h"].iloc[10] == z0["Demand_Volume"].iloc[7]            # t-3
    assert z0["same_hour_yesterday"].iloc[30] == z0["Demand_Volume"].iloc[6]  # t-24
    # at t=40 with horizon 3, the 24 h window covers t-26 .. t-3, i.e. rows 14..37
    assert z0["roll24"].iloc[40] == pytest.approx(z0["Demand_Volume"].iloc[14:38].mean())


def test_feature_columns_are_consistent():
    feats = build_features(synthetic_history(), horizon=1)
    assert all(c in feats.columns for c in FEATURES)
    assert len(FEATURES) == 4 + len(LAG_FEATURES)


# ----------------------------------------------------------------------------
# Field validation
# ----------------------------------------------------------------------------
def _counts_frame(columns=fv.REQUIRED_COLUMNS):
    rows = [
        ["2026-10-14", "Dantokpa", "07:00", "08:00", 100, 50, 10, 5, 0, 1],
        ["2026-10-14", "Godomey", "12:00", "13:00", 60, 40, 5, 2, 3, 0],
        ["2026-10-14", "Unknown place", "07:00", "08:00", 1, 1, 1, 1, 0, 0],
    ]
    return pd.DataFrame(rows, columns=columns)


def test_load_counts_ignores_invalid_rows():
    df, problems = fv.load_counts(_counts_frame(), HUBS)
    assert len(df) == 2 and problems
    assert df.loc[0, "flow_per_hour"] == pytest.approx(165)


def test_load_counts_accepts_former_french_columns():
    french = ["date", "site", "heure_debut", "heure_fin", "motos", "voitures", "bus_minibus",
              "camions", "pluie_mm", "saturation_observee"]
    df, _ = fv.load_counts(_counts_frame(french), HUBS)
    assert len(df) == 2


def test_load_counts_reports_missing_columns():
    df, problems = fv.load_counts(_counts_frame().drop(columns=["trucks"]), HUBS)
    assert df is None and "trucks" in problems[0]


def test_resolve_site_keywords_and_accents():
    names = list(HUBS.keys())
    assert fv.resolve_site("vedoko", names) == "Carrefour Vèdoko (Central Junction)"
    assert fv.resolve_site("DANTOKPA", names) == "Grand Marché de Dantokpa (Commercial Hub)"
    assert fv.resolve_site("nowhere", names) is None


# ----------------------------------------------------------------------------
# Sensitivity analysis
# ----------------------------------------------------------------------------
def _profiles(n=6):
    ranks = np.linspace(0.1, 1.0, n)
    return pd.DataFrame({"demand_rank": ranks, "peak_share_rank": ranks[::-1], "night_share_rank": np.roll(ranks, 2)},
                        index=range(n))


def test_assignment_is_one_to_one():
    profiles = _profiles()
    targets = np.array([[0.6, 0.9, 0.2], [1.0, 0.6, 0.2], [0.8, 0.8, 0.5]])
    idx = sensitivity.assign(sensitivity.cost_matrix(profiles, targets))
    assert len(set(idx)) == len(idx)


def test_no_noise_means_full_stability():
    profiles = _profiles()
    targets = np.array([[0.6, 0.9, 0.2], [1.0, 0.6, 0.2], [0.8, 0.8, 0.5]])
    base = sensitivity.assign(sensitivity.cost_matrix(profiles, targets))
    diff = np.ones((6, 6)) - np.eye(6)
    stability, shift = sensitivity.simulate(profiles, targets, base, diff, sd=0.0, n_sim=20)
    assert np.all(stability == 1.0) and np.all(shift == 0.0)


def test_noise_reduces_stability_not_below_zero():
    profiles = _profiles()
    targets = np.array([[0.6, 0.9, 0.2], [1.0, 0.6, 0.2], [0.8, 0.8, 0.5]])
    base = sensitivity.assign(sensitivity.cost_matrix(profiles, targets))
    diff = np.ones((6, 6)) - np.eye(6)
    stability, _ = sensitivity.simulate(profiles, targets, base, diff, sd=0.5, n_sim=50)
    assert np.all((stability >= 0) & (stability <= 1))


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def test_folds_are_chronological_and_disjoint():
    dates = list(pd.date_range("2026-01-01", periods=30).date)
    folds = evaluate.make_folds(dates)
    assert len(folds) == evaluate.N_FOLDS
    for train, test in folds:
        assert max(train) < min(test) and not set(train) & set(test)
    assert folds[-1][1][-1] == dates[-1]


def test_f1_from_counts():
    assert float(evaluate.f1_from_counts(8, 2, 2)) == pytest.approx(0.8)
    assert float(evaluate.f1_from_counts(0, 0, 0)) == 0.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    dates = list(range(20))
    df = pd.DataFrame({"date": np.repeat(dates, 50), "y": rng.integers(0, 2, 1000)})
    df["pred"] = np.where(rng.random(1000) < 0.8, df["y"], 1 - df["y"])
    counts = evaluate.counts_by_day(df, dates)
    idx = rng.integers(0, len(dates), size=(500, len(dates)))
    boot = evaluate.bootstrap_f1(counts, idx)
    point = float(evaluate.f1_from_counts(counts["tp"].sum(), counts["fp"].sum(), counts["fn"].sum()))
    assert np.percentile(boot, 2.5) <= point <= np.percentile(boot, 97.5)


def test_reliability_of_perfectly_calibrated_probabilities():
    p = np.repeat([0.1, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(900), np.zeros(100)]])
    _, ece = evaluate.reliability(y, p)
    assert ece == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------
# Weather module (offline part only)
# ----------------------------------------------------------------------------
def test_weather_finalize_filters_dates():
    times = pd.date_range("2026-01-01", periods=72, freq="h")
    raw = pd.DataFrame({"time": times, "Rain_mm": 1.0})
    out = weather._finalize(raw, pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-02").date())
    assert len(out) == 24 and set(out["Hour"]) == set(range(24))


# ----------------------------------------------------------------------------
# Trained artifacts (skipped when models/ has not been generated)
# ----------------------------------------------------------------------------
MODELS = os.path.join(ROOT, "models")


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "traffic_rf_model.pkl")), reason="models/ not generated")
def test_trained_model_matches_app_features():
    with open(os.path.join(MODELS, "traffic_rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    assert model.n_features_in_ == 4
    proba = model.predict_proba(pd.DataFrame([[17, 2, 1, 0.0]], columns=["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]))
    assert 0.0 <= proba[0][1] <= 1.0


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "hub_zone_mapping.json")), reason="models/ not generated")
def test_hub_mapping_is_one_to_one_with_targets():
    with open(os.path.join(MODELS, "hub_zone_mapping.json"), encoding="utf-8") as f:
        hubs = json.load(f)
    zones = [h["zone_id"] for h in hubs.values()]
    assert len(zones) == len(set(zones)) == 5
    assert all("target" in h for h in hubs.values())
    