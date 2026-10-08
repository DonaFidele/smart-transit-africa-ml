"""
Automated tests. Run from the project root:   pytest -q

They use small synthetic data, so they need neither the dataset nor the trained models
(except the artifact checks, which are skipped when models/ is absent).
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import choose_target  # noqa: E402
import data_adapter  # noqa: E402
import evaluate  # noqa: E402
import forecast_model  # noqa: E402
import model as training_model  # noqa: E402
import field_validation as fv  # noqa: E402
import sensitivity  # noqa: E402
import weather  # noqa: E402
from forecast_features import (  # noqa: E402
    FEATURES, FEATURES_REL, LAG_FEATURES, REL_LAG_FEATURES, build_features, build_relative_features,
)

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


def test_relative_features_use_train_baseline_and_only_the_past():
    history = synthetic_history()
    baseline = history.groupby(["Zone_ID", history["time"].dt.hour])["Demand_Volume"].mean()
    feats = build_relative_features(history, horizon=3, baseline=baseline)
    z0 = feats[feats["Zone_ID"] == 0].reset_index(drop=True)
    usual = baseline.loc[0].reindex(z0["Hour"]).to_numpy()
    expected_ratio = (z0["Demand_Volume"].to_numpy() + 1) / (usual + 1)
    assert np.allclose(z0["ratio"], expected_ratio)
    assert z0["rel_lag_h"].iloc[10] == pytest.approx(z0["ratio"].iloc[7])           # t-3
    assert z0["rel_same_hour_yesterday"].iloc[30] == pytest.approx(z0["ratio"].iloc[6])  # t-24
    assert len(FEATURES_REL) == 4 + len(REL_LAG_FEATURES)


def test_relative_target_is_less_zone_dependent_than_absolute():
    # Two zones with very different levels but the same daily shape: the absolute target is mostly
    # "which zone", the relative target (ratio to the zone's own level) is not.
    rng = np.random.default_rng(3)
    times = pd.date_range("2026-01-01", periods=20 * 24, freq="h")
    frames = [pd.DataFrame({"time": times, "Zone_ID": z, "Demand_Volume": rng.poisson(level), "Rain_mm": 0.0})
              for z, level in [(0, 5), (1, 50)]]
    history = pd.concat(frames, ignore_index=True)
    baseline = history.groupby(["Zone_ID", history["time"].dt.hour])["Demand_Volume"].mean()
    rel = build_relative_features(history, 1, baseline).dropna(subset=REL_LAG_FEATURES)
    absolute = build_features(history, 1).dropna(subset=LAG_FEATURES)
    gap_abs = absolute.groupby("Zone_ID")["Demand_Volume"].mean().diff().abs().iloc[-1]
    gap_rel = rel.groupby("Zone_ID")["ratio"].mean().diff().abs().iloc[-1]
    assert gap_abs > 20 and gap_rel < 0.2


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


def test_evaluate_horizon_runs_for_both_targets():
    evaluate.N_BOOT = 40  # keep the test fast
    history = synthetic_history(n_days=16, zones=(0, 1, 2, 3))
    history["Rain_mm"] = np.random.default_rng(1).gamma(1, 1, len(history)) * (np.arange(len(history)) % 7 == 0)
    for target in evaluate.TARGETS:
        res = evaluate.evaluate_horizon(history, 1, target)
        names = {r["model"] for r in res["models"]}
        assert evaluate.NO_SKILL in names and evaluate.BASELINE in names
        assert 0.0 <= res["prevalence"] <= 1.0
        assert set(res["zone_swap"]) == {evaluate.RF_LAGS, evaluate.RF_NOLAGS}


def _eval_block(auc, ci_low, no_skill_f1, drop):
    return {
        "models": [
            {"model": "Random Forest (lags)", "auc": auc, "ci_low": ci_low, "f1": ci_low + 0.02},
            {"model": "Always alert (no skill)", "auc": None, "ci_low": no_skill_f1, "f1": no_skill_f1},
        ],
        "zone_swap": {"Random Forest (no lags)": {"drop": drop}},
    }


def _evaluation(rel):
    absolute = _eval_block(0.9, 0.9, 0.4, -0.30)
    return {"horizons": {h: {"targets": {"absolute": absolute, "relative": rel}} for h in ("1", "24")}}


def test_decision_adopts_relative_when_all_criteria_pass():
    result = choose_target.decide(_evaluation(_eval_block(0.75, 0.55, 0.40, -0.02)))
    assert result["target"] == "relative"
    assert all(v["passes"] for v in result["horizons"].values())


def test_decision_keeps_absolute_when_skill_is_missing():
    result = choose_target.decide(_evaluation(_eval_block(0.55, 0.55, 0.40, -0.02)))   # AUC too low
    assert result["target"] == "absolute" and not result["horizons"]["1"]["C1_skill"]


def test_decision_keeps_absolute_when_it_does_not_beat_always_alert():
    result = choose_target.decide(_evaluation(_eval_block(0.80, 0.30, 0.40, -0.02)))   # F1 CI below no-skill
    assert result["target"] == "absolute" and not result["horizons"]["1"]["C2_beats_trivial"]


def test_decision_keeps_absolute_when_not_more_zone_robust():
    result = choose_target.decide(_evaluation(_eval_block(0.80, 0.55, 0.40, -0.50)))   # worse drop than absolute
    assert result["target"] == "absolute" and not result["horizons"]["1"]["C3_zone_robust"]


def test_parse_target_option():
    assert forecast_model.parse_target(["prog"]) == "absolute"
    assert forecast_model.parse_target(["prog", "--target", "relative"]) == "relative"
    with pytest.raises(SystemExit):
        forecast_model.parse_target(["prog", "--target", "banana"])


# ----------------------------------------------------------------------------
# Data adapter layer
# ----------------------------------------------------------------------------
def _tmp_csv(df, name="data.csv"):
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, name)
    df.to_csv(path, index=False)
    return path


def _aggregated_cfg(path, **extra):
    cfg = data_adapter.load_config(os.path.join(tempfile.mkdtemp(), "missing.json"))
    cfg.update({"adapter": "aggregated", "path": path,
                "columns": {"time": "ts", "zone": "site", "value": "v"},
                "weather_lat": 6.37, "weather_lon": 2.39})
    cfg.update(extra)
    return cfg


def test_default_config_reproduces_original_behaviour():
    cfg = data_adapter.load_config(os.path.join(tempfile.mkdtemp(), "no_such_file.json"))
    assert cfg["adapter"] == "events" and cfg["path"] == "data/urban_mobility_raw_data.csv"
    assert cfg["columns"] == {"time": "Date/Time", "lat": "Lat", "lon": "Lon"}
    assert cfg["n_clusters"] == 15 and cfg["sample_n"] == 200000 and cfg["transfer_mode"] == "profile_matching"


def test_config_file_overrides_defaults_and_switches_adapter():
    path = os.path.join(tempfile.mkdtemp(), "cfg.json")
    with open(path, "w") as f:
        json.dump({"adapter": "aggregated", "path": "x.csv", "columns": {"time": "t", "zone": "z", "value": "v"}}, f)
    cfg = data_adapter.load_config(path)
    assert cfg["adapter"] == "aggregated" and cfg["columns"] == {"time": "t", "zone": "z", "value": "v"}


def test_config_validation_rejects_bad_settings():
    base = data_adapter.load_config(os.path.join(tempfile.mkdtemp(), "missing.json"))
    for change in ({"adapter": "banana"}, {"transfer_mode": "teleport"}, {"transfer_mode": "direct"}):
        with pytest.raises(SystemExit):
            data_adapter.validate_config({**base, **change})
    speed = {**base, "adapter": "aggregated", "columns": {"time": "t", "zone": "z", "value": "v"},
             "measure_kind": "speed", "free_flow_speed": None}
    with pytest.raises(SystemExit):
        data_adapter.validate_config(speed)


def test_events_adapter_builds_a_complete_hourly_grid():
    rng = np.random.default_rng(0)
    n = 3000
    times = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 5 * 24 * 3600, n), "s")
    df = pd.DataFrame({"Date/Time": times.strftime("%m/%d/%Y %H:%M:%S"), "Lat": rng.uniform(40.6, 40.9, n),
                       "Lon": rng.uniform(-74.1, -73.8, n)})
    cfg = data_adapter.load_config(os.path.join(tempfile.mkdtemp(), "missing.json"))
    cfg.update({"path": _tmp_csv(df), "n_clusters": 4})
    data = data_adapter.load_canonical(cfg)
    h = data.history
    assert h["Demand_Volume"].sum() == n                               # every event counted exactly once
    assert len(h) == h["time"].nunique() * 4 and h["Zone_ID"].nunique() == 4   # complete grid
    assert data.sparse_events and data.zone_model is not None and list(data.zones.index) == [0, 1, 2, 3]
    again = data_adapter.load_canonical(cfg, zone_model=data.zone_model)      # reuse of the saved zones
    assert again.history.equals(h)


def test_aggregated_counts_are_summed_per_hour_and_gaps_filled_with_zero():
    t = pd.date_range("2026-03-01", periods=2 * 24 * 4, freq="15min")
    df = pd.concat([pd.DataFrame({"ts": t, "site": s, "v": 5}) for s in ("A", "B")])
    df = df[~((df["site"] == "B") & (df["ts"].dt.hour == 3))]          # B has no data at 3 am
    data = data_adapter.load_canonical(_aggregated_cfg(_tmp_csv(df)))
    h = data.history
    a = h[(h["Zone_ID"] == 0) & (h["time"] == pd.Timestamp("2026-03-01 10:00"))]["Demand_Volume"].iloc[0]
    b3 = h[(h["Zone_ID"] == 1) & (h["time"] == pd.Timestamp("2026-03-01 03:00"))]["Demand_Volume"].iloc[0]
    assert a == 20 and b3 == 0                                          # 4 x 5 per hour; gap -> 0
    assert not data.sparse_events and data.unit == "vehicles/h"


def test_aggregated_speed_becomes_a_congestion_load():
    t = pd.date_range("2026-03-01", periods=48, freq="h")
    speeds = np.where(t.hour == 8, 15.0, 55.0)                           # slow at 8 am, faster than free flow otherwise
    df = pd.DataFrame({"ts": t, "site": "seg", "v": speeds})
    data = data_adapter.load_canonical(_aggregated_cfg(_tmp_csv(df), measure_kind="speed", free_flow_speed=40))
    h = data.history.set_index("time")["Demand_Volume"]
    assert h[pd.Timestamp("2026-03-01 08:00")] == pytest.approx(25.0)    # 40 - 15
    assert h[pd.Timestamp("2026-03-01 12:00")] == 0.0                    # above free flow -> no load
    assert data.unit == "km/h below free-flow"


def test_attach_rain_uses_the_cache_without_network():
    t = pd.date_range("2026-03-01", periods=48, freq="h")
    df = pd.DataFrame({"ts": t, "site": "seg", "v": 3})
    data = data_adapter.load_canonical(_aggregated_cfg(_tmp_csv(df)))
    cache = _tmp_csv(pd.DataFrame({"time": t, "Rain_mm": np.arange(48, dtype=float)}), "weather.csv")
    history, weather = data_adapter.attach_rain(data, cache_path=cache)
    assert history["Rain_mm"].tolist() == list(np.arange(48, dtype=float))
    assert len(weather) == 48


def _zones_frame(hub_values):
    zones = pd.DataFrame({"label": [f"z{i}" for i in range(len(hub_values))], "lat": np.nan, "lon": np.nan,
                          "hub": hub_values}, index=pd.Index(range(len(hub_values)), name="Zone_ID"))
    return zones


def test_direct_mode_maps_each_hub_to_its_own_zone():
    values = ["godomey", "Dantokpa", "VEDOKO", "cadjehoun", "Portuaire", np.nan]
    zones = _zones_frame(values)
    profiles = pd.DataFrame({"demand_rank": np.linspace(0.2, 1, 6), "peak_share": 0.3, "night_share": 0.1,
                             "peak_share_rank": 0.5, "night_share_rank": 0.5}, index=zones.index)
    mapping = training_model.map_hubs_direct(profiles, zones)
    assert len(mapping) == 5 and sorted(m["zone_id"] for m in mapping.values()) == [0, 1, 2, 3, 4]
    assert all(m["distance"] == 0.0 and "target" in m for m in mapping.values())
    with pytest.raises(SystemExit):                                       # a hub with no zone -> clear error
        training_model.map_hubs_direct(profiles, _zones_frame(["godomey", "Dantokpa", np.nan, np.nan, np.nan, np.nan]))


def test_zone_profiles_from_the_hourly_history():
    t = pd.date_range("2026-03-02", periods=24, freq="h")                # a Monday
    history = pd.DataFrame({"time": list(t) * 2, "Zone_ID": [0] * 24 + [1] * 24,
                            "Demand_Volume": [1] * 24 + [3] * 24})
    zones = pd.DataFrame({"lat": [1.0, 2.0], "lon": [3.0, 4.0]}, index=pd.Index([0, 1], name="Zone_ID"))
    prof = training_model.build_zone_profiles(history, zones)
    assert prof.loc[1, "demand"] == 72 and prof.loc[0, "demand"] == 24
    assert prof.loc[0, "peak_share"] == pytest.approx(7 / 24)             # 7 peak hours out of 24
    assert prof.loc[0, "weekend_share"] == 0.0 and prof.loc[1, "demand_rank"] == 1.0


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