"""Forecast tab with lags (backtest on the analogue zone of a Cotonou site). Works with both target definitions."""
import json
import pickle

import pandas as pd
import streamlit as st

from forecast_features import (
    FEATURES, FEATURES_REL, LAG_FEATURES, REL_LAG_FEATURES, build_features, build_relative_features,
)

HORIZON_LABELS = {"1 hour ahead": 1, "24 hours ahead": 24}
METHOD_LABELS = {
    "model": "Lag-based model",
    "static": "Historical rate (no lags)",
    "persistence": "Persistence (state at t-h)",
    "same_hour_yesterday": "Same hour yesterday",
    "no_skill": "Always alert (no skill)",
}


@st.cache_resource
def _load_artifacts():
    models = {}
    for h in (1, 24):
        with open(f"models/forecast_rf_h{h}.pkl", "rb") as f:
            models[h] = pickle.load(f)
    with open("models/forecast_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    history = pd.read_csv("models/forecast_history.csv", parse_dates=["time"])
    baseline = None
    if meta.get("target") == "relative":
        baseline = pd.read_csv("models/forecast_baseline.csv").set_index(["Zone_ID", "Hour"])["usual"]
    return models, meta, history, baseline


@st.cache_data(show_spinner=False)
def _zone_features(horizon, zone_id):
    _, meta, history, baseline = _load_artifacts()
    zone_hist = history[history["Zone_ID"] == zone_id]
    if meta.get("target") == "relative":
        feats = build_relative_features(zone_hist, horizon, baseline).dropna(subset=REL_LAG_FEATURES)
    else:
        feats = build_features(zone_hist, horizon).dropna(subset=LAG_FEATURES)
    return feats.set_index("time")


def render(hubs):
    try:
        models, meta, history, baseline = _load_artifacts()
    except FileNotFoundError:
        st.info("Forecast model not found. Run `python src/forecast_model.py` from the project root.")
        return

    relative = meta.get("target") == "relative"
    unit = meta.get("unit", "trips/h")
    cols = FEATURES_REL if relative else FEATURES
    threshold = meta["threshold"]
    cut = pd.Timestamp(meta["cut_date"])

    st.markdown("""
    **Forecast with lags (backtest).** The model predicts a zone's saturation **1 h or 24 h ahead** from recently
    observed demand (lags), hour, weekday and rain. The forecast is replayed on a test period the model has never
    seen, for the training zone matched to the chosen site.
    """)
    if relative:
        st.caption("Target: **surge** — demand clearly above this zone's usual level for that hour "
                   "(ratio to the zone-hour average, top 25% of ratios).")
    else:
        st.caption("Target: **saturation** — demand in the top 25% of all slots.")

    col_a, col_b = st.columns(2)
    with col_a:
        hub_name = st.selectbox("Site (Cotonou)", list(hubs.keys()), key="f_hub")
        zone_id = int(hubs[hub_name]["zone_id"])
        horizon = HORIZON_LABELS[st.radio("Forecast horizon", list(HORIZON_LABELS.keys()), key="f_h")]
        alert_thr = st.slider("Alert threshold (%)", 10, 90, 50, key="f_thr")

    feats = _zone_features(horizon, zone_id)
    test_feats = feats[feats.index >= cut]
    test_dates = sorted({t.date() for t in test_feats.index})

    with col_b:
        day = st.selectbox("Day of the test period", test_dates, key="f_day")
        hour = st.slider("Target hour", 0, 23, 17, key="f_hour")

    target_time = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    if target_time not in test_feats.index:
        st.warning("Time slot unavailable (not enough history to compute the lags).")
        return

    row = test_feats.loc[[target_time]]
    prob = float(models[horizon].predict_proba(row[cols])[0][1]) * 100
    demand = float(row["Demand_Volume"].iloc[0])
    if relative:
        ratio = float(row["ratio"].iloc[0])
        usual = (demand + 1) / ratio - 1
        surge_level = threshold * (usual + 1) - 1
        actually_saturated = ratio > threshold
        persistence = float(row["rel_lag_h"].iloc[0]) > threshold
        yesterday = float(row["rel_same_hour_yesterday"].iloc[0]) > threshold
        level_label, level_value = "Surge level for this hour", surge_level
    else:
        actually_saturated = demand > threshold
        persistence = float(row["lag_h"].iloc[0]) > threshold
        yesterday = float(row["same_hour_yesterday"].iloc[0]) > threshold
        level_label, level_value = "Saturation threshold", threshold

    predicted_alert = prob >= alert_thr
    issued_at = target_time - pd.Timedelta(hours=horizon)
    event = "surge" if relative else "saturation"

    st.caption(f"Forecast issued on {issued_at:%d/%m at %Hh} for {target_time:%d/%m at %Hh}.")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Predicted {event} probability", f"{prob:.1f}%")
    m2.metric("Observed demand", f"{demand:.0f} {unit}")
    m3.metric(level_label, f"{level_value:.0f} {unit}")

    if predicted_alert and actually_saturated:
        st.success(f"🎯 Justified alert: {event} predicted and observed.")
    elif predicted_alert and not actually_saturated:
        st.warning(f"🔔 False alarm: {event} predicted, not observed.")
    elif not predicted_alert and actually_saturated:
        st.error(f"⚠️ Missed {event}: not predicted but observed.")
    else:
        st.info(f"✅ Correct: no alert and no {event}.")

    st.caption(
        f"For comparison: persistence would have predicted “{event if persistence else 'normal'}”, "
        f"same hour yesterday “{event if yesterday else 'normal'}”."
    )

    window = history[(history["Zone_ID"] == zone_id)
                     & (history["time"] >= target_time - pd.Timedelta(hours=48))
                     & (history["time"] <= target_time + pd.Timedelta(hours=6))]
    chart = window.set_index("time")[["Demand_Volume"]].rename(columns={"Demand_Volume": "Observed demand"})
    if relative:
        keys = pd.MultiIndex.from_arrays([[zone_id] * len(window), window["time"].dt.hour])
        usual_series = baseline.reindex(keys).to_numpy()
        chart["Surge level"] = threshold * (usual_series + 1) - 1
    else:
        chart["Saturation threshold"] = threshold
    st.line_chart(chart, color=["#22d3ee", "#ef4444"])

    st.markdown("---")
    st.subheader("Model performance on the test period (all zones)")
    keys_present = [k for k in METHOD_LABELS if k in meta["horizons"]["1"]]
    table = pd.DataFrame({
        f"F1 at {h} h": {METHOD_LABELS[k]: meta["horizons"][str(h)][k]["f1"] for k in keys_present}
        for h in (1, 24)
    })
    st.dataframe(table, use_container_width=True)
    st.caption(
        f"Chronological test from {meta['cut_date']} ({meta['horizons']['1']['n_test']} zone-hour slots). "
        "If the model does not clearly beat the baselines (and “always alert”), the lags add no real gain."
    )

    st.subheader("Limits")
    if meta.get("transfer_mode") == "direct":
        first = "- This is a **backtest on local data**: check that the period covers the situations you care about.\n"
    else:
        first = "- This is a **backtest on training data from outside Cotonou**: the analogue zone is a stand-in.\n"
    st.markdown(
        first
        + "- Real use in Cotonou would require a stream of recent observations (counts, GPS) in the same unit.\n"
        + "- Short period (about one month): no weekly lag, results to be confirmed over a longer span."
    )