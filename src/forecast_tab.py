"""Forecast tab with lags (backtest on the analogue zone of a Cotonou site)."""
import json
import pickle

import pandas as pd
import streamlit as st

from forecast_features import FEATURES, LAG_FEATURES, build_features

HORIZON_LABELS = {"1 hour ahead": 1, "24 hours ahead": 24}
METHOD_LABELS = {
    "model": "Lag-based model",
    "static": "Historical rate (no lags)",
    "persistence": "Persistence (state at t-h)",
    "same_hour_yesterday": "Same hour yesterday",
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
    return models, meta, history


@st.cache_data(show_spinner=False)
def _zone_features(horizon, zone_id):
    _, _, history = _load_artifacts()
    feats = build_features(history[history["Zone_ID"] == zone_id], horizon)
    return feats.dropna(subset=LAG_FEATURES).set_index("time")


def render(hubs):
    st.markdown("""
    **Forecast with lags (backtest).** The model predicts a zone's saturation **1 h or 24 h ahead** from recently
    observed demand (lags), hour, weekday and rain. The forecast is replayed on a test period the model has never
    seen, for the training zone matched to the chosen site.
    """)

    try:
        models, meta, history = _load_artifacts()
    except FileNotFoundError:
        st.info("Forecast model not found. Run `python src/forecast_model.py` from the project root.")
        return

    threshold = meta["threshold"]
    cut = pd.Timestamp(meta["cut_date"])

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
    prob = float(models[horizon].predict_proba(row[FEATURES])[0][1]) * 100
    actual = float(row["Demand_Volume"].iloc[0])
    actually_saturated = actual > threshold
    predicted_alert = prob >= alert_thr
    issued_at = target_time - pd.Timedelta(hours=horizon)

    st.caption(f"Forecast issued on {issued_at:%d/%m at %Hh} for {target_time:%d/%m at %Hh}.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted saturation probability", f"{prob:.1f}%")
    m2.metric("Observed demand", f"{actual:.0f} trips/h")
    m3.metric("Saturation threshold", f"{threshold:.0f} trips/h")

    if predicted_alert and actually_saturated:
        st.success("🎯 Justified alert: saturation predicted and observed.")
    elif predicted_alert and not actually_saturated:
        st.warning("🔔 False alarm: saturation predicted, not observed.")
    elif not predicted_alert and actually_saturated:
        st.error("⚠️ Missed saturation: not predicted but observed.")
    else:
        st.info("✅ Correct: no alert and no saturation.")

    persistence = float(row["lag_h"].iloc[0]) > threshold
    yesterday = float(row["same_hour_yesterday"].iloc[0]) > threshold
    st.caption(
        f"For comparison: persistence would have predicted “{'saturated' if persistence else 'stable'}”, "
        f"same hour yesterday “{'saturated' if yesterday else 'stable'}”."
    )

    window = history[(history["Zone_ID"] == zone_id)
                     & (history["time"] >= target_time - pd.Timedelta(hours=48))
                     & (history["time"] <= target_time + pd.Timedelta(hours=6))]
    chart = window.set_index("time")[["Demand_Volume"]].rename(columns={"Demand_Volume": "Observed demand"})
    chart["Saturation threshold"] = threshold
    st.line_chart(chart, color=["#22d3ee", "#ef4444"])

    st.markdown("---")
    st.subheader("Model performance on the test period (all zones)")
    table = pd.DataFrame({
        f"F1 at {h} h": {METHOD_LABELS[k]: meta["horizons"][str(h)][k]["f1"] for k in METHOD_LABELS}
        for h in (1, 24)
    })
    st.dataframe(table, use_container_width=True)
    st.caption(
        f"Chronological test from {meta['cut_date']} ({meta['horizons']['1']['n_test']} zone-hour slots). "
        "If the model does not clearly beat the baselines, the lags add no real gain."
    )

    st.subheader("Limits")
    st.markdown(
        "- This is a **backtest on training data from outside Cotonou**: the analogue zone is a stand-in.\n"
        "- Real use in Cotonou would require a stream of recent observations (counts, GPS) in the same unit.\n"
        "- Short period (about one month): no weekly lag, results to be confirmed over a longer span."
    )