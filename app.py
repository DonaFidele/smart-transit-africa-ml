import json
import os
import pickle
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import theme  # noqa: E402
from weather import fetch_forecast, HEAVY_RAIN_MM  # noqa: E402
from forecast_tab import render as render_forecast_tab  # noqa: E402
from field_tab import render as render_field_tab  # noqa: E402

st.set_page_config(page_title="SmartTransit Africa · Urban Risk Lab", page_icon="⚡", layout="wide")
theme.inject_css()

C = theme.COLORS
FEATURE_COLUMNS = ["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]
COTONOU_LAT, COTONOU_LON = 6.37, 2.39

DAYS = ["Monday", "Tuesday", "Wednesday (Dantokpa Market Day)", "Thursday",
        "Friday", "Saturday (Market Day)", "Sunday"]
MODE_MANUAL = "Manual scenario"
MODE_FORECAST = "Live forecast (Cotonou)"
PRESETS = {
    "Rush": {"day": 0, "hour": 8, "rain": 0.0},
    "Market": {"day": 2, "hour": 17, "rain": 0.0},
    "Storm": {"day": 4, "hour": 17, "rain": 12.0},
    "Night": {"day": 6, "hour": 3, "rain": 0.0},
}

# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
st.session_state.setdefault("recommendation_history", [])
st.session_state.setdefault("sb_mode", MODE_MANUAL)
st.session_state.setdefault("sb_day", DAYS[0])
st.session_state.setdefault("sb_hour", 17)
st.session_state.setdefault("sb_rain", 0.0)
st.session_state.setdefault("sb_thr", 50)
st.session_state.setdefault("sb_market", True)
st.session_state.setdefault("sb_radj", 0)


def apply_preset(name):
    p = PRESETS[name]
    st.session_state["sb_mode"] = MODE_MANUAL
    st.session_state["sb_day"] = DAYS[p["day"]]
    st.session_state["sb_hour"] = p["hour"]
    st.session_state["sb_rain"] = p["rain"]


def clear_log():
    st.session_state["recommendation_history"] = []


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def rain_label(mm):
    if mm < 0.1:
        return "no rain"
    if mm < 2.5:
        return "light rain"
    if mm < HEAVY_RAIN_MM:
        return "moderate rain"
    return "heavy rain"


def short_name(name):
    base = name.split(" (")[0]
    for prefix in ("Échangeur de ", "Grand Marché de ", "Carrefour ", "Zone "):
        base = base.replace(prefix, "")
    return base


def predict_risk(model, hours, day_index, zone_id, rain_mm):
    """Saturation probability (%) for one or several hours."""
    hours = list(hours) if hasattr(hours, "__iter__") else [hours]
    X = pd.DataFrame({
        "Hour": hours, "Day_of_Week": day_index, "Zone_ID": zone_id, "Rain_mm": rain_mm,
    })[FEATURE_COLUMNS]
    return model.predict_proba(X)[:, 1] * 100


def status_of(risk, threshold):
    if risk >= threshold:
        return "critical", C["red"]
    if risk >= threshold - 10:
        return "watch", C["yellow"]
    return "stable", C["green"]


# ----------------------------------------------------------------------------
# Pop-up report
# ----------------------------------------------------------------------------
@st.dialog("📋 Urban Planning & Traffic Mitigation Guidelines", width="large")
def show_guidelines_modal(loc, risk, rain, market, critical_flag, market_mult):
    st.write(f"### Prescriptive Action Report for: **{loc}**")

    if critical_flag:
        st.error(f"🚨 **CRITICAL CONGESTION WARNING: {risk:.1f}% saturation risk**")

        if rain == 1:
            st.markdown("### ⚠️ Drainage Recommendation")
            st.write(f"High risk of imminent road-network saturation at **{loc}**. Immediately activate the lift pumps at the main station and suspend all ongoing earthworks in this urban sector.")

            st.markdown("### 🔧 Infrastructure Maintenance (Post-Flood)")
            st.write("Prioritize urgent inspection of the surrounding paved roads in Cotonou to detect, map and treat any critical scouring under the roadway before it collapses.")
        else:
            st.markdown("### 🚦 Traffic Regulation")
            st.write(f"Immediately deploy traffic regulation officers at the **{loc}** intersection to manage vehicle flows manually before the network is completely blocked.")

        st.markdown("---")
        st.markdown("#### 🏛️ Long-Term Municipal Structural Policy")
        if market == 1 or (market_mult and ("Dantokpa" in loc or "Portuaire" in loc)):
            st.write("- **Logistical Decoupling:** Marketplace loading zones are oversaturating transit tracks. Plan a decentralized off-dock logistics hub to move freight handling outside systemic peak hours.")
        else:
            st.write("- **Capacity Elasticity Engineering:** Expand the road network's geometric absorption limits or introduce automated dynamic tidal lanes to accommodate exponential commuter growth.")
    else:
        st.success(f"🟢 **STABLE FLOW CONDITIONS: {risk:.1f}% saturation risk**")
        st.markdown("#### Baseline Operational Guidelines")
        st.write("No emergency intervention required in this sector. Maintain routine monitoring and road telemetry through the municipal cameras.")
        st.write("- **Status:** The infrastructure corridor retains enough geometric elasticity to handle current transit velocity curves.")

    if st.session_state.get("is_direct"):
        st.caption("⚠️ Simulation from a model trained on local measurements. Check data coverage and quality before any operational decision.")
    else:
        st.caption("⚠️ Simulation from a demonstration model (method transfer). Field validation is required before any operational decision.")
    st.info("Close this pop-up to see the new entry in the Action Log tab.")


# ----------------------------------------------------------------------------
# Artifacts loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_production_models():
    with open("models/traffic_rf_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        clusterer = pickle.load(f)
    with open("models/hub_zone_mapping.json", "r", encoding="utf-8") as f:
        hubs = json.load(f)
    with open("models/training_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    profiles = pd.read_csv("models/zone_profiles.csv", index_col=0)
    return classifier, clusterer, hubs, profiles, meta


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast():
    """Real forecast for Cotonou; None if the network is unavailable."""
    try:
        return fetch_forecast(COTONOU_LAT, COTONOU_LON, hours=48)
    except Exception:
        return None


try:
    model, kmeans, COTONOU_HUBS, zone_profiles, meta = load_production_models()

    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        st.error("The loaded model does not match this app version. Re-run `python src/model.py` and restart the app.")
        st.stop()

    hub_names = list(COTONOU_HUBS.keys())
    is_direct = meta.get("transfer_mode", "profile_matching") == "direct"
    source_name = meta.get("source_name", "public mobility dataset")
    st.session_state["is_direct"] = is_direct

    # ========================================================================
    # SIDEBAR : controls
    # ========================================================================
    with st.sidebar:
        theme.render(
            '<div class="gel-header"><span class="gel-bolt">⚡</span>'
            '<span class="gel-title" style="font-size:1.1rem">SMARTTRANSIT</span></div>'
            '<div class="gel-sub" style="margin-bottom:.4rem">Urban risk lab v3.0</div>'
        )

        theme.render(theme.label("Preset mode"))
        for col, preset in zip(st.columns(4), PRESETS):
            col.button(preset, key=f"preset_{preset}", on_click=apply_preset, args=(preset,),
                       use_container_width=True)

        theme.render(theme.label("Hub"))
        selected_location = st.selectbox("Target Cotonou hub", hub_names, key="sb_hub",
                                         format_func=short_name, label_visibility="collapsed")
        hub_data = COTONOU_HUBS[selected_location]

        theme.render(theme.label("Scenario"))
        forecast_df = load_forecast()
        mode = st.radio("Scenario source", [MODE_MANUAL, MODE_FORECAST], key="sb_mode",
                        label_visibility="collapsed")
        use_forecast = mode == MODE_FORECAST
        if use_forecast and forecast_df is None:
            st.warning("Forecast unavailable (no connection?). Falling back to the manual scenario.")
            use_forecast = False

        if use_forecast:
            labels = [f"{t:%a %d/%m %Hh} — {mm:.1f} mm/h"
                      for t, mm in zip(forecast_df["time"], forecast_df["Rain_mm"])]
            choice = st.selectbox("Forecast slot (next 48 h)", labels, key="sb_slot")
            row = forecast_df.iloc[labels.index(choice)] if choice in labels else forecast_df.iloc[0]
            day_index = int(row["time"].weekday())
            selected_hour = int(row["time"].hour)
            rain_mm = float(row["Rain_mm"])
            prob = row.get("Rain_prob")
            prob_txt = f", rain probability {prob:.0f}%" if pd.notna(prob) else ""
            st.caption(f"{rain_label(rain_mm).capitalize()} ({rain_mm:.1f} mm/h{prob_txt}).")
        else:
            selected_day = st.selectbox("Day of the week", DAYS, key="sb_day")
            day_index = DAYS.index(selected_day)
            selected_hour = st.slider("Hour of the day", 0, 23, key="sb_hour")
            rain_mm = st.slider("Rainfall (mm/h)", 0.0, 30.0, step=0.5, key="sb_rain")
            st.caption(f"{rain_label(rain_mm).capitalize()} (heavy rain from {HEAVY_RAIN_MM} mm/h).")

        theme.render(theme.label("Alert rules"))
        user_crit_threshold = st.slider("Critical alert threshold (%)", 20, 95, key="sb_thr")
        market_multiplier = st.checkbox("Prioritize marketplace logistics", key="sb_market")
        with st.expander("Advanced"):
            rain_penalty = st.slider(
                "Expert rain adjustment (+ points)", 0, 30, key="sb_radj",
                help="Optional. The model already uses the real rainfall learned during training. "
                     "This adds an expert surcharge (local hypothesis) when rain is heavy.",
            )

        theme.render(theme.label("Controls"))
        run_clicked = st.button("⚡ RUN SIMULATION", type="primary", use_container_width=True, key="run_btn")
        st.caption("The dashboard updates live. Press RUN to log the scenario and open the action report.")
        with st.expander("ℹ️ HOW TO READ THIS"):
            st.markdown(
                "- **Risk score**: probability that the hub's demand exceeds the saturation level (top 25% of slots).\n"
                "- **Alert threshold**: the risk above which the status becomes *critical*.\n"
                "- **Watch**: within 10 points below the threshold.\n"
                "- **Presets**: one-click scenarios (rush hour, market day, storm, night)."
            )

    # ========================================================================
    # Computation
    # ========================================================================
    is_market_day = 1 if day_index in [2, 5] else 0
    heavy_rain_flag = 1 if rain_mm >= HEAVY_RAIN_MM else 0
    extra = rain_penalty if heavy_rain_flag else 0

    base_risk = float(predict_risk(model, [selected_hour], day_index, hub_data["zone_id"], rain_mm)[0])
    risk = min(100.0, base_risk + extra)
    is_critical = risk >= user_crit_threshold
    status, status_color = status_of(risk, user_crit_threshold)

    open_report = False
    if run_clicked:
        entry = {
            "Location": selected_location,
            "Risk": f"{risk:.1f}%",
            "Status": "CRITICAL 🔴" if is_critical else "STABLE 🟢",
            "When": f"{DAYS[day_index].split(' (')[0]} {selected_hour}h, {rain_mm:.1f} mm/h",
            "Details": "",
        }
        if is_critical and heavy_rain_flag == 1:
            entry["Details"] = (
                f"⚠️ **Drainage Recommendation:** Activate the lift pumps and suspend earthworks at {selected_location}.\n\n"
                "🔧 **Infrastructure Maintenance:** Urgent priority inspection of Cotonou's paved roads for pavement scouring."
            )
        elif is_critical:
            entry["Details"] = f"🚦 **Emergency Traffic Regulation:** Traffic officers must be deployed to relieve congestion at {selected_location}."
        else:
            entry["Details"] = f"✅ Smooth and stable flow recorded for {selected_location}. Nothing to report (routine monitoring)."
        st.session_state.recommendation_history.insert(0, entry)
        open_report = True

    history = st.session_state.recommendation_history
    n_crit = sum(1 for e in history if e["Status"].startswith("CRITICAL"))
    n_stable = len(history) - n_crit

    # ========================================================================
    # HEADER, METRIC STRIP, BANNER
    # ========================================================================
    theme.render(theme.header(
        "SmartTransit Africa",
        ["v3.0", "COTONOU · BENIN", "Hourly risk"],
        f"Urban congestion risk lab · {datetime.now(timezone.utc):%d %b %Y %H:%M} UTC",
    ))
    if is_direct:
        theme.render(theme.note(
            f"The model is trained on local measurements ({source_name}), with the real rainfall observed at the same "
            "place and dates. Each hub is its own zone. Check the period covered, the sensor quality and the number of "
            "heavy-rain hours before relying on the results for operational decisions."
        ))
    else:
        theme.render(theme.note(
            "Demonstration prototype (method transfer). The model is trained on a public mobility dataset from outside "
            "Cotonou, with the real rainfall observed at that place and dates. Each Cotonou hub is matched to the training "
            "zone whose demand profile resembles it most. Results illustrate the data → model → decision chain and are not "
            "a validated forecast for Cotonou."
        ))

    strip = st.columns(5)
    strip[0].markdown(theme.card("Selected hub", short_name(selected_location), hub_data["archetype"], C["cyan"]),
                      unsafe_allow_html=True)
    strip[1].markdown(theme.card("Congestion risk", f"{risk:.1f}%",
                                 f"model {base_risk:.1f}%" + (f" + {extra} pts" if extra else ""), status_color),
                      unsafe_allow_html=True)
    strip[2].markdown(theme.card("Rainfall", f"{rain_mm:.1f} mm/h", rain_label(rain_mm),
                                 C["cyan"] if rain_mm >= 0.1 else C["muted"]), unsafe_allow_html=True)
    strip[3].markdown(theme.card("Alert status", status.upper(), f"threshold {user_crit_threshold}%", status_color),
                      unsafe_allow_html=True)
    if is_direct:
        strip[4].markdown(theme.card("Data source", f"Zone {hub_data['zone_id']}", "measured locally", C["green"]),
                          unsafe_allow_html=True)
    else:
        strip[4].markdown(theme.card("Training analogue", f"Zone {hub_data['zone_id']}",
                                     f"profile gap {hub_data['distance']}", C["purple"]), unsafe_allow_html=True)

    if status == "critical":
        banner = theme.banner(
            "danger", "🚨", "CRITICAL SATURATION RISK — ACTION REQUIRED",
            ("Heavy rain + saturation: activate drainage pumps and suspend earthworks (details in the report)."
             if heavy_rain_flag else
             "Deploy traffic regulation officers before the network is fully blocked (details in the report)."),
        )
    elif status == "watch":
        banner = theme.banner(
            "warn", "👁️", "WATCH — APPROACHING THE ALERT THRESHOLD",
            f"Risk is within 10 points of the {user_crit_threshold}% threshold. Keep monitoring this hub.",
        )
    else:
        banner = theme.banner(
            "ok", "🎯", "STABLE FLOW — ROUTINE MONITORING",
            f"Risk is below the {user_crit_threshold}% alert threshold. No intervention required.",
        )
    theme.render(banner)

    # ========================================================================
    # MAIN LAYOUT : tabs (left) + score column (right)
    # ========================================================================
    main_col, side_col = st.columns([3.1, 1])

    with side_col:
        theme.render(theme.score_card(
            risk, "Risk score",
            {"critical": "Critical — act now", "watch": "Approaching the threshold", "stable": "Stable conditions"}[status],
            status_color,
        ))
        theme.render(theme.tiles([
            (n_crit, "Critical", C["red"]),
            (n_stable, "Stable", C["green"]),
            (len(history), "Runs", C["cyan"]),
        ]))
        theme.render(theme.label("Key indicators"))
        margin = user_crit_threshold - risk
        theme.render(theme.side_card(
            "Alert threshold", f"{user_crit_threshold}%",
            f"{abs(margin):.1f} pts {'below' if margin > 0 else 'above'} the threshold", C["red"]))
        theme.render(theme.side_card(
            "Rain condition", rain_label(rain_mm).upper(), f"{rain_mm:.1f} mm/h", C["cyan"]))
        theme.render(theme.side_card(
            "Model quality", f"F1 {meta['f1_model']:.2f}",
            f"naive baseline {meta['f1_baseline']:.2f} (chronological test)", C["yellow"]))

    with main_col:
        tab_risk, tab_map, tab_log, tab_eng, tab_fc, tab_field = st.tabs([
            "📈 Risk profile", "🗺️ Hub map", "📋 Action log",
            "🔬 Engineering", "🔮 Forecast", "🧪 Field data",
        ])

        # ---------------- Risk profile ----------------
        with tab_risk:
            hours = list(range(24))
            profile = predict_risk(model, hours, day_index, hub_data["zone_id"], rain_mm)
            chart_df = pd.DataFrame({
                "Predicted risk (%)": profile,
                "Alert threshold (%)": float(user_crit_threshold),
            }, index=hours)
            chart_df.index.name = "Hour"
            st.caption(f"Predicted saturation risk by hour — {short_name(selected_location)}, "
                       f"{DAYS[day_index].split(' (')[0]}, {rain_mm:.1f} mm/h")
            st.line_chart(chart_df, color=["#22d3ee", "#ef4444"], height=300)

            comparison = pd.Series(
                {short_name(n): float(predict_risk(model, [selected_hour], day_index, h["zone_id"], rain_mm)[0])
                 for n, h in COTONOU_HUBS.items()},
                name="Predicted risk (%)",
            )
            st.caption(f"All hubs at {selected_hour}h, same day and rain")
            st.bar_chart(comparison, color="#22d3ee", height=240)

            with st.expander("ℹ️ Reading this chart"):
                st.markdown(
                    "- **Top chart** — predicted saturation risk for each hour of the selected day. "
                    "The red line is your alert threshold.\n"
                    "- **Bottom chart** — the same hour compared across the five Cotonou hubs.\n\n"
                    "Hours where the cyan curve crosses the red line are the windows that need a regulation plan."
                )

        # ---------------- Hub map ----------------
        with tab_map:
            map_rows = []
            for name, h in COTONOU_HUBS.items():
                r = float(predict_risk(model, [selected_hour], day_index, h["zone_id"], rain_mm)[0])
                if heavy_rain_flag:
                    r = min(100.0, r + rain_penalty)
                s, _ = status_of(r, user_crit_threshold)
                map_rows.append({
                    "lat": h["lat"], "lon": h["lon"], "Location": name,
                    "color": {"critical": "#ef4444", "watch": "#facc15", "stable": "#22c55e"}[s],
                    "size": 160.0 if name == selected_location else 80.0,
                })
            st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon", color="color", size="size", zoom=12)
            st.caption("🔴 critical · 🟡 watch · 🟢 stable — the selected hub is drawn larger.")
            match_line = (f"Measured locally as zone **{hub_data['zone_id']}**." if is_direct else
                          f"Matched to training zone **{hub_data['zone_id']}** (profile gap {hub_data['distance']}).")
            st.markdown(
                f"**{short_name(selected_location)}** — {hub_data['archetype']}  \n"
                f"{hub_data['criteria']}  \n"
                f"{match_line}"
            )

        # ---------------- Action log ----------------
        with tab_log:
            head_a, head_b = st.columns([4, 1])
            head_a.subheader("Municipal log & guidelines archive")
            head_b.button("Clear log", on_click=clear_log, use_container_width=True)
            if history:
                total = len(history)
                for idx, entry in enumerate(history):
                    with st.expander(f"Log #{total - idx} — {short_name(entry['Location'])} ({entry['Status']})",
                                     expanded=(idx == 0)):
                        st.markdown(f"**Saturation probability:** `{entry['Risk']}` · {entry['When']}")
                        st.write(entry["Details"])
            else:
                st.info("No logs yet. Configure the scenario in the sidebar and press RUN SIMULATION.")

        # ---------------- Engineering ----------------
        with tab_eng:
            st.markdown("**Production engineering & model diagnostics.** Raw feature vectors, model output and "
                        "the assumptions behind the Cotonou transfer.")
            col_in, col_out = st.columns(2)
            with col_in:
                st.subheader("Raw feature vector")
                eng_hour = st.slider("Hour", 0, 23, 17, key="e_hour")
                eng_day = st.slider("Day_of_Week (0 = Mon)", 0, 6, 2, key="e_day")
                eng_zone = st.selectbox("Zone_ID", sorted(zone_profiles.index.tolist()), key="e_zone")
                eng_rain = st.slider("Rain_mm (mm/h)", 0.0, 30.0, 0.0, 0.5, key="e_rain")
            with col_out:
                st.subheader("Model output")
                eng_vector = pd.DataFrame([[eng_hour, eng_day, eng_zone, eng_rain]], columns=FEATURE_COLUMNS)
                if st.button("Run diagnostic", key="e_run"):
                    proba = float(model.predict_proba(eng_vector)[0][1]) * 100
                    st.metric("Saturation probability", f"{proba:.1f}%")
                    st.dataframe(eng_vector)
                st.markdown("#### Feature importances")
                importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
                st.bar_chart(importances, color="#22d3ee")

            st.markdown("---")
            st.subheader("🌧️ Real weather & validation")
            m1, m2, m3 = st.columns(3)
            m1.metric("F1 model (chronological test)", f"{meta['f1_model']:.2f}")
            m2.metric("F1 naive baseline", f"{meta['f1_baseline']:.2f}")
            m3.metric("Learned effect of heavy rain", f"{meta['rain_effect_points']:+.1f} pts")
            st.caption(
                f"Real rainfall from {meta['weather_source']} at ({meta['weather_lat']}, {meta['weather_lon']}), "
                f"{meta['date_start']} to {meta['date_end']}: {meta['rain_hours_share']:.0%} rainy hours, "
                f"{meta['heavy_rain_hours']} hours of heavy rain (≥ {meta['heavy_rain_mm']} mm/h). "
                f"Trained on {meta['n_train_days']} days, tested on {meta['n_test_days']} days."
            )
            if meta["heavy_rain_hours"] < 20:
                st.warning("Very few heavy-rain episodes in the training data: the learned rain effect is unreliable.")
            if meta["rain_effect_points"] <= 0:
                if is_direct:
                    st.warning(
                        "The learned effect of heavy rain is zero or negative. With local data this may reflect too few "
                        "heavy-rain hours in the period, or a measure that does not react to rain. Check the heavy-rain "
                        "hours above before drawing conclusions."
                    )
                else:
                    st.warning(
                        "The learned effect of heavy rain is zero or negative. This is consistent with ride demand that "
                        "changes with the weather at the training location, but it says nothing about how traffic behaves "
                        "in Cotonou. Do not read it as evidence."
                    )

            st.markdown("---")
            if is_direct:
                st.subheader("📍 Local zones")
                st.write("The model is trained on measurements taken in Cotonou itself: each hub is its own zone, "
                         "so no profile matching is involved.")
            else:
                st.subheader("🔗 Cotonou → training-zone transfer")
                st.write(
                    "Each Cotonou hub is matched to the training zone whose demand profile (volume, share of peak hours, "
                    "share of night) is closest to the target archetype. The assignment is one-to-one and computed at training time."
                )
            mapping_df = pd.DataFrame([
                {
                    "Hub (Cotonou)": name, "Archetype": h["archetype"], "Criteria": h["criteria"],
                    "Zone": h["zone_id"], "Zone demand rank": h["zone_demand_rank"],
                    "Peak share": h["zone_peak_share"], "Night share": h["zone_night_share"],
                    "Profile gap": h["distance"],
                }
                for name, h in COTONOU_HUBS.items()
            ])
            if is_direct:
                mapping_df = mapping_df.drop(columns=["Profile gap"])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
            with st.expander("Profiles of all zones"):
                st.dataframe(zone_profiles, use_container_width=True)

            eval_path = "models/evaluation_results.json"
            if os.path.exists(eval_path):
                st.markdown("---")
                st.subheader("📊 Statistical evaluation (rolling-origin, 95% CI)")
                with open(eval_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                target_titles = {
                    "absolute": "Absolute target: demand in the top 25% of all slots",
                    "relative": "Relative target: surge above the zone's own usual level",
                }
                for horizon_key, hblock in ev["horizons"].items():
                    for target_key, block in hblock.get("targets", {"absolute": hblock}).items():
                        st.markdown(f"**{horizon_key} h ahead · {target_titles.get(target_key, target_key)}** — "
                                    f"{block['n_folds']} rolling folds, {block['n_test_days']} test days, "
                                    f"{block['n_test_rows']} test rows, {block.get('prevalence', 0):.0%} positives")
                        cols = ["model", "f1_ci", "fold_f1", "auc", "brier", "ece", "diff_ci", "significant"]
                        table = pd.DataFrame(block["models"]).reindex(columns=cols)
                        table.columns = ["Model", "F1 [95% CI]", "F1 per fold", "AUC", "Brier", "ECE",
                                         "Δ F1 vs historical rate", "Significant"]
                        st.dataframe(table, use_container_width=True, hide_index=True)
                        swap = block.get("zone_swap", {}).get("Random Forest (no lags)")
                        if swap:
                            st.caption(
                                f"Zone_ID weight in the model: {block['importances'].get('Zone_ID', 0):.0%}. "
                                f"If the hub is matched to the WRONG zone, F1 changes from {swap['f1_true']:.2f} "
                                f"to {swap['f1_swapped']:.2f} ({swap['drop']:+.2f}) for the no-lags model."
                            )
                st.caption("Significant = the 95% CI of the F1 difference vs the baseline excludes 0. "
                           "'Always alert (no skill)' is the minimum a useful model must beat. "
                           "AUC is comparable across targets (0.5 = chance). "
                           "ECE near 0 = predicted probabilities match observed frequencies. "
                           "Generated by `python src/evaluate.py`.")

            sens_path = "models/sensitivity_results.json"
            if (not is_direct) and os.path.exists(sens_path):
                st.markdown("---")
                st.subheader("🎚️ Sensitivity of the Cotonou → zone transfer")
                with open(sens_path, "r", encoding="utf-8") as f:
                    sens = json.load(f)
                sens_df = pd.DataFrame(sens["hubs"])[[
                    "hub", "assigned_zone", "runner_up_zone", "stability_pct", "expected_shift_noise_pts",
                    "shift_vs_runner_up_pts", "alert_flip_vs_runner_up_pct", "verdict"]]
                sens_df.columns = ["Hub", "Zone", "Runner-up", "Stability (%)", "Expected risk shift (pts)",
                                   "Shift vs runner-up (pts)", "Alert flips vs runner-up (%)", "Verdict"]
                st.dataframe(sens_df, use_container_width=True, hide_index=True)
                st.caption(f"Archetype targets perturbed with noise sd = {sens['noise_sd']} "
                           f"({sens['n_sim']} simulations). Stability = share of simulations where the hub keeps its "
                           "zone. Generated by `python src/sensitivity.py`.")

            st.subheader("Limits")
            if is_direct:
                limit_lines = [
                    f"- Training data come from local measurements ({source_name}): coverage, sensor quality and period matter.",
                    "- The learned rain effect depends on how many heavy-rain hours the local period contains.",
                    "- The target is the top 25% of the measured load: check that it reflects congestion for your measure.",
                ]
            else:
                limit_lines = [
                    "- Training data come from outside Cotonou: local validation is essential.",
                    "- The learned rain effect is that of the training location (temperate climate); its effect on tropical traffic is not demonstrated.",
                    "- The target measures **high demand** (top 25%), not congestion directly.",
                    "- The Cotonou weather forecast is real, but it feeds a model learned elsewhere.",
                ]
            limit_lines.append("- The model reflects hourly and weekly regularities, not the instantaneous state of traffic.")
            st.markdown("\n".join(limit_lines))

        # ---------------- Forecast ----------------
        with tab_fc:
            render_forecast_tab(COTONOU_HUBS)

        # ---------------- Field data ----------------
        with tab_field:
            render_field_tab(COTONOU_HUBS, model, FEATURE_COLUMNS)

    if open_report:
        show_guidelines_modal(selected_location, risk, heavy_rain_flag, is_market_day, is_critical, market_multiplier)

except FileNotFoundError:
    st.error("Model files not found. Run `python src/model.py` from the project root first.")
except Exception as e:
    st.error(f"Unexpected error: {e}")