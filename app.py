import json
import os
import pickle
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from weather import fetch_forecast, HEAVY_RAIN_MM  # noqa: E402

st.set_page_config(page_title="SmartTransit Platform", layout="wide")

FEATURE_COLUMNS = ["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]
COTONOU_LAT, COTONOU_LON = 6.37, 2.39

DAYS = ["Monday", "Tuesday", "Wednesday (Dantokpa Market Day)", "Thursday",
        "Friday", "Saturday (Market Day)", "Sunday"]

# 1. PERMANENT LOG HISTORY INITIALIZATION
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []


def rain_label(mm):
    if mm < 0.1:
        return "aucune pluie"
    if mm < 2.5:
        return "pluie faible"
    if mm < HEAVY_RAIN_MM:
        return "pluie modérée"
    return "forte pluie"


# 2. POP-UP MODAL DEFINITION
@st.dialog("📋 Urban Planning & Traffic Mitigation Guidelines", width="large")
def show_guidelines_modal(loc, risk, rain, market, critical_flag, threshold_val, market_mult):
    st.write(f"### Prescriptive Action Report for: **{loc}**")

    if critical_flag:
        st.error(f"🚨 **CRITICAL CONGESTION WARNING: {risk:.1f}% Saturation Risk**")

        if rain == 1:
            st.markdown("### ⚠️ Recommandation Drainage :")
            st.write(f"Risque élevé de saturation imminente du réseau routier à **{loc}**. Activer immédiatement les pompes de relevage de la station principale et suspendre obligatoirement tous les chantiers de terrassement en cours dans ce secteur urbain.")

            st.markdown("### 🔧 Maintenance Infrastructures (Post-Inondation) :")
            st.write("Prioriser l'inspection urgente des voies pavées de Cotonou environnantes pour détecter, cartographier et traiter d'éventuels affouillements critiques sous la chaussée avant effondrement.")
        else:
            st.markdown("### 🚦 Régulation du Trafic :")
            st.write(f"Déployer immédiatement des brigades d'agents de régulation au niveau de l'intersection de **{loc}** pour gérer manuellement les flux de véhicules avant l'engorgement complet du réseau.")

        st.markdown("---")
        st.markdown("#### 🏛️ Long-Term Municipal Structural Policy")
        if market == 1 or (market_mult and ("Dantokpa" in loc or "Portuaire" in loc)):
            st.write("- **Logistical Decoupling:** Marketplace loading zones are oversaturating transit tracks. Plan a decentralized off-dock logistical hub to move freight handling outside systemic peak hours.")
        else:
            st.write("- **Capacity Elasticity Engineering:** Expand the road network's geometric absorption limits or introduce automated dynamic tidal lanes to accommodate exponential commuter growth.")
    else:
        st.success(f"🟢 **STABLE FLOW CONDITIONS: {risk:.1f}% Saturation Risk**")
        st.markdown("#### Baseline Operational Guidelines")
        st.write("Aucune intervention d'urgence requise dans ce secteur. Maintenir le protocole de surveillance et de télémétrie routière courante via les caméras de la municipalité.")
        st.write("- **Status:** The infrastructure corridor retains adequate geometric elasticity to seamlessly handle current transit velocity curves.")

    st.caption("⚠️ Simulation issue d'un modèle de démonstration (transfert de méthode). Validation terrain requise avant toute décision opérationnelle.")
    st.info("Dismiss or close this pop-up view to log this simulation trace permanently on the dashboard storage below.")


# 3. MODEL ARTIFACTS + FORECAST LOADING
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
    """Prévision réelle pour Cotonou ; None si le réseau est indisponible."""
    try:
        return fetch_forecast(COTONOU_LAT, COTONOU_LON, hours=48)
    except Exception:
        return None


try:
    model, kmeans, COTONOU_HUBS, zone_profiles, meta = load_production_models()

    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        st.error("Le modèle chargé ne correspond pas à cette version de l'application. Relance `python src/model.py` puis redémarre l'app.")
        st.stop()

    st.title("🌍 SmartTransit Africa: Multi-Profile Urban Analytics Platform")

    st.warning(
        "**Prototype de démonstration (transfert de méthode).** Le modèle est entraîné sur un jeu de données "
        "public de mobilité (hors Cotonou), faute de données locales, avec la pluie réelle observée sur ce lieu et "
        "ces dates. Chaque site de Cotonou est associé à la zone d'entraînement dont le profil de demande lui "
        "ressemble le plus. Les résultats illustrent la chaîne données → modèle → décision et ne constituent "
        "pas une prévision validée pour Cotonou."
    )

    tab1, tab2 = st.tabs([
        "📊 Public Policy & Decision Mode (Cotonou Hubs)",
        "🔬 Core Engineering & Model Diagnostics (Raw Data)",
    ])

    # =========================================================================
    # TAB 1: PUBLIC POLICY & PRESCRIPTIVE DECISION MODE (COTONOU)
    # =========================================================================
    with tab1:
        st.markdown("""
        ### Strategic Technology Transfer & Prescriptive Urban Planning
        This deployment profile projects behavioral simulation layers directly onto the infrastructure network of **Cotonou, Benin**.
        """)

        with st.sidebar.expander("⚙️ Customize AI Trigger Thresholds", expanded=True):
            st.markdown("_Fine-tune algorithmic decision parameters for Cotonou municipal rules:_")
            user_crit_threshold = st.slider("Critical Alert Threshold (%)", 20, 95, 50)
            market_multiplier = st.checkbox("Prioritize Marketplace Logistics", value=True)
            rain_penalty = st.slider(
                "Expert rain adjustment (+ points, optional)", 0, 30, 0,
                help="Facultatif. Le modèle utilise déjà la pluie réelle apprise à l'entraînement. "
                     "Ce réglage ajoute une majoration d'expert (hypothèse locale) en cas de forte pluie.",
            )

        col_inputs, col_visuals = st.columns(2)

        with col_inputs:
            st.subheader("Socio-Environmental Stressors")
            selected_location = st.selectbox("Select Target Cotonou Hub", list(COTONOU_HUBS.keys()))
            hub_data = COTONOU_HUBS[selected_location]

            st.caption(
                f"Profil : **{hub_data['archetype']}** → zone d'entraînement n°{hub_data['zone_id']} "
                f"(écart de profil {hub_data['distance']})."
            )

            forecast_df = load_forecast()
            mode_options = ["Scénario manuel", "Prévision météo réelle (Cotonou, Open-Meteo)"]
            mode = st.radio("Source du scénario", mode_options)

            use_forecast = mode == mode_options[1]
            if use_forecast and forecast_df is None:
                st.warning("Prévision indisponible (pas de connexion ?). Retour au scénario manuel.")
                use_forecast = False

            if use_forecast:
                labels = [
                    f"{t:%a %d/%m %Hh} — {mm:.1f} mm/h"
                    for t, mm in zip(forecast_df["time"], forecast_df["Rain_mm"])
                ]
                choice = st.selectbox("Créneau prévu (48 h à venir)", labels)
                row = forecast_df.iloc[labels.index(choice)]
                day_index = int(row["time"].weekday())
                selected_hour = int(row["time"].hour)
                rain_mm = float(row["Rain_mm"])
                prob = row.get("Rain_prob")
                prob_txt = f", probabilité de pluie {prob:.0f}%" if pd.notna(prob) else ""
                st.caption(f"{DAYS[day_index].split(' (')[0]} {selected_hour}h — {rain_label(rain_mm)} ({rain_mm:.1f} mm/h{prob_txt}).")
            else:
                selected_day = st.selectbox("Day of the Week (Cotonou Context)", DAYS)
                day_index = DAYS.index(selected_day)
                selected_hour = st.slider("Target Commuting Window Hour", 0, 23, 17, key="c_hour")
                rain_mm = st.slider("Pluie (mm/h)", 0.0, 30.0, 0.0, 0.5)
                st.caption(f"Équivalent : {rain_label(rain_mm)} (forte pluie à partir de {HEAVY_RAIN_MM} mm/h).")

            is_market_day = 1 if day_index in [2, 5] else 0
            heavy_rain_flag = 1 if rain_mm >= HEAVY_RAIN_MM else 0

            run_c_sim = st.button("🚀 Run Prescriptive Policy Simulation", use_container_width=True)

        with col_visuals:
            st.subheader("Geospatial Node Tracking Map")

            map_color = "#0000FF"  # Neutral blue at startup
            saturation_risk_percentage = 0.0
            is_critical = False

            input_vector = pd.DataFrame(
                [[selected_hour, day_index, hub_data["zone_id"], rain_mm]],
                columns=FEATURE_COLUMNS,
            )

            if run_c_sim:
                base_risk = float(model.predict_proba(input_vector)[0][1]) * 100
                extra = rain_penalty if heavy_rain_flag else 0
                saturation_risk_percentage = min(100.0, base_risk + extra)
                is_critical = bool(saturation_risk_percentage >= user_crit_threshold)
                map_color = "#FF0000" if is_critical else "#00FF00"

            map_df = pd.DataFrame([{
                "lat": hub_data["lat"],
                "lon": hub_data["lon"],
                "Location": selected_location,
                "color": map_color,
            }])
            st.map(map_df, latitude="lat", longitude="lon", color="color", zoom=13)

            if run_c_sim:
                txt = f"Risque modèle : {base_risk:.1f}% (pluie {rain_mm:.1f} mm/h)."
                if extra:
                    txt += f" + majoration d'expert : +{extra} pts."
                st.caption(txt)

                log_entry = {
                    "Location": selected_location,
                    "Risk": f"{saturation_risk_percentage:.1f}%",
                    "Status": "CRITICAL 🔴" if is_critical else "STABLE 🟢",
                    "Details": "",
                }

                if is_critical and heavy_rain_flag == 1:
                    log_entry["Details"] = (
                        f"⚠️ **Recommandation Drainage:** Activer les pompes de relevage et suspendre les chantiers de terrassement à {selected_location}.\n\n"
                        "🔧 **Maintenance Infrastructures:** Inspection prioritaire urgente des voies pavées de Cotonou contre les affouillements de chaussée."
                    )
                elif is_critical:
                    log_entry["Details"] = f"🚦 **Régulation Routière d'Urgence:** Déploiement d'agents de circulation requis pour désengorger {selected_location}."
                else:
                    log_entry["Details"] = f"✅ Flux fluide et stable enregistré pour {selected_location}. RAS (Surveillance de routine)."

                st.session_state.recommendation_history.insert(0, log_entry)

                show_guidelines_modal(
                    selected_location, saturation_risk_percentage, heavy_rain_flag,
                    is_market_day, is_critical, user_crit_threshold, market_multiplier,
                )

        # PERSISTENT LOGS REGISTRY
        st.markdown("---")
        st.subheader("📋 Active Municipal Log & Guidelines Archive")

        if st.session_state.recommendation_history:
            total = len(st.session_state.recommendation_history)
            for idx, entry in enumerate(st.session_state.recommendation_history):
                with st.expander(f"Log #{total - idx} — {entry['Location']} ({entry['Status']})", expanded=(idx == 0)):
                    st.markdown(f"**Algorithmic Traffic Saturation Probability:** `{entry['Risk']}`")
                    st.write(entry["Details"])
        else:
            st.info("No policy logs archived yet. Configure features on the left panel and click 'Run Policy Simulation' to trigger analytical records.")

    # =========================================================================
    # TAB 2: CORE ENGINEERING & TECHNICAL MODE
    # =========================================================================
    with tab2:
        st.markdown("""
        ### Production Engineering & Model Diagnostics Workspace
        This structural space exposes numerical matrix entry arrays, vector columns, and underlying algorithmic boundary conditions.
        """)

        col_eng_inputs, col_eng_metrics = st.columns(2)

        with col_eng_inputs:
            st.subheader("Raw Feature Vector")
            eng_hour = st.slider("Hour", 0, 23, 17, key="e_hour")
            eng_day = st.slider("Day_of_Week (0=Mon)", 0, 6, 2, key="e_day")
            eng_zone = st.selectbox("Zone_ID", sorted(zone_profiles.index.tolist()), key="e_zone")
            eng_rain = st.slider("Rain_mm (mm/h)", 0.0, 30.0, 0.0, 0.5, key="e_rain")

        with col_eng_metrics:
            st.subheader("Model Output")
            eng_vector = pd.DataFrame([[eng_hour, eng_day, eng_zone, eng_rain]], columns=FEATURE_COLUMNS)
            if st.button("Run Diagnostic", key="e_run"):
                proba = float(model.predict_proba(eng_vector)[0][1]) * 100
                st.metric("Saturation probability", f"{proba:.1f}%")
                st.dataframe(eng_vector)

            st.markdown("#### Feature importances")
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
            st.bar_chart(importances)

        st.markdown("---")
        st.subheader("🌧️ Météo réelle et validation")
        m1, m2, m3 = st.columns(3)
        m1.metric("F1 modèle (test chronologique)", f"{meta['f1_model']:.2f}")
        m2.metric("F1 référence naïve", f"{meta['f1_baseline']:.2f}")
        m3.metric("Effet appris d'une forte pluie", f"{meta['rain_effect_points']:+.1f} pts")
        st.caption(
            f"Pluie réelle {meta['weather_source']} au point ({meta['weather_lat']}, {meta['weather_lon']}), "
            f"du {meta['date_start']} au {meta['date_end']} : {meta['rain_hours_share']:.0%} d'heures pluvieuses, "
            f"{meta['heavy_rain_hours']} heures de forte pluie (≥ {meta['heavy_rain_mm']} mm/h). "
            f"Entraînement sur {meta['n_train_days']} jours, test sur {meta['n_test_days']} jours."
        )
        if meta["heavy_rain_hours"] < 20:
            st.warning("Très peu d'épisodes de forte pluie dans les données d'entraînement : l'effet appris de la pluie est peu fiable.")
        if meta["rain_effect_points"] <= 0:
            st.warning(
                "L'effet appris d'une forte pluie est nul ou négatif. C'est cohérent avec une demande de courses "
                "qui évolue avec la météo du lieu d'entraînement, mais cela ne dit rien du comportement de la "
                "circulation à Cotonou. Ne pas interpréter comme une preuve."
            )

        st.markdown("---")
        st.subheader("🔗 Transfert Cotonou → zones d'entraînement")
        st.write(
            "Chaque site de Cotonou est associé à la zone d'entraînement dont le profil de demande "
            "(volume, part des heures de pointe, part de nuit) est le plus proche de l'archétype recherché. "
            "L'affectation est unique (un site = une zone) et calculée à l'entraînement."
        )
        mapping_df = pd.DataFrame([
            {
                "Site (Cotonou)": name,
                "Archétype": h["archetype"],
                "Critère": h["criteria"],
                "Zone": h["zone_id"],
                "Rang demande zone": h["zone_demand_rank"],
                "Part pointe": h["zone_peak_share"],
                "Part nuit": h["zone_night_share"],
                "Écart profil": h["distance"],
            }
            for name, h in COTONOU_HUBS.items()
        ])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        with st.expander("Profils de toutes les zones d'entraînement"):
            st.dataframe(zone_profiles, use_container_width=True)

        st.subheader("Limites")
        st.markdown(
            "- Données d'entraînement hors Cotonou : validation locale indispensable.\n"
            "- La pluie apprise est celle du lieu d'entraînement (climat tempéré) ; son effet sur la circulation tropicale n'est pas démontré.\n"
            "- La cible mesure une **demande élevée** (top 25 %), pas directement la congestion.\n"
            "- La prévision météo de Cotonou est réelle, mais elle alimente un modèle appris ailleurs.\n"
            "- Le modèle reflète des régularités horaires et hebdomadaires, pas l'état instantané du trafic."
        )

except FileNotFoundError:
    st.error("Fichiers du modèle introuvables. Exécute d'abord `python src/model.py` depuis la racine du projet.")
except Exception as e:
    st.error(f"Erreur inattendue : {e}")