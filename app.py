import json
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="SmartTransit Platform", layout="wide")

FEATURE_COLUMNS = ["Hour", "Day_of_Week", "Zone_ID"]

# 1. PERMANENT LOG HISTORY INITIALIZATION
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []


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


# 3. SECURE PRODUCTION MODEL ARTIFACTS LOADING
@st.cache_resource
def load_production_models():
    with open("models/traffic_rf_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        clusterer = pickle.load(f)
    with open("models/hub_zone_mapping.json", "r", encoding="utf-8") as f:
        hubs = json.load(f)
    profiles = pd.read_csv("models/zone_profiles.csv", index_col=0)
    return classifier, clusterer, hubs, profiles


try:
    model, kmeans, COTONOU_HUBS, zone_profiles = load_production_models()

    if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
        st.error("Le modèle chargé ne correspond pas à cette version de l'application. Relance `python train_model.py` puis redémarre l'app.")
        st.stop()

    st.title("🌍 SmartTransit Africa: Multi-Profile Urban Analytics Platform")

    st.warning(
        "**Prototype de démonstration (transfert de méthode).** Le modèle est entraîné sur un jeu de données "
        "public de mobilité (hors Cotonou), faute de données locales. Chaque site de Cotonou est associé à la "
        "zone d'entraînement dont le profil de demande lui ressemble le plus (voir onglet Ingénierie). "
        "Les résultats illustrent la chaîne complète données → modèle → décision et ne constituent pas "
        "une prévision validée pour Cotonou."
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
                "Rain adjustment (+ points of risk)", 0, 30, 10,
                help="Hypothèse d'expert, NON apprise par le modèle : la pluie n'existe pas dans les données d'entraînement. "
                     "Ce réglage est appliqué comme règle de décision après la prédiction.",
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

            days = ["Monday", "Tuesday", "Wednesday (Dantokpa Market Day)", "Thursday",
                    "Friday", "Saturday (Market Day)", "Sunday"]
            selected_day = st.selectbox("Day of the Week (Cotonou Context)", days)
            day_index = days.index(selected_day)

            is_market_day = 1 if day_index in [2, 5] else 0
            selected_hour = st.slider("Target Commuting Window Hour", 0, 23, 17, key="c_hour")
            weather_condition = st.radio(
                "Precipitation Intensity Index",
                ["Dry Conditions / Seasonal Normal", "Heavy Downpour / Flash Floods (Rainy Season Storm)"],
            )
            heavy_rain_flag = 1 if weather_condition.startswith("Heavy Downpour") else 0

            run_c_sim = st.button("🚀 Run Prescriptive Policy Simulation", use_container_width=True)

        with col_visuals:
            st.subheader("Geospatial Node Tracking Map")

            map_color = "#0000FF"  # Neutral blue at startup
            saturation_risk_percentage = 0.0
            is_critical = False

            input_vector = pd.DataFrame(
                [[selected_hour, day_index, hub_data["zone_id"]]],
                columns=FEATURE_COLUMNS,
            )

            if run_c_sim:
                base_risk = float(model.predict_proba(input_vector)[0][1]) * 100
                # Règle d'expert : la pluie majore le risque (non apprise par le modèle)
                saturation_risk_percentage = min(100.0, base_risk + (rain_penalty if heavy_rain_flag else 0))
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
                if heavy_rain_flag and rain_penalty:
                    st.caption(f"Risque modèle : {base_risk:.1f}% + ajustement pluie (règle d'expert) : +{rain_penalty} pts.")
                else:
                    st.caption(f"Risque modèle : {base_risk:.1f}%.")

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

        with col_eng_metrics:
            st.subheader("Model Output")
            eng_vector = pd.DataFrame([[eng_hour, eng_day, eng_zone]], columns=FEATURE_COLUMNS)
            if st.button("Run Diagnostic", key="e_run"):
                proba = float(model.predict_proba(eng_vector)[0][1]) * 100
                st.metric("Saturation probability", f"{proba:.1f}%")
                st.dataframe(eng_vector)

            st.markdown("#### Feature importances")
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
            st.bar_chart(importances)

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
            "- La cible mesure une **demande élevée** (top 25 %), pas directement la congestion.\n"
            "- La pluie n'est **pas apprise** : elle intervient comme règle d'expert paramétrable.\n"
            "- Le modèle reflète des régularités horaires et hebdomadaires, pas une prévision en temps réel."
        )

except FileNotFoundError:
    st.error("Modèles introuvables. Exécute d'abord `python train_model.py` pour générer le dossier `models/`.")
except Exception as e:
    st.error(f"Erreur inattendue : {e}")