import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="SmartTransit Platform", layout="wide")

# Initialize session state for persistent log history
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []

@st.cache_resource
def load_production_models():
    with open("models/traffic_rf_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        clusterer = pickle.load(f)
    return classifier, clusterer

try:
    model, kmeans = load_production_models()

    st.title("🌍 SmartTransit Africa: Multi-Profile Urban Analytics")
    tab1, tab2 = st.tabs(["📊 Public Policy & Decision Mode (Cotonou)", "🔬 Core Engineering & Technical Mode (Raw Data)"])

    # =========================================================================
    # TAB 1: PUBLIC POLICY & DECISION MODE (COTONOU)
    # =========================================================================
    with tab1:
        st.markdown("""
        ### Strategic Technology Transfer & Prescriptive Urban Planning
        This profile projects behavioral simulation layers onto the infrastructure network of **Cotonou, Benin**.
        """)
        
        COTONOU_HUBS = {
            "Échangeur de Godomey (West Gateway)": {"lat": 6.3811, "lon": 2.3522, "zone_id": 1},
            "Grand Marché de Dantokpa (Commercial Hub)": {"lat": 6.3708, "lon": 2.4344, "zone_id": 5},
            "Carrefour Vèdoko (Central Junction)": {"lat": 6.3754, "lon": 2.3881, "zone_id": 3},
            "Carrefour Cadjehoun (Avenue Jean-Paul II)": {"lat": 6.3575, "lon": 2.3980, "zone_id": 8},
            "Zone Portuaire / Akpakpa (Logistics Hub)": {"lat": 6.3650, "lon": 2.4490, "zone_id": 12}
        }
        
        with st.sidebar.expander("⚙️ Customize AI Trigger Thresholds", expanded=False):
            user_crit_threshold = st.slider("Critical Alert Threshold (%)", 40, 95, 50)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Socio-Environmental Stressors")
            selected_location = st.selectbox("Select Target Cotonou Hub", list(COTONOU_HUBS.keys()))
            hub_data = COTONOU_HUBS[selected_location]
            
            days = ["Monday", "Tuesday", "Wednesday (Dantokpa Market Day)", "Thursday", "Friday", "Saturday (Market Day)", "Sunday"]
            selected_day = st.selectbox("Day of the Week (Cotonou)", days)
            day_index = days.index(selected_day)
            is_market_day = 1 if day_index in [2, 5] else 0

            selected_hour = st.slider("Commuting Window Hour", 0, 23, 17, key="c_hour")
            weather_condition = st.radio("Precipitation Intensity Index", ["Dry Conditions", "Heavy Downpour / Flash Floods"])
            heavy_rain_flag = 1 if weather_condition == "Heavy Downpour / Flash Floods" else 0
            
            run_c_sim = st.button("🚀 Run Policy Simulation", use_container_width=True)
            
        with c2:
            st.subheader("Geospatial Node Tracking")
            
            input_data = pd.DataFrame([[selected_hour, day_index, hub_data["zone_id"], is_market_day, heavy_rain_flag]], 
                                     columns=["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"])
            
            probability = model.predict_proba(input_data)
            prediction = model.predict(input_data)
            saturation_risk_percentage = probability * 100
            
            is_critical = saturation_risk_percentage >= user_crit_threshold
            map_color = "#FF0000" if is_critical else "#00FF00"
            
            map_df = pd.DataFrame([{"lat": hub_data["lat"], "lon": hub_data["lon"], "Location": selected_location, "color": map_color}])
            st.map(map_df, latitude="lat", longitude="lon", color="color", zoom=13)
            
            # Modal Pop-up Definition using user's explicit guidelines
            @st.dialog("📋 Urban Planning & Traffic Mitigation Guidelines", width="large")
            def show_guidelines_popup(loc, risk, rain, market):
                st.write(f"### Prescriptive Action Report: {loc}")
                if risk >= user_crit_threshold:
                    st.error(f"🚨 **CRITICAL CONGESTION RISK: {risk:.1f}%**")
                    if rain == 1:
                        st.markdown("### ⚠️ Recommandation Drainage :")
                        st.write(f"Risque de saturation du réseau à **{loc}**. Activer immédiatement les pompes de relevage de la station principale et suspendre tous les chantiers de terrassement en cours dans ce secteur.")
                        st.markdown("### 🔧 Maintenance Infrastructures (Post-Inondation) :")
                        st.write("Prioriser l'inspection urgente des voies pavées de Cotonou environnantes pour détecter et traiter d'éventuels affouillements sous la chaussée.")
                    else:
                        st.markdown("### 🚦 Régulation du Trafic :")
                        st.write(f"Déployer des agents de régulation au niveau de **{loc}** pour gérer les flux avant blocage complet.")
                else:
                    st.success(f"🟢 **STABLE FLOW CONDITIONS: {risk:.1f}% RISK**")
                    st.write("Aucune intervention d'urgence requise. Surveillance de routine via les caméras de téléphométrie.")
                st.info("Close this window to save this report into the platform's history log.")

            # Logic to handle execution and persistent logging
            if run_c_sim:
                # 1. Generate text for the persistent log history
                log_entry = {
                    "Location": selected_location,
                    "Risk": f"{saturation_risk_percentage:.1f}%",
                    "Status": "CRITICAL 🔴" if is_critical else "STABLE 🟢",
                    "Details": ""
                }
                if is_critical and heavy_rain_flag == 1:
                    log_entry["Details"] = f"⚠️ **Recommandation Drainage:** Activer les pompes de relevage et suspendre les chantiers de terrassement à {selected_location}. \n\n🔧 **Maintenance Infrastructures:** Inspection prioritaire des voies pavées de Cotonou contre les affouillements."
                elif is_critical:
                    log_entry["Details"] = f"🚦 **Régulation Routière:** Déploiement d'urgence d'agents de circulation à {selected_location}."
                else:
                    log_entry["Details"] = f"✅ Flux fluide enregistré pour {selected_location}. RAS."
                
                # Append to persistent state history list
                st.session_state.recommendation_history.insert(0, log_entry)
                
                # 2. Trigger the modal popup window
                show_guidelines_popup(selected_location, saturation_risk_percentage, heavy_rain_flag, is_market_day)

        # Permanent Trace/Log Section at the bottom of Tab 1
        st.markdown("---")
        st.subheader("📋 Active Municipal Log & Guidelines Archive")
        st.markdown("_This section keeps a permanent trace of all recommendations generated during your session after pop-ups are dismissed:_")
        
        if st.session_state.recommendation_history:
            for idx, entry in enumerate(st.session_state.recommendation_history):
                with st.expander(f"Log #{len(st.session_state.recommendation_history) - idx} - {entry['Location']} ({entry['Status']})", expanded=(idx==0)):
                    st.write(f"**Calculated Congestion Risk:** {entry['Risk']}")
                    st.write(entry['Details'])
        else:
            st.info("No logs generated yet. Run a policy simulation above to archive prescriptive traces here.")

    # =========================================================================
    # TAB 2: CORE ENGINEERING & TECHNICAL MODE
    # =========================================================================
    with tab2:
        st.markdown("### Production Engineering & Model Diagnostics")
        e1, e2 = st.columns(2)
        with e1:
            raw_hour = st.slider("Feature: Hour (0-23)", 0, 23, 17, key="e_hour")
            raw_day = st.slider("Feature: Day_of_Week (0-6)", 0, 6, 3)
            raw_zone = st.number_input("Feature: Zone_ID", min_value=0, max_value=14, value=1)
            raw_market = st.checkbox("Flag: Local_Market_Day Active")
            raw_rain = st.checkbox("Flag: Heavy_Rain Active")
            run_e_sim = st.button("🔬 Execute Array Inference", use_container_width=True)
            
        with e2:
            feature_vector = pd.DataFrame([[raw_hour, raw_day, raw_zone, int(raw_market), int(raw_rain)]],
                                         columns=["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"])
            st.dataframe(feature_vector, use_container_width=True)
            if run_e_sim:
                raw_prob = model.predict_proba(feature_vector)
                raw_pred = model.predict(feature_vector)
                st.json({
                    "Model_Type": "RandomForestClassifier",
                    "Output_Class_Label": int(raw_pred),
                    "Probability_Distribution": {"Class_0 (Fluid)": float(raw_prob[0][0]), "Class_1 (Saturated)": float(raw_prob[0][1])}
                })

except FileNotFoundError:
    st.error("Infrastructure Error: Trained model artifacts not found. Please run 'python src/model.py'.")
