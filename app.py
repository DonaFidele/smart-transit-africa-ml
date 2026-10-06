import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="SmartTransit Platform", layout="wide")

# 1. Loading production model artifacts
@st.cache_resource
def load_production_models():
    with open("models/traffic_rf_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        clusterer = pickle.load(f)
    return classifier, clusterer

try:
    model, kmeans = load_production_models()

    # 2. Main multi-profile navigation bar
    st.title("🌍 SmartTransit Africa: Multi-Profile Urban Analytics")
    tab1, tab2 = st.tabs(["📊 Public Policy & Decision Mode (Cotonou)", "🔬 Core Engineering & Technical Mode (Raw Data)"])

    # =========================================================================
    # TAB 1: PUBLIC POLICY & DECISION MODE (COTONOU)
    # =========================================================================
    with tab1:
        st.markdown("""
        ### Strategic Technology Transfer Demonstration
        This profile projects behavior simulation layers onto the infrastructure network of **Cotonou, Benin**. 
        It filters complex machine learning inferences into operational insights for municipal authorities and non-technical stakeholders.
        """)
        
        COTONOU_HUBS = {
            "Échangeur de Godomey (West Gateway)": {"lat": 6.3811, "lon": 2.3522, "zone_id": 1},
            "Grand Marché de Dantokpa (Commercial Hub)": {"lat": 6.3708, "lon": 2.4344, "zone_id": 5},
            "Carrefour Vèdoko (Central Junction)": {"lat": 6.3754, "lon": 2.3881, "zone_id": 3},
            "Carrefour Cadjehoun (Avenue Jean-Paul II)": {"lat": 6.3575, "lon": 2.3980, "zone_id": 8},
            "Zone Portuaire / Akpakpa (Logistics Hub)": {"lat": 6.3650, "lon": 2.4490, "zone_id": 12}
        }
        
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
            
            # Pre-compute inference vector to dynamically color code the map marker
            input_data = pd.DataFrame([[selected_hour, day_index, hub_data["zone_id"], is_market_day, heavy_rain_flag]], 
                                     columns=["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"])
            
            probability = model.predict_proba(input_data)
            prediction = model.predict(input_data)
            
            # Target precisely the probability of Class 1 (Saturation Risk)
            saturation_risk_percentage = probability[0][1] * 100
            
            # Dynamic Hex Colors: Red for critical gridlock risk (>50%), Green for optimal traffic flow
            map_color = "#FF0000" if prediction == 1 else "#00FF00"
            
            map_df = pd.DataFrame([{
                "lat": hub_data["lat"], 
                "lon": hub_data["lon"], 
                "Location": selected_location,
                "color": map_color
            }])
            
            st.map(map_df, latitude="lat", longitude="lon", color="color", zoom=13)
            
            st.subheader("Operational Risk Output")
            if run_c_sim:
                if prediction == 1:
                    st.error(f"🔴 **CRITICAL SATURATION ALERT ({saturation_risk_percentage:.1f}% Congestion Risk)**")
                    st.markdown(f"""
                    **Executive Summary for {selected_location}:**  
                    The predictive model indicates a severe threat of systemic gridlock. The compounding effect of the selected temporal window and environmental stressors exceeds the infrastructure's absorption threshold.
                    
                    **Analytical Breakdown:**
                    - **Socio-Economic Catalyst:** The structural traffic baseline is heavily congested.
                    - **Environmental Impact:** Heavy rainfall or localized flash floods have critically reduced free-flow vehicle velocity, accelerating network decay.
                    - **Mitigation Recommendation:** Immediate deployment of traffic traffic wardens or automated proactive dynamic routing signaling is highly advised for this corridor.
                    """)
                else:
                    st.success(f"🟢 **OPERATIONAL FLOW STABLE ({saturation_risk_percentage:.1f}% Congestion Risk)**")
                    st.markdown(f"""
                    **Executive Summary for {selected_location}:**  
                    Urban mobility flow remains within safe, elastic parameters. The risk of major delays or systemic network saturation is minimal.
                    
                    **Analytical Breakdown:**
                    - **Structural Capacity:** The network retains sufficient geometric elasticity to seamlessly absorb current local transit demands.
                    - **Stress Test Status:** Even under potential climate stressors, the selected time frame avoids critical peak accumulation curves.
                    - **Operational Advice:** Normal municipal operations can proceed. No emergency deployment required.
                    """)
            else:
                st.info("ℹ️ Click the button on the left panel to execute the predictive engine and generate the analytical breakdown.")

    # =========================================================================
    # TAB 2: CORE ENGINEERING & TECHNICAL MODE (RAW DATA)
    # =========================================================================
    with tab2:
        st.markdown("""
        ### Production Engineering & Model Diagnostics
        This profile exposes direct matrix inputs, feature spaces, and mathematical evaluation thresholds. 
        Ideal for validating pipeline integrity, troubleshooting overfitting, and reading underlying probabilities.
        """)
        
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("Raw Mathematical Vectors")
            raw_hour = st.slider("Feature: Hour (0-23)", 0, 23, 17, key="e_hour")
            raw_day = st.slider("Feature: Day_of_Week (0-6)", 0, 6, 3, help="0=Monday, 6=Sunday")
            raw_zone = st.number_input("Feature: Zone_ID (K-Means Cluster Index)", min_value=0, max_value=14, value=1)
            
            st.markdown("---")
            raw_market = st.checkbox("Flag: Local_Market_Day Active (1)")
            raw_rain = st.checkbox("Flag: Heavy_Rain Active (1)")
            
            run_e_sim = st.button("🔬 Execute Array Inference", use_container_width=True)
            
        with e2:
            st.subheader("Matrix Execution & Output Array")
            
            feature_vector = pd.DataFrame([[raw_hour, raw_day, raw_zone, int(raw_market), int(raw_rain)]],
                                         columns=["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"])
            
            st.markdown("**Inference Input Array Passed to Classifier:**")
            st.dataframe(feature_vector, use_container_width=True)
            
            if run_e_sim:
                raw_prob = model.predict_proba(feature_vector)
                raw_pred = model.predict(feature_vector)
                
                st.markdown("**Scikit-Learn Classifier Diagnostics:**")
                st.json({
                    "Model_Type": "RandomForestClassifier(n_estimators=150, max_depth=14)",
                    "Output_Class_Label": int(raw_pred),
                    "Probability_Distribution": {
                        "Class_0 (Fluid)": float(raw_prob[0][0]),
                        "Class_1 (Saturated)": float(raw_prob[0][1])
                    },
                    "Active_Decision_Threshold": "Quantile 0.75 Adaptive Gridlock Aggregation"
                })

except FileNotFoundError:
    st.error("Infrastructure Error: Trained model artifacts not found. Please run 'python src/model.py' to generate binary files.")
