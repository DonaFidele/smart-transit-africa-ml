import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="SmartTransit Africa", layout="centered")

st.title("SmartTransit Africa: Urban Congestion Predictive System")
st.markdown("""
This system implements machine learning concepts to anticipate urban traffic saturation.
It integrates geospatial clustering (K-Means) and ensemble learning (Random Forest) to model gridlock patterns based on temporal, environmental, and infrastructure constraints.
""")

@st.cache_resource
def load_production_models():
    with open("models/traffic_rf_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("models/kmeans_zones.pkl", "rb") as f:
        clusterer = pickle.load(f)
    return classifier, clusterer

try:
    model, kmeans = load_production_models()
    
    st.sidebar.header("Simulation Parameters")
    
    days = ["Monday", "Tuesday", "Wednesday (Market Day)", "Thursday", "Friday", "Saturday (Market Day)", "Sunday"]
    selected_day = st.sidebar.selectbox("Day of the Week", days)
    day_index = days.index(selected_day)
    
    is_market_day = 1 if day_index in [2, 5] else 0
    selected_hour = st.sidebar.slider("Target Commuting Hour", 0, 23, 17)
    
    weather_condition = st.sidebar.radio("Precipitation Intensity", ["Dry Conditions", "Heavy Downpour"])
    heavy_rain_flag = 1 if weather_condition == "Heavy Downpour" else 0
    
    st.sidebar.subheader("Target Coordinates")
    latitude = st.sidebar.number_input("Latitude Reference", value=40.7588)
    longitude = st.sidebar.number_input("Longitude Reference", value=-73.9851)

    predicted_zone = kmeans.predict(np.array([[latitude, longitude]]))[0]
    st.info(f"Coordinates mapped to Transport Hub ID: {predicted_zone}")

    if st.button("Run Simulation"):
        input_data = pd.DataFrame([[selected_hour, day_index, predicted_zone, is_market_day, heavy_rain_flag]], 
                                 columns=["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"])
        
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        st.subheader("Simulation Analysis Output:")
        
        if prediction == 1:
            st.error(f"Critical Congestion Warning: High Saturation Risk ({probability * 100:.1f}%)")
            st.write("Compounding factors (peak hour, geographic constraints, environmental stressors) indicate a high probability of systemic bottleneck.")
        else:
            st.success(f"Stable Traffic Estimate: Low Saturation Risk ({probability * 100:.1f}%)")
            st.write("Urban flow capacity is sufficient to handle projected demand volumes under current conditions.")

except FileNotFoundError:
    st.error("Infrastructure Error: Trained model artifacts not found. Please run 'python src/model.py' to generate binary files.")
