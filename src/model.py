import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

def main():
    data_path = "data/urban_mobility_raw_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Missing dataset file at '{data_path}'")

    # Load data and optimize memory
    df = pd.read_csv(data_path, usecols=["Date/Time", "Lat", "Lon"])
    df = df.sample(n=200000, random_state=42)

    # Feature Engineering
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df["Hour"] = df["Date/Time"].dt.hour
    df["Day_of_Week"] = df["Date/Time"].dt.dayofweek
    df["Day_of_Month"] = df["Date/Time"].dt.day

    # Contextual variables simulation
    df["Local_Market_Day"] = df["Day_of_Week"].isin([2, 5]).astype(int)
    
    np.random.seed(42)
    df["Heavy_Rain"] = np.random.choice([0, 1], size=len(df), p=[0.80, 0.20])

    # Spatial clustering
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    df["Zone_ID"] = kmeans.fit_predict(df[["Lat", "Lon"]])

    # Target engineering based on demand density percentile
    traffic_aggregation = df.groupby(["Day_of_Month", "Day_of_Week", "Hour", "Zone_ID", "Local_Market_Day", "Heavy_Rain"]).size().reset_index(name="Demand_Volume")
    
    saturation_threshold = traffic_aggregation["Demand_Volume"].quantile(0.75)
    traffic_aggregation["Traffic_Saturation_Risk"] = (traffic_aggregation["Demand_Volume"] > saturation_threshold).astype(int)

    # Dataset splitting
    X = traffic_aggregation[["Hour", "Day_of_Week", "Zone_ID", "Local_Market_Day", "Heavy_Rain"]]
    y = traffic_aggregation["Traffic_Saturation_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model training
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42)
    rf_model.fit(X_train, y_train)

    # Validation metrics
    predictions = rf_model.predict(X_test)
    print(f"Model Training Complete. Test F1-Score: {f1_score(y_test, predictions):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Export binaries
    os.makedirs("models", exist_ok=True)
    with open("models/traffic_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/kmeans_zones.pkl", "wb") as f:
        pickle.dump(kmeans, f)

if __name__ == "__main__":
    main()
