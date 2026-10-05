# 🌍 SmartTransit Africa: Hybrid Machine Learning for Urban Mobility Optimization

[![Streamlit App](https://streamlit.io)](YOUR_STREAMLIT_APP_LINK_HERE)

## Vision & Problem Statement
Rapidly growing African metropolises face massive economic losses and environmental challenges due to severe traffic congestion. 
This project introduces a **predictive decision-support prototype** for urban planning. While trained on a massive volume of real-world transit data (Uber TLC), the data pipeline uniquely integrates contextual features tailored to emerging markets: **local market-day economic cycles** and **heavy rainy season disruptions**.

## Algorithmic Architecture (Stanford & DeepLearning.AI Validation)
This framework connects key machine learning pillars mastered during my **Stanford & DeepLearning.AI** specialization:

1. **Unsupervised Clustering (K-Means):** Instead of relying on static administrative boundaries, the algorithm dynamically maps the urban space into `K=15` high-activity transport hubs based on millions of geospatial coordinates (Latitude, Longitude).
2. **Contextual Feature Engineering:** Extraction of cyclical time components (Hours, Days) and injection of synthesized socio-economic factors (market days) and weather intensities.
3. **Supervised Ensemble Classification (Random Forest):** Training an ensemble classifier to predict a saturation probability score, using a threshold set at the 75th percentile of high-density demand.

## Evaluation & Metrics
Performance analysis prioritizes the **F1-Score** due to the natural class imbalance of urban traffic gridlocks:
- **Baseline Model (Logistic Regression):** F1-Score ~ 0.58
- **Final Optimized Model (Random Forest):** F1-Score ~ **0.86** 🚀

*Feature importance analysis shows that combined weather anomalies and peak commuting hours multiply the probability of systemic traffic saturation by a factor of 2.4.*

## Repository Structure
```text
├── data/               # Raw datasets (Uber TLC Dataset - gitignored)
├── models/             # Serialized artifact binary files (.pkl - gitignored)
├── src/
│   └── model.py        # Data Engineering & Machine Learning Pipeline
├── app.py              # Interactive Web Interface (Streamlit)
└── requirements.txt    # Production environment dependencies
```

## Installation & Deployment
1. Install dependencies: `pip install -r requirements.txt`
2. Run the training and data pipeline: `python src/model.py`
3. Launch the interactive simulation interface: `streamlit run app.py`

## Author & Credentials
- **Name:** Dona Fidele Houekpoeha
- **Credentials:** Verified Specialist in Machine Learning (**Stanford University & DeepLearning.AI**)
