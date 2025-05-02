# 🌆 Urban Heat Island (UHI) Predictor – CPU Version

A lightweight, containerized machine learning dashboard to analyze and predict Urban Heat Island (UHI) effects using greening effort simulations and clustering — powered by Scikit-learn and Streamlit.

---

## 🚀 Features

- 🔥 Predict neighborhood-level UHI intensity (temperature anomaly)
- 🌱 What-If Simulation: Adjust green cover and analyze the impact
- 🧠 ML Models: Linear Regression, Random Forest, Gradient Boosting, MLP
- 🔍 Clustering: Identify similar neighborhoods based on heat and greenery
- 📊 Interactive dashboard with real-time input (via Streamlit)

---

## 🗂️ Project Structure

uhi-predictor/
├── app/
│ ├── main.py # Streamlit dashboard UI
│ ├── model_utils.py # ML model training, prediction, clustering
│ └── data/
│ └── Louisville_Metro_KY_-_Urban_Heat_Island_Neighborhood_Data.csv
├── Dockerfile # Docker setup (CPU version)
├── requirements.txt # Python dependencies
└── docker-compose.yml # Compose config to run the app

## 🧪 Key Libraries
streamlit – for the interactive dashboard

pandas, numpy – for data processing

scikit-learn – ML models & clustering

matplotlib, seaborn – optional for future plots

## 📊 About the Dataset
Source-https://catalog.data.gov/dataset/louisville-metro-ky-urban-heat-island-neighborhood-data
Neighborhood-level data from Louisville, KY.
Features include:

Urban temperature readings

Green cover percentages

Environmental and demographic factors

## 🛠️ Future Enhancements
Add visualization of clusters on a map

Export predictions as CSV

Enable file uploads for new datasets

