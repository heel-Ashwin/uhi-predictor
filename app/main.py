import streamlit as st
import model_utils

st.title("🌆 Urban Heat Island (UHI) Predictor")

# Load data
df = model_utils.load_data()
st.write("### Column Names in Dataset:")
st.write(df.columns.tolist())
st.dataframe(df.head())

# MODULE 1 - Train and Evaluate Models
st.header("📊 Module 1: Train-Test Split & Model Evaluation")
X_train, X_test, y_train, y_test = model_utils.split_data(df)

models = {
    "Linear Regression": model_utils.train_linear_regression,
    "Random Forest": model_utils.train_random_forest,
    "Gradient Boosting": model_utils.train_gradient_boosting,
    "MLP Regressor": model_utils.train_mlp,
}
if model_utils.xgboost_installed:
    models["XGBoost"] = model_utils.train_xgboost

results = {}
best_r2 = float("-inf")
best_model = None
best_model_name = None

for name, train_fn in models.items():
    model, metrics = train_fn(X_train, y_train, X_test, y_test)
    results[name] = metrics
    if metrics["R2"] is not None and metrics["R2"] > best_r2:
        best_r2 = metrics["R2"]
        best_model = model
        best_model_name = name

st.subheader("Model Evaluation Results")
for name, metrics in results.items():
    st.markdown(f"**{name}**")
    if metrics["R2"] is not None:
        st.write(f"R² Score: {metrics['R2']:.3f}")
        st.write(f"RMSE: {metrics['RMSE']:.3f}")
    else:
        st.warning("XGBoost is not installed.")

st.success(f"✅ Best Model: **{best_model_name}** with R² Score = {best_r2:.3f}")

# Prediction
st.subheader("🔮 Make a Prediction for a Neighborhood")
neighborhoods = df['Neighborhood'].unique()
selected_neigh = st.selectbox("Choose Neighborhood", neighborhoods)
row = df[df['Neighborhood'] == selected_neigh].iloc[0]

trees = st.number_input("Total Trees Planted", value=int(row['Total_Trees_Planted']), step=10)
green_roofs = st.number_input("Total New Green Roofs", value=int(row['Total_New_Green_Roofs']), step=1)
grass = st.number_input("Total Grass Planted (Hectares)", value=float(row['Total_Grass_Planted__Hectares_']), step=1.0)
cool_roofs = st.number_input("Total New Cool Roofs", value=int(row['Total_New_Cool_Roofs']), step=1)
cool_paving = st.number_input("Total Cool Pavement (Hectacres)", value=float(row['Total_Cool_Paving__Hectacres_']), step=1.0)

if st.button("Predict Temperature"):
    features = [[trees, green_roofs, grass, cool_roofs, cool_paving]]
    prediction = best_model.predict(features)
    st.success(f"🌡️ Predicted Avg Temperature for **{selected_neigh}**: **{prediction[0]:.2f} °F** using **{best_model_name}**")

# MODULE 2 - Clustering
st.header("🧭 Module 2: Clustering Analysis")
n_clusters = st.slider("Choose Number of Clusters", min_value=2, max_value=6, value=3)
clustered_df, kmeans_model = model_utils.perform_clustering(df.copy(), n_clusters=n_clusters)
st.subheader("Neighborhood Clusters")
st.dataframe(clustered_df[["Neighborhood", "Cluster"] + list(df.columns[2:])])

# Show cluster descriptions
st.write("### 🧾 Cluster Descriptions")
cluster_summary = model_utils.describe_clusters(clustered_df)
st.dataframe(cluster_summary)

import matplotlib.pyplot as plt
import seaborn as sns

st.write("### 📈 Cluster Visualization (PCA)")
scatter_data = model_utils.get_cluster_scatter_data(clustered_df)

fig, ax = plt.subplots()
sns.scatterplot(
    data=scatter_data,
    x="PC1", y="PC2",
    hue="Cluster",
    palette="viridis",
    legend="full",
    s=100
)
for _, row in scatter_data.iterrows():
    ax.text(row["PC1"], row["PC2"], row["Neighborhood"], fontsize=8, alpha=0.7)

st.pyplot(fig)


#http://localhost:8501/