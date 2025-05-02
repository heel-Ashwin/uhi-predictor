import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    xgboost_installed = True
except ImportError:
    xgboost_installed = False

def load_data():
    df = pd.read_csv("app/data/Louisville_Metro_KY_-_Urban_Heat_Island_Neighborhood_Data.csv")
    df['Avg Temperature (°F)'] = 100 - (
        0.01 * df["Total_Trees_Planted"] +
        0.5 * df["Total_New_Green_Roofs"] +
        0.2 * df["Total_Grass_Planted__Hectares_"] +
        0.3 * df["Total_New_Cool_Roofs"] +
        0.05 * df["Total_Cool_Paving__Hectacres_"]
    ) + np.random.normal(0, 1, size=len(df))
    return df

def split_data(df):
    feature_cols = [
        "Total_Trees_Planted",
        "Total_New_Green_Roofs",
        "Total_Grass_Planted__Hectares_",
        "Total_New_Cool_Roofs",
        "Total_Cool_Paving__Hectacres_"
    ]
    X = df[feature_cols]
    y = df["Avg Temperature (°F)"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return {"R2": r2, "RMSE": rmse}

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)

def train_mlp(X_train, y_train, X_test, y_test):
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)

def train_xgboost(X_train, y_train, X_test, y_test):
    if xgboost_installed:
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, evaluate_model(model, X_test, y_test)
    return None, {"R2": None, "RMSE": None}

def perform_clustering(df, n_clusters=3):
    feature_cols = [
        "Total_Trees_Planted",
        "Total_New_Green_Roofs",
        "Total_Grass_Planted__Hectares_",
        "Total_New_Cool_Roofs",
        "Total_Cool_Paving__Hectacres_"
    ]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[feature_cols])
    return df, kmeans

def describe_clusters(df_clustered):
    descriptions = []
    grouped = df_clustered.groupby("Cluster")

    for cluster_id, group in grouped:
        avg_temp = group["Avg Temperature (°F)"].mean()
        trees = group["Total_Trees_Planted"].mean()
        green_roofs = group["Total_New_Green_Roofs"].mean()
        cool_roofs = group["Total_New_Cool_Roofs"].mean()
        grass = group["Total_Grass_Planted__Hectares_"].mean()

        if avg_temp < 80 and trees > 100:
            label = "🟢 Cool & Green-heavy Neighborhood"
        elif avg_temp > 90 and trees < 50:
            label = "🔥 High-Heat, Low-Greening Area"
        else:
            label = "🌤️ Moderate Heat & Balanced Green Efforts"

        desc = {
            "Cluster": cluster_id,
            "Label": label,
            "Avg Temp (°F)": round(avg_temp, 2),
            "Avg Trees": int(trees),
            "Avg Green Roofs": int(green_roofs),
            "Avg Cool Roofs": int(cool_roofs),
            "Avg Grass (Ha)": round(grass, 1)
        }

        descriptions.append(desc)

    return pd.DataFrame(descriptions)

def get_cluster_scatter_data(df_clustered):
    feature_cols = [
        "Total_Trees_Planted",
        "Total_New_Green_Roofs",
        "Total_Grass_Planted__Hectares_",
        "Total_New_Cool_Roofs",
        "Total_Cool_Paving__Hectacres_"
    ]
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_clustered[feature_cols])
    scatter_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    scatter_df["Cluster"] = df_clustered["Cluster"]
    scatter_df["Neighborhood"] = df_clustered["Neighborhood"]
    return scatter_df
