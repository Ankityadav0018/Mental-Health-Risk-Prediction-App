import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# 1. Load your cleaned + imputed data
df = pd.read_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\student_depression_clustered_final0000.xlsx")

print("✅ Loaded student_depression_clustered_final0000.xlsx:", df.shape)

# 2. Features we will use in the Web App (only mental-health related)
feature_cols = [
    "self_employed",
    "family_history",
    "treatment",
    "days_indoors",
    "changes_habits",
    "mental_health_history",
    "growing_stress",
    "mood_swings",
    "coping_struggles",
    "work_interest",
    "social_weakness",
    "risk_score",   # derived score we already created
]

# Safety check
for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

X = df[feature_cols].values

# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means for 3 risk groups
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["app_kmeans_cluster"] = clusters

# 5. Map clusters -> risk labels based on average risk_score
cluster_risk_mean = df.groupby("app_kmeans_cluster")["risk_score"].mean().sort_values()

# Lowest mean risk_score = Low Risk, etc.
risk_label_order = ["Low Risk", "Moderate Risk", "High Risk"]
risk_map = {}

for i, (cluster_id, _) in enumerate(cluster_risk_mean.items()):
    if i < len(risk_label_order):
        risk_map[int(cluster_id)] = risk_label_order[i]
    else:
        risk_map[int(cluster_id)] = "Unknown"

df["app_risk_label"] = df["app_kmeans_cluster"].map(risk_map)

print("✅ Cluster → Risk mapping:", risk_map)

# 6. Save models and mapping
if not os.path.exists(r"C:\Users\punee\OneDrive\Desktop\mlproject\models"):
    os.makedirs("models")

joblib.dump(scaler, r"C:\Users\punee\OneDrive\Desktop\mlproject\models\app_scaler.joblib")
joblib.dump(kmeans, r"C:\Users\punee\OneDrive\Desktop\mlproject\models\app_kmeans_model.joblib")
joblib.dump(risk_map, r"C:\Users\punee\OneDrive\Desktop\mlproject\models\app_cluster_risk_map.joblib")

# 7. Save a copy of data with app clusters (optional)
df.to_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\student_depression_app_clustered.xlsx", index=False)

print("✅ App-oriented model trained and saved successfully.")
print("   - models/app_scaler.joblib")
print("   - models/app_kmeans_model.joblib")
print("   - models/app_cluster_risk_map.joblib")
