import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load final cleaned data
# ---------------------------
df = pd.read_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\cleaned_student_depression_final000.xlsx")

print("✅ Dataset Loaded:", df.shape)

# ---------------------------
# 2. Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Save scaler
joblib.dump(scaler, r"C:\Users\punee\OneDrive\Desktop\mlproject\models\scaler.joblib")

# ---------------------------
# 3. K-Means Clustering
# ---------------------------
k = 3  # Low, Moderate, High risk
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Save model
joblib.dump(kmeans, r"C:\Users\punee\OneDrive\Desktop\mlproject\models\kmeans_model.joblib")

# ---------------------------
# 4. DBSCAN (Anomaly Clusters)
# ---------------------------
dbscan = DBSCAN(eps=1.2, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

# ---------------------------
# 5. Isolation Forest (Outlier Detection)
# ---------------------------
iso = IsolationForest(contamination=0.15, random_state=42)
df["anomaly"] = iso.fit_predict(X_scaled)  # -1 = High Risk Outlier

# ---------------------------
# 6. Auto Risk Labeling (Using Mean Risk Score)
# ---------------------------
cluster_risk = df.groupby("kmeans_cluster")["risk_score"].mean().sort_values()

risk_map = {
    cluster_risk.index[0]: "Low Risk",
    cluster_risk.index[1]: "Moderate Risk",
    cluster_risk.index[2]: "High Risk"
}

df["risk_label"] = df["kmeans_cluster"].map(risk_map)

# ---------------------------
# 7. PCA Visualization (For Report & Dashboard)
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# Save PCA plot
plt.figure()
plt.scatter(df["pca1"], df["pca2"])
plt.title("Mental Health Risk Clusters (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(r"C:\Users\punee\OneDrive\Desktop\mlproject\models\cluster_visualization.png")
plt.close()

# ---------------------------
# 8. Save Final Clustered File
# ---------------------------
df.to_csv(r"C:\Users\punee\OneDrive\Desktop\mlproject\student_depression_clustered_final0000.csv", index=False)

print("✅ Clustering Completed Successfully!")
print("✅ Output File: student_depression_clustered_final.csv")
print("✅ PCA Image Saved: cluster_visualization.png")
