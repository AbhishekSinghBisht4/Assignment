import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data=pd.read_csv("Mall_Customers.csv")
#print(data)
#print(data.describe)

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

"""
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("PCA Projection of Mall Customers")
plt.xlabel("Plot")
plt.ylabel("Plot 2")
plt.show()

"""
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

"""
plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.title('Elbow Method: Inertia vs Number of Clusters (K)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.show()

"""


chosen_k = 5  
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': labels
})

import seaborn as sns

"""
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='Cluster',
    palette='Set2',
    s=80
)
plt.title(f"K-Means Clusters (K = {chosen_k})")
plt.legend(title='Cluster')
plt.show()

"""
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for K = {chosen_k}: {sil_score:.3f}")
