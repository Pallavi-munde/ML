# Agglomerative Hierarchical Clustering
# Dataset: Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Dataset loaded successfully!")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# 2. Standardize the Data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 3. Create Dendrogram
linked = linkage(X, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(
    linked,
    truncate_mode="level",
    p=5
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# 4. Apply Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

clusters = model.fit_predict(X)

# 5. Add Cluster Labels to Dataset
df["Cluster"] = clusters

print("\nClustered Dataset:")
print(df.head(10))

# 6. Display Number of Data Points in Each Cluster
print("\nNumber of Data Points in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# 7. Visualize Clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    s=50
)

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
