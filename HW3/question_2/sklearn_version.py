import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

train_df = pd.read_csv('HW3/第二大題/kmeans_train.csv')
X = train_df[['x1', 'x2']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-', color='blue')
best_k = k_range[np.argmax(silhouette_scores)]
plt.plot(best_k, max(silhouette_scores), 'go')
plt.title('Silhouette Score')
plt.xlabel('k', fontsize=14, color='red')
plt.ylabel('s', fontsize=14, color='red')
plt.grid(True)

plt.tight_layout()
plt.show()

final_kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
final_labels = final_kmeans.fit_predict(X_scaled)
centroids = final_kmeans.cluster_centers_

plt.figure(figsize=(7, 6))
colors = ['red', 'green', 'blue', 'purple', 'orange']
for i in range(best_k):
    plt.scatter(X_scaled[final_labels == i, 0], X_scaled[final_labels == i, 1],
                color=colors[i], label=f'Cluster {i}', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
plt.title(f'K-Means Clustering (k={best_k})')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()