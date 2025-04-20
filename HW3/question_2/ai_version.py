import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- Step 1: 資料生成 ----------
def generate_data():
    train_df = pd.read_csv('HW3/第二大題/kmeans_train.csv')
    X = train_df[['x1', 'x2']].values
    return X

# ---------- Step 2: 標準化 ----------
def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# ---------- Step 3: 評估不同 k 的 KMeans ----------
def evaluate_kmeans(X_scaled, k_range):
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes

# ---------- Step 4: 繪圖 ----------
def plot_metrics(k_range, inertias, silhouettes):
    best_k = k_range[np.argmax(silhouettes)]

    plt.figure(figsize=(12, 5))

    # Inertia
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.grid(True)

    # Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, 'o-', color='blue')
    plt.plot(best_k, max(silhouettes), 'go', label=f'Best k = {best_k}')
    plt.title('Silhouette Score')
    plt.xlabel('k', fontsize=14, color='red')
    plt.ylabel('s', fontsize=14, color='red')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return best_k

# ---------- Step 5: 執行最佳 KMeans ----------
def plot_final_clustering(X_scaled, best_k):
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    plt.figure(figsize=(8, 6))
    for i in range(best_k):
        plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], color=colors[i], label=f'Cluster {i}', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.title(f'Final Clustering Result (k={best_k})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# ---------- 主流程 ----------
if __name__ == "__main__":
    X = generate_data()
    X_scaled = standardize_data(X)
    k_range = range(2, 7)
    inertias, silhouettes = evaluate_kmeans(X_scaled, k_range)
    best_k = plot_metrics(k_range, inertias, silhouettes)
    plot_final_clustering(X_scaled, best_k)