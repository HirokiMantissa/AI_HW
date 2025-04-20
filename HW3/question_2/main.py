import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

train_df = pd.read_csv('HW3/第二大題/kmeans_train.csv')
X = train_df[['x1', 'x2']].values

# Standard
scaler = StandardScaler()
X = scaler.fit_transform(X)

# init point
k = 3
np.random.seed(0)
initial_indices = np.random.choice(X.shape[0], k, replace=False)
centrol = X[initial_indices]


def assign_clusters(X, centrol):
    distances = np.linalg.norm(X[:, np.newaxis] - centrol, axis=2)
    return np.argmin(distances, axis=1)

def update_centrol(X, labels, k):
    new_centrol = []
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            new_centrol.append(points.mean(axis=0))
        else:
            new_centrol.append(X[np.random.randint(0, X.shape[0])])
    return np.array(new_centrol)

def has_converged(centrol, new_centrol):
    return np.allclose(centrol, new_centrol)

def compute_sse(X, centrol, labels):
    return sum(np.sum((X[labels == i] - centrol[i])**2) for i in range(len(centrol)))

colors = ['red', 'green', 'blue']
sse_list = []

iteration = 0
mutux = False

# Train
while not mutux and iteration < 10:
    labels = assign_clusters(X, centrol)
    new_centrol = update_centrol(X, labels, k)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, label='Data Points')
    plt.scatter(new_centrol[:, 0], new_centrol[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.title(f'Iteration {iteration + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    mutux = has_converged(centrol, new_centrol)
    centrol = new_centrol
    iteration += 1

labels = assign_clusters(X, centrol)
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i}', alpha=0.5)
plt.scatter(centrol[:, 0], centrol[:, 1], c='black', marker='x', s=100, label='Final Centroids')
plt.title('Final Result')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

#  Elbow & Silhouette
k_range = range(2, 11)
sse_list = []
silhouette_list = []

for k in k_range:
    np.random.seed(0)
    centrol = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(100):
        labels = assign_clusters(X, centrol)
        new_centrol = update_centrol(X, labels, k)
        if has_converged(centrol, new_centrol):
            break
        centrol = new_centrol
    sse = compute_sse(X, centrol, labels)
    sse_list.append(sse)
    sil_score = silhouette_score(X, labels)
    silhouette_list.append(sil_score)

plt.figure(figsize=(12, 5))

# Elbow plot
plt.subplot(1, 2, 1)
plt.plot(k_range, sse_list, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.grid(True)

# Silhouette plot
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_list, 'b-', marker='o')
best_k = k_range[np.argmax(silhouette_list)]
best_score = max(silhouette_list)
plt.plot(best_k, best_score, 'go') 
plt.title('Silhouette score')
plt.xlabel('k', fontsize=14)
plt.ylabel('S', fontsize=14)
plt.grid(True)
plt.show()