import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from decision_tree import build_tree, predict_tree

def bootstrap_sample(X, y):
    m = len(X)
    indices = np.random.choice(m, m, replace=True)
    return X[indices], y[indices]

def train_bagging(X, y, n_trees=10, max_depth=4, n_features=None):
    forest = []
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(X_sample, y_sample, max_depth=max_depth, n_features=n_features) 
        forest.append(tree)
    return forest

def predict_forest(forest, X):
    all_preds = []
    for x in X:
        preds = [predict_tree(tree, x) for tree in forest]
        majority_vote = np.bincount(preds).argmax()
        all_preds.append(majority_vote)
    return np.array(all_preds)

def plot_bagging_result(forest, X, y, title="Bagging"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_forest(forest, grid_points)
    Z = Z.reshape(xx.shape)

    # plot 
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['lightblue', 'salmon']), alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['blue', 'red']), edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel("Standardized x1")
    plt.ylabel("Standardized x2")
    plt.grid(True)
    plt.show()