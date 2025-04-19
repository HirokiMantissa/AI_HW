import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from decision_tree import build_tree, predict_tree

def train_boosting(X, y, tree_nums, max_depth=1):
    m = X.shape[0]
    weights = np.ones(m) / m 
    classifiers = []
    alphas = []

    for _ in range(tree_nums):
        indices = np.random.choice(m, m, replace=True, p=weights)
        X_sample, y_sample = X[indices], y[indices]

        stump = build_tree(X_sample, y_sample, max_depth=max_depth)
        pred = np.array([predict_tree(stump, x) for x in X])

        err = np.sum(weights * (pred != y)) / np.sum(weights)
        if err > 0.5:
            continue

        alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
        alphas.append(alpha)
        classifiers.append(stump)

        weights *= np.exp(-alpha * y * (2 * (pred == y) - 1))
        weights /= np.sum(weights)

    return classifiers, alphas


def predict_adaboost(classifiers, alphas, X):
    final = np.zeros(X.shape[0])
    for clf, alpha in zip(classifiers, alphas):
        preds = np.array([predict_tree(clf, x) for x in X])
        preds = 2 * (preds == 1) - 1
        final += alpha * preds
    return (final > 0).astype(int)

def plot_boosting_result(classifiers, alphas, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = predict_adaboost(classifiers, alphas, grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['lightblue', 'salmon']), alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['blue', 'red']), edgecolor='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()