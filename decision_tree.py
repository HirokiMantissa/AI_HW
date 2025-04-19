import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     
        self.threshold = threshold 
        self.left = left           
        self.right = right      
        self.value = value        

def gini_impurity(y):
    class_probs = np.bincount(y) / len(y)
    return 1 - np.sum(class_probs ** 2)

def information_gain(y, y_left, y_right):
    return gini_impurity(y) - (len(y_left) / len(y)) * gini_impurity(y_left) - (len(y_right) / len(y)) * gini_impurity(y_right)

def best_split(X, y, n_features=None):
    m, n = X.shape
    features = np.random.choice(n, n_features, replace=False) if n_features else range(n)
    best_ig = -1
    best_feat, best_thresh = None, None

    for feature in features:
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left, y_right = y[left_mask], y[right_mask]
            ig = information_gain(y, y_left, y_right)

            if ig > best_ig:
                best_ig = ig
                best_feat = feature
                best_thresh = t

    return best_feat, best_thresh

def build_tree(X, y, max_depth=None, depth=0, n_features=None):
    m, n = X.shape
    unique_classes = np.unique(y)

    if len(unique_classes) == 1:
        return Node(value=unique_classes[0])

    if max_depth is not None and depth >= max_depth:
        return Node(value=np.bincount(y).argmax())

    feature, threshold = best_split(X, y, n_features=n_features)
    if feature is None:
        return Node(value=np.bincount(y).argmax())

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], max_depth, depth + 1, n_features)
    right_node = build_tree(X[right_mask], y[right_mask], max_depth, depth + 1, n_features)

    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

def predict_tree(tree, X):
    if tree.value is not None:
        return tree.value
    if X[tree.feature] <= tree.threshold:
        return predict_tree(tree.left, X)
    else:
        return predict_tree(tree.right, X)
      
def plot_decision_tree_result(tree, X, y, title="Decision Tree"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = np.array([predict_tree(tree, np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.axis('equal')
    plt.show()