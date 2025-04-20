import numpy as np
from dataloder import dataloder
from decision_tree import build_tree, plot_decision_tree_result
from bagging import train_bagging, plot_bagging_result
from boosting import train_boosting, plot_boosting_result

#==== dataloader ====
X_train_scaled,  y_train = dataloder().get_train_data()

#==== decision tree ====
tree = build_tree(X_train_scaled, y_train, max_depth=4, n_features=None)
plot_decision_tree_result(tree, X_train_scaled, y_train, title="Decision Tree (trian set)")

#==== bagging ====
n_features = int(np.sqrt(X_train_scaled.shape[1]))
forest = train_bagging(X_train_scaled, y_train, n_trees=10, max_depth=4, n_features=n_features)
plot_bagging_result(forest, X_train_scaled, y_train, title="Bagging (trian set)")

#==== boosting ====
classifiers, alphas = train_boosting(X_train_scaled, y_train, tree_nums=30, max_depth=2)
plot_boosting_result(classifiers, alphas, X_train_scaled, y_train, title="Boosting (trian set)")

#=== test ====
X_test_scaled,  y_test = dataloder().get_test_data()

tree = build_tree(X_test_scaled, y_test, max_depth=4, n_features=None)
plot_decision_tree_result(tree, X_test_scaled, y_test, title="Decision Tree (test set)")

n_features = int(np.sqrt(X_test_scaled.shape[1]))
forest = train_bagging(X_test_scaled, y_test, n_trees=10, max_depth=4, n_features=n_features)
plot_bagging_result(forest, X_test_scaled, y_test, title="Bagging (test set)")

classifiers, alphas = train_boosting(X_test_scaled, y_test, tree_nums=30, max_depth=2)
plot_boosting_result(classifiers, alphas, X_test_scaled, y_test, title="Boosting (test set)")