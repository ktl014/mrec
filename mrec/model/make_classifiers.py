# Standard Dist
import logging

# Third Party Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC

def make_classifiers():
    classifiers = {}
    classifiers.update({"AdaBoost": AdaBoostClassifier()})
    classifiers.update({"Bagging": BaggingClassifier()})
    classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
    classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
    classifiers.update({"Random Forest": RandomForestClassifier()})
    classifiers.update({"KNN": KNeighborsClassifier()})
    classifiers.update({"LSVC": LinearSVC(max_iter=5000)})
    classifiers.update({"NuSVC": NuSVC()})
    classifiers.update({"SVC": SVC()})
    classifiers.update({"DTC": DecisionTreeClassifier()})

    parameters = {}
    parameters.update({"AdaBoost": {
        "base_estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.01, 0.1, 1.0]
    }})
    parameters.update({"Bagging": {
        "base_estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "n_estimators": [10, 50, 100],
        "max_features": [0.5, 1.0],
        "n_jobs": [-1]
    }})
    parameters.update({"Gradient Boosting": {
        "learning_rate": [.1, 0.01],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 6],
        "min_samples_split": [0.01, 0.10],
        "min_samples_leaf": [0.01, 0.10],
        "max_features": ["auto"],
        "subsample": [0.8, 0.9, 1]
    }})
    parameters.update({"Extra Trees Ensemble": {
        "n_estimators": [10, 50, 100],
        "class_weight": [None, "balanced"],
        "max_features": ["auto"],
        "max_depth": [4, 8],
        "min_samples_split": [.01, 0.10],
        "min_samples_leaf": [0.01, 0.10],
        "criterion": ["gini", "entropy"],
        "n_jobs": [-1]
    }})
    parameters.update({"Random Forest": {
        "n_estimators": [50, 150, 500],
        "class_weight": [None, "balanced"],
        "max_features": ["auto"],
        "max_depth": [4, 8],
        "min_samples_split": [0.01, 0.10],
        "min_samples_leaf": [.01, 0.10],
        "criterion": ["gini", "entropy"],
        "n_jobs": [-1]
    }})
    parameters.update({"KNN": {
        "n_neighbors": list(range(1, 31)),
        "p": [2, 5],
        "leaf_size": [5, 15, 30, 50],
        "n_jobs": [-1]
    }})
    parameters.update({"LSVC": {
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1.0, 10]
    }})
    parameters.update({"NuSVC": {
        "nu": [0.25, 0.5, 0.75, 0.9],
        "kernel": ["linear", "rbf", 'poly', 'sigmoid'],
        "degree": [2, 3, 4, 5],
        "gamma": ["scale", "auto"],
        "random_state": [20170428],
        "decision_function_shape": ["ovo", "ovr"]
    }})
    parameters.update({"SVC": {
        "kernel": ["linear", "rbf",],
        "gamma": ["auto"],
        "C": [0.1, 10],
        "degree": [2, 5]
    }})
    parameters.update({"DTC": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "class_weight": [None, "balanced"],
        "max_features": ["auto"],
        "max_depth": [4, 8],
        "min_samples_split": [0.01, 0.10],
        "min_samples_leaf": [0.01, 0.10],
    }})
    return classifiers, parameters
