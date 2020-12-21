"""Take in any given and find best param
"""
# Third Party Imports
from sklearn.model_selection import GridSearchCV


def tuning_parameters(model, param_grid, X, y):
    """Fine-tuning parameter for a given model

    Args:
        model: classifier model
        param_grid: pram of that given model
        X (sparse matrix): countvectorizer of train feature(s)
        y (series): label of train set

    Returns:
        the best estimator
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    return grid_search.best_estimator_


