"""Take in any given and find best param

USAGE
-----

$ python mrec/model/grid_search.py
# Launch an mlflow tracking ui after model results to compare
$ mlflow ui

Add parameters via `model/make_classifiers.py` with its associated model

"""
# Standard Dist Imports
import logging
import os
import joblib
from pprint import pprint
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third Party Imports
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import pandas as pd

# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset
from mrec.model.make_classifiers import make_classifiers
from mrec.model.ml_utils import fetch_logged_data

logger = logging.getLogger(__name__)
logger.root.setLevel(logging.INFO)

SAVE_PATH = 'models/baseline_model/final_model.joblib'
MODEL_NAME = 'NuSVC'
csv_fnames = {'train': 'dataset/processed/train.csv',
              'validation': 'dataset/processed/validation.csv',
              'test': 'dataset/processed/test.csv'}

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

def main():
    experiment_name = 'grid-search'
    mlflow.set_experiment(experiment_name)
    logger.info(f'Beginning experiment {experiment_name} (tracked '
                f'{"remotely" if mlflow.tracking.is_tracking_uri_set() else "locally"})...')
    mlflow.sklearn.autolog()

    logger.info('Preparing datasets and models..')
    train, train_label, val, val_label, test, test_label = preprocessing_dataset(csv_fnames)

    classifiers, parameters = make_classifiers()

    run = mlflow.start_run()

    logger.info(f'Model trained using gridsearch: {MODEL_NAME}')
    final_model = tuning_parameters(classifiers[MODEL_NAME], parameters[MODEL_NAME], train, train_label)

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run.info.run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters[MODEL_NAME].keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])

    # Save the model
    logger.info(f'Saving best model as {SAVE_PATH}')
    joblib.dump(final_model, SAVE_PATH)
    mlflow.log_artifact(SAVE_PATH, artifact_path="baseline_model")

    mlflow.end_run()

if __name__ == '__main__':
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError("Model weights not found. Please run `python select_model.py` to get a baseline model")
    main()
