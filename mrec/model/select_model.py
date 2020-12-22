"""
Train each model, run through val set to select best model

USAGE
-----

$ cd mrec/model
$ python select_model.py
# Launch an mlflow tracking ui after model results to compare
$ mlflow ui

"""

# Standard Dist
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third Party Imports
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset
from mrec.model.make_classifiers import make_classifiers
from mrec.model.grid_search import tuning_parameters

logger = logging.getLogger(__name__)
logger.root.setLevel(logging.INFO)

SAVE_PATH = '../../models/baseline_model/final_model.joblib'
GRID_SEARCH = True
csv_fnames = {'train': '../../dataset/raw/train.csv', 'validation': '../../dataset/raw/validation.csv',
              'test': '../../dataset/raw/test.csv'}

def select_model(grid_search=GRID_SEARCH):
    results = {}

    logger.info('Preparing datasets and models..')
    classifiers, parameters = make_classifiers()
    train, train_label, val, val_label, test, test_label = preprocessing_dataset(csv_fnames)

    # Begin experiment tracking
    experiment_name = 'select-models_and_grid-search' if grid_search else 'select-models'
    mlflow.set_experiment(experiment_name)
    logger.info(f'Beginning experiment {experiment_name}...')
    mlflow.sklearn.autolog()

    parent_run = mlflow.start_run(run_name="SELECT_MODEL_PARENT_RUN")
    mlflow.set_tags(classifiers)

    # result fields
    cv_acc_const = 'cv_acc'

    # Train list of classifiers for model selection
    for classifier_name, classifier in classifiers.items():
        with mlflow.start_run(run_name=classifier_name, nested=True):

            # Train and cross-validate the model
            classifier.fit(train, train_label)
            score = cross_val_score(classifier, val, val_label, scoring='accuracy', cv=10)

            # Log parameters and results
            result = {"Classifier": classifier, cv_acc_const: score.mean()}
            logger.debug(f"{classifier_name:20}: {result[cv_acc_const]:0.3f}")
            results.update({classifier_name: result})

            # Log mlflow results
            for param_name, param in classifier.get_params().items():
                mlflow.log_param(param_name, param)
            mlflow.log_metric(cv_acc_const, result[cv_acc_const])
            mlflow.sklearn.log_model(classifier, classifier_name)

    logger.info('Model selection experiment completed!')

    # Show results of all models
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    classifier_cols = ["tags.mlflow.runName"]
    metric_cols = [f"metrics.{cv_acc_const}"]
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    runs_df = runs[["run_id", *classifier_cols, *metric_cols]].sort_values(f"metrics.{cv_acc_const}", ascending=False)
    logger.info(f'Model selection results\n{runs_df}')

    # Report the best model
    best_score = 0
    best_model = ''
    for classifier_name, classifier_score in results.items():
        if best_score < classifier_score[cv_acc_const]:
            best_score = classifier_score[cv_acc_const]
            best_model = classifier_name
    logger.info(f'Best model: {best_model}; CV-Score: {best_score}')

    # Save results of the best model
    results_file = '../../models/baseline_model/model_selection_results.csv'
    logger.info(f'Results saved to {results_file}')
    runs.to_csv(results_file)
    mlflow.log_artifact(results_file, artifact_path="baseline_model")

    # Save the model
    logger.info(f'Saving best model as {SAVE_PATH}')
    joblib.dump(classifiers[best_model], SAVE_PATH)
    mlflow.log_artifact(SAVE_PATH, artifact_path="baseline_model")
    mlflow.end_run()

    if grid_search:
        logger.info(f'Beginning GridSearch with Best Model ({best_model}). Test set run to follow..')
        # Conduct grid search and report results on test set
        with mlflow.start_run(run_name="GRID_SEARCH_PARENT_RUN", nested=False):
            logger.debug('Tuning models..')
            final_model = tuning_parameters(classifiers[best_model], parameters[best_model], train, train_label)

            logger.debug('Testing models..')
            predictions = final_model.predict(test)
            test_acc = accuracy_score(test_label, predictions)
            logger.info(f'Test set accuracy: {test_acc}')

            mlflow.log_metric("test_acc", test_acc)

            # Save the model
            logger.info(f'Saving best model as {SAVE_PATH}')
            joblib.dump(final_model, SAVE_PATH)
            mlflow.log_artifact(SAVE_PATH, artifact_path="baseline_model")


if __name__ == '__main__':
    select_model()
