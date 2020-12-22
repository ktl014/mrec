"""Train and evaluate MREC (Medical Relation Extraction Classifier)

This script is designed to train and evaluate a classifier.
It begins with loading and partitioning our datasets, then going straight
into training and evaluation. Results will be outputted for the training and
validation set.

Prior to running this script, please ensure the datasets have been made
by running `make_dataset.py` under our `data` directory. The list of datasets
are listed in the documentation of the script.

Usage
-----
>>> SAVE_MODEL = False
>>> # then run the script using the command below
$ python mrec/train_model.py
# Run MlFlow UI after to view results
$ mlflow ui

"""

# Standard dist imports
import sys
from pathlib import Path
import logging
from pprint import pprint
sys.path.insert(0, str(Path(__file__).resolve().parents[1]) + '/')

# Third Party Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC
import mlflow
import mlflow.sklearn

# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset
from mrec.model.ml_utils import fetch_logged_data

logger = logging.getLogger(__name__)
logger.root.setLevel(logging.INFO)

SAVE_MODEL = False
#TODO figure out better way to store the relative paths for these datasets (symlink????)
csv_fnames = {'train': '../dataset/raw/train.csv', 'validation': '../dataset/raw/validation.csv',
              'test': '../dataset/raw/test.csv'}

def print_metric(gtruth, predictions, dset_name):
    """Print 5 scoring metrics: accuracy, roc_auc, f1, precision, and recall

    Args:
        gtruth (array): label (either 0 or 1)
        predictions (array): model prediction (either 0 or 1)
        dset_name: the dataset that is evaluating on
    """
    clsf_metrics = {f'{dset_name}_{metric}':0 for metric in ['accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']}
    clsf_metrics[f"{dset_name}_accuracy"] = round(accuracy_score(gtruth, predictions), 4)
    clsf_metrics[f"{dset_name}_roc_auc"] = round(roc_auc_score(gtruth, predictions), 4)
    clsf_metrics[f"{dset_name}_f1_score"] = round(f1_score(gtruth, predictions), 4)
    clsf_metrics[f"{dset_name}_precision"] = round(precision_score(gtruth, predictions), 4)
    clsf_metrics[f"{dset_name}_recall"] = round(recall_score(gtruth, predictions), 4)
    mlflow.log_metrics(clsf_metrics)

def evaluate_model(model, X, y, dset_name):
    """Evaluate on given model

    Args:
        model: NuSVC()
        X: countvectorizers of feature(s)
        y: label
        dset_name: dataset that is evaluating on
    """
    enc = LabelEncoder()

    predictions = model.predict(X)
    gtruth = enc.fit_transform(y)
    predictions = enc.transform(predictions)

    print_metric(gtruth, predictions, dset_name)

def main():
    """Train the best model"""
    experiment_name = 'train-mrec_v1.0.0'
    mlflow.set_experiment(experiment_name)
    logger.info(f'Beginning experiment {experiment_name}...')

    run_name = 'hyperparamterized-nusvc'
    with mlflow.start_run(run_name=run_name) as run:

        model = NuSVC(degree=2, kernel='rbf', nu=0.25)
        mlflow.log_params(model.get_params())

        train, train_label, val, val_label, test, test_label = preprocessing_dataset(csv_fnames)

        logger.info('Training model..')
        model.fit(train, train_label)

        logger.info('Running evaluations...')
        logger.debug('Evaluating on train set..')
        evaluate_model(model, train, train_label, 'training')

        logger.debug('Evaluating on validation set..')
        evaluate_model(model, val, val_label, 'validation')

        logger.debug('Evaluating on test set..')
        evaluate_model(model, test, test_label, 'test')

    # show data logged in the parent run
    logger.info(f"\n========== {experiment_name} run ==========")
    for key, data in fetch_logged_data(run.info.run_id).items():
        logger.info("\n---------- logged {} ----------".format(key))
        pprint(data)

    if SAVE_MODEL:
        model_path = f'../../models/baseline_model/{run_name}.joblib'
        logger.info(f'Saved model to {model_path}')

if __name__ == '__main__':
    main()
