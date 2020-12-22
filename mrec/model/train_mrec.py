# Standard dist imports
import logging

# Third Party Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC


# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset

logger = logging.getLogger(__name__)


def print_metric(gtruth, predictions, dset_name):
    """Print 5 scoring metrics: accuracy, roc_auc, f1, precision, and recall

    Args:
        gtruth (array): label (either 0 or 1)
        predictions (array): model prediction (either 0 or 1)
        dset_name: the dataset that is evaluating on
    """
    accuracy = round(accuracy_score(gtruth, predictions), 4)
    roc_auc = round(roc_auc_score(gtruth, predictions), 4)
    f1 = round(f1_score(gtruth, predictions), 4)
    precision = round(precision_score(gtruth, predictions), 4)
    recall = round(recall_score(gtruth, predictions), 4)
    print('{:>10} {:>11} {:>12} {:>12} {:>11} {:>12}'.format(dset_name, accuracy, roc_auc, f1, precision, recall))

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

def train_mrec():
    """Train the best model"""
    model = NuSVC()
    train, train_label, val, val_label, test, test_label = preprocessing_dataset()

    logger.debug('Training model..')
    model.fit(train, train_label)

    logger.debug('Evaluating on train set..')
    print('{:>23} {:>12} {:>12} {:>12} {:>10}'.format('Accuracy', 'ROC_AUC', 'F1-score', 'Precision', 'Recall'))
    evaluate_model(model, train, train_label, 'Train')

    logger.debug('Evaluating on validation set..')
    evaluate_model(model, val, val_label, 'Validation')

    logger.debug('Evaluating on test set..')
    evaluate_model(model, test, test_label, 'Test')

if __name__ == '__main__':
    train_mrec()
