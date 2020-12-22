"""Take in any given and find best param
"""
# Standard dist imports
import logging

# Third Party Imports
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC

from mrec.data.make_dataset import preprocessing_dataset
from mrec.model.train_mrec import print_metric, evaluate_model

logger = logging.getLogger(__name__)

SAVE_PATH = '../../models/final_model.joblib'


def tuning_parameters():
    """Fine-tuning parameter for the best model"""
    enc = LabelEncoder()

    model = NuSVC()
    param_grid = {
        "nu": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "kernel": ["linear", "rbf", "sigmoid", "poly"],
        "degree": [1, 2, 3, 4, 5, 6, 7, 8]
    }
    train, train_label, val, val_label, test, test_label = preprocessing_dataset()

    logger.debug('Fine-tuning model..')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train, train_label)

    print('Best param =', grid_search.best_params_)

    final_model = grid_search.best_estimator_

    logger.debug('Evaluating on validation set..')
    print('{:>23} {:>12} {:>12} {:>12} {:>10}'.format('Accuracy', 'ROC_AUC', 'F1-score', 'Precision', 'Recall'))
    evaluate_model(final_model, val, val_label, 'Validation')

    logger.debug('Evaluating on test set..')
    evaluate_model(final_model, test, test_label, 'Test')

    joblib.dump(final_model, SAVE_PATH)

if __name__ == '__main__':
    tuning_parameters()
