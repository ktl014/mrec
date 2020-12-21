"""
Train each model, run through val set to select best model
"""

# Standard Dist
import logging


# Third Party Imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset
from mrec.model.train_mrec import train_mrec
from mrec.model.make_classifiers import make_classifiers
from mrec.model.grid_search import tuning_parameters


logger = logging.getLogger(__name__)

SAVE_PATH = '../../models/final_model.joblib'

def select_model():
    results = {}

    classifiers, parameters = make_classifiers()
    train, train_label, val, val_label, test, test_label = preprocessing_dataset()

    logger.debug('Training models..')

    for classifier_name, classifier in classifiers.items():
        train_mrec(train, train_label, classifier)
        score = cross_val_score(classifier, val, val_label, scoring='accuracy', cv=10)
        result = {"Classifier": classifier,
                  'Val_acc': score.mean()}
        results.update({classifier_name: result})

    best_score = 0
    best_model = ''

    for classifier_name, classifier_score in results.items():
        print(classifier_name + ':', classifier_score['Val_acc'])
        if best_score < classifier_score['Val_acc']:
            best_score = classifier_score['Val_acc']
            best_model = classifier_name
    print('\nBest model:', best_model, '\nBest val score:', best_score)

    classifier = classifiers[best_model]
    param = parameters[best_model]

    logger.debug('Tuning models..')
    final_model = tuning_parameters(classifier, param, train, train_label)

    logger.debug('Testing models..')
    predictions = final_model.predict(test)
    print('\nBest model:', best_model)
    print('Test set accuracy:', accuracy_score(test_label, predictions))

    joblib.dump(final_model, SAVE_PATH)


if __name__ == '__main__':
    select_model()
