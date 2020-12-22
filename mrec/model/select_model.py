"""
Train each model, run through val set to select best model
"""

# Standard Dist
import logging


# Third Party Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.model_selection import cross_val_score

# Project Level Imports
from mrec.data.make_dataset import preprocessing_dataset

logger = logging.getLogger(__name__)

models = [
    SVC(), NuSVC(), LinearSVC(max_iter=5000), GradientBoostingClassifier(),
    KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(),
    BaggingClassifier(), ExtraTreesClassifier(), RandomForestClassifier(),
    AdaBoostClassifier()
]


def select_model():
    """Train a set of model and out put their val score and the best one"""
    results = {}
    train, train_label, val, val_label, test, test_label = preprocessing_dataset()

    print('Training models..')
    for model in models:
        model.fit(train, train_label)
        score = cross_val_score(model, val, val_label, scoring='accuracy', cv=10)

        model_name = model.__class__.__name__
        result = {"Classifier": model_name,
                  'Val_acc': round(score.mean(), 4),
                  'Std': round(score.std(), 3)}
        results.update({model_name: result})

        print('{}: {} (+/-{})'.format(model_name, round(score.mean(), 4), round(score.std(), 3)))

    std = 0
    best_score = 0
    best_model = ''

    for classifier_name, score in results.items():
        if best_score < score['Val_acc']:
            best_score = score['Val_acc']
            std = score['Std']
            best_model = classifier_name

    print('\nBest model: {}'.format(best_model))
    print('Val accuracy score: {} (+/-{})'.format(best_score, std))


if __name__ == '__main__':
    select_model()
