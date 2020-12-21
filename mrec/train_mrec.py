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
>>> ???
>>> ???
>>> # then run the script using the command below
$ python src/train_model.py
"""

# Standard dist imports
import logging
from mrec.data.dataset import load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from mrec.features.transform import clean_text

# Third Party Imports
import joblib

# Setup default

logger = logging.getLogger(__name__)


def main():
    """Train and save the model

    Returns: NULL

    """
    ## Preprocessing dataset
    logger.debug('Preprocessing classifier..')

    count_vect = CountVectorizer(ngram_range=(1, 3), analyzer=clean_text)
    X_counts_train = count_vect.fit_transform(train['sentence'])
    X_counts_validation = count_vect.transform(validation['sentence'])

    ## Train the model
    logger.debug('Training classifier..')

    random_forest = RandomForestClassifier()
    random_forest.fit(X_counts_train, train['relation'])

    ## Evaluate the classifier
    logger.debug('Evaluating classifier..')

    forest_accuracy = cross_val_score(random_forest, X_counts_validation, validation['relation'], scoring="accuracy",
                                      cv=10)
    logger.debug('Accuracy score on validation set:', forest_accuracy.mean())

    ## Save model
    logger.debug('Saving the model..')

    path = '../../models/random_forest.joblib'
    joblib.dump((random_forest, count_vect), path)

if __name__ == '__main__':
    logger.debug('Loading dataset...')

    # Read in training, validation data and labels
    csv_fnames = {'train': 'dataset/raw/train.csv', 'validation': 'dataset/raw/validation.csv',
                  'test': 'dataset/raw/test.csv'}
    dataset = load_data(csv_fnames)
    train, validation = dataset.train, dataset.validation

    # Feature = sentence, target = relation (treats and causes only)
    relation_type = ['causes', 'treats']
    train = train[['sentence', 'relation']][train['relation'].isin(relation_type)].drop_duplicates()
    validation = validation[['sentence', 'relation']][validation['relation'].isin(relation_type)].drop_duplicates()

    logger.debug('Loaded dataset!')

    main()
