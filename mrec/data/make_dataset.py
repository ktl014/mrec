"""Train Dataset Operations

Dataset operations module currently contains functions for the following:
- preprocessing train datasets

"""
# Standard Dist
import logging
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third Party Imports
from sklearn.feature_extraction.text import CountVectorizer

# Project Level Imports
from mrec.data.dataset import load_data
from mrec.features.transform import clean_text

logger = logging.getLogger(__name__)

SAVE_PATH = '../../models/count_vectorizer.joblib'

def preprocessing_dataset(csv_fnames, save=False):
    """Preprocessing dataset and add feature engineering

    Usage
    ------
    >>> from mrec.data.make_dataset import preprocessing_dataset
    >>> X_train, X_train_label, X_val, X_val_label = preprocessing_dataset()

    Returns:
        X_counts_train (sparse matrix): countvectorizer of train feature(s)
        X_train_label (series): label of train set
        X_counts_validation (spare matrix): countvectorizer of validation feature(s)
        X_validation_label (series): label of validation set

    """
    logger.info('Loading dataset...')

    # Read in training, validation data and labels
    dataset = load_data(csv_fnames)
    train, validation, test = dataset.train, dataset.validation, dataset.test

    relation_type = ['causes', 'treats']
    train = train[['sentence', 'relation']][train['relation'].isin(relation_type)].drop_duplicates()
    validation = validation[['sentence', 'relation']][validation['relation'].isin(relation_type)].drop_duplicates()
    test = test[['sentence', 'relation']][test['relation'].isin(relation_type)].drop_duplicates()

    logger.debug('Loaded dataset!')

    logger.debug('Preprocessing dataset..')
    #TODO add feature engineering
    count_vect = CountVectorizer(ngram_range=(1, 3), analyzer=clean_text)
    X_counts_train = count_vect.fit_transform(train['sentence'])
    X_train_label = train['relation']

    X_counts_validation = count_vect.transform(validation['sentence'])
    X_validation_label = validation['relation']

    X_counts_test = count_vect.transform(test['sentence'])
    X_test_label = test['relation']

    logger.debug('Finished preprocessing dataset!')

    #TODO raise error if this path is not found
    if save:
        joblib.dump(count_vect, SAVE_PATH)

    return X_counts_train, X_train_label, X_counts_validation, X_validation_label, X_counts_test, X_test_label


if __name__ == '__main__':
    preprocessing_dataset()
