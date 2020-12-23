"""Train Dataset Operations

Dataset operations module currently contains functions for the following:
- preprocessing train datasets

USAGE
-----

$ python mrec/data/make_dataset.py

"""
# Standard Dist
import logging
import os
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
logging.root.setLevel(logging.DEBUG)

SAVE_PATH = '../../models/count_vectorizer.joblib'
csv_fnames = {'train': 'dataset/raw/train.csv', 'validation': 'dataset/raw/validation.csv',
              'test': 'dataset/raw/test.csv'}

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

    # Filtering for binary case of causes and treats
    # duplicates are dropped here because the dataset includes multiple annotators for 1 sample
    relation_type = ['causes', 'treats']
    train = train[['sentence', 'relation']][train['relation'].isin(relation_type)].drop_duplicates()
    validation = validation[['sentence', 'relation']][validation['relation'].isin(relation_type)].drop_duplicates()
    test = test[['sentence', 'relation']][test['relation'].isin(relation_type)].drop_duplicates()
    logger.debug('Loaded dataset!')

    # Preprocessing entails doing count vectorization
    # vectorizer includes `clean_text` for cleaning out punctuation and lemnatizing words
    logger.debug('Preprocessing dataset..')
    #TODO add feature engineering
    count_vect = CountVectorizer(ngram_range=(1, 3), analyzer=clean_text)
    # train tokenization
    X_counts_train = count_vect.fit_transform(train['sentence'])
    X_train_label = train['relation']
    # validation tokenization
    X_counts_validation = count_vect.transform(validation['sentence'])
    X_validation_label = validation['relation']
    # test set tokenization
    X_counts_test = count_vect.transform(test['sentence'])
    X_test_label = test['relation']

    # preprocessing complete at this point
    logger.debug('Finished preprocessing dataset!')

    #TODO raise error if this path is not found
    if save:
        joblib.dump(count_vect, SAVE_PATH)

    return X_counts_train, X_train_label, X_counts_validation, X_validation_label, X_counts_test, X_test_label

def main():
    """
    remove case1: sentences that have multiple labels assigned within training, val, test
    - remove the samples entirely if they have multiple labels within the dataframe

    get the dataset
    use bim's logic to get the data that have multiple labels (assign a flag if it's bad)
    - - drop duplicates twice

    drop those based on the flag
    """
    cleaned_data_dir = os.path.join(str(Path(__file__).resolve().parents[2]), 'dataset/processed')
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir, exist_ok=True)

    dataset = load_data(csv_fnames)
    train, validation, test = dataset.train, dataset.validation, dataset.test

    def clean_duplicates_and_mislabels(data):
        logger.debug('dataset size before: {}'.format(data.shape[0]))
        relation_type = ['causes', 'treats']
        data = data[data['relation'].isin(relation_type)].drop_duplicates(subset='_unit_id')
        data = data.drop(list(data[data['sentence'].duplicated(False)].index))
        logger.debug('dataset size after: {}'.format(data.shape[0]))
        return data

    def clean_overlapping_sentences(data, training_data):
        df = data.copy()
        overlapped_sentences = df[df['sentence'].isin(training_data['sentence'])]
        logger.debug(f'\tFound overlapped {overlapped_sentences.shape[0]} sentences out of {df.shape[0]}. Now '
                     f'removing...')
        df = df.drop(overlapped_sentences.index)
        logger.debug(f'\tNew dataset size: {df.shape[0]}')
        return df

    csv_file = os.path.join(cleaned_data_dir, '{}.csv')
    data_ = dict(zip(csv_fnames.keys(), [train, validation, test]))
    for k, v in data_.items():
        logger.debug('Starting {} dataset'.format(k))
        clean_data = data_[k] = clean_duplicates_and_mislabels(v)

        if k != 'train':
            logger.debug(f'Cleaning up overlap in {k} dataset')
            clean_data = clean_overlapping_sentences(clean_data, data_['train'])

        logger.debug(f'Dataset saved to {csv_file.format(k)}')
        clean_data.to_csv(csv_file.format(k))



if __name__ == '__main__':
    main()
