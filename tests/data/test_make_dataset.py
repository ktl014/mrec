import sys
import scipy
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import pandas as pd
from mrec.data.make_dataset import *


class TestMakeDataset():

    def test_preprocessing_dataset(self):
        # === Test Inputs ===#
        TEST_CSV_DATASETS = {
            'train': 'dataset/raw/train.csv',
            'validation': 'dataset/raw/validation.csv',
            'test': 'dataset/raw/test.csv'
        }
        SAVE_PATH = 'models/count_vectorizer.joblib'

        X_train, X_train_label, X_val, X_val_label, X_test, X_test_label = preprocessing_dataset(TEST_CSV_DATASETS, save=True, SAVE_PATH=SAVE_PATH)
        assert isinstance(X_train_label, pd.core.series.Series)
        assert isinstance(X_val_label, pd.core.series.Series)
        assert isinstance(X_test_label, pd.core.series.Series)

        assert isinstance(X_train, scipy.sparse.csr.csr_matrix)
        assert isinstance(X_val, scipy.sparse.csr.csr_matrix)
        assert isinstance(X_test, scipy.sparse.csr.csr_matrix)

        with pytest.raises(FileNotFoundError):
            SAVE_PATH = 'WRONG/PATH/count_vectorizer.joblib'
            X_train, X_train_label, X_val, X_val_label, X_test, X_test_label = preprocessing_dataset(TEST_CSV_DATASETS, save=True, SAVE_PATH=SAVE_PATH)

    def test_clean_duplicates_and_mislabels(self):
        # === Test Inputs ===#
        _unit_id = ['1', '2', '3', '4', '2']
        sentences = ['A causes B', 'C treats D', 'A causes B', 'Y is diagnosed by X', 'C treats D']
        relation = ['causes', 'treats', 'treats', 'diagnosed by', 'treats']
        data = pd.DataFrame({'_unit_id': _unit_id, 'sentence': sentences, 'relation': relation})

        # === Expected Output ===#
        expected_output = pd.DataFrame({'_unit_id': _unit_id[1], 'sentence': sentences[1], 'relation': relation[1]}, index=[1])

        # === Trigger Output ===#
        clean_dataset = clean_duplicates_and_mislabels(data)
        assert isinstance(clean_dataset, pd.DataFrame)
        assert clean_dataset.shape == (1, 3) and clean_dataset.index[0] == 1
        assert clean_dataset['sentence'][1] == sentences[1]
        assert expected_output.equals(clean_dataset)

    def test_clean_overlapping_sentences(self):
        #=== Test inputs ===#
        sentence_col = 'sentence'
        sentences = ["I lost my cat", "I lost my dog", "I lost my pig"]
        training_data = pd.DataFrame({sentence_col: sentences[:2]})
        data = pd.DataFrame({sentence_col: sentences})

        #=== Expected Output ===#
        expected_output = pd.DataFrame({sentence_col: sentences[-1:]}, index=[2])

        #=== Trigger Output ===#
        tested_output = clean_overlapping_sentences(data=data, training_data=training_data)
        assert isinstance(tested_output, pd.DataFrame)
        assert sentence_col in tested_output.columns
        assert tested_output.shape == (1, 1) and tested_output.index[0] == 2
        assert tested_output[sentence_col][2] == sentences[-1]
        assert expected_output.equals(tested_output)
