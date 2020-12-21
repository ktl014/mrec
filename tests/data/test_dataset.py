import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from mrec.data.dataset import *

class TestDataset():

    def test_load_data(self):
        #=== Test Inputs ===#
        TEST_CSV_DATASETS = {
            'train': "dataset/raw/train.csv"
        }
        datasets = load_data(TEST_CSV_DATASETS)
        assert hasattr(datasets, 'train')
        assert isinstance(datasets.train, pd.DataFrame)

        with pytest.raises(ValueError, match="Processed dataset"):
            datasets = load_data(TEST_CSV_DATASETS, processed=True)

        with pytest.raises(FileNotFoundError):
            TEST_CSV_DATASETS['val'] = "WRONG/PATH/val.csv"
            datasets = load_data(TEST_CSV_DATASETS)

    def test_load_rel_database(self):
        #=== Test Inputs ===#
        TEST_DATABASE = {
            'mrec': 'dataset/external/mrec.db'
        }
        table_name = "mrec_table"

        dataset = load_rel_database(TEST_DATABASE['mrec'], table_name)
        assert isinstance(dataset, pd.DataFrame)

        with pytest.raises(TypeError):
            datasets = load_rel_database(TEST_DATABASE, table_name)

        with pytest.raises(FileNotFoundError):
            TEST_DATABASE['val'] = "WRONG/PATH/val.db"
            dataset = load_rel_database(TEST_DATABASE['val'], table_name)
