"""Dataset Operations

Dataset operations module currently contains functions for the following:
- reading datasets
- loading & processing datasets

"""
# Standard Dist
from collections import namedtuple
import logging
import os

# Third Party Imports
import pandas as pd

# Project Level Imports
import mrec.mrec
from mrec.data.rel_database import Database

logger = logging.getLogger(__name__)

def load_data(csv_fnames, processed=False):
    """Load dataset into tuple collection object

    Dataset will contain at default the raw dataset. Optionally, the feature
    engineered dataset can be included by turning on the `processed` argument.

    Usage

    >>> from mrec.data.dataset import load_data
    >>> csv_fnames = {'train': '../../dataset/raw/train.csv', 'validation': '../../dataset/raw/validation.csv'}
    >>> dataset = load_data(csv_fnames)
    MRECDataset(train=..., validation=...)

    Args:
        csv_fnames (dict): Dictionary containing csv absolute paths
        processed (bool): Flag for including processed dataset. Default is False.

    Returns:
        MRECDataset: MREC Dataset

    """
    datasets = {}
    for data, csv_path in csv_fnames.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} was not found. Current dir: {os.getcwd()}")

        datasets[data] = pd.read_csv(csv_path)
        logger.debug(f'Loaded dataset ({data}:{csv_path})')

    if processed:
        raise ValueError("Processed dataset loading is not an option yet.")

    MRECDataset = namedtuple('MRECDataset', list(datasets.keys()))
    return MRECDataset(**datasets)


def load_rel_database(db_path :str, table_name):
    """Load dataset from rel database to dataframe

    Usage
    ------
    >>> from mrec.data.dataset import load_rel_database
    >>> table_name = 'mrec_table'
    >>> db_path = '../../dataset/external/mrec.db'
    >>> df = load_rel_database(db_path, table_name)

    Args:
        db_path (str): database file path to load data from
        table_name (str): the name of the table in the database

    Returns:
        DataFrame: unprocessed data from rel database
    """
    if not isinstance(db_path, str):
        raise TypeError("Error found with type of input `db_path` when loading rel database")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"File {db_path} was not found. Current dir: {os.getcwd()}")

    db = Database(db_path)
    SQL_QUERY = "SELECT * FROM " + table_name
    dataset = pd.read_sql(SQL_QUERY, con=db.conn)
    db.close_connection()

    return dataset
