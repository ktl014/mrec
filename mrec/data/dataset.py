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
