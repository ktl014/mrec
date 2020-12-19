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
import os
import datetime
import pickle

# Third Party Imports
import joblib

# SEtup default
import mrec.mrec

logger = logging.getLogger(__name__)

def main():
    # Train the model
    logger.debug('Training classifier..')

    # Evaluate the classifier
    logger.debug('Evaluating classifier..')

    # Save model
    logger.debug('Saving the model..')

if __name__ == '__main__':
    logger.debug('Loading dataset...')
    # Read in training, validation data and labels
    logger.debug('Loaded dataset!')

    main()
