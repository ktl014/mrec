from collections import OrderedDict
import itertools
import random
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings

np.seterr(divide='ignore', invalid='ignore')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def accuracy(gtruth: np.array, predictions: np.array):
    """ Compute accuracy

    Args:
        gtruth: Ground truth
        predictions: Predictions

    Returns:
        float64: Accuracy score

    """
    return (gtruth == predictions).mean()

def compute_cm(gtruth: np.array, predictions: np.array, classes: list, plot=False, save=True):
    """ Compute confusion matrix

    Args:
        gtruth: Gtruth
        predictions: Predictions
        classes: List of classes
        plot: Flag to plot confusion matrix. Default is False
        save: Flag to save confusion matrix image. Default is True

    Returns:
        confusion matrix (normalized)
        average of diagonal along confusion matrix

    """
    # Create array for confusion matrix with dimensions based on number of classes
    num_class = len(classes)
    confusion_matrix_rawcount = np.zeros((num_class, num_class))
    class_count = np.zeros(
        (num_class, 1))  # 1st col represents number of images per class

    # Create confusion matrix
    for t, p in zip(gtruth, predictions):
        class_count[t, 0] += 1
        confusion_matrix_rawcount[t, p] += 1
    confusion_matrix_rate = np.zeros((num_class, num_class))
    for i in range(num_class):
        confusion_matrix_rate[i, :] = (confusion_matrix_rawcount[i, :]) / \
                                      class_count[i, 0] * 100

    confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)

    if plot:
        _plot_confusion_matrix(confusion_matrix_rate, classes=classes, save=save)
    return confusion_matrix_rate, np.nanmean(np.diag(confusion_matrix_rate))

def _plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues, save=False):
    """ Plot the confusion matrix and diagonal class accuracies

    Helper function to plot confusion matrix

    Args:
        cm: Confusion matrix
        classes: List of classes
        cmap: Matplotlib color map
        save: Flag to save confusion matrix image. Default is False.

    Returns:

    """
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2. if not math.isnan(cm.max()) else 50.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Plot diagonal scores alongside it
    plt.subplot(1, 2, 2)
    temp = dict(zip(classes, np.nan_to_num(cm.diagonal())))
    cm_dict = OrderedDict(sorted(temp.items(), key=lambda x: x[1]))
    classes = list(cm_dict.keys())
    cm_diag = list(cm_dict.values())

    ax = pd.Series(cm_diag).plot(kind='barh')
    ax.set_xlabel('Class Accuracy')
    ax.set_yticklabels(classes)
    rects = ax.patches
    # Make some labels.
    for rect, label in zip(rects, cm_diag):
        width = rect.get_width()
        label = np.nan if label == 0 else label
        ax.text(width + 5, rect.get_y() + rect.get_height() / 2, format(label, fmt),
                ha='center', va='bottom')

    if save:
        cm_fname = os.path.join('models', 'confusion.png')
        plt.savefig(cm_fname)
    plt.show()
