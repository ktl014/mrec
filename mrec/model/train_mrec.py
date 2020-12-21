# Standard dist imports
import logging

logger = logging.getLogger(__name__)


def train_mrec(X, y, model):
    """Train the given model

    Args:
        X (sparse matrix): countvectorizer of train feature(s)
        y (series): label of train set
        model: classifier model

    Returns:
        Null
    """
    model.fit(X, y)
