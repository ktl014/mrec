""" MREClassifier module

Loads model to run predictions.
Assumptions are that the model is sklearn based.

"""
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class MREClassifier(object):

    def __init__(self, model_weights: str):
        """ Initializes MREClassifier instance

        Upon initialization, model weights and count vectorizer is loaded.

        Args:
            model_weights:
        """
        if not os.path.exists(model_weights):
            logger.warning(f"File {model_weights} was not found. Current dir: {os.getcwd()}")
            raise FileNotFoundError("Could not initialize MREClassifier because model weights not found.")

        self.model, self.count_vect = joblib.load(model_weights)

    def predict(self, X):
        """Get the model prediction on the data point

        Usage

        >>> from mrec.model.MREClassifier import MREClassifier
        >>> classifier = MREClassifier(model_weights="path/to/model")
        >>> prediction, proba = classifier.predict(X)

        Args:
            X: data point from user's input (contains a sentence and a relation)

        Returns:
            Dictionary of prediction and probability


        """
        X_counts = self.count_vect.transform(X)
        prediction = self.model.predict(X_counts)
        proba = self.model.predict_proba(X_counts)
        if len(prediction) == 1:
            return prediction[0], max(proba[0])
        else:
            return prediction, proba
