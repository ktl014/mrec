import joblib
import logging
import os

logger = logging.getLogger(__name__)

class MREClassifier(object):

    def __init__(self, model_weights: str):
        if not os.path.exists(model_weights):
            logger.warning(f"File {model_weights} was not found. Current dir: {os.getcwd()}")
            raise FileNotFoundError("Could not initalize MREClassifier because model weights not found.")

        self.model, self.count_vect = joblib.load(model_weights)

    def predict(self, X):
        """Get the model prediction on the data point

        Usage

        >>> from mrec.model.MREClassifier import MREClassifier
        >>> prediction, proba = predict(X)

        Args:
            X: data point from user's input (contains a sentence and a relation)

        Returns:
            Dictionary of prediction and probability

        """
        X_counts = self.count_vect.transform(X)
        prediction = self.model.predict(X_counts)
        proba = self.model.predict_proba(X_counts)
        return prediction[0], proba[0]
