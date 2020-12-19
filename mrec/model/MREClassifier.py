import joblib

class MREClassifier(object):

    def __init__(self, model_weights: str):
        self.model, self.count_vect = joblib.load(model_weights)

    def predict(self, X, features_names):
        return self.model.predict_proba(X)


