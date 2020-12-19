import joblib

class MREClassifier(object):

    def __init__(self, model_weights: str):
        # self.model = joblib.load(model_weights)
        self.model = 0

    def predict(self,X,features_names):
        # return self.model.predict_proba(X)
        return "treat"

