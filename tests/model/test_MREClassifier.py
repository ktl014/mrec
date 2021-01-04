import joblib
import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from mrec.model.MREClassifier import MREClassifier
from mrec.features.transform import clean_text


class TestMReClassifier(object):

    def test_init(self):
        with pytest.raises(FileNotFoundError):
            # === Test Inputs ===#
            MODEL_WEIGHT = './WRONG/PATH/random_forest.joblib'

            classifier = MREClassifier(model_weights=MODEL_WEIGHT)

    # def test_predict(self):
    #     # === Test Inputs ===#
    #     MODEL_WEIGHT = './models/random_forest.joblib'
    #
    #     classifier = MREClassifier(model_weights=MODEL_WEIGHT)
    #     X = 'I am a dummy sentence'
    #     expected = 'causes'
    #
    #     prediction, proba = classifier.predict(X=X)
    #     assert prediction == expected
