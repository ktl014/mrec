import pytest

from mrec.model.MREClassifier import MREClassifier

class TestMReClassifier():
    @pytest.fixture(scope="session")
    def classifier(self):
        model_weight = './models/random_forest.joblib'
        classifier = MREClassifier(model_weights=model_weight)
        return classifier

    def test_predict(self, classifier):
        #=== Test Input ===#
        X = "this is a dummy senetence"

        #=== Expected Output ===#
        expected_output = "causes"

        #=== Trigger Output ===#
        output, _ = classifier.predict(X=X)
        assert expected_output == output
