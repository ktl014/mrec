import pytest

from mrec.visualization.medical_term_lookup import *

def test_get_description():
    #=== Test Input ===#
    term = "SPONTANOUS PAIN"

    #=== Trigger test ===#
    description = get_description(term)
    assert "Spontaneous pain" in description

    description = get_description(term='NOT-A-MEDICAL-TERM')
    expected_description = "Description not yet included"
    assert expected_description == description


