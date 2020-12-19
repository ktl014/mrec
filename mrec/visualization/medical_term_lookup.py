import logging
import os

import pandas as pd
import streamlit as st


MEDICAL_LOOKUP = pd.read_csv(os.path.join(os.path.dirname(__file__), 'medical_lookup.csv'))\
    .set_index('term')['description'].to_dict()
print(MEDICAL_LOOKUP)
logger = logging.getLogger(__name__)

def display_medical_terms(term1, term2):
    """Display table of medical terms and description"""
    medical_dict = {
        'term1': [term1, get_description(term1)],
        'term2': [term2, get_description(term2)]
    }
    medical_df = pd.DataFrame(medical_dict, index=['medical term', 'description'])
    st.table(medical_df)

def get_description(term: str):
    if not isinstance(term, str):
        logger.warn(f"Expected str, but got {type(term)} for term")
        raise TypeError("Wrong type passed into `term` when retrieving description")

    try:
        return MEDICAL_LOOKUP[term]
    except KeyError as e:
        logger.warn(f"Medical term ({term}) was not found in the medical_lookup.csv")
        return "Description not yet included"
