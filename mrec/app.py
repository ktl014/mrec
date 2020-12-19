""" Classification Streamlit Application

Module is used to launch classification system

"""
# Standard Dist Imports
from datetime import datetime
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Third party imports
import streamlit as st

# Project level imports
import mrec.mrec
from mrec.data.dataset import load_data
from mrec.visualization import SessionState
from mrec.visualization.medical_term_lookup import display_medical_terms
import mrec.config as config
from mrec.model.MREClassifier import MREClassifier
from mrec.features.transform import clean_text

# Module level constants
FEATURES_LIST = ['_unit_id', 'relation', 'sentence', 'direction', 'term1', 'term2']
logger = logging.getLogger(__name__)

#=== Load dataset as cache ===#
@st.cache
def load_data_to_cache(csv_fnames):
    """ Load dataset for web application

    Args:
        csv_fnames (dict): Dictionary of csv absolute paths to load data from

    Returns:
        Airbnb: Named collection tuple of Airbnb dataset

    """
    dataset = load_data(csv_fnames)
    val = dataset.validation[FEATURES_LIST].drop_duplicates()
    return val

def run():
    data, predictions, idm = None, [], 0
    session_state = SessionState.get(id=id, data=data, predictions=predictions)

    dataset = load_data_to_cache(config.CSV_FNAMES)

    # === Generate USER ID button ===#
    # Get user id
    st.header('Generate User')
    if st.button('Click here to Generate User ID'):
        session_state.data = dataset.sample(n=1, random_state=1)
        session_state.id = session_state.data['_unit_id'].values[0]
        st.write(session_state.id)

    # === View raw data ===#
    st.subheader('Raw Data')
    if st.checkbox('Show data') and (session_state.id != 0):
        st.json(session_state.data.to_dict())

    # === Classification button ===#
    # Recommend based off id
    st.header('Classify Relationship')
    recommend = st.button('Click here to Classify Relationship')
    if recommend:
        model_weight = './models/random_forest.joblib'
        classifier = MREClassifier(model_weights=model_weight)
        session_state.predictions, _ = classifier.predict(X=session_state.data)

        # Report the predictions results
        st.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | USR ID: {session_state.id}] Prediction: {session_state.predictions}')
        st.write(session_state.data['direction'].values[0])

        # === display the medical terminology ===#
        _data = session_state.data
        display_medical_terms(_data["term1"].values[0], _data["term2"].values[0])

    else:
        st.write("Press the above button to classify")


if __name__ == '__main__':
    st.title('MREC: Medical Relation Extraction Classification')
    st.markdown(
        """
            The **purpose** of the MREC is to automate relationship identification
            within the medical field.
        """
    )
    run()
