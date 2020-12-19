""" Classification Streamlit Application

Module is used to launch classification system

"""
# Standard Dist Imports
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Third party imports
import streamlit as st

# Project level imports
from mrec.data.dataset import load_data
from mrec.visualization import SessionState
import mrec.config as config
from mrec.model.MREClassifier import MREClassifier

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
    val = dataset.validation[['_unit_id', 'relation', 'sentence']].drop_duplicates()
    return val

def run():
    data, predictions, id = None, [], 0
    session_state = SessionState.get(id=id, data=data, predictions=predictions)

    dataset = load_data_to_cache(config.CSV_FNAMES)

    # === Generate USER ID button ===#
    # Get user id
    st.header('Generate User')
    if st.button('Click here to Generate User ID'):
        session_state.data = dataset.sample(n=1)
        session_state.id = session_state.data['_unit_id']
        st.write(session_state.id)

    # === View raw data ===#
    st.subheader('Raw Data')
    if st.checkbox('Show data') and (session_state.id != 0).all():
        st.dataframe(session_state.data)

    # === Classification button ===#
    # Recommend based off id
    st.header('Classify Relationship')
    recommend = st.button('Click here to Classify Relationship')
    if recommend:
        classifier = MREClassifier(model_weights='temp-model-weights')
        session_state.predictions = classifier.predict(X=session_state.data, features_names='temp-feature-names')

        # Report the predictions results
        st.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | USR ID: {session_state.id}]')
        st.write(f'Prediction: {session_state.predictions}')

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
