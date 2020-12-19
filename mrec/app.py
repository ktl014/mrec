""" Classification Streamlit Application

Module is used to launch classification system

"""
# Standard Dist Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Third party imports
import streamlit as st

# Project level imports
from mrec.data.dataset import load_data
from mrec.visualization import SessionState
import mrec.config as config

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
    data, id = None, 0
    session_state = SessionState.get(data=data, id=id)

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

if __name__ == '__main__':
    st.title('MREC: Medical Relation Extraction Classification')
    st.markdown(
        """
            The **purpose** of the MREC is to automate relationship identification
            within the medical field.
        """
    )
    run()
