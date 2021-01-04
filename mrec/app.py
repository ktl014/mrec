""" Classification Streamlit Application

Module is used to launch classification system

"""
# Standard Dist Imports
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Third party imports
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Project level imports
import mrec.mrec
from mrec.visualization.medical_term_lookup import display_medical_terms
import mrec.config as config
from mrec.model.MREClassifier import MREClassifier
from mrec.model.score_mrec import accuracy, compute_cm
from mrec.data.dataset import load_data, load_rel_database
from mrec.features.transform import clean_text              # CountVectorizer dependency. Do not remove.

# Module level constants
enc = LabelEncoder()
FEATURES_LIST = ['_unit_id', 'relation', 'sentence', 'direction', 'term1', 'term2']
logger = logging.getLogger(__name__)
SEED = 5
SAMPLE_SIZE = 100


# === Load dataset as cache ===#
@st.cache
def load_data_to_cache(csv_fnames: dict) -> pd.DataFrame:
    """ Load dataset for web application

    Usage

    >>> try:
    >>>     dataset = load_rel_database_to_cache({'validation': 'dataset/raw/validation.csv'})
    >>>     data = dataset.sample(n=SAMPLE_SIZE, random_state=SEED)
    >>> except:
    >>>     st.error(
    >>>            "No such file could be found in the working directory. Make sure it is there and it is a csv-file.")

    Args:
        csv_fnames (dict): Dictionary of csv absolute paths to load data from. Default is to load the validation set

    Returns:
        pd.DataFrame: Validation set

    """
    dataset = load_data(csv_fnames)
    val = dataset.validation[FEATURES_LIST]
    relation_type = ['causes', 'treats']
    val = val[val['relation'].isin(relation_type)].drop_duplicates()

    return val


@st.cache(allow_output_mutation=True)
def load_rel_database_to_cache(db_path: str, table_name: str) -> pd.DataFrame:
    """Load dataset from relational database

    Usage

    >>> dataset = load_rel_database_to_cache(config.DB_PATH["mrec"], 'mrec_table')
    >>> data = dataset.sample(n=SAMPLE_SIZE, random_state=SEED)

    Args:
        db_path (str): database file path to load data from
        table_name (str): the name of the table in the database

    Returns:
        DataFrame: new dataframe after doing majority vote on `direction`
    """
    dataset = load_rel_database(db_path, table_name)
    relation_type = ['causes', 'treats']
    new_dataset = dataset[dataset['relation'].isin(relation_type)]

    # Pick majority vote on `direction`
    new_dataset = new_dataset.groupby(['_unit_id', 'relation', 'sentence', 'term1', 'term2'])['direction'].agg(pd.Series.mode).reset_index()

    return new_dataset


def run():
    st.sidebar.text("© Kevin T. Le & Duy Tran")
    st.title("MREC (Medical Relation Extraction Classifier)")
    st.sidebar.header("Loading data.")
    filename = st.sidebar.selectbox("Choose a file.", ("None", "samples"))
    if filename is not "None":
        try:
            dataset = load_rel_database_to_cache(config.DB_PATH["mrec"], 'mrec_table')
            data = dataset.sample(n=SAMPLE_SIZE, random_state=SEED)
        except:
            st.error(
                "No such file could be found in the working directory. Make sure it is there and it is a csv-file.")
        st.header("Load Data and Model")
        st.subheader("Load data.")
        st.write("Display sample data by ticking the checkbox in the sidebar.")

        agree = st.sidebar.checkbox('Display raw data.')
        if agree:
            st.dataframe(data)
        st.write("The initial data set contains", data['relation'].nunique(), "classes and", data.shape[0],
                 "random samples.")

        # DATA TRANSFORMATION
        st.subheader("Transform data.")
        st.write("Below is how we transform the dataset to feed into our model")
        #TODO come up with a visualization of this

        st.subheader("Load model.")
        st.write("A model was trained previously and is based on **random forest**.")
        with st.spinner("Trained model is being loaded"):
            model_weight = './models/random_forest.joblib'
            classifier = MREClassifier(model_weights=model_weight)
            #TODO Give some visualization of the model (e.g. feature importances, weights, accuracy)
            st.success("Model loaded!")

        # Run predictions on data sample
        st.header("Evaluate Model")
        data['relation_pred'], _ = classifier.predict(X=data['sentence'])
        acc = accuracy(data['relation'], data['relation_pred'])
        st.write(f'The test set accuracy (from out-of-bag samples) is', np.round(acc, 3), ".")

        st.write("Below, the **confusion matrix** for the classification problem is provided.")
        classes = ['causes', 'treats']
        gtruth = enc.fit_transform(data['relation'])
        predictions = enc.transform(data['relation_pred'])
        cm, _ = compute_cm(gtruth, predictions, classes)
        labels_ = enc.inverse_transform(gtruth)
        labels_repeated = []
        for _ in range(np.unique(labels_).shape[0]):
            labels_repeated.append(np.unique(labels_))
        source = pd.DataFrame({'predicted class': np.transpose(np.array(labels_repeated)).ravel(),
                               'true class': np.array(labels_repeated).ravel(),
                               'fraction': np.round(cm.ravel(), 2)})
        heat = alt.Chart(source, width=600, height=600, title="confusion matrix").mark_rect(opacity=0.7).encode(
            x='predicted class:N',
            y='true class:N',
            color=alt.Color('fraction:Q', scale=alt.Scale(scheme='blues')),
            tooltip="fraction").configure_axis(labelFontSize=20, titleFontSize=20)
        st.altair_chart(heat)

        # Single Prediction
        st.header("Prediction")
        st.subheader("Provide sample.")
        st.write("The model can now be used for **prediction** of the medical specialty. Below is another data "
                 "sample with its associated medical terms for the relationship")
        data = dataset.sample(n=1, random_state=SEED).iloc[0].to_dict()
        text = st.text_area("Write some text below", value=data['sentence'])

        relation_pred, probability = classifier.predict(X=[text])

        st.subheader("Assess result.")
        st.write(f"This sample originates from the specialty **{relation_pred}** with a probability of ",
                 np.round(probability, 3), ".")
        display_medical_terms(data["term1"], data["term2"])
    else:
        st.header("Introduction")
        #TODO rewrite this description
        st.write(
            "**This application guides you through the development of a language model that classifies clinical "
            "documents according to their medical relationship.**",
            "It is based on a term frequency–inverse document frequency (tf-idf) approach; tf-idf is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.",
            "It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.",
            "The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.",
            "tf–idf is one of the most popular term-weighting schemes today; 83% of text-based recommender systems in digital libraries use tf–idf. Note that tf-idf ignores the sequential aspect of a language.")

        st.write("The actual model itself is based on a random forest classifier.",
                 "Random forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.",
                 "In particular, trees that are grown very deep tend to learn highly irregular patterns: they overfit their training sets, i.e. have low bias, but very high variance. Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance.",
                 "Random forests can be used to rank the importance of variables in a regression or classification problem in a natural way.")

        st.write(
            "The model is developed with scikit-learn and was trained at an earlier time.")

        st.info("**Start by choosing a file**.")


if __name__ == '__main__':
    run()
