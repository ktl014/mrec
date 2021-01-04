"""Top-level package for mrec."""

__author__ = """Kevin Le"""
__email__ = 'kevin.le@gmail.com'
__version__ = '0.1.0'

import nltk
for nltk_resource in ['stopwords', 'averaged_perceptron_tagger', 'wordnet']:
    try:
        nltk.data.find(nltk_resource)
    except LookupError:
        nltk.download(nltk_resource)
