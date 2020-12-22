import nltk
import string


def get_wordnet_pos(word):
    """Get the correct grammar of a word

    Args:
        word: each word from sentences in the dataset

    Return:
        Grammar of that word

    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def clean_text(text):
    """Remove punctuation, tokenize, remove stopwords and lemmatize

    Args:
        text: a sentence from the dataset

    Returns:
        text: the text after removing punctuation, tokenize, remove stopwords, and lemmatizing

    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = text.split()
    text = [word for word in tokens if word not in stopwords]
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text]
    return text

if __name__ == "__main__":
    import nltk

    for nltk_resource in ['stopwords', 'averaged_perceptron_tagger', 'wordnet']:
        try:
            nltk.data.find(nltk_resource)
        except LookupError:
            nltk.download(nltk_resource)
