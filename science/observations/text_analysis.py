import itertools
import pickle
import re
import string

from pathlib import Path
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from science.utils import FEATURE_SYNONYM_DICT, CONJUNCTIONS, SHAPES, COLORS

sid = SentimentIntensityAnalyzer()

tf_idf_path = Path(__file__).parent / "../../data/tf_idf_vectorizer.pkl"
with tf_idf_path.open('rb') as f:
    utterance_vectorizer = pickle.loads(f.read())

utterance_model_path = Path(__file__).parent / "../../data/utterance_classifier.pkl"
with utterance_model_path.open('rb') as f:
    utterance_classifier = pickle.loads(f.read())

tf_idf_path = Path(__file__).parent / "../../data/tf_idf_vectorizer_augmented.pkl"
with tf_idf_path.open('rb') as f:
    augmented_utterance_vectorizer = pickle.loads(f.read())

utterance_model_path = Path(__file__).parent / "../../data/utterance_classifier_augmented.pkl"
with utterance_model_path.open('rb') as f:
    augmented_utterance_classifier = pickle.loads(f.read())

DEFAULT_SENTIMENT = .5

FEATURE_PRECISION = 1

PHRASE_TYPE_ENUM = {0: 'trajectory',
                    1: 'feature',
                    2: 'action_spatial',
                    3: 'action_behavioral',
                    4: 'unk'}


def limited_punc_tokenization(chat_text):
    """Given a chat message, segment on punctuation and return list of phrases."""

    phrases = re.split('[!.,;|]', chat_text.strip())
    return [p.strip() for p in phrases if p != '']


def classify_phrase_reference_type(phrase):
    """Use sklearn classifier trained on CogSci corpus to classify reference type of utterance."""

    preprocessed = _preprocess_chat_phrase(phrase)
    tf_idf_vector = augmented_utterance_vectorizer.transform([preprocessed])
    classification = augmented_utterance_classifier.predict(tf_idf_vector.reshape(1, -1))[0]

    return PHRASE_TYPE_ENUM[classification]


def reference_vector_from_features(phrase, features=CONJUNCTIONS):

    shape_features = _feature_list_from_dict(_shape_keywords(phrase))
    color_features = _feature_list_from_dict(_color_keywords(phrase))

    if shape_features and not color_features:
        if features == CONJUNCTIONS:
            color_features = COLORS
        else:
            color_features = [""]
    elif color_features and not shape_features:
        if features == CONJUNCTIONS:
            shape_features = SHAPES
        else:
            shape_features = [""]
    elif not color_features and not shape_features:
        return None

    full_phrase_df = pd.DataFrame.from_records(
        [{"shape": k[0], "color": k[1]} for k in itertools.product(shape_features, color_features)])

    full_phrase_df["feature"] = full_phrase_df.apply(lambda x: _feature_from_columns(x), axis=1)

    full_feature_series = pd.Series(data=0, index=features)
    phrase_series = pd.Series(data=1, index=full_phrase_df["feature"])
    full_feature_series = phrase_series.add(full_feature_series, fill_value=0)
    full_feature_series = full_feature_series / full_feature_series.sum()

    return full_feature_series


def feature_observations_from_features(phrase):
    """Given a feature-phrase, return a DataFrame with feature precisions."""

    # Extract features via string matching
    phrase_features = _phrase_to_feature_df(phrase)

    if phrase_features is None:
        return None

    # Set precision to 1 (our max / default) since these are exact string matches.
    phrase_features["feature_precision"] = FEATURE_PRECISION

    return phrase_features


def vader_observation(phrase):

    valence = sid.polarity_scores(phrase)["compound"]

    return valence


def modified_vader_observation(phrase, default_sentiment=DEFAULT_SENTIMENT):
    """Given a phrase, return its sentiment (mean) based on NLTK's Vader sentiment analyzer.

    See the Vader docs for more information: https://github.com/cjhutto/vaderSentiment#about-the-scoring
    """

    # These really *do* reflect a value of zero, so don't put positive bias on them!
    if any(zero in phrase.lower() for zero in ["zero", "worthless", "nothing", " 0"]):
        return 0

    # Vader believes these are negative / positive sentiment respectively, so just delete them.
    # phrase = phrase.replace("lower", "").replace("top", "")

    # Run basic Vader algorithm. If it picks up a valence, return that.
    basic_vader_sentiment = sid.polarity_scores(phrase)["compound"]
    if abs(basic_vader_sentiment) != 0:
        return basic_vader_sentiment

    # If vader doesn't detect a valence, search for negative keywords
    # if any(neg in phrase.lower() for neg in ["don't", "not", "stay away", "avoid", "except", "negative", ' -']):
    #     return -default_sentiment

    # Default to positive sentiment. "Positive implicature" of sorts...
    return default_sentiment


def _spatial_cluster_reference(phrase):

    for cluster_reference in ["top left", "upper left"]:
        if cluster_reference in phrase.lower():
            return 0

    for cluster_reference in ["top right", "upper right"]:
        if cluster_reference in phrase.lower():
            return 1

    for cluster_reference in ["bottom right", "lower right"]:
        if cluster_reference in phrase.lower():
            return 2

    for cluster_reference in ["bottom left", "lower left"]:
        if cluster_reference in phrase.lower():
            return 3

    return None


def _phrase_to_feature_df(phrase):
    """Given a phrase, return a dataframe consisting of shape-color matches."""

    shape_features = _feature_list_from_dict(_shape_keywords(phrase))
    color_features = _feature_list_from_dict(_color_keywords(phrase))

    if shape_features and not color_features:
        color_features = [""]
    elif color_features and not shape_features:
        shape_features = [""]
    elif not color_features and not shape_features:
        return None

    full_phrase_df = pd.DataFrame.from_records(
        [{"shape": k[0], "color": k[1]} for k in itertools.product(shape_features, color_features)])

    full_phrase_df["feature"] = full_phrase_df.apply(lambda x: _feature_from_columns(x), axis=1)

    return full_phrase_df


def _shape_keywords(phrase):
    """Run basic string matching for references to shape features, return boolean dictionary."""

    phrase = phrase.lower()
    return {
        "square": any([s in phrase for s in FEATURE_SYNONYM_DICT["square"]]),
        "circle": any([s in phrase for s in FEATURE_SYNONYM_DICT["circle"]]),
        "triangle": any([s in phrase for s in FEATURE_SYNONYM_DICT["triangle"]]),
    }


def _color_keywords(phrase):
    """Run basic string matching for references to color features, return boolean dictionary."""

    phrase = phrase.lower()
    return {
        "blue": any([s in phrase for s in FEATURE_SYNONYM_DICT["blue"]]),
        "pink": any([s in phrase for s in FEATURE_SYNONYM_DICT["pink"]]),
        "white": any([s in phrase for s in FEATURE_SYNONYM_DICT["white"]]),
        "yellow": any([s in phrase for s in FEATURE_SYNONYM_DICT["yellow"]]),
    }


def _feature_list_from_dict(feature_dict):

    if not feature_dict:
        return None

    return [f for f in feature_dict if feature_dict[f]]


def _preprocess_chat_phrase(chat_phrase):

    # Remove punctuation
    chat_phrase = chat_phrase.translate(str.maketrans('', '', string.punctuation))
    chat_tokens = nltk.word_tokenize(chat_phrase)

    # Lemmaize and filter the words
    chat_tokens = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in chat_tokens]
    chat_tokens = [t for t in chat_tokens if t not in stopwords.words("english")]

    # Convert single-digit numerals to words, i.e. "0" --> "zero"
    number_map = {"0": "zero", "1": "one", "2": "two", "3": "three",
                  "4": "four", "5": "five", "6": "six", "7": "seven",
                  "8": "eight", "9": "nine"}

    chat_tokens = [number_map.get(t, t) for t in chat_tokens]

    return " ".join(chat_tokens)


def _feature_from_columns(row):
    if row["color"] == "":
        return row["shape"]
    elif row["shape"] == "":
        return row["color"]
    else:
        return "{}|{}".format(row["color"], row["shape"])


def nn_preprocess_chat_phrase(chat_phrase):

    chat_phrase = chat_phrase.replace("|", " ")
    chat_phrase = chat_phrase.replace("'", "")

    # Remove punctuation
    chat_phrase = chat_phrase.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower()
    chat_tokens = nltk.word_tokenize(chat_phrase)

    # Lemmaize and filter the words
    chat_tokens = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in chat_tokens]

    # Convert single-digit numerals to words, i.e. "0" --> "zero"
    number_map = {"0": "zero", "1": "one", "2": "two", "3": "three",
                  "4": "four", "5": "five", "6": "six", "7": "seven",
                  "8": "eight", "9": "nine"}

    chat_tokens = [number_map.get(t, t) for t in chat_tokens]

    if not chat_tokens:
        chat_tokens = [""]
    return chat_tokens
