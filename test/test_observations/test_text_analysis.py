import unittest
import pandas as pd

import science.observations.text_analysis as ta
from science.utils import FEATURES

UTTERANCE = "blue circles are good, yellow triangles are bad; cheesecake? is wonderful"

POSITIVE_PHRASE = "magenta circles are great"
NEGATIVE_PHRASE = "yellow triangles are bad"
NEUTRAL_PHRASE = "i took a long walk on the beach yesterday"
CONTEXTUAL_POSITIVE_PHRASE = "yellow circle"
ZERO_PHRASE = "triangles are worth zero"

EMPTY_PHRASE = ""
PUNC_ONLY_PHRASE = ";;;;;!."


class MyTestCase(unittest.TestCase):

    def test_limited_punc_tokenization_splits(self):

        tokenized = ta.limited_punc_tokenization(UTTERANCE)
        self.assertEqual(len(tokenized), 3, msg="Tokenization split to wrong number of phrases.")

    def test_limited_punc_tokenization_edge_case_strings(self):

        tokenized = ta.limited_punc_tokenization(EMPTY_PHRASE)
        self.assertEqual(0, len(tokenized), msg="Tokenization split to wrong number of phrases.")

        tokenized = ta.limited_punc_tokenization(PUNC_ONLY_PHRASE)
        self.assertEqual(0, len(tokenized), msg="Tokenization split to wrong number of phrases.")

    def test_vader_sentiment(self):

        sentiment = ta.vader_observation(POSITIVE_PHRASE)
        self.assertLess(0, sentiment, msg="Basic Vader Sentiment returned {} for {}".format(sentiment, POSITIVE_PHRASE))

        sentiment = ta.vader_observation(CONTEXTUAL_POSITIVE_PHRASE)
        self.assertEqual(0, sentiment, msg="Sentiment returned {} for {}".format(sentiment, CONTEXTUAL_POSITIVE_PHRASE))

    def test_modified_vader_sentiment(self):

        sentiment = ta.modified_vader_observation(POSITIVE_PHRASE)
        self.assertLess(0, sentiment, msg="Sentiment returned {} for {}".format(sentiment, POSITIVE_PHRASE))

        sentiment = ta.modified_vader_observation(CONTEXTUAL_POSITIVE_PHRASE)
        self.assertEqual(ta.DEFAULT_SENTIMENT, sentiment,
                         msg="Sentiment returned {} for {}".format(sentiment, CONTEXTUAL_POSITIVE_PHRASE))

        sentiment = ta.modified_vader_observation(NEGATIVE_PHRASE)
        self.assertGreater(0, sentiment, msg="Sentiment returned positive for {}".format(sentiment, NEGATIVE_PHRASE))

        sentiment = ta.modified_vader_observation(NEUTRAL_PHRASE)
        self.assertEqual(ta.DEFAULT_SENTIMENT, sentiment,
                         msg="Sentiment returned {} for {}".format(sentiment, NEUTRAL_PHRASE))

        sentiment = ta.modified_vader_observation(ZERO_PHRASE)
        self.assertEqual(0, sentiment, msg="Sentiment returned {} for {}".format(sentiment, ZERO_PHRASE))

    def test_feature_extraction(self):

        shape_features = {'square': False, 'circle': True, 'triangle': False}
        color_features = {'blue': False, 'pink': True, 'white': False, 'yellow': False}

        self.assertEqual(shape_features, ta._shape_keywords(POSITIVE_PHRASE), msg="Failed to extract shapes.")
        self.assertEqual(color_features, ta._color_keywords(POSITIVE_PHRASE), msg="Failed to extract colors.")

    def test_feature_dataframe(self):

        df = pd.DataFrame({'shape': {0: 'circle'}, 'color': {0: 'pink'}, 'feature': {0: "pink|circle"}})

        self.assertTrue(df.equals(ta._phrase_to_feature_df(POSITIVE_PHRASE)), msg="Feature extraction dataframe failed.")

    def test_no_feature_dataframe(self):

        # should we return empty DF instead?
        self.assertIsNone(ta._phrase_to_feature_df(NEUTRAL_PHRASE), msg="Failed to return empty feature DF.")

    def test_phrase_reference_type_classifier(self):

        policy_phrase = "great job."
        self.assertEqual('trajectory', ta.classify_phrase_reference_type(policy_phrase))

        feature_phrase = "yellow circles are worth the most"
        self.assertEqual('feature', ta.classify_phrase_reference_type(feature_phrase))

    def test_cluster_references(self):

        top_left_phrases = ["the top left is great", "TOP left best", "avoid upper left"]

        bottom_right_phrases = ['The bottom right set of symbols were all positive or neutral. ',
                                'The bottom right was more valuable.',
                                'The lower right corner had four good ones']

        for p in top_left_phrases:
            self.assertEqual(0, ta._spatial_cluster_reference(p))

        for p in bottom_right_phrases:
            self.assertEqual(2, ta._spatial_cluster_reference(p))

        self.assertEqual(0, ta._spatial_cluster_reference("upper left"))
        self.assertEqual(3, ta._spatial_cluster_reference("bottom left"))

    def test_no_cluster_reference(self):

        no_cluster_phrases = ["5 in the middle of the three.", "all objects in the left", "far left corner"]

        for p in no_cluster_phrases:
            self.assertIsNone(ta._spatial_cluster_reference(p))

    def test_conjunctive_feature_reference_vector_from_features(self):

        vector = ta.reference_vector_from_features("I like circles")
        for c in ta.COLORS:
            self.assertAlmostEqual(1/3, vector["{}|circle".format(c)], msg="Should distribute reference across colors.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")

        vector = ta.reference_vector_from_features("get blue and purple")
        for c in ["blue", "pink"]:
            for s in ta.SHAPES:
                self.assertAlmostEqual(1 / 6, vector["{}|{}".format(c, s)],
                                       msg="Should distribute reference across all shapes of both pink and blue.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")

        vector = ta.reference_vector_from_features("get aqua")
        for s in ta.SHAPES:
            self.assertAlmostEqual(1 / 3, vector["blue|{}".format(s)], msg="Should distribute reference across shapes.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")

        vector = ta.reference_vector_from_features("pink circles are the best")
        self.assertAlmostEqual(1, vector["pink|circle"], msg="Should put whole reference on pink circles.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")

    def test_full_feature_reference_vector_from_features(self):

        vector = ta.reference_vector_from_features("get aqua", features=FEATURES)
        self.assertEqual(1, vector["blue"], msg="Should *NOT* distribute across shapes when using full feature set.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")

        vector = ta.reference_vector_from_features("get blue and purple", features=FEATURES)
        for c in ["blue", "pink"]:
            self.assertAlmostEqual(.5, vector["{}".format(c)],
                                   msg="Should distribute reference across both pink and blue.")
        self.assertAlmostEqual(1, vector.sum(), msg="Vector should sum to 1.")


if __name__ == '__main__':
    unittest.main()
