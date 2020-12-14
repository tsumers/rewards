import unittest

from science.environments.rewards import _feature_magnitude_dict, _feature_sign_dict, \
    SignMagnitudeFeaturizer, PilotFeaturizer

SHAPE_CONFIG = ["s", "^", "o"]
COLOR_CONFIG = ["c", "y", "m"]

TEST_PSITURK_CIRCLE_RECORD = {'cluster_id': 1, 'value': 2, 'x': 5, 'y': 10}
TEST_PSITURK_TRIANGLE_RECORD = {'cluster_id': 4, 'color_value': -10, 'value': 0, 'x': 10, 'y': 71}


def _feature_dict(joint_feature):
    """Helper function to simplify test data"""
    return {f: 1 for f in [joint_feature] + joint_feature.split("|")}


class MyTestCase(unittest.TestCase):

    def test_feature_creation_dicts(self):

        magnitudes = _feature_magnitude_dict(SHAPE_CONFIG)
        self.assertEqual(SHAPE_CONFIG[0], magnitudes[3])
        self.assertEqual(SHAPE_CONFIG[0], magnitudes[-3])
        self.assertEqual(SHAPE_CONFIG[1], magnitudes[-4])
        self.assertEqual(SHAPE_CONFIG[1], magnitudes[7])
        self.assertEqual(SHAPE_CONFIG[1], magnitudes[-7])
        self.assertEqual(SHAPE_CONFIG[2], magnitudes[-8])
        self.assertEqual(SHAPE_CONFIG[2], magnitudes[-10])

        signs = _feature_sign_dict(COLOR_CONFIG)
        self.assertEqual(COLOR_CONFIG[0], signs[-10])
        self.assertEqual(COLOR_CONFIG[0], signs[-1])
        self.assertEqual(COLOR_CONFIG[1], signs[0])
        self.assertEqual(COLOR_CONFIG[2], signs[1])
        self.assertEqual(COLOR_CONFIG[2], signs[10])

    def test_sign_magntiude_featurizer_from_config(self):

        config = {'color_config': 'm_c_y', 'shape_config': '^_s_o'}
        featurizer = SignMagnitudeFeaturizer.from_psiturk_reward_configuration(config)

        # Small-magnitude pos / neg
        self.assertEqual(featurizer.features(-1), _feature_dict("pink|triangle"))
        self.assertEqual(featurizer.features(1), _feature_dict("yellow|triangle"))

        # large magnitude pos / neg
        self.assertEqual(featurizer.features(-10), _feature_dict("pink|circle"))
        self.assertEqual(featurizer.features(10), _feature_dict("yellow|circle"))

        # zeros!
        self.assertEqual(featurizer.features(0, -10), _feature_dict("blue|circle"))
        self.assertEqual(featurizer.features(0, 1), _feature_dict("blue|triangle"))

        with self.assertRaises(ValueError):

            # Should throw an error if we try to create a zero-valued object without a distractor value for magnitude.
            featurizer.features(0)

    def test_pilot_featurizer(self):

        featurizer = PilotFeaturizer()

        cogsci_pink_triangle = {"value": 0, "color_value": 10}
        self.assertEqual(_feature_dict("pink|triangle"), featurizer.features(cogsci_pink_triangle))

        cogsci_yellow_triangle = {"value": 0, "color_value": -10}
        self.assertEqual(_feature_dict("yellow|triangle"), featurizer.features(cogsci_yellow_triangle))

        cogsci_yellow_square = {"value": -10}
        self.assertEqual(_feature_dict("yellow|square"), featurizer.features(cogsci_yellow_square))

        cogsci_blue_circle = {"value": 2}
        self.assertEqual(_feature_dict("blue|circle"), featurizer.features(cogsci_blue_circle))

        # Legacy tests
        self.assertEqual(1, featurizer.features(TEST_PSITURK_CIRCLE_RECORD)["circle"])
        self.assertEqual(1, featurizer.features(TEST_PSITURK_CIRCLE_RECORD)["blue"])
        self.assertEqual(1, featurizer.features(TEST_PSITURK_TRIANGLE_RECORD)["triangle"])
        self.assertEqual(1, featurizer.features(TEST_PSITURK_TRIANGLE_RECORD)["yellow"])


if __name__ == '__main__':
    unittest.main()





