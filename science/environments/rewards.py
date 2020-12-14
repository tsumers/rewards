from science.utils import FEATURE_ABBREVIATION_DICT


class Featurizer(object):
    """Class to wrap various reward configurations. Takes object values and returns appropriate features."""

    def __init__(self):
        pass

    def features(self, reward):
        raise NotImplementedError


class SignMagnitudeFeaturizer(Featurizer):
    """Take configuration strings """

    def __init__(self, feature_sign_list, feature_magnitude_list):

        self.sign_dictionary = _feature_sign_dict(feature_sign_list)
        self.magnitude_dictionary = _feature_magnitude_dict(feature_magnitude_list)

        super().__init__()

    def features(self, object_reward, distractor_reward=None):

        sign_feature = FEATURE_ABBREVIATION_DICT[self.sign_dictionary[object_reward]]

        # For zero-valued objects
        if distractor_reward:
            magnitude_feature = FEATURE_ABBREVIATION_DICT[self.magnitude_dictionary[distractor_reward]]
        else:
            if object_reward == 0:
                raise ValueError("Featurizer given reward of 0 without distractor value. "
                                 "Unable to generate magnitude feature for this object.")

            magnitude_feature = FEATURE_ABBREVIATION_DICT[self.magnitude_dictionary[object_reward]]

        conjunction = "{}|{}".format(sign_feature, magnitude_feature)

        return {f: 1 for f in [sign_feature, magnitude_feature, conjunction]}

    @classmethod
    def from_psiturk_reward_configuration(cls, psiturk_reward_configuration):

        feature_sign_list = psiturk_reward_configuration['color_config'].split("_")
        feature_magitude_list = psiturk_reward_configuration['shape_config'].split("_")

        return cls(feature_sign_list, feature_magitude_list)


class PilotFeaturizer(Featurizer):

    @classmethod
    def features(cls, psiturk_object_record):
        """Given a psiTurk-logged object record from CogSci experiment, "featurize" it into perceptual appearance."""

        if psiturk_object_record["value"] == 0:
            shape = "triangle"
            color_value = psiturk_object_record.get("color_value")
        elif psiturk_object_record["value"] < 0:
            shape = "square"
            color_value = psiturk_object_record["value"]
        else:
            shape = "circle"
            color_value = psiturk_object_record["value"]

        if color_value <= -7:
            color = "yellow"
        elif color_value <= -4:
            color = "white"
        elif color_value <= 3:
            color = "blue"
        elif color_value <= 7:
            color = "white"
        else:
            color = "pink"

        return {shape: 1, color: 1, color + "|" + shape: 1}


def _feature_magnitude_dict(three_feature_list):
    """Given a feature list in order small, medium, large, return dictionary mapping point value to feature."""

    values_list = [[1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10]]

    feature_dict = {}
    for feature, values in zip(three_feature_list, values_list):

        for value in values:
            feature_dict[value] = feature
            feature_dict[-value] = feature

    return feature_dict


def _feature_sign_dict(three_feature_list):
    """Given a feature list in order -/0/+, return dictionary mapping point value to feature."""

    feature_dict = {}

    for i in list(range(1, 11)):
        feature_dict[-i] = three_feature_list[0]

    feature_dict[0] = three_feature_list[1]

    for i in list(range(1, 11)):
        feature_dict[i] = three_feature_list[2]

    return feature_dict