import copy
import itertools
import json
import pandas as pd

from science.utils import PILOT_FEATURES, FEATURES
from science.environments.rewards import PilotFeaturizer, SignMagnitudeFeaturizer


class Environment(object):

    def __init__(self, object_list, psiturk_level_config, features=FEATURES):

        self.features = features
        self.psiturk_level_config = psiturk_level_config
        self.object_list = object_list

        self.object_df = self._dataframe_from_objects(object_list, features=features)
        self.level_number = psiturk_level_config['game_metadata']['level_number']

        self._possible_actions = None

    def to_json(self):

        return json.dumps({"features": self.features,
                           "psiturk_level_config": self.psiturk_level_config,
                           "object_list": self.object_list})

    @property
    def possible_actions(self):

        if self._possible_actions is None:
            self._possible_actions = self._all_possible_choices_on_level()

        return self._possible_actions

    @property
    def max_reward(self):

        return self.possible_actions.reward.max()

    def get_cluster(self, cluster_number):
        """Retrieve actions according to cluster number"""

        return self.object_df[self.object_df.cluster_id == cluster_number]

    @classmethod
    def from_pilot_config(cls, pilot_psiturk_level_config):
        """Given the configuration dictionary used for CogSci experiment levels, return the Environment."""

        pilot_psiturk_level_config = copy.deepcopy(pilot_psiturk_level_config)

        all_objects = pilot_psiturk_level_config["frame_data"]["objects"]
        for obj in all_objects:
            obj.update(PilotFeaturizer.features(obj))

        return cls(all_objects, pilot_psiturk_level_config, features=PILOT_FEATURES)

    @classmethod
    def from_shape_magnitude_config(cls, psiturk_level_config, value_mask_config, features=FEATURES):
        """Given a configuration dict and value mask, return the Environment."""

        psiturk_level_config = copy.deepcopy(psiturk_level_config)
        all_objects = psiturk_level_config["frame_data"]["objects"]

        # Get the reward map used for this game
        featurizer = SignMagnitudeFeaturizer.from_psiturk_reward_configuration(value_mask_config)

        for obj in all_objects:
            obj.update(featurizer.features(obj["value"], obj.get("color_value")))

        return cls(all_objects, psiturk_level_config, features=features)

    @classmethod
    def from_shape_magnitude_game_record(cls, psiturk_game_record, features=FEATURES):
        """Given a gameplay record from a shape-magnitude experiment, return the Environment."""

        psiturk_game_record = copy.deepcopy(psiturk_game_record)

        level_config = psiturk_game_record["config"]
        reward_function_config = psiturk_game_record["value_mask_config"]

        return cls.from_shape_magnitude_config(level_config, reward_function_config, features=features)

    @classmethod
    def _dataframe_from_objects(cls, object_list, features=FEATURES):

        object_df = pd.DataFrame(object_list)

        # Drop the "color value", but don't throw an error if it doesn't exist (i.e. no non-zero objects)
        object_df.drop(["color_value"], axis=1, inplace=True, errors='ignore')
        object_df.rename(columns={"value": "reward"}, inplace=True)
        object_df.fillna(0, inplace=True)
        object_df['cluster_id'] -= 1
        object_df.cluster_id = object_df.cluster_id.astype(float)

        missing_features = [f for f in features if f not in object_df.columns]
        for f in missing_features:
            object_df[f] = 0

        return object_df

    def _all_possible_choices_on_level(self):
        """Return all possible object choice-sets for this level."""

        df_list = []
        for cluster_id, object_dataframe in self.object_df.groupby("cluster_id"):

            cluster_choices = self._within_cluster_choice_set(object_dataframe)
            cluster_choices["cluster_id"] = cluster_id

            df_list.append(cluster_choices)

        return pd.concat(df_list)

    def _within_cluster_choice_set(self, cluster_object_dataframe):
        """Take a cluster of objects and return all possible combinations of 1-5 objects."""

        # Only call this function on a single cluster of objects-- otherwise you'll get a combinatoric explosion.
        assert(cluster_object_dataframe.cluster_id.nunique() == 1)

        combo_list = []
        feature_list = self.features + ["reward"]

        # Iterate over collecting 1-5 objects
        for i in range(1, len(cluster_object_dataframe) + 1):

            # Select all possible combinations of those objects
            for index_list in list(itertools.combinations(cluster_object_dataframe.index, i)):
                resulting_set = cluster_object_dataframe.loc[list(index_list)]
                feature_counts = dict(resulting_set[feature_list].sum())
                feature_counts.update({"locations": [(x, y) for x, y in zip(resulting_set.x, resulting_set.y)]})
                combo_list.append(feature_counts)

        # Glue them back together and return
        combo_df = pd.DataFrame.from_records(combo_list)

        return combo_df
