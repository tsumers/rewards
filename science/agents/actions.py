import copy

import json
import numpy as np
import pandas as pd
from scipy.spatial import distance

from science.observations.behavioral_analysis import _feature_counts
from science.environments.environments import Environment
from science.utils import PILOT_FEATURES


class Trajectory(object):

    def __init__(self, environment, actions=None, feature_counts=None):

        if actions is None and feature_counts is None:
            raise ValueError("Can't create trajectory without either individual actions or feature count.")

        self.environment = environment
        self.actions = actions

        if feature_counts is None:
            self.feature_counts = _feature_counts(self.actions)
            self.reward = self.actions.reward.sum()

        else:
            self.feature_counts = feature_counts
            self.reward = self.feature_counts.get("reward")

    @property
    def pct_max_reward(self):

        return self.reward / self.environment.max_reward

    @classmethod
    def from_pilot_exp_psiturk_record(cls, psiturk_game_record):
        """Generate a trajectory (and contained environment) from a CogSci psiTurk record."""

        environment = Environment.from_pilot_config(psiturk_game_record["config"])
        actions = cls.actions_from_env(environment, psiturk_game_record)

        return cls(environment, actions=actions)

    @classmethod
    def from_exp_psiturk_record(cls, psiturk_game_record, value_mask_config=None):
        """Generate a trajectory (and contained environment) from a NeurIPS psiTurk record."""

        if value_mask_config is not None:
            psiturk_game_record = copy.deepcopy(psiturk_game_record)
            psiturk_game_record["value_mask_config"]["color_config"] = value_mask_config["color_config"]
            psiturk_game_record["value_mask_config"]["shape_config"] = value_mask_config["shape_config"]

        environment = Environment.from_shape_magnitude_game_record(psiturk_game_record)
        actions = cls.actions_from_env(environment, psiturk_game_record)

        return cls(environment, actions=actions)

    @classmethod
    def actions_from_env(cls, env, game_record):

        actions = copy.deepcopy(env.object_df)
        actions["collected"] = 0

        for event in game_record["game_events"]:
            if event["name"] == "object_collected":
                action_data = event["data"]["object"]
                actions.loc[((actions.x == action_data["x"]) & (actions.y == action_data["y"])), 'collected'] = 1

        return actions[actions.collected == 1]

    def to_json(self):

        return json.dumps({"index": self.feature_counts.index.tolist(),
                           "values": self.feature_counts.values.tolist(),
                           "environment": self.environment.to_json()})

    @classmethod
    def from_json(cls, json_blob):

        data = json.loads(json_blob)

        env_json = json.loads(data["environment"])
        environment = Environment(env_json["object_list"],
                                  env_json["psiturk_level_config"],
                                  features=env_json["features"])

        feature_counts = pd.Series(data=data["values"], index=data["index"])

        return cls(environment, feature_counts=feature_counts)

    def to_points(self):
        """Return a trajectory of the form {"x":[],"y":[]} for client-side rendering."""

        if self.actions is not None:
            targets = [(x, y) for x, y in zip(self.actions.x, self.actions.y)]
        else:
            targets = self.feature_counts["locations"]

        # Start at the center of the game board
        trajectory = [[38, 38]]
        while targets:
            # Sort by increasing distance from our last waypoint
            targets = sorted(targets, key=lambda o: distance.euclidean(trajectory[-1], [o[0], o[1]]))
            # Pop the closest and add to our trajectory
            next_waypoint = targets.pop(0)

            # For animation purposes: if this is our first object, add an intermediate waypoint so that Phaser
            # will render with a relatively smooth speed. Annoying technicality to get mitigate Phaser's point-wise
            # interpolation mechanics, which assume evenly spaced points.
            if len(trajectory) == 1:
                intermediate_x = (trajectory[0][0] + next_waypoint[0]) / 2
                intermediate_y = (trajectory[0][1] + next_waypoint[1]) / 2
                trajectory.append([int(intermediate_x), int(intermediate_y)])

            trajectory.append(next_waypoint)

        # Unpack our list of (x,y) tuples into separate x and y lists
        points_to_visit = {"x": [], "y": []}
        for x, y in trajectory:
            points_to_visit["x"].append(x)
            points_to_visit["y"].append(y)

        return points_to_visit


def optimal_choices_under_hypotheses(environment, sampled_reward_hypotheses, features=PILOT_FEATURES):
    """Return a dataframe with each row containing optimal object choices for each reward hypothesis.

    Equivalent to Boltzmann with temp set to 0."""

    possible_choices = environment.possible_actions
    optimal_choices = []

    for i, hypothesis in sampled_reward_hypotheses.iterrows():

        possible_choices["inferred_value"] = np.dot(possible_choices[features], hypothesis[features])
        hypothesis_optimal = possible_choices[possible_choices.inferred_value == possible_choices.inferred_value.max()]
        random_optimal = hypothesis_optimal.sample(n=1).iloc[0]

        optimal_choices.append(dict(random_optimal))

    choice_df = pd.DataFrame.from_records(optimal_choices)

    return choice_df