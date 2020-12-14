import copy
import itertools
import random
import re

import pandas as pd

from science.environments.environments import Environment
from science.utils import FEATURE_SYNONYM_DICT, FEATURE_ABBREVIATION_DICT


def switch_reward_function(shape_magnitude_game_record, chat, new_value_mask_config):
    """Data augmentation: switch (1) object features and (2) chat feature references for this game."""

    old_config = shape_magnitude_game_record["value_mask_config"]
    new_chat = switch_chat_features(chat, old_config, new_value_mask_config)

    new_game_record = copy.deepcopy(shape_magnitude_game_record)
    new_game_record["value_mask_config"]["color_config"] = new_value_mask_config["color_config"]
    new_game_record["value_mask_config"]["shape_config"] = new_value_mask_config["shape_config"]

    return Environment.from_shape_magnitude_game_record(new_game_record), new_chat


def switch_chat_features(chat, old_value_mask_config, new_value_mask_config):
    """Use the literal model's lexicon to swap out direct feature-based references in a piece of text."""

    new_chat = copy.deepcopy(chat.lower())

    old_color_config = old_value_mask_config["color_config"]
    new_color_config = new_value_mask_config["color_config"]
    new_chat = _replace_feature_words(new_chat, old_color_config, new_color_config)

    old_shape_config = old_value_mask_config["shape_config"]
    new_shape_config = new_value_mask_config["shape_config"]
    new_chat = _replace_feature_words(new_chat, old_shape_config, new_shape_config)

    return new_chat


def switch_token_features(chat_tokens, old_feature_config, new_feature_config):
    """Swap chat feature tokens based on lexicon and the new reward configuration."""

    shape_replace_dict = _build_replacement_dict(old_feature_config["shape_config"], new_feature_config["shape_config"])
    color_replace_dict = _build_replacement_dict(old_feature_config["color_config"], new_feature_config["color_config"])

    replace_dict = {**shape_replace_dict, **color_replace_dict}

    # If we have a replacement, use that; otherwise, return the original token
    return [replace_dict.get(t, t) for t in chat_tokens]


def _build_replacement_dict(old_feature_config, new_feature_config):

    replace_dict = {}
    for old, new in zip(old_feature_config.split("_"), new_feature_config.split("_")):

        old_feature_synonyms = FEATURE_SYNONYM_DICT[FEATURE_ABBREVIATION_DICT[old]]
        new_feature_word = random.sample(FEATURE_SYNONYM_DICT[FEATURE_ABBREVIATION_DICT[new]], 1)[0]

        for o in old_feature_synonyms:
            replace_dict[o] = new_feature_word

    return replace_dict


def _replace_feature_words(chat, old_feature_config, new_feature_config):

    replace_dict = _build_replacement_dict(old_feature_config, new_feature_config)
    regex = re.compile("|".join(map(re.escape, replace_dict.keys())))
    return regex.sub(lambda match: replace_dict[match.group(0)], chat)


def get_all_reward_configs():
    """Return all possible shape-color maps."""

    all_permutations = list(itertools.product(
        itertools.permutations(['y', 'c', 'm']),
        itertools.permutations(['s', '^', 'o'])
    ))

    reward_configs = {}
    for i, (colors, shapes) in enumerate(all_permutations):
        color_config = "{}_{}_{}".format(colors[0], colors[1], colors[2])
        shape_config = "{}_{}_{}".format(shapes[0], shapes[1], shapes[2])
        reward_configs[i] = {"color_config": color_config, "shape_config": shape_config}

    return reward_configs


def reward_series_from_config(value_mask_config):
    sign_config = value_mask_config["color_config"]
    magnitude_config = value_mask_config["shape_config"]

    reward_dict = {}
    signs = [-1, 0, 1]
    magnitudes = [2, 5.5, 9]

    for s, f1 in zip(signs, sign_config.split("_")):
        for m, f2 in zip(magnitudes, magnitude_config.split("_")):
            feature_name = "{}|{}".format(FEATURE_ABBREVIATION_DICT[f1], FEATURE_ABBREVIATION_DICT[f2])
            reward_dict[feature_name] = s * m / 10

    # Build a series with feature names as index
    reward_series = pd.Series(reward_dict)

    # Use that to sort (to get consistent order), then return just the values
    return reward_series.sort_index()

