import pandas as pd
import numpy as np

from science.observations.text_analysis import limited_punc_tokenization, classify_phrase_reference_type, \
    feature_observations_from_features, vader_observation, reference_vector_from_features

from science.observations.behavioral_analysis import feature_observations_from_trajectory, \
    feature_count_phi_df_from_trajectory, reference_vector_from_trajectory

from science.observations.spatial_analysis import feature_observations_from_spatial, reference_vector_from_spatial

from science.agents.beliefs import MultivariateNormal


def _reference_vector_from_phrase(phrase, trajectory, features):

    phrase_type = classify_phrase_reference_type(phrase)

    if phrase_type == 'trajectory' or phrase_type == 'action_behavioral':
        return reference_vector_from_trajectory(trajectory, features=features)

    elif phrase_type == 'feature':
        return reference_vector_from_features(phrase, features=features)

    elif phrase_type == 'action_spatial':
        return reference_vector_from_spatial(phrase, trajectory.environment, features=features)

    else:
        print("Unknown phrase type: {}".format(phrase))
        return None


def _observation_from_ref_valence(reference_vector, valence, precision):

    mean = pd.Series(data=reference_vector * valence)

    # Generate a covariance-precision via outer product and scaling
    precision_matrix = np.outer(reference_vector, reference_vector) * precision

    # Create and append a MVN for this observation
    return MultivariateNormal(mean, precision=precision_matrix)


def observations_from_utterance(utterance, trajectory, features, valence_func, valence_scale=1, precision_scale=1,
                                pragmatic_valence=None, pragmatic_precision=None):

    """Given an utterance and trajectory (object collection dataframe), return a list of Gaussian observations."""

    phrases = limited_punc_tokenization(utterance)
    observations = []

    for p in phrases:

        reference_vector = _reference_vector_from_phrase(p, trajectory, features)

        if reference_vector is not None:

            # Scale the (raw) valence output, which is [-1, 1] by our input value
            valence = valence_func(p) * valence_scale
            observation = _observation_from_ref_valence(reference_vector, valence, precision_scale)
            observations.append(observation)

            if pragmatic_valence is not None:

                # Generate an inverse reference vector by looking at whatever features were *not* referenced
                pragmatic_reference = (reference_vector == 0).astype(int)
                pragmatic_reference = pragmatic_reference / pragmatic_reference.sum()

                if pragmatic_precision is None:
                    pragmatic_precision = precision_scale

                # Create a second MVN for the "pragmatic" information
                observation = _observation_from_ref_valence(pragmatic_reference, pragmatic_valence, pragmatic_precision)
                observations.append(observation)

    return observations


def observation_df_from_feedback(utterance, trajectory,
                                 feature_attribution_func=feature_count_phi_df_from_trajectory,
                                 valence_observation_func=vader_observation):
    """Given an utterance and trajectory (object collection dataframe), return a dataframe of feature observations."""

    phrases = limited_punc_tokenization(utterance)
    observation_dfs = []

    for p in phrases:

        observation_df = None
        phrase_type = classify_phrase_reference_type(p)

        if phrase_type == 'trajectory':
            observation_df = feature_observations_from_trajectory(trajectory,
                                                                  feature_attribution_func=feature_attribution_func,
                                                                  features=trajectory.environment.features)

        elif phrase_type == 'feature':
            observation_df = feature_observations_from_features(p)

        elif phrase_type == 'action_spatial':
            observation_df = feature_observations_from_spatial(p, trajectory.environment)

        elif phrase_type == 'action_behavioral':

            # We don't have specific action references yet, so just assume this is related to the policy.
            observation_df = feature_observations_from_trajectory(trajectory,
                                                                  feature_attribution_func=feature_attribution_func,
                                                                  features=trajectory.environment.features)
        elif phrase_type == 'unk':
            print("Unknown phrase type: {}".format(p))

        if observation_df is not None:
            observation_df = _add_valence_and_precision(observation_df, p,
                                                        valence_observation_function=valence_observation_func)

            observation_df["phrase_type"] = phrase_type
            observation_dfs.append(observation_df)

    if observation_dfs:
        return pd.concat([df for df in observation_dfs])


def _add_valence_and_precision(feature_obs, phrase, valence_observation_function=feature_count_phi_df_from_trajectory):

    # Add valence (mu) and precision to the df
    valence = valence_observation_function(phrase)
    feature_obs["mean"] = valence

    # Combine reference and valence precision to get total precision
    feature_obs["precision"] = feature_obs["feature_precision"]
    feature_obs["sigma"] = np.sqrt(1 / feature_obs.precision)

    return feature_obs
