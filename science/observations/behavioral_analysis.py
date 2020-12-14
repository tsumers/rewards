import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from science.utils import PILOT_FEATURES, FEATURES

# Filter out nearly-zero values (e.g. regularized coefficients from nested logit or TF-IDF)
MIN_BEHAVIORAL_PRECISION = .01


def reference_vector_from_trajectory(trajectory, features):

    feature_df = feature_count_phi_df_from_trajectory(trajectory)
    if feature_df is None:
        return None
    series = pd.Series(data=feature_df["feature_precision"].values, index=feature_df["feature"])
    series = series[series.index.isin(features)].sort_index()
    series = series / series.sum()
    return series


def feature_observations_from_trajectory(trajectory, feature_attribution_func, **kwargs):
    """Given a trajectory, return a DataFrame with feature precisions based on the supplied attribution function."""

    # Alternatives:
    # feature_count_phi_df --> naive feature counts
    # tf_idf_phi_df --> use TF-IDF to identify features that "stand out"
    # nested_logit_phi_df --> use nested logit model to estimate utility

    trajectory_features = feature_attribution_func(trajectory, **kwargs)

    # Don't allow close-to-zero values (primarily an issue for TF-IDF)
    if trajectory_features is not None:
        trajectory_features = trajectory_features[trajectory_features["feature_precision"] > MIN_BEHAVIORAL_PRECISION]

    return trajectory_features


def _feature_counts(object_dataframe, features=PILOT_FEATURES):

    return object_dataframe.loc[:, object_dataframe.columns.isin(features)].fillna(0).sum()


def feature_count_phi_df_from_trajectory(trajectory, **kwargs):

    if trajectory.actions is not None:
        return feature_count_phi_df_from_objects(trajectory.actions)

    else:

        num_objects = len(trajectory.feature_counts["locations"])
        feature_counts = trajectory.feature_counts[trajectory.feature_counts.index.isin(FEATURES)]
        normalized_feature_counts = feature_counts / num_objects

        choice_summary = normalized_feature_counts.to_frame(name="feature_precision").astype(float)
        choice_summary.index.name = "feature"

        return choice_summary.reset_index()


def feature_count_phi_df_from_objects(object_dataframe):
    """Return a normalized feature count for this choice set."""

    if len(object_dataframe) == 0:
        return None

    feature_counts = _feature_counts(object_dataframe)

    normalized_feature_counts = feature_counts / len(object_dataframe)

    choice_summary = normalized_feature_counts.to_frame(name="feature_precision")
    choice_summary.index.name = "feature"

    return choice_summary.reset_index()


def nested_logit_phi_df_from_trajectory(trajectory, features=PILOT_FEATURES):
    """Given a single level's object-selection choice dataframe, return normalized nested logit preferences."""

    if len(trajectory.actions) == 0:
        return None

    all_objects = trajectory.environment.object_df
    all_objects["collected"] = trajectory.actions.collected
    all_objects.collected.fillna(0, inplace=True)

    init_theta = np.random.rand(len(features)) - .5
    theta = _fit_model(all_objects, init_theta, features=features)

    # Normalize to 1
    theta /= theta.max()

    df = pd.DataFrame({"feature": features, "feature_precision": theta})

    return df


def _calc_choice_loglike(tdata, theta, lamb=1, features=PILOT_FEATURES):
    x = torch.tensor(tdata[features].values, dtype=torch.float)
    y = torch.tensor(tdata['collected'].values, dtype=torch.float)
    reward = x @ theta
    sel_prob = 1 / (1 + torch.exp(-reward))

    clusters = torch.LongTensor(tdata['cluster_id'].values)
    cluster_val = torch.zeros_like(clusters.unique(), dtype=torch.float)
    cluster_val = cluster_val.scatter_add_(0, clusters, reward * sel_prob)

    # compare to data
    c_collected = tdata.groupby('cluster_id')['collected'].sum()
    sel_cluster = c_collected.to_numpy().nonzero()[0][0]

    c_objs = x[tdata['cluster_id'] == sel_cluster]
    c_obj_choices = y[tdata['cluster_id'] == sel_cluster]

    within_cluster_loglike = \
        torch.log((y * sel_prob + (1 - y) * (1 - sel_prob))) * (clusters == sel_cluster)

    loglike = \
        cluster_val[sel_cluster] \
        - torch.logsumexp(cluster_val, 0) \
        + within_cluster_loglike.sum() \
        - lamb * theta.abs().sum()
    return loglike


def _fit_model(tdata, init_theta, features=PILOT_FEATURES):
    def calc_min_loss(theta):
        res = _calc_choice_loglike(tdata, torch.FloatTensor(theta), features=features, lamb=1)
        return -res.data.numpy().item()

    res = minimize(calc_min_loss,
                   init_theta,
                   method='powell',
                   options={
                       'xtol': .01,
                       'ftol': .0001
                   })
    return res.x
