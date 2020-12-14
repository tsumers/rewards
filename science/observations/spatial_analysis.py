import pandas as pd

from science.observations.text_analysis import _spatial_cluster_reference
from science.observations.behavioral_analysis import feature_count_phi_df_from_objects


def feature_observations_from_spatial(phrase, environment):

    cluster_reference = _spatial_cluster_reference(phrase)

    # We didn't find a reference to a cluster, so can't handle this.
    if cluster_reference is None:
        return None

    cluster_objects = environment.get_cluster(cluster_reference)
    feature_counts = feature_count_phi_df_from_objects(cluster_objects)

    return feature_counts


def reference_vector_from_spatial(phrase, environment, features):

    df = feature_observations_from_spatial(phrase, environment)

    if df is None:
        return None
    series = pd.Series(data=df["feature_precision"].values, index=df["feature"])
    series = series[series.index.isin(features)].sort_index()
    series = series / series.sum()
    return series

