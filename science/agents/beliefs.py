import copy
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from science.utils import PILOT_FEATURES

PRIOR_PRECISION = 1/25
PRIOR_MEAN = 0


class MultivariateNormal(object):

    def __init__(self, mean, covariance=None, precision=None):
        self.mean = mean
        self.covariance = covariance
        self._precision = precision
        self._mvn = None

    def __str__(self):

        return "Mean:\n{}\nCovariance:\n{}\nPrecision:\n{}".format(self.mean, self.covariance, self.precision)

    @classmethod
    def from_labels(cls, labels, mean=0, var=25):
        mean = pd.Series(data=mean, index=sorted(labels))
        covariance = np.identity(len(labels)) * var

        return cls(mean, covariance)

    @property
    def precision(self):
        if self._precision is None:
            self._precision = np.linalg.inv(self.covariance)
        return self._precision

    @property
    def mvn(self):
        if self._mvn is None:
            self._mvn = multivariate_normal(self.mean, self.covariance)
        return self._mvn

    def multiply(self, other):

        posterior_precision = self.precision + other.precision
        posterior_covariance = np.linalg.inv(posterior_precision)

        posterior_mean = posterior_covariance@(self.precision@self.mean + other.precision@other.mean)
        posterior_mean = pd.Series(data=posterior_mean, index=self.mean.index)

        return self.__class__(posterior_mean, covariance=posterior_covariance)

    def update_from_observation(self, obs):

        reference_vector, valence, obs_precision = obs

        new_precision = self.precision + np.outer(reference_vector, reference_vector) * obs_precision
        new_covariance = np.linalg.inv(new_precision)

        obs_mean = reference_vector * valence * obs_precision
        new_mean = new_covariance@(self.precision@self.mean + obs_mean)
        new_mean_series = pd.Series(data=new_mean, index=self.mean.index)

        return self.__class__(new_mean_series, new_covariance)

    def sample_one_belief(self):

        return self.mvn.rvs(1)

    def sample_beliefs(self, n=100, as_df=True):

        rvs = self.mvn.rvs(n)
        if n == 1:
            rvs = [rvs]

        if as_df:
            return pd.DataFrame.from_records(rvs, columns=self.mean.index.values)
        else:
            return rvs

    def to_json(self):

        return json.dumps({"mean": self.mean.values.tolist(),
                           "covariance": self.covariance.tolist(),
                           "features": self.mean.index.tolist()})

    @classmethod
    def from_json(cls, data):

        mean = pd.Series(data=data["mean"], index=data["features"])
        covariance = np.array(data["covariance"])
        return cls(mean, covariance=covariance)


class IndependentNormals(object):

    def __init__(self, features=PILOT_FEATURES, mean=PRIOR_MEAN, precision=PRIOR_PRECISION):
        """Initialize priors {feature: {mu, precision}} for this set of features."""

        feature_params = {"mean": mean, "precision": precision}

        self.beliefs = {f: copy.deepcopy(feature_params) for f in features}

    def __getitem__(self, key):

        return self.beliefs[key]

    def literal_update(self, observation_df):
        """Given a prior {feature:{mu, precision}} and an observation dataframe, perform Bayesian update."""

        new_gaussian = copy.deepcopy(self)

        # See Murphy 2007 conjugate gaussians: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        for k, observation in observation_df.iterrows():

            # Retrieve the prior for this observation
            feature = observation["feature"]
            prior = new_gaussian[feature]

            # Weight prior and likelihood by precision
            weighted_prior_mean = prior["mean"] * prior["precision"]
            weighted_likelihood_mean = observation["mean"] * observation["precision"]

            # Generate posterior
            posterior_precision = prior["precision"] + observation["precision"]
            posterior_mean = (weighted_prior_mean + weighted_likelihood_mean) / posterior_precision

            # Write into dict
            new_gaussian[feature]["precision"] = posterior_precision
            new_gaussian[feature]["mean"] = posterior_mean

        return new_gaussian

    def to_json(self):

        return json.dumps(self.beliefs)

    @classmethod
    def from_json(cls, data):

        new_beliefs = cls()

        new_beliefs.beliefs = json.loads(data)

        return new_beliefs

    def pseudopragmatic_gaussian_update(self, observation_df, implicature_precision=PRIOR_PRECISION,
                                        implicature_valence=None):
        """In addition to literal update, tighten existing beliefs around unmentioned features."""

        new_gaussian = copy.deepcopy(self)
        unmentioned_features = [f for f in new_gaussian.beliefs if f not in observation_df["feature"].values]

        pragmatic_updates = []
        for f in unmentioned_features:

            # If no implicature valence, then use the existing beliefs and just tighten around them.
            if implicature_valence is None:
                implicature_valence = new_gaussian[f]["mean"]
            pragmatic_updates.append({"feature": f, "precision": implicature_precision, "mean": implicature_valence})

        implicature_df = pd.DataFrame.from_records(pragmatic_updates)

        full_update = pd.concat([observation_df, implicature_df])
        posterior = new_gaussian.literal_update(full_update)

        return posterior

    def sample_beliefs(self, n=100, as_df=True):
        """Sample from each (independent) feature-value dictionary and return joint hypotheses."""

        feature_samples = {}
        for feature, params in self.beliefs.items():

            rvs = norm.rvs(params["mean"], np.sqrt(1 / params["precision"]), size=n)
            feature_samples[feature] = rvs

        if as_df:
            return pd.DataFrame(feature_samples)
        else:
            return feature_samples
