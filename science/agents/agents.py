import copy
import numpy as np
import json
import pandas as pd

from science.utils import PILOT_FEATURES, FEATURES, CONJUNCTIONS
from science.observations.observations import observations_from_utterance
from science.observations.text_analysis import modified_vader_observation

from science.agents.actions import Trajectory
from science.agents.beliefs import IndependentNormals, MultivariateNormal

from science.agents.learned_models import TrajectoryFeedbackRewardPredictor


class BaseAgent(object):

    def __init__(self, features=PILOT_FEATURES, prior=None, name=None):

        self._name = name
        self.features = sorted(features)

        if prior is None:
            prior = IndependentNormals(self.features)

        self.initial_beliefs = prior
        self.belief_state = copy.deepcopy(self.initial_beliefs)

    def __str__(self):

        if self._name is not None:
            return self._name
        else:
            return self.__repr__()

    def reset_beliefs(self):

        self.belief_state = self.initial_beliefs

    def update_beliefs(self, utterance, trajectory):
        """Given an utterance and a trajectory, update this agent's belief state."""

        raise NotImplementedError

    def execute_trajectories(self, new_level, n=100):

        # Get a belief and possible trajectories
        beliefs = self.belief_state.sample_beliefs(n=n, as_df=True).values
        possible_actions = new_level.possible_actions

        # Get the estimated value for each trajectory
        inferred_values = possible_actions[self.features].values@beliefs.T

        # Choose the optimal action and generate a new trajectory accordingly
        optimal_actions = np.argmax(inferred_values, axis=0)
        rewards = possible_actions.iloc[optimal_actions].reward
        max_reward = possible_actions.reward.max()

        return rewards, rewards/max_reward

    def execute_trajectory(self, new_level):
        """Given a new level, choose a trajectory according to this agent's beliefs and action-selection policy."""

        # Get a belief and possible trajectories
        belief = self.belief_state.sample_beliefs(n=1, as_df=True).values
        possible_actions = new_level.possible_actions

        # Get the estimated value for each trajectory
        inferred_values = possible_actions[self.features]@belief.T

        # Choose the optimal action and generate a new trajectory accordingly
        action_selection = possible_actions.iloc[np.argmax(inferred_values)]

        return Trajectory(new_level, feature_counts=action_selection)

    def to_json(self):
        """Return the agent's current belief state in JSON form for export."""

        return self.belief_state.to_json()


""" NEURAL NET AGENTS """


class EnsembleFeedforwardNeuralAgent(BaseAgent):

    def __init__(self, name="Ensemble Feedforward NN", precision=2):

        priors = IndependentNormals(features=CONJUNCTIONS)

        super().__init__(CONJUNCTIONS, name=name + " (P{})".format(precision), prior=priors)

        # Choose to load a specific model from a training fold
        self.models = [TrajectoryFeedbackRewardPredictor.from_cv_fold(fold) for fold in range(0, 10)]
        self.precision = precision

    def update_beliefs(self, utterance, trajectory):

        # Average each model's prediction into a single dictionary
        feature_weight_dictionary = {c: {"mean": 0, "precision": self.precision} for c in self.features}
        new_observations = [nn.get_beliefs(utterance, trajectory, precision=self.precision) for nn in self.models]
        for o in new_observations:
            for feature in o.keys():
                feature_weight_dictionary[feature]["mean"] += o[feature]["mean"] / len(self.models)

        reshaped = [dict({"feature": k}, **feature_weight_dictionary[k]) for k in feature_weight_dictionary.keys()]
        obs_df = pd.DataFrame.from_records(reshaped)

        self.belief_state = self.belief_state.pseudopragmatic_gaussian_update(obs_df, implicature_precision=0)

    @classmethod
    def from_json(cls, data):

        new_agent = cls()
        new_agent.belief_state = IndependentNormals.from_json(data)

        return new_agent

class StatefulFeedforwardNeuralAgent(BaseAgent):

    def __init__(self, features=CONJUNCTIONS, name="Stateful Feedforward Neural Net", precision=2, fold=4):

        priors = IndependentNormals(features=features)

        super().__init__(features, name=name + " (P{})".format(precision), prior=priors)

        # Choose to load a specific model from a training fold
        self.nn = TrajectoryFeedbackRewardPredictor.from_cv_fold(fold)
        self.precision = precision

    def update_beliefs(self, utterance, trajectory):

        new_observation = self.nn.get_beliefs(utterance, trajectory, precision=self.precision)
        reshaped = [dict({"feature": k}, **new_observation[k]) for k in new_observation.keys()]
        obs_df = pd.DataFrame.from_records(reshaped)

        self.belief_state = self.belief_state.pseudopragmatic_gaussian_update(obs_df, implicature_precision=0)

    @classmethod
    def from_json(cls, data):

        new_agent = cls()
        new_agent.belief_state = IndependentNormals.from_json(data)

        return new_agent


""" BAYESIAN LR AGENTS """


class MultivariateNormalLearner(BaseAgent):

    def __init__(self, features=FEATURES, prior=None, name="Default Multivariate",
                 valence_func=modified_vader_observation, valence_scale=30, precision_scale=3,
                 pragmatic_valence=None, pragmatic_precision=None, initial_variance=25):

        if prior is None:
            prior = MultivariateNormal.from_labels(features, var=initial_variance)

        self.valence_func = valence_func
        self.valence_scale = valence_scale
        self.precision_scale = precision_scale

        self.pragmatic_valence = pragmatic_valence
        self.pragmatic_precision = pragmatic_precision

        full_name = "{} (V{},P{})".format(name, self.valence_scale, self.precision_scale)
        if self.pragmatic_valence is not None:
            full_name = full_name + "(PragV{}, PragP{})".format(self.pragmatic_valence, self.pragmatic_precision)

        super().__init__(features, name=full_name, prior=prior)

    def update_beliefs(self, utterance, trajectory):

        observations = observations_from_utterance(utterance, trajectory,
                                                   features=self.features,
                                                   valence_func=self.valence_func,
                                                   valence_scale=self.valence_scale,
                                                   precision_scale=self.precision_scale,
                                                   pragmatic_valence=self.pragmatic_valence,
                                                   pragmatic_precision=self.pragmatic_precision)

        for obs in observations:
            self.belief_state = self.belief_state.multiply(obs)
            # self.belief_state = self.belief_state.update_from_observation(obs)

    @classmethod
    def from_json(cls, beliefs):

        beliefs = json.loads(beliefs)
        prior = MultivariateNormal.from_json(beliefs)
        return cls(prior=prior, features=beliefs["features"])


""" EXPERIMENT MODELS """


class ExpEnsembleNeuralLearner(EnsembleFeedforwardNeuralAgent):

    def __init__(self):

        super().__init__(name="ExpEnsembleNeuralLearner", precision=2)


class ExpLiteralLearner(MultivariateNormalLearner):

    def __init__(self, prior=None, features=FEATURES):

        assert sorted(features) == sorted(FEATURES), "Can only create Experiment Learner with full feature set."

        super().__init__(sorted(FEATURES), name="Interactive-Exp Literal", prior=prior,
                         valence_func=modified_vader_observation,
                         valence_scale=30, precision_scale=2)


class ExpPseudoPragmatic(MultivariateNormalLearner):

    def __init__(self, prior=None, features=FEATURES):

        assert sorted(features) == sorted(FEATURES), "Can only create Experiment Learner with full feature set."

        super().__init__(sorted(FEATURES), name="Interactive-Exp Pragmatic", prior=prior,
                         valence_func=modified_vader_observation,
                         valence_scale=30, precision_scale=2,
                         pragmatic_valence=-30, pragmatic_precision=2)







