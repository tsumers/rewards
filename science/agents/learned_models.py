import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import pickle

from science.utils import CONJUNCTIONS, FEATURES
from science.observations.text_analysis import nn_preprocess_chat_phrase

OBJ_FEATURE_INPUT_DIM = len(FEATURES)
EMBEDDING_DIM = 30
REWARD_OUTPUT_DIM = 9


class TrajectoryFeedbackRewardPredictor(nn.Module):

    def __init__(self, words_to_idx, embedding_dim, object_feature_dim, reward_dim):
        super(TrajectoryFeedbackRewardPredictor, self).__init__()

        self._words_to_idx = words_to_idx

        self.embeddings = nn.EmbeddingBag(len(words_to_idx), embedding_dim)
        self.linear1 = nn.Linear(embedding_dim + object_feature_dim, 128)
        self.linear2 = nn.Linear(128, reward_dim)

    def forward(self, word_inputs, feature_inputs):
        """Run a forward pass of the trained reward predictor."""

        # Get embeddings from indexes
        embeds = self.embeddings(word_inputs)

        # Concatenate them with the feature-counts
        concated = torch.cat((embeds, feature_inputs.float()), 1)

        # Run the net
        out = F.relu(self.linear1(concated))
        out = self.linear2(out).double()

        return out

    def get_beliefs(self, utterance, trajectory, precision):
        """Generate a new belief observation based on this U/T tuple."""

        object_features = trajectory.feature_counts.sort_index().values
        object_feature_tensor = torch.tensor([object_features], dtype=torch.long)

        tokens = nn_preprocess_chat_phrase(utterance)
        embedding_indexes = [self._words_to_idx.get(w, self._words_to_idx["UNK"]) for w in tokens]
        text_tensor = torch.tensor([embedding_indexes], dtype=torch.long)

        feature_weights = self(text_tensor, object_feature_tensor).tolist()[0]

        feature_weight_dictionary = {c: {"mean": w * 10, "precision": precision} for c, w in
                                     zip(sorted(CONJUNCTIONS), feature_weights)}

        return feature_weight_dictionary

    @staticmethod
    def from_cv_fold(fold):
        """Load a model from a particular cross-validation training fold."""

        words_to_idx_path = Path(__file__).parent / "../../data/TrajectoryFeedbackRewardPredictor/words_to_idx/splits_{}.pkl".format(fold)

        with words_to_idx_path.open('rb') as f:
            fold_info = pickle.load(f)
            words_to_idx = fold_info["model_word_to_ix"]

        model = TrajectoryFeedbackRewardPredictor(words_to_idx, EMBEDDING_DIM, OBJ_FEATURE_INPUT_DIM, REWARD_OUTPUT_DIM)

        model_path = Path(__file__).parent / "../../data/TrajectoryFeedbackRewardPredictor/weights/joint_model_10fold_split_{}.pt".format(fold)
        model.load_state_dict(torch.load(model_path))

        return model
