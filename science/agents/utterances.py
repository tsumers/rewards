import itertools
from science.utils import FEATURES

features = [f.replace("|", " ") for f in FEATURES]
trajectory = ["job"]
actions = ["{} {}".format(v, h) for v, h in itertools.product(["upper", "bottom"], ["left", "right"])]

reference_terms = list(itertools.chain.from_iterable([features, trajectory, actions]))
valence_terms = ["the WORST :(", "bad", "not good", "zero", "ok", "good", "great", "AWESOME :)"]

utterance_manifold = ["{} {}".format(r, v) for r, v in itertools.product(reference_terms, valence_terms)]
