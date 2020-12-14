import itertools

SHAPES = ['circle', 'square', 'triangle']
COLORS = ['blue', 'pink', 'yellow']
PILOT_COLORS = COLORS + ['white']

PILOT_CONJUNCTIONS = ["{}|{}".format(color, shape) for color, shape in itertools.product(PILOT_COLORS, SHAPES)]
CONJUNCTIONS = sorted(["{}|{}".format(color, shape) for color, shape in itertools.product(COLORS, SHAPES)])

PILOT_FEATURES = SHAPES + PILOT_COLORS + PILOT_CONJUNCTIONS
FEATURES = SHAPES + COLORS + CONJUNCTIONS

FEATURE_ABBREVIATION_DICT = {
    'c': 'blue',
    'y': 'yellow',
    'm': 'pink',
    'w': 'white',
    '^': 'triangle',
    'o': 'circle',
    's': 'square',
}

FEATURE_SYNONYM_DICT = {
    # Shapes
    "square": ["square", "box"],
    "circle": ["circle", "dot"],
    "triangle": ["triangle", "diamond"],
    # Colors
    "blue": ["blue", "cyan", "turquoise", "aqua", "teal"],
    "pink": ["pink", "purple", "magenta", "violet"],
    "white": ["white", "grey"],
    "yellow": ["yellow", "brown", "gold", "amber"]
}