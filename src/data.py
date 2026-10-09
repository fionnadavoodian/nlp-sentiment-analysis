from pathlib import Path

EKMAN_CLASSES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}

DATA_DIR = Path(__file__).parent.parent / "data/raw"

FINE_NAMES = (DATA_DIR / "go_emotions_label_names.txt").read_text(encoding="utf-8").splitlines()

FINE_TO_EKMAN = {}
for ekman_idx, ekman_class in enumerate(EKMAN_CLASSES):
    for fine_name in EKMAN_MAPPING[ekman_class]:
        FINE_TO_EKMAN[FINE_NAMES.index(fine_name)] = ekman_idx


def to_ekman_vector(fine_labels):
    vec = [0] * len(EKMAN_CLASSES)
    for i in fine_labels:
        vec[FINE_TO_EKMAN[i]] = 1
    return vec
