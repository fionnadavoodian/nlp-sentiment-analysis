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

DATA_DIR = Path(__file__).parent / "../data/raw"

_fine_to_ekman = None


def _get_fine_to_ekman():
    global _fine_to_ekman
    if _fine_to_ekman is None:
        fine_names = (DATA_DIR / "go_emotions_label_names.txt").read_text(encoding="utf-8").splitlines()
        _fine_to_ekman = {}
        for ekman_idx, ekman_class in enumerate(EKMAN_CLASSES):
            for fine_name in EKMAN_MAPPING[ekman_class]:
                _fine_to_ekman[fine_names.index(fine_name)] = ekman_idx
    return _fine_to_ekman


def to_ekman_vector(fine_labels):
    fine_to_ekman = _get_fine_to_ekman()
    vec = [0] * len(EKMAN_CLASSES)
    for i in fine_labels:
        vec[fine_to_ekman[i]] = 1
    return vec
