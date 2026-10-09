import numpy as np
from datasets import load_dataset

# GoEmotions ships with a fixed train/val/test split and nothing here shuffles
# or samples rows, so output order is deterministic without needing a seed.

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


def load_go_emotions_ekman():
    # returns {"train": (texts, label_matrix, label_names), "validation": ..., "test": ...}
    # texts: list[str]
    # label_matrix: np.ndarray, shape (n, 7), multi-hot over EKMAN_CLASSES order
    # label_names: list[list[str]], active Ekman class names per row
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    fine_names = ds["train"].features["labels"].feature.names

    fine_to_ekman = {}
    for ekman_idx, ekman_class in enumerate(EKMAN_CLASSES):
        for fine_name in EKMAN_MAPPING[ekman_class]:
            fine_to_ekman[fine_names.index(fine_name)] = ekman_idx

    def encode_split(split):
        texts = list(split["text"])
        label_matrix = np.zeros((len(split), len(EKMAN_CLASSES)), dtype=int)
        for row, fine_labels in enumerate(split["labels"]):
            for fine_idx in fine_labels:
                label_matrix[row, fine_to_ekman[fine_idx]] = 1
        label_names = [
            [EKMAN_CLASSES[i] for i in range(len(EKMAN_CLASSES)) if row[i]]
            for row in label_matrix
        ]
        return texts, label_matrix, label_names

    return {name: encode_split(ds[name]) for name in ["train", "validation", "test"]}


def summarize(texts, label_matrix, label_names, split_name):
    n = len(texts)
    print(f"--- {split_name} ({n} rows) ---")
    for i, cls in enumerate(EKMAN_CLASSES):
        print(f"  {cls}: {label_matrix[:, i].sum()}")

    n_labels = label_matrix.sum(axis=1)
    multi_label_frac = (n_labels > 1).sum() / n
    empty_label_frac = (n_labels == 0).sum() / n
    print(f"  multi-label fraction: {multi_label_frac:.3f}")
    print(f"  empty-label fraction: {empty_label_frac:.3f}")

    neutral_idx = EKMAN_CLASSES.index("neutral")
    has_neutral = label_matrix[:, neutral_idx] == 1
    has_other = (label_matrix.sum(axis=1) - label_matrix[:, neutral_idx]) > 0
    print(f"  rows with neutral + another emotion: {(has_neutral & has_other).sum()}")


if __name__ == "__main__":
    splits = load_go_emotions_ekman()
    for name, (texts, label_matrix, label_names) in splits.items():
        summarize(texts, label_matrix, label_names, name)
