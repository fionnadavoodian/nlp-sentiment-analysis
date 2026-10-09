import os
import pandas as pd
import numpy as np
from datasets import load_dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SPLIT_FILES = {
    "train": "go_emotions_train.csv",
    "validation": "go_emotions_validation.csv",
    "test": "go_emotions_test.csv",
}
LABEL_NAMES_FILE = "go_emotions_label_names.txt"

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


def download_go_emotions():
    # one-time pull from the Hub, written to data/raw/ so it's a fixed,
    # committed snapshot instead of a re-fetch on every run
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    fine_names = ds["train"].features["labels"].feature.names

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, LABEL_NAMES_FILE), "w") as f:
        f.write("\n".join(fine_names))

    for split, filename in SPLIT_FILES.items():
        df = pd.DataFrame({
            "text": ds[split]["text"],
            "labels": [",".join(map(str, labels)) for labels in ds[split]["labels"]],
            "id": ds[split]["id"],
        })
        df.to_csv(os.path.join(RAW_DIR, filename), index=False)


def load_raw_go_emotions():
    # downloads to data/raw/ first if not already there, then reads from disk
    if not all(os.path.exists(os.path.join(RAW_DIR, f)) for f in SPLIT_FILES.values()):
        download_go_emotions()

    with open(os.path.join(RAW_DIR, LABEL_NAMES_FILE)) as f:
        fine_names = f.read().splitlines()

    splits = {}
    for split, filename in SPLIT_FILES.items():
        df = pd.read_csv(os.path.join(RAW_DIR, filename))
        labels = [[int(x) for x in s.split(",")] for s in df["labels"]]
        splits[split] = (df["text"].tolist(), labels)
    return splits, fine_names


def load_go_emotions_ekman():
    # returns {"train": (texts, label_matrix, label_names), "validation": ..., "test": ...}
    # texts: list[str]
    # label_matrix: np.ndarray, shape (n, 7), multi-hot over EKMAN_CLASSES order
    # label_names: list[list[str]], active Ekman class names per row
    splits, fine_names = load_raw_go_emotions()

    fine_to_ekman = {}
    for ekman_idx, ekman_class in enumerate(EKMAN_CLASSES):
        for fine_name in EKMAN_MAPPING[ekman_class]:
            fine_to_ekman[fine_names.index(fine_name)] = ekman_idx

    def encode_split(texts, fine_labels_list):
        label_matrix = np.zeros((len(texts), len(EKMAN_CLASSES)), dtype=int)
        for row, fine_labels in enumerate(fine_labels_list):
            for fine_idx in fine_labels:
                label_matrix[row, fine_to_ekman[fine_idx]] = 1
        label_names = [
            [EKMAN_CLASSES[i] for i in range(len(EKMAN_CLASSES)) if row[i]]
            for row in label_matrix
        ]
        return texts, label_matrix, label_names

    return {name: encode_split(*splits[name]) for name in splits}


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
