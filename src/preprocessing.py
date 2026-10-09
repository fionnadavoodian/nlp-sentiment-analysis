"""Deterministic text preprocessing shared across notebooks.

No function here depends on the corpus or the split - same behavior on
train/val/test/inference text. Corpus-dependent steps (vocabulary, TF-IDF
fitting) and dataset-specific row cleanup (duplicate/fragment removal)
stay out of this file; see 02_preprocessing.ipynb for the reasoning behind
each step below.
"""

import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

CONTRACTIONS = {
    "dont": "do not", "don't": "do not",
    "didnt": "did not", "didn't": "did not",
    "doesnt": "does not", "doesn't": "does not",
    "isnt": "is not", "isn't": "is not",
    "wasnt": "was not", "wasn't": "was not",
    "werent": "were not", "weren't": "were not",
    "arent": "are not", "aren't": "are not",
    "cant": "can not", "can't": "can not",
    "couldnt": "could not", "couldn't": "could not",
    "wouldnt": "would not", "wouldn't": "would not",
    "shouldnt": "should not", "shouldn't": "should not",
    "wont": "will not", "won't": "will not",
    "hadnt": "had not", "hadn't": "had not",
    "hasnt": "has not", "hasn't": "has not",
    "havent": "have not", "haven't": "have not",
    "mustnt": "must not", "mustn't": "must not",
    "neednt": "need not", "needn't": "need not",
    "shant": "shall not", "shan't": "shall not",
    "mightnt": "might not", "mightn't": "might not",
    "aint": "is not", "ain't": "is not",
    "im": "i am", "i'm": "i am",
    "ive": "i have", "i've": "i have",
    "youre": "you are", "you're": "you are",
    "youve": "you have", "you've": "you have",
    "theyre": "they are", "they're": "they are",
    "theyve": "they have", "they've": "they have",
    "hes": "he is", "he's": "he is",
    "shes": "she is", "she's": "she is",
    "lets": "let us", "let's": "let us",
    "whats": "what is", "what's": "what is",
    "whos": "who is", "who's": "who is",
    "thats": "that is", "that's": "that is",
    "theres": "there is", "there's": "there is",
}

_CONTRACTIONS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b"
)

HTML_JUNK = {
    "www", "http", "https", "href", "src", "img",
    "rel", "bookmark", "class", "aligncenter",
    "clairee", "bondmusings",
}

NEGATION_WORDS = {
    "not", "no", "nor", "never", "none", "nothing",
    "neither", "nowhere", "hardly", "scarcely", "barely",
}

_lemmatizer = WordNetLemmatizer()
_all_stop_words = set(stopwords.words("english"))


def expand_contractions(text: str) -> str:
    return _CONTRACTIONS_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)


def normalize(text: str) -> str:
    return expand_contractions(text.lower())


def clean(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", text)


def tokenize_raw(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def remove_html_junk(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in HTML_JUNK]


def get_wordnet_pos(pos_tag: str) -> str:
    if pos_tag.startswith("J"):
        return wordnet.ADJ
    elif pos_tag.startswith("V"):
        return wordnet.VERB
    elif pos_tag.startswith("N"):
        return wordnet.NOUN
    elif pos_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(tokens: list[str]) -> list[str]:
    tagged = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]


def get_stop_words(protect_negation: bool = True) -> set[str]:
    return _all_stop_words - NEGATION_WORDS if protect_negation else _all_stop_words


def remove_stopwords(tokens: list[str], protect_negation: bool = True) -> list[str]:
    stop_words = get_stop_words(protect_negation)
    return [t for t in tokens if t not in stop_words]


def preprocess(
    text: str,
    apply_lemmatization: bool = True,
    apply_stopwords: bool = True,
    protect_negation: bool = True,
) -> list[str]:
    tokens = remove_html_junk(tokenize_raw(clean(normalize(text))))
    if apply_lemmatization:
        tokens = lemmatize(tokens)
    if apply_stopwords:
        tokens = remove_stopwords(tokens, protect_negation=protect_negation)
    return tokens
