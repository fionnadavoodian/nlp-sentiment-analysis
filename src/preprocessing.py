"""Deterministic, corpus-independent text preprocessing.

Every function here depends only on its own input text (plus fixed lookup
tables defined in this module) — never on the rest of the corpus, never on
which split (train/val/test) the text came from. That means the exact same
function call is safe to apply identically to training data, validation
data, test data, or a brand-new string at inference time, with no risk of
leaking information across the train/val/test boundary.

What does NOT belong in this module, and why:
  - Building a vocabulary, fitting a TfidfVectorizer/CountVectorizer, or any
    other step whose output depends on the whole training set. Those are
    corpus-dependent: they must be fit on the training split only, then
    applied (not re-fit) to validation/test. That logic lives with whatever
    owns the split (currently 03_feature_engineering.ipynb).
  - Dropping the 3 contradictory-duplicate pairs or the 6 truncated-fragment
    rows from 02_preprocessing.ipynb. Those are one-time cleaning decisions
    tied to specific rows in this specific raw CSV, not a general text
    transformation that would make sense to apply to a new unseen string.

Configurability: `preprocess()` takes three independent flags so each one's
effect can be measured rather than assumed (Phase 3/4/7 of the project plan):
  - apply_lemmatization: run WordNet lemmatisation or skip it.
  - apply_stopwords: remove stopwords or skip it.
  - protect_negation: when stopwords ARE removed, whether negation words
    (not/no/never/...) are excluded from the stopword set. Irrelevant when
    apply_stopwords is False (nothing is being removed either way).
All three default to True, reproducing the pipeline's original behavior.
"""

import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# --- Fixed lookup tables (not fit on any corpus) ---

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
    """Expand both apostrophe'd (don't) and already-merged (dont) contraction
    forms to their full words. See notebooks/02_preprocessing.ipynb 2.1 for
    the reasoning and the deliberately-excluded ambiguous forms (ill, hell,
    well, its, were, id)."""
    return _CONTRACTIONS_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)


def normalize(text: str) -> str:
    """Lowercase + contraction expansion. No corpus dependency."""
    return expand_contractions(text.lower())


def clean(text: str) -> str:
    """Strip to [a-zA-Z\\s] only. Expects already-normalized text (run
    normalize() first, or contractions will be mangled before they can be
    expanded)."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def tokenize_raw(text: str) -> list[str]:
    """Word-tokenize only. HTML-junk tokens are removed separately by
    remove_html_junk() — kept apart because 02_preprocessing.ipynb teaches
    them as two distinct steps (2.2 then 2.3)."""
    return nltk.word_tokenize(text)


def remove_html_junk(tokens: list[str]) -> list[str]:
    """Drop known HTML-artifact tokens that survive the alpha-only strip
    (www, href, src, ...) because they're purely alphabetic."""
    return [t for t in tokens if t not in HTML_JUNK]


def tokenize(text: str) -> list[str]:
    """Word-tokenize and drop known HTML-junk tokens in one call, for
    callers that don't need the two steps separated (e.g. preprocess())."""
    return remove_html_junk(tokenize_raw(text))


def get_wordnet_pos(pos_tag: str) -> str:
    """Map a Penn Treebank POS tag to WordNet's simpler tag set, defaulting
    to noun for anything that doesn't match."""
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
    """POS-tag then tag-aware lemmatize. Known gap: WordNet's dictionary
    doesn't cover irregular verb forms (felt, went, was survive unchanged)
    — see 02_preprocessing.ipynb 2.5 for the spaCy-verified comparison."""
    tagged = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]


def get_stop_words(protect_negation: bool = True) -> set[str]:
    """The stopword set actually used for filtering. protect_negation=True
    excludes NEGATION_WORDS so negation survives stopword removal."""
    if protect_negation:
        return _all_stop_words - NEGATION_WORDS
    return _all_stop_words


def remove_stopwords(tokens: list[str], protect_negation: bool = True) -> list[str]:
    stop_words = get_stop_words(protect_negation)
    return [t for t in tokens if t not in stop_words]


def preprocess(
    text: str,
    apply_lemmatization: bool = True,
    apply_stopwords: bool = True,
    protect_negation: bool = True,
) -> list[str]:
    """Full deterministic pipeline: normalize -> clean -> tokenize ->
    [lemmatize] -> [stopword removal]. Safe to apply identically to
    train/val/test/inference text — no fitting, no corpus dependency.
    """
    tokens = tokenize(clean(normalize(text)))
    if apply_lemmatization:
        tokens = lemmatize(tokens)
    if apply_stopwords:
        tokens = remove_stopwords(tokens, protect_negation=protect_negation)
    return tokens
