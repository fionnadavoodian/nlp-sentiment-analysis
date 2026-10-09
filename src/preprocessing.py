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

CONTRACTIONS_PATTERN = re.compile(
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

lemmatizer = WordNetLemmatizer()
all_stop_words = set(stopwords.words("english"))


def expand_contractions(text):
    return CONTRACTIONS_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)


def normalize(text):
    return expand_contractions(text.lower())


def clean(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)


def tokenize_raw(text):
    return nltk.word_tokenize(text)


def remove_html_junk(tokens):
    return [t for t in tokens if t not in HTML_JUNK]


def get_wordnet_pos(pos_tag):
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


def lemmatize(tokens):
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]


def get_stop_words(protect_negation=True):
    return all_stop_words - NEGATION_WORDS if protect_negation else all_stop_words


def remove_stopwords(tokens, protect_negation=True):
    stop_words = get_stop_words(protect_negation)
    return [t for t in tokens if t not in stop_words]


# same steps as 02_preprocessing.ipynb, for notebooks that just need final tokens
def preprocess(text, apply_lemmatization=True, apply_stopwords=True, protect_negation=True):
    tokens = remove_html_junk(tokenize_raw(clean(normalize(text))))
    if apply_lemmatization:
        tokens = lemmatize(tokens)
    if apply_stopwords:
        tokens = remove_stopwords(tokens, protect_negation=protect_negation)
    return tokens
