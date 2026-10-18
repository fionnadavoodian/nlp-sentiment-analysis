import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

CONTRACTIONS = {
    "don't": 'do not',
    "didn't": 'did not',
    "doesn't": 'does not',
    "isn't": 'is not',
    "wasn't": 'was not',
    "weren't": 'were not',
    "aren't": 'are not',
    "can't": 'can not',
    "couldn't": 'could not',
    "wouldn't": 'would not',
    "shouldn't": 'should not',
    "won't": 'will not',
    "hadn't": 'had not',
    "hasn't": 'has not',
    "haven't": 'have not',
    "mustn't": 'must not',
    "needn't": 'need not',
    "shan't": 'shall not',
    "mightn't": 'might not',
    "ain't": 'is not',
    "i'm": 'i am',
    "i've": 'i have',
    "you're": 'you are',
    "you've": 'you have',
    "they're": 'they are',
    "they've": 'they have',
    "he's": 'he is',
    "she's": 'she is',
    "let's": 'let us',
    "what's": 'what is',
    "who's": 'who is',
    "that's": 'that is',
    "there's": 'there is',
}

# One maintained mapping; derive matching variants and the compact lookup from it.
# Do not expand bare "lets": it can be an ordinary verb rather than "let us".
_CONTRACTION_LOOKUP = {form.replace("'", ""): expansion for form, expansion in CONTRACTIONS.items()}
CONTRACTIONS_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(form).replace("'", r"(?:'|\s+)" if form == "let's" else r"(?:'|\s+)?")
        for form in sorted(CONTRACTIONS, key=len, reverse=True)
    ) + r")\b"
)

HTML_JUNK = {
    "www", "http", "https", "href", "src", "img",
    "aligncenter",
}

NEGATION_WORDS = {
    "not", "no", "nor", "never", "none", "nothing",
    "neither", "nowhere", "hardly", "scarcely", "barely",
}

lemmatizer = WordNetLemmatizer()
all_stop_words = set(stopwords.words("english"))


def expand_contractions(text):
    def expand_match(match):
        compact = re.sub(r"['\s]", "", match.group(0))
        return _CONTRACTION_LOOKUP[compact]

    return CONTRACTIONS_PATTERN.sub(expand_match, text)


def normalize(text):
    return expand_contractions(text.lower().replace("’", "'"))


def clean(text):
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def tokenize_raw(text):
    return nltk.word_tokenize(text)


def remove_html_junk(tokens):
    return [t for t in tokens if t not in HTML_JUNK]


def get_wordnet_pos(tag):
    return {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }.get(tag[:1])


def lemmatize(tokens):
    result = []
    for word, tag in nltk.pos_tag(tokens):
        pos = get_wordnet_pos(tag)
        result.append(
            lemmatizer.lemmatize(word, pos) if pos else word
        )
    return result


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
