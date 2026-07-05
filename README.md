# Emotion Classification from Text

Three-class sentiment classifier (anger / joy / fear) built from scratch on a scraped social-media comment dataset. Every preprocessing decision is documented with a reason — this README explains the thinking behind each step, what was tested along the way, and where the pipeline has known gaps.

---

## The dataset

5,937 short comments labeled with one of three emotions. Two columns: `Comment` (raw text) and `Emotion` (string label). No nulls. The class split is near-balanced — anger and joy each have 2,000 examples, fear has 1,937 — which means accuracy is a usable metric, though F1 macro is reported alongside it to catch any per-class skew the model might develop.

Text is short. Mean length is 19 words, median 17, max 64. That ceiling matters later: truncation at 64 tokens is safe for any sequence model, and it meant there was no need to spend time investigating padding strategies for long documents.

---

## Exploratory data analysis

Before touching anything, the notebook checks shape, dtypes, null counts, class balance, and text length. A manual sample of five comments per class confirmed that labels are mostly sensible, with a small amount of noise — one typo, a couple of comments where the label feels genuinely wrong, and a few ambiguous cases that could reasonably go either way. That noise was accepted as-is; the dataset is large enough that the model can learn through it, and cleaning every borderline case is expensive without clear payoff.

The more interesting finding came from the duplicate check. There are exactly 3 pairs of comments where the same text appears twice with two different labels — 6 rows total. What made this worth documenting is the pattern: every single conflict is anger versus fear. Not anger versus joy, not joy versus fear — always the same two classes. That's not random noise from a sloppy labeling process. It suggests anger and fear are the two emotions hardest to separate from text alone, and that annotators — or whatever process built this dataset — genuinely disagreed on sentences that sit on the boundary between those two feelings. It was worth flagging because it might foreshadow which class pair the model confuses most when we reach evaluation.

---

## Preprocessing pipeline

### Step 0 — Drop contradictory duplicates

The 6 conflicted rows are dropped first, before any transformation runs. The reason this happens at step zero rather than inside the cleaning pipeline is leakage: if a comment appears in both training and validation sets with different labels, the model gets contradictory signal and validation metrics become unreliable. Dropping them early and cleanly is the safest option.

### Step 1 — Lowercase and strip noise

Lowercase comes before stripping so all patterns are in a consistent case when the regex runs. The regex itself is deliberately simple: keep only `[a-zA-Z\s]`, remove everything else. Punctuation, digits, and html special characters go. The result is stored in a new `Cleaned_Comment` column — the original `Comment` is left untouched throughout, which makes it easy to go back and inspect any row that looks strange later in the pipeline.

One thing the regex cannot handle: purely alphabetic html fragments. Tags like `href`, `src`, `img`, `rel`, and `class` survive because they look like ordinary words to a character-level filter. There are 45 comments containing `http` and around 30 carrying `href` — enough to be worth addressing, but the right place to handle them is after tokenisation, not here.

### Step 2 — Tokenise

`nltk.word_tokenize()` splits each cleaned comment into a list of tokens. The reason for choosing `word_tokenize` over a simple `.split()` is contraction handling: `.split()` leaves `"don't"` as a single token, while `word_tokenize` correctly splits it into `["do", "n't"]`. For sentiment analysis, where negations carry meaning, getting that split right matters.

NLTK is used consistently across tokenisation, POS tagging, and lemmatisation. Mixing tokenisers from different libraries (e.g. spaCy for tokenisation, NLTK for tagging) introduces subtle incompatibilities — the POS tagger's expectations about token boundaries can differ from what another tokeniser produces.

### Step 3 — Remove html-junk tokens

After tokenisation, any token that is a known html artifact is filtered out. The list covers: `www`, `http`, `https`, `href`, `src`, `img`, `rel`, `bookmark`, `class`, `aligncenter`, and a handful of domain-name fragments (`clairee`, `bondmusings`) that appeared in the data.

Filtering at the token level rather than with a broader pre-tokenisation regex is deliberate. A regex that strips anything matching a URL pattern can accidentally clobber real words that share substrings. A fixed token-level set only removes exact matches — surgical rather than broad.

### Step 4 — POS tag

Each token list is passed through `nltk.pos_tag()`, which returns Penn Treebank tags (`VBG`, `NN`, `JJ`, etc.). POS tagging runs before stopword removal, not after. The reason: the tagger uses auxiliary verbs and articles as context clues to correctly label the words around them. Remove stopwords first and you strip away some of that context, which degrades tagging accuracy, which in turn degrades lemmatisation.

### Step 5 — Lemmatise

`WordNetLemmatizer` reduces tokens to their base form using the POS tag for guidance. A helper function `get_wordnet_pos()` maps Penn Treebank tags to WordNet's simpler format (`v`, `n`, `a`, `r`), defaulting to noun for anything unrecognised.

Tag-aware lemmatisation matters here. Without it, `"feeling"` gets treated as a noun and stays as `"feeling"`. With a verb tag, it correctly reduces to `"feel"`. The difference is meaningful for a model counting word frequencies as features.

**Known limitation: irregular verb forms.**

WordNet's lemmatisation dictionary only covers regular verb inflections. It correctly handles forms like `"feeling"` → `"feel"` and `"runs"` → `"run"`, but has no entries for irregular past tenses — so `"felt"`, `"went"`, `"had"`, and `"was"` all come back unchanged even when passed an explicit verb tag.

This was found during verification testing: after confirming that regular forms collapsed as expected, the same test was run on irregular past tenses and the output was wrong. To rule out a bug in the `get_wordnet_pos()` helper or the tagging step, the call was isolated outside the pipeline entirely — `lemmatizer.lemmatize("felt", wordnet.VERB)` run directly in a scratch cell. It still returned `"felt"`. That confirmed the issue is in WordNet's dictionary itself, not in any of the surrounding code.

As a sanity check, the same words were run through spaCy's lemmatiser. spaCy handles all four correctly: `"felt"` → `"feel"`, `"went"` → `"go"`, `"had"` → `"have"`, `"was"` → `"be"`. The gap is specific to WordNet, not to lemmatisation as an approach.

The decision was to document and stay with NLTK rather than switch. Reasons: rebuilding the pipeline around spaCy mid-project carries a real cost (different tokenisation assumptions, different tag formats, re-testing everything downstream); lemmatisation is one step in a larger pipeline and its impact on the final model isn't yet quantified; and the assignment scope doesn't require a perfect lemmatiser — it requires a justified and documented one. If error analysis at evaluation (Step 7) shows that irregular verb tokens are appearing frequently in misclassified examples, switching to spaCy becomes a concrete, evidence-backed next step rather than a speculative improvement made too early.

### Step 6 — Remove stopwords

Common low-signal words are removed using NLTK's English stopword list after lemmatisation. One explicit modification: negation words (`not`, `no`, `nor`, `never`) are removed from the stopword set before filtering. Negations carry direct emotion signal — "I do not feel happy" is meaningfully different from "I feel happy" — and stripping them would lose information the model needs.

### Step 7 — Flag empty or near-empty rows

After the full transformation chain, some comments may end up with zero or one token. Rather than silently dropping these, the pipeline flags them and prints the original comment text. That makes the decision to remove them explicit and documented, rather than a quiet data loss that gets noticed later.

### Step 8 — Label encoding and splitting

String labels are converted to integers using `sklearn`'s `LabelEncoder`. The encoder is saved so predictions can be decoded back to class names at inference time without re-fitting.

The train/validation/test split is 70/15/15 with stratification, which preserves class proportions in all three sets. After splitting, the class ratios are verified explicitly — a failed stratification can silently skew everything downstream and only show up as a confusing result at evaluation time.

---

## Tool choices

**Why NLTK over spaCy for the main pipeline?**

Both are reasonable choices. NLTK was selected because it keeps tokenisation, POS tagging, and lemmatisation tightly coupled — they all share the same tokenisation assumptions, which avoids subtle boundary mismatches.

spaCy's lemmatiser was tested directly and handles irregular verbs that WordNet's dictionary misses. Tested across multiple examples: `"felt"` → `"feel"`, `"went"` → `"go"`, `"had"` → `"have"`, `"was"` → `"be"` — all correctly reduced to their base forms, where NLTK leaves each one unchanged. Despite that advantage, switching just the lemmatiser while keeping NLTK's tagger would introduce a toolchain mismatch: the two tools make different tokenisation assumptions, and mixing them mid-pipeline is the kind of inconsistency that produces subtle errors that are hard to diagnose later.

**Why manual token filtering over vectoriser-level preprocessing?**

The html-junk filter is built as an explicit token-level set rather than relying on a vectoriser's `min_df` threshold or analyzer to quietly discard rare tokens. The reason is visibility: an explicit list documents exactly what's being removed and why, whereas a threshold-based approach silently drops tokens without a record of what was lost.

---

## Known gaps

- **Irregular verb lemmatisation** — `"felt"`, `"went"`, `"had"`, `"was"` survive as distinct tokens because WordNet's dictionary has no entry for them. Confirmed by isolated testing; spaCy handles all four correctly. Documented and accepted for now — worth revisiting if error analysis in Step 7 shows these tokens clustering in misclassified examples.
- **Label noise** — a small number of comments appear to carry incorrect labels (e.g. a comment about physical cold labeled `anger`, an apprehensive comment labeled `anger`). These are accepted as-is. If the confusion matrix at evaluation shows systematic mislabeling patterns, a targeted cleaning pass would be the next step.
- **Anger/fear boundary** — the duplicate analysis already showed these two classes sit close together in semantic space. If the model conflates them more than any other pair, that's an expected result, not a surprise.
