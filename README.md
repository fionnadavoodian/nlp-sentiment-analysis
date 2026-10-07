# Three-Class Emotion Classification from Text

A from-scratch NLP project for classifying short social-media comments into three emotions:

- **Anger**
- **Fear**
- **Joy**

The project is built around a simple question:

> **How much do different text representations, preprocessing strategies, classical machine-learning models, and transformer-based models affect emotion classification performance?**

Rather than applying a preprocessing pipeline blindly, the project documents the reasoning behind each NLP decision, tests alternative approaches, and evaluates whether each additional step actually improves the model.

---

## Project Status

🚧 **In progress**

The project currently contains the dataset investigation, exploratory data analysis, preprocessing experiments, and dataset preparation.

The next stages are classical NLP baselines, feature engineering, systematic evaluation, error analysis, and eventually a Hugging Face transformer model.

The pipeline is intentionally being developed incrementally so that each modeling decision can be evaluated rather than assumed to be beneficial.

---

## Dataset

The dataset contains **5,937 short social-media comments** labeled with one of three emotions:

| Emotion | Approx. Count |
| ------- | ------------: |
| Anger   |         2,000 |
| Joy     |         2,000 |
| Fear    |         1,937 |

The classes are therefore relatively well balanced.

The comments are short, with most containing only a small number of words, making this a suitable problem for comparing both traditional NLP methods and modern transformer-based approaches.

---

## Project Goals

The project has several goals:

1. Understand and investigate the quality of the dataset.
2. Develop a reproducible NLP preprocessing pipeline.
3. Establish simple and strong classical NLP baselines.
4. Compare different text representations and feature-engineering strategies.
5. Measure the actual effect of preprocessing decisions through ablation experiments.
6. Investigate model errors rather than relying only on aggregate accuracy.
7. Examine the ambiguity between **anger** and **fear**.
8. Compare a strong classical NLP model with a Hugging Face transformer.
9. Build a reusable inference pipeline for the final model.
10. Document the complete experimental process.

---

# 1. Exploratory Data Analysis

The first stage investigates the dataset before any modeling is performed.

The EDA covers:

- class distribution
- missing values
- duplicate comments
- contradictory duplicate labels
- text length
- token counts
- unusually short comments
- potentially problematic examples
- label quality

One particularly important finding was that the contradictory duplicate labels were exclusively between **anger and fear**.

This suggests a potential ambiguity between these two classes in the dataset. This is currently treated as a hypothesis rather than a conclusion and will be evaluated later using model confusion matrices and targeted error analysis.

---

# 2. Data Quality

Before modeling, the project investigates whether the dataset contains problematic examples.

This includes:

### Exact duplicates

Identical comments appearing multiple times are investigated to determine whether they contain consistent labels.

### Conflicting duplicates

When the same comment appears with different labels, the conflict is investigated rather than silently allowing the contradictory examples into the training data.

### Potentially problematic examples

Very short, incomplete, or otherwise suspicious comments are reviewed individually.

### Near duplicates

The final pipeline will also investigate highly similar comments to determine whether similar examples appear across train, validation, and test sets.

This is important because random splitting can produce overly optimistic results when nearly identical examples occur in different splits.

---

# 3. Preprocessing

The preprocessing pipeline is designed to preserve useful linguistic information while removing unnecessary noise.

The project investigates:

- lowercasing
- tokenization
- punctuation handling
- contraction handling
- negation
- stopword removal
- lemmatization
- short-document filtering

Importantly, preprocessing choices are treated as **experimental variables** rather than assumptions.

For example, stopword removal and lemmatization may sound beneficial, but they can also remove information useful for emotion classification.

Therefore, their usefulness will ultimately be determined through validation experiments.

---

# 4. Data Splitting

The cleaned dataset is divided into:

- **70% training**
- **15% validation**
- **15% test**

The splits are stratified to preserve the class distribution.

A fixed random seed is used to make the experiments reproducible.

Duplicate and data-quality checks are performed before splitting so that problematic duplicates are not accidentally distributed across different datasets.

The test set is reserved for final evaluation and is not used for model selection.

---

# 5. Classical NLP Baselines

The first modeling stage focuses on traditional NLP methods.

The baseline progression will include:

### Majority-class baseline

Establishes the minimum performance that a useful classifier should exceed.

### Bag-of-Words

Tests raw word-frequency representations.

### TF-IDF

Tests weighted lexical representations.

### Classical classifiers

The project will compare models such as:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

The purpose is not simply to find the highest score, but to understand how different representations and classifiers behave on the dataset.

---

# 6. Feature Engineering

The project investigates several representations:

- binary word presence
- word frequency
- TF
- TF-IDF
- unigrams
- bigrams
- potentially trigrams

The effect of n-grams is particularly relevant because emotion can depend on combinations of words.

For example:

```text
very happy
really afraid
so angry
```

may contain information that individual words alone do not capture.

Feature engineering will therefore be evaluated experimentally rather than assumed to improve performance.

---

# 7. Preprocessing Ablation

One of the main experiments is an ablation study.

Different preprocessing configurations will be compared while keeping the model and evaluation procedure controlled.

For example:

| Pipeline                    | Model               | Macro-F1 |
| --------------------------- | ------------------- | -------: |
| Minimal normalization       | TF-IDF + Linear SVM |        — |
| + punctuation normalization | TF-IDF + Linear SVM |        — |
| + stopword removal          | TF-IDF + Linear SVM |        — |
| + lemmatization             | TF-IDF + Linear SVM |        — |
| + negation handling         | TF-IDF + Linear SVM |        — |
| + bigrams                   | TF-IDF + Linear SVM |        — |

This allows the project to answer:

> **Which preprocessing decisions actually improve emotion classification?**

A preprocessing technique will not be considered beneficial simply because it is commonly used.

---

# 8. Model Evaluation

Models will be evaluated using more than accuracy.

Primary metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- **Macro-F1**
- per-class F1

Macro-F1 is particularly important because it evaluates all three emotion classes independently rather than allowing overall accuracy to hide weaknesses in an individual class.

The final test set will only be used after model selection is complete.

---

# 9. Error Analysis

Aggregate metrics are not sufficient to understand the problem.

The project will investigate:

- false positives
- false negatives
- confusion between anger and fear
- difficult joy examples
- ambiguous language
- mislabeled examples
- very short comments
- negation-related errors
- lexical ambiguity

The goal is to understand **why** the model fails.

This is particularly important because the initial EDA identified contradictory labels exclusively between anger and fear.

The analysis will therefore test whether the model independently exhibits the same ambiguity.

---

# 10. Classical NLP vs Transformer

After establishing a strong classical baseline, the project will introduce a transformer-based classifier using the **Hugging Face ecosystem**.

The transformer pipeline will operate on the original text rather than aggressively preprocessed tokens.

The progression will be:

```text
Original text
      ↓
Hugging Face tokenizer
      ↓
Pretrained transformer
      ↓
Fine-tuning
      ↓
Emotion classification
```

A lightweight pretrained model such as DistilBERT may be used initially, with a stronger model considered if justified by the results and available resources.

The purpose is to determine whether a transformer provides a meaningful improvement over the best classical NLP approach on this relatively small, short-text dataset.

---

# 11. Model Comparison

The final comparison will consider both performance and practical complexity.

| Model               | Representation   | Accuracy | Macro-F1 | Training Time | Inference |
| ------------------- | ---------------- | -------: | -------: | ------------: | --------: |
| Majority            | —                |        — |        — |             — |         — |
| Naive Bayes         | TF-IDF           |        — |        — |             — |         — |
| Logistic Regression | TF-IDF           |        — |        — |             — |         — |
| Linear SVM          | TF-IDF + n-grams |        — |        — |             — |         — |
| Transformer         | Raw text         |        — |        — |             — |         — |

The objective is not automatically to declare the transformer the winner.

If a classical model achieves similar performance at a fraction of the computational cost, that is itself an important result.

---

# 12. Explainability

Interpretability will be approached differently depending on the model.

For linear classical models, important features can be investigated through model coefficients to identify words and n-grams associated with each emotion.

For the transformer, the focus will initially be on qualitative error analysis and model predictions rather than building a large explainability framework.

The goal is to understand model behavior without adding unnecessary complexity.

---

# 13. Inference Pipeline

After selecting the final model, the project will provide a reusable prediction interface.

Conceptually:

```python
predict_emotion(
    "I am really nervous about tomorrow."
)
```

will return something similar to:

```text
Prediction: fear

Confidence:
fear     0.92
anger    0.05
joy      0.03
```

The exact implementation will depend on the final selected model.

---

# 14. Interactive Demo

As a final portfolio component, the project may include a lightweight interactive application allowing users to enter text and receive:

- predicted emotion
- class probabilities/confidence
- model used

The application will be implemented only after the underlying NLP pipeline and evaluation are stable.

---

# Repository Structure

The final project is intended to evolve toward:

```text
nlp-sentiment-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_error_analysis.ipynb
│   └── 07_transformer.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── inference.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── app/
│
├── requirements.txt
├── README.md
└── LICENSE
```

The structure may evolve as the project develops.

---

# Current Known Gaps

The current repository is **not yet a finished end-to-end NLP system**.

The remaining work includes:

- correcting and validating preprocessing decisions
- establishing classical baselines
- feature-engineering experiments
- preprocessing ablation
- hyperparameter tuning
- cross-validation
- final model selection
- near-duplicate analysis
- confusion-matrix analysis
- targeted error analysis
- transformer fine-tuning
- classical-vs-transformer comparison
- inference pipeline
- final documentation
- optional interactive demo

These gaps are intentional and represent the remaining stages of the project.

---

# Final Objective

The finished project should answer a broader question than simply:

> "Can a model classify these comments?"

Instead, it should answer:

> **How do preprocessing, feature representation, classical machine-learning algorithms, and transformer-based representations affect three-class emotion classification, and what kinds of linguistic ambiguity cause the models to fail?**

That comparison is the central theme of the project.
