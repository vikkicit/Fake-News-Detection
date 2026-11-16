"""
MODULE 2: Feature Extraction
=============================
Implements TF-IDF vectorization for the Fake News Detection pipeline.

Design principles
-----------------
- Vectorizer is NEVER fitted on test data (no data leakage).
- fit_vectorizer()   : call ONCE on X_train only.
- transform_data()   : call on any split (train / val / test).
- fit_transform_train() : convenience wrapper for training split.
- Sparse matrices are kept sparse throughout; never densified.
- Fully compatible with the downstream Logistic Regression pipeline.
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PROCESSED_CSV    = DATASET_DIR / "processed_dataset.csv"
VECTORIZER_PATH  = MODELS_DIR  / "tfidf_vectorizer.pkl"

# ── TF-IDF configuration ──────────────────────────────────────────────────────
TFIDF_CONFIG: dict = {
    "ngram_range"  : (1, 2),       # unigrams + bigrams
    "max_features" : 20_000,       # vocabulary cap
    "stop_words"   : "english",    # built-in English stop-word list
    "lowercase"    : True,         # normalise case
    "sublinear_tf" : True,         # apply 1 + log(tf) dampening
    "min_df"       : 2,            # ignore terms appearing < 2 docs
    "max_df"       : 0.95,         # ignore terms in > 95 % of docs
    "strip_accents": "unicode",    # normalise accented characters
    "analyzer"     : "word",       # word-level tokenisation
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",  # alpha-start tokens
}

# ── Train / Val / Test split ratios ──────────────────────────────────────────
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15   # of remaining after train split → 15 % of total
TEST_SIZE  = 0.15   # remainder


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorizer(**overrides) -> TfidfVectorizer:
    """
    Instantiate a fresh (unfitted) TfidfVectorizer with project defaults.

    Parameters
    ----------
    **overrides : Any TFIDF_CONFIG key to override, e.g. max_features=5000.

    Returns
    -------
    TfidfVectorizer (not yet fitted)
    """
    cfg = {**TFIDF_CONFIG, **overrides}
    log.info("Building TfidfVectorizer — config: %s", cfg)
    return TfidfVectorizer(**cfg)


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series | list[str],
) -> TfidfVectorizer:
    """
    Fit the vectorizer on TRAINING data only.

    ⚠  Never pass validation or test data here — that would leak test
       distribution into the vocabulary and IDF weights.

    Parameters
    ----------
    vectorizer : An unfitted TfidfVectorizer instance.
    X_train    : Training text corpus (Series or list of strings).

    Returns
    -------
    The fitted vectorizer (same object, mutated in-place).
    """
    if not isinstance(X_train, (pd.Series, list, np.ndarray)):
        raise TypeError(f"X_train must be Series/list/ndarray, got {type(X_train)}")
    if len(X_train) == 0:
        raise ValueError("X_train is empty — nothing to fit.")

    log.info("Fitting vectorizer on %d training documents …", len(X_train))
    vectorizer.fit(X_train)
    vocab_size = len(vectorizer.vocabulary_)
    log.info("Vocabulary size after fit: %d terms", vocab_size)
    return vectorizer


def transform_data(
    vectorizer: TfidfVectorizer,
    X: pd.Series | list[str],
    split_name: str = "data",
) -> spmatrix:
    """
    Transform text to a TF-IDF sparse matrix using an already-fitted vectorizer.

    Parameters
    ----------
    vectorizer : A *fitted* TfidfVectorizer.
    X          : Text corpus to transform (Series or list of strings).
    split_name : Label for logging ('train' / 'val' / 'test').

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_docs, vocab_size)
    """
    if not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError(
            "Vectorizer has not been fitted yet. "
            "Call fit_vectorizer(vectorizer, X_train) first."
        )
    if len(X) == 0:
        raise ValueError(f"Input corpus for '{split_name}' is empty.")

    log.info("Transforming %s  (%d docs) …", split_name, len(X))
    matrix = vectorizer.transform(X)
    log.info(
        "%s matrix shape: %s  |  dtype: %s  |  density: %.4f%%",
        split_name, matrix.shape, matrix.dtype,
        matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100,
    )
    return matrix


def fit_transform_train(
    X_train: pd.Series | list[str],
    **vectorizer_overrides,
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Convenience function: build → fit → transform in one call for training data.

    Parameters
    ----------
    X_train              : Training text corpus.
    **vectorizer_overrides : Optional TF-IDF config overrides.

    Returns
    -------
    (fitted_vectorizer, train_matrix)
    """
    vec = build_vectorizer(**vectorizer_overrides)
    fit_vectorizer(vec, X_train)
    matrix = transform_data(vec, X_train, split_name="train")
    return vec, matrix


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path | str = VECTORIZER_PATH) -> None:
    """Persist a fitted vectorizer to disk via pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Vectorizer saved → %s", path)


def load_vectorizer(path: Path | str = VECTORIZER_PATH) -> TfidfVectorizer:
    """Load a previously saved vectorizer from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved vectorizer found at: {path}")
    with open(path, "rb") as f:
        vec = pickle.load(f)
    log.info("Vectorizer loaded ← %s  (vocab: %d terms)", path, len(vec.vocabulary_))
    return vec


def get_top_features(
    vectorizer: TfidfVectorizer,
    matrix: spmatrix,
    n: int = 20,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the top-n terms by mean TF-IDF weight across a document set.

    Useful for inspecting which terms are most discriminative per class.

    Parameters
    ----------
    vectorizer : Fitted TfidfVectorizer.
    matrix     : TF-IDF sparse matrix for the document subset.
    n          : Number of top terms to return.
    label      : Optional class label string for the column name.

    Returns
    -------
    DataFrame with columns ['term', 'mean_tfidf'].
    """
    if not issparse(matrix):
        raise TypeError("matrix must be a scipy sparse matrix.")
    feature_names = vectorizer.get_feature_names_out()
    mean_scores   = np.asarray(matrix.mean(axis=0)).flatten()
    top_idx       = mean_scores.argsort()[::-1][:n]
    col           = f"mean_tfidf{'_' + label if label else ''}"
    return pd.DataFrame({
        "term": feature_names[top_idx],
        col   : mean_scores[top_idx],
    })


def prepare_splits(
    df: pd.DataFrame,
    text_col: str = "content",
    label_col: str = "label",
    train_size: float = TRAIN_SIZE,
    val_size:   float = VAL_SIZE,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a processed DataFrame into stratified train / val / test splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame.")
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found in DataFrame.")

    X = df[text_col].fillna("").astype(str)
    y = df[label_col].values

    # First cut: train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1 - train_size,
        random_state=random_state,
        stratify=y,
    )
    # Second cut: val vs test (equal halves of remainder)
    val_ratio = val_size / (val_size + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_ratio,
        random_state=random_state,
        stratify=y_tmp,
    )
    log.info(
        "Split → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (run directly: python src/feature_extraction.py)
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    SEP = "─" * 65

    def section(title: str) -> None:
        print(f"\n{SEP}\n  {title}\n{SEP}")

    # ── Load processed dataset ────────────────────────────────────────────────
    section("TEST 1 — Load processed_dataset.csv")
    if not PROCESSED_CSV.exists():
        print(f"  ✘ {PROCESSED_CSV} not found.  Run preprocessing.py first.")
        return
    df = pd.read_csv(PROCESSED_CSV)
    print(f"  ✔ Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")

    # ── Stratified split ─────────────────────────────────────────────────────
    section("TEST 2 — Stratified Train / Val / Test Split")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df)
    total = len(X_train) + len(X_val) + len(X_test)
    print(f"  {'Split':<8} {'Docs':>8}  {'%':>6}  {'Class-0':>8}  {'Class-1':>8}")
    print(f"  {'─'*8} {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        c0, c1 = (y == 0).sum(), (y == 1).sum()
        print(f"  {name:<8} {len(X):>8,}  {len(X)/total*100:>5.1f}%  {c0:>8,}  {c1:>8,}")

    # ── Leakage guard: confirm splits are disjoint ────────────────────────────
    section("TEST 3 — Data Leakage Guard (index disjointness)")
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    test_idx  = set(X_test.index)
    tv_overlap = train_idx & val_idx
    tt_overlap = train_idx & test_idx
    vt_overlap = val_idx  & test_idx
    all_ok = not tv_overlap and not tt_overlap and not vt_overlap
    print(f"\n  Train ∩ Val  overlap : {len(tv_overlap)}  {'✔' if not tv_overlap else '✘ LEAKAGE!'}")
    print(f"  Train ∩ Test overlap : {len(tt_overlap)}  {'✔' if not tt_overlap else '✘ LEAKAGE!'}")
    print(f"  Val   ∩ Test overlap : {len(vt_overlap)}  {'✔' if not vt_overlap else '✘ LEAKAGE!'}")
    print(f"\n  → {'No data leakage detected ✔' if all_ok else 'DATA LEAKAGE DETECTED ✘'}")

    # ── Fit vectorizer on training data ONLY ─────────────────────────────────
    section("TEST 4 — Fit TF-IDF on Training Data ONLY")
    vec, X_train_tfidf = fit_transform_train(X_train)
    print(f"\n  Vectorizer config:")
    for k, v in TFIDF_CONFIG.items():
        print(f"    {k:<18}: {v}")

    # ── Transform val and test with fitted vectorizer ─────────────────────────
    section("TEST 5 — Transform Val & Test (transform-only, no refit)")
    X_val_tfidf  = transform_data(vec, X_val,  split_name="val")
    X_test_tfidf = transform_data(vec, X_test, split_name="test")

    print(f"\n  {'Split':<8} {'Shape':>18}  {'Non-zeros':>12}  {'Density':>10}")
    print(f"  {'─'*8} {'─'*18}  {'─'*12}  {'─'*10}")
    for name, m in [("train", X_train_tfidf), ("val", X_val_tfidf), ("test", X_test_tfidf)]:
        density = m.nnz / (m.shape[0] * m.shape[1]) * 100
        print(f"  {name:<8} {str(m.shape):>18}  {m.nnz:>12,}  {density:>9.4f}%")

    # ── Top features per class ────────────────────────────────────────────────
    section("TEST 6 — Top 15 TF-IDF Terms per Class (training set)")
    for lbl, lbl_name in [(0, "FAKE (label=0)"), (1, "SYNTHETIC REAL (label=1)")]:
        mask    = y_train == lbl
        sub_mat = X_train_tfidf[mask]
        top_df  = get_top_features(vec, sub_mat, n=15, label=str(lbl))
        print(f"\n  ── {lbl_name}  ({mask.sum():,} docs) ──")
        col = f"mean_tfidf_{lbl}"
        for _, row in top_df.iterrows():
            bar = "█" * int(row[col] * 400)
            print(f"    {row['term']:<30}  {row[col]:.6f}  {bar}")

    # ── Vocabulary consistency check ──────────────────────────────────────────
    section("TEST 7 — Vocabulary Consistency")
    vocab_size = len(vec.vocabulary_)
    print(f"\n  Vocabulary size     : {vocab_size:,}")
    print(f"  Max configured      : {TFIDF_CONFIG['max_features']:,}")
    print(f"  Val  feature cols   : {X_val_tfidf.shape[1]:,}")
    print(f"  Test feature cols   : {X_test_tfidf.shape[1]:,}")
    consistent = (X_train_tfidf.shape[1] == X_val_tfidf.shape[1] == X_test_tfidf.shape[1])
    print(f"\n  All splits share same feature space: {'✔' if consistent else '✘'}")

    # ── Save fitted vectorizer ────────────────────────────────────────────────
    section("TEST 8 — Save & Reload Vectorizer")
    save_vectorizer(vec)
    vec2 = load_vectorizer()
    reload_ok = len(vec2.vocabulary_) == len(vec.vocabulary_)
    print(f"\n  Original vocab size  : {len(vec.vocabulary_):,}")
    print(f"  Reloaded vocab size  : {len(vec2.vocabulary_):,}")
    print(f"  Reload integrity     : {'✔ PASS' if reload_ok else '✘ FAIL'}")

    # ── Final summary ─────────────────────────────────────────────────────────
    section("FEATURE EXTRACTION — TEST SUMMARY")
    results = {
        "CSV loaded"              : True,
        "Stratified split"        : True,
        "No data leakage"         : all_ok,
        "Vectorizer fitted"       : hasattr(vec, "vocabulary_"),
        "Val/Test transform-only" : True,
        "Feature space consistent": consistent,
        "Vectorizer persisted"    : reload_ok,
    }
    print()
    for check, passed in results.items():
        status = "✔ PASS" if passed else "✘ FAIL"
        print(f"  [{status}]  {check}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED ✔' if all_passed else 'SOME TESTS FAILED ✘'}")
    print(f"\n  Vectorizer path  : {VECTORIZER_PATH}")
    print(f"  Ready for MODULE 3 — Model Training (Logistic Regression)\n")


if __name__ == "__main__":
    _run_tests()
