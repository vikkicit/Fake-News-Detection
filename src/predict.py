"""
MODULE 4: Prediction Engine
============================
Provides a reusable `predict_news(text)` function that loads the trained
Logistic Regression model and fitted TF-IDF vectorizer, then returns a
structured prediction result.

IMPORTANT DISCLAIMER
--------------------
This model was trained on two datasets:
  - Fake.csv : Scraped fake / misleading news articles  (label 0)
  - True.csv : AI-generated synthetic articles           (label 1)

The classifier distinguishes **writing style differences between these two
corpora**.  It does NOT independently verify factual truth and should NOT
be treated as a production-grade fact-checker.  High confidence scores
reflect stylistic separation, not journalistic verification.
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
MODELS_DIR      = BASE_DIR / "models"
MODEL_PATH      = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 20
LABEL_MAP = {0: "Fake", 1: "Synthetic"}

DISCLAIMER = (
    "This model classifies text based on writing-style differences between "
    "a scraped fake-news corpus (label 0) and AI-generated synthetic articles "
    "(label 1). It does NOT verify factual accuracy. Results should be "
    "interpreted as stylistic analysis, not truth verification."
)

# ── Cached model / vectorizer singletons ──────────────────────────────────────
_model = None
_vectorizer = None


class PredictionError(Exception):
    """Raised when the prediction pipeline encounters an unrecoverable error."""
    pass


def _load_artifact(path: Path, name: str):
    """Load a pickled artifact with clear error messaging."""
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}. "
            f"Run src/train.py to generate model artifacts first."
        )
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        log.info("Loaded %s from %s", name, path)
        return artifact
    except Exception as e:
        raise PredictionError(f"Failed to load {name} from {path}: {e}") from e


def load_model(force_reload: bool = False):
    """
    Load the trained Logistic Regression model (cached singleton).

    Parameters
    ----------
    force_reload : bool
        If True, reload from disk even if already cached.

    Returns
    -------
    The trained sklearn LogisticRegression model.
    """
    global _model
    if _model is None or force_reload:
        _model = _load_artifact(MODEL_PATH, "Logistic Regression model")
    return _model


def load_vectorizer(force_reload: bool = False):
    """
    Load the fitted TF-IDF vectorizer (cached singleton).

    Parameters
    ----------
    force_reload : bool
        If True, reload from disk even if already cached.

    Returns
    -------
    The fitted sklearn TfidfVectorizer.
    """
    global _vectorizer
    if _vectorizer is None or force_reload:
        _vectorizer = _load_artifact(VECTORIZER_PATH, "TF-IDF vectorizer")
    return _vectorizer


def predict_news(text: str) -> Dict[str, Any]:
    """
    Classify a news article as Fake (0) or Synthetic (1).

    Parameters
    ----------
    text : str
        The article text to classify.  Must be a non-empty string of at
        least 20 characters after stripping whitespace.

    Returns
    -------
    dict with keys:
        prediction           : int   — 0 (Fake) or 1 (Synthetic)
        label                : str   — "Fake" or "Synthetic"
        confidence           : float — highest class probability (0–1)
        fake_probability     : float — P(class=0)
        synthetic_probability: float — P(class=1)
        warning              : str   — disclaimer about model limitations

    Raises / returns error dict
    ---------------------------
    If the input is invalid (empty, too short, wrong type) the function
    returns a dict with prediction=-1, label="Error", and an explanatory
    warning — it does NOT raise, so callers can handle errors uniformly.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(text, str):
        return _error_result(f"Input must be a string, got {type(text).__name__}.")

    cleaned = text.strip()

    if not cleaned:
        return _error_result("Input text is empty. Please provide a news article.")

    if len(cleaned) < MIN_TEXT_LENGTH:
        return _error_result(
            f"Input text is too short ({len(cleaned)} chars). "
            f"Minimum required: {MIN_TEXT_LENGTH} characters."
        )

    # ── Load artifacts ────────────────────────────────────────────────────
    try:
        model = load_model()
        vectorizer = load_vectorizer()
    except (FileNotFoundError, PredictionError) as e:
        return _error_result(f"Model loading failed: {e}")

    # ── Transform & predict ───────────────────────────────────────────────
    try:
        text_vectorized = vectorizer.transform([cleaned])
        prediction = int(model.predict(text_vectorized)[0])
        probabilities = model.predict_proba(text_vectorized)[0]

        fake_prob = float(probabilities[0])
        synthetic_prob = float(probabilities[1])
        confidence = float(max(fake_prob, synthetic_prob))

        return {
            "prediction": prediction,
            "label": LABEL_MAP.get(prediction, "Unknown"),
            "confidence": round(confidence, 6),
            "fake_probability": round(fake_prob, 6),
            "synthetic_probability": round(synthetic_prob, 6),
            "warning": DISCLAIMER,
        }
    except Exception as e:
        return _error_result(f"Prediction failed: {e}")


def _error_result(message: str) -> Dict[str, Any]:
    """Build a standardised error-result dictionary."""
    log.warning("Prediction error: %s", message)
    return {
        "prediction": -1,
        "label": "Error",
        "confidence": 0.0,
        "fake_probability": 0.0,
        "synthetic_probability": 0.0,
        "warning": message,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python src/predict.py)
# ══════════════════════════════════════════════════════════════════════════════
def _run_demo():
    SEP = "─" * 70

    def section(title):
        print(f"\n{SEP}\n  {title}\n{SEP}")

    section("PREDICTION ENGINE DEMO")

    samples = [
        (
            "Fake-style",
            "Trump supporters spread a viral video claiming massive election "
            "fraud, but the footage was from a completely different country and "
            "had nothing to do with the US election.",
        ),
        (
            "Synthetic-style",
            "The municipal council published its quarterly service update in "
            "Northfield, covering road repairs and drainage maintenance. "
            "Residents can review the schedule at the local office.",
        ),
        ("Empty", ""),
        ("Too short", "hello world"),
        ("Non-string", None),
    ]

    for label, text in samples:
        print(f"\n  ── {label} ──")
        if text is not None:
            preview = text[:80] + ("…" if len(text) > 80 else "")
            print(f"  Input : \"{preview}\"")
        else:
            print(f"  Input : None")
        result = predict_news(text) if isinstance(text, str) else predict_news(text)
        for k, v in result.items():
            if k == "warning":
                v_str = v[:90] + "…" if len(str(v)) > 90 else v
                print(f"    {k:<25}: {v_str}")
            else:
                print(f"    {k:<25}: {v}")

    print(f"\n{SEP}")
    print("  Prediction engine is working correctly.")
    print(SEP)


if __name__ == "__main__":
    _run_demo()
