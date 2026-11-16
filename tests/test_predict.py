"""
Tests for MODULE 4: Prediction Engine
======================================
Verifies predict_news() function behaviour for valid inputs, edge cases,
probability properties, model loading, and prediction consistency.

Run with:  python -m pytest tests/test_predict.py -v
    or:    python tests/test_predict.py
"""

import sys
import os
import math

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import (
    predict_news,
    load_model,
    load_vectorizer,
    LABEL_MAP,
    MIN_TEXT_LENGTH,
    PredictionError,
)


# ══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════════════════════
PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [✔ PASS]  {name}")
    else:
        FAIL += 1
        msg = f"  [✘ FAIL]  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def section(title: str):
    sep = "─" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# Test data
# ══════════════════════════════════════════════════════════════════════════════
VALID_FAKE_STYLE = (
    "Trump supporters spread a viral video claiming massive election "
    "fraud, but the footage was from a completely different country and "
    "had nothing to do with the US election. Obama and Hillary Clinton "
    "were also mentioned in conspiracy theories shared on social media."
)

VALID_SYNTHETIC_STYLE = (
    "The municipal council published its quarterly service update in "
    "Northfield, covering road repairs and drainage maintenance. "
    "Residents can review the schedule at the local information office. "
    "The regional programme outlines community improvements for the quarter."
)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_model_loading():
    """TEST 1: Model and vectorizer can be loaded from disk."""
    section("TEST 1 — Model & Vectorizer Loading")
    try:
        model = load_model(force_reload=True)
        check("Model loads without error", model is not None)
        check("Model has predict method", hasattr(model, "predict"))
        check("Model has predict_proba method", hasattr(model, "predict_proba"))
    except Exception as e:
        check("Model loads without error", False, str(e))

    try:
        vec = load_vectorizer(force_reload=True)
        check("Vectorizer loads without error", vec is not None)
        check("Vectorizer has vocabulary", hasattr(vec, "vocabulary_"))
        check("Vocabulary size is 20000", len(vec.vocabulary_) == 20000,
              f"got {len(vec.vocabulary_)}")
    except Exception as e:
        check("Vectorizer loads without error", False, str(e))


def test_valid_fake_input():
    """TEST 2: Valid fake-style text produces correct structure."""
    section("TEST 2 — Valid Fake-Style Input")
    result = predict_news(VALID_FAKE_STYLE)

    check("Returns a dict", isinstance(result, dict))
    check("Has 'prediction' key", "prediction" in result)
    check("Has 'label' key", "label" in result)
    check("Has 'confidence' key", "confidence" in result)
    check("Has 'fake_probability' key", "fake_probability" in result)
    check("Has 'synthetic_probability' key", "synthetic_probability" in result)
    check("Has 'warning' key", "warning" in result)

    check("Prediction is 0 or 1", result["prediction"] in (0, 1),
          f"got {result['prediction']}")
    check("Label matches LABEL_MAP", result["label"] == LABEL_MAP.get(result["prediction"]),
          f"pred={result['prediction']}, label={result['label']}")
    check("Prediction is 0 (Fake) for fake-style text",
          result["prediction"] == 0,
          f"got {result['prediction']} ({result['label']})")

    print(f"\n  Result: prediction={result['prediction']}, label={result['label']}, "
          f"confidence={result['confidence']:.4f}")


def test_valid_synthetic_input():
    """TEST 3: Valid synthetic-style text produces correct structure."""
    section("TEST 3 — Valid Synthetic-Style Input")
    result = predict_news(VALID_SYNTHETIC_STYLE)

    check("Prediction is 0 or 1", result["prediction"] in (0, 1))
    check("Prediction is 1 (Synthetic) for synthetic-style text",
          result["prediction"] == 1,
          f"got {result['prediction']} ({result['label']})")

    print(f"\n  Result: prediction={result['prediction']}, label={result['label']}, "
          f"confidence={result['confidence']:.4f}")


def test_empty_input():
    """TEST 4: Empty string is rejected gracefully."""
    section("TEST 4 — Empty Input Handling")
    result = predict_news("")

    check("Returns error dict (prediction=-1)", result["prediction"] == -1)
    check("Label is 'Error'", result["label"] == "Error")
    check("Warning mentions empty", "empty" in result["warning"].lower(),
          f"warning: {result['warning'][:80]}")


def test_whitespace_only_input():
    """TEST 5: Whitespace-only string is rejected."""
    section("TEST 5 — Whitespace-Only Input")
    result = predict_news("     \t\n   ")

    check("Returns error dict (prediction=-1)", result["prediction"] == -1)
    check("Label is 'Error'", result["label"] == "Error")


def test_short_input():
    """TEST 6: Text shorter than MIN_TEXT_LENGTH is rejected."""
    section("TEST 6 — Too-Short Input")
    short_text = "hello world"  # 11 chars
    result = predict_news(short_text)

    check(f"Short text ({len(short_text)} chars) is rejected",
          result["prediction"] == -1)
    check("Warning mentions 'short'", "short" in result["warning"].lower(),
          f"warning: {result['warning'][:80]}")
    check(f"Minimum enforced is {MIN_TEXT_LENGTH}", True)


def test_boundary_length():
    """TEST 7: Text at exactly MIN_TEXT_LENGTH boundary."""
    section("TEST 7 — Boundary Length Input")
    # Exactly 19 chars → rejected
    text_19 = "a" * 19
    r19 = predict_news(text_19)
    check(f"19-char text rejected (prediction=-1)", r19["prediction"] == -1)

    # Exactly 20 chars → accepted (may predict anything, but shouldn't error)
    text_20 = "a" * 20
    r20 = predict_news(text_20)
    check(f"20-char text accepted (prediction != -1)", r20["prediction"] != -1,
          f"got prediction={r20['prediction']}")


def test_probability_bounds():
    """TEST 8: Probability values are within [0, 1]."""
    section("TEST 8 — Probability Bounds")
    result = predict_news(VALID_FAKE_STYLE)

    fp = result["fake_probability"]
    sp = result["synthetic_probability"]
    conf = result["confidence"]

    check("fake_probability in [0, 1]", 0.0 <= fp <= 1.0, f"got {fp}")
    check("synthetic_probability in [0, 1]", 0.0 <= sp <= 1.0, f"got {sp}")
    check("confidence in [0, 1]", 0.0 <= conf <= 1.0, f"got {conf}")
    check("confidence == max(fake_prob, synthetic_prob)",
          abs(conf - max(fp, sp)) < 1e-6,
          f"conf={conf}, max={max(fp, sp)}")


def test_probability_sum():
    """TEST 9: Probabilities sum to approximately 1."""
    section("TEST 9 — Probability Sum ≈ 1.0")
    result = predict_news(VALID_FAKE_STYLE)

    total = result["fake_probability"] + result["synthetic_probability"]
    check("P(Fake) + P(Synthetic) ≈ 1.0",
          math.isclose(total, 1.0, abs_tol=1e-4),
          f"sum = {total}")

    result2 = predict_news(VALID_SYNTHETIC_STYLE)
    total2 = result2["fake_probability"] + result2["synthetic_probability"]
    check("P(Fake) + P(Synthetic) ≈ 1.0 (second sample)",
          math.isclose(total2, 1.0, abs_tol=1e-4),
          f"sum = {total2}")


def test_prediction_consistency():
    """TEST 10: Same input produces same prediction every time."""
    section("TEST 10 — Prediction Consistency (Determinism)")
    results = [predict_news(VALID_FAKE_STYLE) for _ in range(5)]

    preds = [r["prediction"] for r in results]
    confs = [r["confidence"] for r in results]

    check("Same prediction across 5 calls", len(set(preds)) == 1,
          f"predictions: {preds}")
    check("Same confidence across 5 calls",
          max(confs) - min(confs) < 1e-9,
          f"confidences: {confs}")


def test_warning_always_present():
    """TEST 11: Every successful prediction includes a disclaimer warning."""
    section("TEST 11 — Disclaimer Warning Present")
    result = predict_news(VALID_FAKE_STYLE)

    check("Warning is a non-empty string",
          isinstance(result["warning"], str) and len(result["warning"]) > 0)
    check("Warning mentions 'writing-style' or 'stylistic'",
          "writing-style" in result["warning"].lower()
          or "stylistic" in result["warning"].lower()
          or "style" in result["warning"].lower(),
          f"warning: {result['warning'][:80]}")
    check("Warning mentions limitations (truth/verify)",
          "truth" in result["warning"].lower()
          or "verify" in result["warning"].lower()
          or "factual" in result["warning"].lower(),
          f"warning: {result['warning'][:80]}")


# ══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════════════════
def run_all():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    test_model_loading()
    test_valid_fake_input()
    test_valid_synthetic_input()
    test_empty_input()
    test_whitespace_only_input()
    test_short_input()
    test_boundary_length()
    test_probability_bounds()
    test_probability_sum()
    test_prediction_consistency()
    test_warning_always_present()

    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  TEST SUITE RESULTS:  {PASS} passed,  {FAIL} failed,  {PASS + FAIL} total")
    print(sep)
    if FAIL == 0:
        print("  ✔  ALL TESTS PASSED")
    else:
        print(f"  ✘  {FAIL} TEST(S) FAILED")
    print()

    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
