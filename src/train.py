"""
MODULE 3: Model Training and Evaluation
=======================================
Trains and evaluates a Logistic Regression classifier on the TF-IDF features
generated from the processed fake news dataset.

Architectural highlights:
- Strict split reproducibility: Stratified 70/15/15 split (seed=42).
- Zero data leakage: Reuses fitted TF-IDF vectorizer (or fits strictly on X_train).
- Hyperparameter tuning: Evaluates regularization strength C across [0.01, 0.1, 1.0, 10.0, 100.0]
  on Validation F1-score only.
- Final evaluation: Evaluated ONCE on untouched Test set.
- Honest appraisal: Documents dataset bias (AI-generated synthetic text vs fake news corpus).
- Reports: Saves metrics.json, confusion_matrix.png, classification_report.txt, and fake_news_model.pkl.
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
import logging
from pathlib import Path

# Configure matplotlib cache directory to avoid permissions warnings on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# Import module components
from feature_extraction import (
    prepare_splits,
    load_vectorizer,
    transform_data,
    fit_transform_train,
    save_vectorizer,
    PROCESSED_CSV,
    VECTORIZER_PATH,
    MODELS_DIR,
    BASE_DIR,
)

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "fake_news_model.pkl"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# ── Model Configuration Space ─────────────────────────────────────────────────
PARAM_GRID_C = [0.01, 0.1, 1.0, 10.0, 100.0]
BASE_CONFIG = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "liblinear",
}


def section(title: str) -> None:
    sep = "─" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


def train_and_evaluate():
    section("STEP 1: Load Data & Reproduce Stratified Split")
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Missing {PROCESSED_CSV}. Run preprocessing.py first.")

    df = pd.read_csv(PROCESSED_CSV)
    log.info("Loaded processed dataset: %d rows", len(df))

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
        df, text_col="content", label_col="label", train_size=0.70, val_size=0.15, random_state=42
    )

    log.info("Split verification -> Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    section("STEP 2: Load / Verify Fitted TF-IDF Vectorizer")
    if VECTORIZER_PATH.exists():
        vec = load_vectorizer(VECTORIZER_PATH)
        log.info("Reusing existing fitted vectorizer (vocab size: %d)", len(vec.vocabulary_))
    else:
        log.info("Existing vectorizer not found, fitting strictly on X_train...")
        vec, _ = fit_transform_train(X_train)
        save_vectorizer(vec, VECTORIZER_PATH)

    section("STEP 3: Transform Train, Val, and Test Splits (Zero Leakage)")
    X_train_vec = transform_data(vec, X_train, split_name="train")
    X_val_vec   = transform_data(vec, X_val,   split_name="val")
    X_test_vec  = transform_data(vec, X_test,  split_name="test")

    section("STEP 4: Hyperparameter Tuning on Validation Set")
    print(f"  Searching over regularization C: {PARAM_GRID_C}")
    print(f"  {'C':<8} {'Train F1':>10} {'Val F1':>10} {'Val Acc':>10} {'Fit Time (s)':>14}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*14}")

    best_c = None
    best_val_f1 = -1.0
    best_model = None
    tuning_records = []

    for c in PARAM_GRID_C:
        cfg = {**BASE_CONFIG, "C": c}
        clf = LogisticRegression(**cfg)
        
        t0 = time.perf_counter()
        clf.fit(X_train_vec, y_train)
        fit_time = time.perf_counter() - t0

        train_pred = clf.predict(X_train_vec)
        val_pred = clf.predict(X_val_vec)

        train_f1 = f1_score(y_train, train_pred)
        val_f1   = f1_score(y_val, val_pred)
        val_acc  = accuracy_score(y_val, val_pred)

        tuning_records.append({
            "C": c,
            "train_f1": float(train_f1),
            "val_f1": float(val_f1),
            "val_acc": float(val_acc),
            "fit_time_seconds": round(fit_time, 4),
        })

        print(f"  {c:<8} {train_f1:>10.4f} {val_f1:>10.4f} {val_acc:>10.4f} {fit_time:>14.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c
            best_model = clf

    print(f"\n  Selected Optimal Model: C = {best_c} (Validation F1 = {best_val_f1:.4f})")

    section("STEP 5: Final Evaluation on Untouched Test Set")
    t0_train = time.perf_counter()
    # Retrain on full train set with optimal C
    final_model = LogisticRegression(**{**BASE_CONFIG, "C": best_c})
    final_model.fit(X_train_vec, y_train)
    total_train_time = time.perf_counter() - t0_train

    # Measure inference / prediction time
    t0_pred = time.perf_counter()
    y_test_pred = final_model.predict(X_test_vec)
    y_test_proba = final_model.predict_proba(X_test_vec)[:, 1]
    total_pred_time = time.perf_counter() - t0_pred
    per_sample_pred_time_ms = (total_pred_time / len(X_test)) * 1000

    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)

    target_names = ["Class 0: Fake News", "Class 1: Synthetic"]
    report_str = classification_report(y_test, y_test_pred, target_names=target_names, digits=4)

    print("\n  TEST SET METRICS:")
    print(f"    Accuracy         : {acc:.4%}")
    print(f"    Precision        : {prec:.4%}")
    print(f"    Recall           : {rec:.4%}")
    print(f"    F1-Score         : {f1:.4%}")
    print(f"    ROC-AUC Score    : {auc:.4f}")
    print(f"    Training Time    : {total_train_time:.3f} s")
    print(f"    Inference Time   : {total_pred_time:.4f} s total ({per_sample_pred_time_ms:.3f} ms / sample)")
    print("\n  CLASSIFICATION REPORT:")
    print(report_str)
    print("\n  CONFUSION MATRIX (Rows: True, Cols: Pred):")
    print(f"    TN: {cm[0, 0]:>5} (Fake as Fake)       | FP: {cm[0, 1]:>5} (Fake as Synthetic)")
    print(f"    FN: {cm[1, 0]:>5} (Synthetic as Fake)  | TP: {cm[1, 1]:>5} (Synthetic as Synthetic)")

    section("STEP 6: Model Interpretability & Bias Audit")
    feature_names = np.array(vec.get_feature_names_out())
    weights = final_model.coef_[0]

    top_synthetic_idx = weights.argsort()[::-1][:15]
    top_fake_idx = weights.argsort()[:15]

    print("\n  Top 15 Features Indicative of Class 1 (Synthetic):")
    for idx in top_synthetic_idx:
        print(f"    + {feature_names[idx]:<25} weight = {weights[idx]:.4f}")

    print("\n  Top 15 Features Indicative of Class 0 (Fake News):")
    for idx in top_fake_idx:
        print(f"    - {feature_names[idx]:<25} weight = {weights[idx]:.4f}")

    section("STEP 7: Save Model Artifacts and Reports")
    # Save Model
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved final model -> %s", MODEL_OUTPUT_PATH)

    # Save Classification Report
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== LOGISTIC REGRESSION FAKE NEWS / SYNTHETIC CLASSIFIER ===\n\n")
        f.write(f"Hyperparameters: {final_model.get_params()}\n\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0, 0]} | FP: {cm[0, 1]}\n")
        f.write(f"FN: {cm[1, 0]} | TP: {cm[1, 1]}\n")
        f.write("\n\n--- CRITICAL DATASET LIMITATION NOTICE ---\n")
        f.write("The dataset consists of real scraped fake news (Fake.csv) and AI-generated synthetic news (True.csv).\n")
        f.write("High test accuracy reflects distinguishing AI generated prose from sensational fake news,\n")
        f.write("NOT generalizable real-world truth verification.\n")
    log.info("Saved classification report -> %s", CLASSIFICATION_REPORT_PATH)

    # Save Metrics JSON
    metrics_payload = {
        "model_type": "LogisticRegression",
        "best_hyperparameters": {"C": best_c, **BASE_CONFIG},
        "tuning_history": tuning_records,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "training_time_seconds": round(total_train_time, 4),
            "prediction_time_seconds": round(total_pred_time, 4),
            "latency_ms_per_sample": round(per_sample_pred_time_ms, 4),
        },
        "dataset_metadata": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "label_0_name": "Fake News (scraped)",
            "label_1_name": "Synthetic (AI-generated)",
        },
        "dataset_limitations": (
            "True.csv contains AI-generated synthetic articles rather than independently verified real-world news. "
            "High test metrics demonstrate successful classification between synthetic writing style and fake news corpus, "
            "not a production-ready real-world truth verifier."
        ),
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("Saved metrics JSON -> %s", METRICS_JSON_PATH)

    # Generate and Save Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size="large", weight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Fake (0)", "Pred: Synthetic (1)"])
    ax.set_yticklabels(["True: Fake (0)", "True: Synthetic (1)"])
    plt.title("Confusion Matrix - Test Split", pad=16, weight="bold")
    plt.xlabel("Predicted Label", labelpad=8)
    plt.ylabel("True Label", labelpad=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    log.info("Saved confusion matrix image -> %s", CONFUSION_MATRIX_PATH)

    section("STEP 8: Verification & Sanity Checks")
    # 1. Model reload verification
    with open(MODEL_OUTPUT_PATH, "rb") as f:
        reloaded_model = pickle.load(f)
    sample_preds = reloaded_model.predict(X_test_vec[:5])
    sample_probs = reloaded_model.predict_proba(X_test_vec[:5])
    assert np.array_equal(sample_preds, y_test_pred[:5]), "Reloaded model output mismatch!"
    print("  ✔ Model serialization & loading verified.")
    print("  ✔ Sample unseen test predictions:", sample_preds)
    print("  ✔ Probability outputs for unseen test samples:")
    for idx, (p0, p1) in enumerate(sample_probs):
        pred_label = "Synthetic (1)" if p1 >= 0.5 else "Fake (0)"
        print(f"      Sample {idx}: P(Fake)={p0:.4f}, P(Synthetic)={p1:.4f} -> Pred: {pred_label}")

    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
