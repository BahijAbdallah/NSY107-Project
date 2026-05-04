"""
evaluate_models.py
Loads saved models and evaluates them on the test set.

Metrics computed for each model:
  accuracy, precision, recall, F1-score, ROC-AUC, false positive rate

Best model selection criteria:
  Strong F1-score + high recall + reasonable false positive rate
  (NOT accuracy alone — a model predicting all-normal can get high accuracy
   while missing every attack)

Outputs:
  results/model_comparison.csv    — metrics table
  results/best_model_summary.txt  — winner + reasoning
  results/best_model_name.txt     — machine-readable best model name
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Paths ──
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data():
    """Load supervised and unsupervised test sets, plus supervised train set."""
    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))

    X_te_sup   = read("X_test_sup.csv").values
    y_te_sup   = read("y_test_sup.csv").values.ravel()
    X_te_unsup = read("X_test_unsup.csv").values
    y_te_unsup = read("y_test_unsup.csv").values.ravel()

    # Training data — used only to print the overfitting gap for RF
    X_tr_sup = read("X_train_sup.csv").values
    y_tr_sup = read("y_train_sup.csv").values.ravel()

    return X_te_sup, y_te_sup, X_te_unsup, y_te_unsup, X_tr_sup, y_tr_sup


def compute_metrics(name, y_true, y_pred, y_scores=None):
    """Compute classification metrics and return as a dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    auc = float("nan")
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            pass

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "    ROC-AUC   : N/A")
    print(f"    FPR       : {fpr:.4f}")
    print(f"    Confusion : TN={tn} FP={fp} FN={fn} TP={tp}")

    return {
        "model":     name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(auc,  4) if not np.isnan(auc) else "N/A",
        "fpr":       round(fpr,  4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def eval_random_forest(X_test, y_test, X_train=None, y_train=None):
    path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(path):
        print("  [SKIP] random_forest.joblib not found")
        return None
    rf = joblib.load(path)

    # ── Overfitting gap: train F1 vs test F1 ──────────────────────────────────
    if X_train is not None and y_train is not None:
        y_tr_pred  = rf.predict(X_train)
        train_f1   = f1_score(y_train, y_tr_pred, zero_division=0)
        y_te_pred  = rf.predict(X_test)
        test_f1    = f1_score(y_test,  y_te_pred,  zero_division=0)
        gap        = train_f1 - test_f1
        flag       = "OK" if gap <= 0.10 else "OVERFITTING WARNING"
        print(f"\n  [Random Forest — Overfitting Check]")
        print(f"    Train F1 : {train_f1:.4f}")
        print(f"    Test  F1 : {test_f1:.4f}")
        print(f"    Gap      : {gap:.4f}  [{flag}]")

    y_pred   = rf.predict(X_test)
    y_scores = rf.predict_proba(X_test)[:, 1]
    return compute_metrics("Random Forest", y_test, y_pred, y_scores)


def eval_gradient_boosting(X_test, y_test, X_train=None, y_train=None):
    path = os.path.join(MODELS_DIR, "gradient_boosting.joblib")
    if not os.path.exists(path):
        print("  [SKIP] gradient_boosting.joblib not found")
        return None
    hgb = joblib.load(path)

    if X_train is not None and y_train is not None:
        y_tr_pred = hgb.predict(X_train)
        train_f1  = f1_score(y_train, y_tr_pred, zero_division=0)
        y_te_pred = hgb.predict(X_test)
        test_f1   = f1_score(y_test,  y_te_pred, zero_division=0)
        gap       = train_f1 - test_f1
        flag      = "OK" if gap <= 0.10 else "OVERFITTING WARNING"
        print(f"\n  [Hist GradientBoosting — Overfitting Check]")
        print(f"    Train F1 : {train_f1:.4f}")
        print(f"    Test  F1 : {test_f1:.4f}")
        print(f"    Gap      : {gap:.4f}  [{flag}]")

    y_pred   = hgb.predict(X_test)
    y_scores = hgb.predict_proba(X_test)[:, 1]
    return compute_metrics("Hist GradientBoosting", y_test, y_pred, y_scores)


def eval_isolation_forest(X_test, y_test):
    path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if not os.path.exists(path):
        print("  [SKIP] isolation_forest.joblib not found")
        return None
    iso = joblib.load(path)

    # IsolationForest: predict() returns +1 (normal) / -1 (anomaly)
    raw      = iso.predict(X_test)
    y_pred   = np.where(raw == -1, 1, 0)           # convert to 0/1
    y_scores = -iso.score_samples(X_test)           # higher = more anomalous
    return compute_metrics("Isolation Forest", y_test, y_pred, y_scores)


def eval_autoencoder(X_test, y_test):
    ae_path  = os.path.join(MODELS_DIR, "autoencoder.keras")
    thr_path = os.path.join(MODELS_DIR, "autoencoder_threshold.json")

    if not os.path.exists(ae_path):
        print("  [SKIP] autoencoder.keras not found")
        return None

    try:
        import tensorflow as tf
        ae = tf.keras.models.load_model(ae_path)
    except ImportError:
        print("  [SKIP] tensorflow not installed — autoencoder evaluation skipped")
        return None

    with open(thr_path) as f:
        threshold = json.load(f)["threshold"]

    recon    = ae.predict(X_test, verbose=0)
    errors   = np.mean((X_test - recon) ** 2, axis=1)
    y_pred   = (errors > threshold).astype(int)
    y_scores = errors
    return compute_metrics("Autoencoder", y_test, y_pred, y_scores)


def select_best(results):
    """
    Best model = highest F1 with recall >= 0.60 and FPR <= 0.30.
    Falls back to highest F1 if no model meets all thresholds.
    """
    valid = [r for r in results if r["recall"] >= 0.60 and r["fpr"] <= 0.30]
    pool  = valid if valid else results
    return max(pool, key=lambda r: r["f1"])


if __name__ == "__main__":
    print("Evaluating models on test data...")
    X_te_sup, y_te_sup, X_te_unsup, y_te_unsup, X_tr_sup, y_tr_sup = load_test_data()

    results = []

    r = eval_random_forest(X_te_sup, y_te_sup, X_tr_sup, y_tr_sup)
    if r: results.append(r)

    r = eval_gradient_boosting(X_te_sup, y_te_sup, X_tr_sup, y_tr_sup)
    if r: results.append(r)

    r = eval_isolation_forest(X_te_unsup, y_te_unsup)
    if r: results.append(r)

    r = eval_autoencoder(X_te_unsup, y_te_unsup)
    if r: results.append(r)

    if not results:
        print("\n[ERROR] No models were evaluated. Train models first.")
        exit(1)

    # Save comparison table
    df_cmp = pd.DataFrame(results)
    cmp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_cmp.to_csv(cmp_path, index=False)
    print(f"\n  Comparison table -> {cmp_path}")
    print(df_cmp[["model","accuracy","precision","recall","f1","roc_auc","fpr"]].to_string(index=False))

    # Identify and save best model
    best = select_best(results)
    print(f"\n  Best model: {best['model']}  (F1={best['f1']}, Recall={best['recall']}, FPR={best['fpr']})")

    summary_path = os.path.join(RESULTS_DIR, "best_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("BEST MODEL SELECTION\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Winner: {best['model']}\n\n")
        f.write("Selection criteria: highest F1 with Recall >= 0.60 and FPR <= 0.30\n\n")
        f.write("Metrics:\n")
        for k, v in best.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nAll model results:\n")
        f.write(df_cmp.to_string(index=False))

    name_path = os.path.join(RESULTS_DIR, "best_model_name.txt")
    with open(name_path, "w") as f:
        f.write(best["model"])

    print(f"  Summary        -> {summary_path}")
    print(f"  Best model name -> {name_path}")
    print("\n[DONE] Evaluation complete.")
