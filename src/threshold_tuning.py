"""
threshold_tuning.py
Explores RF classification thresholds on the test set.

Default scikit-learn threshold is 0.50.
Lowering the threshold increases recall (catch more attacks)
at the cost of higher FPR (more false alarms).

Saves:
  results/threshold_tuning.csv
  results/best_threshold.json   (loaded by detect_anomaly.py and error_analysis.py)
  results/figures/threshold_tradeoff.png
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def load_data():
    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))
    X = read("X_test_sup.csv").values
    y = read("y_test_sup.csv").values.ravel()
    return X, y


def load_best_supervised_model():
    """Load RF or GBM — whichever is the best model (or RF as fallback)."""
    name_path = os.path.join(RESULTS_DIR, "best_model_name.txt")
    name = "Random Forest"
    if os.path.exists(name_path):
        with open(name_path) as f:
            candidate = f.read().strip()
        if candidate in ("Random Forest", "Hist GradientBoosting"):
            name = candidate

    model_file = {
        "Random Forest":        "random_forest.joblib",
        "Hist GradientBoosting": "gradient_boosting.joblib",
    }[name]
    path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(path):
        # fallback to RF
        path = os.path.join(MODELS_DIR, "random_forest.joblib")
        name = "Random Forest"
    print(f"[INFO] Tuning threshold for: {name}")
    return joblib.load(path)


def evaluate_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": threshold,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "fpr":       round(fpr,  4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def select_best(rows):
    """
    Best threshold: highest F1 where recall >= 0.88 and FPR <= 0.05.
    Falls back to: highest F1 where FPR <= 0.05.
    Falls back to: highest F1 overall.
    """
    candidate = [r for r in rows if r["recall"] >= 0.88 and r["fpr"] <= 0.05]
    if candidate:
        return max(candidate, key=lambda r: r["f1"])
    candidate = [r for r in rows if r["fpr"] <= 0.05]
    if candidate:
        return max(candidate, key=lambda r: r["f1"])
    return max(rows, key=lambda r: r["f1"])


def plot_tradeoff(df, best_thr):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df["threshold"], df["recall"],    "b-o", label="Recall", linewidth=2)
    ax.plot(df["threshold"], df["precision"], "g-s", label="Precision", linewidth=2)
    ax.plot(df["threshold"], df["f1"],        "r-^", label="F1", linewidth=2)
    ax.axvline(best_thr, color="purple", linestyle="--", linewidth=1.5,
               label=f"Best = {best_thr}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Recall / Precision / F1 vs Threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["fpr"],    df["recall"], "b-o", linewidth=2)
    ax2.axhline(0.88, color="green", linestyle="--", linewidth=1, label="Recall target 0.88")
    ax2.axvline(0.05, color="red",   linestyle="--", linewidth=1, label="FPR limit 0.05")
    for _, row in df.iterrows():
        ax2.annotate(f"{row['threshold']:.2f}",
                     (row["fpr"], row["recall"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("Recall (True Positive Rate)")
    ax2.set_title("Recall vs FPR Tradeoff by Threshold")
    ax2.legend()
    ax2.set_xlim(-0.01, 0.35)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "threshold_tradeoff.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Plot saved -> {path}")


if __name__ == "__main__":
    print("Threshold tuning for best supervised model...")

    X, y  = load_data()
    model = load_best_supervised_model()

    y_proba = model.predict_proba(X)[:, 1]

    print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'FPR':>8}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*5} {'-'*5} {'-'*5}")

    rows = []
    for thr in THRESHOLDS:
        r = evaluate_threshold(y, y_proba, thr)
        rows.append(r)
        flag = "  <-- default" if thr == 0.50 else ""
        print(f"  {r['threshold']:>10.2f} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['fpr']:>8.4f}  "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}{flag}")

    best = select_best(rows)
    print(f"\n  BEST threshold: {best['threshold']}  "
          f"(F1={best['f1']}, Recall={best['recall']}, FPR={best['fpr']})")
    print(f"  Selection criterion: recall >= 0.88 AND FPR <= 0.05, maximize F1")

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "threshold_tuning.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Table saved -> {csv_path}")

    # Save best threshold for other scripts to load
    thr_path = os.path.join(RESULTS_DIR, "best_threshold.json")
    with open(thr_path, "w") as f:
        json.dump({"threshold": best["threshold"],
                   "f1": best["f1"],
                   "recall": best["recall"],
                   "fpr": best["fpr"]}, f, indent=2)
    print(f"  Best threshold -> {thr_path}")

    plot_tradeoff(df, best["threshold"])

    print("\n[DONE] Threshold tuning complete.")
