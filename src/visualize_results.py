"""
visualize_results.py
Generates all evaluation plots and saves them to results/figures/.

Plots produced:
  01_metrics_comparison.png       — bar chart: accuracy/precision/recall/F1 per model
  02_confusion_matrix_<model>.png — confusion matrix for each model
  03_roc_curves.png               — ROC curves
  04_precision_recall_curves.png  — Precision-Recall curves
  05_anomaly_scores_iso.png       — Isolation Forest anomaly score distribution
  06_anomaly_scores_ae.png        — Autoencoder reconstruction error distribution
  07_pca_2d.png                   — PCA 2D view: normal vs anomaly coloring
  08_feature_importance.png       — RF feature importances (all 21 features)
  09_ae_training_loss.png         — Autoencoder train/val loss curves
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── Paths ──
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def load_data():
    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))

    X_sup   = read("X_test_sup.csv").values
    y_sup   = read("y_test_sup.csv").values.ravel()
    X_unsup = read("X_test_unsup.csv").values
    y_unsup = read("y_test_unsup.csv").values.ravel()
    return X_sup, y_sup, X_unsup, y_unsup


# ── 1. Metrics comparison bar chart ──
def plot_metrics_comparison():
    cmp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(cmp_path):
        print("  [SKIP] model_comparison.csv not found — run evaluate_models.py first")
        return

    df = pd.read_csv(cmp_path)
    metrics = ["accuracy", "precision", "recall", "f1"]
    df_plot = df[["model"] + metrics].copy()
    # Convert roc_auc if numeric
    if "roc_auc" in df.columns:
        try:
            df_plot["roc_auc"] = pd.to_numeric(df["roc_auc"], errors="coerce")
            metrics.append("roc_auc")
            df_plot = df[["model"] + metrics].copy()
        except Exception:
            pass

    df_melt = df_plot.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_melt, x="metric", y="score", hue="model")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=14)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.legend(title="Model")
    plt.tight_layout()
    savefig("01_metrics_comparison.png")


# ── 2. Confusion matrices ──
def plot_confusion_matrices(X_sup, y_sup, X_unsup, y_unsup):
    tasks = []

    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        tasks.append(("Random_Forest", rf.predict(X_sup), y_sup))

    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
        raw = iso.predict(X_unsup)
        tasks.append(("Isolation_Forest", np.where(raw == -1, 1, 0), y_unsup))

    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            import tensorflow as tf
            ae = tf.keras.models.load_model(ae_path)
            with open(os.path.join(MODELS_DIR, "autoencoder_threshold.json")) as f:
                thr = json.load(f)["threshold"]
            recon  = ae.predict(X_unsup, verbose=0)
            errors = np.mean((X_unsup - recon) ** 2, axis=1)
            tasks.append(("Autoencoder", (errors > thr).astype(int), y_unsup))
        except ImportError:
            pass

    for model_name, y_pred, y_true in tasks:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"]
        )
        plt.title(f"Confusion Matrix — {model_name.replace('_', ' ')}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        savefig(f"02_confusion_matrix_{model_name}.png")


# ── 3. ROC curves ──
def plot_roc_curves(X_sup, y_sup, X_unsup, y_unsup):
    plt.figure(figsize=(7, 5))
    plotted = False

    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        scores = rf.predict_proba(X_sup)[:, 1]
        fpr, tpr, _ = roc_curve(y_sup, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc:.3f})")
        plotted = True

    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
        scores = -iso.score_samples(X_unsup)
        fpr, tpr, _ = roc_curve(y_unsup, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Isolation Forest (AUC={roc_auc:.3f})")
        plotted = True

    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            import tensorflow as tf
            ae = tf.keras.models.load_model(ae_path)
            recon  = ae.predict(X_unsup, verbose=0)
            scores = np.mean((X_unsup - recon) ** 2, axis=1)
            fpr, tpr, _ = roc_curve(y_unsup, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Autoencoder (AUC={roc_auc:.3f})")
            plotted = True
        except ImportError:
            pass

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    savefig("03_roc_curves.png")


# ── 4. Precision-Recall curves ──
def plot_pr_curves(X_sup, y_sup, X_unsup, y_unsup):
    plt.figure(figsize=(7, 5))
    plotted = False

    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        scores = rf.predict_proba(X_sup)[:, 1]
        prec, rec, _ = precision_recall_curve(y_sup, scores)
        plt.plot(rec, prec, label="Random Forest")
        plotted = True

    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
        scores = -iso.score_samples(X_unsup)
        prec, rec, _ = precision_recall_curve(y_unsup, scores)
        plt.plot(rec, prec, label="Isolation Forest")
        plotted = True

    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            import tensorflow as tf
            ae = tf.keras.models.load_model(ae_path)
            recon  = ae.predict(X_unsup, verbose=0)
            scores = np.mean((X_unsup - recon) ** 2, axis=1)
            prec, rec, _ = precision_recall_curve(y_unsup, scores)
            plt.plot(rec, prec, label="Autoencoder")
            plotted = True
        except ImportError:
            pass

    if not plotted:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    savefig("04_precision_recall_curves.png")


# ── 5. Anomaly score distributions ──
def plot_score_distributions(X_unsup, y_unsup):
    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
        scores = -iso.score_samples(X_unsup)
        df = pd.DataFrame({"score": scores, "label": y_unsup})

        plt.figure(figsize=(8, 4))
        for lbl, grp in df.groupby("label"):
            name = "Attack" if lbl == 1 else "Normal"
            plt.hist(grp["score"], bins=50, alpha=0.6, label=name, density=True)
        plt.xlabel("Anomaly Score (higher = more anomalous)")
        plt.ylabel("Density")
        plt.title("Isolation Forest — Anomaly Score Distribution")
        plt.legend()
        plt.tight_layout()
        savefig("05_anomaly_scores_iso.png")

    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            import tensorflow as tf
            ae  = tf.keras.models.load_model(ae_path)
            recon  = ae.predict(X_unsup, verbose=0)
            errors = np.mean((X_unsup - recon) ** 2, axis=1)
            df = pd.DataFrame({"error": errors, "label": y_unsup})

            with open(os.path.join(MODELS_DIR, "autoencoder_threshold.json")) as f:
                thr = json.load(f)["threshold"]

            plt.figure(figsize=(8, 4))
            for lbl, grp in df.groupby("label"):
                name = "Attack" if lbl == 1 else "Normal"
                plt.hist(grp["error"], bins=50, alpha=0.6, label=name, density=True)
            plt.axvline(thr, color="red", linestyle="--", label=f"Threshold={thr:.4f}")
            plt.xlabel("Reconstruction Error (MSE)")
            plt.ylabel("Density")
            plt.title("Autoencoder — Reconstruction Error Distribution")
            plt.legend()
            plt.tight_layout()
            savefig("06_anomaly_scores_ae.png")
        except ImportError:
            pass


# ── 6. PCA 2D scatter ──
def plot_pca_2d(X_unsup, y_unsup):
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_unsup)
    df  = pd.DataFrame({"PC1": X2d[:, 0], "PC2": X2d[:, 1], "label": y_unsup})

    plt.figure(figsize=(8, 6))
    colors = {0: "steelblue", 1: "tomato"}
    labels = {0: "Normal", 1: "Attack"}
    for lbl in [0, 1]:
        sub = df[df["label"] == lbl]
        plt.scatter(sub["PC1"], sub["PC2"], c=colors[lbl],
                    label=labels[lbl], alpha=0.4, s=10)
    var = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    plt.title("PCA 2D — Normal vs Attack Traffic")
    plt.legend()
    plt.tight_layout()
    savefig("07_pca_2d.png")


# ── Main ──
if __name__ == "__main__":
    print("Generating visualizations...")
    X_sup, y_sup, X_unsup, y_unsup = load_data()

    plot_metrics_comparison()
    plot_confusion_matrices(X_sup, y_sup, X_unsup, y_unsup)
    plot_roc_curves(X_sup, y_sup, X_unsup, y_unsup)
    plot_pr_curves(X_sup, y_sup, X_unsup, y_unsup)
    plot_score_distributions(X_unsup, y_unsup)
    plot_pca_2d(X_unsup, y_unsup)

    print(f"\n[DONE] All figures saved to {FIG_DIR}")
