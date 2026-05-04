"""
error_analysis.py
Analyzes false negatives (missed attacks) and false positives (false alarms)
from the best trained model (Random Forest) on the test set.

attack_type column is used for reporting ONLY -- it is NEVER passed to any model.

Outputs:
  results/error_analysis.txt
  results/attack_type_metrics.csv  (per-attack-type recall, precision, F1)
  results/figures/error_fn_fp_by_type.png
  results/figures/error_fp_feature_dist.png
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURE_COLS = [
    "error_rate", "unauthorized_rate", "throttle_rate",
    "forbidden_rate", "success_rate", "requests_per_route",
    "avg_gap_seconds", "min_gap_seconds", "latency_std",
    "post_ratio", "login_ratio", "unique_status_codes",
    "avg_latency", "max_latency", "unique_routes",
    "route_entropy",
    "route_switches_ratio", "repeated_route_ratio",
    "prev_requests_per_route", "delta_requests_per_route",
    "rolling2_unique_routes", "rolling2_route_entropy", "rolling2_error_rate",
]

_lines = []


def log(text=""):
    print(text)
    _lines.append(str(text))


def save_report():
    path = os.path.join(RESULTS_DIR, "error_analysis.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print(f"\n  Report saved -> {path}")


def load_inputs():
    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))

    X_sc   = read("X_test_sup.csv").values
    y      = read("y_test_sup.csv").values.ravel()
    X_raw  = read("X_test_raw.csv").values

    at_path = os.path.join(DATA_DIR, "attack_type_test.csv")
    if os.path.exists(at_path):
        attack_types = read("attack_type_test.csv").values.ravel()
    else:
        attack_types = np.where(y == 0, "normal", "unknown")

    return X_sc, y, X_raw, attack_types


def load_rf():
    path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RF model not found: {path}")
    return joblib.load(path)


def load_threshold():
    path = os.path.join(RESULTS_DIR, "best_threshold.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["threshold"]
    return 0.5


def predict_with_threshold(rf, X, threshold):
    proba  = rf.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return y_pred, proba


def section_overview(y, y_pred):
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    log("=" * 60)
    log("ERROR ANALYSIS REPORT")
    log("=" * 60)
    log(f"\n[1] OVERALL CONFUSION")
    log(f"  TP (attack caught)  : {tp}")
    log(f"  FP (false alarm)    : {fp}")
    log(f"  FN (missed attack)  : {fn}")
    log(f"  TN (correct normal) : {tn}")
    log(f"  FN rate (missed/total attacks) : {fn/(fn+tp+1e-9):.3f}")
    log(f"  FP rate (false alarm / normal) : {fp/(fp+tn+1e-9):.3f}")
    return tp, fp, fn, tn


def section_per_attack_type(y, y_pred, attack_types):
    log("\n[2] METRICS PER ATTACK TYPE")
    attack_type_list = ["flood", "slow_brute", "credential", "recon"]
    rows = []
    for atype in attack_type_list:
        mask = attack_types == atype
        if mask.sum() == 0:
            log(f"  {atype}: NO SAMPLES in test set")
            continue
        y_true_t = y[mask]
        y_pred_t = y_pred[mask]

        tp_t = int(((y_true_t == 1) & (y_pred_t == 1)).sum())
        fn_t = int(((y_true_t == 1) & (y_pred_t == 0)).sum())
        fp_t = int(((y_true_t == 0) & (y_pred_t == 1)).sum())
        tn_t = int(((y_true_t == 0) & (y_pred_t == 0)).sum())

        total_attack = int((y_true_t == 1).sum())
        recall_t = tp_t / (tp_t + fn_t + 1e-9)
        prec_t   = tp_t / (tp_t + fp_t + 1e-9) if (tp_t + fp_t) > 0 else 0.0
        f1_t     = 2 * prec_t * recall_t / (prec_t + recall_t + 1e-9)

        log(f"\n  [{atype}]")
        log(f"    Windows in test : {int(mask.sum())} "
            f"(attack={total_attack}, normal={int((y_true_t==0).sum())})")
        log(f"    TP={tp_t}  FN={fn_t}  FP={fp_t}  TN={tn_t}")
        log(f"    Recall    : {recall_t:.4f}")
        log(f"    Precision : {prec_t:.4f}")
        log(f"    F1        : {f1_t:.4f}")
        if fn_t > 0:
            log(f"    *** MISSED {fn_t} attack windows ***")

        rows.append({
            "attack_type": atype,
            "total_windows": int(mask.sum()),
            "attack_windows": total_attack,
            "TP": tp_t, "FN": fn_t, "FP": fp_t, "TN": tn_t,
            "recall": round(recall_t, 4),
            "precision": round(prec_t, 4),
            "f1": round(f1_t, 4),
        })

    df_at = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "attack_type_metrics.csv")
    df_at.to_csv(csv_path, index=False)
    log(f"\n  Per-attack metrics saved -> {csv_path}")
    return df_at


def section_fn_examples(y, y_pred, y_proba, X_raw, attack_types, n=10):
    log(f"\n[3] TOP {n} FALSE NEGATIVES (missed attacks)")
    fn_idx = np.where((y == 1) & (y_pred == 0))[0]
    if len(fn_idx) == 0:
        log("  No false negatives!")
        return

    # Sort by probability (lowest confidence of being attack = most confused)
    fn_idx = fn_idx[np.argsort(y_proba[fn_idx])][:n]

    for rank, idx in enumerate(fn_idx, 1):
        log(f"\n  FN #{rank}  attack_type={attack_types[idx]}  "
            f"RF_prob={y_proba[idx]:.4f}")
        top_features = sorted(
            zip(FEATURE_COLS, X_raw[idx]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]
        feat_str = "  ".join(f"{k}={v:.3f}" for k, v in top_features)
        log(f"    {feat_str}")


def section_fp_examples(y, y_pred, y_proba, X_raw, n=10):
    log(f"\n[4] TOP {n} FALSE POSITIVES (normal flagged as attack)")
    fp_idx = np.where((y == 0) & (y_pred == 1))[0]
    if len(fp_idx) == 0:
        log("  No false positives!")
        return

    # Sort by probability (highest confidence = most confident false alarm)
    fp_idx = fp_idx[np.argsort(y_proba[fp_idx])[::-1]][:n]

    for rank, idx in enumerate(fp_idx, 1):
        log(f"\n  FP #{rank}  RF_prob={y_proba[idx]:.4f}")
        top_features = sorted(
            zip(FEATURE_COLS, X_raw[idx]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]
        feat_str = "  ".join(f"{k}={v:.3f}" for k, v in top_features)
        log(f"    {feat_str}")


def section_fp_analysis(y, y_pred, X_raw):
    log("\n[5] FALSE POSITIVE FEATURE ANALYSIS")
    fp_mask = (y == 0) & (y_pred == 1)
    tn_mask = (y == 0) & (y_pred == 0)

    if fp_mask.sum() == 0:
        log("  No false positives to analyze.")
        return

    fp_df = pd.DataFrame(X_raw[fp_mask], columns=FEATURE_COLS)
    tn_df = pd.DataFrame(X_raw[tn_mask], columns=FEATURE_COLS)

    log("  Feature means:  FP (falsely flagged normal) vs TN (correct normal)")
    log(f"  {'Feature':<22}  {'FP mean':>10}  {'TN mean':>10}  {'ratio FP/TN':>12}")
    for col in FEATURE_COLS:
        fp_m = fp_df[col].mean()
        tn_m = tn_df[col].mean()
        ratio = fp_m / (tn_m + 1e-9)
        log(f"  {col:<22}  {fp_m:>10.4f}  {tn_m:>10.4f}  {ratio:>12.2f}")


def section_diagnosis(df_at):
    log("\n[6] DIAGNOSIS")
    if df_at is None or len(df_at) == 0:
        log("  No per-type metrics available.")
        return

    worst = df_at.loc[df_at["recall"].idxmin()] if "recall" in df_at.columns else None
    if worst is not None:
        log(f"  Hardest attack type: {worst['attack_type']}  "
            f"(recall={worst['recall']:.4f}  FN={worst['FN']})")
    if df_at["FN"].sum() > 0:
        by_fn = df_at.sort_values("FN", ascending=False)
        log("  FN count by type:")
        for _, row in by_fn.iterrows():
            log(f"    {row['attack_type']}: {row['FN']} missed")


# ---- Plots ------------------------------------------------------------------

def plot_fn_fp_by_type(df_at):
    if df_at is None or len(df_at) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Recall per attack type
    ax = axes[0]
    colors = ["firebrick" if r < 0.80 else "steelblue" for r in df_at["recall"]]
    ax.bar(df_at["attack_type"], df_at["recall"], color=colors)
    ax.axhline(0.80, color="orange", linestyle="--", linewidth=1, label="0.80 target")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall per Attack Type")
    ax.set_ylabel("Recall")
    ax.legend()
    for i, v in enumerate(df_at["recall"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # FN/TP counts per attack type
    ax2 = axes[1]
    x = np.arange(len(df_at))
    w = 0.35
    ax2.bar(x - w/2, df_at["TP"], w, label="TP (caught)", color="steelblue")
    ax2.bar(x + w/2, df_at["FN"], w, label="FN (missed)", color="firebrick")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_at["attack_type"])
    ax2.set_title("TP vs FN by Attack Type")
    ax2.set_ylabel("Window Count")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "error_fn_fp_by_type.png")
    plt.savefig(path, dpi=100)
    plt.close()
    log(f"\n  Plot: {path}")


def plot_fp_features(y, y_pred, X_raw):
    fp_mask = (y == 0) & (y_pred == 1)
    tn_mask = (y == 0) & (y_pred == 0)
    if fp_mask.sum() == 0:
        return

    fp_df = pd.DataFrame(X_raw[fp_mask], columns=FEATURE_COLS)
    tn_df = pd.DataFrame(X_raw[tn_mask], columns=FEATURE_COLS)

    # Show top 6 most different features
    diff = abs(fp_df.mean() - tn_df.mean())
    top6 = diff.nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), top6):
        ax.hist(tn_df[col], bins=20, alpha=0.6, label="TN (correct normal)",
                color="steelblue", density=True)
        ax.hist(fp_df[col], bins=20, alpha=0.6, label="FP (false alarm)",
                color="firebrick", density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle("Feature Distributions: FP (false alarms) vs TN (correct normal)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "error_fp_feature_dist.png")
    plt.savefig(path, dpi=100)
    plt.close()
    log(f"  Plot: {path}")


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Running error analysis on Random Forest test predictions...")

    X_sc, y, X_raw, attack_types = load_inputs()
    rf = load_rf()
    threshold = load_threshold()
    log(f"  Using classification threshold: {threshold}")

    y_pred, y_proba = predict_with_threshold(rf, X_sc, threshold)

    tp, fp, fn, tn = section_overview(y, y_pred)
    df_at = section_per_attack_type(y, y_pred, attack_types)
    section_fn_examples(y, y_pred, y_proba, X_raw, attack_types)
    section_fp_examples(y, y_pred, y_proba, X_raw)
    section_fp_analysis(y, y_pred, X_raw)
    section_diagnosis(df_at)

    plot_fn_fp_by_type(df_at)
    plot_fp_features(y, y_pred, X_raw)

    save_report()
    print("\n[DONE] Error analysis complete.")
