"""
audit_dataset.py
Audits api_gateway_features.csv to diagnose suspicious ML results (e.g. F1=1.0).

Checks:
  - Dataset shape, columns, class distribution
  - Missing values and duplicate rows
  - Feature statistics by class (mean, std)
  - Feature-label correlation (point-biserial r)
  - Single-feature AUROC (separability per feature)
  - Random Forest feature importances on full dataset
  - Data leakage check (label not in feature set)

Outputs:
  results/audit_report.txt
  results/figures/audit/audit_01_class_distribution.png
  results/figures/audit/audit_02_feature_distributions.png
  results/figures/audit/audit_03_feature_correlation.png
  results/figures/audit/audit_04_feature_importance.png
  results/figures/audit/audit_05_single_feature_separability.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings("ignore")

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(ROOT, "data")
RESULTS_DIR  = os.path.join(ROOT, "results")
AUDIT_DIR    = os.path.join(RESULTS_DIR, "figures", "audit")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = [
    # Rate / behavioral features (volume-normalized)
    "error_rate", "unauthorized_rate", "throttle_rate",
    "forbidden_rate", "success_rate", "requests_per_route",
    "avg_gap_seconds", "min_gap_seconds", "latency_std",
    "post_ratio", "login_ratio", "unique_status_codes",
    # Continuous features
    "avg_latency", "max_latency", "unique_routes",
    # Route diversity (v4)
    "route_entropy",
    # Within-window route sequence (v5)
    "route_switches_ratio",
    "repeated_route_ratio",
    # Per-IP rolling/lag features (v5)
    "prev_requests_per_route",
    "delta_requests_per_route",
    "rolling2_unique_routes",
    "rolling2_route_entropy",
    "rolling2_error_rate",
]
LABEL_COL = "label"

_report_lines = []


def log(text=""):
    print(text)
    _report_lines.append(str(text))


def save_report():
    path = os.path.join(RESULTS_DIR, "audit_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    log(f"\n  Audit report saved -> {path}")


# ---- 1. Load ----------------------------------------------------------------

def load_data():
    path = os.path.join(DATA_DIR, "api_gateway_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run build_dataset.py first.")
    return pd.read_csv(path)


# ---- 2. Basic info ----------------------------------------------------------

def section_basic(df):
    log("=" * 60)
    log("DATASET AUDIT REPORT")
    log("=" * 60)
    log("\n[1] BASIC INFO")
    log(f"  Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Columns       : {list(df.columns)}")
    log(f"  Feature cols  : {FEATURE_COLS}")
    log(f"  Target col    : {LABEL_COL}")
    log(f"\n  dtypes:\n{df.dtypes.to_string()}")


# ---- 3. Class distribution --------------------------------------------------

def section_class_dist(df):
    log("\n[2] CLASS DISTRIBUTION")
    n_normal = int((df[LABEL_COL] == 0).sum())
    n_attack = int((df[LABEL_COL] == 1).sum())
    total    = len(df)
    log(f"  Normal (0) : {n_normal:,}  ({100*n_normal/total:.1f}%)")
    log(f"  Attack (1) : {n_attack:,}  ({100*n_attack/total:.1f}%)")
    log(f"  Imbalance  : {n_normal/n_attack:.1f}:1")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Normal (0)", "Attack (1)"], [n_normal, n_attack],
           color=["steelblue", "firebrick"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Window Count")
    for i, v in enumerate([n_normal, n_attack]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(AUDIT_DIR, "audit_01_class_distribution.png"), dpi=100)
    plt.close()
    log("  Saved: audit_01_class_distribution.png")


# ---- 4. Data quality --------------------------------------------------------

def section_quality(df):
    log("\n[3] DATA QUALITY")
    missing = df[FEATURE_COLS + [LABEL_COL]].isna().sum()
    n_dupes = df.duplicated().sum()
    log(f"  Duplicate rows  : {n_dupes}")
    log(f"  Missing values  :\n{missing.to_string()}")


# ---- 5. Feature stats by class ----------------------------------------------

def section_feature_stats(df):
    log("\n[4] FEATURE MEANS BY CLASS")
    y      = df[LABEL_COL]
    norm_  = df[y == 0][FEATURE_COLS]
    atk_   = df[y == 1][FEATURE_COLS]

    rows = []
    for col in FEATURE_COLS:
        rows.append({
            "feature"     : col,
            "normal_mean" : round(norm_[col].mean(), 3),
            "normal_std"  : round(norm_[col].std(),  3),
            "attack_mean" : round(atk_[col].mean(),  3),
            "attack_std"  : round(atk_[col].std(),   3),
        })
    stats = pd.DataFrame(rows)
    log(stats.to_string(index=False))

    # Plot distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, col in zip(axes.flatten(), FEATURE_COLS):
        ax.hist(norm_[col], bins=30, alpha=0.6, label="Normal",
                color="steelblue", density=True)
        ax.hist(atk_[col],  bins=30, alpha=0.6, label="Attack",
                color="firebrick", density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle("Feature Distributions by Class (density)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(AUDIT_DIR, "audit_02_feature_distributions.png"), dpi=100)
    plt.close()
    log("\n  Saved: audit_02_feature_distributions.png")


# ---- 6. Feature-label correlation -------------------------------------------

def section_correlation(df):
    log("\n[5] FEATURE-LABEL CORRELATION (point-biserial r)")
    y    = df[LABEL_COL]
    rows = []
    for col in FEATURE_COLS:
        r, p = pointbiserialr(df[col].fillna(0), y)
        rows.append({"feature": col, "r": round(r, 4), "p_value": round(p, 6)})
    corr = pd.DataFrame(rows).sort_values("r", ascending=False)
    log(corr.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["firebrick" if v >= 0 else "steelblue" for v in corr["r"]]
    ax.barh(corr["feature"], corr["r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Point-Biserial Correlation with Label")
    ax.set_xlabel("Correlation (r)")
    plt.tight_layout()
    plt.savefig(os.path.join(AUDIT_DIR, "audit_03_feature_correlation.png"), dpi=100)
    plt.close()
    log("  Saved: audit_03_feature_correlation.png")

    return corr


# ---- 7. Single-feature AUROC ------------------------------------------------

def section_separability(df):
    log("\n[6] SINGLE-FEATURE SEPARABILITY (AUROC)")
    y    = df[LABEL_COL]
    rows = []
    for col in FEATURE_COLS:
        vals = df[col].fillna(0).values
        try:
            auroc = roc_auc_score(y, vals)
            if auroc < 0.5:
                auroc = 1.0 - auroc   # flip direction
        except Exception:
            auroc = float("nan")
        rows.append({"feature": col, "auroc": round(auroc, 4)})
    sep = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    log(sep.to_string(index=False))

    perfect = sep[sep["auroc"] >= 0.95]
    if len(perfect) > 0:
        log(f"\n  WARNING: {len(perfect)} feature(s) achieve AUROC >= 0.95 alone:")
        for _, row in perfect.iterrows():
            log(f"    {row['feature']}: {row['auroc']}")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["firebrick" if a >= 0.95 else "steelblue" for a in sep["auroc"]]
    ax.barh(sep["feature"], sep["auroc"], color=colors)
    ax.axvline(0.5,  color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.95, color="red",  linestyle="--", linewidth=0.8,
               label="0.95 danger line")
    ax.set_xlim(0, 1.05)
    ax.set_title("Single-Feature AUROC  (red = near-perfect separator)")
    ax.set_xlabel("AUROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(AUDIT_DIR, "audit_05_single_feature_separability.png"),
                dpi=100)
    plt.close()
    log("  Saved: audit_05_single_feature_separability.png")

    return sep


# ---- 8. RF feature importances ----------------------------------------------

def section_rf_importances(df):
    log("\n[7] RANDOM FOREST FEATURE IMPORTANCES (full dataset)")
    X  = df[FEATURE_COLS].values
    y  = df[LABEL_COL].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    log(imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="steelblue")
    ax.set_title("Random Forest Feature Importances (trained on full dataset)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(AUDIT_DIR, "audit_04_feature_importance.png"), dpi=100)
    plt.close()
    log("  Saved: audit_04_feature_importance.png")


# ---- 9. Leakage check -------------------------------------------------------

def section_leakage(df):
    log("\n[8] DATA LEAKAGE CHECK")
    in_features = LABEL_COL in FEATURE_COLS
    log(f"  'label' in FEATURE_COLS : {in_features}  (should be False)")
    extra = [c for c in df.columns if c not in FEATURE_COLS and c != LABEL_COL]
    log(f"  Extra columns           : {extra}")
    if in_features:
        log("  CRITICAL: label column is in feature set -- direct leakage!")
    else:
        log("  OK: no direct label leakage detected.")


# ---- 10. Diagnosis ----------------------------------------------------------

def section_diagnosis(sep_df):
    log("\n[9] ROOT CAUSE DIAGNOSIS")
    top = sep_df.iloc[0]
    log(f"  Top single-feature separator: '{top['feature']}' (AUROC={top['auroc']})")

    if top["auroc"] >= 0.99:
        log("  VERDICT: Perfect single-feature separation.")
        log("  One feature alone splits normal vs attack with AUROC >= 0.99.")
        log("  This is the root cause of F1=1.0 -- the task is trivially easy.")
        log("  FIX: Redesign dataset so attack features overlap with normal.")
    elif top["auroc"] >= 0.95:
        log("  VERDICT: Near-perfect single-feature separation (AUROC >= 0.95).")
        log("  Dataset is too easy. Model memorizes a threshold, not a pattern.")
    else:
        log("  VERDICT: No single feature achieves AUROC >= 0.95.")
        log("  Dataset has sufficient overlap. F1=1.0 would be surprising.")


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()

    section_basic(df)
    section_class_dist(df)
    section_quality(df)
    section_feature_stats(df)
    section_correlation(df)
    sep_df = section_separability(df)
    section_rf_importances(df)
    section_leakage(df)
    section_diagnosis(sep_df)

    save_report()
    print("\n[DONE] Audit complete.")
