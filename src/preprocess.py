"""
preprocess.py  (v3 — 21 features, time-based split, metadata saved)

Changes from v2:
  - FEATURE_COLS expanded from 9 to 21 (rate + behavioral features)
  - attack_type metadata saved to data/attack_type_test.csv (for error analysis)
  - Unscaled test features saved to data/X_test_raw.csv (for interpretability)
  - Time-based split unchanged (80th percentile of window_start)
  - Scaler still fitted on training set only

Outputs in data/:
  X_train_sup.csv   y_train_sup.csv
  X_test_sup.csv    y_test_sup.csv
  X_test_raw.csv    (unscaled test features — for error analysis)
  X_train_unsup.csv
  X_test_unsup.csv  y_test_unsup.csv
  attack_type_test.csv
  scaler.joblib

results/feature_list.txt  updated with 21 feature names
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FEAT_CSV    = os.path.join(DATA_DIR, "api_gateway_features.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = [
    # Rate / behavioral features — generalize across traffic volumes
    "error_rate", "unauthorized_rate", "throttle_rate",
    "forbidden_rate", "success_rate", "requests_per_route",
    "avg_gap_seconds", "min_gap_seconds", "latency_std",
    "post_ratio", "login_ratio", "unique_status_codes",
    # Continuous features (not raw event counts)
    "avg_latency", "max_latency", "unique_routes",
    # Route diversity (v4)
    "route_entropy",
    # Within-window route sequence (v5)
    "route_switches_ratio",
    "repeated_route_ratio",
    # Per-IP rolling/lag features (v5) — past windows only, no future leakage
    "prev_requests_per_route",
    "delta_requests_per_route",
    "rolling2_unique_routes",
    "rolling2_route_entropy",
    "rolling2_error_rate",
]
LABEL_COL = "label"
META_COLS = ["attack_type"]   # metadata — not features, not label


def load_and_validate():
    if not os.path.exists(FEAT_CSV):
        raise FileNotFoundError(
            f"{FEAT_CSV} not found.\nRun:  python src/build_dataset.py  first."
        )
    df = pd.read_csv(FEAT_CSV)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing values.")

    print(f"  Loaded {len(df)} rows | "
          f"normal: {(df[LABEL_COL]==0).sum()} | "
          f"attack: {(df[LABEL_COL]==1).sum()}")
    return df


def time_based_split(df, split_quantile=0.80):
    """Split at the split_quantile percentile of window_start."""
    if "window_start" in df.columns:
        df["window_start"] = pd.to_datetime(df["window_start"])
        df = df.sort_values("window_start").reset_index(drop=True)
        cutoff     = df["window_start"].quantile(split_quantile)
        train_mask = df["window_start"] <= cutoff
    else:
        n          = len(df)
        cutoff_idx = int(n * split_quantile)
        train_mask = df.index < cutoff_idx

    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)


if __name__ == "__main__":
    print("Preprocessing dataset (v5 — 23 features, time-based split)...")
    df = load_and_validate()

    # ── Time-based split ──────────────────────────────────────────────────────
    df_train, df_test = time_based_split(df, split_quantile=0.80)
    print(f"  Split: train={len(df_train)} windows | test={len(df_test)} windows")

    X_tr = df_train[FEATURE_COLS].values
    y_tr = df_train[LABEL_COL].values
    X_te = df_test[FEATURE_COLS].values
    y_te = df_test[LABEL_COL].values

    print(f"  Train: normal={int((y_tr==0).sum())} | attack={int((y_tr==1).sum())}")
    print(f"  Test : normal={int((y_te==0).sum())} | attack={int((y_te==1).sum())}")

    if "attack_type" in df_test.columns:
        for atype in ["flood", "slow_brute", "credential", "recon"]:
            cnt = int((df_test["attack_type"] == atype).sum())
            if cnt > 0:
                print(f"    Test {atype}: {cnt} windows")

    # ── Scaler: fit on train only ─────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # ── Unsupervised split ────────────────────────────────────────────────────
    normal_mask   = (y_tr == 0)
    X_tr_unsup_sc = scaler.transform(X_tr[normal_mask])
    X_te_unsup_sc = scaler.transform(X_te)
    y_te_unsup    = y_te

    print(f"  Unsupervised train (normal only): {len(X_tr_unsup_sc)}")

    # ── Save CSV splits ───────────────────────────────────────────────────────
    def save(arr, path):
        pd.DataFrame(arr, columns=FEATURE_COLS).to_csv(path, index=False)

    save(X_tr_sc,  os.path.join(DATA_DIR, "X_train_sup.csv"))
    save(X_te_sc,  os.path.join(DATA_DIR, "X_test_sup.csv"))
    save(X_tr_unsup_sc, os.path.join(DATA_DIR, "X_train_unsup.csv"))
    save(X_te_unsup_sc, os.path.join(DATA_DIR, "X_test_unsup.csv"))

    # Unscaled test features for interpretable error analysis
    pd.DataFrame(X_te, columns=FEATURE_COLS).to_csv(
        os.path.join(DATA_DIR, "X_test_raw.csv"), index=False)

    pd.Series(y_tr,      name="label").to_csv(
        os.path.join(DATA_DIR, "y_train_sup.csv"),  index=False)
    pd.Series(y_te,      name="label").to_csv(
        os.path.join(DATA_DIR, "y_test_sup.csv"),   index=False)
    pd.Series(y_te_unsup, name="label").to_csv(
        os.path.join(DATA_DIR, "y_test_unsup.csv"), index=False)

    # Attack type metadata for error analysis
    if "attack_type" in df_test.columns:
        df_test["attack_type"].reset_index(drop=True).to_csv(
            os.path.join(DATA_DIR, "attack_type_test.csv"), index=False)
        print(f"  attack_type_test.csv saved")

    # ── Save scaler ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(DATA_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler -> {scaler_path}")

    # ── Save feature list ─────────────────────────────────────────────────────
    feat_list_path = os.path.join(RESULTS_DIR, "feature_list.txt")
    with open(feat_list_path, "w") as f:
        for col in FEATURE_COLS:
            f.write(col + "\n")
    print(f"  Feature list ({len(FEATURE_COLS)} features) -> {feat_list_path}")

    print("\n[DONE] Preprocessing complete.")
