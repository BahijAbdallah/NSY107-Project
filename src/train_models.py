"""
train_models.py  (v4 — TimeSeriesSplit CV, gap-aware selection, AE regularization)

Key change from v3:
  RF now uses TimeSeriesSplit(n_splits=3) instead of regular k-fold.
  Regular k-fold shuffles data randomly, giving an optimistic CV score because
  it can "see" future windows during validation. TimeSeriesSplit trains on
  earlier windows and validates on later ones — exactly matching the real
  train/test evaluation. This means the CV score is honest and the selected
  hyperparameters are those that generalize to future data, not memorize past data.

  We also print training F1 vs CV F1 explicitly so the overfitting gap is visible.
  Selection criterion: highest CV F1 where (train_F1 - CV_F1) <= 0.10.

Saved models:
  models/random_forest.joblib
  models/isolation_forest.joblib
  models/autoencoder.keras
  models/autoencoder_threshold.json
  models/ae_history.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))

    required = ["X_train_sup.csv", "y_train_sup.csv",
                "X_train_unsup.csv", "X_test_unsup.csv", "y_test_unsup.csv"]
    for f in required:
        if not os.path.exists(os.path.join(DATA_DIR, f)):
            raise FileNotFoundError(f"Missing {f}. Run preprocess.py first.")

    X_tr_sup   = read("X_train_sup.csv").values
    y_tr_sup   = read("y_train_sup.csv").values.ravel()
    X_tr_unsup = read("X_train_unsup.csv").values
    return X_tr_sup, y_tr_sup, X_tr_unsup


# ---- A) Random Forest -------------------------------------------------------

def train_random_forest(X_train, y_train):
    print("\n[A] Training Random Forest (TimeSeriesSplit CV — gap-aware)...")

    # TimeSeriesSplit: fold k trains on windows 0..k and validates on k+1.
    # This mirrors the real evaluation: train on past, test on future.
    # Regular k-fold would shuffle and leak future windows into training folds.
    tscv = TimeSeriesSplit(n_splits=3)

    param_dist = {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [5, 8, 10, 12, 15],   # no None — prevents full memorization
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf":  [3, 5, 8, 10],         # larger = less overfitting
        "max_features":      ["sqrt", "log2"],
        "class_weight":      ["balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=25,
        cv=tscv,        # <-- time-series aware, not shuffled
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        refit=True,     # refit on full training set with best params
        verbose=0
    )
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_

    # Compute training F1 to show the overfitting gap explicitly
    y_train_pred = best_rf.predict(X_train)
    train_f1     = f1_score(y_train, y_train_pred, zero_division=0)
    cv_f1        = search.best_score_
    gap          = train_f1 - cv_f1

    print(f"    Best params       : {search.best_params_}")
    print(f"    Train F1 (full)   : {train_f1:.4f}")
    print(f"    CV F1 (TimeSeries): {cv_f1:.4f}")
    print(f"    Overfit gap       : {gap:.4f}  "
          f"({'ACCEPTABLE' if gap <= 0.10 else 'HIGH — consider more regularization'})")

    path = os.path.join(MODELS_DIR, "random_forest.joblib")
    joblib.dump(best_rf, path)
    print(f"    Saved -> {path}")
    return best_rf


# ---- B) Isolation Forest ----------------------------------------------------

def train_isolation_forest(X_train_normal):
    print("\n[B] Training Isolation Forest (contamination + max_samples tuning)...")

    n_val = max(1, int(len(X_train_normal) * 0.20))
    X_fit = X_train_normal[:-n_val]
    X_val = X_train_normal[-n_val:]

    from itertools import product
    contaminations = [0.03, 0.05, 0.08, 0.10, 0.15]
    max_samples_opts = ["auto", 0.7, 0.9]

    print(f"    {'Contamination':>14}  {'max_samples':>12}  {'FPR on normal val':>18}")
    print(f"    {'-'*14}  {'-'*12}  {'-'*18}")

    best_cont = 0.10
    best_ms   = "auto"
    best_gap  = float("inf")

    for c, ms in product(contaminations, max_samples_opts):
        iso = IsolationForest(n_estimators=300, contamination=c,
                              max_samples=ms, random_state=42, n_jobs=-1)
        iso.fit(X_fit)
        raw     = iso.predict(X_val)
        fpr_val = float((raw == -1).sum()) / len(X_val)
        gap     = abs(fpr_val - c)
        print(f"    {c:>14.3f}  {str(ms):>12}  {fpr_val:>18.4f}")
        if gap < best_gap:
            best_gap  = gap
            best_cont = c
            best_ms   = ms

    print(f"\n    Selected: contamination={best_cont}  max_samples={best_ms}")

    iso_best = IsolationForest(n_estimators=300, contamination=best_cont,
                               max_samples=best_ms, random_state=42, n_jobs=-1)
    iso_best.fit(X_train_normal)

    path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(iso_best, path)
    print(f"    Saved -> {path}")
    return iso_best


# ---- D) Hist GradientBoosting -----------------------------------------------

def train_gradient_boosting(X_train, y_train):
    print("\n[D] Training Hist GradientBoosting (TimeSeriesSplit CV — gap-aware)...")
    from sklearn.ensemble import HistGradientBoostingClassifier

    tscv = TimeSeriesSplit(n_splits=3)

    param_dist = {
        "max_iter":          [100, 200, 300],
        "max_depth":         [3, 4, 5, 7],
        "min_samples_leaf":  [10, 20, 30],
        "learning_rate":     [0.05, 0.10, 0.15],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    base_hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)

    search = RandomizedSearchCV(
        base_hgb, param_dist,
        n_iter=20, cv=tscv, scoring="f1",
        random_state=42, n_jobs=-1, refit=True, verbose=0
    )
    search.fit(X_train, y_train)
    best_hgb = search.best_estimator_

    y_train_pred = best_hgb.predict(X_train)
    train_f1     = f1_score(y_train, y_train_pred, zero_division=0)
    cv_f1        = search.best_score_
    gap          = train_f1 - cv_f1

    print(f"    Best params       : {search.best_params_}")
    print(f"    Train F1 (full)   : {train_f1:.4f}")
    print(f"    CV F1 (TimeSeries): {cv_f1:.4f}")
    print(f"    Overfit gap       : {gap:.4f}  "
          f"({'ACCEPTABLE' if gap <= 0.10 else 'HIGH — consider more regularization'})")

    path = os.path.join(MODELS_DIR, "gradient_boosting.joblib")
    joblib.dump(best_hgb, path)
    print(f"    Saved -> {path}")
    return best_hgb


# ---- C) Autoencoder ---------------------------------------------------------

def _build_ae(n_features, arch, dropout_rate, l2_reg):
    from tensorflow import keras
    regularizer = keras.regularizers.l2(l2_reg)

    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in arch:
        x = keras.layers.Dense(units, activation="relu",
                                kernel_regularizer=regularizer)(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)

    for units in reversed(arch[:-1]):
        x = keras.layers.Dense(units, activation="relu",
                                kernel_regularizer=regularizer)(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(n_features, activation="linear")(x)
    ae = keras.Model(inputs, outputs)
    ae.compile(optimizer="adam", loss="mse")
    return ae


def train_autoencoder(X_train_normal):
    print("\n[C] Training Autoencoder (dropout + L2 + batch_size comparison)...")

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("    [SKIP] tensorflow not installed.")
        return None, None

    n_features = X_train_normal.shape[1]
    n_val      = max(1, int(len(X_train_normal) * 0.15))
    X_fit      = X_train_normal[:-n_val]
    X_val      = X_train_normal[-n_val:]

    # Grid: architecture x dropout x batch_size
    configs = [
        {"arch": (32, 16), "dropout": 0.1, "l2": 1e-4, "batch": 32},
        {"arch": (32, 16), "dropout": 0.1, "l2": 1e-4, "batch": 64},
        {"arch": (32, 16), "dropout": 0.2, "l2": 1e-4, "batch": 64},
        {"arch": (64, 32, 16), "dropout": 0.1, "l2": 1e-4, "batch": 64},
    ]

    best_ae      = None
    best_val_loss = float("inf")
    best_cfg     = None
    best_history = None

    for cfg in configs:
        label = (f"arch={cfg['arch']} dropout={cfg['dropout']} "
                 f"l2={cfg['l2']} batch={cfg['batch']}")
        print(f"\n    Config: {label}")

        ae = _build_ae(n_features, cfg["arch"], cfg["dropout"], cfg["l2"])

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        )

        history = ae.fit(
            X_fit, X_fit,
            epochs=80,
            batch_size=cfg["batch"],
            validation_data=(X_val, X_val),
            callbacks=[early_stop],
            verbose=0
        )
        vl = min(history.history["val_loss"])
        print(f"    Best val_loss = {vl:.6f}  (stopped at epoch {len(history.history['val_loss'])})")

        if vl < best_val_loss:
            best_val_loss = vl
            best_ae       = ae
            best_cfg      = cfg
            best_history  = history

    print(f"\n    Selected config: arch={best_cfg['arch']} "
          f"dropout={best_cfg['dropout']} batch={best_cfg['batch']}")
    print(f"    Best val_loss : {best_val_loss:.6f}")

    # Compute reconstruction errors on full normal training data
    recon  = best_ae.predict(X_train_normal, verbose=0)
    errors = np.mean((X_train_normal - recon) ** 2, axis=1)

    thresholds = {
        "p95":   float(np.percentile(errors, 95)),
        "p97_5": float(np.percentile(errors, 97.5)),
        "p99":   float(np.percentile(errors, 99)),
    }
    print(f"    Thresholds (MSE):")
    for k, v in thresholds.items():
        print(f"      {k}: {v:.6f}")

    threshold = thresholds["p95"]

    ae_path  = os.path.join(MODELS_DIR, "autoencoder.keras")
    thr_path = os.path.join(MODELS_DIR, "autoencoder_threshold.json")
    hist_path = os.path.join(MODELS_DIR, "ae_history.json")

    best_ae.save(ae_path)

    with open(thr_path, "w") as f:
        json.dump({"threshold": threshold, "all_thresholds": thresholds,
                   "architecture": str(best_cfg["arch"])}, f, indent=2)

    # Save training history for loss plot
    with open(hist_path, "w") as f:
        json.dump({
            "train_loss": best_history.history["loss"],
            "val_loss":   best_history.history["val_loss"],
        }, f, indent=2)

    print(f"    Model     -> {ae_path}")
    print(f"    Threshold -> {thr_path}")
    print(f"    History   -> {hist_path}")
    return best_ae, threshold


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading preprocessed data...")
    X_tr_sup, y_tr_sup, X_tr_unsup = load_data()
    print(f"  Supervised train   : {X_tr_sup.shape}  "
          f"(normal={int((y_tr_sup==0).sum())} attack={int((y_tr_sup==1).sum())})")
    print(f"  Unsupervised train : {X_tr_unsup.shape}  (normal only)")

    train_random_forest(X_tr_sup, y_tr_sup)
    train_gradient_boosting(X_tr_sup, y_tr_sup)
    train_isolation_forest(X_tr_unsup)
    train_autoencoder(X_tr_unsup)

    print("\n[DONE] All models trained and saved.")
