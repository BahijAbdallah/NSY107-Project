"""
detect_anomaly.py
Loads the best trained model and classifies new API log windows.

Usage:
  python src/detect_anomaly.py                   # runs on built-in test samples
  python src/detect_anomaly.py --csv path.csv    # runs on a CSV file

Input CSV must have the same columns as api_gateway_features.csv:
  request_count, success_count, error_count,
  status_401_count, status_403_count, status_429_count,
  avg_latency, max_latency, unique_routes

Output per row:
  NORMAL  or  ANOMALY DETECTED  + suspicious indicators
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

# ── Paths ──
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

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

# ── Built-in test samples ──
# All 15 features are behavioral/rate values — no raw counts.
DEMO_SAMPLES = [
    # 1. Normal session: low error rate, high success, human-paced gaps
    {
        "error_rate": 0.0, "unauthorized_rate": 0.0, "throttle_rate": 0.0,
        "forbidden_rate": 0.0, "success_rate": 1.0, "requests_per_route": 1.5,
        "avg_gap_seconds": 90.0, "min_gap_seconds": 60.0, "latency_std": 12.0,
        "post_ratio": 0.33, "login_ratio": 0.33, "unique_status_codes": 1,
        "avg_latency": 85.0, "max_latency": 110.0, "unique_routes": 2,
        "route_entropy": 1.0,
        "route_switches_ratio": 0.5,   # alternating 2 routes
        "repeated_route_ratio": 0.33,  # some repeats in normal browsing
        "prev_requests_per_route": 1.5, "delta_requests_per_route": 0.0,
        "rolling2_unique_routes": 2.0, "rolling2_route_entropy": 1.0,
        "rolling2_error_rate": 0.0,
        "_description": "Typical normal user session"
    },
    # 2. Slow brute force: 100% 401, all POST to /login
    {
        "error_rate": 1.0, "unauthorized_rate": 1.0, "throttle_rate": 0.0,
        "forbidden_rate": 0.0, "success_rate": 0.0, "requests_per_route": 4.0,
        "avg_gap_seconds": 60.0, "min_gap_seconds": 30.0, "latency_std": 20.0,
        "post_ratio": 1.0, "login_ratio": 1.0, "unique_status_codes": 1,
        "avg_latency": 150.0, "max_latency": 200.0, "unique_routes": 1,
        "route_entropy": 0.0,
        "route_switches_ratio": 0.0,   # single route, never switches
        "repeated_route_ratio": 0.75,  # 3/4 requests repeat same route
        "prev_requests_per_route": 4.0, "delta_requests_per_route": 0.0,
        "rolling2_unique_routes": 1.0, "rolling2_route_entropy": 0.0,
        "rolling2_error_rate": 1.0,
        "_description": "Slow brute force login  --  100% 401 Unauthorized"
    },
    # 3. Credential stuffing: mixed 200/401 (stealthy)
    {
        "error_rate": 0.583, "unauthorized_rate": 0.417, "throttle_rate": 0.083,
        "forbidden_rate": 0.083, "success_rate": 0.417, "requests_per_route": 6.0,
        "avg_gap_seconds": 22.0, "min_gap_seconds": 5.0, "latency_std": 18.0,
        "post_ratio": 0.75, "login_ratio": 0.67, "unique_status_codes": 4,
        "avg_latency": 100.0, "max_latency": 150.0, "unique_routes": 2,
        "route_entropy": 0.72,
        "route_switches_ratio": 0.4,   # mostly /login, occasional /secure
        "repeated_route_ratio": 0.67,  # 4/6 requests repeat /login
        "prev_requests_per_route": 6.0, "delta_requests_per_route": 0.0,
        "rolling2_unique_routes": 2.0, "rolling2_route_entropy": 0.72,
        "rolling2_error_rate": 0.583,
        "_description": "Credential stuffing  --  mixed 200/401 responses"
    },
    # 4. Flood: high error rate, rapid requests, high latency
    {
        "error_rate": 0.941, "unauthorized_rate": 0.471, "throttle_rate": 0.353,
        "forbidden_rate": 0.118, "success_rate": 0.059, "requests_per_route": 85.0,
        "avg_gap_seconds": 3.5, "min_gap_seconds": 0.1, "latency_std": 60.0,
        "post_ratio": 0.35, "login_ratio": 0.35, "unique_status_codes": 5,
        "avg_latency": 210.0, "max_latency": 550.0, "unique_routes": 1,
        "route_entropy": 0.0,
        "route_switches_ratio": 0.0,   # hammering single route
        "repeated_route_ratio": 0.99,  # almost all requests repeat same route
        "prev_requests_per_route": 85.0, "delta_requests_per_route": 0.0,
        "rolling2_unique_routes": 1.0, "rolling2_route_entropy": 0.0,
        "rolling2_error_rate": 0.941,
        "_description": "High-volume flood  --  85 req in 5 min, 94% errors"
    },
]


def load_best_model():
    """Load the best model based on results/best_model_name.txt."""
    name_path = os.path.join(RESULTS_DIR, "best_model_name.txt")

    if os.path.exists(name_path):
        with open(name_path) as f:
            best_name = f.read().strip()
    else:
        # Fallback: try each model in preference order
        best_name = "Random Forest"

    print(f"[INFO] Using model: {best_name}")

    if best_name == "Random Forest":
        path = os.path.join(MODELS_DIR, "random_forest.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}\nRun train_models.py first.")
        model = joblib.load(path)
        return "rf", model, None

    elif best_name == "Isolation Forest":
        path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}\nRun train_models.py first.")
        model = joblib.load(path)
        return "iso", model, None

    elif best_name == "Hist GradientBoosting":
        path = os.path.join(MODELS_DIR, "gradient_boosting.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}\nRun train_models.py first.")
        model = joblib.load(path)
        return "hgb", model, None

    elif best_name == "Autoencoder":
        ae_path  = os.path.join(MODELS_DIR, "autoencoder.keras")
        thr_path = os.path.join(MODELS_DIR, "autoencoder_threshold.json")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(ae_path)
        except ImportError:
            raise ImportError("tensorflow not installed  --  cannot load Autoencoder.")
        with open(thr_path) as f:
            threshold = json.load(f)["threshold"]
        return "ae", model, threshold

    else:
        raise ValueError(f"Unknown model name: {best_name}")


def load_scaler():
    path = os.path.join(DATA_DIR, "scaler.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found: {path}\nRun preprocess.py first.")
    return joblib.load(path)


def load_rf_threshold():
    """Load tuned threshold from threshold_tuning.py, default 0.5."""
    path = os.path.join(RESULTS_DIR, "best_threshold.json")
    if os.path.exists(path):
        with open(path) as f:
            thr = json.load(f)["threshold"]
        print(f"[INFO] Using tuned RF threshold: {thr}")
        return thr
    return 0.5


def predict(model_type, model, threshold, X_scaled):
    """Return binary predictions (0=normal, 1=anomaly)."""
    if model_type in ("rf", "hgb"):
        thr = load_rf_threshold()
        proba = model.predict_proba(X_scaled)[:, 1]
        return (proba >= thr).astype(int)
    elif model_type == "iso":
        raw = model.predict(X_scaled)
        return np.where(raw == -1, 1, 0)
    elif model_type == "ae":
        recon  = model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        return (errors > threshold).astype(int)


def explain(row: dict, pred: int) -> str:
    """Generate a short human-readable reason for the prediction."""
    reasons = []

    if pred == 0:
        return "NORMAL"

    if row.get("request_count", 0) > 20:
        reasons.append(f"high request count ({row['request_count']})")
    if row.get("status_401_count", 0) > 5:
        reasons.append(f"many 401 Unauthorized ({row['status_401_count']})")
    if row.get("status_429_count", 0) > 3:
        reasons.append(f"rate-limit hits ({row['status_429_count']} x 429)")
    if row.get("status_403_count", 0) > 3:
        reasons.append(f"access denied responses ({row['status_403_count']} x 403)")
    if row.get("avg_latency", 0) > 300:
        reasons.append(f"high avg latency ({row['avg_latency']:.0f} ms)")
    if row.get("error_count", 0) > 10:
        reasons.append(f"many errors ({row['error_count']} total 4xx/5xx)")
    if row.get("unique_routes", 0) == 1 and row.get("request_count", 0) > 10:
        reasons.append("targeting single route (possible DDoS)")

    reason_str = "; ".join(reasons) if reasons else "unusual traffic pattern"
    return f"ANOMALY DETECTED  --  {reason_str}"


def run_detection(rows: list[dict], model_type, model, threshold, scaler):
    """Run detection on a list of row dicts and print results."""
    df = pd.DataFrame(rows)
    X  = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    preds = predict(model_type, model, threshold, X_scaled)

    print("\n" + "=" * 60)
    for i, (row, pred) in enumerate(zip(rows, preds)):
        desc = row.get("_description", f"Sample {i+1}")
        result = explain(row, int(pred))
        icon   = "[ALERT]" if pred == 1 else "[OK]"
        print(f"\n{icon} [{i+1}] {desc}")
        print(f"   Input : {{{', '.join(f'{k}: {row[k]}' for k in FEATURE_COLS)}}}")
        print(f"   Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect API anomalies using trained ML model")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to input CSV with feature columns")
    args = parser.parse_args()

    scaler = load_scaler()
    model_type, model, threshold = load_best_model()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[ERROR] File not found: {args.csv}")
            sys.exit(1)
        df_input = pd.read_csv(args.csv)
        rows = df_input.to_dict(orient="records")
    else:
        print("[INFO] No --csv provided. Running on built-in demo samples.")
        rows = DEMO_SAMPLES

    run_detection(rows, model_type, model, threshold, scaler)
