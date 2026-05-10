import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "gradient_boosting.joblib")
SCALER_PATH = os.path.join(ROOT, "data", "scaler.joblib")

FEATURE_COLS = [
    "error_rate", "unauthorized_rate", "throttle_rate", "forbidden_rate",
    "success_rate", "requests_per_route", "avg_gap_seconds",
    "min_gap_seconds", "latency_std", "post_ratio", "login_ratio",
    "unique_status_codes", "avg_latency", "max_latency", "unique_routes",
    "route_entropy", "route_switches_ratio", "repeated_route_ratio",
    "prev_requests_per_route", "delta_requests_per_route",
    "rolling2_unique_routes", "rolling2_route_entropy", "rolling2_error_rate",
]

THRESHOLD = 0.55


def parse_cloudwatch_csv(csv_path):
    df = pd.read_csv(csv_path)

    message_col = "@message" if "@message" in df.columns else "message"
    timestamp_col = "@timestamp" if "@timestamp" in df.columns else "timestamp"

    rows = []

    for _, r in df.iterrows():
        try:
            msg = json.loads(r[message_col])

            route_key = msg.get("routeKey", "")
            parts = route_key.split(" ", 1)

            method = msg.get("httpMethod") or (parts[0] if len(parts) > 0 else "")
            route = parts[1] if len(parts) > 1 else route_key

            rows.append({
                "timestamp": r.get(timestamp_col) or msg.get("requestTime"),
                "ip": msg.get("ip"),
                "method": method,
                "route": route,
                "status": int(msg.get("status", 0)),
                "latency": float(msg.get("responseLatency", 0)),
            })

        except Exception:
            continue

    logs = pd.DataFrame(rows)

    if logs.empty:
        raise ValueError("No valid CloudWatch JSON log rows found.")

    logs = logs.dropna()
    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
    logs = logs.dropna(subset=["timestamp"])

    return logs.sort_values("timestamp").reset_index(drop=True)


def build_features(logs):
    logs["window"] = logs["timestamp"].dt.floor("5min")

    rows = []

    for (ip, window), g in logs.groupby(["ip", "window"]):
        g = g.sort_values("timestamp")
        n = len(g)

        success = int(g["status"].isin([200, 201]).sum())
        errors = int((g["status"] >= 400).sum())
        s401 = int((g["status"] == 401).sum())
        s403 = int((g["status"] == 403).sum())
        s429 = int((g["status"] == 429).sum())

        unique_routes = int(g["route"].nunique())

        diffs = g["timestamp"].diff().dropna().dt.total_seconds()
        avg_gap = float(diffs.mean()) if len(diffs) else 0.0
        min_gap = float(diffs.min()) if len(diffs) else 0.0

        route_probs = g["route"].value_counts(normalize=True).values
        route_entropy = float(-(route_probs * np.log2(route_probs + 1e-10)).sum())

        routes = g["route"].tolist()

        if len(routes) > 1:
            switches = sum(routes[i] != routes[i - 1] for i in range(1, len(routes)))
            route_switches_ratio = switches / (len(routes) - 1)
        else:
            route_switches_ratio = 0.0

        seen = set()
        repeats = 0

        for route in routes:
            if route in seen:
                repeats += 1
            seen.add(route)

        repeated_route_ratio = repeats / n

        rows.append({
            "ip": ip,
            "window": window,

            "error_rate": errors / n,
            "unauthorized_rate": s401 / n,
            "throttle_rate": s429 / n,
            "forbidden_rate": s403 / n,
            "success_rate": success / n,
            "requests_per_route": n / max(unique_routes, 1),

            "avg_gap_seconds": avg_gap,
            "min_gap_seconds": min_gap,
            "latency_std": float(np.std(g["latency"])),

            "post_ratio": int((g["method"] == "POST").sum()) / n,
            "login_ratio": int((g["route"] == "/login").sum()) / n,
            "unique_status_codes": int(g["status"].nunique()),

            "avg_latency": float(g["latency"].mean()),
            "max_latency": float(g["latency"].max()),
            "unique_routes": unique_routes,

            "route_entropy": route_entropy,
            "route_switches_ratio": route_switches_ratio,
            "repeated_route_ratio": repeated_route_ratio,
        })

    features = pd.DataFrame(rows)
    features = features.sort_values(["ip", "window"]).reset_index(drop=True)

    features["prev_requests_per_route"] = (
        features.groupby("ip")["requests_per_route"].shift(1).fillna(0)
    )

    features["delta_requests_per_route"] = (
        features["requests_per_route"] - features["prev_requests_per_route"]
    )

    for src, dst in [
        ("unique_routes", "rolling2_unique_routes"),
        ("route_entropy", "rolling2_route_entropy"),
        ("error_rate", "rolling2_error_rate"),
    ]:
        features[dst] = (
            features.groupby("ip")[src]
            .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
            .fillna(0)
        )

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CloudWatch CSV export file")
    args = parser.parse_args()

    print("[INFO] Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("[INFO] Parsing CloudWatch logs...")
    logs = parse_cloudwatch_csv(args.csv)
    print(f"[INFO] Parsed log rows: {len(logs)}")

    print("[INFO] Building 5-minute feature windows...")
    features = build_features(logs)
    print(f"[INFO] Feature windows: {len(features)}")

    X = features[FEATURE_COLS].fillna(0)
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    print("\n========== AWS LOG ANOMALY DETECTION RESULTS ==========")

    for i, row in features.iterrows():
        result = "ANOMALY" if predictions[i] == 1 else "NORMAL"

        print(f"\n[{result}]")
        print(f"IP: {row['ip']}")
        print(f"Window: {row['window']}")
        print(f"Anomaly probability: {probabilities[i]:.4f}")
        print(
            f"error_rate={row['error_rate']:.2f}, "
            f"401_rate={row['unauthorized_rate']:.2f}, "
            f"429_rate={row['throttle_rate']:.2f}, "
            f"avg_latency={row['avg_latency']:.2f}, "
            f"unique_routes={row['unique_routes']}"
        )

    print("\n[DONE] Real AWS logs processed.")


if __name__ == "__main__":
    main()