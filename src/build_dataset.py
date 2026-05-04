"""
build_dataset.py  (v5 — 23 features + attack_type metadata)

Changes from v2:
  - 12 new behavioral / rate features added to aggregated windows
  - attack_type column added to features CSV for error analysis (NOT a training feature)
  - Flood attack bursts spread to hours 4, 10, 16, 22 so hour-22 burst falls in test window
  - Normal traffic gets realistic timing (batched sessions)

New features (in addition to original 9):
  error_rate, unauthorized_rate, throttle_rate, forbidden_rate, success_rate,
  requests_per_route, avg_gap_seconds, min_gap_seconds, latency_std,
  post_ratio, login_ratio, unique_status_codes,
  route_entropy  (NEW v4 — Shannon entropy of route distribution;
                  recon scans all routes uniformly → max entropy;
                  brute force hits only /login → entropy ≈ 0)

attack_type values: "normal", "flood", "slow_brute", "credential", "recon"
  -- saved in features CSV for error analysis ONLY, never passed to models

Outputs:
  data/api_gateway_dataset.csv
  data/api_gateway_features.csv  (22 ML features + window_start + label + attack_type)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_CSV  = os.path.join(DATA_DIR, "api_gateway_dataset.csv")
FEAT_CSV = os.path.join(DATA_DIR, "api_gateway_features.csv")

random.seed(42)
np.random.seed(42)

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0)
ROUTES    = ["/public", "/login", "/secure", "/orders"]


def _row(ip, ts, route, status, latency, label, attack_type="normal"):
    method = "GET" if route in ["/public", "/secure"] else "POST"
    return {
        "timestamp":     ts.isoformat(),
        "ip":            ip,
        "route":         route,
        "method":        method,
        "status":        status,
        "latency":       latency,
        "response_size": random.randint(100, 2000),
        "label":         label,
        "attack_type":   attack_type,
    }


# ---- Normal traffic ---------------------------------------------------------

def generate_normal_traffic(n=8000):
    """
    50 IPs, spread across 24 h with human-paced sessions.
    Includes realistic occasional 401 (6%) and 429 (2%) to overlap with
    stealthy attacks when rate-normalized features are compared.
    """
    ips = [f"10.0.{random.randint(0,5)}.{random.randint(1,254)}"
           for _ in range(50)]
    records = []
    for _ in range(n):
        ts     = BASE_TIME + timedelta(seconds=random.randint(0, 86400))
        ip     = random.choice(ips)
        route  = random.choices(ROUTES, weights=[0.30, 0.15, 0.38, 0.17])[0]
        status = random.choices(
            [200, 201, 400, 401, 403, 429],
            weights=[0.72, 0.08, 0.09, 0.06, 0.03, 0.02]
        )[0]
        latency = max(5, int(np.random.normal(80, 25)))
        records.append(_row(ip, ts, route, status, latency, label=0,
                            attack_type="normal"))
    return records


# ---- Attack Type A: Flood ---------------------------------------------------

def generate_flood_attack():
    """
    16 IPs, 4 bursts across the day (hours 4, 10, 16, 22).
    5-15 requests per burst per IP — overlaps with normal volume so the
    signal comes from error/throttle rates and latency, not raw count.
    """
    ips         = [f"192.168.1.{i}" for i in range(1, 17)]
    burst_hours = [4, 10, 16, 22]
    records     = []
    for ip in ips:
        for hour in burst_hours:
            offset = hour * 3600
            count  = random.randint(5, 15)
            for _ in range(count):
                ts     = BASE_TIME + timedelta(seconds=offset + random.randint(0, 299))
                route  = random.choices(ROUTES, weights=[0.10, 0.30, 0.45, 0.15])[0]
                status = random.choices(
                    [200, 400, 401, 403, 429, 500],
                    weights=[0.10, 0.15, 0.25, 0.20, 0.25, 0.05]
                )[0]
                latency = max(10, int(np.random.normal(130, 40)))
                records.append(_row(ip, ts, route, status, latency,
                                    label=1, attack_type="flood"))
    return records


# ---- Attack Type B: Slow Brute Force ----------------------------------------

def generate_slow_brute(n=2000):
    """
    12 IPs, sessions of 2-5 requests per 5-min window.
    88% 401 errors — rate-based feature unauthorized_rate is the key signal.
    request_count overlaps with normal (2-5 vs 1-4), making this stealthy.
    """
    ips     = [f"172.16.0.{i}" for i in range(1, 13)]
    records = []
    while len(records) < n:
        ip     = random.choice(ips)
        offset = random.randint(3 * 3600, 21 * 3600 - 300)
        count  = random.randint(2, 5)
        for _ in range(count):
            ts     = BASE_TIME + timedelta(seconds=offset + random.randint(0, 299))
            route  = random.choices(["/login", "/secure"], weights=[0.70, 0.30])[0]
            status = random.choices(
                [200, 400, 401, 403],
                weights=[0.03, 0.03, 0.88, 0.06]
            )[0]
            latency = max(10, int(np.random.normal(110, 35)))
            records.append(_row(ip, ts, route, status, latency,
                                label=1, attack_type="slow_brute"))
    return records[:n]


# ---- Attack Type C: Credential Stuffing -------------------------------------

def generate_credential_stuffing(n=1500):
    """
    6 IPs, sessions of 5-20 req/window.
    40% succeed (valid stolen credentials) — stealthy because success_rate looks normal.
    The combination success_rate=0.40 + unauthorized_rate=0.40 is the signal.
    Increased to 1000 (from 800) for better test coverage.
    """
    ips     = [f"172.20.0.{i}" for i in range(1, 7)]
    records = []
    while len(records) < n:
        ip     = random.choice(ips)
        offset = random.randint(0, 86400 - 300)
        count  = random.randint(5, 20)
        for _ in range(count):
            ts     = BASE_TIME + timedelta(seconds=offset + random.randint(0, 299))
            route  = random.choices(["/login", "/secure"], weights=[0.80, 0.20])[0]
            status = random.choices(
                [200, 400, 401, 403, 429],
                weights=[0.40, 0.05, 0.40, 0.10, 0.05]
            )[0]
            latency = max(5, int(np.random.normal(100, 30)))
            records.append(_row(ip, ts, route, status, latency,
                                label=1, attack_type="credential"))
    return records[:n]


# ---- Attack Type D: Recon Scan ----------------------------------------------

def generate_recon_scan(n=800):
    """
    4 IPs, sessions of 3-10 req/window hitting ALL 4 routes.
    unique_routes=4 is distinctive; login_ratio ~0.25 is uniform (unusual).
    Increased to 500 (from 400) for better test coverage.
    """
    ips     = [f"192.0.2.{i}" for i in range(1, 5)]
    records = []
    while len(records) < n:
        ip     = random.choice(ips)
        offset = random.randint(0, 86400 - 300)
        # Minimum 4 ensures all 4 routes are always visited at least once,
        # making route_entropy = log2(4) = 2.0 a reliable recon signal.
        count  = random.randint(4, 10)
        base_routes = ROUTES.copy()
        random.shuffle(base_routes)
        for j in range(count):
            route  = base_routes[j % 4]
            ts     = BASE_TIME + timedelta(seconds=offset + random.randint(0, 299))
            status = random.choices(
                [200, 400, 401, 403],
                weights=[0.30, 0.25, 0.25, 0.20]
            )[0]
            latency = max(5, int(np.random.normal(90, 20)))
            records.append(_row(ip, ts, route, status, latency,
                                label=1, attack_type="recon"))
    return records[:n]


# ---- Aggregation ------------------------------------------------------------

def _compute_window(g):
    """Compute all 23 ML features + label + attack_type for one (ip, window) group."""
    n = len(g)

    # Timing gaps between consecutive requests within the window
    sorted_ts = g["timestamp"].sort_values()
    diffs     = sorted_ts.diff().dropna().dt.total_seconds()
    avg_gap   = float(diffs.mean()) if len(diffs) > 0 else 0.0
    min_gap   = float(diffs.min())  if len(diffs) > 0 else 0.0
    if np.isnan(avg_gap): avg_gap = 0.0
    if np.isnan(min_gap): min_gap = 0.0

    rc   = n
    sc   = int((g["status"].isin([200, 201])).sum())
    ec   = int((g["status"] >= 400).sum())
    s401 = int((g["status"] == 401).sum())
    s403 = int((g["status"] == 403).sum())
    s429 = int((g["status"] == 429).sum())
    ur   = int(g["route"].nunique())

    # Route entropy: 0 when all requests hit one route (brute force);
    # log2(n_routes) when perfectly uniform (recon scans all routes equally).
    route_probs = g["route"].value_counts(normalize=True).values
    r_entropy   = float(-(route_probs * np.log2(route_probs + 1e-10)).sum())

    # Within-window route sequence features (sorted by timestamp)
    sorted_routes = g.sort_values("timestamp")["route"].values
    n_req = len(sorted_routes)
    if n_req > 1:
        sw = sum(sorted_routes[i] != sorted_routes[i-1] for i in range(1, n_req))
        r_sw_ratio = sw / (n_req - 1)
    else:
        r_sw_ratio = 0.0
    # repeated_route_ratio: fraction of requests going to an already-visited route.
    # Recon cycles through all routes once → 0.0; brute force repeats /login → high.
    seen_r, repeats = set(), 0
    for r in sorted_routes:
        if r in seen_r:
            repeats += 1
        seen_r.add(r)
    rep_rt_ratio = repeats / n_req

    post_n  = int((g["method"] == "POST").sum())
    login_n = int((g["route"] == "/login").sum())

    return pd.Series({
        # ---- Original 9 features ----
        "request_count":    rc,
        "success_count":    sc,
        "error_count":      ec,
        "status_401_count": s401,
        "status_403_count": s403,
        "status_429_count": s429,
        "avg_latency":      round(float(g["latency"].mean()), 2),
        "max_latency":      int(g["latency"].max()),
        "unique_routes":    ur,
        # ---- New 12 behavioral/rate features ----
        "error_rate":          round(ec   / rc, 4),
        "unauthorized_rate":   round(s401 / rc, 4),
        "throttle_rate":       round(s429 / rc, 4),
        "forbidden_rate":      round(s403 / rc, 4),
        "success_rate":        round(sc   / rc, 4),
        "requests_per_route":  round(rc   / max(ur, 1), 2),
        "avg_gap_seconds":     round(avg_gap, 2),
        "min_gap_seconds":     round(min_gap, 2),
        "latency_std":         round(float(np.std(g["latency"].values)), 2),
        "post_ratio":          round(post_n  / rc, 4),
        "login_ratio":         round(login_n / rc, 4),
        "unique_status_codes": int(g["status"].nunique()),
        "route_entropy":          round(r_entropy, 4),
        "route_switches_ratio":   round(r_sw_ratio, 4),
        "repeated_route_ratio":   round(rep_rt_ratio, 4),
        # ---- Metadata (NOT ML features) ----
        "label":              int(g["label"].max()),
        "attack_type":        str(g["attack_type"].mode().iloc[0]),
    })


def aggregate_features(df):
    """
    Group raw requests by (ip, 5-min window) and compute features.
    Keeps window_start (for time-based split) and attack_type (for error analysis).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["window"]    = df["timestamp"].dt.floor("5min")

    agg = (df.groupby(["ip", "window"])
             .apply(_compute_window, include_groups=False)
             .reset_index())

    agg = agg.rename(columns={"window": "window_start"})
    # Sort per-IP chronologically before computing lag/rolling features.
    agg = agg.sort_values(["ip", "window_start"]).reset_index(drop=True)

    # Lag feature: previous window's request density per IP.
    # Uses shift(1) so only PAST windows are seen — no future leakage.
    agg["prev_requests_per_route"] = (
        agg.groupby("ip")["requests_per_route"].shift(1).fillna(0)
    )
    agg["delta_requests_per_route"] = (
        agg["requests_per_route"] - agg["prev_requests_per_route"]
    )

    # Rolling mean of the 2 most recent PAST windows per IP.
    # shift(1) shifts the series back by one before rolling, so the current
    # window is never included in its own rolling mean.
    for src, dst in [
        ("unique_routes", "rolling2_unique_routes"),
        ("route_entropy", "rolling2_route_entropy"),
        ("error_rate",    "rolling2_error_rate"),
    ]:
        agg[dst] = (
            agg.groupby("ip")[src]
               .transform(lambda g: g.shift(1).rolling(2, min_periods=1).mean())
               .fillna(0)
        )

    agg = agg.sort_values("window_start").drop(columns=["ip"]).reset_index(drop=True)
    return agg


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":
    print("Building API gateway dataset v5 (23 features + attack_type)...")

    normal     = generate_normal_traffic(n=8000)
    flood      = generate_flood_attack()
    slow_brute = generate_slow_brute(n=2000)
    credential = generate_credential_stuffing(n=1500)
    recon      = generate_recon_scan(n=800)

    all_raw = normal + flood + slow_brute + credential + recon
    df_raw  = pd.DataFrame(all_raw)
    df_raw  = df_raw.sort_values("timestamp").reset_index(drop=True)
    df_raw.to_csv(RAW_CSV, index=False)

    n_atk = len(flood) + len(slow_brute) + len(credential) + len(recon)
    print(f"  Raw : {len(df_raw):,} rows  (normal={len(normal):,}  attacks={n_atk:,})")
    print(f"    Flood={len(flood)}, SlowBrute={len(slow_brute)}, "
          f"Credential={len(credential)}, Recon={len(recon)}")

    df_feat = aggregate_features(df_raw)

    # Remove duplicate windows — identical feature vectors inflate performance
    # when the same row appears in both train and test after the time split.
    feat_cols = [c for c in df_feat.columns
                 if c not in ("window_start", "label", "attack_type")]
    before_dedup = len(df_feat)
    df_feat = df_feat.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"  Dedup: {before_dedup} -> {len(df_feat)} windows "
          f"(removed {before_dedup - len(df_feat)} duplicates)")

    df_feat.to_csv(FEAT_CSV, index=False)

    n_nw = int((df_feat["label"] == 0).sum())
    n_aw = int((df_feat["label"] == 1).sum())
    print(f"  Windows : {len(df_feat):,}  (normal={n_nw:,}  attack={n_aw:,})")

    for atype in ["flood", "slow_brute", "credential", "recon"]:
        cnt = int((df_feat["attack_type"] == atype).sum())
        print(f"    {atype}: {cnt} windows")

    print(f"  Columns : {list(df_feat.columns)}")
    print(f"  -> {FEAT_CSV}")
    print("\n[DONE] Dataset ready.")
