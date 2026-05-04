# API Gateway Anomaly Detection

AI-powered intrusion detection system for cloud API gateways. Aggregates per-IP traffic into 5-minute windows and classifies them as normal or attack using four machine learning models — Hist GradientBoosting, Random Forest, Isolation Forest, and an Autoencoder.

---

## Final Conclusion (v5)

The final model is **Hist GradientBoosting at threshold 0.55**, trained on 23 behavioral and historical features extracted from per-IP 5-minute traffic windows.

**Final metrics:**

| Metric | Value |
|---|---|
| Test F1 | **0.9719** |
| Recall | **99.1%** |
| Precision | **95.3%** |
| FPR (false alarm rate) | **1.33%** |
| Train − Test gap | **0.0281** |

**Per-attack-type recall (Hist GradientBoosting, threshold = 0.55):**

| Attack Type | Recall |
|---|---|
| Flood | 100% |
| Slow Brute Force | 100% |
| Credential Stuffing | 100% |
| Recon Scan | 95.1% |

**Why v5 improved over v4 (F1 = 0.920 → 0.972):**
Five per-IP historical features were added — rolling means of error rate, unique routes, and route entropy over the two preceding windows, plus the previous window's request density and its delta. The most impactful is `rolling2_error_rate`: attack IPs generate elevated errors consistently across multiple windows, a pattern invisible when analyzing one window at a time. These features use only past windows (shift-before-rolling), so there is no future leakage.

**Leakage audit — all clean:**
- Rolling features use only past windows (`shift(1)` before rolling — current window never sees itself)
- Scaler fitted on training data only, applied to test data without refitting
- TimeSeriesSplit validates on future windows in every fold — no shuffling
- Labels (`label`, `attack_type`) are not in the feature set — verified in code
- IP addresses are dropped from the dataset before any model sees the data

**Main limitation:**
The dataset is synthetic. Each attack IP exclusively attacks, giving its rolling features a perfectly consistent signal. Real production logs may contain mixed behavior (an attacker with some legitimate prior traffic, a legitimate user with a burst of 401 errors), which would reduce the discriminative power of rolling features. Real-world performance would likely be somewhat lower than 97.2%.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| **Hist GradientBoosting** | **98.6%** | **95.3%** | **99.1%** | **97.2%** | **99.9%** | **1.3%** |
| Random Forest | 98.2% | 93.7% | 98.2% | 95.9% | 99.9% | 1.8% |
| Isolation Forest | 96.0% | 87.8% | 94.7% | 91.1% | 99.4% | 3.6% |
| Autoencoder | 94.7% | 83.8% | 93.4% | 88.3% | 98.7% | 5.0% |

**Best model: Hist GradientBoosting** — selected by highest F1 with Recall ≥ 60% and FPR ≤ 30%.

**Overfitting check (honest metrics):**
- Train F1: 1.0000 | CV F1 (TimeSeriesSplit): 0.9809 | Test F1: 0.9677
- Train − Test gap: **0.0323** — well within the 0.10 acceptable threshold.

**After threshold tuning (threshold = 0.55):** F1 = 0.9719, Recall = 99.1%, FPR = 1.33%

**Per-attack-type recall (Hist GradientBoosting, threshold 0.55):**
| Attack Type | Recall | Note |
|---|---|---|
| Flood | 100% | Always detected |
| Slow Brute Force | 100% | Persistent 401 history caught by rolling error rate |
| Credential Stuffing | 100% | Route sequence features expose repeated /login |
| Recon Scan | 95.1% | Rolling unique-routes history reveals multi-window scanning |

---

## Attack Types Detected

| Type | Behavior | Key Signal |
|---|---|---|
| **Flood** | 5–15 requests/window in short bursts | High error rate + throttle rate |
| **Slow Brute Force** | 2–5 req/window, 88% 401 errors | Elevated `unauthorized_rate` |
| **Credential Stuffing** | 5–20 req/window, 40% succeed | Mixed `success_rate` + `unauthorized_rate` |
| **Recon Scan** | Visits all 4 routes per session | High `unique_routes` + `requests_per_route` |

---

## Architecture

```
Client
  │
  ▼
AWS API Gateway  (throttling, rate limiting, CloudWatch logs)
  │
  ▼
EC2 — Node.js / Express Backend  (JWT auth, input validation, logging)
  │
  ▼
CloudWatch Logs  →  api_gateway_features.csv  (per IP / 5-min window)
  │
  ├── Random Forest   (supervised)
  ├── Isolation Forest (unsupervised)
  └── Autoencoder      (unsupervised)
        │
        ▼
  Best Model → NORMAL / ANOMALY DETECTED
```

---

## Features Used (23)

The models train exclusively on **rate, behavioral, and historical features** — raw event counts are excluded to ensure the model learns patterns rather than volume thresholds.

**Rate / behavioral (12):**
| Feature | Description |
|---|---|
| `error_rate` | Fraction of requests returning 4xx/5xx |
| `unauthorized_rate` | Fraction returning 401 |
| `throttle_rate` | Fraction returning 429 |
| `forbidden_rate` | Fraction returning 403 |
| `success_rate` | Fraction returning 200/201 |
| `requests_per_route` | Average requests per unique route |
| `avg_gap_seconds` | Mean time between consecutive requests |
| `min_gap_seconds` | Minimum time between requests |
| `latency_std` | Standard deviation of response latency |
| `post_ratio` | Fraction of POST requests |
| `login_ratio` | Fraction of requests to /login |
| `unique_status_codes` | Number of distinct status codes seen |

**Continuous (3):**
| Feature | Description |
|---|---|
| `avg_latency` | Mean response time (ms) |
| `max_latency` | Peak response time (ms) |
| `unique_routes` | Number of distinct endpoints accessed |

**Route diversity (1, added v4):**
| Feature | Description |
|---|---|
| `route_entropy` | Shannon entropy of route distribution — 0 for single-route attacks (brute force), log₂(4)=2 for recon scans that visit all routes uniformly |

**Within-window route sequence (2, added v5):**
| Feature | Description |
|---|---|
| `route_switches_ratio` | Consecutive route changes / (n−1). Recon alternates routes → 1.0; brute force stays on /login → 0.0 |
| `repeated_route_ratio` | Fraction of requests going to an already-visited route. Recon visits each route once → 0.0; brute force → 0.75+ |

**Per-IP rolling/lag history (5, added v5 — no future leakage):**
| Feature | Description |
|---|---|
| `prev_requests_per_route` | Request density from this IP's previous window |
| `delta_requests_per_route` | Change in request density vs previous window |
| `rolling2_unique_routes` | Mean unique routes over 2 most recent past windows |
| `rolling2_route_entropy` | Mean route entropy over 2 most recent past windows |
| `rolling2_error_rate` | Mean error rate over 2 most recent past windows |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Start the backend
```bash
cd backend
npm install
node server.js
```
Runs on `http://localhost:3000`

### 3. (Optional) Simulate traffic
```bash
python3 scripts/normal_traffic.py
python3 scripts/attack_traffic.py
```

### 4. Run the full ML pipeline
```bash
python3 src/build_dataset.py       # generate dataset (3,558 deduplicated windows)
python3 src/preprocess.py          # clean, scale, time-based split
python3 src/audit_dataset.py       # verify dataset quality (no trivial separation)
python3 src/train_models.py        # train all 3 models (TimeSeriesSplit CV)
python3 src/evaluate_models.py     # print metrics + overfitting gap check
python3 src/threshold_tuning.py    # find optimal classification threshold
python3 src/error_analysis.py      # FN/FP breakdown by attack type
python3 src/visualize_results.py   # generate plots → results/figures/
python3 src/detect_anomaly.py      # run demo detection
```

---

## Backend API

| Method | Route | Auth | Description |
|---|---|---|---|
| GET | `/public` | None | Health check |
| POST | `/login` | None | Returns JWT token |
| GET | `/secure` | JWT | Protected resource |
| POST | `/orders` | JWT | Create order (validated) |

Demo credentials: `admin / password123` or `user1 / pass456`

---

## Project Structure

```
Project6_API_Anomaly_Detection/
├── backend/
│   ├── server.js               Express server (JWT auth, logging)
│   └── package.json
├── scripts/
│   ├── normal_traffic.py       Simulates legitimate user sessions
│   └── attack_traffic.py       Simulates flood, brute force, recon
├── src/
│   ├── build_dataset.py        Generates raw logs + aggregates features
│   ├── preprocess.py           Dedup, scale, time-based train/test split
│   ├── train_models.py         Trains RF / Isolation Forest / Autoencoder
│   ├── evaluate_models.py      Computes metrics, selects best model
│   ├── visualize_results.py    Saves plots to results/figures/
│   ├── detect_anomaly.py       Live detection with human-readable alerts
│   ├── audit_dataset.py        Dataset health checks (separability, leakage)
│   ├── threshold_tuning.py     Autoencoder threshold analysis
│   └── error_analysis.py       Per-attack-type breakdown
├── data/                       Generated CSV splits + scaler
├── models/                     Saved trained models
├── results/
│   ├── figures/                PNG plots (metrics, ROC, confusion matrices)
│   ├── model_comparison.csv
│   └── best_model_summary.txt
├── notebooks/
└── requirements.txt
```

---

## Model Details

### Random Forest (Supervised)
- Trained on labeled normal + attack windows
- Hyperparameters selected via `RandomizedSearchCV` (25 iterations, **TimeSeriesSplit(3)**, F1 scoring)
- `class_weight='balanced'` to handle class imbalance
- TimeSeriesSplit ensures CV trains on past windows and validates on future — prevents temporal data leakage

### Hist GradientBoosting (Supervised — comparison model)
- Also trained with TimeSeriesSplit CV (20 iterations)
- `class_weight='balanced'`; L2 regularization parameter tuned
- Similar F1 to RF but slightly higher recall and slightly higher FPR — RF wins on F1+FPR balance

### Isolation Forest (Unsupervised)
- Trained on **normal traffic only** — no attack labels required
- Contamination parameter tuned by minimizing gap between target and actual FPR on a held-out normal validation set
- 300 estimators

### Autoencoder (Unsupervised)
- Trained on **normal traffic only**
- Learns to reconstruct normal windows; attacks produce high reconstruction error (MSE)
- Architecture: 15 → 32 → 16 → 32 → 15 with Dropout(0.1) + L2 regularization
- Max 80 epochs, early stopping (patience=7), batch size=64
- Detection threshold: 95th percentile of training reconstruction errors

---

## Data Pipeline

Raw HTTP requests are aggregated into **per-IP / 5-minute windows**. Each window becomes one training sample. The split is **time-based** (80/20 by `window_start`) — the model trains on past traffic and is evaluated on future traffic, which mirrors production deployment.

**Dataset (after deduplication):**
- 5,312 windows total — 4,108 normal (77.3%) / 1,204 attack (22.7%)
- Train: 4,259 windows | Test: 1,053 windows
- Duplicate windows removed after rolling feature computation (two windows with the same within-window stats but different IP history are genuinely distinct samples and are kept)
- Rolling features computed strictly on past windows — shift(1) before rolling(2) ensures the current window never contributes to its own historical features

---

## Limitations

- **Synthetic dataset.** CloudWatch logs are not available locally, so traffic was simulated. Each attack IP exclusively attacks, making rolling features highly discriminative. In real production logs, attackers may have mixed histories and legitimate users may generate occasional errors, which would reduce performance somewhat.
- **Two-window history only.** The rolling features look back at the two most recent windows per IP. Attacks that escalate gradually over longer periods would not be fully captured. An LSTM or longer rolling window would help.
- **Batch pipeline.** The current system reads a CSV and runs inference offline. A production deployment would require streaming (e.g., CloudWatch → Kinesis → Lambda → model).

---

## Future Work

- **Real logs.** Replace the simulated dataset with actual AWS CloudWatch / API Gateway access logs. This is the single most impactful improvement.
- **Mixed IP behavior.** Generate a synthetic dataset where some IPs mix normal and attack traffic, to stress-test the rolling features under realistic conditions.
- **Sequence model (LSTM).** Learn attack escalation patterns across many consecutive windows per IP, not just the previous two.
- **AWS WAF integration.** Automatically block IPs flagged by the model without manual intervention.
- **Real-time pipeline.** CloudWatch → Kinesis Data Streams → Lambda → model inference → SNS alert.
- **Analyst feedback loop.** Allow security analysts to label false positives and negatives to continuously retrain the model on real-world edge cases.
