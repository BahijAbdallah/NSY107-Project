"""
attack_traffic.py
Simulates API abuse scenarios for security testing:
  1. Rapid flooding (rate-limit trigger)
  2. Missing JWT
  3. Invalid / tampered JWT
  4. Brute-force login attempts
  5. Malformed / broken JSON payloads
  6. Invalid order payloads (negative quantity, empty itemName, etc.)
  7. Rapid authenticated requests (token abuse)

Change API_BASE_URL to your API Gateway URL or local backend address.
WARNING: Run only against systems you are authorized to test.
"""

import requests
import time
import random

# ── Change this to your API Gateway URL or http://localhost:3000 ──
API_BASE_URL = "https://3fuaxiet0j.execute-api.us-east-1.amazonaws.com"

_valid_token = None   # obtained once for mixed-attack scenarios


def _login_valid():
    """Obtain a legitimate token for mixed-scenario tests."""
    global _valid_token
    try:
        r = requests.post(
            f"{API_BASE_URL}/login",
            json={"username": "admin", "password": "password123"},
            timeout=10
        )
        if r.status_code == 200:
            _valid_token = r.json().get("token")
            print("[SETUP] Valid token obtained for mixed-attack tests\n")
    except requests.exceptions.RequestException:
        pass


# ── Scenario 1: Rapid flooding ──
def rapid_flood():
    print("[ATTACK 1] Rapid flooding of /public (30 requests, no delay)")
    for i in range(30):
        try:
            r = requests.get(f"{API_BASE_URL}/public", timeout=5)
            print(f"  [{i+1:02d}] /public -> {r.status_code}")
        except Exception as e:
            print(f"  [{i+1:02d}] ERROR: {e}")
    # No sleep — intentionally rapid


# ── Scenario 2: Missing JWT ──
def missing_jwt():
    print("\n[ATTACK 2] GET /secure with no Authorization header (5x)")
    for i in range(5):
        r = requests.get(f"{API_BASE_URL}/secure", timeout=5)
        print(f"  [{i+1}] /secure (no token) -> {r.status_code}")
        time.sleep(0.1)


# ── Scenario 3: Invalid / tampered JWTs ──
def invalid_jwt():
    print("\n[ATTACK 3] GET /secure with invalid/tampered tokens")
    bad_tokens = [
        "Bearer invalidtoken123",
        "Bearer eyJhbGciOiJIUzI1NiJ9.tampered.badsignature",
        "Bearer ",
        "NotBearer abc",
        "Bearer null",
        "",
    ]
    for token in bad_tokens:
        headers = {"Authorization": token} if token else {}
        r = requests.get(f"{API_BASE_URL}/secure", headers=headers, timeout=5)
        preview = token[:45] if token else "(empty header)"
        print(f"  /secure -> {r.status_code}  | token: {preview}")
        time.sleep(0.1)


# ── Scenario 4: Brute-force login ──
def brute_force_login():
    print("\n[ATTACK 4] Brute-force login attempts")
    attempts = [
        {"username": "admin",  "password": "wrongpass"},
        {"username": "admin",  "password": "123456"},
        {"username": "root",   "password": "root"},
        {"username": "admin",  "password": ""},
        {"username": "",       "password": ""},
        {"username": "user1",  "password": "hacked"},
        {"username": "' OR 1=1--", "password": "sql_injection"},
    ]
    for attempt in attempts:
        r = requests.post(f"{API_BASE_URL}/login", json=attempt, timeout=5)
        print(f"  /login {attempt} -> {r.status_code}")
        time.sleep(0.15)


# ── Scenario 5: Malformed JSON ──
def malformed_json():
    print("\n[ATTACK 5] Malformed JSON to POST /orders")
    auth_header = {"Authorization": f"Bearer {_valid_token}"} if _valid_token else {}
    headers = {**auth_header, "Content-Type": "application/json"}

    bad_bodies = [
        b'{itemName: laptop, quantity: 1}',        # unquoted keys
        b'{"itemName": "laptop", "quantity":}',    # syntax error
        b'not_json_at_all',
        b'',                                        # empty body
        b'{"itemName": null, "quantity": null}',
        b'[1,2,3]',                                # array instead of object
    ]
    for body in bad_bodies:
        try:
            r = requests.post(f"{API_BASE_URL}/orders",
                              data=body, headers=headers, timeout=5)
            print(f"  /orders (malformed) -> {r.status_code}")
        except Exception as e:
            print(f"  /orders (malformed) -> ERROR: {e}")
        time.sleep(0.1)


# ── Scenario 6: Invalid order payloads ──
def invalid_orders():
    print("\n[ATTACK 6] Valid JWT but invalid order payloads")
    if not _valid_token:
        print("  [SKIP] No valid token available")
        return

    headers = {"Authorization": f"Bearer {_valid_token}"}
    bad_orders = [
        {"itemName": "",       "quantity": 1},        # empty string
        {"itemName": "   ",    "quantity": 1},         # whitespace only
        {"itemName": "laptop", "quantity": -5},        # negative
        {"itemName": "laptop", "quantity": 0},         # zero
        {"itemName": "laptop", "quantity": 1.5},       # float
        {"itemName": 12345,    "quantity": 1},          # wrong type
        {"quantity": 1},                               # missing itemName
        {"itemName": "laptop"},                        # missing quantity
    ]
    for order in bad_orders:
        r = requests.post(f"{API_BASE_URL}/orders", json=order,
                          headers=headers, timeout=5)
        err = r.json().get("error", "") if r.headers.get("content-type", "").startswith("application/json") else ""
        print(f"  /orders {order} -> {r.status_code}  {err}")
        time.sleep(0.1)


# ── Scenario 7: Rapid authenticated flood ──
def rapid_auth_flood():
    print("\n[ATTACK 7] Rapid authenticated requests to /secure (20 requests, no delay)")
    if not _valid_token:
        print("  [SKIP] No valid token available")
        return
    headers = {"Authorization": f"Bearer {_valid_token}"}
    for i in range(20):
        r = requests.get(f"{API_BASE_URL}/secure", headers=headers, timeout=5)
        print(f"  [{i+1:02d}] /secure -> {r.status_code}")


if __name__ == "__main__":
    print(f"ATTACK traffic simulation -> {API_BASE_URL}")
    print("WARNING: Authorized testing only!")
    print("=" * 55)

    try:
        _login_valid()
        rapid_flood()
        missing_jwt()
        invalid_jwt()
        brute_force_login()
        malformed_json()
        invalid_orders()
        rapid_auth_flood()
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot reach {API_BASE_URL} — is the backend running?")

    print("\n[DONE] Attack simulation complete.")
