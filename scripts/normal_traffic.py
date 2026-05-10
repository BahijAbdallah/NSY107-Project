"""
normal_traffic.py
Simulates normal, human-like API usage:
  - valid login -> JWT -> /secure -> /orders with realistic payloads
  - delays between requests to mimic a real user

Change API_BASE_URL to your API Gateway URL or local backend address.
"""

import requests
import time
import random

# ── Change this to your API Gateway URL or http://localhost:3000 ──
API_BASE_URL = "https://3fuaxiet0j.execute-api.us-east-1.amazonaws.com"

USERS = [
     {"username": "admin", "password": "123456"},
]

ITEMS = ["laptop", "mouse", "keyboard", "monitor", "headset", "webcam", "desk", "chair"]


def login(user):
    """POST /login and return JWT token, or None on failure."""
    try:
        r = requests.post(f"{API_BASE_URL}/login", json=user, timeout=10)
        print(f"  [LOGIN]  {user['username']} -> HTTP {r.status_code}")
        if r.status_code == 200:
            return r.json().get("token")
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR]  login failed: {e}")
    return None


def call_public():
    r = requests.get(f"{API_BASE_URL}/public", timeout=10)
    print(f"  [PUBLIC] -> HTTP {r.status_code}")


def call_secure(token):
    r = requests.get(
        f"{API_BASE_URL}/secure",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )
    print(f"  [SECURE] -> HTTP {r.status_code}")


def place_order(token):
    payload = {"itemName": random.choice(ITEMS), "quantity": random.randint(1, 10)}
    r = requests.post(
        f"{API_BASE_URL}/orders",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )
    print(f"  [ORDER]  {payload} -> HTTP {r.status_code}")


def run_session(user):
    """One full normal user session."""
    token = login(user)
    if not token:
        return

    time.sleep(random.uniform(0.4, 1.2))
    call_public()
    time.sleep(random.uniform(0.3, 0.8))
    call_secure(token)
    time.sleep(random.uniform(0.5, 1.5))

    for _ in range(random.randint(1, 3)):
        place_order(token)
        time.sleep(random.uniform(0.5, 2.0))


if __name__ == "__main__":
    print(f"Normal traffic simulation -> {API_BASE_URL}")
    print("=" * 55)

    for cycle in range(20):
        user = random.choice(USERS)
        print(f"\n--- Session {cycle + 1} / 20  (user: {user['username']}) ---")
        try:
            run_session(user)
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot reach {API_BASE_URL} — is the backend running?")
            break
        time.sleep(random.uniform(1.0, 3.0))

    print("\n[DONE] Normal traffic simulation complete.")
