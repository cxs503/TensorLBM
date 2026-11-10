"""Publish the P269W2-N04 report to the Tensor API (no secrets printed)."""

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(url, payload=None, token=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, r.read().decode()


status, body = call(f"{BASE}/auth/login", {"username": USER, "password": PASS})
token = json.loads(body)["access_token"]
print("login", status)

payload = json.load(open(os.environ.get("P269_PAYLOAD", "/tmp/report_p269w2_n04.json")))
status, body = call(f"{BASE}/tensor/reports", payload, token)
print("report post", status, body[:400])