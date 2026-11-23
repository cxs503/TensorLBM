#!/usr/bin/env python3
"""Debug login response shape."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

req = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({"username": USER, "password": PASS}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
        print("STATUS", r.status)
        print("BODY", body[:800])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:500])
