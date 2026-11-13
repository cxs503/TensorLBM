#!/usr/bin/env python3
"""Post the audit report payload to the platform report API (single call)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
print("login ok")
payload = json.load(open("/tmp/report_m18158.json"))
res = post("/tensor/reports", payload, tok)
print({k: res.get(k) for k in ("id", "title", "mission_id", "figures")} if isinstance(res, dict) else res)
print("keys:", list(res)[:8] if isinstance(res, dict) else type(res))