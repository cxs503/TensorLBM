#!/usr/bin/env python3
"""Publish board-audit report for m18152 (audit of m17992)."""
import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def post(path, payload, token=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()[:400]

# login: try json body first, then form
try:
    st, body = post("/auth/login", {"username": user, "password": pw})
    token = json.loads(body).get("token", "")
except Exception:
    import urllib.parse
    data = urllib.parse.urlencode({"username": user, "password": pw}).encode()
    req = urllib.request.Request(base + "/auth/login", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read().decode()).get("token", "")

if not token:
    raise SystemExit("login failed")

st, body = post("/tensor/reports", json.load(open("/tmp/report.json")), token)
print("status:", st)
print(body)
