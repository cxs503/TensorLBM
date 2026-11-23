#!/usr/bin/env python3
"""Query tensor platform ledgers: feature_ledger / knowledge_base / negative_knowledge."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get(path, token=None):
    req = urllib.request.Request(BASE + path, method="GET")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS}).get("access_token", "")
print("LOGIN_OK")

QUERIES = [
    "/features?domain=LBM",
    "/features?q=collision",
    "/knowledge/?q=collision",
    "/knowledge/search?q=collision",
    "/negative-knowledge?q=collision",
    "/negative_knowledge?q=collision",
    "/ledger/entries?q=collision",
    "/feature-ledger?q=collision",
]
for q in QUERIES:
    try:
        r = get(q, tok)
        s = json.dumps(r, ensure_ascii=False)
        print("GET", q, "->", s[:1500])
    except Exception as e:
        print("GET", q, "FAIL", e)
