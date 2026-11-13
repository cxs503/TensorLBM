#!/usr/bin/env python3
"""m18068: find correct KB endpoints via openapi."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


tok = post("/auth/login", {"username": USER, "password": PASS}).get("access_token", "")
try:
    spec = get("/openapi.json", tok)
    paths = sorted(spec.get("paths", {}).keys())
    for p in paths:
        if any(k in p for k in ("knowledge", "oss", "ledger", "capab", "solution", "negative")):
            print(p)
except Exception as e:
    print("openapi ERR", e)
