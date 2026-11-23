#!/usr/bin/env python3
"""Publish the mission report to the TensorPlatform reports API."""

import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
body = json.load(open("/tmp/report_m21300.json"))
res = post("/tensor/reports", body, tok)
print(json.dumps(res)[:600])
