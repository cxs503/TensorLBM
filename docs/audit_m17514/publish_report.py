#!/usr/bin/env python3
"""Publish the m17572 audit report (paper-grade six sections + 3 figures)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
body = json.load(open("/tmp/report.json"))


def post(url, payload, token=None):
    req = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read() or b"{}")


tok = post(BASE + "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
out = post(BASE + "/tensor/reports", body, tok)
print(json.dumps({k: out.get(k) for k in ("id", "title", "mission_id", "status")}, ensure_ascii=False))