#!/usr/bin/env python3
"""m18068: 开工三查 (knowledge_base / negative_knowledge_ledger / oss_solutions)."""
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
print("token_len", len(tok))

for path in [
    "/knowledge_base/capabilities?domain=TensorLBM",
    "/knowledge_base/capabilities?domain=lbm",
    "/negative_knowledge_ledger?keywords=TensorLBM",
    "/negative_knowledge_ledger?keywords=2b-t",
    "/oss_solutions?domain=TensorLBM",
    "/oss_solutions?keywords=2b-t",
]:
    try:
        data = get(path, tok)
        s = json.dumps(data, ensure_ascii=False)
        print("==", path, "len", len(s))
        print(s[:1800])
    except Exception as e:
        print("==", path, "ERR", e)
