#!/usr/bin/env python3
"""m18195 开工三查: knowledge_base capability + negative_knowledge_ledger + oss_solutions.

Anchor: XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT
Writes /tmp/tok.txt for later report publishing.
"""
import json
import os
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(path, data=None, token=None, method=None):
    url = BASE + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.loads(f.read().decode())


tok = req("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
open("/tmp/tok.txt", "w").write(token)
print("login ok:", bool(token))

QUERIES = [
    "pressure tensor",
    "shan chen pressure",
    "multiphase equilibrium force",
    "pseudopotential",
    "laplace pressure tensor",
]

for q in QUERIES:
    for ep in ["/knowledge_base/search", "/oss_solutions/search"]:
        try:
            res = req(ep + "?q=" + urllib.parse.quote(q) + "&limit=5", token=token)
            txt = json.dumps(res, ensure_ascii=False)
            print(f"--- {ep} q={q}: {txt[:700]}")
        except Exception as e:  # noqa: BLE001
            print(f"--- {ep} q={q}: ERR {e}")

for kw in ["TensorLBM", "pressure-tensor", "shan-chen"]:
    for ep in ["/knowledge_base/negative_ledger", "/negative_knowledge_ledger/search",
               "/knowledge_base/negative_knowledge_ledger"]:
        try:
            res = req(ep + "?keywords=" + urllib.parse.quote(kw) + "&limit=5", token=token)
            print(f"--- NEG {ep} kw={kw}: {json.dumps(res, ensure_ascii=False)[:700]}")
        except Exception as e:  # noqa: BLE001
            print(f"--- NEG {ep} kw={kw}: ERR {e}")