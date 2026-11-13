#!/usr/bin/env python3
"""Three-checks before start: knowledge_base capability + negative_knowledge_ledger + oss_solutions."""
import json
import os
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
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.loads(f.read().decode())


tok = req("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
print("login keys:", list(tok)[:8])
open("/tmp/tok.txt", "w").write(token)

for q in ["compressed", "porous", "neighbour", "neighbor", "openLBMPM", "sparse"]:
    for ep in ["/knowledge_base/search", "/oss_solutions/search"]:
        try:
            res = req(ep + "?q=" + q + "&limit=5", token=token)
            txt = json.dumps(res, ensure_ascii=False)
            print(f"--- {ep} q={q}: {txt[:600]}")
        except Exception as e:
            print(f"--- {ep} q={q}: ERR {e}")