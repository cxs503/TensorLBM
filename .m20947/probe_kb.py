#!/usr/bin/env python3
"""m20947 开工三查：negative_knowledge_ledger(TensorLBM) + oss_inventory(MACHINE) + oss_solutions 写入口探测。"""
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
token = tok.get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(token or "")
print("login ok:", bool(token))

for kw in ["TensorLBM", "job-status", "checkpoint", "MF-LBM", "wall-clock"]:
    for ep in ["/knowledge_base/negative_ledger", "/knowledge_base/search"]:
        try:
            res = req(ep + "?" + ("keywords" if "negative" in ep else "q") + "=" + urllib.parse.quote(kw) + "&limit=4", token=token)
            s = json.dumps(res, ensure_ascii=False)
            print(f"--- {ep} kw={kw}: {s[:600]}")
        except Exception as e:  # noqa: BLE001
            print(f"--- {ep} kw={kw}: ERR {e}")

for q in ["job status checkpoint", "MACHINE", "checkpoint restart"]:
    for ep in ["/oss_solutions/search", "/oss_inventory/search"]:
        try:
            res = req(ep + "?q=" + urllib.parse.quote(q) + "&limit=4", token=token)
            print(f"--- {ep} q={q}: {json.dumps(res, ensure_ascii=False)[:700]}")
        except Exception as e:  # noqa: BLE001
            print(f"--- {ep} q={q}: ERR {e}")

# openapi: find oss_solutions write paths
try:
    with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as f:
        d = json.load(f)
    for p, v in d.get("paths", {}).items():
        if "oss" in p or "negative" in p:
            print("PATH", p, list(v.keys()))
except Exception as e:  # noqa: BLE001
    print("openapi ERR", e)