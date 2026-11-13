#!/usr/bin/env python3
"""m18213 交差核验探针：读取 m17990 的 mission 记录 / 报告 / 基准用例。"""
import json
import os
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:  # noqa
        return None, str(e)


def get(path, token=None):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw[:2000]
    except Exception as e:  # noqa
        return None, str(e)


st, tok = post("/auth/login", {"username": USER, "password": PWD})
token = tok["access_token"]
print("login", st)
json.dump({"access_token": token}, open("/tmp/tok.json", "w"))

for p in ["/tensor/missions/17990", "/tensor/dev-missions/17990", "/missions/17990",
          "/tensor/missions?mission_id=17990", "/tensor/benchmark/cases?anchor_slug=XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE"]:
    st, d = get(p, token)
    print("==", p, st)
    s = json.dumps(d, ensure_ascii=False) if not isinstance(d, str) else d
    print(s[:2500])