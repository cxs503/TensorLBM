#!/usr/bin/env python3
"""读取 m17990 mission 全量结果 + 平台 openapi 路由清单。"""
import json
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
token = json.load(open("/tmp/tok.json"))["access_token"]


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:
        return None, str(e)


st, d = get("/tensor/missions/17990")
json.dump(d, open("/tmp/m17990.json", "w"), ensure_ascii=False, indent=1)
res = d.get("result") or {}
print("status", d.get("status"), "keys", list(res.keys()))
ev = res.get("evidence") or {}
print("evidence keys:", list(ev.keys()))
print(json.dumps(ev, ensure_ascii=False, indent=1)[:6000])
print("--- summary keys:", list((ev.get("summary") or {}).keys()))

st, oa = get("/openapi.json") if False else (None, None)
req = urllib.request.Request("http://127.0.0.1:8010/openapi.json")
with urllib.request.urlopen(req, timeout=60) as r:
    spec = json.loads(r.read().decode())
paths = [p for p in spec["paths"] if any(k in p for k in ("benchmark", "maturity", "verdict", "report"))]
print("--- routes:")
for p in sorted(paths):
    print(p, list(spec["paths"][p].keys()))