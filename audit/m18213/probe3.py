#!/usr/bin/env python3
"""读平台报告 16909 + l4-verdict for 17990。"""
import json
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
token = json.load(open("/tmp/tok.json"))["access_token"]


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:
        return None, str(e)


for p in ["/tensor/bench/l4-verdict/17990", "/tensor/reports/16909"]:
    st, d = get(p)
    print("==", p, st)
    if isinstance(d, dict):
        d2 = {k: (v if k != "markdown" else str(v)[:1500]) for k, v in d.items() if k != "figures"}
        print(json.dumps(d2, ensure_ascii=False, indent=1)[:4000])
    else:
        print(str(d)[:800])
    json.dump(d, open("/tmp/%s.json" % p.strip("/").replace("/", "_"), "w"), ensure_ascii=False)