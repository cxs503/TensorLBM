#!/usr/bin/env python3
"""Fetch the published K1-CAVCLOS-R2 report record (verify title/figures)."""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def call(method, path, token=None):
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


st, body = call("POST", "/auth/login",
                {"username": os.environ["TENSOR_AGENT_USER"],
                 "password": os.environ["TENSOR_AGENT_PASS"]})
tok = json.loads(body).get("access_token", "")
print("login:", st)
for p in ["/tensor/reports?mission_id=27075",
          "/tensor/reports?limit=3",
          "/tensor/reports"]:
    st2, body2 = call("GET", p, tok)
    print(p, "->", st2, body2[:200].replace("\n", " "))
    if st2 == 200:
        try:
            d = json.loads(body2)
            items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
            for it in items[:6]:
                print("   -", it.get("id"), "|", (it.get("title") or "")[:60],
                      "| figs:", len(it.get("figures") or []), "|", it.get("created_at"))
        except Exception as e:
            print("   parse err:", e)
        break
