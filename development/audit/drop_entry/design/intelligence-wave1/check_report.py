#!/usr/bin/env python3
"""Inspect the report stored under mission 26724 (is it the INT W1 report?)."""
import json, os, urllib.request, urllib.error

base = os.environ.get("TENSOR_API_BASE", "")

def req(method, path, payload=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

st, body = req("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                       "password": os.environ["TENSOR_AGENT_PASS"]})
token = json.loads(body).get("access_token", "")
st, body = req("GET", "/tensor/reports?limit=200&mission_id=26724", None, token)
d = json.loads(body)
items = d if isinstance(d, list) else (d.get("items") or d.get("data") or d.get("reports") or [])
for it in items:
    print("id:", it.get("id"))
    print("mission_id:", it.get("mission_id"))
    print("title:", it.get("title"))
    print("scenario_key:", it.get("scenario_key"))
    print("created:", it.get("created_at") or it.get("created"))
    md = it.get("markdown") or ""
    print("md_len:", len(md))
    print("md_head:", md[:200].replace("\n", " | "))
    print("figures:", len(it.get("figures") or []))
