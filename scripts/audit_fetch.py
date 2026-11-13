#!/usr/bin/env python3
"""Audit helper (m17668): fetch mission evidence for XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER.

Usage: python3 scripts/audit_fetch.py mission <id>
       python3 scripts/audit_fetch.py get <path>
       python3 scripts/audit_fetch.py reports <mission_id>
       python3 scripts/audit_fetch.py sql "<SELECT ...>"
Env: TENSOR_API_BASE, TENSOR_AGENT_USER, TENSOR_AGENT_PASS
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(path, body=None, method=None):
    url = BASE + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    tok = _token()
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:2000]}


_CACHE = {}


def _token():
    if "t" in _CACHE:
        return _CACHE["t"]
    req = urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps({"username": USER, "password": PASS}).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("LOGIN FAIL", e.code, e.read().decode()[:500], file=sys.stderr)
        _CACHE["t"] = None
        return None
    tok = d.get("access_token") or d.get("token") or d.get("accessToken")
    _CACHE["t"] = tok
    return tok


def main():
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    if cmd == "mission":
        out = call("missions/" + arg)
    elif cmd == "get":
        out = call(arg)
    elif cmd == "reports":
        out = call("tensor/reports?mission_id=" + arg)
    elif cmd == "sql":
        out = call("tensor/query", {"sql": arg})
    else:
        out = {"err": "unknown cmd"}
    print(json.dumps(out, ensure_ascii=False, indent=1)[:20000])


if __name__ == "__main__":
    main()