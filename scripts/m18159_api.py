#!/usr/bin/env python3
"""Minimal helper for mission m18159 board-audit: API access + DB-backed reads.

Reads via the TensorPlatform HTTP API (DB tables are permission-gated for the
agent role). Usage:
    python3 scripts/m18159_api.py login
    python3 scripts/m18159_api.py get  <path-without-/api/v1>
    python3 scripts/m18159_api.py post <path> <json-file>
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
TOKEN_CACHE = "/tmp/m18159_token"


def _req(url, data=None, token=None, method=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:2000]


def login():
    st, d = _req(BASE + "/auth/login", {"username": USER, "password": PASS})
    if st != 200:
        print("login failed", st, d)
        sys.exit(1)
    tok = d.get("access_token") or d.get("token") or d.get("data", {}).get("access_token")
    open(TOKEN_CACHE, "w").write(tok)
    print("token saved, len", len(tok))
    return tok


def token():
    if os.path.exists(TOKEN_CACHE):
        return open(TOKEN_CACHE).read().strip()
    return login()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "login":
        login()
    elif cmd == "get":
        st, d = _req(BASE + "/" + sys.argv[2], token=token())
        print("status", st)
        print(json.dumps(d, indent=1)[:20000] if not isinstance(d, str) else d)
    elif cmd == "post":
        payload = json.load(open(sys.argv[3]))
        print(json.dumps(_req(BASE + "/" + sys.argv[2], payload, token=token()), indent=1)[:8000])