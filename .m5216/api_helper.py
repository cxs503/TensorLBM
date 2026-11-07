#!/usr/bin/env python3
"""m5216 helper: login + campaign/mission queries (read-only)."""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def req(method, path, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:  # noqa: BLE001
        return -1, {"error": str(e)}


def login():
    st, d = req("POST", "/auth/login", {"username": USER, "password": PASS})
    if st != 200:
        print("LOGIN FAIL", st, d)
        sys.exit(1)
    tok = d.get("access_token") or d.get("token")
    with open("/tmp/m5216_token", "w") as f:
        f.write(tok)
    return tok


def main():
    tok = login()
    what = sys.argv[1] if len(sys.argv) > 1 else "campaigns"
    if what == "campaigns":
        st, d = req("GET", "/tensor/campaigns", token=tok)
        print(st, json.dumps(d, ensure_ascii=False)[:4000])
    elif what == "campaign":
        cid = sys.argv[2]
        st, d = req("GET", f"/tensor/campaigns/{cid}", token=tok)
        print(st, json.dumps(d, ensure_ascii=False)[:8000])
    elif what == "nodes":
        st, d = req("GET", "/tensor/campaigns/nodes", token=tok)
        print(st, json.dumps(d, ensure_ascii=False)[:8000])
    elif what == "mission":
        st, d = req("GET", "/tensor/missions/5216", token=tok)
        print(st, json.dumps(d, ensure_ascii=False)[:8000])


if __name__ == "__main__":
    main()
