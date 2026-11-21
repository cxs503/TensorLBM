#!/usr/bin/env python3
"""Login + publish DE-NXT-INT-W1 report."""
import json, os, urllib.request, urllib.error

base = os.environ.get("TENSOR_API_BASE", "")
user = os.environ.get("TENSOR_AGENT_USER", "")
pw = os.environ.get("TENSOR_AGENT_PASS", "")

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

payload = {"username": user, "password": pw}
token = ""
for path in ("/tensor/auth/login", "/tensor/login", "/api/v1/auth/login", "/auth/login"):
    st, body = req("POST", path, payload)
    print(path, st, body[:150].replace("\n", " "))
    if st == 200:
        try:
            d = json.loads(body)
        except Exception:
            d = {}
        token = (d.get("access_token") or d.get("token")
                 or (d.get("data") or {}).get("access_token")
                 or (d.get("data") or {}).get("token") or "")
        if token:
            break
if not token:
    raise SystemExit("login failed on all paths")
print("LOGIN_OK")

rep = json.load(open("/tmp/report.json"))
st2, body2 = req("POST", "/tensor/reports", rep, token)
print("REPORT", st2, body2[:400])
