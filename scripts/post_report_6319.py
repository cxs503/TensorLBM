#!/usr/bin/env python3
"""POST /tmp/report.json to the tensor platform (writes /tmp/report_post.log)."""
import json, os, urllib.request, urllib.error

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, r.read().decode()[:1200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:1200]


st, body = post("/auth/login", {"username": user, "password": pw})
print("login", st)
tok = ""
try:
    d = json.loads(body)
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token", "")
except Exception:
    pass
if not tok:
    print("no token:", body)
    raise SystemExit(1)

payload = json.loads(open("/tmp/report.json").read())
st, body = post("/tensor/reports", payload, tok)
print("report", st, body)
open("/tmp/report_post.log", "w").write(f"{st}\n{body}\n")