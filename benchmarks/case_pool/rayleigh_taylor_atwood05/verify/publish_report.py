"""Publish the mission report for CASE-RT-INSTABILITY-ATWOOD (proposer seat)."""

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
rep = json.load(open("/tmp/report.json", encoding="utf-8"))
out = post("/tensor/reports", rep, token)
print(
    json.dumps(
        {k: out.get(k) for k in ("id", "report_id", "status", "title", "ok")}, ensure_ascii=False
    )
)
print(str(out)[:400])
