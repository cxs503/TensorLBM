"""Publish REPORT_V3 to the platform report API (credentials from the environment)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
payload = json.loads(open("_scratch/report_v3.json").read())
res = post("/tensor/reports", payload, token)
print("report id:", res.get("id"), "| title:", (res.get("title") or "")[:60])