import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


st, d = call("/auth/login", {"username": USER, "password": PASS})
tok = d.get("access_token", "")
print("login", st, "token_len", len(tok))
payload = json.load(open("xval/cupy_dropin_no_kernel_fusion/artifacts/report_payload.json"))
st2, out = call("/tensor/reports", payload, tok)
print("report", st2, json.dumps(out)[:400])