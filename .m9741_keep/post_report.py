import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PWD = os.environ.get("TENSOR_AGENT_PASS", "")
DATA = json.dumps({"username": USER, "password": PWD}).encode()
req = urllib.request.Request(BASE.rstrip("/") + "/auth/login", data=DATA,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    login = json.loads(resp.read().decode())
token = (login.get("access_token") or login.get("token")
         or (login.get("data") or {}).get("access_token"))
print("login ok, token length", len(token or ""))

body = open("/tmp/report.json", "rb").read()
req = urllib.request.Request(BASE.rstrip("/") + "/tensor/reports", data=body,
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + (token or "")})
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        print("report", resp.status, resp.read().decode()[:400])
except urllib.error.HTTPError as exc:
    print("report FAILED", exc.code, exc.read().decode()[:600])