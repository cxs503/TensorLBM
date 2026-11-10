import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PWD = os.environ.get("TENSOR_AGENT_PASS", "")


def token():
    data = json.dumps({"username": USER, "password": PWD}).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/auth/login", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        login = json.loads(resp.read().decode())
    return (login.get("access_token") or login.get("token")
            or (login.get("data") or {}).get("access_token"))


tok = token()
rep = json.loads(open("/tmp/report.json").read())
# keep the payload small enough for a single request: figures are optional in the
# API call, the png files live in the repository under artifacts/invariance
small = dict(rep)
small["figures"] = []
body = json.dumps(small).encode()
print("payload KB", len(body) // 1024)
req = urllib.request.Request(BASE.rstrip("/") + "/tensor/reports", data=body,
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        print("POST", resp.status, resp.read().decode()[:300])
except urllib.error.HTTPError as exc:
    print("POST failed", exc.code, exc.read().decode()[:400])

req = urllib.request.Request(BASE.rstrip("/") + "/tensor/reports?mission_id=9741&limit=5",
                             headers={"Authorization": "Bearer " + tok})
with urllib.request.urlopen(req, timeout=120) as resp:
    print("verify", resp.read().decode()[:400])