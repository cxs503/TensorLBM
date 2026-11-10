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

for path in ("/tensor/reports?mission_id=9741&limit=5",
             "/tensor/reports?limit=3"):
    req = urllib.request.Request(BASE.rstrip("/") + path,
                                 headers={"Authorization": "Bearer " + (token or "")})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
        print(path, "->", body[:700])
    except urllib.error.HTTPError as exc:
        print(path, "HTTP", exc.code, exc.read().decode()[:200])