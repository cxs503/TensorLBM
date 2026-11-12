import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


login = post(
    "/auth/login",
    {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    },
)
tok = login.get("access_token") or login.get("token")
print("login keys:", list(login)[:8], "token?", bool(tok))
open("/tmp/tok.txt", "w").write(tok or "")


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


spec = get("/openapi.json")
paths = list(spec.get("paths", {})) if isinstance(spec, dict) else []
print("n paths", len(paths))
for p in sorted(paths):
    if any(k in p for k in ("knowledge", "oss", "ledger", "report", "negative")):
        print(" ", p)