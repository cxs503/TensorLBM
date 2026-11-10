"""Post the campaign report to the platform API (env credentials, no secrets on the CLI)."""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


def main():
    payload_path = sys.argv[1] if len(sys.argv) > 1 else (
        "artifacts/pseudopotential_edm_hr_upgrade/report_payload.json"
    )
    with open(payload_path) as fh:
        payload = json.load(fh)

    token = ""
    for path, body in (
        ("/auth/login", {"username": USER, "password": PASS}),
        ("/login", {"username": USER, "password": PASS}),
    ):
        try:
            st, txt = post(path, body)
            d = json.loads(txt)
            token = d.get("access_token") or d.get("token") or ""
            print("login", path, st, "token_len", len(token))
            if token:
                break
        except Exception as exc:  # noqa: BLE001
            print("login failed", path, type(exc).__name__, exc)

    try:
        st, txt = post("/tensor/reports", payload, token)
        print("report", st, txt[:600])
        return 0
    except Exception as exc:  # noqa: BLE001
        print("report POST failed:", type(exc).__name__, exc)
        if hasattr(exc, "read"):
            print(exc.read().decode()[:400])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())