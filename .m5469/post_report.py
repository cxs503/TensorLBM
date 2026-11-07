"""Post the IN38-N04 report to the platform (credentials from the environment).

Kept outside the committed tree: it is a publishing utility, not a deliverable.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def request(url: str, data: dict | None = None, token: str | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.loads(fh.read().decode())


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    pw = os.environ["TENSOR_AGENT_PASS"]
    tok = request(f"{base}/auth/login", {"username": user, "password": pw})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        print("login failed:", list(tok)[:8])
        return 1
    print("login ok (token length %d)" % len(token))
    with open("/tmp/report.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    try:
        res = request(f"{base}/tensor/reports", payload, token)
    except urllib.error.HTTPError as exc:
        print("POST failed", exc.code, exc.read().decode()[:600])
        return 1
    print("posted:", json.dumps({k: res[k] for k in list(res)[:8]})[:600])
    try:
        back = request(f"{base}/tensor/reports?mission_id=5469", None, token)
        items = back if isinstance(back, list) else back.get("items", back)
        print("readback count:", len(items) if hasattr(items, "__len__") else "?")
    except Exception as exc:                     # readback is best effort
        print("readback skipped:", type(exc).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())