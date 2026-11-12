"""Post the mission report to the platform (login -> POST /tensor/reports).

Kept out of the committed tree? no: it is a small reusable helper placed under
scripts/ so future rounds can re-post reports without re-deriving the routes.
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def _req(url, payload=None, tok=None, method="POST"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"null")


def login():
    body = {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    for path in ("/tensor/auth/login", "/login", "/auth/login"):
        try:
            d = _req(BASE + path, body)
            tok = (d or {}).get("access_token") or (d or {}).get("token")
            if tok:
                print("login ok:", path)
                return tok
        except Exception as e:
            print("login fail", path, e)
    raise SystemExit("login failed")


def main(report_path: str) -> None:
    tok = login()
    payload = json.load(open(report_path))
    d = _req(BASE + "/tensor/reports", payload, tok)
    print(json.dumps(d)[:600])


if __name__ == "__main__":
    main(sys.argv[1])