"""Audit probe: read the machine referee (tensor_benchmark_cases) and L4 verdict for m15470."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def call(path, body=None, token=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except Exception as e:  # noqa: BLE001
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode()[:400]
            except Exception:  # noqa: BLE001
                detail = ""
        return getattr(e, "code", "ERR"), f"{e} {detail}"


tok = None
for cred_key in ("username", "user", "email"):
    st, d = call("/auth/login", {cred_key: USER, "password": PASS})
    if st == 200 and isinstance(d, dict):
        tok = d.get("access_token") or d.get("token")
        if tok:
            print("login ok via", cred_key)
            break
print("token:", "YES" if tok else "NO")

paths = [
    "/tensor/bench/cases?anchor_slug=CASE-MCMP-CONTACT-ANGLE",
    "/tensor/bench/l4-verdict/15470",
    "/tensor/bench/anchors?slug=CASE-MCMP-CONTACT-ANGLE",
    "/tensor/bench/results?mission_id=15470",
]
for p in paths:
    st, d = call(p, token=tok)
    print("\n==", p, "->", st)
    print(json.dumps(d, ensure_ascii=False)[:3000] if not isinstance(d, str) else d[:1500])