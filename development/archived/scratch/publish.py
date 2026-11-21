import base64, json, os, urllib.request, pathlib

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[:400]
    except Exception as exc:
        body = ""
        if hasattr(exc, "read"):
            body = exc.read().decode()[:300]
        return getattr(exc, "code", "ERR"), f"{exc} {body}"


tok = None
for path in ("/auth/login", "/login"):
    st, body = post(BASE + path, {"username": USER, "password": PASS})
    print(path, st, str(body)[:120])
    if st == 200:
        try:
            d = json.loads(body if isinstance(body, str) and body.startswith("{") else body)
        except Exception:
            d = None
        tok = None
    if st == 200:
        break

# re-login cleanly to capture the full body
st, body = post(BASE + "/auth/login", {"username": USER, "password": PASS})
print("login", st, body[:200])
