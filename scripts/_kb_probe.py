import json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")


def req(path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.status, json.loads(f.read().decode())
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        return "ERR", str(e) + " " + body.decode()[:400]


st, d = req("/auth/login", "POST", {"username": USER, "password": PASS})
print("login", st)
tok = None
if isinstance(d, dict):
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
print("token?", bool(tok))
if tok:
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tok"), "w").write(tok)
    for p in ["/knowledge_base/capabilities?q=free_surface",
              "/knowledge_base/search?q=fill%20level%20reconstruct",
              "/oss_solutions?q=walberla"]:
        s, dd = req(p, token=tok)
        print("==", p, s)
        print(json.dumps(dd, ensure_ascii=False)[:2000])