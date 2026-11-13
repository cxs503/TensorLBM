import json, os, urllib.request, urllib.parse

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def req(path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read().decode())
    except Exception as e:
        b = getattr(e, "read", lambda: b"")()
        return "ERR", str(e) + " " + b.decode()[:500]

st, d = req("/auth/login", "POST", {"username": USER, "password": PASS})
print("login", st)
tok = None
if isinstance(d, dict):
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
open("/root/tp-agent/work/m18225/.tok", "w").write(tok or "")
print("token?", bool(tok))

for q in ["TensorLBM", "maxwell", "density ratio", "eos", "coexistence"]:
    s, dd = req("/knowledge_base/negative?q=" + urllib.parse.quote(q) + "&limit=10", token=tok)
    print("== neg", q, s)
    print(json.dumps(dd, ensure_ascii=False)[:2500])
