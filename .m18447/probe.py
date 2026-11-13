import json, os, urllib.request, urllib.parse

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PWD = os.environ.get("TENSOR_AGENT_PASS", "")


def req(path, body=None, token=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode())


tok = req("/auth/login", {"username": USER, "password": PWD})
token = tok.get("access_token") or tok.get("token") or ""
print("token ok", bool(token))

queries = [
    "/tensor/knowledge_base?q=codegen&limit=8",
    "/tensor/knowledge_base?q=sympy&limit=8",
    "/tensor/knowledge_base?q=pylbm&limit=8",
    "/tensor/negative_knowledge_ledger?keywords=TensorLBM&limit=8",
    "/tensor/oss_solutions?q=pylbm&limit=8",
    "/tensor/oss_solutions?q=codegen&limit=8",
    "/tensor/capabilities?domain=TensorLBM&limit=10",
]
for path in queries:
    try:
        d = req(path, token=token)
        s = json.dumps(d, ensure_ascii=False)[:900]
        print("OK", path, "\n  ", s, "\n")
    except Exception as e:
        print("ERR", path, repr(e)[:200], "\n")
