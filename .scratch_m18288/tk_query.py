import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def get(path, tok):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

tok = post("/auth/login", {"username": user, "password": pw}).get("access_token", "")
print("token_len", len(tok))

try:
    d = get("/knowledge_base?ktype=capability&domain=TensorLBM", tok)
    rows = d if isinstance(d, list) else d.get("items", d.get("data", []))
    print("== capability KB rows:", len(rows))
    for r in rows[:40]:
        print("-", r.get("kb_key"), "|", r.get("title"))
except Exception as e:
    print("kb err", e)

try:
    d = get("/negative_knowledge_ledger?keywords=TensorLBM", tok)
    rows = d if isinstance(d, list) else d.get("items", d.get("data", []))
    print("== negative ledger rows:", len(rows))
    for r in rows[:40]:
        print("-", r.get("title"), "||", str(r.get("failure_reason"))[:160])
except Exception as e:
    print("neg err", e)
