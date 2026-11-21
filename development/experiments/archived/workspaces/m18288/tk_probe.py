import json, os, urllib.request

def get(url, tok=None, timeout=6):
    h = {"Authorization": "Bearer " + tok} if tok else {}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]
req = urllib.request.Request(base + "/auth/login", data=json.dumps({"username": user, "password": pw}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
tok = json.load(urllib.request.urlopen(req)).get("access_token", "")
print("token_len", len(tok))

# probe 8021 knowledge engine
for path in ["", "health", "search", "knowledge_base", "negative-knowledge/search"]:
    try:
        st, d = get("http://127.0.0.1:8021/" + path, tok)
        print("8021/" + path, "->", st, str(d)[:200])
    except Exception as e:
        print("8021/" + path, "ERR", str(e)[:120])

# probe 8010 api v1 paths
for path in ["knowledge", "knowledge-base", "kb", "negative-knowledge", "negative-knowledge/search",
             "tensor/knowledge", "knowledge/search", "search/knowledge"]:
    try:
        st, d = get(base + "/" + path, tok)
        print("8010/" + path, "->", st, str(d)[:200])
    except Exception as e:
        print("8010/" + path, "ERR", str(e)[:120])
