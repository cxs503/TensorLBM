import json, os, urllib.request, urllib.parse

def get(url, tok=None, timeout=10):
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

# capability KB for TensorLBM
st, d = get(base + "/knowledge/search?" + urllib.parse.urlencode({"query": "", "domain": "TensorLBM", "limit": 50}), tok)
print("== capability/domain TensorLBM:", d.get("total"))
for it in d.get("items", []):
    print("-", it.get("kb_key") or it.get("key"), "|", it.get("title"))
    c = (it.get("content") or "")[:200].replace("\n", " ")
    print("   ", c)

# negative knowledge ledger
for path in ["negative-knowledge/search", "negative-knowledge", "negative_knowledge_ledger"]:
    try:
        st, d = get(base + "/" + path + "?" + urllib.parse.urlencode({"keywords": "TensorLBM"}), tok)
        print("==", path, st, str(d)[:400])
        break
    except Exception as e:
        print("==", path, "ERR", str(e)[:100])
