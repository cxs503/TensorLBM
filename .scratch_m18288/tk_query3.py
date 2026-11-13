import json, os, urllib.request, urllib.parse

def get(url, tok=None, timeout=10):
    h = {"Authorization": "Bearer " + tok} if tok else {}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]; pw = os.environ["TENSOR_AGENT_PASS"]
req = urllib.request.Request(base + "/auth/login", data=json.dumps({"username": user, "password": pw}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
tok = json.load(urllib.request.urlopen(req)).get("access_token", "")

# capability KB: try query variants
for q in ["", "LBM", "method selection", "benchmark"]:
    for dom in ["TensorLBM", None]:
        params = {"query": q, "limit": 30}
        if dom: params["domain"] = dom
        try:
            st, d = get(base + "/knowledge/search?" + urllib.parse.urlencode(params), tok)
            items = d.get("items", [])
            print(f"q={q!r} dom={dom} total={d.get('total')}")
            for it in items[:15]:
                print("   -", it.get("kb_key") or it.get("key"), "|", it.get("title"))
        except Exception as e:
            print("err", q, dom, str(e)[:80])

# negative ledger via knowledge/search with ktype
try:
    st, d = get(base + "/knowledge/search?" + urllib.parse.urlencode({"query": "TensorLBM", "ktype": "negative", "limit": 30}), tok)
    print("neg via ktype=negative total=", d.get("total"))
    for it in d.get("items", [])[:15]:
        print("   -", it.get("title"), "||", str(it.get("content"))[:120])
except Exception as e:
    print("neg err", str(e)[:100])
