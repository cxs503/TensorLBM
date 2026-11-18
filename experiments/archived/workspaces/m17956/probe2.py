import json, os, urllib.request, urllib.parse

BASE = "http://127.0.0.1:8010/api/v1"


def req(path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(BASE + path, data=data, headers=h)
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.loads(resp.read().decode())


tok = req("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
print("=== status ===")
print(json.dumps(req("/tensor/knowledge/status", token=token))[:800])
for q in ["mask bounce-back", "bounceback", "vivsim", "obstacle bounce"]:
    for kind in ["capability", "negative", "oss", None]:
        p = "/tensor/knowledge/search?q=%s&limit=6" % urllib.parse.quote(q)
        if kind:
            p += "&kind=" + kind
        try:
            d = req(p, token=token)
            hits = d.get("results") or d.get("hits") or d.get("items") or d
            print("\n--- q=%s kind=%s n=%s" % (q, kind, len(hits) if isinstance(hits, list) else "?"))
            txt = json.dumps(hits)[:1500]
            print(txt)
        except Exception as e:
            print("ERR", q, kind, repr(e)[:150])