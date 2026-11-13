import json, os, sys, urllib.request, urllib.parse

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(path, data=None, token=None, method=None):
    url = BASE + path
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode())


tok = req("/auth/login", {"username": USER, "password": PASS})
tok = tok.get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(tok or "")
print("login ok", bool(tok))

q = sys.argv[1:] or ["regularised boundary condition LBM"]
for kw in q:
    for ep in ["/knowledge_base/search", "/oss_solutions/search"]:
        try:
            d = req(ep + "?keywords=" + urllib.parse.quote(kw) + "&limit=10", token=tok)
            print("==", ep, kw)
            print(json.dumps(d, ensure_ascii=False)[:2500])
        except Exception as e:
            print("ERR", ep, repr(e)[:200])