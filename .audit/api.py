import json, os, sys, urllib.request, urllib.parse
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", ""); pwd = os.environ.get("TENSOR_AGENT_PASS", "")
def _req(path, method="GET", body=None, token=None):
    h = {"Content-Type": "application/json"}
    if token: h["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as r: raw = r.read().decode()
    except urllib.error.HTTPError as e:
        return {"__status": e.code, "__body": e.read().decode()[:2000]}
    try: return json.loads(raw)
    except Exception: return {"__raw": raw[:3000]}
def login():
    d = _req("/auth/login", "POST", {"username": user, "password": pwd})
    return d.get("access_token") or d.get("token") or d.get("__body")
TOKEN = None
def get(path, params=None):
    global TOKEN
    if TOKEN is None: TOKEN = login()
    if params: path = path + "?" + urllib.parse.urlencode(params)
    return _req(path, "GET", None, TOKEN)
def post(path, body):
    global TOKEN
    if TOKEN is None: TOKEN = login()
    return _req(path, "POST", body, TOKEN)
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "get": print(json.dumps(get(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None), ensure_ascii=False)[:8000])
    elif cmd == "post": print(json.dumps(post(sys.argv[2], json.loads(sys.argv[3])), ensure_ascii=False)[:8000])
    elif cmd == "paths":
        if TOKEN is None: TOKEN = login()
        d = _req("/openapi.json","GET",None,TOKEN)
        for p in sorted(d.get("paths",{})):
            if any(k in p for k in ("benchmark","verdict","report","maturity")): print(p)
