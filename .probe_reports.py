import json, os, sys, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", "")
pwd = os.environ.get("TENSOR_AGENT_PASS", "")

def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def get(path, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + path, headers=h)
    with urllib.request.urlopen(req) as r:
        return json.load(r)

tok = post("/auth/login", {"username": user, "password": pwd})
token = tok.get("access_token") or tok.get("token") or ""
rid = sys.argv[1]
try:
    d = get(f"/tensor/reports/{rid}", token)
    md = d.get("markdown") or ""
    print("LEN", len(md))
    print(md[:9000])
except Exception as e:
    print("ERR", e)
