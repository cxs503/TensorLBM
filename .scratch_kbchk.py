import os, json, urllib.request, urllib.parse

base = "http://127.0.0.1:8010/api/v1"
env = {}
for line in open("/root/tp-agent/agent.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k] = v

def call(path, body=None, token=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

tok = call("/auth/login", {"username": env["TENSOR_AGENT_USER"], "password": env["TENSOR_AGENT_PASS"]})
t = tok.get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(t or "")
print("login ok:", bool(t))

for path in [
    "/knowledge_base/capabilities?keyword=TensorLBM",
    "/knowledge_base/negative_knowledge_ledger?keyword=TensorLBM",
    "/knowledge_base/oss_solutions?keyword=coexistence",
    "/knowledge_base/oss_solutions?keyword=LBM",
]:
    r = call(path, token=t)
    s = json.dumps(r, ensure_ascii=False)
    print("===", path, "len", len(s))
    print(s[:1500])