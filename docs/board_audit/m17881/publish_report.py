import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")


def call(method, path, payload=None, token=None):
    req = urllib.request.Request(base + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = call("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                   "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
body = json.load(open("/tmp/report_m17881.json"))
out = call("POST", "/tensor/reports", body, token=tok)
print({k: out[k] for k in ("id", "title", "mission_id", "qa_status") if k in out})