import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


tok = post(base + "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
print("token ok:", bool(token))
rep = json.load(open("/tmp/report.json"))
print(json.dumps(post(base + "/tensor/reports", rep, token))[:600])