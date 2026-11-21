"""Login and publish the T5 solo progress report."""
import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

body = json.dumps({"username": user, "password": pw}).encode()
req = urllib.request.Request(
    base + "/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST"
)
with urllib.request.urlopen(req, timeout=30) as r:
    token = json.load(r).get("access_token", "")
print("token_len", len(token))

report = json.load(open("/tmp/report.json"))
data = json.dumps(report).encode()
req2 = urllib.request.Request(
    base + "/tensor/reports",
    data=data,
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req2, timeout=60) as r:
        print("HTTP", r.status)
        print(r.read().decode()[:600])
except urllib.error.HTTPError as e:
    print("HTTPError", e.code)
    print(e.read().decode()[:600])
