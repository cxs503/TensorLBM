"""Publish the T5 solo progress report (auth disabled -> anonymous admin)."""
import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"]
report = json.load(open("/tmp/report.json"))
data = json.dumps(report).encode()
req = urllib.request.Request(
    base + "/tensor/reports",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("HTTP", r.status)
        print(r.read().decode()[:600])
except urllib.error.HTTPError as e:
    print("HTTPError", e.code)
    print(e.read().decode()[:600])
