#!/usr/bin/env python
"""Submit report to Tensor API."""
import json
import urllib.request
import urllib.error

# Login
login_url = "http://127.0.0.1:8010/api/v1/auth/login"
login_data = json.dumps({"username": "eco_admin", "password": "EcoTest!2026pass"}).encode()
req = urllib.request.Request(login_url, data=login_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    token = json.loads(resp.read())["access_token"]
print(f"Got token: {token[:20]}...")

# Submit report
report_url = "http://127.0.0.1:8010/api/v1/tensor/reports"
with open("report_m238.json") as f:
    report_data = json.load(f)

req = urllib.request.Request(
    report_url,
    data=json.dumps(report_data).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    method="POST"
)
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Report submitted: {result}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
