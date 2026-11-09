import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
req = urllib.request.Request(
    f"{BASE}/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=20) as resp:
    tok = json.loads(resp.read()).get("access_token")

req = urllib.request.Request(f"{BASE}/tensor/reports?limit=50",
                             headers={"Authorization": f"Bearer {tok}"})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read())
items = data.get("items", data if isinstance(data, list) else [])
for it in items[:50]:
    t = it.get("title", "")
    if "P66" in t or "冰区碎冰场" in t:
        print("FOUND", it.get("id"), it.get("mission_id"), t[:80])