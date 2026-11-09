import json
import os
import time
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

rep = json.load(open("/tmp/report.json"))
stamp = time.strftime("%Y-%m-%d %H:%M:%S")
rep["markdown"] = rep["markdown"] + f"\n\n---\n*报告生成时间：{stamp}；工作树 `agent/m7292`（TensorLBM），提交 22c03a1a。*\n"
json.dump(rep, open("/tmp/report_post.json", "w"))

req = urllib.request.Request(
    f"{BASE}/auth/login",
    data=json.dumps({"username": user, "password": pw}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=20) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
print("token ok", bool(tok))

payload = json.dumps(rep).encode()
req = urllib.request.Request(
    f"{BASE}/tensor/reports", data=payload,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        print("posted:", resp.status, resp.read()[:400])
except Exception as exc:  # noqa: BLE001
    print("post failed:", type(exc).__name__, str(exc)[:200])
    try:
        print(exc.read()[:400])
    except Exception:  # noqa: BLE001
        pass