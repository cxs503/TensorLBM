"""Publish the m17664 board-audit report (mission m17603 grading) to /tensor/reports."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path("/root/tp-agent/work/m17664")


def post(path, payload, token=None):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]

md = (HERE / "docs/board_audit/m17664/AUDIT_m17664.md").read_text()
fig = HERE / "docs/board_audit/m17664/fig_audit_m17664.png"
rep = {
    "title": "m17664 交差核验：m17603 XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE 判级 L2",
    "summary": (
        "独立复核 m17603（Inamuro 湿节点 vs lbm-principles-practice OSS）：重编上游 film_inamuro.cpp 重跑，"
        "4 个 checkpoint 参考场 sha256 与归档逐一吻合；经生产模块重跑 12000 步 rel L2=3.17e-11/5.12e-11/"
        "4.36e-11/4.22e-11，与归档逐项差 0.0；控制探针（删闭包）→0.9967 证明鉴别力；pytest 14 passed。"
        "判级 L2：L3 无集成调用点（未进 boundary_registry），L4 被三道门挡住——本锚裁判行 0 行、"
        "anti-bounce-back 速度臂 not_run、四件交付物全 untracked 且 README 一步复现崩（FileExistsError）。"
        "处置 request_changes，verdict 已落表。"
    ),
    "scenario_key": "board_audit",
    "mission_id": 17664,
    "tags": ["audit", "TensorLBM", "inamuro", "wet-node", "anti-bounce-back",
             "xval", "maturity-gate", "cross-validation"],
    "markdown": md,
    "figures": [{
        "name": "fig_audit_m17664.png",
        "caption": ("上排：step 12000 浓度场——左=审计经生产模块重跑，中=审计重新编译并运行的上游 OSS 二进制，"
                    "右=逐点 |Δc|（log 色标，max 1.2e-10）。下排：四个 checkpoint 的 rel L2/Linf（log 轴）"
                    "与 15% 验收线相距 10 个数量级，另标控制探针（闭包删除 → rel L2 0.9967），"
                    "证明 1e-11 的通过不是宽松容差所致。全部数字来自 audit.json，图由 gen_figure.py 程序化生成。"),
        "png_base64": base64.b64encode(fig.read_bytes()).decode(),
    }],
}
try:
    res = post("/tensor/reports", rep, tok)
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read().decode()[:600])
    raise
print(json.dumps(res, ensure_ascii=False)[:400])