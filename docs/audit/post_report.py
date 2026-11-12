import base64, json, os, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
md = open("docs/audit/M16323_AUDIT_REPORT.md").read()

tok = json.load(urllib.request.urlopen(urllib.request.Request(
    base + "/auth/login", data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})))["access_token"]

png = base64.b64encode(open("docs/audit/figs/m16323_audit_overview.png", "rb").read()).decode()
body = {
    "title": "m16323 交差核验：CASE-LBM-WOMERSLEY-PULSATILE 独立复核 → L2（判据 <5% 三算例全通过，集成缺失）",
    "summary": ("独立复核 m16323：以自写有限差分 ODE 参考（无 Bessel 依赖）重算 Womersley 解析解，"
                "与任务自报振幅/相位误差逐位吻合（参考差 3e-6 %），三算例 alpha=5 中心线振幅比误差 "
                "0.0386/0.0536/0.0719 %、相位误差 0.0871/0.7692/0.3121 %，全部 <5% 判据。"
                "但交付未落 commit（全部 untracked，ECO 主仓 0 命中）、无集成调用方、声明 repro_cmd 指向不存在脚本、"
                "交付测试 1 failed(KeyError 'finite') ⇒ 等级裁定 L2，不是 L4。"),
    "scenario_key": "CASE-LBM-WOMERSLEY-PULSATILE",
    "mission_id": 16338,
    "tags": ["TensorLBM", "Womersley", "audit", "maturity", "board-audit", "m16323"],
    "markdown": md,
    "figures": [{"name": "m16323_audit_overview.png",
                 "caption": "审计图：(a) 槽/管中心线相位平均波形 vs 独立 Womersley 参考；(b) 三算例振幅/相位误差 vs 5 % 判据带；(c) 管 R=16.5 径向振幅剖面 LBM vs 任务解析 vs 审计 FD 独立解。",
                 "png_base64": png}],
}
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
r = json.load(urllib.request.urlopen(req, timeout=180))
print("report id:", r.get("id"), r.get("status"))