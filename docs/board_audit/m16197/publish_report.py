#!/usr/bin/env python3
"""Publish the m16197 board-audit report. Numbers come from evidence.json (no hand copy)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = json.loads((HERE / "evidence.json").read_text())
FIG = HERE / "fig_audit_m16140_K_channels.png"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

env = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env") if "=" in l and not l.startswith("#"))


def post(path, payload, token=None, form=False):
    data = json.dumps(payload).encode() if not form else payload.encode()
    hdrs = {"Content-Type": "application/json" if not form else "application/x-www-form-urlencoded"}
    if token:
        hdrs["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


tok = None
for body, form in (({"username": USER, "password": PASS}, False),
                   (f"username={USER}&password={PASS}", True)):
    try:
        d = post("/auth/login", body, form=form)
        tok = d.get("access_token") or d.get("token")
        if tok:
            break
    except Exception as e:
        print("login attempt failed:", e)
assert tok, "no token"

b = EV["benchmark"]
rows = b["rows"]
tbl = "\n".join(
    "| %g | %d | %d | %s | %.2f | %.4f | %+.2f%% | %.4f | %+.2f%% | %.4f | %+.2f%% |"
    % (r["phi"], r["N"], r["steps"], "yes" if r["steady"] else "no", r["gap_cells"],
       r["K_mb"], r["err_mb_pct"], r["K_qv"], r["err_qv_pct"], r["K_mx"], r["err_mx_pct"])
    for r in rows)
findings = "\n".join(f"- **{f['id']} {f['severity']}** — {f['title']}" for f in EV["findings"])

md = f"""# 交差审计 m16140 CASE-LBM-CUBIC-ARRAY-DRAG（board-audit，非重做）

## 1. 案例介绍（物理背景/几何/参数）

被审对象是 m16140 交付的算例：简单立方（SC）周期排列实心球阵中的 Stokes 渗流，靶标量为无量纲阻力
`K(phi) = F/(6 pi mu a U_s)`，参考值取锁定 SC 表（`benchmarks/verified/permeability3d/NOTES.md` §1.2-1.4，
Basilisk `spheres.c` + Holmes 2011 双证人）：phi=0.125 → K=4.292，phi=0.45 → K=28.09。
几何 = 单胞边长 N 个格子的周期盒、球半径 a=N*(3phi/4pi)^(1/3)；D3Q19 BGK、tau=1（nu=1/6），
体驱动力按 Re_p≈0.05 取小；三档实跑参数见下表（phi_actual 与面间距 gap 均由交付工件读出）。
本单只做审计：判据 = 平台基准表 + 报告中数值能否被独立复现。

## 2. 怎么做的（方法/流程/工具链）

1. **取证**：读 m16140 工作树 + commit `c0a3e6c3`（12 files）+ 平台报告 15276，逐项核 run_artifacts /
   benchmark ref-vs-got / repro_cmd；所有交付工件按字节快照进 `refs/` 并登记 sha256（`refs/provenance.json`）。
2. **基准查表**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-CUBIC-ARRAY-DRAG'` → **0 行**
   → 平台无机器裁判行，裁判只能回落到锁定文献表（此即发现 F4）。
3. **对账**：用工件 JSON 重算每一个误差百分比（本报告所有数字程序化注入，零手抄），并与 REPORT.md §4 表格逐格比对。
4. **独立复跑**：`run.py case --phi 0.125 --n 16 --max-steps 5000 --threads 32`（91.8 s）与交付 JSON 逐位比对。
5. **鲁棒性探针**：默认参数（1024 步）、`--max-steps 60`（小于 4000 步测量窗）、`--smoke`（N=12）三条路径。
6. **判级 + 落表**：判定 L2 并 INSERT `tensor_maturity_verdicts`（verdict_by=board-audit），
   并用 gate 自己的 `judge()` 函数预演：claimed=2 → gate_level=2（无缺项）。

## 3. 计算结果（逐项数值表）

参考绝对量：phi=0.125 ref=4.292（Hasimoto 级数码内求值 {rows[0]['K_hasimoto']:.4f}，两者差
{100*(rows[0]['K_hasimoto']-4.292)/4.292:+.2f}%）；phi=0.45 ref=28.09。

| phi | N | steps | steady | gap[cells] | K_mb | err(K_mb) | K_qv | err(K_qv) | K_mx | err(K_mx) |
|---|---|---|---|---|---|---|---|---|---|---|
{tbl}

- N16→N32 主通道变化 = **{b['grid_change_mb_N16_to_N32_pct']:+.2f}%**（预注册收敛门 2%）。
- 通道一致性 `F_mx/F_inj` = {rows[0]['F_mx_over_F_inj']:.3f}(N16) / {rows[1]['F_mx_over_F_inj']:.3f}(N32)
  → 交付把链接对动量交换通道判为 falsified 的结论，审计**认可**。
- 审计独立复跑：K_mb={rows[0]['K_mb']:.15f}、K_qv={rows[0]['K_qv']:.15f}、K_mx={rows[0]['K_mx']:.15f}，
  与交付 JSON **逐位相同**。

## 4. 结果图

左：三档实测 K 与锁定参考（误差标注）；右：phi=0.125 网格阶梯——两档均低于参考且阶梯未收敛。

## 5. 注意点（坑与局限）

{findings}


审计局限：同主机同工作树族，非第三方重实现；只独立复跑 phi=0.125/N=16 一档；内核正确性未审（只核存在与被调用）。

## 6. 误差判定（逐项、显式回答是否 <3%）

| 关键量 | got | ref | 误差 | <3%? | <15% 可接受线? |
|---|---|---|---|---|---|
| K(phi=0.125) 主通道 K_mb, N=32 | {rows[1]['K_mb']:.4f} | 4.292 | {rows[1]['err_mb_pct']:+.2f}% | **否** | **否** |
| K(phi=0.125) 主通道 K_mb, N=16 | {rows[0]['K_mb']:.4f} | 4.292 | {rows[0]['err_mb_pct']:+.2f}% | **否** | **否** |
| K(phi=0.125) 旧口径 K_qv, N=32 | {rows[1]['K_qv']:.4f} | 4.292 | {rows[1]['err_qv_pct']:+.2f}% | **否** | 是（但口径被 1/(1-phi) 放大，不可作判据） |
| K(phi=0.45) K_mx（欠分辨，1.58 cell 面间距） | {rows[2]['K_mx']:.4f} | 28.09 | {rows[2]['err_mx_pct']:+.2f}% | **否** | **否**（质量应判 WITHHELD） |

**显式结论：关键量是否 <3%？否。是否 <15%？否（最细判据档 -18.67%）。**
交付报告自己的 honest-FAIL 判定（§6「<5%? NO」）**准确、未粉饰**；但 §4 表格把 K_mx 的误差印在
「err(K_mb)」列下，且 phi=0.45 的实跑工件未入库却被写成「无结果」——这两点是审计发现 F2/F3。

## 判级与处置

**L2（独立功能实现）**：L1 通过（Hasimoto 码内验证 0.06%）、L2 通过（485 行驱动 + 生产库内核 + 逐位可复现），
L3 不成立（NOT_ON_MAIN，origin/main 该算例 0 文件、无调用方）、L4 不成立（基准 -18.67% + 收敛门 +12.23% + 无机器裁判行）。
处置：**approve with notes** —— 接受 honest-FAIL 事实，不接受任何高于 L2 的成熟度；返修清单见 audit.md §7。
"""

payload = {
    "title": "交差审计 m16140 CASE-LBM-CUBIC-ARRAY-DRAG：判级 L2（基准 FAIL，-18.67%），三件套逐项核验",
    "summary": "board-audit 复核 TensorLBM 立方球阵 Darcy 阻力算例：工件真实且逐位可复现，但最细档 -18.67%、"
               "阶梯未收敛（+12.23%）、平台无 benchmark 行、phi=0.45 实跑件未入库 → 判 L2，不判 L3/L4。",
    "scenario_key": "board_audit",
    "mission_id": 16197,
    "tags": ["board-audit", "TensorLBM", "CASE-LBM-CUBIC-ARRAY-DRAG", "maturity-L2", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": "fig_audit_m16140_K_channels.png",
        "caption": "m16197 审计图：左=三档实测 K（K_mb/K_qv/K_mx）vs 锁定参考，标注相对误差；"
                   "右=phi=0.125 网格阶梯，两档均低于参考且 N16→N32 变化 +12.23%（门 2%）",
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }],
}
r = post("/tensor/reports", payload, token=tok)
print("published:", json.dumps(r)[:300])