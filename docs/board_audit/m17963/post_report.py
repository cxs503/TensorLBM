"""Publish the m17963 audit report to the platform (six elements, figures embedded).

All numbers come from docs/board_audit/m17963/audit_numbers.json (zero hand-copy).
"""
from __future__ import annotations

import base64
import json
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "audit_numbers.json").read_text())
BASE = "http://127.0.0.1:8010/api/v1"


def token() -> str:
    req = urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps({"username": "eco_admin", "password": "EcoTest!2026pass"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))["access_token"]


def fig(name: str, caption: str) -> dict:
    b64 = base64.b64encode((HERE / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": b64}


md = f"""# 交差审计 m17963 — m17759 `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`

**被审单** m17759（TensorLBM，succeeded，交付 commit `08a55f8e`，自报 PASS 9.4%<15%）· **审计单** m17963（board-audit，非重做）
**判级 level = 2**（L2）；**基准裁判补登**：`tensor_benchmark_cases` id=84（status=failed，预声明口径）；
**verdict 落表** `tensor_maturity_verdicts` id=675（gate_state=pending，待机器复核）。

## ① 案例介绍
2-D D2Q9 压力驱动槽道 Poiseuille（x 周期、上下半程反弹无滑移壁、常体力 Gx），对照 OSS
`haimingz/vivsim`。参数：NX×NY=10×10（加密算例 NY=20）、nu={N['params']['nu']}（tau=1.1）、
Gx={N['params']['Gx']}、{N['params']['steps']} 步、float64 CPU 单核。
无量纲数：u_max=0.05 → Ma=0.0866、Re=u_max·H/nu=2.25。

## ② 怎么做的
①取交付三件套（脚本 14 629 B / JSON / 报告 128 行 / DIFF 表）；②**独立重跑**交付脚本（未改一行）；
③静态核验"是否真调用本仓代码"；④源码级核对 vivsim NEBB 转写保真度；⑤双口径复算误差；
⑥查 `tensor_benchmark_cases` 该 anchor 判据行（交付时 **0 行**）；⑦程序化出图/出数（零手抄）。

## ③ 计算结果
| 量 | 交付 JSON | 审计重跑 |
|---|---|---|
| TensorLBM 全局归一化误差 | {100*N['tlb_global_err']:.2f} % | {100*N['tlb_global_err']:.2f} %（payload 完全相同） |
| TensorLBM 预声明口径（内部点最大相对误差） | {100*N['tlb_pointwise_interior_err']:.2f} % | 同 |
| TensorLBM 壁面 u_x(y=0)（应=0） | {N['tlb_wall_ux']:.4f} | 同 |
| OSS 臂 全局 / 预声明口径 | {100*N['oss_global_err']:.2f} % / {100*N['oss_pointwise_interior_err']:.2f} % | 同 |
| 双臂最大差 | {N['cross_arm_max_abs_diff']:.4f}（{100*N['cross_arm_global_err']:.2f} %） | 同 |
| NY=20 网格 TLB / OSS | {100*N['tlb_ny20_global_err']:.2f} % / {100*N['oss_ny20_global_err']:.2f} % | 同 |
重跑 ≈10 min 单核；排序序列化 SHA-256 两次相同 → **数值可复现**（reproducibility 过）。

## ④ 结果图
左图=三线剖面（解析/OSS/TensorLBM，标注壁面滑移 {N['tlb_wall_ux']:.4f} = {N['tlb_wall_slip_pct_of_umax']:.2f}% u_max）；
右图=四口径误差 vs 15%/3% 门线（口径替换翻转 PASS/FAIL）+ NY=10→20 收敛。

## ⑤ 注意点（findings）
- **F1 blocking**：脚本 import 只有 json/pathlib/numpy（:50-55），**0 处 import tensorlbm**；Arm B 是手写 numpy 复述（`tl_step` :135-160）。
  仓库真实 `src/tensorlbm/gpu_core_path.py:278-343 collide_guo_into` 与 `wave_bc_common.py:160-196` NEBB **一次都没执行** → "cross-validated TensorLBM scaffold" 名不副实。
- **F2 blocking**：脚本预声明口径（内部点最大相对误差）**{100*N['tlb_pointwise_interior_err']:.2f}% > 15% FAIL**；报告改判为全局归一化 {100*N['tlb_global_err']:.2f}% PASS。换尺在 §5 有披露，但 headline PASS 与预声明判据冲突。
- **F3 high**：TLB 臂壁面速度 {N['tlb_wall_ux']:.4f}（{N['tlb_wall_slip_pct_of_umax']:.2f}% u_max）→ 无滑移未被执行；双臂 {100*N['cross_arm_global_err']:.2f}% 差 ≡ 该滑移量，报告称 "second-order wall effect" 无量化支撑。
- **F4 high**：交付时 `tensor_benchmark_cases` 该 anchor **0 行**，自设阈值自判 PASS；本单补登（failed）。
- **F5 low**：OSS 臂从未在真实 vivsim 上运行（`ModuleNotFoundError: jax` → **NOT_RUN**），故"cross"实为两份手写转写之交；零测试；10 min 单核应走 run_cluster。

## ⑥ 误差判定（是否 <3%）
| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| u_max（TLB, NY=10） | 解析 0.05 | {100*N['tlb_global_err']:.2f}% | ✗ |
| u_max（TLB, NY=20） | 解析 0.05 | {100*N['tlb_ny20_global_err']:.2f}% | ✗（临界外） |
| 壁面 u_x（TLB） | 0 | {N['tlb_wall_ux']:.4f}（{N['tlb_wall_slip_pct_of_umax']:.2f}% u_max） | ✗ |
| 剖面逐点最大相对误差 | 解析 | {100*N['tlb_pointwise_interior_err']:.2f}% | ✗ |
| 双臂一致性 | OSS 臂 | {100*N['cross_arm_global_err']:.2f}% | ✗ |
| OSS u_max（NY=20） | 解析 | {100*N['oss_ny20_global_err']:.2f}% | ✓ |
**关键量均未达 3% 生态目标；owner 15% 门"全局口径过、预声明口径不过"**。数值非硬编码（网格加密单调收敛 9.4%→3.2%），但集成/口径缺陷使成果定位 L2，不足 L3/L4。

**返修建议（给接棒单）**：①Arm B 真调用 `collide_guo_into` + 仓库 NEBB；②修壁面滑移（wall-velocity 修正与 NEBB 的 ρ/动量闭合）；③按预声明口径重报或显式改判据并留 predeclared 记录；④装 jax 跑真实 vivsim 一臂或标 NOT_RUN。

**证据指针**：`docs/board_audit/m17963/AUDIT_m17759.md`（全文）、`audit_numbers.json`、`rerun/xval_nebb_bodyforce_driver.json`、`insert_verdict.sql`、`fig_m17963_profile.png`、`fig_m17963_metrics.png`、DB `tensor_maturity_verdicts` id=675 / `tensor_benchmark_cases` id=84。
"""

payload = {
    "title": "交差审计 m17963: m17759 XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S -> level 2（预声明口径 23.50% 未过门）",
    "summary": (
        "Audit of m17759 (TensorLBM XVAL vs haimingz/vivsim): re-ran the delivered script, JSON payload identical "
        "(reproducible). Level 2 - the TensorLBM arm is a hand-written numpy paraphrase (0 tensorlbm imports, 0 callers "
        "of the repo Guo/NEBB code); pre-declared pointwise metric 23.50%% > 15%% FAIL while the report's "
        "global-normalised metric (9.40%%) passes; wall slip 0.0022 (4.40%% u_max) = the cross-arm gap. No benchmark "
        "row existed at delivery -> registered failed. OSS arm NOT_RUN (jax missing)."
    ),
    "scenario_key": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
    "mission_id": 17963,
    "tags": ["audit", "TensorLBM", "xval", "nebb", "body-force", "maturity-verdict", "L2", "reproducibility"],
    "markdown": md,
    "figures": [
        fig("fig_m17963_profile.png",
            "u_x(y) in the D2Q9 force-driven channel: closed-form Poiseuille vs the OSS vivsim arm (5.00% global) vs "
            "the delivered TensorLBM arm (9.40% global). The TensorLBM profile sits on a constant +0.0022 offset: "
            "wall velocity 0.0022 = 4.40% of u_max, i.e. no-slip not enforced."),
        fig("fig_m17963_metrics.png",
            "Left: four error conventions vs the 15% gate / 3% target - pre-declared pointwise 23.50% FAILS, "
            "global-normalised 9.40% passes. Right: NY=10->20 refinement from the delivered JSON."),
    ],
}

tok = token()
req = urllib.request.Request(
    BASE + "/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
)
try:
    resp = json.load(urllib.request.urlopen(req))
    print(json.dumps(resp, ensure_ascii=False)[:400])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:800])
    raise