#!/usr/bin/env python3
"""m18353: publish the audit report (numbers injected from the audit JSON, not hand-copied)."""
from __future__ import annotations

import base64
import json
import os
import sys

HERE = "/root/tp-agent/work/m18353"
OUT = HERE + "/docs/board_audit/m18353"
sys.path.insert(0, HERE + "/.audit_tmp")
import tapi  # noqa: E402

ev = json.load(open(OUT + "/audit_m18236.json"))
pr = json.load(open(OUT + "/probe_nan_onset.json"))
b = ev["benchmark"]
png = base64.b64encode(open(OUT + "/fig_m18353_m18236_audit.png", "rb").read()).decode()

md = f"""# 交差审计 m18236 — XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT（TensorLBM）判 L{ev['delivered_level']}

**席位**：board-audit（交差核验，非重做）· **审计单**：m18353 · **被审单**：m18236（succeeded）
· **平台报告**：tensor_reports id=17163 · **机器裁定**：maturity_gate 预演 gate_level=L{ev['delivered_level']}（无 fail reason）

## ① 案例介绍（物理背景 / 几何 / 参数）

被审能力 = 把 `je-santos/MPLBM-UT :: quasi-static-pc-ladder-with-checkpoint` 的**协议**移植到 TensorLBM 的
CAC2D 二相求解器上：把驱动毛细压力按 `num_pc_steps` 个 rung 阶梯抬升（每 rung 的台阶 = 最小喉道进气压力
`ΔP = 2σcosθ/r_min`），每 rung 按「平均速度相对变化率 < conv（%/iteration）」判稳，**rung 间写全状态检查点**
（`(f, g, φ, u_prev)` + SHA-256）以便长阶梯断点续算。

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| ρ_l / ρ_g | 1.0 / 0.1 | 界面宽 W | 3.0（W/r_min = 0.375） |
| σ | 0.05 | 接触角 θ | 60°（`wall_phi_policy="contact"`） |
| 网格 | 44 × 128（D2Q9） | 通道半宽 R(x) | R_wide=20 → r_min=8，喉道 x=78 ± 16 |
| 液塞 | x∈[26, 56]（L={56-26}） | rung/gate | 6 rung；conv 1e-4（另有 1e-2 臂）；it_conv=50, it_max=200 |
| 驱动 | 体力 g = ΔP/(Δρ·L)，周期性环流 | 验收量 | 逐 rung `σ(κ_lead−κ_trail)` vs `Δρ g L` |

## ② 怎么做的（审计方法）

1. **取证**：读 m18236 工作树 5 个交付路径（628 行 `pc_ladder.py`、91 行测试、313 行 xval runner、
   24 个检查点目录、162 行文档）+ 平台报告 17163，逐项核「run_artifacts / ref-vs-got / repro_cmd」三件套。
2. **单测复核**：测试文件沙箱副本 + `PYTHONPATH=<m18236>/src` 复跑（被审树零写入）→ **4 passed in 6.29 s**。
3. **工件复现**：把 `run_xval.py` 复制到审计沙箱重跑（rc=0，80.6 s），与交付 `results.json` 做 683 个标量逐值比对
   → {ev['determinism']['identical_including_nan']} 个相同（NaN==NaN），{683-ev['determinism']['identical_including_nan']} 个差异全是
   `wall_time_s` 与绝对 `checkpoint.path` 字符串。
4. **检查点完整性**：用 `load_ladder_state(verify_digest=True)` 恢复交付 `ckpt/main/rung_000` 与 `rung_005` → 摘要校验通过。
5. **基准裁判**：`tensor_benchmark_cases` 按 anchor_slug 查询 → **{b['registered_cases_for_anchor']} 行**（全表 {b['tensor_benchmark_cases_total']} 行，TensorLBM {b['tensor_benchmark_cases_tensorlbm']} 行）。
6. **对照实验（被审方没做的那个）**：用同一 `build()` 在 **g=0**（不施加任何压力）与 **g=rung1=1.929e-5** 下各演化 600 步，
   记录首次非有限步 —— 判定液塞解体是否由压力 ramp 触发。

## ③ 计算结果（数字表，全部程序化注入自 audit_m18236.json / probe_nan_onset.json）

| 量 | ref / 判据 | got | 误差 | 判定 |
|---|---|---|---|---|
| 曲率 κ（静态液滴 R_eq=20） | 1/R_eq = {pr['e0']['kappa_from_Req']:.6f} | {pr['e0']['kappa_band']:.6f} | **{b['E0_kappa_vs_1_over_Req_pct']:.3f}%** | ✅ <3% |
| Laplace 压差 Δp=σκ | σκ = {pr['e0']['sigma_kappa']:.3e} | 场压 p = {pr['e0']['dp_field']:.3e} | **{b['E0_laplace_pressure_pct']:.2f}%** | ❌ >15%（σ_eff={b['E0_sigma_effective_from_dp']:.4f} vs 输入 {b['E0_sigma_input']}） |
| E1 rung 0 的 Y-L 残差 | — | {b['E1_rung0_relerr_divide_by_zero']:.3e} | — | 除零产物（Δρ g L=0），无意义 |
| E1 rung 1–6 | <15% | x_lead=nan, κ=nan | nan | **无度量** |
| E3 单发（全压一次加） | <15% | nan | nan | **无度量** |
| E4 rungs4 | <15% | nan | nan | **无度量** |
| E4 conv1e-2 rung1 / rung2 | <15% | — | **{b['E4_conv1e-2_rung1_rung2_relerr'][0]*100:.1f}% / {b['E4_conv1e-2_rung1_rung2_relerr'][1]*100:.1f}%** | ❌ |
| 检查点摘要 | 逐位一致 | rung0/rung5 verify 通过（rung5 step=1200，f 已非有限） | 0 | ✅ 完整性 |
| 续算等价（E2） | resume≈连续 | `TypeError: 'RungRecord' object is not subscriptable` | — | ❌ 未度量 |
| 注册基准 | ≥1 行 | {b['registered_cases_for_anchor']} 行 | — | ❌ 裁判缺席 |

## ④ 结果图

左：E0 测量链两条路线的相对误差 vs 15% owner 门与 3% 生态门（曲率 0.05% 过，压力场 38.36% 不过）。
右：被审方未做的对照实验 —— 液塞在量化锥形通道上的 `u_max` 演化；**无外加压力**（g=0）时 step {pr['trace_g0']['first_nonfinite_step']}
即发散，施加 rung1 压力后 step {pr['trace_g1']['first_nonfinite_step']}（仅差 10 步）。

![audit](fig)

## ⑤ 注意点（坑与局限）

1. **因果归因被否证**（F2，high）：报告把液塞解体归给「接触角壁面在压力 ramp 下把伪流抬到 1.3e-2」；
   对照实验显示 **g=0 时 step {pr['trace_g0']['first_nonfinite_step']} 已发散**，加压力只提前 10 步（≈2.6%）。
   返修必须先稳住液塞初值/壁面模型，否则任何 rung 排程都无效。
2. **『with-checkpoint』的核心不变量无度量**（F3）：E2 在交付脚本内崩溃；被拿来替代的单测只断言
   rung id / 摘要长度 / 可 load / 篡改抛错，**不断言**其 docstring 声称的「resume 复现连续轨迹」。
   已验证的只是检查点写读与 SHA-256 完整性。
3. **除零未防护**（F4）：`young_laplace_residual` 在 ΔρgL=0 时返回 7.3e25 而非 nan，该数字被写进报告对照表。
4. **溯源瑕疵**（F5）：交付 `run.log` 是修 bug 前的崩溃日志（13:18:34），`results.json` 来自 13:27 修复后的运行（13:28:48）——
   日志不对应数字；且 5 个交付路径全 untracked、不在 main，工件只在 m18236 工作树可达。
5. 局限：同主机同解释器复跑，未写第二实现；OSS `ShanChen.cpp` 行号未独立重读；未上集群（单核 81 s）。
   未修复发散（越界，属返修轮）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

* 曲率 κ vs 1/R_eq：**{b['E0_kappa_vs_1_over_Req_pct']:.3f}% → 是（<3%，也 <15%）**。
* Laplace 压差（压力路线）：**{b['E0_laplace_pressure_pct']:.2f}% → 否（>15%，也 >3%）**，该路线按报告建议退役。
* 逐 rung Young-Laplace 残差（本锚点验收量）：**无法判定 —— rung1-6 全 NaN，无数字可判**（不粉饰）。
* E4 conv1e-2 唯一可测 rung：**{b['E4_conv1e-2_rung1_rung2_relerr'][0]*100:.1f}% / {b['E4_conv1e-2_rung1_rung2_relerr'][1]*100:.1f}% → 否**。
* 续算等价：**未度量**（E2 脚本缺陷），不宣称通过。
* 基准裁判：**缺席**（本锚点 tensor_benchmark_cases 0 行）。

## 判级与落表

**L{ev['delivered_level']}**：L1 ✅（4/4 单测复跑）· L2 ✅（628 行真代码 + 可执行入口 + 5 类工件 + 物理数值位级可复现）·
L3 ❌（零调用方 / 未导出 / 不在 main）· L4 ❌（注册基准 0 行 + 验收量 NaN）· L5 ❌。
落表：`tensor_maturity_verdicts` id=741（repo=TensorLBM, scope=m18236 XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT,
mission_id=18353, level=2, verdict_by=board-audit）；maturity_gate 预演裁定 L2（无 fail reason）。
处置：**approve with notes**（honest-FAIL 的协议交付），返修须补：g=0 液塞稳定性、resume⇒连续轨迹等价断言、本锚点基准注册行。
"""

body = {"title": "交差审计 m18236 — XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT 判 L2（协议移植成立·物理 NaN 复现确认·归因被否证）",
        "summary": ("board-audit 交差核验 m18236：4/4 单测复跑通过、683 个交付标量复现 657 个逐值相同、"
                    "检查点 SHA-256 完整性通过；判 L2（无调用方 ⇒ 非 L3；tensor_benchmark_cases 本锚 0 行且验收量 NaN ⇒ 非 L4）。"
                    "对照实验否证其失败归因：g=0 时液塞 step 380 自行发散，加 rung1 压力仅提前到 step 390。"),
        "scenario_key": "audit_m18236_quasi_static_pc_ladder",
        "mission_id": 18353,
        "tags": ["audit", "board-audit", "TensorLBM", "m18236", "quasi-static-pc-ladder", "checkpoint", "L2", "honest-FAIL", "falsified-attribution"],
        "markdown": md,
        "figures": [{"name": "fig_m18353_m18236_audit.png",
                     "caption": "左：E0 测量链误差 vs 15%/3% 门；右：g=0 与 g=rung1 下液塞 u_max 演化（无压力也发散）",
                     "png_base64": png}]}

st, d = tapi.call("/tensor/reports", "POST", body)
print("report POST", st, str(d)[:300])