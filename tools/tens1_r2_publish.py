#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1 R2 报告发布（m26894）：数字从 runs/tens1_csf_r2/TENS1-CSF-R2.json 程序化注入，图 base64 内嵌。"""
import base64
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
J = json.load(open(os.path.join(ROOT, "runs", "tens1_csf_r2", "TENS1-CSF-R2.json")))
ea, eb = J["arms"]
d = J["cross_arm_history_diff"]
a = J["ctrl_anchor_check"]
p = J["prereg"]

TABLE = """| 臂 | stop | 窗长 abs/pc ms | 峰深 D | 峰时 abs/pc ms | 误差%% (vs 1.857434D) | mach_max | Δmach vs ctrl |
|---|---|---|---|---|---|---|---|
| ctrl | %s | %.3f/%.3f | %.6f | %.4f/%.4f | %+.2f | %.4f | — |
| csf_smooth | %s | %.3f/%.3f | %.6f | %.4f/%.4f | %+.2f | %.4f | %+.2f%% |""" % (
    ea["stop_reason"], ea["window_abs_ms"], ea["postcontact_healthy_ms"], ea["peak_depth_D"],
    ea["peak_time_abs_ms"], ea["peak_time_pc_ms"], ea["peak_err_pct"], ea["mach_max"],
    eb["stop_reason"], eb["window_abs_ms"], eb["postcontact_healthy_ms"], eb["peak_depth_D"],
    eb["peak_time_abs_ms"], eb["peak_time_pc_ms"], eb["peak_err_pct"], eb["mach_max"],
    J["mach_drop_pct_csf_vs_ctrl"])

ANCHOR = ("峰深 %.16f D（位级一致=%s）；峰时 %.4f ms（锚 11.0906）；满窗 pc %.4f ms（锚 24.005）；"
          "mach %.4f（锚 0.2598）→ **ctrl 位级复现基线锚**" % (
              a["peak_depth_D"], a["peak_depth_bit_match"], a["peak_time_abs_ms"],
              a["postcontact_healthy_ms"], a["mach_max"]))

MD = """# TENS1-CSF-SMOOTH-R2: CSF 曲率 3 点滑动平均滤波满窗双臂定谳（FALSIFIED·接线缺陷）

## ① 案例介绍（物理背景 / 几何 / 参数）

**物理背景**：液滴入水（water entry）是空穴动力学与冲击载荷的经典基准。Santini et al. (2001) case-3 实验（REF-WEB-01）给出空穴峰深 1.857434 D @ 10.394 ms（接触后时轴）。本战役（DE 多线并行攻关·表面张力与相间力线）的基线（Körner D3Q19 自由表面 LBM，cells=8）峰深 1.435220 D，**固有欠深 22.73%**（MMC3 全臂差分卡已定谳：缺口不在一阶界面动量交换、不在可调参数，唯一变深入口是网格离散化，渐近地板 8.6–14.3%）。

**本单假设（R1 提出，R2 定谳）**：CSF 模型中曲率 κ 的数值噪声是伪电流（parasitic currents）主因（Brackbill et al. 1992, JCP 100, 335）；对 κ 做 3 点滑动平均滤波可抑制界面非物理速度场，从而 mach_max 降 20%+ 且峰深误差 <15%。

**几何/参数**（Santini case-3，程序化提取自 benchmarks/water_entry/single_drop/protocol.json）：

| 量 | 值 |
|---|---|
| 液滴直径 D | 2.27 mm（aspect 0.96） |
| 入水速度 | 1.4 m/s |
| ρ_w / ρ_a | 997 / 1.17 kg/m³ |
| μ_w / μ_a | 0.853e-3 / 1.86e-5 Pa·s |
| σ | 71.42 mN/m（σ_lat=4.637e-4，CSF 力激活但量级小） |
| g | 9.81 m/s² |
| 网格 | cells=8（dx=2.8375e-4 m），span=4D，pool_depth=3D，gap=0.18D |
| 时标 | speed=0.06 → dt=1.2161e-5 s；满窗=接触后 24 ms（probe 内建 window_complete 门），contact=45 → step 2019（2020 行，24.552 ms abs） |
| 健康门 | mach≤0.3，volume_drift≤1e-4 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

- **双臂设计（唯一变量=滤波开关）**：
  - `ctrl`：当前树（fe3d13b64）中 `free_surface_lbm_2d.py`/`free_surface_lbm_27.py` 恢复为 32349e84e~1 预滤波态（kappa_smooth 计数=0），其余文件与主树 diff 恒等；
  - `csf_smooth`：当前树原样（R1 滤波在 2d/27 两文件）。
- **车道**：SDAA（太初卡）车道。首跑走共享 wrapper `/root/tp-agent/cluster/sdaa_fs_probe.py`，双臂各 ~1200 步后主动 kill 重排（当时误把「steps10000」当满窗步数算出 ETA>7200s 硬顶；事后核算满窗实际 wall≈1900–2000 s，共享 wrapper 本可容纳——误判如实披露）。重排改用**分相位 wrapper**（tools/sdaa_fs_probe_split.py，md5 544ec7aeb10db10010317414d39058a8，与 m26882 K1-DIAG-B 字节级一致；同 host/镜像/卡锁/driver/产出路径，仅拆 launch/poll/collect 三次短调用）。
- **预登记（跑前写死，沿用 R1 卡）**：HIT=csf_smooth 峰深误差<15% 且 mach_max 降 20%+；证伪=峰深误差>20% 或 mach_max 不降；ctrl 须位级复现 1.435220D@11.0906ms/24.005ms/mach 0.2598。
- **零手抄**：全部数字由 `tens1_csf_r2_adjudicate.py` 从 runs/{ctrl,csf_smooth}/result.json 计算（机读 runs/tens1_csf_r2/TENS1-CSF-R2.json）；判决卡=BB/X/TENS1-CSF-SMOOTH-R2.md（共享正典位+树内双写）。

## ③ 计算结果（数字表，程序化注入）

__TABLE__

**ctrl 锚检查（位级）**：__ANCHOR__

**双臂 history 逐行逐字段最大绝对差**：__MAXDIFF__（__NROW__ 行全窗）→ 双臂**位级恒等**。

**判决**：预登记证伪判据**触发**（峰深误差 -22.73% > 20% 且 mach 降幅 0.00% 不降）→ **FALSIFIED（按预登记）**。但根因不是物理否定：滤波代码不在 probe 代码路径上（§⑤-1），物理假设在本基准上**未被检验**。

## ④ 结果图

- 图 1：空穴深度 vs 时间（ctrl vs csf_smooth vs 实验峰 1.857434D）——两臂曲线完全重合（滤波未执行）。
- 图 2：最大格点 Mach vs 时间（伪电流指标）——两臂完全重合。
- **无场量云图的原因（声明）**：满窗 probe 按车道 driver 固定 `planes_path=None`（不采平面快照），工件仅含标量 history；本单为定谳单（禁改代码），未加采场开关，故无云图，以时间序列比较曲线替代。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **接线缺陷（本单核心发现）**：R1 滤波落在 `free_surface_lbm_2d/27.py`，而 probe 实际路径是 `tensorlbm.free_surface_lbm`（D3Q19，内联 κ→delta_rho，L1387–1404）。静态证据：probe 导入闭包 24 文件（AST 解析）无一引用 2d/27/csf.py；动态证据：双臂满窗 history 位级恒等（max_abs_diff=0.0）。R1 的「50 步无崩溃」冒烟测的是死代码。
2. **满窗口径**：「steps10000」是步数上限，非窗长；probe 内建 window_complete 门=(step−contact)×dt≥0.024 s（接触后 24 ms）。本单首跑因此误算 ETA 并多花一次 kill/重排（~20 min）。
3. **σ_lat≈4.6e-4**：cells=8 下 CSF 力本身是小量，即便滤波接对，其效应也预期远小于 22.73% 欠深缺口——物理先验上该假设命中概率低（MMC3 方向学：一阶界面力改动系统性变浅）。
4. **节点负载**：SDAA 节点（128 核）负载 55–75 时 pace≈0.93 s/step（满窗 ~1900 s）；负载低时 ~0.19 s/step（历史满窗 1918–1979 s）。
5. 本单未改任何物理/滤波代码（R2 令）；判决对「滤波接错模块」这一事实成立，对「滤波物理效果」不构成否定。R3 建议：把滤波移植到 free_surface_lbm.py 的 κ 路径（L1387–1404）后再做满窗双臂。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | ctrl | csf_smooth | 误差 | <3%? |
|---|---|---|---|---|---|
| 峰深 | 1.857434 D（实验） | 1.435220 D | 1.435220 D | -22.73% | **否**（基线固有欠深，MMC3 已定谳为网格离散化地板族） |
| 峰时 | 10.394 pc ms（实验） | 10.5436 ms | 10.5436 ms | +0.150 ms | 在基线 38 步平台带内（MMC3 §4②），非独立判据 |
| mach_max | 无实验基准（声明：实验未提供） | 0.2598 | 0.2598 | Δ=0.00% | 不适用（无基准） |
| ctrl 锚复现 | 1.435220D@11.0906/24.005/0.2598 | 全同 | — | 位级一致 | 是 |

**显式回答**：峰深误差 -22.73% **不满足 <3%**（也不满足预登记 HIT 的 <15%）；mach_max 降幅 0.00% **不满足降 20%+**（双臂位级恒等）。**honest-FAIL**：预登记证伪判据触发，根因=接线缺陷（滤波 dead code）而非物理否定。
"""
MD = (MD.replace("__TABLE__", TABLE)
        .replace("__ANCHOR__", ANCHOR)
        .replace("__MAXDIFF__", repr(d["max_abs_diff"]))
        .replace("__NROW__", str(d["n_rows"])))

figs = []
for name, cap in (
    ("fig_depth.png", "图1 空穴深度 vs 时间（pc 口径）：ctrl（蓝实线）与 csf_smooth（红虚线）位级重合；黑点线=实验峰 1.857434D@10.394pc。两臂重合=滤波未在 probe 路径执行。"),
    ("fig_mach.png", "图2 最大格点 Mach vs 时间：两臂完全重合（mach_max=0.2598，降幅 0.00%）。"),
):
    b64 = base64.b64encode(open(os.path.join(ROOT, "runs", "tens1_csf_r2", name), "rb").read()).decode()
    figs.append({"name": name, "caption": cap, "png_base64": b64})

report = {
    "title": "TENS1-CSF-SMOOTH-R2: CSF曲率3点滑动平均滤波满窗双臂定谳——FALSIFIED（接线缺陷，非物理否定）",
    "summary": "满窗双臂（ctrl vs csf_smooth，cells=8，SDAA 车道）：ctrl 位级复现基线锚 1.435220D@11.0906ms/24.005ms/mach 0.2598；csf_smooth 与 ctrl 满窗 history 位级恒等（2020 行 max_abs_diff=0.0）→ 预登记证伪判据触发。根因：R1 滤波落在 free_surface_lbm_2d/27.py，而 probe 实际路径是 tensorlbm.free_surface_lbm（D3Q19 内联 κ 路径）——滤波从未执行。物理假设未被检验，R3 应移植到正确模块后重审。",
    "scenario_key": "DE-X-TENS1-CSF-SMOOTH-R2",
    "mission_id": 26894,
    "tags": ["TensorLBM", "water-entry", "CSF", "surface-tension", "free-surface", "SDAA", "adjudication", "falsified"],
    "markdown": MD,
    "figures": figs,
}
json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("report.json ready:", len(MD), "chars,", len(figs), "figures")
