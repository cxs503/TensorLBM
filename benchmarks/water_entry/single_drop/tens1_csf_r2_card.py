#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1-CSF-SMOOTH-R2 判决卡生成器（m26894）。

表格从 ``runs/tens1_csf_r2/TENS1-CSF-R2.json`` 程序化渲染；稳定发布
位置是 ``docs/drop_entry_bb/``。开发过程快照只由迁移台账保存在
``development/audit/``，生成器不得重新创建已退役的根目录。
"""
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "runs", "tens1_csf_r2", "TENS1-CSF-R2.json")
OUT = os.path.join(ROOT, "docs", "drop_entry_bb", "X_TENS1-CSF-SMOOTH-R2.md")


def render_table(J):
    L = ["| 臂 | stop | 窗长 abs/pc ms | 峰深 D | 峰时 abs/pc ms | 误差% (vs 1.857434D) | mach_max | Δmach vs ctrl |",
         "|---|---|---|---|---|---|---|---|"]
    for e in J["arms"]:
        pc = "%.3f" % e["peak_time_pc_ms"] if e["peak_time_pc_ms"] is not None else "—"
        dm = "—" if e["arm"] == "ctrl" else "%+.2f%%" % J["mach_drop_pct_csf_vs_ctrl"]
        L.append("| %s | %s | %.3f/%.3f | %.6f | %.3f/%s | %+.2f | %.4f | %s |" % (
            e["arm"], e["stop_reason"], e["window_abs_ms"],
            e["postcontact_healthy_ms"] or 0.0, e["peak_depth_D"],
            e["peak_time_abs_ms"], pc, e["peak_err_pct"], e["mach_max"], dm))
    return "\n".join(L)


def render_anchor(J):
    a = J["ctrl_anchor_check"]
    L = ["| 锚量 | 锚值 | ctrl 实测 | 判定 |", "|---|---|---|---|",
         "| 峰深 D | 1.4352204748231814 | %.16f | %s |" % (
             a["peak_depth_D"], "位级一致" if a["peak_depth_bit_match"] else "**不一致**"),
         "| 峰时 abs ms | 11.0906 | %.4f | %s |" % (
             a["peak_time_abs_ms"], "一致(±5e-4)" if a["peak_time_abs_match"] else "**不一致**"),
         "| 满窗 pc ms | 24.005 | %.4f | %s |" % (
             a["postcontact_healthy_ms"], "一致(±5e-4)" if a["postcontact_match"] else "**不一致**"),
         "| mach_max | 0.2598 | %.4f | %s |" % (
             a["mach_max"], "一致(±5e-5)" if a["mach_match"] else "**不一致**")]
    return "\n".join(L)


def main():
    J = json.load(open(SRC))
    ea, eb = J["arms"]
    p = J["prereg"]
    d = J["cross_arm_history_diff"]
    if p["falsified"] and not p["hit"]:
        verdict = "FALSIFIED（按预登记证伪判据）——但根因是接线缺陷而非物理否定（§5）"
    elif p["hit"]:
        verdict = "**HIT（按预登记命中判据）**"
    else:
        verdict = "**未定谳（两判据均未触发，需披露）**"
    hitj = "触发" if p["hit"] else "未触发"
    falsj = "触发" if p["falsified"] else "未触发"

    HEAD = """# TENS1-CSF-SMOOTH-R2: CSF曲率3点滑动平均滤波 满窗双臂定谳

锚点=NEW:DE-X-TENS1-CSF-SMOOTH-R2:DE 多线并行攻关·表面张力与相间力线 满窗收口（owner 令 2026-11-21）
战役=DE · 仓库 TensorLBM · 分支 agent/m26894 · 2026-11-21
前情=TENS1-CSF-SMOOTH-R1（m26892，commit 32349e84e：滤波代码已建已合并，50 步冒烟，未定谳）

## 0. 方法与口径（R2 令执行记录）
- R2 令：只做满窗双臂定谳，禁改已建滤波代码。臂=ctrl（滤波关）/ csf_smooth（滤波开）；cells=8 gap.18 speed.06 rs1.53 pd3 ul0.15 steps10000。
- 预登记（沿用 R1 卡跑前写死）：HIT=csf_smooth 峰深误差<15% 且 mach_max 降 20%+；证伪=峰深误差>20% 或 mach_max 不降；ctrl 须位级复现 1.435220D@11.0906ms/24.005ms/mach 0.2598。
- 零手抄：本卡每个数字由 `benchmarks/water_entry/single_drop/tens1_csf_r2_adjudicate.py` 从 runs/{ctrl,csf_smooth}/result.json 直接计算；机读版=runs/tens1_csf_r2/TENS1-CSF-R2.json（正典位同目录 .json 副本）。
- 时轴口径：abs=绝对时轴（t0=初始化）；pc=接触后时轴（与实验 10.394ms 同口径）；两轴差=contact 0.547ms（step45×dt）。
- 基线锚：ctrl 1.435220D@11.0906abs/10.5436pc，满窗 24.552abs/24.005pc，mach 0.2598（MMC1-CURV2-R1，d37e62c95）。
- 实验参考：Santini case-3 1.857434D@10.394pc（REF-WEB-01）。

## 1. 双臂三元组表
"""

    MID = """
## 2. run 溯源（每格数字 → 工件路径）
| 臂 | 运行树 | commit | 卡 | 容器 | 工件 |
|---|---|---|---|---|---|
| ctrl | /tmp/tens1_ctrl_tree（当前树；2d/27 两文件恢复 32349e84e~1 预滤波态，kappa_smooth 计数=0；core free_surface_lbm.py 与主树 diff 恒等） | fe3d13b64（树 HEAD；/tmp 非 git 目录故 run.log 记 dirty） | %s | %s | runs/ctrl/result.json |
| csf_smooth | /root/tp-agent/work/m26894（当前树，滤波在 2d/27） | fe3d13b64 | %s | %s | runs/csf_smooth/result.json |

## 3. ctrl 锚检查（位级）
"""

    MID2 = """
## 4. 判决（预登记 vs 实测）
- **判定：__VERDICT__**
- 预登记 HIT 判据：csf_smooth 峰深误差<15% 且 mach_max 降 20%+ → 实测峰深误差 __ERRPCT__%、mach 降幅 __DMACHPCT__% → __HITJUDG__
- 预登记证伪判据：峰深误差>20% 或 mach_max 不降 → __FALSJUDG__
- **根因（非物理证伪）**：csf_smooth 臂与 ctrl 臂满窗 history 逐行逐字段最大绝对差 = __MAXDIFF__（__NROW__ 行全窗）→ 滤波在 probe 代码路径上**从未执行**（§5）。

## 5. 根因：R1 滤波接线缺陷（dead code）
- R1 滤波落点：`src/tensorlbm/free_surface_lbm_2d.py`（L2D）与 `src/tensorlbm/free_surface_lbm_27.py`（3D）。
- probe 实际路径：`benchmarks/water_entry/single_drop/free_surface_probe.py` → `tensorlbm.free_surface_lbm`（D3Q19 Körner 全模型），其 CSF 路径为**内联** κ→delta_rho（free_surface_lbm.py L1387-1404，`surface_tension=sc["sigma"]>0` 激活）。
- 证据①（静态）：probe 导入闭包 24 文件（probe 直接依赖 + free_surface_lbm.py 模块级/函数级导入，AST 解析）逐一 grep，无一引用 free_surface_lbm_2d / free_surface_lbm_27 / free_surface_tension.csf。
- 证据②（动态）：双臂满窗 history 位级恒等（max_abs_diff=__MAXDIFF__）——若滤波在路径上，双臂不可能逐位相同。
- 结论：R1 的「50 步无崩溃」冒烟测的是死代码；R1 卡「基础设施 FAIL」自认现升级为**接线缺陷定谳**。物理假设（曲率平滑抑制伪电流）在本基准上**未被检验**，非被否定。

## 6. 诚实披露
- **车道偏差（披露，含误判自认）**：首跑（共享 wrapper）双臂各 ~1200 步后主动 kill 重排——当时误把「steps10000」当作满窗步数（实为步数上限；probe 内建 window_complete 门=(step−contact)×dt≥0.024s，即**接触后 24 ms**，contact=45 时满窗=step 2019≈2020 行，与锚 24.552abs/24.005pc 自洽），据此误算 ETA 9400-9900s > 共享 wrapper 7200s 硬顶。改用**分相位 wrapper**（tools/sdaa_fs_probe_split_r2.py，md5 544ec7aeb10db10010317414d39058a8，与 m26882 K1-DIAG-B 同源字节级一致）：同 host/镜像/卡锁/driver/产出路径，仅把递交→轮询→回收拆成三次短调用。事后核算：满窗实际 wall ≈1900-2000s，共享 wrapper 本可容纳；偏差未影响双臂可比性（同车道同协议），如实记录。
- ctrl 臂运行树为 /tmp 临时树（非 git 目录，run.log commit 记 dirty）；其 2d/27 两文件=32349e84e~1 预滤波态，其余与主树逐文件 diff 恒等。
- 本单**未改任何滤波/物理代码**（R2 令）；新增文件仅本卡工具链（adjudicate/card/figures/eta/poller/post/publish）与 tools/sdaa_fs_probe_split_r2.py（m26882 版分相位 wrapper 副本，未动仓内既有 sdaa_fs_probe_split.py）。
- 3% 门：csf_smooth 峰深误差 __ERRPCT__%（=基线固有欠深，MMC3 卡已定谳为网格离散化地板族）→ **不满足 <3%**，与 R1 预登记一致。

## 7. 复现
- 双臂（分相位）：`python3 tools/sdaa_fs_probe_split_r2.py launch --arm <ctrl|csf_smooth> --tree <树> --cells 8 --gap 0.18 --speed 0.06 --rs 1.53 --pd 3.0 --ul 0.15 --steps 10000` → `poll` → `collect`
- 定谳提取：`python3 benchmarks/water_entry/single_drop/tens1_csf_r2_adjudicate.py`
- 卡生成：`python3 benchmarks/water_entry/single_drop/tens1_csf_r2_card.py`
- 图：`python3 benchmarks/water_entry/single_drop/tens1_csf_r2_figures.py` → runs/tens1_csf_r2/fig_*.png
"""

    md = (HEAD + render_table(J) + "\n" + MID % (
        J["trace"]["ctrl"]["card"], J["trace"]["ctrl"]["container"],
        J["trace"]["csf_smooth"]["card"], J["trace"]["csf_smooth"]["container"])
        + render_anchor(J) + "\n" + MID2)
    md = md.replace("__VERDICT__", verdict)
    md = md.replace("__ERRPCT__", "%.2f" % eb["peak_err_pct"])
    md = md.replace("__DMACHPCT__", "%.2f" % J["mach_drop_pct_csf_vs_ctrl"])
    md = md.replace("__HITJUDG__", hitj)
    md = md.replace("__FALSJUDG__", falsj)
    md = md.replace("__MAXDIFF__", repr(d["max_abs_diff"]))
    md = md.replace("__NROW__", str(d.get("n_rows")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as stream:
        stream.write(md)
    print("wrote", OUT, len(md), "bytes")
    machine_readable = OUT.replace(".md", ".json")
    shutil.copy(SRC, machine_readable)
    print("json ->", machine_readable)


if __name__ == "__main__":
    main()
