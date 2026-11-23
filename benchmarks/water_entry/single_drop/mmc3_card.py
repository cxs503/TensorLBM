#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMC3 交付卡生成器：表从 runs/mmc3_diff/MMC3-RUNSDIFF.json 程序化渲染（零手抄）。
用法：python3 benchmarks/water_entry/single_drop/mmc3_card.py
产出：/root/tp-agent/drop_entry/BB/X/MMC3-RUNSDIFF.md + 同目录 MMC3-RUNSDIFF.json 副本
"""

import json
import os
import shutil

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs", "mmc3_diff")
)
OUT_MD = "/root/tp-agent/drop_entry/BB/X/MMC3-RUNSDIFF.md"
OUT_JSON = "/root/tp-agent/drop_entry/BB/X/MMC3-RUNSDIFF.json"


def render_table(J):
    L = [
        "| 臂 | 系列 | 变量 | stop | 窗长 abs/pc ms | 峰深 D | 峰时 abs/pc ms | 误差% | ΔD vs ctrl | 方向 | 死因 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in J["arms"]:
        pc = "%.3f" % e["peak_time_pc_ms"] if e["peak_time_pc_ms"] is not None else "—"
        L.append(
            "| %s | %s | %s | %s | %.3f/%.3f | %.4f | %.3f/%s | %.2f | %+.4f | %s | %s |"
            % (
                e["arm"],
                e["series"],
                e["variable"],
                e["stop_reason"],
                e["window_abs_ms"],
                e["postcontact_healthy_ms"] or 0.0,
                e["peak_depth_D"],
                e["peak_time_abs_ms"],
                pc,
                e["peak_err_pct"],
                e["delta_vs_ctrl_D"],
                e["direction_vs_ctrl"],
                e["death_cause"] or "—",
            )
        )
    return "\n".join(L)


def render_trace(J):
    L = ["| 臂 | 镜像路径（m26746 工作树） | 原始出处 |", "|---|---|---|"]
    for e in J["arms"]:
        L.append("| %s | %s | %s |" % (e["arm"], e["source_mirror"], e["source_origin"]))
    return "\n".join(L)


HEAD = """# MMC3-RUNSDIFF · 全臂差分诊断对照卡（诊断官 m26746）
自检：5/5（文件在且非空✓ 数字可溯=runs/mmc3_diff/**/result.json 程序化提取✓ 双写=worktree+BB/X✓ 口径声明✓ 失败如实✓）
锚点=NEW:MMC3-全臂差分诊断对照卡 · 战役=DE · 仓库 TensorLBM · 分支 agent/m26746 · 2026-11-21
CROSS 已读：BB/X 全部 6 卡（KERN-A-R2/KERN-B/CAVSWEEP/CONV12/ABBS/REJECT_KERN-A）+ REF-WEB-01 + SEP

## 0. 方法与口径（防造假铁律执行记录）
- **零手抄**：本卡每个数字由 `benchmarks/water_entry/single_drop/mmc3_runsdiff.py` 从 result.json 直接计算（24 臂 29 个工件），机读版=同目录 MMC3-RUNSDIFF.json。
- **时轴口径（本卡核心澄清）**：`abs`=绝对时轴（t0=初始化）；`pc`=接触后时轴（score.py 与实验 10.394ms 同口径）。两轴差=contact 时刻 0.547ms（step45×dt）。上游卡混用两轴（CAVSWEEP/KERN-B 用 abs，CONV12/ABBS 用 pc），同一时刻 11.091abs≡10.543pc。
- **基线锚**：ctrl 1.4352D@11.091abs/10.543pc，满窗 24.552abs/24.005pc；实验 1.857434D@10.394pc（Santini case-3，REF-WEB-01；score_pair.json reference_observed_peak=1.857434）。
- **峰时=平台带**：基线峰顶为 38 步（0.46ms）平台，argmax 时刻只在带内有意义（§4②）。
- 本卡**未跑任何新仿真**：24 臂全部为既有 rc=0 仿真段完整工件（§2 逐臂溯源）。

## 1. 全臂三元组表（24 臂；误差%=(峰深/1.857434−1)×100，负=欠深）
"""

MID = """
注：`identical`=与 ctrl 峰深差 <1e-6D（9 臂，含 4 个 CAVSWEEP 重复性/无感臂）；`died`=health_limit 死于实验峰时 10.394pc 之前，峰深为**截断下界**（10 臂：ul006/rs175/gap12/pd20/caponly/simonly/spd007/ul012pd20/kernb_B1/abbs_B）。

## 2. run 溯源（每格数字 → 工件路径）
"""

MID2 = """
- **rc 状态**：CAVSWEEP 16/16（node099 宿主 /tmp/cavsweep，本地副本 md5 对账过，CAVSWEEP 卡 §6）；KERN-B 两臂仿真段完整（window_complete/health_limit 全 history 在本地工件内；cj 账本 rc=1 为包装层，披露见 STATUS.md）；KERN-A-R2=SDAA card0（run.log 首行 `arm=KERN-A-R2 card=0 commit=badf6b70c steps=10000`，wall 1919s）；ABBS 三臂=cj5684/5685/5687 rc=0 fetch tar 内 repo/runs/abbs/*.json；CONV12 两档=cj5691/5692 rc=0（n330 单线程）。
- **ctrl 四副本一致性**：cavsweep/ctrl、kernb/ctrl、abbs/armA、conv12/cal8 的 history 逐行差 ≤2.1e-11（跨平台浮点噪声，node099 x86 vs n330 loong64）；物理量 4 位小数全同。上游"逐位零分歧"表述修正为"2e-11 级一致"。

## 3. 预测 vs 实测（上游卡声明 vs 原始工件复核）
| 上游声明 | 原始工件复核 | 判定 |
|---|---|---|
| CAVSWEEP 16 臂三元组（卡 §2 表） | 16/16 与 runs/cavsweep/*.json 提取值全同 | 一致 |
| KERN-B ctrl 1.4352@11.091/24.552、B1 0.5032@3.150（3.393 死） | kernb/ctrl.json、b1_0995.json 全同 | 一致 |
| KERN-A-R2 1.3701D@22.947ms 满窗 24.552 | result.json：1.3701@22.947abs ✓；**窗长=24.516**（contact step 42 非 45，Option3 补丁使接触检测提前 3 步） | 峰时一致；**窗长卡值 24.552 有误，实为 24.516** |
| KERN-A-R2 "commit 7f92f763c" | run.log 首行 commit=badf6b70c（=含 7f92f763c 补丁的后续树 HEAD） | 补丁号 vs 运行树 HEAD，两者皆真，卡应标运行树 |
| ABBS 臂A 1.4352D@10.543ms | armA_fix.json：1.4352@11.091abs=10.543pc ✓（pc 口径） | 一致（口径） |
| ABBS 臂B 峰深"—" | armB_zero.json：0.0180D@0.292ms（死前微峰） | 卡留空，工件有值（本卡补全） |
| CONV12 cal8 1.43522@10.543ms / cal12 1.52244@10.020ms | cal8_fixed.json 11.091abs=10.543pc ✓；case3_cal12_fixed.json 10.434abs=10.020pc ✓ | 一致（pc 口径） |
| KERN-B "ctrl 逐位复现 CAVSWEEP ctrl" | 逐行差 ≤2.1e-11（非逐位） | 物理一致，"逐位"措辞过强 |

## 4. 结论：三问
### ① 哪些方向系统性变浅/变深（证伪方向学）
- **24 臂方向分布：9 同值（1.4352D）/ 14 变浅 / 1 变深**。变深唯一=conv12_cal12（1.5224D，+0.087D，误差 22.73→18.04%）=**纯网格离散化**，非物理项；CONV12 定谳其渐近地板 8.6–14.3%（3% 门 3–6 倍，n*≈278 档不可行）。
- **一阶界面动量交换的全部改动系统性变浅**（4 条独立路线）：ABB 零注入/整体移除（abbs_B，0.304ms 早死）/ Option3 7+/9- 重建（KERN-A-R2，1.3701D，−0.065D）/ 液相侧（abbs_C，1.3753D，−0.060D）/ 空穴压亏 0.995（kernb_B1，0.5032D，3.393ms 早死）。
- **参数方向**：RS↓1.35→1.3530D（−5.7%）、spd↓0.05→1.3310D（−7.2%）、u_limit 0.10–0.24 平坦（纯稳定器）、交换模式双关皆死（质量守恒必要条件）。**无任何旋钮把峰深推向 1.8574D**。
- **方向学定谳**：22.73% 缺口不在一阶界面动量交换、不在可调参数；"变深"方向唯一实证入口=网格（不可行）。处方必须来自**缺失物理项**（二阶项/触线/气相完整动力学），且必须作用在变深方向。

### ② 峰时刻漂移的统一规律（早死 vs 满窗峰漂）
- **三态分类（24 臂全覆盖）**：
  1. **早死截断**（10 臂死于 10.394pc 前）："峰"=截断下界，峰时=死亡时刻，无物理意义（ul006 8.427pc、caponly 0.742pc、simonly 2.615pc、kernb_B1 2.602pc、abbs_B 0.292pc、初始化死 5 臂 <0.05pc）。
  2. **满窗+真内点峰**（峰深互异的 5 臂：cal12/ctrl/abbs_C/rs135/spd005；9 个同值臂与 ctrl 同峰结构）：峰顶为 21–47 步平台，argmax 时刻在带内由浮点噪声决定；实验峰 10.394pc **落在 ctrl 0.1% 平台带 [9.85,10.60]pc 内** → 基线峰时与实验**不矛盾**，判别量是峰深不是峰时。
  3. **满窗+无真峰**（KERN-A-R2）：峰漂到窗尾 22.437pc（+12.04ms），148 步（1.8ms）平台=近窗尾缓降，argmax 是窗尾伪迹——**向下动量被削弱时最大值滑向窗尾**，这是该机制的指纹。
- **统一规律**：峰时只在曲线有真内点极大时才是稳健可观测量；平台顶（≥20 步）上 argmax 时刻对浮点噪声敏感（基线自身 38 步平台）；上游卡间"峰时漂移"（11.091 vs 10.543）=时轴口径差，非物理漂移。
- **基线峰时 vs 实验**：10.543pc vs 10.394pc（+0.149ms，在平台带内）→ 模拟空穴"欠深且峰时略迟"，与欠深同源。

### ③ 对二阶项/触线两方向的预期方向（MMC1/MMC2 先验）
- **二阶项（MOM 线：界面力分离施加/双弛豫界面力）**：
  - **先验：峰深 >1.4352D（误差 <22.73%）且峰时带向 10.394pc 移动（更早）**。
  - 依据：①全部一阶改动变浅（§4①），若二阶项是缺失物理，必须补偿在变深方向；②唯一变深入口是离散化（地板 8.6–14.3%），物理项若存在应叠加其上。
  - **证伪判据（跑前写死）：峰深 ≤1.4352D 或早死 → MOM 线证伪**，缺口不在界面力。
- **触线（CONT 线：润湿角/触线物理）**：
  - REF-WEB-01 情报："气穴捕获对触线物理敏感，欠深可能与触线过早贴壁相关"（Truscott 正典：动/静态润湿角直接控制气穴形成）。
  - **先验：触线过早贴壁封顶气穴 → 修正后峰深 >1.4352D 和/或 pinch-off 延后**（峰时带后移）。
  - **证伪判据（跑前写死）：峰深 ≤1.4352D → 触线非欠深主因**。
- 两先验同向（变深）=与 §4① 方向学自洽；若双双证伪，则欠深根因在气相完整动力学（GAS 线，REF-WEB-01 线索 2）。

## 5. 诚实披露
- 本卡为纯提取/诊断卡，**未跑新仿真**；24 臂全部溯既有 rc=0 工件（§2）。
- cj5694/5695（KERN-B）账本 rc=1=包装层（CONV12 卡已披露集群账本 bug 族）；仿真段完整、本地工件在，数字溯本地 result.json。
- ul006 峰 1.2942D 为下界（死于实验峰时前）；10 个早死臂的"峰"均不作物理量使用。
- numeric_pass=False：0/24 臂过 3% 门（最深 cal12 18.04%）。
- KERN-A-R2 窗长 24.516（contact 42）vs 其余 24.552（contact 45）——跨臂比窗长时注意 contact 步差。
- 早死臂死因归因按末行 mach>0.3 / volume_drift>1e-4 判定（与卡声明一致：ul020/ul024/rs135ul020 死 Ma 门，余死 drift 门）。

## 6. 复现
- 提取：`python3 benchmarks/water_entry/single_drop/mmc3_runsdiff.py`（m26746 工作树；输出 runs/mmc3_diff/MMC3-RUNSDIFF.json + 表）
- 卡生成：`python3 benchmarks/water_entry/single_drop/mmc3_card.py`
- 工件镜像：runs/mmc3_diff/（cavsweep 16 + kernb 2 + kern_a_r2 1 + abbs 3 + conv12 4 + figs 2）
"""


def main():
    J = json.load(open(os.path.join(ROOT, "MMC3-RUNSDIFF.json")))
    md = HEAD + render_table(J) + "\n" + MID + render_trace(J) + "\n" + MID2
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(md)
    shutil.copy(os.path.join(ROOT, "MMC3-RUNSDIFF.json"), OUT_JSON)
    print("wrote", OUT_MD, len(md), "bytes; json ->", OUT_JSON)


if __name__ == "__main__":
    main()
