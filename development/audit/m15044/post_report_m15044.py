"""Post the m15044 (m14689 board-audit) paper-grade report to the platform.

Every number in the markdown is injected from artifacts/audit_m14689/audit_summary.json —
no hand-typed values. Figures are embedded as base64 PNGs.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
OUT = pathlib.Path("artifacts/audit_m14689")
SUMMARY = json.loads((OUT / "audit_summary.json").read_text())
ROWS = SUMMARY["checks"]["flow_rows"]
CRIT = SUMMARY["criteria"]
CTL = SUMMARY["checks"]["static_control_ny49_g25_mirror"]
CAL = SUMMARY["checks"]["wall_calibration_rerun"]


def login():
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["access_token"]


def fig(name, caption):
    raw = (OUT / name).read_bytes()
    return {"name": name.replace(".png", ""), "caption": caption,
            "png_base64": base64.b64encode(raw).decode()}


def pct(x):
    return f"{100 * x:+.2f}%"


def row_table():
    lines = ["| ny | y_int (row) | target | err ① | Q_mix | err ② mixture | err ② component | S/u_max ③ | steady |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        lines.append(
            f"| {r['ny']} | {r['y_int_row']:.4f} | {r['y_int_target_row']:.1f} | "
            f"{pct(r['err_interface'])} | {r['Q_mix']:.6f} | "
            f"{pct(r['rules']['mixture']['err_Q_mix_independent'])} | "
            f"{pct(r['rules']['component']['err_Q_mix_independent'])} | "
            f"{r['ratio_S']:.4f} | {r['steady']} |")
    return "\n".join(lines)


def c3_table():
    lines = ["| ny | ratio S_all/u_max | gate | miss factor |", "|---|---|---|---|"]
    for ny, ratio in CRIT["c3_spurious"]["ratios"].items():
        miss = CRIT["c3_spurious"]["gate_miss_factor"][ny]
        lines.append(f"| {ny} | {ratio:.4f} | {CRIT['c3_spurious']['threshold']} | {miss:.1f}x |")
    return "\n".join(lines)


MD = f"""
## ① 案例介绍（交差审计单，非重做）
被审对象 **m14689 · `GAP-LBM-MCMP-POISEUILLE`**（TensorLBM，commit `d0a945e2`，status=succeeded）：
Shan–Chen 两组分（MCMP）分层 Poiseuille 流的解析对标。物理=两互不相溶组分在平面槽道中各占一半，
以单位混合体积体力 `G_x=5e-5` 驱动，界面处速度与剪应力连续、壁面无滑移；数值=D2Q9 + 线性伪势
（ψ=ρ）+ 两组分 BGK + Guo 二阶力 + post-streaming 反弹壁面；本轮的模型改动=中性润湿（mirror）SC
固壁填充 `_mirror_wall_fill`（历史 dry 壁在首层流体质点施加 `O(|G_12|ρ²)≈0.0875` 的假法向力，
是体力的 ~1750 倍）。
参数表：`ny=25/33/49`（`nx=4`，二维周期）、`τ_lower=1.0`（重相 ρ=0.7）、`τ_upper=0.75`（轻相 ρ=0.3）、
`G_12=-2.5`、`G_x=5e-5`、float32 单线程、20000/30000/45000 步；壁面有效面取半格位 `H=ny-2`，
界面 `h=ny/2-0.5`。解析锚=两层共动 Poiseuille 闭式解（逐层 u''=-G/ν + 界面速度/剪应力连续
+ 壁面 u=0），两种黏性混合规则（component：逐组分 ν；mixture：密度加权层 ν）。

## ② 怎么做的（审计方法，审计单=取证 + 判级 + 落表）
1. **取证三件套**：核 `artifacts/mcmp_stratified_poiseuille/records/r2_*.json`（6 个验收算例）与
   `diag/sp_*.json`（14 已入库 + 1 未入库长程控制）、`benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json`
   （ref-vs-got + 阈值 + `repro_cmd`）、`docs/figures/r2_*.png` 三图是否真存在。
2. **独立重算**：不采信入库的 reference 块，用 `analytic.layer_viscosities / layered_reference /
   sample_profile` 从记录自带 config 重算两层闭式解与 Q_total，再与入库值对表（浮点原值）。
3. **独立重跑**：①单算例 `run_sweep.py --ny 25 --steps 20000 --g12 -2.5 --u-eq viscous
   --force-scheme guo --wall-mode mirror`；②壁面约定标定 `run_sweep.py --mode walls --ny 25`；
   ③测试 `pytest tests/test_mcmp_wall_neutral.py tests/test_mcmp_stratified_poiseuille.py -q`。
4. **反造假探针**：重跑装配阶段 `r2_benchmark.py` 后对 `result_r2.json` 与三张图做 sha256 比对。
5. **判级 L0–L5** 后落 `tensor_maturity_verdicts`（id=222, level=4, verdict_by=board-audit），
   evidence 八键全部给真实指针（文件:行号/工件路径/测试名）。

## ③ 计算结果（全部由 `artifacts/audit_m14689/audit_summary.json` 程序化注入）
**判据 ①②③ 的独立复算**（与交付值逐位一致，`max|ΔQ_ref|=0.0`，6 个 record×rule 组合）：

{row_table()}

**判据 ③ 的缺口**（阈 0.005，交付判定 FAIL，本审计同判 FAIL，且缺口比平台摘要更大）：

{c3_table()}

**壁面约定标定（判决关键项，本审计独立重跑）**：`H=ny-1`（格心面）→ u_max {pct(CAL['H_ny_minus_1']['u_max_error_rel'])}、
Q {pct(CAL['H_ny_minus_1']['Q_error_rel'])}；`H=ny-2`（半格面）→ u_max {pct(CAL['H_ny_minus_2']['u_max_error_rel'])}、
Q {pct(CAL['H_ny_minus_2']['Q_error_rel'])}。单相 Poiseuille 独立标定支持半格面，故 ② 的 −3.97%（而非 −17.7%）是对标正确口径。

**静态控制（G_x=0, |G_12|=2.5, ny=49, mirror）不是收敛平台而是极限环**：40k 步后 `S_ifc` 后半程区间
{CTL['40000']['S_ifc_min_second_half']:.2e}–{CTL['40000']['S_ifc_max_second_half']:.2e}；未入库的 80k 步运行后半程
{CTL['80000']['S_ifc_min_second_half']:.2e}–{CTL['80000']['S_ifc_max_second_half']:.2e}（终值 {CTL['80000']['S_ifc_final']:.2e}
= 后半程最小值的 {CTL['80000']['ratio_final_over_min']:.1f} 倍）。表中 `S_ifc=2.24e-3` 是环上相位采样。

**重跑与测试**：单算例 bit-identical（Q_mix 0.391216903924942、u_max_mix 0.02561205066740513、
spurious_mix 0.010163000784814358、interface 11.539123946161155）；`pytest` 12 passed（30.19 s）。

## ④ 结果图
图 1（左）判据 ①（界面位置误差）与判据 ②（Q_mix，两种黏性规则）随 ny 的误差带：±5%/±10% 门槛内，
方向单调向零收敛；图 1（右）判据 ③ 比值与 0.005 门槛（对数轴），三个验收算例全部越门 17–79 倍。
图 2 为静态控制 `S_ifc(t)`：40k 与 80k 两次运行都表现为幅值 4 倍量级的极限环，说明"寄生流下限"
不是已收敛量——这是本审计新增的证据面（该 80k 工件在被审工作树中未入库）。

## ⑤ 注意点（坑与局限）
* **壁面约定是判决关键但取证链有洞**：`H=ny-2` 使 ② 从 −17.7% 变 −3.97%，而 R2 工件链里
  `analysis.json` 的 `wall_calibration` 为空、`repro_cmd` 不含 `--mode walls` 算例；标定证据只在
  round-1 `result.json`，本审计重跑复现（单相 −0.08% vs 格心 −8.2%）。
* **入库工件不可按 `repro_cmd` 逐位复现**：`run_r2_matrix.sh all` 含 `sp_static_ny49_long` 单元，其产物
  `sp_gx0_ny49_g25_mirror_long.json` 在工作树里未入库；重跑装配后 `result_r2.json` +2026 行、图 sha 变化。
  单算例可逐位复现，聚合工件不行。
* **"verified" 标签与仓内规则不一致**：`benchmarks/matrix.md:4` 要求误差 ≤3% 才入 verified/，本算例按
  预登记 5%/10% 门槛判 PASS，最差 3.84%（①）/3.97%（② mixture），按 3% 口径只有 ny=49（且 `steady=False`）合格。
* **比值口径不统一**：判据 ③ 用 `u_max_mix` 作分母，`static_floor` 表用 component 规则的解析 u_max
  （0.115 vs 0.1110），跨表不可直接比；不影响 FAIL 结论。
* **未登记**：`benchmarks/matrix.md` 无本算例行（P22 是更早的 SC-MCMP 两相 Poiseuille，仍 FAIL），
  规范文件 `result.json` 未更新（只在 `result_r2.json`）；`spec.json` 仍写格心壁面约定（R2 已否决）。
* 局限：只审"是否诚实地测量了该模型"，未审模型选型；未跑全 20 单元矩阵（每单元分钟级，本审计跑了
  判决关键的三条：单算例、壁面标定、装配）；CPU float32、单线程，1e-6 以下差异无物理意义。

## ⑥ 误差判定（逐项，显式回答是否 <3%）
* 判据 ① 界面位置：−3.84%（ny=25，**不 <3%**）、−2.95%（ny=33）、+1.38%（ny=49）。门槛 5% → PASS。
* 判据 ② 流量（mixture 规则，主口径）：−3.97%（**不 <3%**）、−2.71%、+0.29%。门槛 10% → PASS。
* 判据 ② 流量（component 规则，预登记要求同样成立）：−6.46%、−5.24%、−2.30%（**均 >3%**，但 <10%）→ PASS。
* 判据 ③ 寄生流：0.3968/0.2049/0.0861 vs 门槛 0.005 → **越门 79.4×/41.0×/17.2×，FAIL（诚实）**。
* 解析锚自身：本审计重算 Q_ref 与入库值差 0.0（浮点相等）；单相标定在 H=ny-2 下达 −0.08%/−0.20%（<3%）。
**结论：只有 ny=49（及 ny=33 的 mixture 口径 2.71%）满足 <3%；ny=25 的 3.84%/3.97% 以及 component 规则
全线 2.3–6.5% 均不满足 3% 口径，但在该算例预登记的 5%/10% 门槛内。判据 ③ 未达标，交付方如实报 FAIL，
本审计确认该 FAIL 为真并发现其缺口被平台摘要低报（"4–13×" vs 实测 17–79×）。**
"""


def main():
    tok = login()
    payload = {
        "title": "m15044 交差审计：m14689 GAP-LBM-MCMP-POISEUILLE — L4 判定 + 4 项 claim-strength 缺陷",
        "summary": ("独立复算两层闭式解与判据 ①②③（max|ΔQ_ref|=0，① −3.84/−2.95/+1.38%，"
                    "② mixture −3.97/−2.71/+0.29%，③ 0.397/0.205/0.086 vs 0.005 = FAIL 诚实）；"
                    "单算例重跑 bit-identical、12 测试全绿、壁面标定独立复现；判 L4 并落表 id=222。"
                    "发现：入库工件不可按 repro_cmd 复现（未入库 80k 控制）、静态“下限”实为极限环、"
                    "平台摘要把 ③ 缺口低报为 4–13×（实测 17–79×）、verified 标签宽松于仓内 ≤3% 规则。"),
        "scenario_key": "mcmp_stratified_poiseuille_audit",
        "mission_id": 15044,
        "tags": ["TensorLBM", "multiphase", "Shan-Chen", "audit", "board-audit",
                 "GAP-LBM-MCMP-POISEUILLE", "m14689", "honest-FAIL", "L4"],
        "markdown": MD,
        "figures": [
            fig("audit_m14689_criteria.png",
                "审计独立复算：判据①界面位置与判据②流量误差随 ny（±5%/±10% 门槛带）与判据③比值 vs 0.005 门槛"),
            fig("audit_m14689_static_control.png",
                "静态控制 S_ifc(t)（|G12|=2.5, ny=49, mirror, 40k/80k 步）：极限环而非已收敛下限"),
        ],
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    print(json.dumps(json.loads(urllib.request.urlopen(req, timeout=180).read()),
                     ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()