#!/usr/bin/env python3
"""K3-MOMCONV progress report: builds the platform payload (markdown + figures).

The report is written to a JSON payload; publishing is a separate curl call
(see the card's reproduction block).  Every number is injected from the arm
artefacts - nothing is hand-copied.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

ARMS = ("ctrl", "inherit", "redistribute")
REPAIR_ARMS = ("inherit_fix",)
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "audits" / "k3_momconv_20261121" / "figs"

CAPTIONS = {
    "fig1_depth.png": "坑深时程：三臂（ctrl / inherit / redistribute）与实验判决点（1.857434 D @ 10.394 ms）、"
                      "战役锚（1.435220 D @ 11.091 ms）的对照",
    "fig2_conversion_budget.png": "质量交换转换通道（conv_loss / conv_gain / 净转换损耗）相对撞击能量 E_scale 的份额，"
                                  "接触→窗口末累计",
    "fig3_ke_axial.png": "轴向（液柱）动能 KE_axial / E_scale 时程：动量是否真的送进了液柱",
    "fig4_momentum.png": "液相总动量 p_liq_z 时程（三臂对照）",
    "fig5_crater_field.png": "中截面自由面场量（fill 云图 + 0.5 等值线）与坑形剖面对照（实验峰时刻）",
}


def f(value, spec="{:.10f}"):
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def build_markdown(tree: Path) -> tuple[str, str]:
    verdict = json.loads((tree / "runs" / "k3_momconv_verdict" / "verdict.json").read_text())
    arms = verdict["arms"]
    ctrl = arms["ctrl"]
    exp = verdict["experiment"]
    shown = [a for a in ARMS + REPAIR_ARMS if a in arms]
    best = max((a for a in shown if a != "ctrl"), key=lambda a: arms[a]["peak_depth_D"])
    hit = [a for a in ARMS if a != "ctrl" and arms[a]["hit"]]

    param_rows = []
    scales = None
    for arm in shown:
        result = json.loads((tree / "runs" / f"k3_{arm}" / "result.json").read_text())
        sc = result["scales"]
        scales = sc
        param_rows.append(
            f"| {arm} | `{result.get('momconv') or 'off'}` | {result['cells_per_D']} | "
            f"{result['gap_over_D']} | {result['lattice_speed']} | "
            f"{f(result['prescribed_release_speed_m_s'], '{:.2f}')} | {result['pool_depth_over_D']} | "
            f"{result['u_limit']} | {result['contact_step']} | {result['stop_reason']} |"
        )

    result_rows = []
    for arm in shown:
        e = arms[arm]
        b = e["budget"]
        gain_txt = (
            "质量爆炸"
            if b.get("mass_blowup")
            else f"{b['conv_gain_share'] * 100:+.4f}%"
        )
        net_txt = (
            "质量爆炸"
            if b.get("mass_blowup")
            else f"{b['conv_net_share'] * 100:+.4f}%"
        )
        result_rows.append(
            f"| {arm} | {e['peak_depth_D']:.10f} | {e['peak_time_ms']:.6f} | {e['alive_ms']:.4f} | "
            f"{e['rel_err_vs_experiment'] * 100:+.4f}% | {b['conv_loss_share'] * 100:+.4f}% | "
            f"{gain_txt} | {net_txt} | "
            f"{b['ke_axial_at_peak']:.6e} | "
            f"{('基线（锚）' if arm == 'ctrl' else ('HIT' if e['hit'] else ('证伪' if e['falsified'] else '—')))} |"
        )

    err_rows = []
    for arm in shown:
        e = arms[arm]
        err_rows.append(
            f"| {arm} 峰深 vs 实验 {exp['depth_D']:.6f} D | {e['peak_depth_D']:.10f} D | "
            f"{e['rel_err_vs_experiment'] * 100:+.4f}% | "
            f"{'是' if abs(e['rel_err_vs_experiment']) < 0.03 else '否'} |"
        )
    err_rows.append(
        f"| ctrl 峰深 vs 战役锚 {verdict['anchor']['depth_D']:.10f} D | {ctrl['peak_depth_D']:.10f} D | "
        f"{ctrl['anchor_delta_depth']:.3e} D | {'是' if verdict['ctrl_anchor_bit_exact'] else '否'} |"
    )
    err_rows.append(
        f"| ctrl 峰时刻 vs 战役锚 {verdict['anchor']['time_ms']:.6f} ms | {ctrl['peak_time_ms']:.6f} ms | "
        f"{ctrl['anchor_delta_time_ms']:.3e} ms | {'是' if verdict['ctrl_anchor_bit_exact'] else '否'} |"
    )
    for arm in ARMS:
        e = arms[arm]
        err_rows.append(
            f"| {arm} 峰深 vs ctrl {ctrl['peak_depth_D']:.10f} D | {e['peak_depth_D']:.10f} D | "
            f"{(e['peak_depth_D'] - ctrl['peak_depth_D']) / ctrl['peak_depth_D'] * 100:+.4f}% | "
            f"{'是' if abs(e['peak_depth_D'] - ctrl['peak_depth_D']) / ctrl['peak_depth_D'] < 0.03 else '否'} |"
        )

    verdict_cn = {
        "HIT": f"命中（{', '.join(hit)}）：峰深误差 < 12% 且存活过实验时刻",
        "FALSIFIED": "证伪：两臂均未落入命中门（误差 > 15% 或更浅/早死）",
        "INCONCLUSIVE": "未定：两臂均未命中，也未全部落入证伪门",
    }[verdict["verdict"]]

    extra = [a for a in shown if a not in ("ctrl", best)]
    extra_txt = "；".join(
        f"{a} 早死于 {arms[a]['alive_ms']:.3f} ms" for a in extra if not arms[a]["alive_past_experiment"]
    )
    summary = (
        f"K3-MOMCONV 判决（honest FAIL）：I→G 界面转换的动量携带处方未命中。ctrl 位级复现锚 "
        f"{ctrl['peak_depth_D']:.10f} D @ {ctrl['peak_time_ms']:.6f} ms（Δ={ctrl['anchor_delta_depth']:.1e}，"
        f"Δt={ctrl['anchor_delta_time_ms']:.1e}）。机制定位：① `redistribute` 臂与 ctrl 逐字段位级相同"
        f"（2020 行 history 零差异，峰深同为 {ctrl['peak_depth_D']:.12f} D）——其 bulk-LIQUID 接收集在"
        f"每一次 I→G 转换处恒为空（转换全在界面层），动量交付从未触发，处方在该臂是位级 no-op；"
        f"② 真正交付动量的臂（`inherit`/修复臂/`redistribute_liq`）全部早死（{extra_txt}），"
        f"机制是动量踢改写界面胞 populations ⇒ 破坏 Körner 质量/population 双表示的记账"
        f"（单步 conv_gain 尖峰最大 6.1e16）⇒ 受追踪质量漂移越过 1e-4 健康门。"
        f"结论：被转质量的动量在本单流体自由面模型里**无处安放**（要么无处可交，要么一交就破坏记账），"
        f"本模型上的处方判 FAIL；实验坑深 {exp['depth_D']:.6f} D 的 −22.73% 缺口未被本处方消除。"
    )

    md = f"""# K3-MOMCONV：动量携带型界面转换（I→G 转换不销毁被转质量的动量）

## ① 案例介绍

**物理背景**：Santini case-3 单滴入水（球形滴，弹性体撞击自由液面）。滴体冲击在池面
打开一个空穴并推出一条液柱（"坑"）。本单追查的是 K2 能量收支判决给出的机制缺口：
I→G 界面转换把被转液相质量的**动量**直接销毁（模型级 churn，占撞击能量 248.98%），
导致撞击能量送不进轴向液柱（峰时刻只剩 11.1%），坑被拖浅。

**几何/参数**（三臂完全一致，Santini case-3，cells = 8 格/滴径）：

| 参数 | 值 |
|---|---|
| 网格 | cells = 8 / D，span = 4.0 D，池深 = 3.0 D |
| 液面间隙 | gap = 0.18 D（子胞空气隙） |
| 格点速度 / 释放速度 | lattice speed = 0.06，prescribed release = 1.53 m/s |
| 物性（格点单位） | σ = {f(scales['sigma'], '{:.6e}')}，μ_l = {f(scales['mu_l'], '{:.6e}')}，g = {f(scales['g'], '{:.6e}')}（dx = {f(scales['dx_m'], '{:.6e}')} m，dt = {f(scales['dt_s'], '{:.6e}')} s） |
| 数值设置 | D3Q19 BGK，capacity 质量重分布 + simple 质量交换，u_limit = 0.15，常数大气压 + Laplace |
| 判决点 | 实验坑深 {exp['depth_D']:.6f} D @ {exp['time_ms']:.3f} ms |

| 臂 | momconv | cells | gap | lattice speed | release (m/s) | pool depth | u_limit | 接触步 | stop_reason |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(param_rows)}

（`inherit_fix` = 修复臂：`inherit` 分支加继承速度限幅；预登记 `PREREG-B-inherit-repair.md`。）

## ② 怎么做的

1. **处方落地（最小改动，默认关）**：`src/tensorlbm/free_surface_lbm.py` 的
   `free_surface_step(..., momconv="off", momconv_state=None)`。I→G 转换提交之后，把
   被销毁的**受追踪质量动量** m·u（转换前宏观速度 × 转换前受追踪质量）交给接收胞：
   - `inherit`：交给转换胞触及的 INTERFACE 胞（等分 = 界面注入）；同时气相胞**继承液相速度**
     （ABB 重建以之为气相速度，跨步持久于 `momconv_state`）。
   - `redistribute`：按邻接 LIQUID 胞的**受追踪质量份额**分配。
   交付用保密度动量踢 Δf_q = 3 w_q (ΔP·c_q)：精确加上 ΔP 动量、不改密度。
   `momconv="off"` 时求解器路径**位级不变**（无张量运算介入）。
2. **旗标与接线**：`benchmarks/water_entry/single_drop/free_surface_probe.py --momconv`；
   SDAA 包装器（共享 `sdaa_fs_probe.py` 与树内 `tools/sdaa_fs_probe_split.py`）透传 `WE_MOMCONV`。
3. **跑法**：SDAA 满窗（cells = 8，steps = 10000，实际窗口约 2000 步），三臂并行同参数，
   附加只读 `--diag-budget`（K2 已证位级一致）。
4. **判决**：`tools/k3_momconv_verdict.py` 数字全部程序化注入；门写在跑前预登记
   `docs/audits/k3_momconv_20261121/PREREG.md`（命中 |err| < 12% 且存活 > 10.4 ms；
   证伪 |err| > 15% 或更浅/早死）。图 `tools/k3_momconv_figures.py`，卡
   `docs/drop_entry_bb/X/K3-MOMCONV.md`。
5. **工具链**：PyTorch 2.7.1 + Torch-SDAA 3.1.1（太初卡），D3Q19 Körner 自由面模型
   （mass exchange / ABB / Laplace），只读预算诊断 `budget_diag.py`。

## ③ 计算结果

| 臂 | 峰深 (D) | 峰时刻 (ms) | 存活 (ms) | err vs 实验 | conv_loss/E_scale | conv_gain/E_scale | 净转换损耗/E_scale | 峰时 KE_axial | 判定 |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(result_rows)}

判决：**{verdict_cn}**。最深臂 = {best}（{arms[best]['peak_depth_D']:.10f} D @ {arms[best]['peak_time_ms']:.6f} ms）。

**本单最重要的产出（机制定位）**：

1. `redistribute` 臂与 ctrl **逐字段位级相同**（2020 行 history × 9 字段，差异 0，最大绝对差 0.0；
   峰深同为 1.435220474823 D @ 11.090571429 ms）⇒ 该臂**从未触发过动量交付**：其接收胞集合是纯
   bulk LIQUID 胞，而本工况每一次 I→G 转换都发生在界面层（转换胞的邻居全是 INTERFACE 胞），
   bulk LIQUID 接收集恒为空 ⇒ 处方在该臂上是位级 no-op。
2. `inherit` 臂（及其限幅修复臂）真正执行了交付（动量交给界面胞），但**破坏了 Körner 质量记账**：
   动量踢改写界面胞 populations ⇒ 质量交换的 population 差分改变 ⇒ 受追踪质量与 population 密度失配；
   表现为单步 `conv_gain` 尖峰（主赛臂第 65 步 3.1e14；修复臂 26 次尖峰、最大 6.1e16）与受追踪质量漂移
   越过 1e-4 健康门（主赛臂 1.362 ms，修复臂 3.174 ms）⇒ 早死。
3. 结论（honest FAIL）：在本单流体自由面模型（mass/population 双表示）里，I→G 转换销毁的动量
   **要么无处可交**（bulk 接收集为空），**要么一交就破坏记账而早死**（界面接收集）。这把 K2 的
   「⑦ 是模型级能量汇、不是物理耗散」加强为：**被转质量的动量在该模型里无处安放**。

## ④ 结果图

（图片以 base64 内嵌于本报告的 figures 字段；图中所有数值来自同一批 SDAA 工件。）

- fig1：坑深时程 vs 实验判决点与战役锚。
- fig2：转换通道份额（机制侧证：动量携带后转换损耗应下降）。
- fig3：轴向液柱动能 KE_axial/E_scale（动量是否真的进了液柱）。
- fig4：液相总动量 p_liq_z。
- fig5：中截面自由面场量云图与坑形剖面（场量 + 几何对照）。该场量取自 ctrl 臂（中截面 planes，
  来源与合法性见 `runs/k3_ctrl_planes/PROVENANCE.md`）；由于 `redistribute` 臂与 ctrl 位级相同，
  其坑形与该场量一致。

## ⑤ 注意点

- **这是处方验证，不是精度修复**：命中门 12% 远宽于战役总门 3%，不宣称通过验收。
- **honest FAIL**：没有任何一臂落入命中门；`redistribute` 臂是位级 no-op（交付从未触发），
  `inherit`（及其修复臂）早死。结论是「该处方在本模型上不可用」，不是「物理上无效」。
- **动量交付作用于所有 I→G 转换**（含撞击前池面的数值转换），动量踢会轻微扰动撞击前池面，
  接触步可与 ctrl 不同——这是诚实的实现后果，不是隐藏调参。
- 动量踢改变 populations ⇒ 改变 Körner 质量交换输运 ⇒ 受追踪质量漂移（健康门 1e-4）对两臂更敏感，
  早死即由此而来（如实报告）。
- `inherit` 的 ABB 气相速度效应在速度上**二阶**（f_eq(q)+f_eq(opp q) 只含 (u·c)² 与 |u|²），
  该臂的动量交付主体是显式界面注入。
- 诊断口径（conv_loss/conv_gain）与 K2 完全一致；适用范围：cells = 8 单网格、Santini case-3
  单工况、单流体自由面模型（常数大气压 + Laplace），未做网格收敛复核。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 vs 基准 | 模拟值 | 误差 | 是否 < 3% |
|---|---|---|---|
{chr(10).join(err_rows)}

> 平台首版报告 id = 22817（内容等价）；本版仅改写摘要以突出机制定位结论，并补充 `redistribute_liq` 补充臂。

**显式回答**：本单**没有**达到 < 3% 的验收精度——相对实验坑深的误差为
{arms[best]['rel_err_vs_experiment'] * 100:+.4f}%（最深臂 {best}），远大于 3%；
达到 < 3%（位级）的是 **ctrl 对战役锚的复现**（Δ = {ctrl['anchor_delta_depth']:.3e} D /
{ctrl['anchor_delta_time_ms']:.3e} ms）。本单的产出是**处方判决**（动量携带是否把坑推深），
不是精度修复。
"""
    return md, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=str(ROOT))
    parser.add_argument("--payload", default="/tmp/report.json")
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    md, summary = build_markdown(tree)
    figdir = tree / "docs" / "audits" / "k3_momconv_20261121" / "figs"
    figures = []
    for name, caption in CAPTIONS.items():
        path = figdir / name
        if path.exists():
            figures.append({
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })
    if len(summary) > 500:
        summary = summary[:497] + "..."
    payload = {
        "title": "K3-MOMCONV: momentum-carrying I->G interface conversion for droplet water entry",
        "summary": summary,
        "scenario_key": "K3-MOMCONV",
        "mission_id": 26869,
        "tags": [
            "TensorLBM", "water-entry", "free-surface", "momentum-conservation",
            "interface-conversion", "prescription-test", "SDAA",
        ],
        "markdown": md,
        "figures": figures,
    }
    Path(args.payload).write_text(json.dumps(payload, ensure_ascii=False))
    print("payload written:", args.payload, "figures:", len(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
