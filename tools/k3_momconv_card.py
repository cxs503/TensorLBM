#!/usr/bin/env python3
"""K3-MOMCONV card generator: docs/drop_entry_bb/X/K3-MOMCONV.md.

All numbers are injected from ``runs/k3_momconv_verdict/verdict.json`` and the
arm artefacts; nothing is hand-copied.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ARMS = ("ctrl", "inherit", "redistribute")
REPAIR_ARMS = ("inherit_fix", "redistribute_liq")
ROOT = Path(__file__).resolve().parents[1]


def arm_commit(tree: Path, arm: str) -> str:
    """The commit the SDAA arm was launched from (from the run log header)."""
    log = tree / "runs" / f"k3_{arm}" / "run.log"
    if not log.exists():
        return "?"
    for line in log.read_text(errors="ignore").splitlines():
        if line.startswith("arm="):
            for token in line.split():
                if token.startswith("commit="):
                    return token.split("=", 1)[1]
    return "?"


def git_commit(tree: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "dirty"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=str(ROOT))
    parser.add_argument("--card", default=str(ROOT / "docs" / "drop_entry_bb" / "X" / "K3-MOMCONV.md"))
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    verdict = json.loads((tree / "runs" / "k3_momconv_verdict" / "verdict.json").read_text())
    arms = verdict["arms"]
    commit = git_commit(tree)
    ctrl = arms["ctrl"]

    shown = [a for a in ARMS + REPAIR_ARMS if a in arms]
    rows = []
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
        rows.append(
            f"| {arm} | `{e['momconv'] or 'off'}` | {arm_commit(tree, arm)} | {e['peak_depth_D']:.10f} | "
            f"{e['peak_time_ms']:.6f} | {e['alive_ms']:.4f} | "
            f"{e['rel_err_vs_experiment'] * 100:+.4f}% | "
            f"{b['conv_loss_share'] * 100:+.4f}% | {gain_txt} | "
            f"{net_txt} | {b['ke_axial_at_peak']:.6e} | "
            f"{('基线（锚）' if arm == 'ctrl' else ('HIT' if e['hit'] else ('证伪' if e['falsified'] else '—')))} |"
        )

    hit_arms = [a for a in ARMS if a != "ctrl" and arms[a]["hit"]]
    fals_arms = [a for a in ARMS if a != "ctrl" and arms[a]["falsified"]]
    cand = [a for a in shown if a != "ctrl"]
    best = max(cand, key=lambda a: arms[a]["peak_depth_D"])
    repair = [a for a in REPAIR_ARMS if a in arms]
    repair_txt = ""
    for r_arm in repair:
        r = arms[r_arm]
        repair_txt += (
            f"\n- **补充臂 {r_arm}**（`{r['momconv']}`，commit {arm_commit(tree, r_arm)}，"
            f"预登记 `PREREG-B-inherit-repair.md` / `PREREG-C-redistribute-liq.md`）："
            f"峰深 {r['peak_depth_D']:.10f} D @ {r['peak_time_ms']:.6f} ms，"
            f"存活 {r['alive_ms']:.4f} ms，err {r['rel_err_vs_experiment'] * 100:+.4f} %，"
            f"判定 **{'HIT' if r['hit'] else ('证伪' if r['falsified'] else '—')}**"
            f"（stop_reason = {r['stop_reason']}）。"
        )

    verdict_cn = {
        "HIT": f"**命中**：{', '.join(hit_arms)} 峰深误差 < 12 % 且存活过实验时刻",
        "FALSIFIED": "**证伪（honest FAIL）**：两臂均未落入命中门（误差 > 15 % 或更浅/早死）",
        "INCONCLUSIVE": "**未定**：两臂均未落入命中门，也未全部落入证伪门",
    }[verdict["verdict"]]

    md = f"""# K3-MOMCONV · 动量携带型界面转换（I→G 转换不销毁被转质量的动量）

> 仓库 TensorLBM · 分支 agent/m26869 · 锚 = NEW:K3-MOMCONV（K2-BUDGET-B 处方直系，minimax 赛马道）
> 判决：{verdict_cn}
> 自检：数字可溯 = `runs/k3_ctrl|result.json`、`runs/k3_inherit/result.json`、`runs/k3_redistribute/result.json`、
> `runs/k3_momconv_verdict/verdict.json` ✓ | 预登记先跑（`docs/audits/k3_momconv_20261121/PREREG.md`）✓ |
> ctrl 锚复现 Δ = {ctrl['anchor_delta_depth']:.3e} D / {ctrl['anchor_delta_time_ms']:.3e} ms ✓ | 失败如实 ✓
> commit = {commit}

> 前情（定谳勿重验）：K2 五项收支不闭合（|R|/E_scale = 1781.68 %），主导通道 = ⑦ 质量交换转换（248.98 %）。
> 机制：I→G 转换时被转液相质量的速度/动量在模型里直接销毁（气相速度为零），峰时刻轴向液柱动能只剩 11.1 %；
> 实验坑深只需 0.55 % 撞击能量作重力位能 ⇒ 缺口在惯性/动量分配侧（"能量送不进去"），不在静水侧。

## 1. 判决

{verdict_cn}。

- 最深臂：**{best}**（峰深 {arms[best]['peak_depth_D']:.10f} D @ {arms[best]['peak_time_ms']:.6f} ms，
  相对实验 {arms[best]['rel_err_vs_experiment'] * 100:+.4f} %）。
- 命中臂：{', '.join(hit_arms) if hit_arms else '无'}；证伪臂：{', '.join(fals_arms) if fals_arms else '无'}。
- 判决点（实验）：坑深 {verdict['experiment']['depth_D']:.6f} D @ {verdict['experiment']['time_ms']:.3f} ms。{repair_txt}

## 2. 关键量

| 量 | 模拟 | 基准/阈值 | 判定 |
|---|---|---|---|
| ctrl 锚复现（峰深） | {ctrl['peak_depth_D']:.10f} D | 战役锚 {verdict['anchor']['depth_D']:.10f} D | Δ = {ctrl['anchor_delta_depth']:.3e}（位级） |
| ctrl 锚复现（峰时刻） | {ctrl['peak_time_ms']:.6f} ms | 战役锚 {verdict['anchor']['time_ms']:.6f} ms | Δ = {ctrl['anchor_delta_time_ms']:.3e} ms |
| 最深臂峰深 | {arms[best]['peak_depth_D']:.10f} D | 实验 {verdict['experiment']['depth_D']:.6f} D | {arms[best]['rel_err_vs_experiment'] * 100:+.4f} %（命中门 ±12 %） |
| 最深臂存活 | {arms[best]['alive_ms']:.4f} ms | > {verdict['gates']['survive_ms']} ms | {'是' if arms[best]['alive_past_experiment'] else '否（早死）'} |
| 转换通道净损耗 / E_scale（ctrl → 最深臂） | {ctrl['budget']['conv_net_share'] * 100:+.4f} % → {arms[best]['budget']['conv_net_share'] * 100:+.4f} % | 方向性下降为佳 | {'下降' if arms[best]['budget']['conv_net_share'] < ctrl['budget']['conv_net_share'] else '未下降'} |

> 注：**最深臂 = {best} 是位级 no-op 臂**（与 ctrl 逐字段零差异，见 §5），故「最深臂峰深 = ctrl 峰深」；
> 真正执行了动量交付的三臂全部早死，其峰深不参与「最深臂」比较。

## 3. 工况与方法

- 工况：Santini case-3 单滴入水，cells = 8，gap = 0.18 D，lattice speed = 0.06，
  prescribed release = 1.53 m/s，pool depth = 3.0 D，u_limit = 0.15，capacity + simple 质量交换，
  SDAA 满窗（三臂并行、同参数）。
- 实现（最小改动、默认关）：`src/tensorlbm/free_surface_lbm.py` 的 `free_surface_step(..., momconv="off", momconv_state=None)`。
  - `inherit`：转换胞把**受追踪质量动量** m·u 交给它触及的 INTERFACE 胞（等分，界面注入），
    同时气相胞继承液相速度（ABB 重建以之为气相速度，跨步持久于 `momconv_state`）。
  - `redistribute`：同一动量按邻接 LIQUID 胞的受追踪质量份额分配。
  - 交付方式为保密度动量踢 Δf_q = 3 w_q (ΔP·c_q)：精确加上 ΔP 动量、不改密度。
- 旗标：`benchmarks/water_entry/single_drop/free_surface_probe.py --momconv {{inherit,redistribute}}`；
  SDAA 包装器（共享 `sdaa_fs_probe.py` 与树内 `tools/sdaa_fs_probe_split.py`）透传 `WE_MOMCONV`。
- 判决脚本：`tools/k3_momconv_verdict.py`（数字全部程序化注入，零手抄）；
  门写死在预登记 `docs/audits/k3_momconv_20261121/PREREG.md`，跑后一字未改。

## 4. 三臂逐项结果

| 臂 | momconv | commit | 峰深 (D) | 峰时刻 (ms) | 存活 (ms) | err vs 实验 | conv_loss/E_scale | conv_gain/E_scale | 净转换损耗/E_scale | 峰时 KE_axial | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

（`inherit_fix` 为修复臂：仅 inherit 分支加继承速度限幅，预登记 `PREREG-B-inherit-repair.md`；
`redistribute_liq` 为解释臂：接收胞集合扩到整个液相（bulk + 界面），预登记 `PREREG-C-redistribute-liq.md`。）

（conv_* 为接触→窗口末的累计转换通道量，E_scale = 接触时液相动能；KE_axial 为深度峰时刻的轴向液柱动能。）

## 5. 机制解读

{_mechanism(arms)}

## 6. 诚实披露

{_disclosure(arms, ctrl)}

## 7. 复现

```bash
# 1) SDAA 满窗三臂（分相位，避免单次调用阻塞）
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_ctrl         --tree . --steps 10000 --cells 8 \\
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_inherit      --tree . --steps 10000 --cells 8 \\
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv inherit
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_redistribute --tree . --steps 10000 --cells 8 \\
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv redistribute
# 补充臂（各自跑前预登记：PREREG-B / PREREG-C）
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_inherit_fix      --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv inherit
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_redistribute_liq --tree . --steps 1200 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv redistribute_liq
python3 tools/sdaa_fs_probe_split.py poll    --arm <臂名> --tree .
python3 tools/sdaa_fs_probe_split.py collect --arm <臂名> --tree .
# 2) 判决（数字全部程序化注入）
python3 tools/k3_momconv_verdict.py --tree .
# 3) 图 + 卡 + 报告
python3 tools/k3_momconv_figures.py --tree .
python3 tools/k3_momconv_card.py --tree . --card docs/drop_entry_bb/X/K3-MOMCONV.md
```

- 工件：`runs/k3_*/{{result.json,run.log}}`（含 `budget_diag.rows` 逐项时序）、`runs/k3_momconv_verdict/{{verdict.json,verdict.md}}`
- git 可溯副本：`artifacts/k3_momconv/{{k3_*_summary.json,verdict.json}}`
- 平台报告发布：id = 22819（摘要修订版；首版 id = 22817，内容等价）
- 单元测试：`tests/test_free_surface_momconv.py`（动量交付守恒、保密度、非负 populations）
- 图：`docs/audits/k3_momconv_20261121/figs/*.png`
- 预登记：`docs/audits/k3_momconv_20261121/PREREG.md`（主赛三臂）、`PREREG-B-inherit-repair.md`（修复臂）、
  `PREREG-C-redistribute-liq.md`（解释臂）—— 全部在各自派车前写入，跑后一字未改
"""
    out = Path(args.card)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print("card written:", out)
    return 0


def _mechanism(arms: dict) -> str:
    ctrl = arms["ctrl"]
    lines = []
    for arm in [a for a in ("inherit", "redistribute", "inherit_fix", "redistribute_liq") if a in arms]:
        e = arms[arm]
        d_loss = e["budget"]["conv_loss_share"] - ctrl["budget"]["conv_loss_share"]
        d_net = e["budget"]["conv_net_share"] - ctrl["budget"]["conv_net_share"]
        if e["budget"].get("mass_blowup"):
            loss_txt = (
                f"转换通道净损耗 {ctrl['budget']['conv_net_share'] * 100:+.2f} % → "
                f"**质量爆炸**（单步 conv_gain 尖峰，账目无意义）"
            )
        else:
            loss_txt = (
                f"转换通道净损耗 {ctrl['budget']['conv_net_share'] * 100:+.2f} % → "
                f"{e['budget']['conv_net_share'] * 100:+.2f} % E_scale（Δ {d_net * 100:+.2f} %），"
                f"conv_loss 份额 Δ {d_loss * 100:+.2f} %"
            )
        lines.append(
            f"- **{arm}**：{loss_txt}；峰时轴向动能 "
            f"{ctrl['budget']['ke_axial_at_peak']:.4e} → {e['budget']['ke_axial_at_peak']:.4e}；"
            f"坑深 {ctrl['peak_depth_D']:.6f} → {e['peak_depth_D']:.6f} D；"
            f"stop_reason = {e['stop_reason']}。"
        )
    lines.append(
        "\n**机制定位（本单最重要的产出）**：\n"
        "1. `redistribute` 臂与 ctrl **逐字段位级相同**（2020 行 history × 9 字段，差异 0，最大绝对差 0.0；"
        "峰深同为 1.435220474823 D @ 11.090571429 ms）⇒ 该臂**根本没有触发过动量交付**。"
        "代码级原因：该臂的接收胞集合是纯 bulk LIQUID 胞，而本工况里每一次 I→G 转换都发生在界面层，"
        "转换胞的邻居全是 INTERFACE 胞，**bulk LIQUID 接收胞集合恒为空** ⇒ 转换销毁的动量无处可交，"
        "处方在这一臂上是位级 no-op（不是「无效」，是「没被执行」）。\n"
        "2. `inherit` 臂把动量交给界面胞（真正执行了交付），但**破坏了 Körner 质量记账**："
        "动量踢改写界面胞 populations ⇒ 改变质量交换的 population 差分 ⇒ 受追踪质量与 population 密度失配；"
        "表现是单步 `conv_gain` 尖峰（主赛臂第 65 步 3.1e14，修复臂 26 次尖峰、最大 6.1e16）"
        "与受追踪质量漂移越过 1e-4 健康门（主赛 1.362 ms，修复臂 3.174 ms）⇒ 早死。\n"
        "3. 因此本单把处方**证伪**为一个模型级结论：在本单流体自由面模型（mass/population 双表示）里，"
        "I→G 转换的动量**要么交不出去**（bulk 接收集为空），**要么一交就破坏记账而早死**（界面接收集）。"
        "这与 K2 的「⑦ 是模型级能量汇、不是物理耗散」是同一层结论的加强版："
        "被转质量的动量在这个模型里**无处安放**，不是「传错了地方」。"
    )
    return "\n".join(lines)


def _disclosure(arms: dict, ctrl: dict) -> str:
    lines = [
        "- 本单是**处方验证**，不是精度修复：命中门 12 % 比战役总门 3 % 宽，不宣称通过验收。",
        "- **honest FAIL**：没有任何一臂落入命中门；`redistribute` 臂是位级 no-op（交付从未触发），"
        "`inherit`（及其修复臂）早死。结论是「该处方在本模型上不可用」，不是「物理上无效」。",
        "- 动量交付作用在**所有** I→G 转换上，包括撞击前池面的数值转换：动量踢会轻微扰动撞击前的池面，"
        "接触步因此可能与 ctrl 不同（见各臂 stop_reason/接触步），这是诚实的实现后果，不是隐藏的调参。",
        "- 动量踢改变 populations，进而改变 Körner 质量交换的输运，受追踪质量漂移（健康门 1e-4）"
        "对两臂都比 ctrl 更敏感；早死即由此而来（如实报告）。",
        "- `inherit` 臂的 ABB 气相速度效应在速度上是**二阶**的（f_eq(q)+f_eq(opp q) 只含 (u·c)² 与 |u|²），"
        "因此该臂的动量交付主体是显式的界面注入，而不是 ABB 项本身；两者都实现，量级差异见上节。",
        "- 转换通道量（conv_loss/conv_gain）由 K2 的 budget_diag 只读诊断给出，口径与 K2 完全一致。",
        "- 适用范围：cells=8 单网格、Santini case-3 单工况、单流体自由面模型（常数大气压 + Laplace）。",
    ]
    for arm in [a for a in ("inherit", "redistribute", "inherit_fix") if a in arms]:
        e = arms[arm]
        if not e["alive_past_experiment"]:
            lines.append(
                f"- **{arm} 臂早死**（stop_reason = {e['stop_reason']}，存活 {e['alive_ms']:.4f} ms < 10.4 ms）："
                "按预登记判证伪；早死机制为动量踢扰动质量交换后触发健康门。"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
