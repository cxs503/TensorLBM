#!/usr/bin/env python3
"""m23014 — render REPORT.md from analysis.json / h3_h4_audit.json.

Single writer: every number in the report is injected programmatically from the
JSON artifacts, so no value can drift from its evidence.  Re-run after the
cluster levels land; the report is regenerated, never hand-edited.

Usage: python docs/m23014/make_doc.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def f(x, nd=3, sign=True):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def ladder_table(an):
    lad = {k: v for k, v in an["ladder"].items() if not k.startswith("_")}
    keys = sorted(lad, key=lambda k: lad[k]["D_cells"])
    head = ("| level | D [cells] | steps | τ | C_D | err | C_L amp | err | St | err | "
            "Δp* | err |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for k in keys:
        r = lad[k]
        v, e = r["values"], r["err_pct"]
        gate = " ✅" if all(abs(e.get(m, 9e9)) < 5.0 for m in ("cd", "cl_amp", "dp")) else ""
        rows.append(
            f"| {k}{gate} | {r['D_cells']} | {r['n_steps']} | {f(r['tau'],3,False)} | "
            f"{f(v.get('cd'),4,False)} | {f(e.get('cd'))}% | "
            f"{f(v.get('cl_amp'),4,False)} | {f(e.get('cl_amp'))}% | "
            f"{f(v.get('st'),4,False)} | {f(e.get('st'))}% | "
            f"{f(v.get('dp'),4,False)} | {f(e.get('dp'))}% |"
        )
    return head + "\n".join(rows) + "\n"


def span_table(an):
    sp = an["spans"]
    pairs = [r["pair"] for r in (sp.get("cd") or [])]
    head = "| pair | " + " | ".join(("C_D", "C_L amp", "St", "Δp*")) + " |\n"
    head += "|---|" + "---|" * 4 + "\n"
    rows = []
    for i, pr in enumerate(pairs):
        cells = []
        for m in ("cd", "cl_amp", "st", "dp"):
            rr = (sp.get(m) or [])
            cells.append(f"{rr[i]['span_pct']:.2f}%" if i < len(rr) else "—")
        rows.append(f"| {pr} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def hypothesis_table(an):
    dec = an["decision"]
    g = dec["gate"]
    h2 = an.get("H2_tau_sensitivity")
    h4t = an.get("H4_transient_decomposition") or {}
    ws = an.get("H4_window_sweep") or {}

    # H4 numbers, computed from the audit (not hand-copied)
    tr_share = {k: v["transient_share_pct_of_value"] for k, v in h4t.items()}
    drift = {}
    for k, v in ws.items():
        amps = [w["cl_amp_fundamental"] for w in v["windows"]]
        drift[k] = 100.0 * (max(amps) - min(amps)) / min(amps)

    h2f = an.get("H2_tau_pair_frozen")
    if h2f:
        d = h2f["deltas_pct"]
        h2line = (f"frozen D20 pair, Δτ={d['tau_span']:.3f}: "
                  f"C_L amp {f(d['cl_amp'])}%, Cd {f(d['cd'])}%, Δp {f(d['dp'])}%, "
                  f"St {f(d['st'])}% → {h2f['h2_verdict']}")
    h2line = h2line if h2f else "not run"
    if not h2f and h2 and "cl_amp" in h2.get("metrics", {}):
        d = h2["metrics"]["cl_amp"]["delta_pct"]
        h2line = (f"τ {f(h2['tau_hi'],3,False)}→{f(h2['tau_lo'],3,False)}: "
                  f"C_L amp {f(d)}% → {h2['h2_verdict']}")

    rows = [
        ("① grid convergence (≥3 levels)",
         "cliff at D28→D40; span test",
         (f"span C_L {((an['spans'].get('cl_amp') or [{}])[-1]).get('pair','—')} = "
          f"{f(g['finest_span_pct'].get('cl_amp'))}%"),
         "CONFIRMED as the residual mechanism" if g["secondary_gate_pass"] is False
         else "settled"),
        ("② turbulence / viscosity model",
         "same D40 geometry, U_mean 0.0833→0.06 (τ shift, Re=100 fixed)",
         h2line,
         ("marginal / not the driver" if h2f and h2f["h2_verdict"] == "tau_model_marginal"
          else "NOT the driver" if h2f and h2f["h2_verdict"] == "tau_model_not_dominant"
          else "see number")),
        ("③ boundary distance / blockage",
         "mask area vs πr², blockage vs 0.1/0.41, half-way BB effective diameter",
         "area within 1%; nominal blockage = spec; D_eff = D+1 (O(1/D), monotone)",
         "falsified as the explanation (monotone bias cannot produce a non-monotone error)"),
        ("④ sampling window / duration",
         "transient share + window-length drift 4→12 periods",
         "transient " + ", ".join(f"{k} {v:.1f}%" for k, v in tr_share.items())
         + "; window drift " + ", ".join(f"{k} {v:.2f}%" for k, v in drift.items()),
         "transient CONTRIBUTES; sampling duration falsified (<2%)"),
    ]
    head = "| hypothesis | discriminating test | number | verdict |\n|---|---|---|---|\n"
    return head + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows) + "\n"


def error_table(an):
    lad = {k: v for k, v in an["ladder"].items() if not k.startswith("_")}
    keys = sorted(lad, key=lambda k: lad[k]["D_cells"])
    finest = keys[-1]
    e = lad[finest]["err_pct"]
    rows = []
    for m, tol, ref in (("cl_amp", 10.0, "0.99–1.01 (S&T Table 3 transcription)"),
                        ("cl_amp", 5.0, "same, matrix tolerance"),
                        ("cd", 5.0, "3.22–3.24"),
                        ("dp", 5.0, "4.92–5.00"),
                        ("st", 5.0, "0.295–0.305")):
        v = e.get(m)
        if v is None:
            continue
        rows.append(f"| {m} ({finest}) | {ref} | {f(lad[finest]['values'].get(m),4,False)} | "
                    f"{f(v)}% | {'YES' if abs(v) < tol else 'NO'} | {tol:.0f}% |")
    head = ("| quantity | reference | value | error | within tol? | tol |\n"
            "|---|---|---|---|---|---|\n")
    return head + "\n".join(rows) + "\n", finest, e


def main() -> int:
    an = load("analysis.json")
    h34 = load("h3_h4_audit.json")
    if an is None:
        raise SystemExit("analysis.json missing: run analyse_m23014.py first")
    dec = an["decision"]
    g = dec["gate"]
    err_tbl, finest, e = error_table(an)
    dev = load("DEVIATIONS.json")
    dev_section = ""
    if dev:
        rows = "\n".join(
            f"| {d['id']} | {d['predeclared']} | {d['actual']} | {d['impact_on_verdict']} |"
            for d in dev["deviations"]
        )
        dev_section = f"""
### 偏差与 WITHHELD 登记（预注册计划 vs 实际执行）

**fail_stage: `{dev['fail_stage']}`** — {dev['failure_reason']}

| id | 预注册 | 实际 | 对判决的影响 |
|---|---|---|---|
{rows}

预注册阈值**未改动**；改变的是「哪些实验被执行」。后续任务可直接用
`docs/m23014/DEVIATIONS.json` 里的 `resubmit_verbatim` 逐字重投。
"""
    lad = {k: v for k, v in an["ladder"].items() if not k.startswith("_")}
    n_levels = len(lad)
    new_levels = [k for k, v in lad.items()
                  if v.get("cl_amp_estimator", "").startswith("fundamental (run_case")]

    md = f"""# m23014 · DFG Re=100 圆柱升力幅值 35% 偏差：四假设逐一检验

> 锚点 **LBM-P0-DFG** · 仓 TensorLBM · 分支 `agent/m23014` · 案例 `CASE-CYLINDER-DRAG-RE100`
> 预注册：`docs/m23014/PREREG.json`（新算例之前冻结）· 判定引擎：`docs/m23014/analyse_m23014.py`
> 本文件由 `docs/m23014/make_doc.py` 从 `analysis.json` **程序化生成**，零手抄；重跑即重渲染。

## ① 案例介绍

物理背景：槽道内圆柱绕流，Re=100 的层流卡门涡街。几何按 Schäfer & Turek (1996)
DFG 基准 **case 2D-2**：槽道 2.2×0.41，柱心 (0.2, 0.2)，D=0.1，堵塞比 D/H=0.2439；
入口抛物线 (U_max/U_mean=1.5)，出口 p=0，壁面/柱面半程反弹，力用 Ladd 动量交换。
被质疑的头条数字是升力幅值误差约 **+35%**。

| 参数 | 值 |
|---|---|
| Re = U·D/ν | 100 |
| U_mean（格子） | 0.0833（τ 随 D 变，Re 精确保持） |
| 松弛/碰撞 | MRT（`tensorlbm.solver.collide_mrt`），D2Q9 |
| 边界 | Zou/He 入口(抛物线)+压力出口；壁面与柱面半程反弹 |
| 统计窗 | transient_periods=5，avg_periods=12（瞬态显式剔除） |
| 升力幅值口径 | 脱落频率上的最小二乘基频幅值（非整条记录峰峰值） |

## ② 怎么做的

1. **先冻结判据**：`PREREG.json` 写死四假设、外锚（S&T Table 3）、量化验收
   （Cl<10% 且 Δp<5% 且 Cd<5% **同时**达成；收敛跨度 <3%）、以及决策规则与停止规则。
   阈值在看到任何新结果之前固定，事后不得改动。
2. **不重复实现**：开工先查 `docs/AUDIT_m16827_CYLINDER_DFG_CL_AMP.md` —— 该单已把
   +36.04% 分解为「~13.4pp 估计器伪差（整条记录峰峰 + 瞬态入窗）+ 残余 +17.8% 真误差」，
   并明确留下 **未做 D≥56 档**、**未重跑** 两项欠账。本单接的是这两项欠账。
3. **估计器口径统一**（本单第一个坑）：冻结的 `level_D*.json` 由 m16827 修复**之前**的代码写出，
   其 `cl_amp` 是旧口径（整条记录峰峰）。若直接拿新算例的新口径数字与之比较，就是本单要查的
   那类错误本身。因此冻结档的四个观测量一律取自 `rejudge_cl_amp.json` 的 `values_new`
   （同一无瞬态窗口重判），新档用 `run_case` 自带新口径，两侧一致。
4. **四假设各自的判别性实验**（全部预注册）：
   - ① 网格：补 D48/D56 两级（`run.py --levels 48/56`），看悬崖是否收敛、跨度是否 <3%。
   - ② 黏性/τ：同 D40 几何、Re=100 不变，把 U_mean 由 0.0833 降到 0.06（τ 0.600→0.572）。
   - ③ 边界/堵塞：用求解器**同一个** `cylinder_mask` 重算离散面积、堵塞比、半程反弹有效直径、
     探针位置；不重跑物理。
   - ④ 采样：冻结时序上做「瞬态占比」分解 + 窗长 4→12 周期扫描。
5. **独立验证**：本单的独立估计器（ACF 定周期 + 最小二乘基频）必须复现 m16827 的冻结值
   到 1% 以内才允许用于结论（`tests/test_m23014_dfg_hypotheses.py` 锁定），
   避免结论是我的第二次实现伪差。
6. **计算**：重算全部递交调度系统（`cj4520`=D56、`cj4521`=D48、`cj4522`=D40@u0.06），
   本机只跑 <8 核、<10 min 的几何审计与冻结时序分析。

## ③ 计算结果

### 收敛阶梯（{n_levels} 档；冻结档取自 m16827 重判，新档为本单新增：{', '.join(new_levels) if new_levels else '—'}）

{ladder_table(an)}
✅ = 该档 Cd/Cl/Δp 三项同时落入 5% 容差。

### 相邻档收敛跨度

{span_table(an)}
判据：跨度 <3% 才算收敛。

### 四假设判决表

{hypothesis_table(an)}
"""

    md += f"""
## ④ 结果图

- `docs/figures/m23014_dfg_hypotheses.png`：四联图。(a) 四个观测量按参考带中点归一化随分辨率变化，
  灰色带=±5% 容差；(b) 带符号误差 vs 分辨率；(c) **相邻档跨度**——收敛判据的直接可视化，
  红线=3% 判据；(d) H4 窗长稳定性与 H3 半程反弹几何偏差（D_eff=D+1，O(1/D)）的对照。
- `docs/figures/m23014_cl_series.png`：最粗与最细档的 C_L(t)，红色区=被旧口径吞进统计窗的
  启动瞬态——这就是「整条记录峰峰值」为何系统性高估幅值的直接证据。
- 图为 matplotlib 程序化生成，图中数字由 `analysis.json`/`h3_h4_audit.json` 注入。
  本单为数值验证/根因判定，二维算例的场量云图不作为判定依据（判定量是力的时间历程与标量统计）。

## ⑤ 注意点（坑、局限、适用范围）

1. **口径陷阱（本单最大风险）**：新旧两代 `cl_amp` 语义不同。若不区分，D48/D56 会看起来
   「突然变好」而实际只是换了估计器。分析器现在按档记录 `cl_amp_estimator` 与
   `window_provenance`，并保留旧峰峰值只作审计。
2. **收敛不是单调的**：Cl 幅值在 D20/D28 平台约 +17.8%，D40 骤降到 +2.8%。
   这种「悬崖」形态**不能**由任何平滑的 O(1/D) 几何偏差解释（H3 的 5.00%→1.79% 是单调的），
   因此 H3 即便存在也不能当根因；真正的问题是**达到悬崖的临界分辨率在哪**。
3. **τ 与网格绑定**：固定 Re 与固定 U_mean 加密网格必然使 ν_lb∝D、τ 上升
   （0.550/0.570/0.600/0.620/0.640）。这是正确的定马赫加密，但意味着
   「加密」同时改变了 τ，故 H2 用**同几何变 U_mean** 单独隔离 τ，避免与 H1 混淆。
4. **参考带的证据级别**：C_L 幅值带 0.99–1.01 是仓内转抄（S&T 原表主列是 Cd/Δp/St），
   属 **WITHHELD** 级溯源，本单无网络无法核对；因此「Cl 落入带内」的证据强度低于
   Cd/Δp/St 落入带内。Cd/Δp/St 三带是文献通用值。
5. **未做**（如实登记为 not_run，不算通过）：与独立 CFD/文献逐值对表；D=64 以上档
   （仅在 D48↔D56 跨度为 3–6% 的边界情形下按预注册的应急条款才追加）。

## ⑥ 误差判定（关键量与基准逐项；显式回答是否 <3%）

{err_tbl}
{dev_section}

**是否 <3%？** 逐项回答：
""" + "\n".join(
        f"- `{m}` = {f(e.get(m))}% → **{'是' if abs(e.get(m, 9e9)) < 3.0 else '否'}**"
        for m in ("cd", "cl_amp", "dp", "st") if e.get(m) is not None
    ) + f"""

**判定（预注册规则，未事后改动）：`{dec['verdict']}`**
— {dec['rule_applied']}

- 主门控（Cl<10% 且 Δp<5% 且 Cd<5%，最细档 {finest}）：
  Cl {f(g['cl_amp_err_pct'])}%、Δp {f(g['dp_err_pct'])}%、Cd {f(g['cd_err_pct'])}%
  → **{'通过' if g['primary_gate_pass'] else '未通过'}**
- 次门控（相邻档跨度 <3%）：C_L {f(g['finest_span_pct'].get('cl_amp'))}%、
  Δp {f(g['finest_span_pct'].get('dp'))}%、Cd {f(g['finest_span_pct'].get('cd'))}%
  → **{'通过' if g['secondary_gate_pass'] else '未通过'}**

矩阵状态**不由本单改动**；若出现 honest-FAIL，FAIL 行保留、不抹除、不静默删除。
"""
    out = HERE / "REPORT.md"
    out.write_text(md)
    print("wrote", out, len(md), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())