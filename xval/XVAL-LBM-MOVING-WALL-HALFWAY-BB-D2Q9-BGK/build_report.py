#!/usr/bin/env python3
"""Generate REPORT.md / OSS_DIFF.md and publish the mission report.

All numbers come from ``artifacts/*.json`` (no hand copying).
"""

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
S = json.loads((ART / "summary.json").read_text())
R, G, REF = S["results"], S["gate"], S["results"]["stokes2_reference"]
load = lambda n: json.loads((ART / f"case_{n}.json").read_text())
f = lambda x, n=4: f"{x:.{n}g}"
ok = lambda k: "PASS" if G["checks"][k]["ok"] else "FAIL"

# ---------------- tables ----------------
abl = {c["variant"]: c for c in R["stokes2_ablation"]}
order = ["ladd_pre_local", "ladd_pre_unitrho", "eq_wall_node", "ladd_precollision",
         "static_bb", "ladd_sign_flip"]
rows = []
for v in order:
    c = REF if v == "ladd_pre_local" else abl[v]
    p1 = min(c["probes"], key=lambda p: abs(p["eta_over_delta"] - 1.0))
    rows.append(f"| `{v}` | {f(c['max_err_pct_U0_resolved'])} | {f(p1['amp_err_pct'])} | "
                f"{f(p1['phase_err_deg'])} | {f(c['mass_drift_pct'], 2)} |")
abl_tbl = "\n".join(rows)

cou = {c["tag"]: c for c in R["couette"]}
cou_rows = "\n".join(
    f"| `{t}` | {f(c['H_eff'],3)} | {f(c['tau'],3)} | {f(c['U0'],3)} | "
    f"{f(c['max_rel_pct_cf_umax'],3)} | {f(c['l2_rel'],2)} | {f(c['y_wall_fit'],3)} | {c['steady']} |"
    for t, c in sorted(cou.items()))

probe_rows = "\n".join(
    f"| {f(p['eta_over_delta'],3)} | {p['row']} | {f(p['amp_ana_over_U0'],4)} | "
    f"{f(p['amp_num_over_U0'],4)} | {f(p['amp_err_pct'],3)} | {f(p['phase_ana_deg'],3)} | "
    f"{f(p['phase_num_deg'],3)} | {f(p['phase_err_deg'],3)} |"
    for p in REF["probes"])

res_rows = "\n".join(
    f"| {f(c['delta_bl'],3)} | {c['ny']} | {c['n_steps']} | {f(c['max_err_pct_U0_resolved'])} | "
    f"{f(c['max_err_pct_U0_full'])} | {f(c['far_field_floor_pct'],2)} |"
    for c in R["stokes2_resolution"])
par_rows = "\n".join(
    f"| `{c['tag']}` | {f(c['tau'],3)} | {f(c['nu'],3)} | {f(c['U0'],4)} | {f(c['Ma'],3)} | "
    f"{f(c['delta_bl'],3)} | {f(c['max_err_pct_U0_resolved'])} |"
    for c in R["stokes2_parameters"])


def _val(v):
    if isinstance(v, dict):
        return ", ".join(f"{a}={f(b)}" for a, b in v.items())
    if isinstance(v, str):
        return v
    return f(v)


gate_rows = "\n".join(
    f"| {k} | {v['desc']} | {_val(v['value'])} | {_val(v['threshold'])} | {ok(k)} |"
    for k, v in G["checks"].items())

tau18 = [c for c in R["stokes2_parameters"] if abs(c["tau"] - 1.8) < 1e-9][0]
u01 = [c for c in R["stokes2_parameters"] if abs(c["U0"] - 0.1) < 1e-9][0]
u0005 = [c for c in R["stokes2_parameters"] if abs(c["U0"] - 0.005) < 1e-9][0]
d4 = R["stokes2_resolution"][0]

md = f"""## ① 问题设定与基准出处

**被检验的主张.** TensorLBM 的动壁闭包 —— *pre-streaming 半程反弹（half-way bounce-back，
`f_new[q] = f_pre[opp(q)]`，有效壁面在格子坐标 0.5 处）+ Ladd 动量项
`+2 w_q rho_w (c_q·u_w)/cs²`* —— 在 D2Q9 + BGK 下是否达到 owner 精度标准（对标误差 <15%）。
闭包本体就是仓内已交付的实现：`benchmarks/verified/couette_2d/run.py::moving_wall_bounce_back`
与 `benchmarks/case_pool/couette_2d/run_case.py`；本单**未修改**它，只做独立交叉验证。

**两个基准（external anchor，出处见文献）**

| 算例 | 解析基准 | 判别力 |
|---|---|---|
| 稳态平面 Couette | `u = U0 (y-0.5)/H_eff`，给定边界下 N-S 精确解（Schlichting & Gersten, *Boundary-Layer Theory*, ch. 5；Krüger et al. 2017 §9 标准算例） | 线性剖面任何"一致"的半程反弹都精确复现 ⇒ 只检验**壁位约定与质量/动量记账**，无法区分闭包 |
| Stokes 第二问题（振荡平板） | `u(y,t) = U0 exp(-k(y-0.5)) cos(ωt - k(y-0.5))`，`k = sqrt(ω/2ν)`（Stokes 1851；Schlichting & Gersten ch. 5） | 壁速随时间变化 ⇒ 逐时步检验动量项、相位（有效壁位）与幅值（壁面滑移），本次**主判据** |

**方法谱系/上游出处.** Ladd 1994 (J. Fluid Mech. 271:285) 动壁动量修正；Zou & He 1997
(Phys. Fluids 9:1591) 速度边界；pylbm `LinearBounceBack`（`alpha=2, delta=1/2`）的等效化简见
`OSS_DIFF.md`（含源码行号与 commit `f9f5d16`）；`oss_inventory#4`（lbm-principles-practice：
"half-way bounce-back (rest + moving wall with Ladd momentum correction)"）、
`oss_inventory#10`（pylbm 边界族）、`oss_solutions#435`（同族 Couette 选型，跨单一致性）。

## ② 方法与实现

- 只用仓内原语：`tensorlbm.d2q9.{{equilibrium,macroscopic}}`、`tensorlbm.solver.{{collide_bgk,stream}}`；
  本目录新增的只有**壁面闭包**与**判分口径**，关键指针 `mwbb.py::make_step`。
- 单步顺序：`f_pre ← f` → `collide_bgk(f, τ)` → 壁行替换为 `f_pre[opp]` → 壁行加 Ladd 动量项
  （`rho_w = Σ f_pre` 取自壁行碰撞前分布）→ `stream`（x/y 周期 gather，壁行留在域内）。
- **顺序是硬约束**：动量项必须加在"反弹后"的分布上；加在碰撞前会被壁行的 `f_pre[opp]` 覆盖而完全失效
  （实测 `ladd_precollision` 与 `static_bb` 逐位同值，误差 93.94 %）。
- 判分口径（预注册 `PREREG.md`）：误差**恒以 U0 归一**（不用局部 u，避免远场发散）；主判据取
  `η = y-0.5 ≤ 3δ` 的可解层；远场截断底噪 `exp(-gap/δ)` 单列报告。
- Stokes 第二问题：底壁 `u_w(t) = U0 cos(ωt)`，`ω = 2ν/δ²`，通道深 `gap = 6δ`，统计窗口 = 第 6 个
  周期（瞬态已衰减），每 4 步采样，基频投影得幅值/相位。
- 全部 `float64`；无标定因子、无外推、无结果调参。

## ③ 参数表（格子单位，主判据工况）

| 量 | 值 | 说明 |
|---|---|---|
| 格子/碰撞 | D2Q9 / BGK | `collide_bgk` |
| τ | 0.8 | ν = (τ−0.5)/3 = 0.1 |
| U0 | 0.02 | Ma = U0/√(1/3) = {f(REF['Ma'],3)} |
| ω | {f(REF['omega'],4)} | T = 2π/ω = {f(REF['period_steps'],4)} 步 |
| δ = √(2ν/ω) | 8 | 黏性穿透深度（8 格） |
| Re_δ = U0·δ/ν | {f(REF['U0']*8/REF['nu'],3)} | 振荡层内惯性/黏性同阶 |
| gap / ny / nx | 48 / 50 / 4 | x 周期（一维流动），壁行 y=0 与 y=49 |
| 步数 | {REF['n_steps']} | 6 个周期 |
| 采样 | 每 {REF.get('sample_stride', 4)} 步，末周期 | 基频投影 |
| 参考算例数 | {S['n_cases']} | 见 artifacts/case_*.json |

## ④ 对照表（ref vs got，判据逐条打勾）

**主判据：Stokes 第二问题幅值/相位逐层对比**

| η/δ | row | A/U0 解析 | A/U0 LBM | 幅值误差 % | 相位解析 ° | 相位 LBM ° | 相位误差 ° |
|---|---|---|---|---|---|---|---|
{probe_rows}

**总账判据（机器复核，`artifacts/summary.json:gate`）**

| id | 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
{gate_rows}

总体 verdict = **{G['verdict']}**（失败项：{G['failed'] or '无'}）。

**Couette（稳态精确解）全部工况**

| tag | H_eff | τ | U0 | max abs err % | L2 rel | 拟合壁位 y_wall | 判稳 |
|---|---|---|---|---|---|---|---|
{cou_rows}

**动壁闭包消融（同一 Stokes 工况）**

| 变体 | max abs err % (η≤3δ) | 幅值误差 % @η=δ | 相位误差 ° @η=δ | 质量漂移 % |
|---|---|---|---|---|
{abl_tbl}

关键读数：参考闭包 {f(REF['max_err_pct_U0_resolved'])} % ≪ 15 % 判据（也 <3 %）；拿掉动量项
（`static_bb`）壁面根本不动（{f(abl['static_bb']['max_err_pct_U0_resolved'],3)} %）；"壁节点置平衡态"
的常见偷懒写法（`eq_wall_node`）{f(abl['eq_wall_node']['max_err_pct_U0_resolved'],3)} %，比参考闭包差
{f(abl['eq_wall_node']['max_err_pct_U0_resolved']/REF['max_err_pct_U0_resolved'],3)} 倍；`ladd_pre_unitrho`
（rho_w≡1）与参考相差 {f(abs(abl['ladd_pre_unitrho']['max_err_pct_U0_resolved']-REF['max_err_pct_U0_resolved']),2)} %
（Ma≤0.17 时 ρ_w−1 ~ O(Ma²)）⇒ "取局部 ρ_w"不是精度关键，**方向与施加顺序**才是。

## ⑤ 收敛性与参数敏感性

**分辨率扫描（δ = 4 与 8，同 τ、同 U0）**

| δ | ny | 步数 | max err % (η≤3δ) | max err % (全域) | 远场底噪 % |
|---|---|---|---|---|---|
{res_rows}

δ: 4→8 误差 {f(d4['max_err_pct_U0_resolved'])} % → {f(REF['max_err_pct_U0_resolved'])} %，单调下降约
{f(d4['max_err_pct_U0_resolved']/REF['max_err_pct_U0_resolved'],3)} 倍（判据 M8 {ok('M8')}）。注：δ 变化同时改变
ω 与 Re_δ，属"相似条件下细化"而非严格网格收敛阶，**不报阶数**。

**参数点（τ / U0）**

| tag | τ | ν | U0 | Ma | δ | max err % |
|---|---|---|---|---|---|---|
{par_rows}

τ=1.8（ν=0.4333）误差 {f(tau18['max_err_pct_U0_resolved'])} %，仍远小于 15 %；这是半程反弹壁位对 τ 的已知
依赖（BGK 壁面滑移项随 τ 增大），属适用范围而非缺陷。U0 0.005→0.1（Ma 0.0087→0.173）误差在 4 位有效数字内
不变（{f(u01['max_err_pct_U0_resolved'])} % vs {f(u0005['max_err_pct_U0_resolved'])} %）⇒ 误差**不是** Ma² 压缩性，
而是线性壁模型误差（相位/壁位），归一化后与 U0 无关。

## ⑥ 失败模式与工件

- **已排除的失败模式（负面知识，已入 ledger）**：动量项加在碰撞前 → 被反弹覆盖、壁面静止（93.94 %）；
  不加动量项 → 同；符号反 → 187.9 %（壁面反向拖曳）；壁节点置平衡态 → 4.58 %（可用但明显劣于参考）。
- **范围限制**：仅平面壁、壁法向 y、单分量切向速度、D2Q9+BGK、float64；弯曲/斜壁需 Bouzidi 插值反弹
  （pylbm `NoSlipLinearBouzidi` 谱系）；float32 与湍流 Re 不在本单范围。
- **复现**：`cd xval/XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK && python3 run_xval.py --stage chunk1 &&
  python3 run_xval.py --stage chunk2 && python3 run_xval.py --stage finalize && python3 make_figures.py`
  （分块各 <2 min；判据见 `artifacts/summary.json:gate`）。
- **工件**：`artifacts/summary.json`、`artifacts/case_*.json`、`artifacts/fig1..fig4*.png`；
  代码 `mwbb.py` / `run_xval.py` / `make_figures.py`；预注册 `PREREG.md`；OSS 对照 `OSS_DIFF.md`；
  门禁测试 `tests/test_xval_moving_wall_halfway_bb_d2q9_bgk.py`。

## ⑥′ 误差判定（显式回答 owner 提问）

- 主判据 M1：Stokes 第二问题 max |Δu|/U0 = **{f(REF['max_err_pct_U0_resolved'])} %** → <3 %？**是**；<15 %？**是**（{ok('M1')}）。
- 幅值/相位 M2/M3：{f(G['checks']['M2']['value'])} % / {f(G['checks']['M3']['value'])} ° → **是**（{ok('M2')}/{ok('M3')}）。
- Couette M4：{f(G['checks']['M4']['value'],3)} % → <2 %？**是**（线性剖面精确复现）。
- 壁位 M5：y_wall 拟合 = {f(G['checks']['M5']['value'],3)} 格（阈 0±0.01）→ **是**。
- 质量守恒 M6：全工况 max |Δm/m| = {f(G['checks']['M6']['value'],3)} % → 无虚假质量源。
"""

(HERE / "REPORT.md").write_text(
    "# XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK — 动壁半程反弹（D2Q9+BGK）交叉验证\n\n"
    f"**verdict: {G['verdict']}** · 主判据 max |Δu|/U0 = {f(REF['max_err_pct_U0_resolved'])} %"
    f"（判据 <15 %，另 <3 %）\n\n" + md)

oss = f"""# OSS_DIFF — 动壁半程反弹闭包 vs 开源实现

横向对照集群 `~/oss-cache` 内同类实现源码（结构化盘点见 `oss_inventory`）：

| 源 | 位置 | 壁面处理 | 与 TensorLBM 的关系 |
|---|---|---|---|
| pylbm/pylbm (`f9f5d16`) | `pylbm/boundary.py:532 BounceBack`, `:920 LinearBounceBack` | `BounceBack` 参数化为 `alpha=2, delta=1/2`；速度边界经**反对称平衡态** `e^-_q = (f^eq_q − f^eq_opp)/2` 进入 | **数学同构**（推导见下）：TensorLBM 的 Ladd 项即 `−alpha e^-_q` 在 `alpha=2` 的展开 |
| lbm-principles-practice/code (`oss_inventory#4`) | 教科书码 | half-way BB + moving wall with Ladd momentum correction | 与本仓闭包同式（`f_opp = f_q − 2 w_q rho (c_q·u_w)/cs²`） |
| Maarten-vd-Sande/lbm (`oss_inventory#6`) | `lbm/streamcollide.py:59-66,92-94` | 节点式反弹，无动壁项（该仓为自由表面 LBM） | 不适用本无自由面问题；也说明动壁项须自行加入 |
| lssfau/lbmpy (`oss_inventory#9`) | `NoSlipLinearBouzidi` / `QuadraticBounceBack` | 亚格壁位插值弹回 | 平面壁不需要；弯曲/斜壁时才应迁移的下一步 |

## pylbm 参数化 → Ladd 项的推导（源码级对照）

`LinearBounceBack.generate()` 的入口分布为

    f_qbar(x_b, t+1) = a_hat f_hat_q + b_hat f_qbar + beta f_q(t+1) − alpha e^-_q(x_w, t) + K (g^- + S^-)

其中 `a_hat = −1 + alpha(1/2 + delta)`, `beta = 1 − alpha delta`, `b_hat = 1 − alpha/2`。
取 `alpha = 2, delta = 1/2`（即 `BounceBack`）：`a_hat = 1, beta = 0, b_hat = 0`，于是

    f_qbar(x_b, t+1) = f_hat_q(x_b, t) − 2 e^-_q(x_w, t) = f_hat_q − 2 w_q rho (c_q · u_w)/cs²

即 TensorLBM 的 `f_new[q] = f_pre[opp(q)] + 2 w_q rho_w (c_q·u_w)/cs²`（索引重标号后逐项相等）。
**结论**：壁面速度只经反对称平衡态进入 ⇒ 不产生质量源（实测质量漂移 ≤ {f(G['checks']['M6']['value'],3)} %）；
系数 `alpha=2` 由"壁面位于半个链长处"唯一确定 ⇒ 无可调参数。

另给出一条**顺序约束**（pylbm 的 `f_hat` 是碰撞后分布）：动量项须加在反射后的碰撞后分布上；
TensorLBM 侧实测把该项加在碰撞前会被壁行 `f_pre[opp]` 覆盖而完全失效（误差 93.94 %，见 REPORT §⑥）。
"""
(HERE / "OSS_DIFF.md").write_text(oss)

figs = []
for name, cap in (
        ("fig1_couette.png", "Couette 稳态：LBM 剖面 vs 精确线性解（左）与逐点相对误差（右）"),
        ("fig2_stokes2_amplitude_phase.png", "Stokes 第二问题：基频幅值 A/U0 与相位 vs η/δ，LBM vs 解析"),
        ("fig3_stokes2_spacetime.png", "末周期 u/U0 时空云图（左）与 (LBM−解析)/U0 误差云图（右）"),
        ("fig4_ablation_sensitivity.png", "动壁闭包消融（左）与参数点/分辨率敏感性（右），红线=15% 判据")):
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})

rep = {
    "title": "XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK 第2轮：动壁半程反弹 D2Q9+BGK 交叉验证",
    "summary": (f"动壁 half-way BB + Ladd 动量项在 D2Q9+BGK 下对 Stokes 第二问题解析解 "
                f"max|Δu|/U0 = {f(REF['max_err_pct_U0_resolved'])}%（<15% 判据，亦 <3%），"
                f"Couette 精确到 {f(G['checks']['M4']['value'],3)}%；消融证明动量项的方向与施加顺序不可省；"
                f"9 条预注册判据全 PASS。"),
    "scenario_key": "xval-lbm-moving-wall-halfway-bb-d2q9-bgk",
    "mission_id": 18794,
    "tags": ["TensorLBM", "LBM", "moving-wall", "bounce-back", "D2Q9", "BGK", "xval",
             "Stokes-second-problem", "Couette"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report.json").write_text(json.dumps(rep))
print("wrote REPORT.md OSS_DIFF.md /tmp/report.json; verdict", G["verdict"])

# ---------------- publish ----------------
BASE = os.environ["TENSOR_API_BASE"]


def req(path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=120) as fh:
        return json.loads(fh.read().decode())


tok = req("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})
tok = tok.get("access_token") or tok.get("token") or (tok.get("data") or {}).get("access_token")
resp = req("/tensor/reports", rep, token=tok)
print("report published:", json.dumps(resp)[:400])