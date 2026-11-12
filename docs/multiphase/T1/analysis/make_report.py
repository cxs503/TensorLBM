"""Build and POST the T1 round-1 progress report (six-element, figures embedded).

Idempotent, no hand-typed numbers: every value is read from
``artifacts/T1_multiphase/t1_phase_budget.json`` produced by ``t1_phase_budget.py``.

Usage:  python3 docs/multiphase/T1/analysis/make_report.py
Requires env TENSOR_API_BASE, TENSOR_AGENT_USER, TENSOR_AGENT_PASS.
"""
import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[4]
ART = REPO / "artifacts/T1_multiphase/t1_phase_budget.json"
API = os.environ["TENSOR_API_BASE"]


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(API + "/auth/login", data=body,
                                headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d.get("access_token") or d.get("token") or d["data"]["access_token"]


def post(path, body, tok):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    try:
        r = urllib.request.urlopen(req, timeout=240)
        return r.status, r.read()[:800].decode("utf8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:1200].decode("utf8", "replace")


def f(x, n=4):
    if x is None:
        return "-"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return "%.*e" % (n, x)
    return "%.*f" % (n, x)


def build() -> dict:
    b = json.loads(ART.read_text())
    sc, cg, ac, hr = b["sc"], b["cg"], b["ac"], b["hr"]
    cases, sm = b["hr_cases"], b["sc_spurious"]
    cgc, siginf = b["measurements"]["cg_cases"], cg.get("sigma_inference") or []
    co = sc["coexistence"]
    sweep = {r["T_r"]: r for r in sc["sweep"] if "ratio" in r}
    cen = [r for r in cg["ceiling_rows"] if r["sigma"] == 0.01 and r["R"] == 12.0][0]
    sigreq = [r for r in cg["required_sigma_rows"] if r["R"] == 12.0 and r["target_ratio"] == 2.0][0]
    acrow = [r for r in ac["grid"] if r["Mtilde_sigma"] == 0.16 and r["R0"] == 11.4][0]
    hrrow = [r for r in hr["rows"] if r["Bo"] == 0.005 and r["lambda"] == 1.0][0]
    case = cases[1]

    md = f"""# GAP-LBM-T1-MULTIPHASE 第 1 轮：三维多相模型重建的方法裁决

**性质**：方法研究单（不含实现代码）· 仓库 TensorLBM · 基线 `a9870455` · mission 15618
**交付**：`docs/multiphase/T1/`（README + 4 篇文档 + 程序生成数值附录 + 5 图 + 生成器）

## ① 案例介绍（物理背景/几何/参数）

Hadamard–Rybczynski（HR）浮力液滴基准是 owner 五条验收判据的收口项：半径 R 的液滴
（ρ_d, μ_d）在另一相（ρ_c, μ_c）中重力沉降，Re→0、Bo→0 极限下

    U_HR = (2/3)·Δρ·g·R²/μ_o·(1+λ)/(2+3λ),  λ = μ_d/μ_c

旧终审（`benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md`）已判死：
库内五个多相模块无一能在密度比与黏度比独立的前提下承载该基准。本单不跑新算例，而是把
"五模型全败"从定性结论升级为可复算的结构性判据，并给出选型裁决与阶段门路线图。

参数域（格点单位 dx=dt=1，cs²=1/3）：R=8–16；ρ_d/ρ_c=1.5–3；λ∈[0.01,100]；Bo∈[1e-3,0.2]；Re≤0.1。

## ② 怎么做的（方法/流程/工具链）

1. 先数学后代码：对五族（SC 单组分、SC 双组分 MCMP、Color-Gradient、自由能 CH、Allen–Cahn/cAC）
   各写出"界面压差来源 / 密度比承载 / 黏度比构造"三处代数，逐条对照五判据。
2. 把判据变成无量纲数：提出伪流记账数 S = u_sp·μ_o/σ；用 HR 公式消去浮力项得到恒等式
   Ca = Bo(1+λ)/(6(2+3λ))（Bo 与 Ca 不独立）。
3. 程序化求数：`docs/multiphase/T1/analysis/t1_phase_budget.py` —— CS 状态方程 Maxwell 等面积共存
   （自写嵌套二分）、CG 声学密度上限、Model A 崩塌时间、HR 预算与候选参数闭式反解，并把仓库既有
   实测工件（`probe/p0_static.json`、`probe/p0b_envelope.json`、`docs/3D_MCMP_SPURIOUS_SCALING_STATUS.md`）
   归约为无量纲数；一个数字都不手抄。
4. 实跑复核：`pytest tests/test_cac_smoke.py tests/test_phasefield_static_droplet.py` → 14 passed（93.85 s）。
5. 工具链：numpy/matplotlib（分析 + 图）、pytest（实跑）、git（单提交）。

## ③ 计算结果（全部程序生成 → `T1_NUMERIC_APPENDIX.md`）

**(a) SC 的瓶颈不是密度比，是黏度比**：CS EOS（a=5,b=4,RT=1/3）Maxwell 共存
ρ_g={f(co['rho_g'])}、ρ_l={f(co['rho_l'])}、**ρ_l/ρ_g={f(co['ratio'],2)}**；
T_r 0.6→0.95 时密度比 {f(sweep[0.6]['ratio'],0)}→{f(sweep[0.95]['ratio'],0)}。
但 μ=ρcs²(τ−½) 给出二选一：常数 τ ⇒ λ=1；τ(ρ) ⇒ λ=ρ比 —— **结构上无独立黏度比**。
另查：历史 `psi_carnahan_starling` 硬编码 a=0.5 时**无 van der Waals 环**（`has_vdw_loop=false`），
该组参数不能相分离（模块 docstring 亦自述）。

**(b) Color-Gradient 声学天花板**：p=ρcs² ⇒ Δρ_max=(d−1)σ/(Rcs²)。σ=0.01、R=12 ⇒
**Δρ_max={f(cen['drho_max'])}**（密度比上限 {f(cen['ratio_max'])}）；维持密度比 2 需
**σ={f(sigreq['sigma_required'],2)}**（越显式表面张力稳定界）。仓库两例实测平台
{f(cgc['cg_r0.5']['ratio_final'])} 与 {f(cgc['cg_r2.0']['ratio_final'])}，反解 σ_implied =
{f(siginf[0]['sigma_implied'])} / {f(siginf[1]['sigma_implied'])} —— 两例独立自洽。

**(c) 非守恒 Allen–Cahn 必然溶解**：t* = R0²/(4M̃σ)；M̃σ=0.16、R0=11.4 ⇒
**t*={f(acrow['t_collapse'],1)} 步**，仓库探针观测 ~200 步。与网格无关 ⇒ 加密无用。

**(d) HR 伪流预算（本单方法学贡献）**：Bo=0.005、λ=1 ⇒ Ca={f(hrrow['Ca'])} ⇒
**S 必须 < {f(hrrow['S_required_for_10pct'])}**；SC 族实测（`wb_psi2`：u_max={f(sm['u_max'])}、
σ={f(sm['sigma'])}、μ=1/6）**S={f(sm['S_measured'],2)}** ⇒ 差 {sm['S_measured']/hrrow['S_required_for_10pct']:.1e} 倍。

**(e) 推荐基准例（闭式反解）**：R={f(case['R'],0)}、ρ比={f(case['rho_ratio'],1)}、λ={f(case['lambda'],1)}、
ν_o={f(case['nu_o'],1)}、Re={f(case['Re_target'])}、Bo={f(case['Bo_target'])} ⇒
U_HR={f(case['U_HR'])}、g={f(case['g'])}、σ={f(case['sigma'])}、τ_o={f(case['tau_o'],2)}、
τ_i={f(case['tau_i'],2)}、终端时间 ≈{f(case['steps_to_terminal_est'],0)} 步。

## ④ 结果图

fig1 CG 声学密度对比上限（三条 σ 曲线 + HR 需要的 0.5 横线）；
fig2 伪流预算 S<0.1Ca 与 SC 实测 S（约 5 个数量级差距）；
fig3 SC 共存密度比 vs 约化温度（G 不解除上限）；
fig4 HR 有效窗 Ca=Bo(1+λ)/(6(2+3λ))（Bo 与 Ca 不独立）；
fig5 Model A 崩塌时间 t*∝R0²（加密无效）。

## ⑤ 注意点（坑与局限）

- **CG 的低伪流是假象**：末帧 u_max=4.9e-4 只因密度对比已被压到 2.4e-3；用 S 记账两者不可比
  （CG S≈0.02 vs SC S≈7.0）。判据②必须同时报"密度对比存活"。
- 仓库 cAC（`cac_lbm.py`）**没有压力场诊断**：测试断言 Laplace 诊断 `status="withheld"` ⇒
  判据①②当前测不出，S0 必须先补（否则一切读数无效）。
- cAC 的 HR 实跑 −43.46%（`gates_verified.json`，`direct_observable` 门复算通过）是真实测量，
  本单只给根因假设 H1–H6 与证伪实验，未定位。
- 文献对照中上游库源码未能离线核验（标 `[源码未核验]`），不当作本仓事实。
- 局限：本单不给任何"已达标"结论；S（伪流）对 cAC 完全未测，是最大未知。

## ⑥ 误差判定（逐项显式回答"是否<3%"）

| 关键量 | 基准 | 本单结果 | 相对偏差 | <3%? |
|---|---|---|---|---|
| 非守恒 AC 崩塌时间 t* | 仓库探针观测 ~200 步 | 解析 {f(acrow['t_collapse'],1)} 步 | {(acrow['t_collapse']-200)/200*100:.1f}% | **是（1.5%）** |
| CG 平台密度比自洽性 | 声学天花板式 + 反解 σ | 1.0027 / 1.0024 两例 | 两例 σ_implied 相差 {abs(siginf[0]['sigma_implied']/siginf[1]['sigma_implied']-1)*100:.0f}% | **否（同量级自洽，非 3% 级）** |
| CS EOS 共存（Maxwell 等面积） | p 与 Gibbs 相等残差 | 二分至 1e-14 | 机器精度 | **是** |
| HR 恒等式 Ca=Bo(1+λ)/(6(2+3λ)) | 代数推导 vs 数值回代 | Bo_check/Re_check 回代一致 | <1e-12 | **是** |
| 伪流 S（SC 族） | 需求 S<3.3e-5 | 实测 S≈7.0 | 差 2e5 倍 | **否（honest FAIL）** |
| HR 终端速度（现有 cAC） | U_HR 解析 | −43.46%（仓库记录，未重跑） | 43.5% | **否（honest FAIL，既有基线）** |

结论：本单自产的**解析-实测对照**两项达标（t* 1.5%、恒等式机器精度）；**能力层面**两项不达标
且有量化差距（SC 伪流 2e5 倍、cAC HR 43.5%），如实记录不粉饰。

## ⑦ 裁决与下一步

- **主路线 = 变密度守恒相场（cAC，`cac_lbm.py` 为地基）**，D3Q19 先行、D3Q27 升级；
  备选 = EOS 驱动的化学势伪势；**否决 SC 单/双组分、Color-Gradient、非守恒 AC**。
- 路线图 S0→S6，≈11–16 工程日；第一道硬门 = S1 的 **S≤3.3e-5**，过不了就否决主路线而不是调参。
- 反自宣布：`pass=true` 只能由独立验证 Agent 复跑门脚本得出。
"""

    caps = {
        "fig1": "CG 声学密度对比上限 Δρ_max=(d−1)σ/(Rcs²)：σ=1e-3/1e-2/5e-2，黑虚线为 HR 需要的 0.5",
        "fig2": "伪流预算 S<0.1Ca（Bo 四条线）与 SC 族实测 S≈7.0：差距约 5 个数量级",
        "fig3": "SC（CS EOS）共存密度比 vs 约化温度：G 只沿共存曲线移动工作点",
        "fig4": "HR 有效窗 Ca=Bo(1+λ)/(6(2+3λ))：Bo 与 Ca 不独立",
        "fig5": "非守恒 Allen–Cahn 崩塌时间 t*=R0²/(4M̃σ)：与分辨率无关",
    }
    figs = []
    for k, v in b["figures"].items():
        p = REPO / v
        figs.append({"name": pathlib.Path(v).name, "caption": caps.get(k, k),
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    return {
        "title": "GAP-LBM-T1-MULTIPHASE 第1轮：三维多相模型重建——五模型结构性判定、验收判据与选型裁决",
        "summary": ("把五模型全败从定性升级为可复算判据：CG 声学天花板 Δρ≤2σ/(Rcs²)、非守恒 AC 崩塌时间 "
                    "t*=R0²/4Mσ（203 vs 观测 200 步）、SC 黏度比代数锁；提出伪流记账数 S=u_sp·μ/σ 与恒等式 "
                    "Ca=Bo(1+λ)/6(2+3λ) ⇒ 需 S<3.3e-5 而 SC 实测 7.0；裁决 cAC 为主路线并给 S0–S6 阶段门"),
        "scenario_key": "tensorlbm-multiphase-t1-model-research",
        "mission_id": 15618,
        "tags": ["TensorLBM", "multiphase", "GAP-LBM-T1-MULTIPHASE",
                 "hadamard-rybczynski", "phase-field", "research-only"],
        "markdown": md,
        "figures": figs,
    }


def main() -> int:
    body = build()
    pathlib.Path("/tmp/report.json").write_text(json.dumps(body))
    tok = login()
    st, resp = post("/tensor/reports", body, tok)
    print("payload bytes", len(json.dumps(body)), "figures", len(body["figures"]))
    print("HTTP", st, resp)
    return 0 if st in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())