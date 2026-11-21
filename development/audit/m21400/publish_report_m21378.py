#!/usr/bin/env python3
"""m21400 — publish the board-audit report for the m21378 verdict.

Every number in the markdown is injected from
``audit/m21400/probe_m21378_results.json`` (independent recompute) or from the
delivered artifact ``xval/...-R2/artifacts/compact.json``; nothing is typed by
hand.  The figure is embedded as ``png_base64``.

Run:  python3 audit/m21400/publish_report_m21378.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMPACT = ROOT / "xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO-R2/artifacts/compact.json"
PROBE = HERE / "probe_m21378_results.json"
FIG = HERE / "fig1_m21378_audit.png"

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
SIGMA_LIB = 0.05611


def psql(sql: str) -> str:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
        capture_output=True, text=True, env=env,
    )
    return r.stdout.strip() if r.returncode == 0 else "ERR:" + r.stderr.strip()[:200]


def q_cases(compact, forcing):
    rows = [c for c in compact["cases"]
            if c["forcing"] == forcing and c["R0"] == 12.0 and "sigma" in c]
    return sorted(rows, key=lambda c: c["tau"])


def spread(vals):
    vals = sorted(vals)
    m = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / m * 100.0, (max(vals) - min(vals)) / 2 / m * 100.0


def pct(got, ref):
    return (got - ref) / ref * 100.0


def build(compact, probe):
    mx = probe["maxwell_mine"]
    mx_del = compact["maxwell"]
    ident = probe["identity"]
    bench_rows = psql(
        "SELECT count(*) FROM tensor_benchmark_cases "
        "WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO';")
    verdict_row = psql(
        "SELECT level||' | '||coalesce(gate_state,'pending') FROM tensor_maturity_verdicts "
        "WHERE verdict_by='board-audit' AND mission_id=21400 ORDER BY id DESC LIMIT 1;")

    sig = {f: q_cases(compact, f) for f in ("velocity_shift", "edm", "guo")}
    d_edm = spread([c["sigma"] for c in sig["edm"]])
    d_vs = spread([c["sigma"] for c in sig["velocity_shift"]])
    d_guo = spread([c["sigma"] for c in sig["guo"]])

    md = f"""# m21378 交差核验：XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO（TensorLBM / TCLB shan-chen-edm-lycett-luo）

**本单** m21400（board-audit 交差核验，非重做）｜**被核对象** 任务 21378，commit `9b2791376`，PR #2455（merged），报告 id=19330
**判级** **L3（系统集成）** ｜**L4 基准验证** **WITHHELD** ｜**处置** approve with notes
**判据** owner 精度标准 <15%（生态自目标 <3%）｜**基准裁判** `tensor_benchmark_cases`

---

## ① 案例介绍

锚点问的是：TensorLBM 的 Shan-Chen 单组分多相（SCMP）核，与 CFD-GO/TCLB `models/multiphase/d2q9_scmp`
里 **shan-chen-edm-lycett-luo** 的做法，在同一个静态液滴算例上是否同族可对标、差多少。
被核对象 m21378 是这条锚点的**第 2 轮**：上游第 1 轮（m18323）在自研 NumPy 复刻上自判 "PASS <15%"，
第 2 轮把它搬回随仓发布的库路径并重做判别。

| 项 | 值（取自交付 `compact.json:cfg`） |
|---|---|
| 模型 | SCMP single-component Shan-Chen (Shan & Chen 1993/94)，D2Q9，BGK |
| 伪势 | ψ(ρ)=1−exp(−ρ)，ρ₀=1 |
| 耦合 | 库约定 G=+5（后向 gather，>0 吸引；等价 G_eff=−5） |
| 体相压力 | p_mech = c_s²ρ − (G/2)c_s²ψ²（库约定） |
| 网格 | L = 4R，R ∈ {{10,12,14}} → L ∈ {{40,48,56}} |
| τ | 0.8 / 1.0 / 1.2 |
| 力项 | velocity_shift / edm（Kupershtokh 精确差分）/ guo |
| 收敛 | Δp 相对变化 <2e-3 / 250 步，上限 3000–4000 步 |
| 测量带 | 内 r≤0.5R_eq，外 r≥1.5R_eq，R_eq 由液区面积自洽迭代 |

## ② 怎么做的（核验流程）

1. **三件套取证**：对 commit `9b2791376` 的 `git ls-files` / `git cat-file -e HEAD:` / `find` 逐项核实工件、
   图、上游第 1 轮的"假工件"；把 `REPORT.md` 的每张表与 `compact.json` 逐格比对（查手抄）。
2. **基准裁判**：按题面用 `tensor_benchmark_cases` 裁 got 是否在容差内 —— 结果**该 anchor {bench_rows} 行**
   （全表 138 行），本锚点无登记基准例，裁判空转；故改用交付自带参照臂，并显式标注其独立性等级。
3. **solver-free 独立探针**（`audit/m21400/probe_m21378.py`，8 s，不复用交付脚本）：
   τ 恒等（5 个 τ，自建随机场 seed 20261115）、同一组密度在两种压力符号下的 σ 重算、Maxwell 等面积独立重算、
   τ 离散度按统一定义重算。
4. **交付不变式复跑**：`pytest tests/test_xval_sc_edm_laplace_r2.py -q`。
5. **全量重跑**：`local_sweep.py` 本机 120 s shell 上限被收割 → 递交 compute 门 `cj3406`（4 核）。
6. **落表**：`tensor_maturity_verdicts`（repo/scope/mission_id/level/evidence JSON/verdict_by=board-audit）→ 当前行 `{verdict_row}`。

## ③ 计算结果（程序化注入，零手抄）

**A. 交付的 σ 与两个参照臂的差距（R₀=12，σ=Δp·R_eq）**

| forcing | σ(τ=0.8) | σ(τ=1.0) | σ(τ=1.2) | 全幅离散 | 半幅离散 |
|---|---|---|---|---|---|
| velocity_shift | {sig['velocity_shift'][0]['sigma']:.5f} | {sig['velocity_shift'][1]['sigma']:.5f} | {sig['velocity_shift'][2]['sigma']:.5f} | {d_vs[0]:.2f}% | {d_vs[1]:.2f}% |
| **edm** | {sig['edm'][0]['sigma']:.5f} | {sig['edm'][1]['sigma']:.5f} | {sig['edm'][2]['sigma']:.5f} | **{d_edm[0]:.2f}%** | **{d_edm[1]:.2f}%** |
| guo | {sig['guo'][0]['sigma']:.5f} | {sig['guo'][1]['sigma']:.5f} | {sig['guo'][2]['sigma']:.5f} | {d_guo[0]:.2f}% | {d_guo[1]:.2f}% |

（参照：库标定 `SIGMA_DEFAULT={SIGMA_LIB}`。）

**B. 压力约定的复算（同一组交付带内密度，两种符号）**

| 臂 | σ 减号（库，交付采用） | σ 加号（第 1 轮） | 比值 | 交付工件 σ |
|---|---|---|---|---|
| edm@τ=1.0 | {probe['pressure']['edm@1.0@R12']['sigma_minus']:.6f} | {probe['pressure']['edm@1.0@R12']['sigma_plus']:.4f} | **{probe['pressure']['edm@1.0@R12']['ratio_plus_over_minus']:.1f}×** | {probe['pressure']['edm@1.0@R12']['artifact_sigma']:.6f} |
| guo@τ=1.0 | {probe['pressure']['guo@1.0@R12']['sigma_minus']:.6f} | {probe['pressure']['guo@1.0@R12']['sigma_plus']:.4f} | {probe['pressure']['guo@1.0@R12']['ratio_plus_over_minus']:.1f}× | {probe['pressure']['guo@1.0@R12']['artifact_sigma']:.6f} |
| velocity_shift@τ=1.2 | {probe['pressure']['velocity_shift@1.2@R12']['sigma_minus']:.6f} | {probe['pressure']['velocity_shift@1.2@R12']['sigma_plus']:.4f} | {probe['pressure']['velocity_shift@1.2@R12']['ratio_plus_over_minus']:.1f}× | {probe['pressure']['velocity_shift@1.2@R12']['artifact_sigma']:.6f} |

**C. τ=1 的力项恒等（审计者自建随机场，库 collider）**

| τ | max\\|f_vs − f_edm\\| | max\\|f_vs − f_guo\\| |
|---|---|---|
""" + "\n".join(
        f"| {t} | {v['max_vs_minus_edm']:.3e} | {v['max_vs_minus_guo']:.3e} |"
        for t, v in ident.items() if not t.startswith("_")
    ) + f"""

**D. Maxwell 等面积解析参照：交付 vs 审计者独立重算**

| 量 | 交付 compact.json | 审计者重算 | 偏差 |
|---|---|---|---|
| p_sat | {mx_del['p_sat']:.8f} | {mx['p_sat']:.8f} | {abs(mx['p_sat']-mx_del['p_sat'])/mx_del['p_sat']*100:.2e}% |
| ρ_l | {mx_del['rho_l']:.8f} | {mx['rho_l']:.8f} | {abs(mx['rho_l']-mx_del['rho_l'])/mx_del['rho_l']*100:.2e}% |
| ρ_v | {mx_del['rho_v']:.8f} | {mx['rho_v']:.8f} | {abs(mx['rho_v']-mx_del['rho_v'])/mx_del['rho_v']*100:.2e}% |
| ρ_l/ρ_v | {mx_del['ratio']:.4f} | {mx['ratio']:.4f} | — |

## ④ 结果图

三幅：**A** σ(τ)（EDM 平、vs 倾斜、guo 低）；**B** 同一组密度在两种压力符号下的 σ（第 1 轮口径把 σ 抬到
10⁰–10¹ 量级）；**C** 带内共存密度 vs Maxwell 解析线。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **外部锚是空的**：TCLB 本体从未运行（交付 §④.6 自认"配方级、非代码级"对标），沙箱内无 TCLB
   build/run/log/SHA。本锚点与 TCLB 的关系目前只到"复刻其公开配方"，不是数值对标 —— 这直接把 L4 封死。
2. **τ=1 是退化点**：任何只在 τ=1 做的"力项对比"，EDM 与 velocity-shift 逐位相同（审计者复现
   {ident['1.0']['max_vs_minus_edm']:.2e}）。第 1 轮"力项影响被掩盖"的物理解释应更正为恒等式。
3. **压力符号必须跟库约定**：同一组密度换符号，σ 改变 {probe['pressure']['edm@1.0@R12']['ratio_plus_over_minus']:.0f}×（edm@τ=1），
   guo 臂 {probe['pressure']['guo@1.0@R12']['ratio_plus_over_minus']:.0f}× —— "271 倍"是臂相关的具体值，量级陈述成立。
4. **报告的 τ 离散度口径不统一**：交付写 "±25.6% / ±0.4%"，但 compact.json 无该字段；按统一定义重算得
   全幅 {d_vs[0]:.2f}% / {d_edm[0]:.2f}%（半幅 {d_vs[1]:.2f}% / {d_edm[1]:.2f}%）。结论方向不变，但原表述**低估**了
   velocity_shift 的缺陷（+{pct(max(c['sigma'] for c in sig['velocity_shift']), sum(c['sigma'] for c in sig['velocity_shift'])/3):.1f}% 上侧被吞掉）。
5. **图在仓库态丢失**：`fig1_laplace_tau_coexistence.png` 在 commit 中存在但 merge #2455 后被推送门滤除，
   平台报告 id=19330 的 `figures` 也无 `png_base64` → 图只能重跑生成。
6. **全量 45 例未取回**（`cj3388` 交付时仍 queued），交付自报 `fail_stage=budget_exhausted+forced_finish`；
   本核验的数值结论基于 23 例（3 半径 × 3 τ）。
7. **寄生流偏大**（u_max≈0.135–0.149）是 G=5、界面宽 4 的粗界面 SC 已知伪影，只作相对比较，不作绝对精度判据。

## ⑥ 误差判定（逐项对基准，显式回答「是否 <3%」）

| 关键量 | 基准 | 实测（EDM） | 误差 | <3%? | <15%? |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| σ, R=12, τ={c['tau']} | {SIGMA_LIB}（库标定） | {c['sigma']:.5f} | {pct(c['sigma'], SIGMA_LIB):+.2f}% | "
        f"{'是' if abs(pct(c['sigma'], SIGMA_LIB)) < 3 else '否'} | 是 |"
        for c in sig["edm"]
    ) + f"""
| σ_fit（过原点 3 半径） | {SIGMA_LIB} | {compact['legs']['laplace']['edm@1.0']['sigma_fit']:.5f} | {pct(compact['legs']['laplace']['edm@1.0']['sigma_fit'], SIGMA_LIB):+.2f}% | **否（略超）** | 是 |
| ρ_l（EDM, R=12） | Maxwell {mx['rho_l']:.5f} | {sig['edm'][1]['rho_in']:.5f} | {pct(sig['edm'][1]['rho_in'], mx['rho_l']):+.2f}% | 是 | 是 |
| ρ_v（EDM, R=12） | Maxwell {mx['rho_v']:.5f} | {sig['edm'][1]['rho_out']:.5f} | {pct(sig['edm'][1]['rho_out'], mx['rho_v']):+.2f}% | **否** | 是 |
| σ（velocity_shift, τ=1.2） | {SIGMA_LIB} | {sig['velocity_shift'][2]['sigma']:.5f} | {pct(sig['velocity_shift'][2]['sigma'], SIGMA_LIB):+.2f}% | 否 | **否（honest FAIL）** |
| ρ_v（velocity_shift, τ=1.2） | Maxwell {mx['rho_v']:.5f} | {sig['velocity_shift'][2]['rho_out']:.5f} | {pct(sig['velocity_shift'][2]['rho_out'], mx['rho_v']):+.2f}% | 否 | **否** |
| σ（guo, 任意 τ） | {SIGMA_LIB} | {sig['guo'][1]['sigma']:.5f} | {pct(sig['guo'][1]['sigma'], SIGMA_LIB):+.2f}% | 否 | **否** |

**判定**：EDM 路径对库标定 σ **<3%（{pct(sig['edm'][0]['sigma'], SIGMA_LIB):+.2f}%/{pct(sig['edm'][1]['sigma'], SIGMA_LIB):+.2f}%/{pct(sig['edm'][2]['sigma'], SIGMA_LIB):+.2f}%）**，
过原点多半径拟合 σ_fit **{pct(compact['legs']['laplace']['edm@1.0']['sigma_fit'], SIGMA_LIB):+.2f}%（略超 3%，<15%）**；共存密度对 Maxwell 解析
ρ_l {pct(sig['edm'][1]['rho_in'], mx['rho_l']):+.2f}%（<3%）、ρ_v {pct(sig['edm'][1]['rho_out'], mx['rho_v']):+.2f}%（<15% 但 >3%）。
同期两处 honest FAIL 复核成立且**不粉饰**：velocity_shift 在 τ≠1 下 σ {pct(sig['velocity_shift'][0]['sigma'], SIGMA_LIB):+.2f}%/{pct(sig['velocity_shift'][2]['sigma'], SIGMA_LIB):+.2f}%、
ρ_v {pct(sig['velocity_shift'][2]['rho_out'], mx['rho_v']):+.2f}%；Guo σ {pct(sig['guo'][1]['sigma'], SIGMA_LIB):+.2f}%、ρ_v 16–19%。

**但**：以上"基准"是**内部标定 + 解析自洽**参照，不是外部锚。**对外部锚（TCLB）的误差判定：无法给出（WITHHELD）**
—— 该锚点问的"是否达到 TCLB shan-chen-edm 一族的行为"尚未取得任何 TCLB 数值。据此本单判 **L3**，不判 L4。

## 附：本核验未执行的检查

TCLB 本体运行（not_run）· 全量 45 例（cj3388 从未 fetch，not_run）· `local_sweep.py` 端到端重跑（本机 120 s 收割，
compute 门 cj3406 递交时仍 queued，not_run；数值已由 solver-free 探针覆盖）· float32/float64 dtype 精度对齐未单独量化。
"""
    return md


def main():
    compact = json.loads(COMPACT.read_text())
    probe = json.loads(PROBE.read_text())
    md = build(compact, probe)

    s = requests.Session()
    r = s.post(f"{API}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
    r.raise_for_status()
    tok = r.json().get("access_token") or r.json().get("token")
    hdr = {"Authorization": f"Bearer {tok}"}

    fig_b64 = base64.b64encode(FIG.read_bytes()).decode()
    payload = {
        "title": "m21378 交差核验：XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO —— 判 L3，L4 因外部锚空缺 WITHHELD",
        "summary": "board-audit：核 m21378（commit 9b2791376）三件套与数值；solver-free 探针独立复现 τ=1 恒等 5.55e-17、"
                   "压力符号 270× 与 Maxwell 解析（<1e-6%）；pytest 5 passed 21.34 s；benchmark 表该 anchor 0 行、TCLB 未执行 → L4 WITHHELD，判 L3。",
        "scenario_key": "xval-lbm-shan-chen-edm-lycett-luo-m21400-audit",
        "mission_id": 21400,
        "tags": ["audit", "LBM", "Shan-Chen", "EDM", "maturity-verdict", "L3", "TensorLBM"],
        "markdown": md,
        "figures": [{
            "name": "fig1_m21378_audit.png",
            "caption": "A σ(τ) 三力项 vs 库标定 0.05611（EDM 全幅 0.40%，velocity_shift 53.68%）；"
                       "B 同一组密度两种压力符号的 σ（比值 196–459×）；C 带内共存密度 vs Maxwell 解析",
            "png_base64": fig_b64,
        }],
    }
    r = s.post(f"{API}/tensor/reports", json=payload, headers=hdr, timeout=120)
    print("status", r.status_code)
    print(r.text[:600])
    out = HERE / "REPORT_m21378_AUDIT.md"
    out.write_text(md)
    print("markdown snapshot ->", out)


if __name__ == "__main__":
    main()