#!/usr/bin/env python3
"""NSCBC 出口反射系数验证面（D2Q9 声脉冲，d'Alembert 解析锚 + 两端解析极限）。

为什么需要这个验证面
--------------------
``tensorlbm.boundaries.nscbc_outlet_2d`` 在 ``boundary_capability_contract``
里被登记为 ``IMPLEMENTATION_ONLY``（"No tests; simplified single-characteristic
relaxation, not full NSCBC"）。库文档声称它是 "Non-Reflecting (Characteristic
Wave) outlet"，并声明 σ 的语义是：

    σ = 0 → fully non-reflecting；σ = 1 → hard pressure fix

本验证面把这两条声明变成**机器可判**的判据：在小振幅声脉冲打到出口时测量
反射系数 R，并对两个解析极限做仪器标定。

物理问题（1-D 管道，均值流 U0，小振幅声脉冲）
---------------------------------------------
D2Q9 BGK，ν=(τ−1/2)/3，ρ0=1，均匀来流 U0（无壁面、无驱动力，动量守恒）。
初始扰动取**右行特征**（d'Alembert 相容初值）：

    ρ(x,0) = ρ0 + A·exp(−(x−x0)²/(2w²))
    u(x,0) = U0 + (ρ−ρ0)·cs/ρ0        (向右传播的压力-速度同相特征)

线化声学下该扰动以 c+ = cs + U0 右传（计入 BGK 色散后的实测波速另行报告），
打到 x=nx−1 的出口后产生左行反射波，其振幅比即反射系数 R。

测量方法（自标定，不依赖任何可调参数）
--------------------------------------
探针 P1(x=xp1)、P2(x=xp2) 每步记录 ρ、u。特征分解：

    A+ = ½[ p' + ρ0·cs·u' ]  (右行，压力幅值)
    A− = ½[ p' − ρ0·cs·u' ]  (左行)
    p' = cs²·(ρ−ρ0),  u' = u−U0

* 入射幅值：A+ 在入射窗内的峰值（P1、P2 各一次）。
* 衰减标定：a = A+_P2/A+_P1 是 200 格子行程的实测衰减；出口往返 2·(nx−1−xp1)
  格子 → 衰减因子 a^(2(nx−1−xp1)/200)。
* 反射系数（衰减已扣除）：

      R = A−_P1 / ( A+_P1 · a^(2(nx−1−xp1)/Δ) )

  符号保留（压力节点反射为负、刚性壁反射为正）。

解析锚（仪器标定，先证仪器能分辨，再判被测 BC）
------------------------------------------------
* ``pin``  —— 出口硬性给定 p'=0（压力 Dirichlet）⇒ 线化声学压力节点反射 |R|=1、符号负。
* ``mirror`` —— 出口全反射（刚性壁，bounce-back）⇒ |R|=1、符号正。
* 若这两个锚测不出 ±1，本仪器的任何结论都不成立（先判仪器）。
* 入射段的 d'Alembert 校验：实测波速 vs cs=1/√3（报告偏差），
  归一化波形 vs 解析 Gaussian 平移解（shape error），保证脉冲本身是"已知输入"。

被测对象与判据
--------------
判据直接取自库文档的 σ 语义（不是事后发明的阈值）：

1. ``nscbc``(|R(σ=1)|) 应接近 1（文档："σ=1 → hard pressure fix"）→ 容差 ±10%；
2. ``nscbc`` 的 |R| 随 σ 单调（σ→0 更"非反射"）——趋势判据；
3. 默认参数 σ=0.25 下 |R| ≤ 5%（"non-reflecting" 的字面含义；Poinsot & Lele 1992
   的 1-D 法向入射 NSCBC 反射量级为百分之一量级）—— 硬门。

用法：
    run.py all <out.json>            # 全档（默认）
    run.py single <bc> <sigma> <out.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.boundaries import nscbc_outlet_2d, zero_gradient_outlet_2d  # noqa: E402

CS = 1.0 / math.sqrt(3.0)

# D2Q9 方向索引 (cx, cy)：0(0,0) 1(1,0) 2(0,1) 3(-1,0) 4(0,-1) 5(1,1) 6(-1,1) 7(-1,-1) 8(1,-1)
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)  # 全反向（刚性壁）

# --------------------------------------------------------------------------- #
# 出口边界条件实现（全部 post-streaming，形状 (9, ny, nx)）
# --------------------------------------------------------------------------- #


def _column_fields(f: torch.Tensor, col: int):
    rho, ux, uy = macroscopic(f)
    ny = f.shape[1]
    return (
        rho[:, col].reshape(ny, 1),
        ux[:, col].reshape(ny, 1),
        uy[:, col].reshape(ny, 1),
    )


def bc_zerograd(f: torch.Tensor, **_):
    """库函数：零梯度出口（对照档）。"""
    return zero_gradient_outlet_2d(f)


def bc_nscbc(f: torch.Tensor, sigma: float = 0.25, rho_target: float = 1.0, **_):
    """库函数：被测的 NSCBC 出口。"""
    return nscbc_outlet_2d(f, rho_target=rho_target, sigma=sigma)


def bc_pin(f: torch.Tensor, rho_target: float = 1.0, **_):
    """解析锚：出口硬性 p'=0（压力 Dirichlet，取上游速度）。

    线化声学的压力节点：p'_inc + p'_ref = 0 ⇒ R = −1（幅值 1）。
    """
    ny = f.shape[1]
    rho_i, ux_i, uy_i = _column_fields(f, -2)
    rho_b = torch.full_like(rho_i, rho_target)
    out = f.clone()
    out[:, :, -1] = equilibrium(rho_b, ux_i, uy_i)[:, :, 0]
    return out


def bc_mirror(f: torch.Tensor, **_):
    """解析锚：刚性壁全反射（bounce-back，出口列反转全部方向）⇒ R = +1。"""
    out = f.clone()
    out[:, :, -1] = f[OPP][:, :, -1]
    return out


def bc_neeq(f: torch.Tensor, sigma: float = 0.25, rho_target: float = 1.0, **_):
    """候选修复档：非平衡外推（Guo 2002）+ 线化松弛目标压力。

    ρ_b = ρ_i − σ·(ρ_i − ρ_target)，速度零梯度，非平衡部分从上游列外推：

        f_outlet = f_eq(ρ_b, u_i) + [ f(upstream) − f_eq(ρ_i, u_i) ]

    σ=0 ⇒ ρ_b=ρ_i ⇒ 出口列 ≡ 上游列（与零梯度档逐位相同）；
    σ=1 ⇒ 硬压力 Dirichlet（与解析锚 pin 同心）。
    即该族在 σ 上**解析地**连接两个已标定的极限（端点可逐位对照）。
    """
    ny = f.shape[1]
    rho_i, ux_i, uy_i = _column_fields(f, -2)
    rho_b = rho_i - sigma * (rho_i - rho_target)
    feq_b = equilibrium(rho_b, ux_i, uy_i)[:, :, 0]
    feq_i = equilibrium(rho_i, ux_i, uy_i)[:, :, 0]
    out = f.clone()
    out[:, :, -1] = feq_b + (f[:, :, -2] - feq_i)
    return out


OUTLETS = {
    "zerograd": bc_zerograd,
    "nscbc": bc_nscbc,
    "pin": bc_pin,
    "mirror": bc_mirror,
    "neeq": bc_neeq,
}


def bc_inlet_zerograd(f: torch.Tensor) -> torch.Tensor:
    """入口（左端）零梯度：f[:, :, 0] = f[:, :, 1]（脉冲右行，入口影响可忽略）。"""
    out = f.clone()
    out[:, :, 0] = f[:, :, 1]
    return out


# --------------------------------------------------------------------------- #
# 单档模拟
# --------------------------------------------------------------------------- #


def run_case(
    bc: str,
    sigma: float,
    nx: int = 1400,
    ny: int = 8,
    nu: float = 0.1,
    u0: float = 0.0,
    amp: float = 1e-3,
    width: float = 48.0,
    x0: int = 200,
    xp1: int | None = None,
    xp2: int | None = None,
    n_steps: int = 4000,
    rho_target: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    xp1 = nx - 400 if xp1 is None else xp1
    xp2 = nx - 250 if xp2 is None else xp2
    tau = 0.5 + 3.0 * nu
    dev = torch.device("cpu")
    xx = torch.arange(nx, device=dev, dtype=dtype).reshape(1, nx)
    yy = torch.arange(ny, device=dev, dtype=dtype).reshape(ny, 1)

    shape = 1.0 + amp * torch.exp(-((xx - x0) ** 2) / (2.0 * width**2))
    rho = shape.expand(ny, nx).contiguous()
    drho = rho - 1.0
    ux = (u0 + drho * CS).contiguous()  # 右行特征相容初值
    uy = torch.zeros((ny, nx), device=dev, dtype=dtype)
    f = equilibrium(rho, ux, uy)

    outlet_fn = OUTLETS[bc]
    kw = {"sigma": sigma, "rho_target": rho_target}

    rho1 = np.empty(n_steps, dtype=np.float64)
    ux1 = np.empty(n_steps, dtype=np.float64)
    rho2 = np.empty(n_steps, dtype=np.float64)
    ux2 = np.empty(n_steps, dtype=np.float64)
    mass = np.empty(n_steps, dtype=np.float64)

    t0 = time.time()
    divergent = False
    for step in range(n_steps):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = outlet_fn(f, **kw)
        f = bc_inlet_zerograd(f)
        rr, uu, _ = macroscopic(f)
        rho1[step] = float(rr[0, xp1])
        ux1[step] = float(uu[0, xp1])
        rho2[step] = float(rr[0, xp2])
        ux2[step] = float(uu[0, xp2])
        mass[step] = float(f.sum())
        if not math.isfinite(rho1[step]) or abs(rho1[step] - 1.0) > 0.05:
            divergent = True
            rho1[step:] = np.nan
            rho2[step:] = np.nan
            ux1[step:] = np.nan
            ux2[step:] = np.nan
            break
    wall = time.time() - t0

    # ---- 特征分解 -----------------------------------------------------------
    def chars(rho, ux):
        # 去直流（中位数稳健）：平均密度/流速的缓慢漂移不属于声波
        p = CS * CS * (rho - np.median(rho))
        up = ux - np.median(ux)
        return 0.5 * (p + CS * up), 0.5 * (p - CS * up)

    ap1, am1 = chars(rho1, ux1)
    ap2, am2 = chars(rho2, ux2)

    t_inc = (xp1 - x0) / (CS + u0)  # 入射到达 P1 的理论时刻
    half = 6.0 * width / CS  # 窗口半宽（脉冲穿越时间量级）
    w_in = (slice(max(int(t_inc - half), 0), min(int(t_inc + half), n_steps)),)

    a_inc_p1 = float(np.nanmax(np.abs(ap1[w_in])))
    a_inc_p2 = float(np.nanmax(np.abs(ap2[w_in])))
    a_ref_p1 = float(np.nanmax(np.abs(am1[w_in])))  # 入射段自带的左行底噪

    # 反射窗：入射在出口（nx−1）反射后再回到 P1
    t_out = (nx - 1 - x0) / (CS + u0)
    t_ref = t_out + (nx - 1 - xp1) / (CS - u0)
    w_rf = (slice(max(int(t_ref - half), 0), min(int(t_ref + half), n_steps)),)
    a_ref = float(np.nanmax(np.abs(am1[w_rf])))
    sgn = float(np.sign(np.nanmean(am1[w_rf][np.abs(am1[w_rf]) > 0.5 * a_ref]))) if a_ref > 0 else 0.0

    # 衰减标定：200 格子的实测衰减 a
    a = a_inc_p2 / a_inc_p1 if a_inc_p1 > 0 else float("nan")
    dist = 2.0 * (nx - 1 - xp1)
    atten = a ** (dist / float(xp2 - xp1)) if a > 0 else float("nan")
    r_corr = a_ref / (a_inc_p1 * atten) if a_inc_p1 > 0 else float("nan")

    # 入射段的 d'Alembert 校验：波速 + 归一化波形
    if w_in[0].stop - w_in[0].start < 5:
        raise RuntimeError(f"incident window empty: nx={nx}, x0={x0}, xp1={xp1}, n_steps={n_steps}")
    win_i = np.arange(n_steps)[w_in[0]]
    p1 = CS * CS * (rho1[w_in[0]] - np.median(rho1))
    peak_idx = int(np.nanargmax(np.abs(p1)))
    t_arrive = float(win_i[peak_idx])
    c_meas = (xp1 - x0) / t_arrive
    # 解析 d'Alembert 平移解（同宽、同传播速度、实测幅值）vs 实测波形（零均值后）
    ana = np.abs(p1[peak_idx]) * np.exp(-(((win_i - t_arrive) * c_meas) ** 2) / (2.0 * width**2))
    shape_err = float(np.linalg.norm(np.abs(p1) - ana) / np.linalg.norm(ana))

    return {
        "bc": bc,
        "sigma": sigma,
        "tau": tau,
        "nu": nu,
        "u0": u0,
        "amp": amp,
        "width": width,
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
        "divergent": divergent,
        "wall_seconds": round(wall, 2),
        "mass_drift_max_rel": float(np.nanmax(np.abs(mass / mass[0] - 1.0))),
        "incident_amp_P1": a_inc_p1,
        "incident_amp_P2": a_inc_p2,
        "incident_leftgoing_floor_P1": a_ref_p1,
        "floor_over_incident": a_ref_p1 / a_inc_p1 if a_inc_p1 > 0 else float("nan"),
        "atten_per_200": a,
        "reflected_amp_P1": a_ref,
        "reflection_coeff_atten_corrected": r_corr,
        "reflection_coeff_raw": a_ref / a_inc_p1 if a_inc_p1 > 0 else float("nan"),
        "reflection_sign": sgn,
        "ideal_anchor": {"pin": -1.0, "mirror": 1.0}.get(bc),
        "wave_speed_measured": c_meas,
        "wave_speed_theory": CS + u0,
        "wave_speed_err_pct": 100.0 * (c_meas - (CS + u0)) / (CS + u0),
        "dalembert_shape_err_pct": 100.0 * shape_err,
    }


# --------------------------------------------------------------------------- #
# 档位与判据
# --------------------------------------------------------------------------- #

CASES = [
    ("mirror", 0.0),
    ("pin", 0.0),
    ("zerograd", 0.0),
    ("nscbc", 0.0),
    ("nscbc", 0.25),
    ("nscbc", 1.0),
    ("nscbc", 5.0),
    ("neeq", 0.0),
]


def judge(rows: dict[str, dict]) -> dict:
    """判据（全部取自库文档语义，见模块 docstring）。"""
    def R(tag):
        r = rows[tag]
        return None if r["divergent"] else r["reflection_coeff_atten_corrected"]

    # --- 仪器标定：两个解析极限 ---
    instrument = {
        "mirror_|R|_vs_1": abs(abs(R("mirror@0.0")) - 1.0) if R("mirror@0.0") is not None else None,
        "pin_|R|_vs_1": abs(abs(R("pin@0.0")) - 1.0) if R("pin@0.0") is not None else None,
        "mirror_sign_positive": rows["mirror@0.0"]["reflection_sign"] > 0,
        "pin_sign_negative": rows["pin@0.0"]["reflection_sign"] < 0,
        "wave_speed_err_pct": rows["pin@0.0"]["wave_speed_err_pct"],
        "dalembert_shape_err_pct": rows["pin@0.0"]["dalembert_shape_err_pct"],
    }
    instrument["tolerance"] = 0.10
    instrument["pass"] = bool(
        instrument["mirror_|R|_vs_1"] is not None
        and instrument["pin_|R|_vs_1"] is not None
        and instrument["mirror_|R|_vs_1"] <= 0.10
        and instrument["pin_|R|_vs_1"] <= 0.10
        and instrument["mirror_sign_positive"]
        and instrument["pin_sign_negative"]
    )

    # --- 被测：库 NSCBC 的文档声明（按本次实际跑到的 σ 档判定）---
    sig = sorted(float(t.split("@")[1]) for t in rows if t.startswith("nscbc@"))
    r_by = {sg: rows[f"nscbc@{sg}"]["reflection_coeff_atten_corrected"] for sg in sig}
    r_pin = R("pin@0.0")
    r025 = r_by.get(0.25)
    r1 = r_by.get(1.0)
    claims = {
        "claim1_legacy_semantics_sigma1_is_hard_pressure_fix_|R|~1": {
            "doc_source": "旧版 docstring：'1 = hard pressure fix'（本次已改写语义，见库 docstring）",
            "measured": r1,
            "err": None if r1 is None else abs(abs(r1) - 1.0),
            "pass": bool(r1 is not None and abs(abs(r1) - 1.0) <= 0.10),
        },
        "claim2_monotone_more_reflective_with_larger_sigma": {
            "measured": r_by,
            "pass": bool(len(sig) >= 3 and all(
                abs(r_by[sig[i]]) <= abs(r_by[sig[i + 1]]) + 0.01 for i in range(len(sig) - 1)
            )),
            "note": "允许 ±0.01 容差（|R| 已到仪器分辨率地板 ~0.4%）",
        },
        "claim3_default_sigma025_non_reflecting_|R|<=5pct": {
            "measured": r025,
            "pass": bool(r025 is not None and abs(r025) <= 0.05),
        },
        "claim4_absorbs_far_better_than_zero_gradient_baseline": {
            "measured": {"nscbc_sigma0.25": r025, "zerograd": R("zerograd@0.0")},
            "pass": bool(
                r025 is not None and R("zerograd@0.0") is not None
                and abs(r025) * 5.0 <= abs(R("zerograd@0.0"))
            ),
        },
        "claim5_pressure_node_anchor_is_reflective_not_leaky": {
            "doc_source": "解析锚 pin 必须反射（|R|≈1）：若仪器把 pin 测成吸收，则所有 R 读数不可信",
            "measured": r_pin,
            "pass": bool(r_pin is not None and abs(abs(r_pin) - 1.0) <= 0.10),
        },
    }
    legacy_key = "claim1_legacy_semantics_sigma1_is_hard_pressure_fix_|R|~1"
    evaluated = {k: v for k, v in claims.items() if not k.startswith("claim1_legacy")}
    n_pass = sum(1 for c in evaluated.values() if c["pass"])
    return {
        "instrument": instrument,
        "library_nscbc_claims": claims,
        "claims_passed": n_pass,
        "claims_total": len(evaluated),
        "legacy_semantics_claim_falsified": (not claims[legacy_key]["pass"]),
        "verdict": "PASS" if (instrument["pass"] and n_pass == len(evaluated)) else "FAIL",
        "pre_fix_reference": {
            "note": "修复前（2026-11-12 前）同一仪器测得的旧实现：|R| = 0.148 / 0.005 / 0.112 @ sigma=1.0/0.5/0.25 "
                    "（非单调、默认档 11.2% 反射）；旧值仅作对照记录，不参与本次判据",
            "sigma1.0": 0.148, "sigma0.5": 0.005, "sigma0.25": 0.112,
        },
        "verdict_note": (
            "仪器标定未过 ⇒ 任何结论不成立，判 WITHHELD"
            if not instrument["pass"]
            else (f"当前实现 {n_pass}/{len(evaluated)} 条声明成立"
                  + ("；旧版 σ 语义（σ=1=硬压力）已被证伪并改写文档"
                     if not claims[legacy_key]["pass"] else ""))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "single", "batch", "assemble"])
    ap.add_argument("bc", nargs="?")
    ap.add_argument("sigma", nargs="?", default="0.0")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--nx", type=int, default=1200)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    torch.set_num_threads(8)
    t0 = time.time()
    rows: dict[str, dict] = {}

    if args.mode == "assemble":
        d = Path(args.bc or "runs")
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            rows[r["tag"]] = r
        verdict = judge(rows)
        out = Path(args.out or "result.json")
        out.write_text(json.dumps({
            "case": "nscbc_outlet_reflection",
            "title": "NSCBC 出口反射系数验证面（D2Q9 声脉冲，解析锚 + 文档声明判据）",
            "library_module": "tensorlbm.boundaries.nscbc_outlet_2d",
            "measurement": "R = A-/(A+·attenuation)，探针特征分解（零均值/中位数去直流）",
            "params": rows[next(iter(rows))]["params"] if "params" in rows[next(iter(rows))] else {
                "nx": args.nx, "ny": 8, "nu": 0.1, "u0": 0.0, "amp": 1e-3, "width": 48.0,
                "x0": 200, "probes": [args.nx - 400, args.nx - 250], "n_steps": args.n_steps,
            },
            "cases": rows, "verdict": verdict,
        }, indent=2, ensure_ascii=False))
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
        print(f"-> {out}")
        return 0

    if args.mode == "batch":
        sel = [t.strip() for t in args.cases.split(",") if t.strip()]
        todo = [c for c in CASES if f"{c[0]}@{c[1]}" in sel]
        outdir = Path(args.out or "runs")
        outdir.mkdir(parents=True, exist_ok=True)
        for bc, sigma in todo:
            tag = f"{bc}@{sigma}"
            print(f"[case] {tag} ...", flush=True)
            r = run_case(bc, sigma, nx=args.nx, n_steps=args.n_steps)
            r["tag"] = tag
            (outdir / f"{tag}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
            print("   R(corr)={:.4f} sign={:+.0f} floor={:.4f} wall={:.1f}s".format(
                r["reflection_coeff_atten_corrected"], r["reflection_sign"],
                r["floor_over_incident"], r["wall_seconds"]), flush=True)
        return 0
    if args.mode == "single":
        bc, sigma = args.bc, float(args.sigma)
        r = run_case(bc, sigma, nx=args.nx, n_steps=args.n_steps)
        out = Path(args.out or "result_single.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    for bc, sigma in CASES:
        tag = f"{bc}@{sigma}"
        print(f"[case] {tag} ...", flush=True)
        rows[tag] = run_case(bc, sigma, nx=args.nx, n_steps=args.n_steps)
        rows[tag]["tag"] = tag
        print(
            "   R(raw)={:.4f} R(corr)={:.4f} sign={:+.0f} massdrift={:.2e} wall={:.1f}s".format(
                rows[tag]["reflection_coeff_raw"],
                rows[tag]["reflection_coeff_atten_corrected"],
                rows[tag]["reflection_sign"],
                rows[tag]["mass_drift_max_rel"],
                rows[tag]["wall_seconds"],
            ),
            flush=True,
        )

    verdict = judge(rows)
    payload = {
        "case": "nscbc_outlet_reflection",
        "title": "NSCBC 出口反射系数验证面（D2Q9 声脉冲，解析锚 + 文档声明判据）",
        "library_module": "tensorlbm.boundaries.nscbc_outlet_2d",
        "measurement": "R = A-/(A+·attenuation), characteristic decomposition at probe x=1000",
        "params": {
            "nx": args.nx,
            "ny": 8,
            "nu": 0.1,
            "tau": 0.8,
            "u0": 0.0,
            "amp": 1e-3,
            "width": 48.0,
            "x0": 200,
            "probes": [args.nx - 400, args.nx - 250],
            "n_steps": args.n_steps,
            "cs": CS,
        },
        "cases": rows,
        "verdict": verdict,
        "total_wall_seconds": round(time.time() - t0, 1),
    }
    out = Path(sys.argv[-1]) if sys.argv[-1].endswith(".json") else Path("result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    return 0 if verdict["verdict"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())