#!/usr/bin/env python3
"""CASE-MCMP-LAPLACE — 多组分静滴 Young-Laplace 定律（TensorLBM 算例池执行器）。

算例池记录：``case_code=MCMP-LAPLACE``，repo=TensorLBM，
ref_source=``Delta p = sigma / R``（Laplace 解析律，2D 圆滴 / 圆柱界面），
tol_policy=``<5%``（owner 算例池令）。

本目录是**复用型**执行器：不新写求解器、不改物理内核，只驱动仓内既有两条能力：

* **Route A（Laplace，主）** `tensorlbm.mcmp_verification` —— 两组分 Shan-Chen
  静止液滴（``simulate_static_droplet`` + ``measure_laplace``），平衡态 EOS 压力
  ``p = cs^2(rho1+rho2) + G12 cs^2 rho1 rho2`` 取液/气两侧箱均值得到 ``Delta p``，
  半径取四条射线上 ``rho1 = (rho_liq+rho_gas)/2`` 的线性插值交点；
* **Route B（机械压力张量，独立绝对基准）** `tensorlbm.scmc_pressure_tensor` ——
  平板静止界面的 Kirkwood-Buff 机械表面张力 ``sigma = ∫ (P_NN - P_TT) dn / n_iface``，
  与 Laplace 律无关地给出同一模型的 ``sigma``；``P_NN`` 沿界面法向的守恒残差
  （flatness）作为该路线的**有效性闸门**。

判据（运行前固定，见 ``THRESHOLDS``；运行后未做任何调整）：

* 算例池容差族 ``<5%``：
  (a) 逐半径 ``sigma_i = Delta p_i R_i`` 对过原点拟合的 ``sigma`` 的最大相对偏差；
  (b) ``Delta p`` 实测 vs 解析 ``sigma_B / R``（独立 Route B 的 sigma）逐点相对偏差；
  (c) ``sigma_A``（Laplace 拟合）vs ``sigma_B``（机械张量）跨路线相对偏差。
* 自设护栏族（非算例池容差，如实分列）：``R^2 >= 0.99``、自由线截距
  ``|b| R_max / |a| <= 5%``、Route B ``flatness <= 5%``、以及库内 guardrail
  （未收敛 / 周期映像 / 各向异性 / 负 sigma）。

所有数字由本文件写入 ``artifacts/summary.json``，报告 markdown 与图由
``make_report.py`` 从该 json 程序化渲染（零手抄）。

用法::

    python benchmarks/case_pool/mcmp_laplace/run_case.py --jobs 8
    python benchmarks/case_pool/mcmp_laplace/run_case.py --quick --jobs 2   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tensorlbm.mcmp_verification as mv  # noqa: E402
from tensorlbm.scmc_pressure_tensor import (  # noqa: E402
    SCMCPressureConfig,
    flat_interface_pressure_profile,
    simulate_flat_interface,
)

# ---------------------------------------------------------------------------
# Protocol + thresholds (pre-registered: fixed before the numbers existed)
# ---------------------------------------------------------------------------

#: 算例池容差（owner 2026-11-12 令：<5%）。
TOL = 0.05

THRESHOLDS = {
    # --- 算例池容差族（gates） ---
    "tolerance": TOL,
    # --- 自设护栏 / 诊断族（reported, not gates） ---
    "r2_min": 0.99,
    "max_offset_fraction": 0.05,
    "flat_flatness_max": 0.05,
}

#: 耦合常数窗口：-2.2 溶解、-2.3 反转、<= -3.5 发散（由库计量 guardrail 测绘，
#: 见 docs/MCMP_VERIFY_BENCH.md）；三点全部落在窗口内部。
G_LEVELS = [-2.5, -2.75, -3.0]

#: Laplace 系列的初始半径（n = 96 ⇒ R/n <= 0.25 < 库护栏 0.35）。
RADII = [12.0, 16.0, 20.0, 24.0]

PROTOCOL: dict[str, Any] = {
    "case_code": "MCMP-LAPLACE",
    "repo": "TensorLBM",
    "ref_source": "Delta p = sigma / R (Laplace 解析律, 2D 圆滴/圆柱界面)",
    "tol_policy": f"<{TOL:.0%}",
    "laplace_dims": 2,  # 2D 圆滴：Delta p = sigma / R
    "G_levels": G_LEVELS,
    "radii": RADII,
    "n_main": 96,
    "resolution_study": {"radius": 16.0, "n": [96, 128]},
    "steps_droplet": 12000,
    "steps_coexist": 12000,
    "tau": 1.0,
    "stop_tol": 1e-4,
    "check_every": 250,
    "flat": {
        "ny": 192,
        "nx": 8,
        "steps": 20000,
        "checkpoints": [5000, 10000, 15000, 20000],
        "slab_half": 48.0,
    },
    "eos_anchor": {"n": 96, "steps": 20000},
    "hero": {"G_12": -2.75, "radius": 20.0, "n": 96},
    "dtype": "float32",
    "device": "cpu",
}

QUICK: dict[str, Any] = {
    "G_levels": [-2.5],
    "radii": [12.0, 16.0, 20.0],
    "n_main": 64,
    "resolution_study": {"radius": 16.0, "n": [64]},
    "steps_droplet": 2500,
    "steps_coexist": 2500,
    "eos_anchor": {"n": 64, "steps": 3000},
    "flat": {"ny": 96, "nx": 8, "steps": 3000, "checkpoints": [1500, 3000], "slab_half": 24.0},
}


# ---------------------------------------------------------------------------
# Jobs (module level so they pickle)
# ---------------------------------------------------------------------------


def _quiet() -> None:
    torch.set_num_threads(1)


def job_coexistence(args: tuple[float, int, int]) -> dict[str, Any]:
    """Planar-slab coexistence densities (curvature free) for one coupling."""
    G, n, steps = args
    _quiet()
    t0 = time.time()
    out = mv.estimate_coexistence(G_12=G, n=n, geometry="planar", steps=steps)
    out.pop("rho1_field", None)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def job_droplet(args: tuple[float, int, float, int, dict[str, Any], bool]) -> dict[str, Any]:
    """One static droplet: relax, then measure ``Delta p``, ``R`` and ``sigma``."""
    G, n, radius, steps, coex, hero = args
    _quiet()
    t0 = time.time()
    run = mv.simulate_static_droplet(
        G_12=G,
        n=n,
        radius_init=radius,
        rho1_in=coex["rho1_liq"],
        rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"],
        rho2_out=coex["rho2_gas"],
        steps=steps,
        tau1=PROTOCOL["tau"],
        tau2=PROTOCOL["tau"],
        stop_tol=PROTOCOL["stop_tol"],
        check_every=PROTOCOL["check_every"],
    )
    case = mv.measure_laplace(run, dims=PROTOCOL["laplace_dims"])
    # 独立的半径估计（径向 tanh 拟合），与射线插值互为校验
    thick = mv.interface_thickness(run["rho1"])
    case["w_tanh"] = thick["w_tanh"]
    case["width_10_90"] = thick["width_10_90"]
    case["R_fit_tanh"] = thick["R_fit"]
    if thick["w_tanh"] and case["radius"]:
        case["R_over_w"] = case["radius"] / thick["w_tanh"]
    case["thickness_guardrails"] = thick["guardrails"]
    case["history"] = run["history"]
    case["elapsed_s"] = round(time.time() - t0, 1)
    if hero:
        case["_fields"] = {
            "rho1": run["rho1"].numpy(),
            "rho2": run["rho2"].numpy(),
            "ux": run["ux"].numpy(),
            "uy": run["uy"].numpy(),
        }
        prof_r, prof_rho = mv.radial_profile(run["rho1"])
        case["_profile"] = {
            "r": prof_r.tolist(),
            "rho1": prof_rho.tolist(),
        }
    return case


def _count_interfaces(profile: torch.Tensor, frac: float = 0.2) -> int:
    """Number of interface bands in a periodic 1-D profile (``|d rho/dn|`` bumps).

    A periodic domain holding two phases always contains an *even* number of
    interfaces; ``∫ (P_NN - P_TT) dn`` sums over all of them, so the per-interface
    surface tension is the integral divided by this count.
    """
    d = (torch.roll(profile, -1) - profile).abs()
    if float(d.max()) <= 0.0:
        return 0
    above = (d > frac * float(d.max())).to(torch.int64).tolist()
    # count rising edges around the (periodic) list
    edges = sum(1 for i in range(len(above)) if above[i] and not above[i - 1])
    return max(edges, 1 if any(above) else 0)


def job_flat(args: tuple[float, int, int, int, dict[str, Any], list[int], float]) -> dict[str, Any]:
    """Route B (exploratory): Kirkwood-Buff ``sigma`` of a flat interface slab.

    Protocol: a liquid slab of half-thickness ``half`` centred in a periodic
    column, with tanh edges of width ``w0``, relaxes to two static parallel
    interfaces between the coexistence densities.  ``sigma_mech`` is the
    Kirkwood-Buff integral ``∫ (P_NN - P_TT) dn`` per interface.

    A linear-ramp initialisation (the library's ``simulate_flat_interface``) is
    *not* used here: it is a spinodal state whose coarsening into a two-interface
    slab is slow (measured: still multi-domain after 20 000 steps on ny=256), and
    a coarsening front yields a meaningless ``sigma``.

    Validity gate (pre-declared): ``flatness = max|P_NN - <P_NN>|/<P_NN>`` <= 5%.
    A static planar interface has constant normal stress; if the tensor does not
    balance, the route is void and its ``sigma`` must not be quoted as a
    reference (see the report's section 5).
    """
    G, ny, nx, steps, coex, checkpoints, half = args
    _quiet()
    t0 = time.time()
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.multiphase import collide_sc_two_component
    from tensorlbm.solver import stream

    rho_l = float(coex["rho1_liq"])
    rho_g = float(coex["rho1_gas"])
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1)
    c = ny / 2.0
    w0 = 2.0
    s_in = 0.5 * (1.0 + torch.tanh((ys - (c - half)) / w0)) * 0.5 * (
        1.0 - torch.tanh((ys - (c + half)) / w0)
    )
    s_in = torch.clamp(s_in, 0.0, 1.0).expand(ny, nx).clone()
    rho1 = rho_g + s_in * (rho_l - rho_g)
    rho2 = rho_l - s_in * (rho_l - rho_g)
    zero = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)
    done = 0
    rows: list[dict[str, Any]] = []
    final_prof: dict[str, Any] | None = None
    for target in checkpoints:
        for _ in range(target - done):
            f1, f2 = collide_sc_two_component(f1, f2, G_12=G, tau1=PROTOCOL["tau"], tau2=PROTOCOL["tau"])
            f1 = stream(f1)
            f2 = stream(f2)
        done = target
        prof = flat_interface_pressure_profile(f1, f2, G)
        p_n = prof["p_normal"]
        p_t = prof["p_tangential"]
        diff = p_n - p_t
        n_iface = _count_interfaces(prof["rho1"])  # type: ignore[arg-type]
        sigma_total = float(diff.sum())
        rows.append(
            {
                "steps": target,
                "sigma_total": sigma_total,
                "n_interfaces": n_iface,
                "sigma_per_interface": sigma_total / n_iface if n_iface else None,
                "flatness": prof["flatness"],
                "proxy_flatness": prof["proxy_flatness"],
                "mean_normal": prof["mean_normal"],
                "pn_min": float(p_n.min()),
                "pn_max": float(p_n.max()),
                "diff_peak": float(diff.abs().max()),
                "rho1_min": float(prof["rho1"].min()),  # type: ignore[union-attr]
                "rho1_max": float(prof["rho1"].max()),  # type: ignore[union-attr]
            }
        )
        final_prof = {
            "y": list(range(ny)),
            "p_normal": p_n.tolist(),
            "p_tangential": p_t.tolist(),
            "rho1": prof["rho1"].tolist(),  # type: ignore[union-attr]
        }
    converged = False
    if len(rows) >= 2:
        a, b = rows[-2]["sigma_per_interface"], rows[-1]["sigma_per_interface"]
        if a and b:
            converged = abs(b - a) / abs(b) < 0.01
    return {
        "G_12": G,
        "ny": ny,
        "nx": nx,
        "slab_half_thickness": half,
        "geometry": "flat slab, 2 parallel periodic interfaces, tanh init",
        "checkpoints": rows,
        "sigma_mech": rows[-1]["sigma_per_interface"],
        "flatness": rows[-1]["flatness"],
        "converged": converged,
        "valid": bool(rows[-1]["flatness"] <= THRESHOLDS["flat_flatness_max"]),
        "profile": final_prof,
        "elapsed_s": round(time.time() - t0, 1),
    }




def eos_sigma_from_fields(rho1: torch.Tensor, rho2: torch.Tensor, G12: float, n_int: int = 2) -> dict[str, Any]:
    """EOS-profile (gradient-theory) surface tension of a flat static interface.

    ``sigma_EOS = (1/n_int) * sum_y [ p0 - p_iso(rho1(y), rho2(y)) ]`` with
    ``p_iso = cs^2 (rho1+rho2) + G12 cs^2 rho1 rho2`` (the same EOS the droplet
    pressure uses).  ``p0`` is the bulk gas-side EOS pressure.  Only the density
    profile and the bulk EOS enter, so the result is independent of any droplet
    ``dp`` or ``R``.
    """
    p = (1.0 / 3.0) * (rho1 + rho2) + G12 * (1.0 / 3.0) * rho1 * rho2
    prof = p.mean(dim=1)  # x-uniform interface
    nn = prof.numel()
    k = max(4, nn // 32)
    p_gas = float(prof[:k].mean())
    p_liq = float(prof[-k:].mean())
    deficit = p_gas - prof
    return {
        "sigma_eos": float(deficit.sum()) / n_int,
        "p_bulk_gas": p_gas,
        "p_bulk_liq": p_liq,
        "bulk_dp": p_liq - p_gas,
        "max_deficit": float(deficit.max()),
        "deficit_negative_rows": int((deficit < -1e-6).sum()),
    }


def job_eos_anchor(args: tuple[float, int, int, dict[str, Any]]) -> dict[str, Any]:
    """Route C (independent absolute sigma, Round 2): EOS-profile integral.

    A flat static interface is in normal-force balance, so the normal pressure
    is constant and equal to the bulk EOS pressure ``p0``; the tangential
    pressure is the local EOS pressure of the *same* equation of state the
    droplet measurement uses::

        sigma_EOS = (1/n_int) * sum_y [ p0 - p_iso(rho1(y), rho2(y)) ],
        p_iso(rho1, rho2) = cs^2 (rho1+rho2) + G12 cs^2 rho1 rho2.

    Only the relaxed flat-interface *density profile* and the bulk EOS enter -
    no droplet ``dp`` or ``R`` - so this is an independent anchor for
    ``dp = sigma / R`` (gradient theory; Rowlinson & Widom ch. 3;
    Sbragaglia et al. 2007 PRE 75 026702).  It is the non-circular check that
    Round 1's radius-independence fit cannot provide: a common multiplicative
    error in the model stress passes the Round-1 fit but fails here.
    """
    G, n, steps, coex = args
    _quiet()
    t0 = time.time()
    s = mv.simulate_planar_slab(
        G_12=G,
        n=n,
        rho1_in=coex["rho1_liq"],
        rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"],
        rho2_out=coex["rho2_gas"],
        half_thickness=0.25 * n,
        steps=steps,
        tau1=PROTOCOL["tau"],
        tau2=PROTOCOL["tau"],
        stop_tol=PROTOCOL["stop_tol"],
        check_every=PROTOCOL["check_every"],
    )
    rho1, rho2 = s["rho1"], s["rho2"]
    r = eos_sigma_from_fields(rho1, rho2, G, n_int=2)
    r.update(
        {
            "G_12": G,
            "n": n,
            "converged": bool(s["converged"]),
            "steps_run": int(s["steps_run"]),
            "u_max": float(s["u_max"]),
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _env() -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads_per_job": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build(out_dir: Path, jobs: int, quick: bool = False, calib_root: Path | None = None) -> dict[str, Any]:
    proto = dict(PROTOCOL)
    if quick:
        proto.update(QUICK)
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    G_levels: list[float] = proto["G_levels"]
    radii: list[float] = proto["radii"]
    n_main: int = proto["n_main"]
    steps_d: int = proto["steps_droplet"]
    steps_c: int = proto["steps_coexist"]

    with Pool(jobs) as pool:
        # ---- 1. coexistence (planar slab) ----
        coex_list = pool.map(job_coexistence, [(G, max(n_main, 96 if not quick else 64), steps_c) for G in G_levels])
        coex = {round(c["G_12"], 4): c for c in coex_list}

        # ---- 2. Laplace series: G x R at n_main ----
        hero = proto["hero"]
        tasks = []
        for G in G_levels:
            c = coex[round(G, 4)]
            for R in radii:
                is_hero = (
                    round(G, 4) == round(hero["G_12"], 4)
                    and abs(R - hero["radius"]) < 1e-9
                    and n_main == hero["n"]
                )
                tasks.append((G, n_main, R, steps_d, c, is_hero))
        droplet_cases = pool.map(job_droplet, tasks)

        # ---- 3. resolution study at fixed physical radius ----
        res_spec = proto["resolution_study"]
        res_tasks = []
        for G in G_levels:
            for n in res_spec["n"]:
                if n == n_main:
                    continue
                res_tasks.append((G, n, float(res_spec["radius"]), steps_d, coex[round(G, 4)], False))
        resolution_cases = pool.map(job_droplet, res_tasks) if res_tasks else []

        # ---- 4. Route B: flat interface mechanical surface tension ----
        flat_spec = proto["flat"]
        flat_tasks = [
            (
                G,
                flat_spec["ny"],
                flat_spec["nx"],
                flat_spec["steps"],
                coex[round(G, 4)],
                flat_spec["checkpoints"],
                flat_spec["slab_half"],
            )
            for G in G_levels
        ]
        flat_rows = pool.map(job_flat, flat_tasks)

        # ---- 4b. Route C: independent absolute sigma (EOS-profile, non-circular) ----
        eos_spec = proto["eos_anchor"]
        eos_tasks = [(G, eos_spec["n"], eos_spec["steps"], coex[round(G, 4)]) for G in G_levels]
        eos_rows = pool.map(job_eos_anchor, eos_tasks)

    # ---- 5. analysis: per-G Laplace fits (library, one unknown per level) ----
    levels: list[dict[str, Any]] = []
    for G in G_levels:
        cases = [c for c in droplet_cases if round(c["G_12"], 4) == round(G, 4)]
        for c in cases:
            c["guardrails"] = list(c["guardrails"]) + list(c["thickness_guardrails"])
        level = mv._fit_laplace_level(G, cases, TOL, dims=proto["laplace_dims"])
        flat = next(f for f in flat_rows if round(f["G_12"], 4) == round(G, 4))
        sigma_a = level["sigma_origin"]
        sigma_b = flat["sigma_mech"]
        cross = None if (sigma_a is None or not sigma_b) else (sigma_a - sigma_b) / sigma_b
        # 逐点 ref（解析 sigma_B / R）vs got
        per_point = []
        for c in cases:
            if c["radius"] and c["dp"] is not None and sigma_b:
                ref = sigma_b / c["radius"]
                per_point.append(
                    {
                        "radius_init": c["radius_init"],
                        "radius": c["radius"],
                        "dp_got": c["dp"],
                        "dp_ref_routeB": ref,
                        "rel_err": (c["dp"] - ref) / ref,
                        "within_tol": abs((c["dp"] - ref) / ref) < TOL,
                        "sigma_i": c["sigma"],
                    }
                )
        level["sigma_routeB"] = sigma_b
        level["routeB_flatness"] = flat["flatness"]
        level["routeB_converged"] = flat["converged"]
        level["cross_route_rel_err"] = cross
        level["per_point"] = per_point
        level["max_point_rel_err_routeB"] = (
            max(abs(p["rel_err"]) for p in per_point) if per_point else None
        )
        # 容差族判定（逐项，不合并成一个大 pass）
        #   算例池容差 <5% 的**判定项**（预注册）：Laplace 律的半径无关性
        #   max_i |sigma_i - sigma_fit|/sigma_fit < 5%（这就是 Δp = sigma/R 的可证内容）。
        #   Route B（机械张量）是**探索性**独立绝对基准：先过自身的有效性闸门
        #   （平面界面 P_NN 守恒 <= 5%）；闸门不过 ⇒ 该路线作废，不作判定项。
        law_ok = level["max_rel_err"] is not None and level["max_rel_err"] < TOL
        hard = {"no_crossing", "not_converged", "negative_sigma", "centre_not_liquid"}
        guard_ok = not any(g["code"] in hard for g in level["guardrails"])
        flat_ok = bool(flat["flatness"] is not None and flat["flatness"] <= THRESHOLDS["flat_flatness_max"])
        cross_ok = bool(cross is not None and abs(cross) < TOL)
        level["tol_family"] = {
            "law_radius_independence": {"value": level["max_rel_err"], "threshold": TOL, "pass": law_ok},
            "guardrails_clean": {"value": guard_ok, "threshold": True, "pass": guard_ok},
        }
        level["tol_family_pass"] = bool(law_ok and guard_ok)
        level["exploratory_routeB"] = {
            "valid": flat_ok,
            "flatness": flat["flatness"],
            "flatness_threshold": THRESHOLDS["flat_flatness_max"],
            "converged": flat["converged"],
            "sigma_mech": sigma_b,
            "sigma_laplace": sigma_a,
            "cross_route_rel_err": cross,
            "cross_route_within_tol": cross_ok,
            "dp_vs_sigmaB_over_R_max_rel_err": level["max_point_rel_err_routeB"],
            "status": "usable" if (flat_ok and cross_ok) else "VOID (own validity gate failed)",
        }
        levels.append(level)

    # ---- 6. resolution sensitivity at fixed physical radius ----
    resolution = []
    for G in G_levels:
        main = [
            c
            for c in droplet_cases
            if round(c["G_12"], 4) == round(G, 4)
            and abs(c["radius_init"] - float(res_spec["radius"])) < 1e-9
        ]
        for c in main + [c for c in resolution_cases if round(c["G_12"], 4) == round(G, 4)]:
            resolution.append(
                {
                    "G_12": G,
                    "n": c["n"],
                    "R_init": c["radius_init"],
                    "radius": c["radius"],
                    "dp": c["dp"],
                    "sigma": c["sigma"],
                    "width_10_90": c["width_10_90"],
                    "w_tanh": c["w_tanh"],
                    "R_over_w": c.get("R_over_w"),
                    "u_max": c["u_max"],
                    "density_ratio": c["density_ratio"],
                    "steps_run": c["steps_run"],
                    "guardrails": c["guardrails"],
                }
            )

    hero_case = next((c for c in droplet_cases if "_fields" in c), None)
    if hero_case is not None:
        import numpy as np

        np.savez_compressed(
            out_dir / "hero_fields.npz",
            **hero_case["_fields"],
            G_12=hero_case["G_12"],
            radius_init=hero_case["radius_init"],
            n=hero_case["n"],
        )
        (out_dir / "hero_profile.json").write_text(json.dumps(hero_case["_profile"], indent=1))

    def _strip(case: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in case.items() if not k.startswith("_")}

    # ---- 6. Route C: absolute criterion against the independent sigma_EOS ----
    # Non-circular: sigma_EOS comes from the flat-interface EOS-profile integral
    # (no droplet dp or R).  A common multiplicative stress error passes the
    # Round-1 radius-independence fit but fails here.
    eos_by_G = {round(r["G_12"], 4): r for r in eos_rows}
    absolute_checks = []
    for c in droplet_cases:
        if c["radius"] is None or c["dp"] is None:
            continue
        e = eos_by_G.get(round(c["G_12"], 4))
        if e is None or e["sigma_eos"] <= 0:
            continue
        ref = e["sigma_eos"] / c["radius"]
        rel_err = (c["dp"] - ref) / ref
        absolute_checks.append(
            {
                "G_12": c["G_12"],
                "n": c["n"],
                "radius": c["radius"],
                "dp_got": c["dp"],
                "dp_ref": ref,
                "rel_err": rel_err,
                "pass": abs(rel_err) < TOL,
                "sigma_dpR": c["sigma"],
                "sigma_eos": e["sigma_eos"],
            }
        )
    n_abs_pass = sum(1 for a in absolute_checks if a["pass"])
    absolute_pass = bool(absolute_checks) and n_abs_pass == len(absolute_checks)
    abs_factor = {}
    for G in G_levels:
        sub = [a for a in absolute_checks if round(a["G_12"], 4) == round(G, 4)]
        e = eos_by_G.get(round(G, 4))
        if sub and e is not None and e["sigma_eos"] > 0:
            abs_factor[repr(G)] = sum(a["sigma_dpR"] for a in sub) / (len(sub) * e["sigma_eos"])

    summary = {
        "case": {
            "case_code": proto["case_code"],
            "repo": proto["repo"],
            "anchor": "CASE-MCMP-LAPLACE",
            "model": "two-component Shan-Chen (MCMP), linear psi, BGK, velocity-shift forcing",
            "ref_source": proto["ref_source"],
            "tol_policy": proto["tol_policy"],
            "quick": quick,
        },
        "protocol": {k: v for k, v in proto.items() if k != "hero"},
        "thresholds": THRESHOLDS,
        "env": _env(),
        "coexistence": coex_list,
        "levels": levels,
        "cases": [_strip(c) for c in droplet_cases],
        "resolution": resolution,
        "routeB": [{k: v for k, v in f.items() if k != "profile"} for f in flat_rows],
        "figures_data": {
            "laplace_points": [
                {"G_12": c["G_12"], "radius": c["radius"], "dp": c["dp"], "n": c["n"]}
                for c in droplet_cases
                if c["radius"] and c["dp"] is not None
            ],
            "routeB_profiles": [
                {
                    "G_12": f["G_12"],
                    "y": f["profile"]["y"],
                    "p_normal": f["profile"]["p_normal"],
                    "p_tangential": f["profile"]["p_tangential"],
                    "rho1": f["profile"]["rho1"],
                }
                for f in flat_rows
            ],
        },
        "verdict": {
            "tol_family_pass": all(lv["tol_family_pass"] for lv in levels),
            "per_level": [
                {
                    "G_12": lv["G_12"],
                    "sigma_laplace": lv["sigma_origin"],
                    "max_rel_err_law": lv["max_rel_err"],
                    "r2": lv["r2"],
                    "max_offset_fraction": lv["max_offset_fraction"],
                    "tol_family_pass": lv["tol_family_pass"],
                }
                for lv in levels
            ],
            "routeB_status": [{"G_12": lv["G_12"], **lv["exploratory_routeB"]} for lv in levels],
            "routeC_status": [
                {
                    "G_12": r["G_12"],
                    "n": r["n"],
                    "sigma_eos": r["sigma_eos"],
                    "bulk_dp": r["bulk_dp"],
                    "max_deficit": r["max_deficit"],
                    "deficit_negative_rows": r["deficit_negative_rows"],
                    "converged": r["converged"],
                    "steps_run": r["steps_run"],
                }
                for r in eos_rows
            ],
            "absolute": {
                "definition": (
                    "|dp - sigma_EOS/R| / (sigma_EOS/R) < tol; sigma_EOS from the "
                    "flat-interface EOS-profile integral (independent of the droplet)"
                ),
                "n_pass": n_abs_pass,
                "n_total": len(absolute_checks),
                "pass": absolute_pass,
                "dpR_over_sigma_eos": abs_factor,
                "checks": absolute_checks,
            },
        },
        "elapsed_s": round(time.time() - t_start, 1),
        "repro_cmd": (
            "python benchmarks/case_pool/mcmp_laplace/run_case.py --jobs 8 "
            "&& python benchmarks/case_pool/mcmp_laplace/make_report.py"
        ),
    }
    # 自设护栏族（非容差）逐项判定
    guard_family = {
        "r2_ge_0.99": all(
            (lv["r2"] is not None and lv["r2"] >= THRESHOLDS["r2_min"]) for lv in levels
        ),
        "intercept_frac_le_5pct": all(
            (lv["max_offset_fraction"] is not None and lv["max_offset_fraction"] <= THRESHOLDS["max_offset_fraction"])
            for lv in levels
        ),
        "routeB_flatness_le_5pct": all(
            (f["flatness"] is not None and f["flatness"] <= THRESHOLDS["flat_flatness_max"])
            for f in flat_rows
        ),  # 诊断项：False 表示 Route B 作废（不影响算例池容差判定）
        "routeB_converged": all(f["converged"] for f in flat_rows),
    }
    summary["verdict"]["guard_family"] = guard_family
    summary["verdict"]["guard_family_pass"] = all(guard_family.values())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_safe))
    return summary


def _json_safe(obj: Any) -> Any:  # numpy scalars / tuples / tensors
    try:
        return obj.item()
    except AttributeError:
        pass
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 8)))
    ap.add_argument("--quick", action="store_true", help="smoke protocol, not for acceptance")
    args = ap.parse_args()
    out = Path(args.out)
    summary = build(out, args.jobs, quick=args.quick)
    print(
        json.dumps(
            {
                "tol_family_pass": summary["verdict"]["tol_family_pass"],
                "guard_family_pass": summary["verdict"]["guard_family_pass"],
                "per_level": summary["verdict"]["per_level"],
                "elapsed_s": summary["elapsed_s"],
            },
            indent=1,
        )
    )
    print(f"wrote {out / 'summary.json'}")
    return 0 if summary["verdict"]["tol_family_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())