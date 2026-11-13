#!/usr/bin/env python3
"""XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT -- driver.

Runs the arms predeclared in ``predeclared.json`` and writes ``results.json``
plus three figures.  Every number in the report comes from this file's output
(no hand-typed values).

Usage::

    PYTHONPATH=src python3 xval/XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT/run_xval.py
"""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import torch

from tensorlbm.warp_shift_streaming import (
    bgk_stream_step,
    kinetic_energy,
    oss_literal_stream,
    pull_stream,
    taylor_green_field,
    warp_shift_stream,
)

HERE = Path(__file__).resolve().parent
TILES = (8, 16, 32)


def step_pull(f: torch.Tensor, tau: float) -> torch.Tensor:
    from tensorlbm.d2q9 import equilibrium, macroscopic

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return pull_stream(f + (feq - f) / tau)


def k1_k2() -> dict:
    """K1 (vectorised == OSS literal) and K2 (periodic == pull reference)."""
    torch.manual_seed(20261113)
    out = {"K1": [], "K2": []}
    fields = {
        "random_96x40": torch.rand(9, 40, 96, dtype=torch.float64),
        "taylor_green_L64": taylor_green_field(64, u0=0.04),
    }
    for name, f in fields.items():
        for tile in TILES:
            if f.shape[2] % tile:
                continue
            lit = oss_literal_stream(f, tile=tile)
            vec = warp_shift_stream(f, tile=tile, x_periodic=False, y_periodic=False)
            out["K1"].append(
                {
                    "field": name,
                    "tile": tile,
                    "max_abs_diff": float((vec - lit).abs().max()),
                    "bitwise_equal": bool(torch.equal(vec, lit)),
                }
            )
            per = warp_shift_stream(f, tile=tile, x_periodic=True, y_periodic=True)
            ref = pull_stream(f)
            out["K2"].append(
                {
                    "field": name,
                    "tile": tile,
                    "arm": "single_step",
                    "max_abs_diff": float((per - ref).abs().max()),
                    "bitwise_equal": bool(torch.equal(per, ref)),
                }
            )
    # coupled run: 200 BGK steps, warp-shift vs pull
    for tile in (16, 32):
        a = taylor_green_field(64, u0=0.04)
        b = a.clone()
        for _ in range(200):
            a = bgk_stream_step(a, tau=0.8, tile=tile)
            b = step_pull(b, tau=0.8)
        out["K2"].append(
            {
                "field": "taylor_green_L64_after200steps",
                "tile": tile,
                "arm": "coupled_200_steps",
                "max_abs_diff": float((a - b).abs().max()),
                "bitwise_equal": bool(torch.equal(a, b)),
            }
        )
    return out


def decay_rate(
    f0: torch.Tensor, steps: int, tau: float, tile: int, backend: str
) -> dict:
    e0 = float(kinetic_energy(f0))
    f = f0
    t0 = time.perf_counter()
    for _ in range(steps):
        f = (
            bgk_stream_step(f, tau=tau, tile=tile)
            if backend == "warp"
            else step_pull(f, tau)
        )
    wall = time.perf_counter() - t0
    e1 = float(kinetic_energy(f))
    lam = -math.log(e1 / e0) / (2.0 * steps)
    mass0 = float(f0.sum())
    mass1 = float(f.sum())
    return {
        "E0": e0,
        "E1": e1,
        "lambda_measured": lam,
        "mass_drift_rel": abs(mass1 - mass0) / abs(mass0),
        "wall_s": wall,
        "field": f,
    }


def k3_k4() -> dict:
    """K3 analytic decay on the (L, u0) grid; K4 defect of the as-is kernel."""
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    steps = 800
    rows = []
    curves = {}
    for L in (64, 128, 256):
        for u0 in (0.02, 0.04, 0.08):
            f0 = taylor_green_field(L, u0=u0)
            k = 2.0 * math.pi / L
            # velocity mode decays as exp(-nu (kx^2+ky^2) t) = exp(-2 nu k^2 t);
            # with E = E0 exp(-2 lambda t) the analytic rate is lambda = 2 nu k^2
            lam_ref = 2.0 * nu * k * k
            r = decay_rate(f0, steps, tau, 32, "warp")
            err = abs(r["lambda_measured"] - lam_ref) / lam_ref
            rows.append(
                {
                    "L": L,
                    "u0": u0,
                    "Ma": u0 * math.sqrt(3.0),
                    "Re": u0 * L / nu,
                    "tau": tau,
                    "nu": nu,
                    "k": k,
                    "steps": steps,
                    "lambda_analytic": lam_ref,
                    "lambda_measured": r["lambda_measured"],
                    "rel_err": err,
                    "pass_15pct": err < 0.15,
                    "mass_drift_rel": r["mass_drift_rel"],
                    "wall_s": r["wall_s"],
                }
            )
            if u0 == 0.04:
                curves[L] = {"analytic": lam_ref}
    # decay curves for the figure: analytic / warp-shift / as-is verbatim
    L, u0 = 64, 0.04
    k = 2.0 * math.pi / L
    lam = 2.0 * nu * k * k
    f0 = taylor_green_field(L, u0=u0)
    e0 = float(kinetic_energy(f0))
    steps_c = 400
    f = f0
    g = f0
    curve = {"t": [], "analytic": [], "warp_shift": [], "oss_verbatim": []}
    for n in range(steps_c + 1):
        if n % 20 == 0:
            curve["t"].append(n)
            curve["analytic"].append(e0 * math.exp(-2.0 * lam * n))
            curve["warp_shift"].append(float(kinetic_energy(f)))
            ev = float(kinetic_energy(g))
            curve["oss_verbatim"].append(ev if math.isfinite(ev) else None)
        if n == steps_c:
            break
        f = bgk_stream_step(f, tau=tau, tile=32)
        # as-is upstream kernel: no domain wrap, no cross-edge repair
        from tensorlbm.d2q9 import equilibrium, macroscopic

        rho, ux, uy = macroscopic(g)
        feq = equilibrium(rho, ux, uy)
        g = warp_shift_stream(
            g + (feq - g) / tau, tile=32, x_periodic=False, y_periodic=False
        )
        if not torch.isfinite(g).all():
            g = torch.zeros_like(g) + float("nan")
    curve["lam_analytic"] = lam
    curve["L"] = L
    curve["u0"] = u0
    curve["tau"] = tau
    return {"K3": rows, "decay_curve": curve}


def k4_verbatim() -> dict:
    """Defect of the as-is kernel: mass drift, edge mismatch and time-to-divergence."""
    tau = 0.8
    L, u0, max_steps = 64, 0.04, 100
    f0 = taylor_green_field(L, u0=u0)
    from tensorlbm.d2q9 import equilibrium, macroscopic

    m0 = float(f0.sum())
    per = f0
    for _ in range(10):
        per = bgk_stream_step(per, tau=tau, tile=32)
    g = f0
    n10 = None
    steps_to_nonfinite = None
    mass10 = None
    rec = []
    for n in range(1, max_steps + 1):
        rho, ux, uy = macroscopic(g)
        feq = equilibrium(rho, ux, uy)
        g = warp_shift_stream(
            g + (feq - g) / tau, tile=32, x_periodic=False, y_periodic=False
        )
        if not torch.isfinite(g).all():
            steps_to_nonfinite = n
            break
        if n == 10:
            mass10 = float(g.sum())
            n10 = g.clone()
            rec.append(mass10)
    out = {
        "steps": 10,
        "L": L,
        "u0": u0,
        "tau": tau,
        "mass0": m0,
        "mass_after_10_steps": mass10,
        "mass_drift_rel_10_steps": abs(mass10 - m0) / abs(m0) if mass10 else None,
        "steps_to_nonfinite": steps_to_nonfinite,
        "kinetic_energy_verbatim_10": float(kinetic_energy(n10)) if n10 is not None else None,
        "kinetic_energy_periodic_10": float(kinetic_energy(per)),
    }
    if n10 is not None:
        interior = slice(1, L - 1)
        out["interior_max_abs_diff_vs_periodic"] = float(
            (n10[:, interior, interior] - per[:, interior, interior]).abs().max()
        )
        out["edge_max_abs_diff_vs_periodic"] = float(
            (n10[:, [0, -1], :] - per[:, [0, -1], :]).abs().max()
        )
    return out


def k5_timing() -> dict:
    """MLUPS of pull vs warp-shift at several tile widths (median of 5)."""
    tau = 0.8
    steps = 30
    reps = 5
    rows = []
    for L in (128, 256):
        for backend, tile in [("pull", 0)] + [("warp", t) for t in (8, 16, 32, 64, 128)]:
            if backend == "warp" and L % tile:
                continue
            f = taylor_green_field(L, u0=0.04)
            times = []
            for _ in range(reps):
                g = f
                t0 = time.perf_counter()
                for _ in range(steps):
                    g = (
                        bgk_stream_step(g, tau=tau, tile=tile)
                        if backend == "warp"
                        else step_pull(g, tau)
                    )
                times.append(time.perf_counter() - t0)
            times.sort()
            med = times[reps // 2]
            rows.append(
                {
                    "L": L,
                    "backend": backend,
                    "tile": tile if backend == "warp" else None,
                    "steps": steps,
                    "reps": reps,
                    "median_wall_s": med,
                    "min_wall_s": times[0],
                    "max_wall_s": times[-1],
                    "mlups": L * L * steps / med / 1e6,
                    "mlups_best": L * L * steps / times[0] / 1e6,
                }
            )
    base = {(r["L"]): r["mlups"] for r in rows if r["backend"] == "pull"}
    base_best = {(r["L"]): r["mlups_best"] for r in rows if r["backend"] == "pull"}
    for r in rows:
        r["speedup_vs_pull"] = r["mlups"] / base[r["L"]]
        r["speedup_vs_pull_best"] = r["mlups_best"] / base_best[r["L"]]
        r["time_spread_rel"] = (r["max_wall_s"] - r["min_wall_s"]) / r["median_wall_s"]
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)  # timing arm: one CPU thread, no oversubscription
    t_start = time.perf_counter()
    k12 = k1_k2()
    k34 = k3_k4()
    k4 = k4_verbatim()
    k5 = k5_timing()
    elapsed = time.perf_counter() - t_start
    thr = json.loads((HERE / "predeclared.json").read_text())["predeclared_thresholds"]

    k1_worst = max(r["max_abs_diff"] for r in k12["K1"])
    k2_worst = max(r["max_abs_diff"] for r in k12["K2"])
    k3_worst = max(r["rel_err"] for r in k34["K3"])
    checks = {
        "K1_max_abs_diff": k1_worst,
        "K1_pass": k1_worst <= thr["K1_max_abs_diff"],
        "K2_max_abs_diff": k2_worst,
        "K2_pass": k2_worst <= thr["K2_max_abs_diff"],
        "K3_worst_rel_err": k3_worst,
        "K3_pass": k3_worst < thr["K3_decay_rel_err"],
        "K4_verbatim_mass_drift_rel": k4["mass_drift_rel_10_steps"],
        "K4_as_is_rejected": (k4["mass_drift_rel_10_steps"] or 0.0) > thr["K4_verbatim_mass_drift_rel"],
        "K5_best_warp_speedup_median": max(r["speedup_vs_pull"] for r in k5 if r["backend"] == "warp"),
        "K5_best_warp_speedup_best": max(r["speedup_vs_pull_best"] for r in k5 if r["backend"] == "warp"),
        "K5_worst_time_spread_rel": max(r["time_spread_rel"] for r in k5),
    }
    checks["K5_cpu_adopt"] = (
        checks["K5_best_warp_speedup_median"] >= 1.0
        and checks["K5_worst_time_spread_rel"] < 0.20
    )

    verdict = (
        "D2: adopt the rule (row-tile staging + one-lane modular shift + cross-tile "
        "repair) with both domain wraps closed as a documented streaming backend; "
        "as-is (D1) rejected because the domain edges are never repaired; CPU keeps "
        "the periodic pull operator as reference."
        if checks["K1_pass"] and checks["K2_pass"] and checks["K3_pass"]
        else "H0 falsified: a KPI is outside its predeclared threshold"
    )

    results = {
        "anchor": "XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT",
        "mission": 18329,
        "oss": {
            "project": "PorousMediaSimulation/openLBMPM",
            "commit": "3d841891daafe7b6e9f65e51dd9bf20f2b0e1792",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "dtype": "float64",
            "torch_num_threads": 1,
        },
        "checks": checks,
        "thresholds": thr,
        "K1_K2": k12,
        "K3": k34["K3"],
        "decay_curve": {k: v for k, v in k34["decay_curve"].items()},
        "K4_verbatim_defect": k4,
        "K5_timing": k5,
        "verdict": verdict,
        "wall_s_total": elapsed,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "decay_curve"}, indent=2)[:6000])


if __name__ == "__main__":
    main()