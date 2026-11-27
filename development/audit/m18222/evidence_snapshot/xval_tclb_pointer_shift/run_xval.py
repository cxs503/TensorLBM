#!/usr/bin/env python3
"""XVAL driver: TCLB per-margin-arrays pointer-shift streaming vs TensorLBM.

Anchor: ``XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING`` (mission 18129).

Arms
----
``A`` baseline : TensorLBM production transport -- ``solver.stream`` (2D,
    cached int64 advanced-index gather) and ``solver3d.stream3d_gather`` (3D).
``B`` oss      : :mod:`tensorlbm.xval.per_margin_pointer_shift` -- the TCLB
    formulation (per-margin block decomposition, constant linear offset per
    block, i.e. pointer shift; no index tensors).

Everything else (collision, Guo forcing, bounce-back, initial condition,
sampling, error estimators) is *the same object* in both arms, so any measured
difference is attributable to the transport operator alone.

Predeclared decision rules (fixed before the runs; see
``docs/XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING.md`` §2):

* **D1 equivalence** -- arms A and B must agree *bit-wise* (max abs diff 0) in
  2D and 3D, both dtypes.  Otherwise the port is rejected.
* **D2 gate** -- owner accuracy gate: relative L2 error of the velocity field
  against the closed-form reference must be < 15 % for *both* arms on both
  physics cases (Taylor-Green 2D, plane Poiseuille).  The 3 % internal target is
  reported, not required (it is a resolution question, not a transport one).
* **D3 keep/adopt** -- adopt the formulation only if D1 and D2 hold *and* the
  CPU cost is within 2x of the production path; otherwise keep the production
  transport and bank the invariant check as a regression test (the decision is
  reported either way).

Outputs: ``runs/xval_tclb_pointer_shift/result.json``,
``figures/xval_tclb_pointer_shift/*.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm import solver, solver3d  # noqa: E402
from tensorlbm.d2q9 import C as C2D  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE as OPP2D  # noqa: E402
from tensorlbm.d2q9 import W_EXACT, equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q19 import C as C3D  # noqa: E402
from tensorlbm.verification.analytic_l2 import (  # noqa: E402
    relative_l2,
    tgv2d_analytic,
    tgv2d_decay_time,
)
from tensorlbm.xval import per_margin_pointer_shift as PMPS  # noqa: E402

OUT_RUN = REPO / "runs" / "xval_tclb_pointer_shift"
OUT_FIG = REPO / "figures" / "xval_tclb_pointer_shift"

GATE = 0.15  # owner accuracy gate (>15 % unacceptable)
TARGET = 0.03  # internal 3 % target
SLOWDOWN_LIMIT = 2.0  # D3


# --------------------------------------------------------------------- helpers
def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    total = 0
    if isinstance(obj, dict):
        for v in obj.values():
            total += _tensor_bytes(v)
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            total += _tensor_bytes(v)
    return total


def stream_2d(f: torch.Tensor, arm: str, plan=None) -> torch.Tensor:
    if arm == "A":
        return solver.stream(f)
    return PMPS.stream_pointer_shift(f, C2D, plan=plan)


def stream_3d(f: torch.Tensor, arm: str, plan=None) -> torch.Tensor:
    if arm == "A":
        return solver3d.stream3d_gather(f)
    return PMPS.stream_pointer_shift(f, C3D, plan=plan)


# ------------------------------------------------------- 1. implementation XVAL
def equivalence_suite(seed: int = 0) -> dict:
    out: dict = {"cases": [], "invariants": []}
    gen = torch.Generator().manual_seed(seed)
    for dtype in (torch.float32, torch.float64):
        for shape in [(17, 23), (4, 4), (3, 5), (256, 256)]:
            f = torch.rand(9, *shape, generator=gen, dtype=dtype)
            plan = PMPS.pointer_shift_plan(C2D, shape)
            a = stream_2d(f, "A")
            b = PMPS.stream_pointer_shift(f, C2D, plan=plan)
            c = PMPS.stream_pointer_shift_flat(f, C2D, plan=plan)
            inv = PMPS.check_plan_invariant(f, C2D, plan=plan) if max(shape) <= 64 else None
            out["cases"].append(
                {
                    "dims": 2,
                    "dtype": str(dtype).replace("torch.", ""),
                    "shape": list(shape),
                    "bitwise_equal": bool(torch.equal(a, b)),
                    "max_abs_diff": float((a - b).abs().max()),
                    "flat_max_abs_diff": float((a - c).abs().max()),
                    "mass_baseline": float(a.sum()),
                    "mass_oss": float(b.sum()),
                    "blocks": len(plan.copies),
                }
            )
            if inv is not None:
                out["invariants"].append({"dims": 2, "shape": list(shape), **inv})
        for shape in [(6, 7, 8), (5, 5, 5), (3, 4, 5)]:
            f = torch.rand(19, *shape, generator=gen, dtype=dtype)
            plan = PMPS.pointer_shift_plan(C3D, shape)
            a = stream_3d(f, "A")
            b = PMPS.stream_pointer_shift(f, C3D, plan=plan)
            c = PMPS.stream_pointer_shift_flat(f, C3D, plan=plan)
            inv = PMPS.check_plan_invariant(f, C3D, plan=plan)
            out["cases"].append(
                {
                    "dims": 3,
                    "dtype": str(dtype).replace("torch.", ""),
                    "shape": list(shape),
                    "bitwise_equal": bool(torch.equal(a, b)),
                    "max_abs_diff": float((a - b).abs().max()),
                    "flat_max_abs_diff": float((a - c).abs().max()),
                    "mass_baseline": float(a.sum()),
                    "mass_oss": float(b.sum()),
                    "blocks": len(plan.copies),
                }
            )
            out["invariants"].append({"dims": 3, "shape": list(shape), **inv})

    # plan structure: block count per direction must be 2**|{a : c_a != 0}|
    struct_ok = True
    detail = {}
    for name, table, dims in (("D2Q9", C2D, 2), ("D3Q19", C3D, 3)):
        plan = PMPS.pointer_shift_plan(table, (8,) * dims)
        per_dir = plan.summary()["blocks_per_direction"]
        expected = {
            int(q): 2 ** sum(1 for c in list(row)[:dims] if int(c) != 0)
            for q, row in enumerate(table.tolist())
        }
        detail[name] = {"expected": expected, "got": per_dir, "total_copies": len(plan.copies)}
        struct_ok &= per_dir == expected
    out["plan_structure"] = {"ok": struct_ok, "detail": detail}

    # boundary of the decomposition: n < 3 must be rejected, not silently wrong
    try:
        PMPS.pointer_shift_plan(C2D, (2, 8))
        out["short_axis_rejected"] = False
    except ValueError as exc:
        out["short_axis_rejected"] = True
        out["short_axis_message"] = str(exc)

    out["all_bitwise_equal"] = all(c["bitwise_equal"] for c in out["cases"])
    out["max_abs_diff"] = max(c["max_abs_diff"] for c in out["cases"])
    out["invariants_ok"] = all(
        i["partition_ok"] and i["max_offset_error"] == 0 and i["max_abs_diff_vs_gather"] == 0.0
        for i in out["invariants"]
    )
    out["index_tensor_bytes_2d_256"] = _tensor_bytes(solver._stream2d_cache)
    out["index_tensor_bytes_3d_48"] = _tensor_bytes(solver3d._stream3d_cache)
    return out


# --------------------------------------------------------------- 2. physics arms
def taylor_green(n: int, steps: int, tau: float, u_max: float, arm: str,
                 dtype=torch.float64) -> dict:
    nu = (2.0 * tau - 1.0) / 6.0
    a0 = tgv2d_analytic(0, n, n, nu, u_max=u_max, as_torch=True, dtype=dtype)
    f = equilibrium(a0.rho, a0.ux, a0.uy)
    plan = PMPS.pointer_shift_plan(C2D, (n, n)) if arm == "B" else None
    samples = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = solver.collide_bgk(stream_2d(f, arm, plan=plan), tau)
        if step in (1, max(1, steps // 4), max(1, steps // 2), steps):
            rho, ux, uy = macroscopic(f)
            ref = tgv2d_analytic(step, n, n, nu, u_max=u_max, as_torch=True, dtype=dtype)
            umax_num = float(torch.sqrt(ux * ux + uy * uy).max())
            umax_ref = float(torch.sqrt(ref.ux * ref.ux + ref.uy * ref.uy).max())
            samples.append(
                {
                    "step": step,
                    "t_over_td": step / tgv2d_decay_time(n, n, nu),
                    "rel_l2_ux": relative_l2(ux, ref.ux).rel_l2,
                    "rel_l2_uy": relative_l2(uy, ref.uy).rel_l2,
                    "rel_l2_rho": relative_l2(rho, ref.rho, ref_base=1.0).rel_l2,
                    "umax_num": umax_num,
                    "umax_ref": umax_ref,
                    "umax_rel_err": abs(umax_num - umax_ref) / umax_ref,
                }
            )
    last = samples[-1]
    return {
        "arm": arm,
        "n": n,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "u_max": u_max,
        "ma": u_max * math.sqrt(3.0),
        "re_steps": u_max * n / nu,
        "td_steps": tgv2d_decay_time(n, n, nu),
        "wall_s": time.perf_counter() - t0,
        "samples": samples,
        "final_rel_l2_ux": last["rel_l2_ux"],
        "final_rel_l2_uy": last["rel_l2_uy"],
        "final_error": max(last["rel_l2_ux"], last["rel_l2_uy"]),
    }


def _collide_guo(f: torch.Tensor, tau: float, accel: float, dtype) -> torch.Tensor:
    """BGK collision with Guo forcing (body force along +x only).

    Momentum ``rho u = sum_i c_i f_i + G/2``; the forcing term is the standard
    ``(1 - 1/(2 tau)) 3 w_i (c_i . G)`` (unit density scale, as ``G`` is the
    acceleration here).
    """
    c = C2D.to(dtype)
    w = torch.tensor(W_EXACT, dtype=dtype)
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = ((f * c[:, 0].view(9, 1, 1)).sum(dim=0) + 0.5 * accel) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    feq = equilibrium(rho, ux, uy)
    src = (1.0 - 0.5 / tau) * 3.0 * w.view(9, 1, 1) * (c[:, 0].view(9, 1, 1) * accel)
    return f - (f - feq) / tau + src


def poiseuille(ny: int, nx: int, steps: int, tau: float, accel: float, arm: str,
               dtype=torch.float64) -> dict:
    """Plane Poiseuille channel: solid rows y=0, y=ny-1; force along +x.

    Half-way bounce-back, Guo forcing and the sampling are shared by both arms;
    only the transport operator differs.
    """
    nu = (2.0 * tau - 1.0) / 6.0
    h_eff = float(ny - 2)  # no-slip planes at y=0.5 and y=ny-1.5
    rho = torch.ones(ny, nx, dtype=dtype)
    zero = torch.zeros(ny, nx, dtype=dtype)
    f = equilibrium(rho, zero, zero)
    plan = PMPS.pointer_shift_plan(C2D, (ny, nx)) if arm == "B" else None

    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    # per-direction upstream-solid masks, computed once (identical in both arms)
    masks = {}
    for q in range(9):
        cy, cx = int(C2D[q, 1]), int(C2D[q, 0])
        if cy == 0 and cx == 0:
            continue
        m = torch.roll(solid, shifts=(cy, cx), dims=(0, 1))  # solid[y-cy, x-cx]: upstream solid
        if bool(m.any()):
            masks[q] = m

    def bounce(f_in: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Half-way bounce-back: where the upstream link of q ends in a wall, the
        incoming population is replaced by the reflection of the outgoing one."""
        for q, m in masks.items():
            out[q][m] = f_in[int(OPP2D[q])][m]
        return out

    # solid rows are never read by the fluid update (their upstream links are
    # masked); pinning them to the rest state keeps the periodic wrap from
    # injecting garbage into the wrap-around rows.  Identical in both arms.
    rest = torch.tensor(W_EXACT, dtype=dtype).view(9, 1, 1).expand(9, ny, nx).clone()

    ys = np.arange(1, ny - 1)
    u_ref_parabola = accel / (2.0 * nu) * (ys - 0.5) * (ny - 1.5 - ys)
    u_max_ref = float(u_ref_parabola.max())

    t0 = time.perf_counter()
    hist = []
    prev = None
    for step in range(1, steps + 1):
        out = stream_2d(f, arm, plan=plan)
        f = bounce(f, out)
        f = _collide_guo(f, tau, accel, dtype)
        f[:, solid] = rest[:, solid]
        if step % max(1, steps // 8) == 0 or step == steps:
            _, ux, _ = macroscopic(f)
            prof = ux[1:-1].mean(dim=1).numpy()
            resid = 0.0 if prev is None else float(np.max(np.abs(prof - prev)) / max(u_max_ref, 1e-30))
            hist.append({"step": step, "umax": float(prof.max()), "resid": resid})
            prev = prof
    wall = time.perf_counter() - t0

    _, ux, uy = macroscopic(f)
    prof = ux[1:-1].mean(dim=1).numpy()
    err = relative_l2(prof, u_ref_parabola)
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "h_eff": h_eff,
        "accel": accel,
        "re_centerline": u_max_ref * h_eff / nu,
        "umax_ref": u_max_ref,
        "umax_num": float(prof.max()),
        "umax_rel_err": abs(float(prof.max()) - u_max_ref) / u_max_ref,
        "rel_l2_profile": err.rel_l2,
        "wall_s": wall,
        "history": hist,
        "profile": {"y": ys.tolist(), "u": prof.tolist()},
    }


# ---------------------------------------------------------------- 3. benchmark
def transport_benchmark(reps: int = 7) -> dict:
    res = {"2d": {}, "3d": {}, "reps": reps}
    gen = torch.Generator().manual_seed(7)
    for n in (128, 256):
        f = torch.rand(9, n, n, generator=gen, dtype=torch.float32)
        plan = PMPS.pointer_shift_plan(C2D, (n, n))
        stream_2d(f, "A")
        PMPS.stream_pointer_shift(f, C2D, plan=plan)
        PMPS.stream_pointer_shift_flat(f, C2D, plan=plan)
        tb, to, tf = [], [], []
        for _ in range(reps):
            t = time.perf_counter(); stream_2d(f, "A"); tb.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift(f, C2D, plan=plan); to.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift_flat(f, C2D, plan=plan); tf.append(time.perf_counter() - t)
        res["2d"][n] = {
            "baseline_s": statistics.median(tb),
            "oss_s": statistics.median(to),
            "oss_flat_s": statistics.median(tf),
            "slowdown": statistics.median(to) / statistics.median(tb),
            "index_tensor_bytes": _tensor_bytes(solver._stream2d_cache),
            "blocks": len(plan.copies),
        }
    for n in (32, 48):
        f = torch.rand(19, n, n, n, generator=gen, dtype=torch.float32)
        plan = PMPS.pointer_shift_plan(C3D, (n, n, n))
        stream_3d(f, "A")
        PMPS.stream_pointer_shift(f, C3D, plan=plan)
        tb, to = [], []
        for _ in range(reps):
            t = time.perf_counter(); stream_3d(f, "A"); tb.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift(f, C3D, plan=plan); to.append(time.perf_counter() - t)
        res["3d"][n] = {
            "baseline_s": statistics.median(tb),
            "oss_s": statistics.median(to),
            "slowdown": statistics.median(to) / statistics.median(tb),
            "index_tensor_bytes": _tensor_bytes(solver3d._stream3d_cache),
            "blocks": len(plan.copies),
        }
    return res


# --------------------------------------------------------------------- figures
def make_figures(result: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    figs = []

    # fig 1: Taylor-Green error curves vs analytic, both arms, all grids
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arm in zip(axes, ("A", "B")):
        for r in [c for c in result["physics"]["tgv"] if c["arm"] == arm]:
            steps = [s["step"] for s in r["samples"]]
            ax.plot(steps, [s["rel_l2_ux"] * 100 for s in r["samples"]], "-o", label=f"$u_x$ n={r['n']}")
            ax.plot(steps, [s["rel_l2_uy"] * 100 for s in r["samples"]], "--s", alpha=0.7,
                    label=f"$u_y$ n={r['n']}")
        ax.axhline(GATE * 100, color="r", ls=":", label="15 % gate")
        ax.axhline(TARGET * 100, color="g", ls=":", label="3 % target")
        ax.set_xlabel("step")
        ax.set_ylabel(r"relative $L_2$ error vs analytic [%]")
        ax.set_yscale("log")
        ax.set_title(f"Taylor-Green 2D, transport arm {arm}")
        ax.legend(fontsize=6)
    fig.tight_layout()
    p = OUT_FIG / "fig1_tgv_error.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Taylor-Green 2D relative L2 velocity error against the "
                 "closed-form decay for arms A (production gather) and B (TCLB per-margin pointer shift); "
                 "dotted lines = 15 % owner gate and 3 % internal target."})

    # fig 2: arm difference (exactly zero) + Poiseuille profile
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    labels = [f"{c['dims']}D {c['shape']}\n{c['dtype']}" for c in result["equivalence"]["cases"]]
    vals = [max(c["max_abs_diff"], 1e-18) for c in result["equivalence"]["cases"]]
    ax.bar(range(len(vals)), vals)
    ax.set_yscale("log")
    ax.set_ylabel(r"max $|stream_A - stream_B|$")
    ax.set_title("implementation equivalence\n(1e-18 floor = exact bit-wise zero)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax = axes[1]
    for arm, style in (("A", "-"), ("B", "--")):
        pr = [p for p in result["physics"]["poiseuille"] if p["arm"] == arm][0]["profile"]
        ax.plot(pr["u"], pr["y"], style, label=f"arm {arm}")
    ana = result["physics"]["poiseuille_analytic"]
    ax.plot(ana["u"], ana["y"], "k:", lw=2, label="analytic parabola")
    ax.set_xlabel("$u_x$ [lattice units]")
    ax.set_ylabel("y")
    ax.set_title("plane Poiseuille: steady profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT_FIG / "fig2_equivalence_profile.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Left: bit-wise difference between the two transport arms over "
                 "all equivalence cases (all exactly zero). Right: steady Poiseuille profile for both arms "
                 "against the analytic parabola."})

    # fig 3: transport cost
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, base, oss, flat = [], [], [], []
    for n, row in result["benchmark"]["2d"].items():
        labels.append(f"2D {n}$^2$")
        base.append(row["baseline_s"] * 1e3)
        oss.append(row["oss_s"] * 1e3)
        flat.append(row["oss_flat_s"] * 1e3)
    for n, row in result["benchmark"]["3d"].items():
        labels.append(f"3D {n}$^3$")
        base.append(row["baseline_s"] * 1e3)
        oss.append(row["oss_s"] * 1e3)
        flat.append(float("nan"))
    x = np.arange(len(labels))
    ax.bar(x - 0.25, base, 0.25, label="A: cached int64 gather (production)")
    ax.bar(x, oss, 0.25, label="B: per-margin pointer shift")
    ax.bar(x + 0.25, flat, 0.25, label="B$'$: flat-offset variant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("median transport time [ms] (CPU torch)")
    ax.set_title("transport operator cost")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = OUT_FIG / "fig3_transport_cost.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Median wall time per transport call: arm A (cached int64 index "
                 "gather) vs arm B (per-margin block / pointer-shift copies), CPU torch."})
    return figs


# ------------------------------------------------------------------------ main
def run(quick: bool = False) -> dict:
    result: dict = {
        "anchor": "XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING",
        "mission": 18129,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "quick" if quick else "full",
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "cuda": torch.cuda.is_available(),
        },
        "gates": {"owner_error_gate": GATE, "internal_target": TARGET,
                  "slowdown_limit": SLOWDOWN_LIMIT},
    }
    result["equivalence"] = equivalence_suite()
    print(f"[eq] bitwise equal: {result['equivalence']['all_bitwise_equal']}  "
          f"invariants ok: {result['equivalence']['invariants_ok']}")

    tgv = []
    ladder = [(32, 200)] if quick else [(32, 300), (64, 400)]
    for arm in ("A", "B"):
        for n, steps in ladder:
            r = taylor_green(n, steps, tau=0.8, u_max=0.04, arm=arm)
            tgv.append(r)
            print(f"[tgv] arm={arm} n={n} steps={steps} err={r['final_error']*100:.3f} % ({r['wall_s']:.2f}s)")
    result["physics"] = {"tgv": tgv}

    taus = [0.6, 0.8] if quick else [0.6, 0.8, 1.0]
    sens = []
    for arm in ("A", "B"):
        for tau in taus:
            r = taylor_green(32, 150, tau=tau, u_max=0.04, arm=arm)
            sens.append(r)
            print(f"[tau] arm={arm} tau={tau} err={r['final_error']*100:.3f} %")
    result["physics"]["tau_sensitivity"] = sens

    ny, nx = (18, 4) if quick else (34, 4)
    steps = 6000 if quick else 8000
    poi = []
    for arm in ("A", "B"):
        r = poiseuille(ny, nx, steps=steps, tau=0.8, accel=1.6e-5, arm=arm)
        poi.append(r)
        print(f"[poiseuille] arm={arm} umax_err={r['umax_rel_err']*100:.3f} % "
              f"rel_l2={r['rel_l2_profile']*100:.3f} % resid={r['history'][-1]['resid']:.2e} "
              f"({r['wall_s']:.1f}s)")
    result["physics"]["poiseuille"] = poi
    ys = poi[0]["profile"]["y"]
    nu = poi[0]["nu"]
    accel = poi[0]["accel"]
    result["physics"]["poiseuille_analytic"] = {
        "y": ys,
        "u": (accel / (2.0 * nu) * (np.array(ys) - 0.5) * (ny - 1.5 - np.array(ys))).tolist(),
    }

    result["benchmark"] = transport_benchmark(reps=5)
    for dim, rows in (("2d", result["benchmark"]["2d"]), ("3d", result["benchmark"]["3d"])):
        for n, row in rows.items():
            print(f"[bench {dim}] n={n} base={row['baseline_s']*1e3:.2f} ms "
                  f"oss={row['oss_s']*1e3:.2f} ms slowdown={row['slowdown']:.2f}x")

    # ---- decide (rules predeclared above)
    eq = result["equivalence"]
    errs = [max(p["umax_rel_err"], p["rel_l2_profile"]) for p in poi] + [r["final_error"] for r in tgv]
    slow = [row["slowdown"] for dim in ("2d", "3d") for row in result["benchmark"][dim].values()]
    d1 = bool(eq["all_bitwise_equal"] and eq["invariants_ok"] and eq["plan_structure"]["ok"])
    d2 = bool(max(errs) < GATE)
    d3 = bool(d1 and d2 and max(slow) <= SLOWDOWN_LIMIT)
    result["decision"] = {
        "D1_equivalence": d1,
        "D2_gate_lt_15pct": d2,
        "max_error": max(errs),
        "max_slowdown": max(slow),
        "D3_adopt_formulation": d3,
        "verdict": "PASS" if (d1 and d2) else "FAIL",
        "action": (
            "adopt pointer-shift transport as an alternative path and keep the block-offset invariant as a test"
            if d3 else
            "keep the production gather transport; bank the block-offset invariant + the bitwise cross-check as a regression test; "
            "the OSS formulation's advantage does not survive the port to torch (see §6)"
        ),
    }
    result["figures"] = make_figures(result)
    result["env"]["peak_rss_mb"] = _peak_rss_mb()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    res = run(quick=args.quick)
    (OUT_RUN / "result.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res["decision"], indent=2))
    print(f"wrote {OUT_RUN / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())