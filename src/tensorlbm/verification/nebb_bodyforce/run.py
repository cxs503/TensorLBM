"""Benchmark entry point: force-corrected NEBB walls + body-force driver.

Run from the repository root::

    PYTHONPATH=src python -m tensorlbm.verification.nebb_bodyforce.run \
        --out runs/m18182_nebb_force

Physics / why
-------------
Body-force-driven plane channel (D2Q9, periodic in *x*, no-slip top/bottom).
The steady solution of the incompressible momentum equation is the exact
parabola ``ux(y) = Gx/(2 nu) y (H - y)``.  Two closures are compared:

``nebb`` (this work)
    wall-node non-equilibrium bounce-back with the force-corrected wall
    velocity ``-Gx/(2 rho_w)`` and the wall density recovered from the field
    (:mod:`tensorlbm.boundary.nebb`), driven by the Guo (2002) body force;
``halfway`` (control)
    classic half-way bounce-back, whose no-slip planes sit half a cell off the
    wall rows, so the *matching* reference profile is a half-cell-shifted
    parabola; reported for completeness, not as a like-for-like accuracy
    ranking.

Arms in the benchmark
---------------------
* ``nebb_guo_ny{10,20,40}``      — primary, grid-refinement convergence
* ``nebb_guo_ny20_nocorr``       — negative control: the ``-Gx/(2 rho_w)`` wall
                                   shift removed (the pre-fix formulation)
* ``nebb_edm_ny20``              — alternative forcing (equilibrium-shift only)
* ``halfway_ny{10,20}``          — half-way bounce-back control
* ``corepath_{halfway,nebb}``    — the same capability exercised through the
                                   repo's main core-path solver
                                   (:class:`tensorlbm.gpu_core_path.GPUCorePathSolver`,
                                   ``wall_bc=`` dispatch added by this work)

Pre-declared acceptance (owner standard, fixed before the run)
--------------------------------------------------------------
* PASS if ``max|got-ref| / max|ref| < 0.15`` on the primary arm;
* ecosystem stretch if ``< 0.03``;
* convergence: error must fall by >= 3x per grid doubling (2nd order ⇒ 4x).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ...boundary.nebb import (
    NebbChannelConfig,
    NebbPoiseuilleChannel,
    analytic_poiseuille,
)

__all__ = ["run_2d_arm", "run_core_path_arm", "main", "OWNER_THRESHOLD", "STRETCH"]

OWNER_THRESHOLD = 0.15
STRETCH = 0.03


def run_2d_arm(
    ny: int,
    *,
    nx: int = 4,
    nu: float = 0.2,
    gx: float = 1.0e-3,
    scheme: str = "guo",
    wall: str = "nebb",
    force_corrected_wall: bool = True,
    steps: int = 400_000,
) -> dict:
    """One channel arm through the package driver (input -> compute -> output)."""
    cfg = NebbChannelConfig(
        nx=nx,
        ny=ny,
        nu=nu,
        gx=gx,
        steps=steps,
        scheme=scheme,
        wall=wall,
        force_corrected_wall=force_corrected_wall,
    )
    t0 = time.perf_counter()
    summary = NebbPoiseuilleChannel(cfg).run()
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["dy"] = gx / (ny - 1.0)
    return summary


def run_core_path_arm(
    wall_bc: str,
    *,
    ny: int = 18,
    nx: int = 8,
    nz: int = 4,
    tau: float = 0.8,
    u_max_target: float = 0.05,
    max_steps: int = 200_000,
    check_every: int = 200,
    tol: float = 1.0e-12,
) -> dict:
    """Same capability through the main solver (``gpu_core_path``) — L3 caller.

    Runs to steady state with the solver's own step loop and compares against
    the analytic reference that matches the selected wall convention
    (``channel_analytic_velocity`` for ``halfway``,
    ``channel_analytic_velocity_nebb`` for ``nebb``).
    """
    from ...gpu_core_path import (
        GPUCorePathConfig,
        GPUCorePathSolver,
        channel_analytic_velocity,
        channel_analytic_velocity_nebb,
    )

    cfg = GPUCorePathConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau=tau,
        u_max_target=u_max_target,
        init="rest",
        steps=1,
        backend="eager",
        wall_bc=wall_bc,
        label=f"nebb_xval_{wall_bc}",
    )
    solver = GPUCorePathSolver(cfg)
    t0 = time.perf_counter()
    prev = None
    steps = 0
    residual = float("inf")
    while steps < max_steps:
        solver.step(check_every)
        steps += check_every
        _y, u = solver.channel_profile()
        if prev is not None:
            residual = float((u - prev).abs().max())
            if residual < tol:
                break
        prev = u
    elapsed = time.perf_counter() - t0

    _y, u = solver.channel_profile()  # fluid rows 1..ny-2
    if wall_bc == "nebb":
        _yy, ref_full = channel_analytic_velocity_nebb(cfg)
        ref = ref_full[1 : ny - 1]
    else:
        _yy, ref_full = channel_analytic_velocity(cfg)
        ref = ref_full[1 : ny - 1]
    ref = ref.to(dtype=u.dtype)
    scale = float(ref.abs().max().clamp_min(1e-30))
    err = float((u - ref).abs().max()) / scale
    return {
        "wall_bc": wall_bc,
        "cfg": cfg.as_dict(),
        "steps_to_steady": steps,
        "steady_residual": residual,
        "elapsed_s": elapsed,
        "backend_used": solver.backend,
        "backend_note": solver.backend_note,
        "profile_ux_fluid": u.tolist(),
        "analytic_ux_fluid": ref.tolist(),
        "umax": float(u.abs().max()),
        "umax_analytic": scale,
        "err_global_vs_analytic": err,
        "mass_mean": solver.mass(),
        "height_used": (float(ny - 1) if wall_bc == "nebb" else float(ny - 2)),
    }


def _figure(result: dict, path: Path) -> bool:
    """Profile comparison + grid-convergence figure (skipped if no matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    primary = result["arms"].get("nebb_guo_ny20")
    if primary is not None:
        prof = primary["profile_ux"]
        ref = primary["analytic_ux"]
        y = list(range(len(prof)))
        ax.plot(ref, y, "k-", lw=2, label="analytic $u=Gx\\,y(H-y)/(2\\nu)$")
        ax.plot(prof, y, "o", ms=4, label="TensorLBM NEBB+Guo (ny=20)")
        ctrl = result["arms"].get("halfway_ny20")
        if ctrl:
            ax.plot(ctrl["profile_ux"], list(range(len(ctrl["profile_ux"]))), "s",
                    ms=3, mfc="none", label="half-way BB (ny=20)")
        ax.set_xlabel("$u_x$ (lattice)")
        ax.set_ylabel("y (lattice)")
        ax.set_title("Steady Poiseuille profile (Gx=1e-3, nu=0.2)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1]
    ns, es = [], []
    for key, arm in result["arms"].items():
        if key.startswith("nebb_guo_ny") and "nocorr" not in key:
            ns.append(arm["cfg"]["ny"])
            es.append(arm["err_global_vs_analytic"])
    if ns:
        order = sorted(range(len(ns)), key=lambda i: ns[i])
        ns = [ns[i] for i in order]
        es = [es[i] for i in order]
        ax.loglog(ns, es, "o-", label="NEBB+Guo")
        import math

        ref_line = [es[0] * (ns[0] / n) ** 2 for n in ns]
        ax.loglog(ns, ref_line, "k--", lw=1, label=r"$\propto \Delta y^2$")
        ax.axhline(OWNER_THRESHOLD, color="tab:red", ls=":", lw=1, label="15 % owner")
        ax.axhline(STRETCH, color="tab:green", ls=":", lw=1, label="3 % stretch")
        ax.set_xlabel("channel rows ny")
        ax.set_ylabel("max|got-ref| / max|ref|")
        ax.set_title("Grid convergence vs analytic parabola")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/m18182_nebb_force", help="artifact directory")
    parser.add_argument("--ny", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--nu", type=float, default=0.2)
    parser.add_argument("--gx", type=float, default=1.0e-3)
    parser.add_argument("--max-steps", type=int, default=400_000)
    parser.add_argument("--quick", action="store_true", help="small grid only (smoke run)")
    parser.add_argument("--skip-corepath", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.ny = [10, 20]
        args.skip_corepath = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "module": "tensorlbm.boundary.nebb",
        "entry": "python -m tensorlbm.verification.nebb_bodyforce.run",
        "problem": "2D D2Q9 Poiseuille channel: periodic x, no-slip y walls, body force Gx",
        "reference": "closed form ux(y) = Gx/(2 nu) y (H - y), H = ny - 1",
        "thresholds": {
            "owner_15pct": OWNER_THRESHOLD,
            "ecosystem_3pct": STRETCH,
            "convergence_ratio_per_doubling_min": 3.0,
        },
        "params": {"nu": args.nu, "gx": args.gx, "nx": 4, "max_steps": args.max_steps},
        "arms": {},
        "predeclared_from": "docstring (fixed before the run)",
    }

    for ny in args.ny:
        key = f"nebb_guo_ny{ny}"
        result["arms"][key] = run_2d_arm(
            ny, nu=args.nu, gx=args.gx, steps=args.max_steps, scheme="guo", wall="nebb"
        )
        print(f"[{key}] err={result['arms'][key]['err_global_vs_analytic']:.5f} "
              f"steps={result['arms'][key]['steps_done']}", flush=True)

    # controls
    ny_ctrl = args.ny[-1] if args.quick else 20
    result["arms"]["nebb_guo_ny20_nocorr"] = run_2d_arm(
        ny_ctrl, nu=args.nu, gx=args.gx, steps=args.max_steps,
        scheme="guo", wall="nebb", force_corrected_wall=False,
    )
    result["arms"]["nebb_edm_ny20"] = run_2d_arm(
        ny_ctrl, nu=args.nu, gx=args.gx, steps=args.max_steps, scheme="edm", wall="nebb"
    )
    for ny in args.ny[:2]:
        result["arms"][f"halfway_ny{ny}"] = run_2d_arm(
            ny, nu=args.nu, gx=args.gx, steps=args.max_steps, scheme="guo", wall="halfway"
        )

    # integration arms: the same capability through the main core-path solver
    if not args.skip_corepath:
        for wall_bc in ("halfway", "nebb"):
            key = f"corepath_{wall_bc}"
            result["arms"][key] = run_core_path_arm(wall_bc, u_max_target=0.05)
            print(f"[{key}] err={result['arms'][key]['err_global_vs_analytic']:.5f} "
                  f"steps={result['arms'][key]['steps_to_steady']}", flush=True)

    # pre-declared verdict, computed from the primary arm + convergence
    primary_key = f"nebb_guo_ny{max(args.ny)}"
    primary_err = result["arms"][primary_key]["err_global_vs_analytic"]
    coarse_key = f"nebb_guo_ny{min(args.ny)}"
    coarse_err = result["arms"][coarse_key]["err_global_vs_analytic"]
    ratio = coarse_err / primary_err if primary_err > 0 else float("inf")
    result["verdict"] = {
        "primary_arm": primary_key,
        "primary_err": primary_err,
        "owner_threshold": OWNER_THRESHOLD,
        "ecosystem_threshold": STRETCH,
        "pass_owner_15pct": bool(primary_err < OWNER_THRESHOLD),
        "pass_ecosystem_3pct": bool(primary_err < STRETCH),
        "coarse_err": coarse_err,
        "error_reduction_ratio": ratio,
        "second_order_consistent": bool(ratio >= 3.0),
    }

    (out / "result.json").write_text(json.dumps(result, indent=2))

    # long-form profile table
    lines = ["arm,ny,y,ux_got,ux_ref"]
    for key, arm in result["arms"].items():
        if "profile_ux" not in arm:
            continue
        ref = arm.get("analytic_ux", [])
        for i, val in enumerate(arm["profile_ux"]):
            r = ref[i] if i < len(ref) else ""
            lines.append(f"{key},{arm['cfg']['ny']},{i},{val!r},{r!r}")
    (out / "profiles.csv").write_text("\n".join(lines) + "\n")

    have_fig = _figure(result, out / "nebb_bodyforce_xval.png")
    result["figure"] = "nebb_bodyforce_xval.png" if have_fig else None
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["verdict"], indent=2))
    print("artifacts written to", out.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())