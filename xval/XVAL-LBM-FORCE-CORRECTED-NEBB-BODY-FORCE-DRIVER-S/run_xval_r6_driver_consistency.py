"""Round 6: driving-normalisation and velocity-diagnostic consistency of the
force-corrected NEBB / body-force driver.

Anchor: ``XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S``.

Round 5 closed every level check but left one pre-declared item failing:

    D3Q19 grid-convergence rate = 1.63 / 1.25   (target 1.8 - 2.2)

and it left a residual error of ~3e-3 on *both* D3Q19 wall closures at ny=18,
even though the plane-Poiseuille steady state is an exact quadratic of the
BGK lattice (Holdych, Noble & Georgiadis 1998, J. Comput. Phys. 145, 406: the
Poiseuille velocity field carries no truncation error).  An error that is not
truncation must be a *definition* error.  This harness measures the two
candidates without assuming either:

A. **Driving normalisation.**  ``GPUCorePathConfig.body_force`` derives the
   acceleration from ``height``.  ``height`` is the half-way-bounce-back width
   ``ny - 2``, but the ``nebb`` closure puts the no-slip planes *on* the wall
   rows, so its width is ``ny - 1``.  Measured quantity::

       ratio = u_max_measured / u_max_target
       prediction if the drive ignores the closure: ((ny-1)/(ny-2))**2

B. **Velocity vs. momentum definition.**  The collision evaluates the
   equilibrium at ``u* = u_raw + F/(2 rho)`` (Guo, Zheng & Shi 2002), i.e. the
   *physical* velocity is the raw moment shifted by ``F/(2 rho) = a/2``.  The
   public ``macroscopic()`` / ``channel_profile()`` return the raw moment, so
   the reported velocity is low by ``a/2``, which is ``4 nu / H**2`` relative
   to ``u_max`` (1.25 % at ny=10, 0.31 % at ny=18, 0.08 % at ny=34).  Measured
   quantity::

       deficit_ratio = (u_phys - u_raw) / (a/2)      # 1 => raw moment reported
                                                     # 0 => already physical

   ``u_phys`` is reconstructed here *outside* the solver
   (``u_raw + a/(2 rho)``), so this measurement is valid on both the pre-fix
   and post-fix tree and does not presuppose the fix.

C. **Order of accuracy**, two protocols:
   * ``legacy``  - fixed ``u_max_target``, refine ``ny`` (Round 4's protocol).
   * ``proper``  - refine with ``u_max_target ~ 1/H`` at fixed ``tau``, so every
     error term (truncation, compressibility, half-force) shrinks together.

D2Q9 arm (``tensorlbm.nebb_bodyforce.BodyForceChannel``) is carried alongside as
the cross-implementation control: it already reports the corrected velocity
(``ChannelResult.ux_corrected``), so its ratio and deficit are the reference
answer the D3Q19 core path must reproduce.

Usage
-----
    python run_xval_r6_driver_consistency.py --out <dir> [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from tensorlbm.gpu_core_path import (
    GPUCorePathConfig,
    GPUCorePathSolver,
    _macroscopic,
    channel_analytic_velocity,
    channel_analytic_velocity_nebb,
)

TOL_STEADY = 1e-12
CHECK_EVERY = 200


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rel_err_global(got: np.ndarray, ref: np.ndarray) -> float:
    """max|got-ref| / max|ref| over the compared rows (R3/R5 convention)."""
    scale = max(float(np.abs(ref).max()), 1e-300)
    return float(np.abs(np.asarray(got) - np.asarray(ref)).max() / scale)


def rel_err_pointwise(got: np.ndarray, ref: np.ndarray) -> float:
    """max|got-ref| / |ref| per row, rows with |ref| < 0.2 max dropped.

    Pointwise (R3's stricter metric); rows near the wall where the parabola
    vanishes are excluded from the denominator, not from the numerator.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    keep = np.abs(ref) > 0.2 * np.abs(ref).max()
    if not keep.any():  # pragma: no cover - degenerate
        return float("nan")
    return float((np.abs(got - ref)[keep] / np.abs(ref)[keep]).max())


def run_core_path(ny: int, wall_bc: str, *, u_max: float = 0.05, tau: float = 1.1,
                  nx: int = 8, nz: int = 4, max_steps: int = 200_000) -> dict:
    """Run the main-solver channel to steady state and record both moments."""
    cfg = GPUCorePathConfig(
        nx=nx, ny=ny, nz=nz, tau=tau, u_max_target=u_max, init="rest",
        precision="FP64FP64", backend="eager", steps=max_steps,
        wall_bc=wall_bc, label=f"r6_{wall_bc}_ny{ny}",
    )
    cfg.validate()
    solver = GPUCorePathSolver(cfg)

    # own steady-state loop so the residual is recorded (the built-in run()
    # does not expose it for this solver)
    prev, residual, steps_used = None, float("nan"), max_steps
    for k in range(max_steps):
        solver.step()
        if (k + 1) % CHECK_EVERY == 0:
            _, u_now = solver.channel_profile()
            cur = u_now.numpy().copy()
            if prev is not None:
                residual = float(np.abs(cur - prev).max())
                if residual < TOL_STEADY:
                    steps_used = k + 1
                    break
            prev = cur

    with __import__("torch").no_grad():
        f = solver._current()
        rho, ux_raw, _, _ = _macroscopic(f, solver.C)
    a = cfg.effective_body_force
    # physical velocity of this collision variant: equilibrium sees
    # u* = u_raw + a/2 (uniform acceleration => F/(2 rho) == a/2 exactly)
    ux_phys = (ux_raw + 0.5 * a).clone()

    rows = slice(1, ny - 1)          # fluid rows used by channel_profile()
    y = np.arange(1, ny - 1, dtype=np.float64)
    u_raw_prof = ux_raw.mean(dim=(0, 2))[rows].numpy().copy()
    u_phys_prof = ux_phys.mean(dim=(0, 2))[rows].numpy().copy()

    if wall_bc == "nebb":
        _, ref_full = channel_analytic_velocity_nebb(cfg)
        ref = ref_full[rows].numpy().copy()
        closure_width = float(ny - 1)
    else:
        _, ref_full = channel_analytic_velocity(cfg)
        ref = ref_full[rows].numpy().copy()
        closure_width = float(ny - 2)

    # the solver's own public profile must agree with the raw moment
    _, u_public = solver.channel_profile()
    public_matches_raw = float(np.abs(u_public.numpy() - u_raw_prof).max())

    a2 = 0.5 * a
    return {
        "arm": "corepath",
        "wall_bc": wall_bc,
        "ny": ny, "ny_fluid_rows": int(rows.stop - rows.start),
        "nx": nx, "nz": nz,
        "tau": cfg.tau, "nu": cfg.nu, "u_max_target": u_max,
        "body_force": a,
        "height_config": cfg.height,
        "closure_width": closure_width,
        "height_matches_closure": bool(abs(cfg.height - closure_width) < 1e-12),
        "steps_used": steps_used,
        "steady_residual": residual,
        "u_max_raw": float(u_raw_prof.max()),
        "u_max_phys": float(u_phys_prof.max()),
        "ratio_raw_to_target": float(u_raw_prof.max() / u_max),
        "ratio_phys_to_target": float(u_phys_prof.max() / u_max),
        "ratio_predicted_width_mismatch": float((closure_width / cfg.height) ** 2),
        "a_over_2": float(a2),
        "deficit_raw_vs_phys": float(u_phys_prof.max() - u_raw_prof.max()),
        "deficit_ratio": float((u_phys_prof.max() - u_raw_prof.max()) / a2) if a2 else float("nan"),
        "deficit_ratio_predicted": float(4.0 * cfg.nu / closure_width ** 2 * u_max / a2) if a2 else float("nan"),
        "err_raw_vs_analytic": rel_err_global(u_raw_prof, ref),
        "err_phys_vs_analytic": rel_err_global(u_phys_prof, ref),
        "err_phys_pointwise": rel_err_pointwise(u_phys_prof, ref),
        "err_raw_pointwise": rel_err_pointwise(u_raw_prof, ref),
        "public_profile_equals_raw": public_matches_raw,
        "profile_phys": u_phys_prof.tolist(),
        "profile_raw": u_raw_prof.tolist(),
        "profile_ref": ref.tolist(),
        "y": y.tolist(),
    }


def run_d2q9(ny: int, *, nu: float = 0.2, u_max: float = 0.05,
             scheme: str = "guo", max_steps: int = 200_000) -> dict:
    """Cross-implementation control: the in-repo D2Q9 NEBB channel."""
    from tensorlbm.nebb_bodyforce import BodyForceChannel, ChannelConfig, analytic_poiseuille

    H = ny - 1
    yc = 0.5 * H
    gx = 2.0 * nu * u_max / (yc * (H - yc))     # centre-line == u_max by construction
    cfg = ChannelConfig(nx=8, ny=ny, nu=nu, gx=gx, scheme=scheme,
                        wall="force_corrected_nebb", init="force_shifted",
                        steps=max_steps, steady_tol=1e-14, check_every=200)
    res = BodyForceChannel(cfg).run()
    ref = analytic_poiseuille(ny, gx, nu, 0.0)
    prof_raw = res.ux.mean(axis=0)
    prof_phys = res.ux_corrected.mean(axis=0)
    a2 = 0.5 * gx
    return {
        "arm": "d2q9_nebb_module",
        "wall_bc": "force_corrected_nebb", "scheme": scheme,
        "ny": ny, "tau": cfg.tau, "nu": nu, "gx": gx, "u_max_target": u_max,
        "steps_used": res.steps_used, "steady_residual": res.steady_residual,
        "u_max_raw": float(prof_raw.max()), "u_max_phys": float(prof_phys.max()),
        "ratio_raw_to_target": float(prof_raw.max() / u_max),
        "ratio_phys_to_target": float(prof_phys.max() / u_max),
        "deficit_raw_vs_phys": float(prof_phys.max() - prof_raw.max()),
        "deficit_ratio": float((prof_phys.max() - prof_raw.max()) / a2),
        "deficit_ratio_predicted": float(4.0 * nu / H ** 2 * u_max / a2),
        "err_raw_vs_analytic": rel_err_global(prof_raw, ref),
        "err_phys_vs_analytic": rel_err_global(prof_phys, ref),
        "profile_phys": prof_phys.tolist(), "profile_raw": prof_raw.tolist(),
        "profile_ref": ref.tolist(), "y": list(range(ny)),
    }


def order_study(protocol: str, wall_bcs=("halfway", "nebb"), nys=(10, 20),
                tau: float = 1.1, u_max_ref: float = 0.05, u_max_ny_ref: int = 20,
                nx: int = 4, nz: int = 2, max_steps: int = 200_000) -> list[dict]:
    """Measured order of accuracy of the physical velocity.

    ``legacy``: fixed ``u_max_target`` (Round 4 protocol).
    ``proper``: ``u_max_target = u_max_ref * H_ref/H`` at fixed ``tau``.
    """
    out = []
    for wbc in wall_bcs:
        for ny in nys:
            H = float(ny - 1 if wbc == "nebb" else ny - 2)
            H_ref = float(u_max_ny_ref - 1 if wbc == "nebb" else u_max_ny_ref - 2)
            umax = u_max_ref if protocol == "legacy" else u_max_ref * H_ref / H
            row = run_core_path(ny, wbc, u_max=umax, tau=tau, nx=nx, nz=nz,
                                max_steps=max_steps)
            row["protocol"] = protocol
            row["H"] = H
            out.append(row)
    # rates within each wall_bc
    for wbc in wall_bcs:
        series = sorted([r for r in out if r["wall_bc"] == wbc], key=lambda r: r["ny"])
        for a, b in zip(series, series[1:]):
            for key in ("err_phys_vs_analytic", "err_raw_vs_analytic"):
                e0, e1 = a[key], b[key]
                rate = (math.log(e0 / e1) / math.log(a["H"] / b["H"])
                        if e0 > 0 and e1 > 0 else float("nan"))
                b[f"rate_{key}_{a['ny']}to{b['ny']}"] = float(rate)
            b[f"rate_ny_{a['ny']}to{b['ny']}"] = a["ny"]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "artifacts", "r6"))
    ap.add_argument("--quick", action="store_true", help="small grids / short budgets")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    nys = (10, 18) if args.quick else (10, 18, 26)
    payload: dict = {"quick": bool(args.quick), "tol_steady": TOL_STEADY}

    # ---- A/B: driving normalisation + velocity definition -------------------
    drives = []
    for wbc in ("halfway", "nebb"):
        for ny in nys:
            row = run_core_path(ny, wbc, max_steps=40_000 if args.quick else 200_000)
            drives.append(row)
            print(f"[drive] {wbc:8s} ny={ny:3d} H_cfg={row['height_config']:5.1f} "
                  f"H_closure={row['closure_width']:5.1f} "
                  f"ratio_phys={row['ratio_phys_to_target']:.6f} "
                  f"ratio_raw={row['ratio_raw_to_target']:.6f} "
                  f"deficit/a2={row['deficit_ratio']:.4f} "
                  f"err_phys={row['err_phys_vs_analytic']:.3e} "
                  f"err_raw={row['err_raw_vs_analytic']:.3e}", flush=True)
    payload["drives"] = drives

    d2 = []
    for ny in (10, 18):
        row = run_d2q9(ny, max_steps=40_000 if args.quick else 200_000)
        d2.append(row)
        print(f"[d2q9 ] ny={ny:3d} ratio_phys={row['ratio_phys_to_target']:.6f} "
              f"deficit/a2={row['deficit_ratio']:.4f} "
              f"err_phys={row['err_phys_vs_analytic']:.3e} "
              f"err_raw={row['err_raw_vs_analytic']:.3e}", flush=True)
    payload["d2q9"] = d2

    # ---- C: order of accuracy ---------------------------------------------
    orders = {}
    for protocol in ("legacy", "proper"):
        rows = order_study(protocol, nys=(10, 20) if args.quick else (10, 20, 40),
                           max_steps=40_000 if args.quick else 200_000)
        orders[protocol] = rows
        for r in rows:
            print(f"[order:{protocol:6s}] {r['wall_bc']:8s} ny={r['ny']:3d} "
                  f"u_max_t={r['u_max_target']:.5f} err_phys={r['err_phys_vs_analytic']:.3e} "
                  f"err_raw={r['err_raw_vs_analytic']:.3e}", flush=True)
    payload["orders"] = orders

    path = os.path.join(args.out, "r6_driver_consistency.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())