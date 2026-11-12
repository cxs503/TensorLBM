#!/usr/bin/env python3
"""Curved-boundary (BFL) sphere drag ladder — production driver.

Anchor ``GAP-LBM-SPHERE``, mission m16085.  Frozen protocol and criteria:
``prereg.md`` in this directory (written before the runs).

Physical case
-------------
One sphere of **exact geometric radius** ``R`` (implicit surface, not a voxel
mask) is held fixed at the centre of a periodic cube of side ``L``.  A uniform
body force density ``g`` drives the flow; ``g`` is adjusted by a drag
feed-forward controller so that the mean fluid velocity equals ``u_char``.
The no-slip surface is imposed with the repository's D3Q19 BFL interpolated
bounce-back (:func:`tensorlbm.interpolated_bc.compute_q_sphere` +
:func:`tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19`), which places the wall at
``q`` fractional distance from the boundary node — i.e. at exactly ``R``.

Force bookkeeping
-----------------
Two independent estimators, both written to the case record:

1. **Global momentum balance.**  At steady state the periodic cell stores no
   momentum, so everything the Guo source injects is removed by the wall.
   This library's source is ``(1 - 1/(2 tau)) w (3 c.F + 9 (c.u)(c.F))``, whose
   first moment is ``(1 - 1/(2 tau)) F`` exactly (the ``(c.u)(c.F)`` term has
   zero first moment by lattice symmetry), hence

       ``F_mb = (1 - 1/(2 tau)) * g_tail * V_fluid``

   ``V_fluid`` counts an SDF cell as fluid iff its centre is outside the
   sphere.  ``F_mb`` is the verdict quantity.
2. **BFL momentum exchange** — the ``return_force`` ledger of
   ``bouzidi_bounce_back_d3q19``.  Consistency check only (criterion S2).

Drag coefficient and Reynolds number use the **geometric** radius

    ``C_D = 2 F / (rho u^2 pi R^2)``,   ``Re = rho u (2 R) / mu``,

with both the interstitial fluid mean velocity (``u_fluid``, what the sphere
sees) and the superficial cell velocity ``u_sup = u_fluid (1 - phi_solid)``
(the convention of the Hasimoto array reference used by criterion S1) written
out explicitly — the two differ by ~2 % at ``lambda = 1/6``, which is not
negligible against a 3 % criterion and must not be silently picked.

Nothing in this file reads a drag correlation: the closure only picks which
``Re`` to simulate and supplies the initial guess for ``g``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19  # noqa: E402
from tensorlbm.d3q19 import C, W  # noqa: E402
from tensorlbm.drag_closure import cd_schiller_naumann  # noqa: E402
from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

CS2 = 1.0 / 3.0
CS4 = 1.0 / 9.0

#: Relaxation of the drag feed-forward controller (per sample, see
#: :func:`_controller_force`).
CTRL_RELAX = 0.15

#: Steady gate: relative drift of ``u`` and ``g`` between the two halves of the
#: tail window must be below this.
STEADY_TOL = 1.0e-2


def hasimoto_K(lambda_: float) -> float:
    """Hasimoto (1959) drag enhancement of a sphere in a simple-cubic array.

    ``F / F_Stokes = (1 - 2.837 lam + 4.19 lam^3 - 27.4 lam^5)^-1`` with
    ``lam = a / L_cell`` (a = sphere radius).  Written in volume fraction
    ``phi = 4 pi lam^3 / 3`` the leading term is the familiar
    ``1.7601 phi^(1/3)`` (identity used as a unit test).
    """
    lam = float(lambda_)
    den = 1.0 - 2.837 * lam + 4.19 * lam**3 - 27.4 * lam**5
    if den <= 0.0:
        raise ValueError(f"Hasimoto series out of range at lambda={lam}")
    return 1.0 / den


def analytic_drag_force(u: float, R: float, nu: float, lam: float) -> float:
    """Absolute analytic drag on the sphere at the *measured* velocity.

    ``F = 0.5 rho u^2 pi R^2 C_D(u)`` with ``rho = 1`` in lattice units and

    * ``C_D = 24/Re * K(lam)`` (Stokes + Hasimoto confinement) for ``Re < 1``,
    * ``C_D = 24/Re (1 + 0.15 Re^0.687)`` (Schiller-Naumann) for ``Re >= 1``.

    This is the third, *correlation-based* force instrument.  It is a
    diagnostic only: it is never fed back into the controller (a controller
    that pins ``u`` to a value it derives from a correlation cannot test that
    correlation).  It is what discriminates between the two bookkeeping
    instruments ``F_mb`` (source momentum) and ``F_bfl`` (link ledger) when
    they disagree: whichever of the two agrees with the analytic drag is the
    one that is measuring the wall force.
    """
    if not (u > 0.0):
        return 0.0
    re = u * 2.0 * R / nu
    if re < 1.0:
        cd = (24.0 / re) * hasimoto_K(lam) if re > 0.0 else float("inf")
    else:
        cd = cd_schiller_naumann(re)
    return 0.5 * 1.0 * u * u * math.pi * R * R * cd


def _sphere_sdf_cells(L: int, cy: float, R: float) -> torch.Tensor:
    """Boolean solid mask: cell centres strictly inside the sphere."""
    idx = torch.arange(L, dtype=torch.float64) + 0.5
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    c = 0.5 * L
    return ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= R * R


def _fused_bgk_guo(f, tau, g, cvec, wvec):
    """One BGK collision + Guo (2002) source for a +x force density ``g``.

    Returns ``(f_post, rho, ux, uy, uz)``.  The source is the exact form used by
    the certified staircase study (``benchmarks/pending/sphere_drag_closure``)
    so the two geometries stay comparable: ``(1 - 1/(2 tau)) w (3 c.F + 9 (c.u)(c.F))``.
    """
    rho = f.sum(dim=0)
    inv_rho = 1.0 / rho
    ux = (f * cvec[:, 0]).sum(dim=0) * inv_rho
    uy = (f * cvec[:, 1]).sum(dim=0) * inv_rho
    uz = (f * cvec[:, 2]).sum(dim=0) * inv_rho
    cu = cvec[:, 0] * ux + cvec[:, 1] * uy + cvec[:, 2] * uz
    usq = ux * ux + uy * uy + uz * uz
    feq = wvec * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    src = (1.0 - 0.5 / tau) * wvec * (cvec[:, 0] * g) * (
        1.0 / CS2 + cu / CS4
    )
    return f - (f - feq) / tau + src, rho, ux, uy, uz


def _controller_force(g_old: float, g_ff: float, u_target: float, u_filt: float) -> float:
    """Drag feed-forward + scale-invariant velocity trim.

    ``g_ff`` comes from the measured wall force (the drag the sphere actually
    exerts), so the box inertia is never in the loop; the velocity trim is a
    *multiplicative* correction on the ratio ``u_target/u_filt`` — the drag
    scales as ``u^2``, so a ratio-based correction keeps the same gain over the
    whole ``Re`` ladder, unlike an additive trim whose gain has to be retuned
    per case (the staircase study's ``1e-2`` additive gain is 15 % of ``g`` at
    ``L = 36``: that is what makes its controller hunt).
    """
    if not (g_ff > 0.0) or not (u_filt > 0.0):
        return g_old
    ratio = max(0.5, min(2.0, u_target / u_filt))
    return g_ff * math.sqrt(ratio)


def _drift(vals) -> float:
    """Relative drift between the halves of a tail window (0 = flat)."""
    n = len(vals)
    if n < 4:
        return float("inf")
    h = n // 2
    a = sum(vals[:h]) / h
    b = sum(vals[h:]) / (n - h)
    if abs(b) < 1e-300:
        return float("inf")
    return abs(b - a) / abs(b)


def run_case(spec: dict) -> dict:
    """Run one sphere drag case and return its record (JSON-serialisable)."""
    torch.set_num_threads(int(spec.get("threads", 1)))
    tag = spec["tag"]
    L = int(spec["L"])
    R = float(spec["R"])
    Re = float(spec["Re"])
    u_target = float(spec.get("u", 0.05))
    steps = int(spec.get("steps", 8000))
    sample_every = int(spec.get("sample_every", 20))
    scheme = spec.get("scheme", "curved")
    zero_force = bool(spec.get("zero_force", False))
    tail = int(spec.get("tail", 50))
    #: ``bfl``      -- legacy: feed-forward from the BFL link ledger (the
    #:                 instrument whose absolute accuracy is under test here);
    #: ``momentum`` -- feed-forward from the analytic drag at ``u_filt``, i.e.
    #:                 no measured force in the control loop (prereg amendment
    #:                 A1, docs/GAP-LBM-SPHERE_v3_converged_control_verdict.md);
    #: ``fixed``    -- prescribed body-force density ``g``, starved of feedback.
    controller = str(spec.get("controller", "bfl"))
    if controller not in {"bfl", "momentum", "fixed"}:
        raise ValueError(f"unknown controller {controller!r}")

    dev = torch.device("cpu")
    cvec = C.to(dev).float().view(19, 3, 1, 1, 1)
    wvec = W.to(dev).float().view(19, 1, 1, 1)

    solid = _sphere_sdf_cells(L, L / 2.0, R).to(dev)
    fluid = ~solid
    n_solid = int(solid.sum())
    v_fluid = float(fluid.sum())
    phi_solid = n_solid / float(L**3)

    if scheme != "curved":
        raise ValueError("this driver implements the curved (BFL) scheme only")

    mask, q = compute_q_sphere(L, L, L, L / 2.0, L / 2.0, L / 2.0, R, dev)
    n_links = int(mask.sum())

    D = 2.0 * R
    nu = u_target * D / Re
    tau = 0.5 + 3.0 * nu
    force_factor = 1.0 - 0.5 / tau

    # initial force density guess: unbounded Schiller-Naumann drag on the
    # geometric sphere, spread over the fluid volume
    cd_guess = cd_schiller_naumann(Re) if not zero_force else 0.0
    f_guess = 0.5 * 1.0 * u_target * u_target * math.pi * R * R * cd_guess
    g = 0.0 if zero_force else f_guess / (force_factor * v_fluid)
    if controller == "fixed" and not zero_force:
        g = float(spec["g_fixed"])
    g_fixed = g

    rho0 = torch.ones((L, L, L), dtype=torch.float32)
    uz0 = torch.zeros_like(rho0)
    ux0 = torch.full_like(rho0, u_target)
    # equilibrium at rest-fluid + uniform stream
    cu0 = cvec[:, 0] * ux0
    f = wvec * rho0 * (1.0 + 3.0 * cu0 + 4.5 * cu0 * cu0 - 1.5 * ux0 * ux0)
    f = f + 0.0 * torch.zeros_like(f)
    del uz0

    f_rest = wvec.expand(19, L, L, L).contiguous()
    g_ff = g
    u_filt = u_target
    u_hist: list[float] = []
    g_hist: list[float] = []
    bfl_hist: list[float] = []
    f_bfl_last = 0.0
    f_bfl_ema = None
    mem_every = int(spec.get("mem_every", 100))
    diverged = False
    t0 = time.time()
    step_i = 0
    for step_i in range(1, steps + 1):
        f_new, rho, ux, uy, uz = _fused_bgk_guo(f, tau, g, cvec, wvec)
        # The solid interior carries no fluid.  It is reset to the *rest*
        # equilibrium every step rather than frozen: freezing it (the
        # convention of benchmarks/pending/sphere_drag_closure) lets the
        # interior accumulate the momentum that streams in from the fluid and
        # never relax it, which is a momentum sink that does not appear in the
        # wall ledger -- see docs/GAP-LBM-SPHERE_curved_boundary_verification.md
        # §5.1.  With the reset, everything the source injects leaves through
        # the BFL wall links, so estimator 1 measures the wall force.
        f_new[:, solid] = f_rest[:, solid]
        u_mean = float(ux[fluid].mean())
        u_filt += 0.02 * (u_mean - u_filt)
        if not zero_force and controller == "momentum":
            # exact momentum-balance feed-forward: the force the analytic
            # correlation says the sphere exerts, no measured F in the loop
            g_ff = analytic_drag_force(u_filt, R, nu, R / L) / (force_factor * v_fluid)
            g = _controller_force(g, g_ff, u_target, u_filt)
        elif not zero_force and controller == "bfl":
            g = _controller_force(g, g_ff, u_target, u_filt)
        f_pre = f_new
        f_stream = stream3d(f_pre)
        f, force = bouzidi_bounce_back_d3q19(
            f_stream, f_pre, mask, q, return_force=True
        )
        if step_i % sample_every == 0:
            u_hist.append(u_mean)
            g_hist.append(g)
            if step_i % mem_every == 0:
                f_bfl_last = float(force[0])
                bfl_hist.append(f_bfl_last)
                if not zero_force and controller == "bfl":
                    f_bfl_ema = abs(f_bfl_last) if f_bfl_ema is None else (
                        (1.0 - CTRL_RELAX) * f_bfl_ema + CTRL_RELAX * abs(f_bfl_last)
                    )
                    g_ff = f_bfl_ema / (force_factor * v_fluid)
        if step_i % 200 == 0 and not bool(torch.isfinite(f).all()):
            diverged = True
            break
        if step_i % max(steps // 6, 1) == 0:
            print(
                f"[{tag}] step={step_i}/{steps} U={u_mean:.6f} g={g:.5e} "
                f"F_bfl={f_bfl_last:.5e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    wall_s = time.time() - t0

    u_tail = u_hist[-tail:]
    g_tail = g_hist[-tail:]
    bfl_tail = bfl_hist[-min(len(bfl_hist), max(tail // 5, 3)) :] if bfl_hist else []
    u_meas = sum(u_tail) / len(u_tail) if u_tail else float("nan")
    g_meas = sum(g_tail) / len(g_tail) if g_tail else float("nan")
    f_bfl = sum(bfl_tail) / len(bfl_tail) if bfl_tail else float("nan")

    f_mb = force_factor * g_meas * v_fluid
    u_sup = u_meas * (1.0 - phi_solid)
    # dynamic viscosity: in lattice units rho = 1, so mu = rho * nu = nu.
    # (m16085 used ``nu / 3.0`` here, which reported Re_measured 3x too high --
    #  0.2944 instead of 0.0981 for the stokes_Re0.1_L36 case.  See
    #  docs/GAP-LBM-SPHERE_v2_root_cause_and_fix.md.)
    mu = nu

    def cd(u_ref: float, force: float) -> float:
        if not (u_ref > 0.0) or not math.isfinite(force):
            return float("nan")
        return 2.0 * force / (1.0 * u_ref * u_ref * math.pi * R * R)

    rec = {
        "tag": tag,
        "scheme": scheme,
        "L": L,
        "R": R,
        "D": D,
        "lambda": D / L,
        "phi_solid": phi_solid,
        "Re_target": Re,
        "nu": nu,
        "tau": tau,
        "u_target": u_target,
        "u_measured": u_meas,
        "u_superficial": u_sup,
        "Re_measured": u_meas * D / mu,
        "Re_superficial": u_sup * D / mu,
        "steps": step_i,
        "wall_s": wall_s,
        "diverged": diverged,
        "zero_force": zero_force,
        "n_solid_cells": n_solid,
        "n_bfl_links": n_links,
        "v_fluid": v_fluid,
        "force_factor": force_factor,
        "g_tail": g_meas,
        "F_momentum_balance": f_mb,
        "F_bfl": f_bfl,
        "bfl_rel_diff_pct": (
            100.0 * (f_bfl - f_mb) / f_mb if f_mb not in (0.0,) and math.isfinite(f_bfl) else float("nan")
        ),
        "controller": controller,
        "F_analytic": analytic_drag_force(u_meas, R, nu, R / L),
        "analytic_vs_mb_pct": (
            100.0 * (f_mb - analytic_drag_force(u_meas, R, nu, R / L)) / analytic_drag_force(u_meas, R, nu, R / L)
            if analytic_drag_force(u_meas, R, nu, R / L) > 0.0
            else float("nan")
        ),
        "analytic_vs_bfl_pct": (
            100.0 * (f_bfl - analytic_drag_force(u_meas, R, nu, R / L)) / analytic_drag_force(u_meas, R, nu, R / L)
            if analytic_drag_force(u_meas, R, nu, R / L) > 0.0
            else float("nan")
        ),
        "cd_mb_fluid": cd(u_meas, f_mb),
        "cd_mb_sup": cd(u_sup, f_mb),
        "cd_bfl_fluid": cd(u_meas, f_bfl),
        "u_tail_drift": _drift(u_tail),
        "g_tail_drift": _drift(g_tail),
        "steady": bool(
            math.isfinite(_drift(u_tail))
            and math.isfinite(_drift(g_tail))
            and _drift(u_tail) < STEADY_TOL
            and (zero_force or _drift(g_tail) < STEADY_TOL)
            and not diverged
        ),
        "hasimoto_K": hasimoto_K(R / L),
        # mean free path of the kinetic model vs the body radius: the BGK
        # viscosity (tau - 1/2)/3 is only the hydrodynamic (k -> 0) value, and
        # it is *not* what the flow sees once c_s*tau is comparable to R
        "mfp_over_R": (tau / math.sqrt(3.0)) / R,
        "hydrodynamic_regime": bool((tau / math.sqrt(3.0)) / R < 0.5),
        "cd_stokes_confined": (
            24.0 / Re * hasimoto_K(R / L) if not zero_force else float("nan")
        ),
        "cd_schiller_naumann": cd_schiller_naumann(Re) if not zero_force else float("nan"),
        "u_hist_tail": [round(v, 8) for v in u_tail],
        "g_hist_tail": [float(f"{v:.10e}") for v in g_tail],
        "protocol": "prereg.md §4",
        "torch": torch.__version__,
        "nproc": os.cpu_count(),
    }
    return rec


def _stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic D3Q19 streaming via roll (the driver's own copy so the case
    records are independent of solver3d internals)."""
    out = torch.empty_like(f)
    for i in range(f.shape[0]):
        dx = int(C[i, 0])
        dy = int(C[i, 1])
        dz = int(C[i, 2])
        if dx == 0 and dy == 0 and dz == 0:
            out[i] = f[i]
        else:
            out[i] = torch.roll(f[i], shifts=(dz, dy, dx), dims=(0, 1, 2))
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--R", type=float, default=6.0)
    ap.add_argument("--Re", type=float, required=True)
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--zero-force", action="store_true")
    ap.add_argument(
        "--controller",
        default="bfl",
        choices=("bfl", "momentum", "fixed"),
        help="bfl=legacy ledger feed-forward; momentum=analytic-drag feed-forward; "
        "fixed=prescribed --g (no feedback)",
    )
    ap.add_argument("--g", type=float, default=None, help="body-force density for --controller fixed")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    spec = {
        "tag": args.tag,
        "L": args.L,
        "R": args.R,
        "Re": args.Re,
        "u": args.u,
        "steps": args.steps,
        "threads": args.threads,
        "zero_force": args.zero_force,
        "controller": args.controller,
    }
    if args.controller == "fixed":
        if args.g is None:
            ap.error("--controller fixed requires --g")
        spec["g_fixed"] = args.g
    rec = run_case(spec)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"{args.tag}.json"
    p.write_text(json.dumps(rec, indent=1))
    print(
        f"[{args.tag}] cd_mb_sup={rec['cd_mb_sup']:.4f} cd_mb_fluid={rec['cd_mb_fluid']:.4f} "
        f"steady={rec['steady']} bfl_diff={rec['bfl_rel_diff_pct']:.2f}% "
        f"analytic_vs_mb={rec['analytic_vs_mb_pct']:.2f}% "
        f"analytic_vs_bfl={rec['analytic_vs_bfl_pct']:.2f}% "
        f"wall={rec['wall_s']:.0f}s -> {p}",
        flush=True,
    )
    rec.setdefault("sha256_self", sha256_file(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())