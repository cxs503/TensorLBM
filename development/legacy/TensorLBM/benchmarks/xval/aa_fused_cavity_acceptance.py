"""Acceptance sweep for the AA-pattern fused stream-collide kernel.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM
``aa-pattern-fused-stream-collide``).  Acceptance: ``< 15 %`` against a
reference (owner precision standard).

Why this script exists
----------------------
``docs/xval_aa_pattern_fused_stream_collide_ordering.md`` fixed the operator
ordering and the wall treatment of ``tensorlbm.aa_fused_stream_collide`` but
left item 1 of its "remaining work" open: *the cavity ablation never finished,
so the ``< 15 %`` acceptance of this anchor was never demonstrated*.  The old
``L2_u = 0.247`` / ``L2_v = 0.239`` numbers of the 2026-11-13 report came from a
kernel with (a) the wrong operator ordering ``C[S[f]]`` and (b) a mass-leaking
post-stream bounce-back, so they are neither a pass nor a fail for the fixed
kernel.  This driver produces the missing number.

Three gates, in order of logical dependence
-------------------------------------------
``couette``   **Exact-solution gate for the moving wall.**  Halfway bounce-back
              with a moving wall injects momentum with a sign that a
              self-consistent "fused vs materialised" test cannot see (both
              share the code path).  Plane Couette flow has the exact solution
              ``u_x(y) = U (y - 1/2) / L``; a wrong sign cannot reproduce it.
              Nothing downstream is meaningful unless this passes.
``equality``  Fused kernel vs the canonical materialised step inside the cavity
              configuration (bit-equality, ``torch.equal``), so the accepted
              number is shown to come from a kernel that *is* collide-then-stream.
``cavity``    Lid-driven cavity, Re = 100, U = 0.1, L = 32 / 64 / 128 / 192,
              run to a monitored steady state, center-line profiles vs Ghia et
              al. (1982), ``L2`` normalised by ``U``.

Metric conventions (they matter, see ``notes`` in the output)
------------------------------------------------------------
* ``u`` is reported in lattice units and normalised by ``U`` *before* being
  compared with Ghia: ``sqrt(mean((u/U - u_ghia)^2))``.  (The audit driver
  ``aa_fused_stream_collide_ordering_audit.py::_l2_profile`` divided *after*
  subtracting, which inflated every error by ``1/U``; fixed there too.)
* ``*_full`` uses all Ghia sample points, including ``y = 0`` and ``y = 1``,
  where a cell-centred solver has no data and must extrapolate from the
  half-cell node.  That is a genuine discretisation error but it is also a
  *floor*: an exact profile fed back through the metric still scores
  ``floor_*`` (reported per grid).  ``*_interior`` excludes the two wall
  endpoints and isolates the interior accuracy.
* Mass and finiteness are evaluated on **fluid nodes only**.  Populations on
  solid nodes are never used as a source by the halfway bounce-back, so they
  carry no physical meaning and can diverge harmlessly;
  ``test_solid_node_junk_does_not_leak_into_fluid`` pins that containment.

Usage
-----
``python benchmarks/xval/aa_fused_cavity_acceptance.py --task-index 3``
runs one task (the scheduler runs one task per shard); ``--task-index -1`` runs
the two cheap gates.

Writes ``TensorLBM/artifacts/xval_aa_pattern/cavity_accept_<idx>.json`` (plus
``.npz`` velocity fields for the contour figure when ``L <= 128``).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

# --------------------------------------------------------------------------
# Thread discipline (2026-11-15, m22295)
# --------------------------------------------------------------------------
# One step of this kernel is 9 x (L+2)^2 float64 *and* ~60 torch kernel
# launches, i.e. it is launch-overhead bound, not flop bound.  On a host whose
# default intra-op pool is sized to the core count (128 here) the pool
# hand-off dominates: measured with ``aa_fused_step`` at L = 64 on one and the
# same idle-ish process,
#
#     torch threads | ms / step
#     --------------+-----------
#         128       |   26.1
#          16       |    6.5
#           8       |    6.2
#           4       |    5.9
#           2       |    5.9
#           1       |    7.0
#
# i.e. pinning costs 4.4x of wall clock over the default.  Sweeps submitted
# without this pin were observed to sit in ``running`` for > 6 h for a task
# that needs minutes, which is how the ``< 15 %`` acceptance of this anchor
# stayed unpaid.  Keep it small (1..4) and overridable.
_THREADS = max(1, int(os.environ.get("TLBM_THREADS", "2")))
torch.set_num_threads(_THREADS)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from aa_fused_stream_collide_ordering_audit import (  # noqa: E402
    GHIA_U,
    GHIA_V,
    _cavity_field,
    _cavity_geometry,
    _moments,
    _sample_profiles,
)
from tensorlbm.aa_fused_stream_collide import (  # noqa: E402
    W as W_TENSOR,
    aa_fused_step,
    separate_step,
)

torch.set_default_dtype(torch.float64)

W9 = [float(x) for x in W_TENSOR]

ART = os.path.abspath(os.path.join(_HERE, "..", "..", "artifacts", "xval_aa_pattern"))


# --------------------------------------------------------------------------
# metric (normalise BEFORE subtracting -- see module docstring)
# --------------------------------------------------------------------------
def profile_error(numeric_lattice, y_star, ref, U: float,
                  interior: bool = False) -> float:
    """``L2`` of ``u/U`` against a normalised reference table.

    Parameters
    ----------
    numeric_lattice : (L,) array
        Centre-line velocity in *lattice* units on the cell-centred points
        ``y_star``.
    ref : (n, 2) array
        ``[y, u/U]`` reference table (Ghia).
    interior : bool
        Drop the ``y = 0`` and ``y = 1`` reference points (extrapolated).
    """
    num = np.asarray(numeric_lattice, dtype=float) / U
    ys = np.asarray(y_star, dtype=float)
    vals = np.interp(ref[:, 0], ys, num)
    vals = np.where(ref[:, 0] >= ys[-1], num[-1], vals)
    sel = (ref[:, 0] > 0.0) & (ref[:, 0] < 1.0) if interior else np.ones(len(ref), bool)
    return float(np.sqrt(np.mean((vals[sel] - ref[sel, 1]) ** 2)))


def metric_floor(L: int, U: float = 0.1) -> dict:
    """What the metric reports for a *perfect* solution on an ``L`` grid.

    Feeding the Ghia profile interpolated to the cell centres back into the
    metric leaves only the extrapolation to ``y = 0, 1`` -- the floor below
    which no solver can score.
    """
    ys = (np.arange(L) + 0.5) / L
    out = {}
    for name, ref in (("u", GHIA_U), ("v", GHIA_V)):
        exact = np.interp(ys, ref[:, 0], ref[:, 1]) * U
        out[f"floor_{name}"] = profile_error(exact, ys, ref, U)
        out[f"floor_{name}_interior"] = profile_error(exact, ys, ref, U, interior=True)
    return out


def _fluid_mass(f: torch.Tensor, fluid: torch.Tensor) -> float:
    return float(f[:, fluid].sum())


# --------------------------------------------------------------------------
# Task 0: exact-solution gate for the moving wall (plane Couette flow)
# --------------------------------------------------------------------------
def _couette_geometry(L: int) -> tuple:
    """Two infinite planar walls: ``L`` fluid rows, one solid row each side, one
    node in the streamwise direction (periodic, so the unroll is trivial).

    Row 0 (bottom) is at rest, row ``L + 1`` (top) moves with ``u_w = (U, 0)``;
    the wall *faces* sit halfway between the solid row and the first fluid row,
    i.e. at ``y = 1/2`` and ``y = L + 1/2``, so ``H = L``.
    """
    n = L + 2
    solid = torch.zeros((n, 1), dtype=torch.bool)
    solid[0, 0] = True
    solid[-1, 0] = True
    f = torch.zeros((9, n, 1))
    f[0] = 4.0 / 9.0
    for i in range(1, 9):
        f[i] = W9[i]
    return n, solid, f


def task_couette(L: int = 32, U: float = 0.05, tau: float = 0.8,
                 steps: int = 12000) -> dict:
    """Plane Couette flow between a stationary and a moving planar wall.

    Exact steady state ``u_x(y) = U (y - 1/2) / L``; fluid row ``j = 1..L`` sits
    at ``y_j = j``, so ``u_x(j) = U (j - 1/2) / L``.

    This is the gate for the *moving-wall* bounce-back: the momentum sign of
    ``+6 w_i rho (c_i . u_w)`` is invisible to a fused-vs-materialised
    comparison (both share the code path) but a flipped sign cannot reproduce
    the exact linear profile.
    """
    n, solid, f = _couette_geometry(L)
    fluid = ~solid
    m0 = _fluid_mass(f, fluid)
    for _ in range(steps):
        f = aa_fused_step(f, tau, solid, u_lid=U, lid_row=-1)
    _, ux, _ = _moments(f)
    u_num = ux[1:L + 1, 0].numpy()
    u_ana = U * (np.arange(1, L + 1) - 0.5) / L
    mass = _fluid_mass(f, fluid)
    return {
        "task": "couette",
        "L": L, "U": U, "tau": tau, "nu": (tau - 0.5) / 3.0, "steps": steps,
        "rel_err_max": float(np.abs(u_num - u_ana).max() / U),
        "rel_err_l2": float(np.sqrt(np.mean((u_num - u_ana) ** 2)) / U),
        "u_top_numeric": float(u_num[-1]),
        "u_top_analytic": float(u_ana[-1]),
        "u_mid_numeric": float(u_num[L // 2]),
        "u_mid_analytic": float(u_ana[L // 2]),
        "profile_numeric": u_num.tolist(),
        "profile_analytic": u_ana.tolist(),
        "fluid_mass_initial": m0,
        "fluid_mass_final": mass,
        "fluid_mass_drift_rel": float(abs(mass - m0) / m0),
        "gate_1pct": bool(np.abs(u_num - u_ana).max() / U < 1e-2),
    }


# --------------------------------------------------------------------------
# Task 1: fused vs materialised canonical step inside the cavity setup
# --------------------------------------------------------------------------
def task_equality(L: int = 32, U: float = 0.1, Re: float = 100.0,
                  steps: int = 20000) -> dict:
    tau = 0.5 + 3.0 * (U * L / Re)
    n, solid, _ = _cavity_geometry(L, U)
    f_fused = _cavity_field(n)
    f_sep = f_fused.clone()
    max_diff = 0.0
    first_diff = None
    for s in range(1, steps + 1):
        f_fused = aa_fused_step(f_fused, tau, solid, u_lid=U, lid_row=-1)
        f_sep = separate_step(f_sep, tau, solid, u_lid=U, lid_row=-1)
        if s % 1000 == 0 or s == 1:
            d = float((f_fused - f_sep).abs().max())
            max_diff = max(max_diff, d)
            if d != 0.0 and first_diff is None:
                first_diff = {"step": s, "max_abs_diff": d}
    return {
        "task": "equality_cavity",
        "L": L, "U": U, "Re": Re, "tau": tau, "steps": steps,
        "max_abs_diff": max_diff,
        "bit_equal": bool(max_diff == 0.0),
        "first_nonzero_diff": first_diff,
    }


# --------------------------------------------------------------------------
# Task 2..5: lid-driven cavity vs Ghia et al. (1982), Re = 100
# --------------------------------------------------------------------------
def task_cavity(L: int, steps: int, U: float = 0.1, Re: float = 100.0,
                monitor: int = 1000, rtol: float = 1e-9, min_steps: int = 20000,
                ramp: int = 2000, save_field: bool = True) -> dict:
    tau = 0.5 + 3.0 * (U * L / Re)
    nu = (tau - 0.5) / 3.0
    n, solid, _ = _cavity_geometry(L, U)
    wt = (n - L) // 2
    fluid = ~solid
    f = _cavity_field(n)
    mass0 = _fluid_mass(f, fluid)
    hist = []
    prev = None
    t0 = time.time()
    converged_at = None
    stable_hits = 0
    fluid_finite = True
    s = 0

    for s in range(1, steps + 1):
        u_lid = U * min(1.0, s / ramp) if ramp else U
        f = aa_fused_step(f, tau, solid, u_lid=u_lid, lid_row=-1)
        if s % monitor == 0:
            if not bool(torch.isfinite(f[:, fluid]).all()):
                fluid_finite = False
                hist.append({"step": s, "nonfinite": True})
                break
            _, ux, uy = _moments(f)
            u = np.stack([ux.numpy(), uy.numpy()])
            res = float("nan") if prev is None else float(np.abs(u - prev).max() / U)
            uxc, uyc, ys = _sample_profiles(f, L, wt)
            hist.append({
                "step": s, "residual": res,
                "L2_u_full": profile_error(uxc, ys, GHIA_U, U),
                "L2_v_full": profile_error(uyc, ys, GHIA_V, U),
                "fluid_mass_rel_drift": abs(_fluid_mass(f, fluid) - mass0) / mass0,
                "rho_min": float(f.sum(0)[fluid].min()),
                "rho_max": float(f.sum(0)[fluid].max()),
            })
            prev = u
            if s >= min_steps and np.isfinite(res) and res < rtol:
                stable_hits += 1
                if stable_hits >= 2:
                    converged_at = s
                    break
            else:
                stable_hits = 0

    elapsed = time.time() - t0
    out = {
        "task": "cavity", "L": L, "U": U, "Re": Re, "tau": tau, "nu": nu,
        "steps_planned": steps, "steps_run": s, "fluid_finite": fluid_finite,
        "converged_at": converged_at, "monitor": monitor, "rtol": rtol,
        "ramp_steps": ramp, "elapsed_s": elapsed,
        "metric_floor": metric_floor(L, U),
        "history": hist,
    }
    if fluid_finite:
        uxc, uyc, ys = _sample_profiles(f, L, wt)
        rho, ux, uy = _moments(f)
        out.update({
            "L2_u_full": profile_error(uxc, ys, GHIA_U, U),
            "L2_v_full": profile_error(uyc, ys, GHIA_V, U),
            "L2_u_interior": profile_error(uxc, ys, GHIA_U, U, interior=True),
            "L2_v_interior": profile_error(uyc, ys, GHIA_V, U, interior=True),
            "Linf_u_full": float(max(
                abs(np.interp(GHIA_U[:, 0], ys, uxc / U)[:-1] - GHIA_U[:-1, 1]).max(),
                abs(uxc[-1] / U - GHIA_U[-1, 1]))),
            "umax_over_U": float(np.abs(ux[fluid]).max() / U),
            "fluid_mass_rel_drift": abs(_fluid_mass(f, fluid) - mass0) / mass0,
            "rho_fluid_range": [float(rho[fluid].min()), float(rho[fluid].max())],
            "profiles": {"y_star": ys.tolist(), "ux": uxc.tolist(),
                         "uy": uyc.tolist()},
            "gate_15pct": bool(max(profile_error(uxc, ys, GHIA_U, U),
                                   profile_error(uyc, ys, GHIA_V, U)) < 0.15),
            "gate_3pct": bool(max(profile_error(uxc, ys, GHIA_U, U),
                                  profile_error(uyc, ys, GHIA_V, U)) < 0.03),
        })
        if save_field and L <= 128:
            np.savez_compressed(
                os.path.join(ART, f"cavity_accept_field_L{L}.npz"),
                ux=ux.numpy(), uy=uy.numpy(), L=L, wt=wt, U=U)
    return out


TASKS = {
    0: ("couette", lambda: task_couette()),
    1: ("equality_cavity_L32", lambda: task_equality()),
    2: ("cavity_L32", lambda: task_cavity(32, 120_000)),
    3: ("cavity_L64", lambda: task_cavity(64, 150_000)),
    4: ("cavity_L128", lambda: task_cavity(128, 200_000)),
    5: ("cavity_L192", lambda: task_cavity(192, 250_000, save_field=False)),
}


def run_task(idx: int) -> dict:
    name, fn = TASKS[idx]
    r = fn()
    r["task_index"] = idx
    r["task_name"] = name
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, f"cavity_accept_{idx}.json")
    with open(path, "w") as fh:
        json.dump(r, fh, indent=2)
    brief = {k: r[k] for k in ("task_name", "steps_run", "converged_at",
                               "L2_u_full", "L2_v_full", "L2_u_interior",
                               "rel_err_max", "bit_equal") if k in r}
    print("wrote", path, json.dumps(brief))
    return r


def main() -> dict:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=-1,
                    help="-1 = cheap gates (0 and 1) locally; otherwise one task")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        return {i: run_task(i) for i in TASKS}
    if a.task_index < 0:
        return {i: run_task(i) for i in (0, 1)}
    return run_task(a.task_index)


if __name__ == "__main__":
    main()