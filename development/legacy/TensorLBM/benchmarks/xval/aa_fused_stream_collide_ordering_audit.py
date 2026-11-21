"""XVAL audit: is ``tensorlbm.aa_fused_stream_collide`` the AA-pattern scheme?

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM fused
stream-collide).  Acceptance: <15 % against experiment/reference.

Motivation (adversarial re-read of the 2026-11-13 XVAL report)
-------------------------------------------------------------
``docs/xval_aa_pattern_fused_stream_collide.md`` claims

  "For periodic BCs the AA-pattern and separate approach produce bit-identical
   results (max diff 5.55e-17)"

and records an honest FAIL for the lid-driven cavity at n=33
(``L2_u = 0.2472`` vs Ghia, versus ``0.0957`` for the separate scheme), blamed on
grid coarseness.  Two things look wrong on inspection:

1. The equivalence test in ``tests/test_aa_fused_stream_collide.py`` starts
   from ``f = 0.1 * ones`` with ``f[0] = 0.5`` -- a **spatially uniform**
   state.  Under a uniform state every stencil (roll) is the identity, so the
   comparison cannot see an operator-ordering difference.  The 5.55e-17 is a
   tautology, not evidence.
2. ``aa_fused_stream_collide`` pulls **pre-collision** populations and then
   collides at the destination node.  The AA-pattern fused pull form is

       f_i(x, t+1) = f_i(x - c_i, t) + omega * [f_i^eq(x - c_i, t) - f_i(x - c_i, t)]

   i.e. the *collision increment is taken at the source node* (Kruger et al.
   2017 ch. 5; ``tensorlbm.aa_fused_push_pull.fused_push_pull_step``).  The
   module under audit instead computes ``C[S[f]]`` (stream, then collide),
   which does not commute with ``S[C[f]]`` because the BGK equilibrium is a
   nonlinear functional of the *local* moments.  This script measures how large
   that difference is in a physically observable quantity (effective viscosity).

Experiments
-----------
E1  vacuity of the published equivalence test (uniform vs non-uniform state)
E2  periodic equivalence against the canonical collide-then-stream operator
E3  analytic shear-wave viscous decay: nu_eff / nu - 1 for each ordering
E4  lid-driven cavity vs Ghia et al. (1982), Re=100, L=64 and L=128

Run:
    python benchmarks/xval/aa_fused_stream_collide_ordering_audit.py
Writes ``artifacts/xval_aa_pattern/ordering_audit.json``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.aa_fused_push_pull import (  # noqa: E402
    fused_push_pull_step,
    collide_stream_step,
    CX,
    CY,
    W9,
    OPP,
)
import tensorlbm.aa_fused_stream_collide as legacy_mod  # noqa: E402

torch.set_default_dtype(torch.float64)

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "xval_aa_pattern")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a - b).abs().max().item()
    s = max(b.abs().max().item(), 1e-300)
    return d / s


# --------------------------------------------------------------------------
# E1/E2: operator-ordering comparison on periodic domains
# --------------------------------------------------------------------------
def _uniform_state(n: int) -> torch.Tensor:
    f = torch.ones((9, n, n)) * 0.1
    f[0] = 0.5
    return f


def _nonuniform_state(n: int) -> torch.Tensor:
    """MAC-clean non-uniform state: u = U sin(2 pi y / n) e_x, rho = 1."""
    y = torch.arange(n, dtype=torch.float64).view(n, 1).expand(n, n)
    ux = 0.05 * torch.sin(2.0 * math.pi * y / n)
    uy = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    c = torch.tensor([(CX[i], CY[i]) for i in range(9)])
    w = torch.tensor(W9)
    cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    u2 = ux**2 + uy**2
    feq = w.view(-1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0))
    # add a non-equilibrium part so collision is not idle
    return feq + 1e-3 * (feq - feq.mean(0, keepdim=True))


def exp_e1_e2(n: int = 32, tau: float = 0.8) -> dict:
    out = {}
    for label, state in (("uniform", _uniform_state(n)), ("nonuniform", _nonuniform_state(n))):
        ref = state.clone()
        f_can = collide_stream_step(ref.clone(), tau)          # S[C[f]] (canonical)
        f_fused = fused_push_pull_step(ref.clone(), tau)       # AA fused pull
        f_leg = legacy_mod.stream_then_collide_legacy(ref.clone(), tau)  # C[S[f]] (audited)
        e = {
            "max_rel_diff_canonical_vs_separate_fns": _rel(
                f_can, legacy_mod.separate_stream_collide(ref.clone(), tau)),
            "max_rel_diff_aa_fused_vs_canonical": _rel(f_fused, f_can),
            "max_rel_diff_audited_vs_canonical": _rel(f_leg, f_can),
            "max_rel_diff_audited_vs_aa_fused": _rel(f_leg, f_fused),
        }
        out[label] = e
    # multi-step karma: how fast do the two orderings diverge from a physical IC?
    f_a = _nonuniform_state(n)
    f_b = f_a.clone()
    trace = []
    for step in range(1, 201):
        f_a = fused_push_pull_step(f_a, tau)
        f_b = legacy_mod.stream_then_collide_legacy(f_b, tau)
        if step in (1, 2, 5, 10, 20, 50, 100, 200):
            trace.append({"step": step, "max_rel_diff": _rel(f_b, f_a)})
    out["divergence_trace_nonuniform"] = trace
    return out


# --------------------------------------------------------------------------
# E3: analytic shear-wave decay (exact solution of the incompressible NSE)
# --------------------------------------------------------------------------
def _shear_ic(ny: int, nx: int, U: float) -> torch.Tensor:
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1).expand(ny, nx)
    ux = U * torch.sin(2.0 * math.pi * y / ny)
    uy = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    c = torch.tensor([(CX[i], CY[i]) for i in range(9)])
    w = torch.tensor(W9)
    cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    u2 = ux**2 + uy**2
    return w.view(-1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0))


def _fundamental(f: torch.Tensor) -> float:
    """Amplitude of the sin(2 pi y / ny) mode of the x-momentum signal."""
    rho, ux, _ = (lambda g: (g.sum(0),
                             (g * torch.tensor([CX[i] for i in range(9)]).view(-1, 1, 1)).sum(0)
                             / g.sum(0),
                             (g * torch.tensor([CY[i] for i in range(9)]).view(-1, 1, 1)).sum(0)
                             / g.sum(0)))(f)
    prof = ux[:, 0]  # one column; field is x-uniform
    ny = prof.numel()
    y = np.arange(ny)
    s = np.sin(2 * np.pi * y / ny)
    return float(2.0 / ny * np.dot(prof.numpy(), s))


def exp_e3(ny: int = 64, nx: int = 8, U: float = 0.02, nu_target: float = 0.05,
           steps: int = 4000, scheme: str = "fused") -> dict:
    tau = 0.5 + 3.0 * nu_target
    k = 2.0 * math.pi / ny
    f = _shear_ic(ny, nx, U)
    a0 = _fundamental(f)
    times, amps = [0], [a0]
    t0 = time.time()
    for s in range(1, steps + 1):
        if scheme == "fused":
            f = fused_push_pull_step(f, tau)
        elif scheme == "canonical":
            f = collide_stream_step(f, tau)
        elif scheme == "audited":
            f = legacy_mod.stream_then_collide_legacy(f, tau)
        else:
            raise ValueError(scheme)
        if s % 200 == 0:
            times.append(s)
            amps.append(_fundamental(f))
    elapsed = time.time() - t0
    t = np.array(times[1:], dtype=float)
    a = np.abs(np.array(amps[1:], dtype=float))
    keep = a > 1e-8
    slope, intercept = np.polyfit(t[keep], np.log(a[keep]), 1)
    nu_eff = -slope / k**2
    # amplitude error vs analytic at the final time
    t_end = float(times[-1])
    a_exact = a0 * math.exp(-nu_target * k**2 * t_end)
    return {
        "scheme": scheme,
        "ny": ny, "U": U, "nu_target": nu_target, "tau": tau, "steps": steps,
        "amplitude_initial": a0,
        "amplitude_final": float(amps[-1]),
        "amplitude_exact_final": a_exact,
        "amp_rel_error_final": float(abs(amps[-1] - a_exact) / a_exact),
        "nu_eff": float(nu_eff),
        "nu_rel_error": float(nu_eff / nu_target - 1.0),
        "elapsed_s": elapsed,
    }


# --------------------------------------------------------------------------
# E4: lid-driven cavity vs Ghia et al. (1982), Re = 100
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Reference table: SINGLE SOURCE OF TRUTH (defect fix, 2026-11-16)
# --------------------------------------------------------------------------
# This module used to carry a hand-embedded copy of Ghia's Table I/II.  The
# Table I copy (``u``) happened to be right; the Table II copy (``v``) was
# *scrambled*: values were attached to the wrong ``x/L`` stations, so the
# ``v``-profile metric could not fall below ~4.5 % on any grid (a
# grid-independent plateau that was mistaken for a solver error).
#
# The right half of the horizontal centre line is where the two copies part.
# BOUND table (this one):  v(0.8047) = -0.24533, v(0.8594) = -0.22445,
# v(0.9063) = -0.16914, v(0.9453) = -0.10313, v(0.9531) = -0.08864,
# v(0.9609) = -0.07391, v(0.9688) = -0.05906, v(1.0) = 0 -- monotone towards the
# wall, negative everywhere on the right half (the flow descends there).
# Retired copy: v(0.9531) = -0.03717, v(0.9609) = **+0.00730**,
# v(0.9688) = **+0.02526**, v(0.9766) = **+0.05248**, v(1.0) = 0 -- a sign
# change to positive one cell away from a solid wall, where v must approach
# zero from below.  A converged solver cannot match that, which is exactly the
# grid-independent ~4.5 % floor that was read as a solver defect.
#
# The BOUND table is ``tensorlbm.verification.lid_driven_cavity.reference_data.
# GHIA_RE100`` (provenance ``"BOUND"``: mechanically extracted OCR of Ghia et
# al. 1982, sha256 recorded in ``benchmarks/anchors/ghia1982_provenance.json``).
# Import it here instead of re-typing it; see
# ``tests/test_ghia_reference_single_source.py`` and
# ``docs/xval_aa_pattern_ghia_table_defect.md``.
_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

from tensorlbm.verification.lid_driven_cavity.reference_data import (  # noqa: E402
    GHIA_RE100 as _GHIA_RE100,
)


def _as_table(coord, value) -> np.ndarray:
    """``[coordinate, normalised velocity]`` as the rest of this module expects."""
    return np.column_stack([np.asarray(coord, dtype=float), np.asarray(value, dtype=float)])


GHIA_U = _as_table(_GHIA_RE100.y, _GHIA_RE100.u)   # Table I  (u at x/L = 1/2)
GHIA_V = _as_table(_GHIA_RE100.x, _GHIA_RE100.v)   # Table II (v at y/L = 1/2)

#: The scrambled Table II copy that used to live here, kept ONLY so that the
#: regression test can assert it is *not* what the metric compares against.
GHIA_V_RETIRED_SCRAMBLED = np.array([
    [0.0000, 0.00000], [0.0625, 0.09233], [0.0703, 0.10091], [0.0781, 0.10890],
    [0.0938, 0.12317], [0.1563, 0.16077], [0.2266, 0.17507], [0.2344, 0.17527],
    [0.5000, 0.05454], [0.7344, -0.24533], [0.8516, -0.18866], [0.9531, -0.03717],
    [0.9609, 0.00730], [0.9688, 0.02526], [0.9766, 0.05248], [1.0000, 0.00000],
])


def _stream_then_collide_step(f: torch.Tensor, tau: float, solid=None,
                              wall_u=None) -> torch.Tensor:
    """Stream-then-collide (``C o S``) with the *same* halfway bounce-back
    structure as the correct fused kernel -- used to isolate the operator
    ordering from the boundary treatment."""
    from tensorlbm.aa_fused_push_pull import _pull, macroscopic, _collision_increment
    rho_loc, _, _ = macroscopic(f)
    g = torch.empty_like(f)
    g[0] = f[0]
    for i in range(1, 9):
        g[i] = _pull(f[i], i)
    bb = {}
    if solid is not None:
        for i in range(1, 9):
            up = _pull(solid, i)
            if not bool(up.any()):
                continue
            j = OPP[i]
            val = f[j].clone()
            if wall_u is not None:
                val = val + 6.0 * W9[i] * rho_loc * (
                    CX[i] * _pull(wall_u[0], i) + CY[i] * _pull(wall_u[1], i))
            bb[i] = (up, val)
    rho, ux, uy = macroscopic(g)
    inc = _collision_increment(g, rho, ux, uy, tau)
    out = g + inc
    for i, (up, val) in bb.items():
        out[i] = torch.where(up, val, out[i])
    return out


def _legacy_bc(f: torch.Tensor, solid, U: float, lid_row: int = -1) -> torch.Tensor:
    """The boundary treatment of the audited module, applied to any step."""
    out = f.clone()
    for i in range(1, 9):
        opp = int(OPP[i])
        out[i, solid] = out[opp, solid]
    if U != 0.0:
        rho, _, _ = (lambda g: (g.sum(0),
                                (g * torch.tensor([CX[k] for k in range(9)]).view(-1, 1, 1)).sum(0) / g.sum(0),
                                (g * torch.tensor([CY[k] for k in range(9)]).view(-1, 1, 1)).sum(0) / g.sum(0)))(out)
        c = torch.tensor([(CX[k], CY[k]) for k in range(9)], dtype=out.dtype)
        w = torch.tensor(W9, dtype=out.dtype)
        feq = w.view(-1, 1, 1) * rho * (1 + 3 * c[:, 0].view(-1, 1, 1) * U
                                        + 4.5 * (c[:, 0].view(-1, 1, 1) * U) ** 2
                                        - 1.5 * U * U)
        for i in (3, 7, 8):
            out[i, lid_row] = feq[i, lid_row]
    return out


def _cavity_field(n_nodes: int):
    rho = torch.ones((n_nodes, n_nodes))
    z = torch.zeros_like(rho)
    c = torch.tensor([(CX[i], CY[i]) for i in range(9)], dtype=rho.dtype)
    w = torch.tensor(W9, dtype=rho.dtype)
    return w.view(-1, 1, 1) * rho * torch.ones((9, 1, 1))


def _cavity_geometry(L: int, U: float, wt: int = 1):
    n = L + 2 * wt
    solid = torch.zeros((n, n), dtype=torch.bool)
    solid[:wt, :] = solid[-wt:, :] = True
    solid[:, :wt] = solid[:, -wt:] = True
    wall_u = torch.zeros((2, n, n))
    wall_u[0, -wt:, :] = U
    return n, solid, wall_u


def _moments(f):
    c = torch.tensor([(CX[i], CY[i]) for i in range(9)], dtype=f.dtype)
    rho = f.sum(0)
    ux = (f * c[:, 0].view(-1, 1, 1)).sum(0) / rho
    uy = (f * c[:, 1].view(-1, 1, 1)).sum(0) / rho
    return rho, ux, uy


def _centre_line_weights(L: int) -> tuple:
    """Linear-interpolation weights that place the sampling line *exactly* on
    ``x/H = 1/2`` (and, by symmetry, ``y/H = 1/2``).

    Cell centres sit at ``(j + 1/2) / L``, so for an even ``L`` no node lies on
    the centre line: the old code took row/column ``L // 2`` of the padded array,
    which is one half cell *off* the centre line (``+/- 1/(2L)``).
    On the horizontal centre line that half-cell offset is *not* harmless -- near
    the top wall ``|dv/dy|`` is large, and the offset alone contributed
    ``~8e-3`` (in units of ``U``) at ``L = 128``, decreasing only as ``O(1/L)``.
    Returns ``(j, w)`` such that ``(1 - w) * phi[j] + w * phi[j + 1]`` is the
    centre-line value.
    """
    if L % 2 == 1:
        return L // 2, 0.0
    return L // 2 - 1, 0.5


def _sample_profiles(f, L, wt):
    """Centre-line profiles on the *exact* centre lines ``x/H = y/H = 1/2``.

    Returns ``(u_x(y), u_y(x), y_star)`` with ``y_star = (j + 1/2) / L`` the
    cell-centre stations of the profiles.
    """
    _, ux, uy = _moments(f)
    return _sample_profiles_from_moments(ux, uy, L, wt)


def _sample_profiles_from_moments(ux, uy, L, wt):
    """Same as :func:`_sample_profiles` but from an already known moment field
    (used by the offline metric re-computation, which stores ``u`` rather than
    the nine planes)."""
    blk_x = ux[wt:wt + L, wt:wt + L]
    blk_y = uy[wt:wt + L, wt:wt + L]
    j, w = _centre_line_weights(L)
    if w == 0.0:
        uxc = blk_x[:, j].numpy()
        uyc = blk_y[j, :].numpy()
    else:
        uxc = ((1.0 - w) * blk_x[:, j] + w * blk_x[:, j + 1]).numpy()
        uyc = ((1.0 - w) * blk_y[j, :] + w * blk_y[j + 1, :]).numpy()
    y_star = (np.arange(L) + 0.5) / L
    return uxc, uyc, y_star


def _l2_profile(numeric, y_star, ref, U):
    """``L2`` of ``u/U`` vs a normalised reference table (Ghia).

    FIXED 2026-11-15 (m21739): the previous version subtracted the normalised
    reference from the *dimensional* profile and divided the residual by ``U``
    afterwards, i.e. it reported ``sqrt(mean((u - u_ghia)^2))/U`` and so
    inflated every error by a factor ``~1/U`` (=10 for the lid velocity used
    here).  The normalisation must happen before the subtraction, otherwise a
    converged cavity scores ``L2 ~ 0.9`` instead of ``~0.09`` and the ``< 15 %``
    acceptance can never be met by *any* solver.
    """
    num = np.asarray(numeric, dtype=float) / U
    ys = np.asarray(y_star, dtype=float)
    vals = np.interp(ref[:, 0], ys, num)
    vals = np.where(ref[:, 0] >= ys[-1], num[-1], vals)
    return float(np.sqrt(np.mean((vals - ref[:, 1]) ** 2)))


ARMS = {
    # ordering, boundary treatment
    "fused+halfwayBB": ("fused", "halfway"),
    "streamThenCollide+halfwayBB": ("stc", "halfway"),
    "fused+legacyBC": ("fused", "legacy"),
    "audited(legacy module)": ("audited", "legacy"),
}


def exp_e4(L: int = 64, U: float = 0.1, Re: float = 100.0, steps: int = 20000,
           arm: str = "fused+halfwayBB", checkpoints=(1000, 5000, 20000)) -> dict:
    nu = U * L / Re
    tau = 0.5 + 3.0 * nu
    n, solid, wall_u = _cavity_geometry(L, U)
    wt = (n - L) // 2
    f = _cavity_field(n)
    order, bc = ARMS[arm]
    rec = {}
    diverged = False
    t0 = time.time()
    for s in range(1, steps + 1):
        if order == "fused":
            f = fused_push_pull_step(f, tau, solid=solid if bc == "halfway" else None,
                                     wall_u=wall_u if bc == "halfway" else None)
            if bc == "legacy":
                f = _legacy_bc(f, solid, U)
        elif order == "stc":
            f = _stream_then_collide_step(f, tau, solid=solid, wall_u=wall_u)
        else:
            f = legacy_mod.stream_then_collide_legacy(f, tau)
            f = _legacy_bc(f, solid, U)
        if s % 500 == 0 and not torch.isfinite(f).all():
            diverged = True
            break
        if s in checkpoints:
            uxc, uyc, ys = _sample_profiles(f, L, wt)
            rec[str(s)] = {"L2_u_vs_ghia": _l2_profile(uxc, ys, GHIA_U, U),
                           "L2_v_vs_ghia": _l2_profile(uyc, ys, GHIA_V, U),
                           "max_ux": float(np.abs(uxc).max())}
    elapsed = time.time() - t0
    out = {"arm": arm, "L": L, "U": U, "Re": Re, "tau": tau, "steps": steps,
           "diverged": diverged, "checkpoints": rec, "elapsed_s": elapsed}
    if not diverged:
        uxc, uyc, ys = _sample_profiles(f, L, wt)
        out["profiles"] = {"y_star": ys.tolist(), "ux_centerline": uxc.tolist(),
                           "uy_centerline": uyc.tolist()}
    return out


CAVITY_TASKS = [
    ("fused+halfwayBB", 64, 150000),
    ("streamThenCollide+halfwayBB", 64, 150000),
    ("fused+legacyBC", 64, 150000),
    ("audited(legacy module)", 64, 150000),
    ("fused+halfwayBB", 128, 200000),
]


def run_cavity_task(idx: int) -> dict:
    arm, L, steps = CAVITY_TASKS[idx]
    ck = tuple(int(steps * f) for f in (0.0833, 0.1667, 0.3333, 0.5, 0.75, 1.0))
    r = exp_e4(L=L, steps=steps, arm=arm, checkpoints=ck)
    r["task_index"] = idx
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, f"cavity_arm_{idx}.json")
    with open(path, "w") as fh:
        json.dump(r, fh, indent=2)
    print("wrote", path, "L2_u checkpoints:",
          {k: round(v["L2_u_vs_ghia"], 4) for k, v in r["checkpoints"].items()})
    return r


def main() -> dict:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-index", type=int, default=None,
                    help="run cavity task index (see CAVITY_TASKS) instead of the local suite")
    ap.add_argument("--skip-cavity", action="store_true")
    a = ap.parse_args()
    if a.arm_index is not None:
        return run_cavity_task(a.arm_index)
    t_all = time.time()
    res = {"anchor": "XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE",
           "note": "audit of tensorlbm.aa_fused_stream_collide ordering",
           "torch": torch.__version__}
    res["E1_E2_operator_ordering"] = exp_e1_e2()
    res["E3_shear_wave"] = [
        exp_e3(scheme="canonical"),
        exp_e3(scheme="fused"),
        exp_e3(scheme="audited"),
    ]
    if not a.skip_cavity:
        res["E4_cavity_L64"] = [exp_e4(L=64, steps=20000, arm=arm) for arm in ARMS]
    res["elapsed_s"] = time.time() - t_all
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, "ordering_audit.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2)[:6000])
    print("wrote", path)
    return res


if __name__ == "__main__":
    main()