#!/usr/bin/env python3
"""Acceptance gates for the 3-D load-exchange interface (GAP-LBM-T5-ICE-COUPLING, round 1).

Each gate compares an *independent* reference (analytic integral, closed-form
linear algebra, or Euler-Bernoulli beam theory) against the value produced by the
interface code.  Nothing is hand-copied: the JSON this script writes
(``runs/three_way_interface3d/gates.json``) is the only source of the numbers
quoted in ``docs/GAP-LBM-T5-ICE-COUPLING_round1.md`` and in the mission report.

Run::

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/three_way_interface3d_gates.py

Gates
-----
I1  closed-surface traction quadrature (∮n dA = 0) and open-hemisphere pressure
    resultant (p·πa²) with marker refinement
I2  Q1-hex consistent surface load: resultant + moment exact
I3  affine patch test: interface work = ∫t·u dA (transfer vs Gauss reference)
I4  DEM point load: consistent vs nearest-node control (moment conservation)
I5  added-mass stability: marching growth vs closed-form spectral radius,
    threshold bisection, and the relaxation bound 2/(1+μ)
I6  hull-girder leg: Hermite beam vs Euler-Bernoulli theory, loads delivered
    through the 3-D surface interface (two marker resolutions)
I7  interface work ledger: W = ΔE with second-order dt convergence
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.beam_fem import BeamProperties, EulerBernoulliBeamFEM  # noqa: E402
from tensorlbm.three_way.interface3d import (  # noqa: E402
    CouplingClock,
    HexGrid3D,
    InterfaceLedger,
    added_mass_ratio,
    explicit_staggered_stable,
    gauss_reference,
    march_one_dof,
    project_to_beam,
    rectangle_surface,
    relaxation_bound,
    sphere_surface,
    staggered_spectral_radius,
)

OUT = ROOT / "runs" / "three_way_interface3d"
FIG = OUT / "figures"
DTYPE = torch.float64


def rel(got: float, ref: float) -> float:
    return abs(got - ref) / max(abs(ref), 1e-300)


def order(e1: float, e2: float, ratio: float) -> float:
    if e1 <= 0 or e2 <= 0:
        return float("nan")
    return math.log(e1 / e2) / math.log(ratio)


# ---------------------------------------------------------------------------
# I1 — surface quadrature
# ---------------------------------------------------------------------------


def gate_i1() -> dict:
    a = 1.0
    closed = sphere_surface((0, 0, 0), a, n_theta=12, n_phi=24)
    closed_sum = float(torch.linalg.norm((closed.normals * closed.dA.view(-1, 1)).sum(0)))
    rows = []
    for n_th, n_ph in ((8, 16), (16, 32), (32, 64)):
        hemi = sphere_surface((0, 0, 0), a, n_theta=n_th, n_phi=n_ph, hemisphere=True)
        fz = float(hemi.pressure_force(1.0)[2])
        rows.append({"n_theta": n_th, "n_phi": n_ph, "markers": len(hemi),
                     "got": -fz, "ref": math.pi * a * a, "err": rel(fz, -math.pi * a * a),
                     "area_err": rel(hemi.area, 2 * math.pi * a * a)})
    for i in range(len(rows) - 1):
        rows[i]["convergence_order"] = order(rows[i]["err"], rows[i + 1]["err"], 2.0)
    return {
        "closed_surface_sum_ndA": closed_sum,
        "closed_surface_ref": 0.0,
        "closed_surface_rel": closed_sum / (4 * math.pi * a * a),
        "closed_surface_exact": closed_sum < 1e-12,
        "hemisphere_rows": rows,
        "pass": closed_sum < 1e-12 and rows[-1]["err"] < 1e-2 and rows[-1]["err"] < rows[0]["err"],
    }


# ---------------------------------------------------------------------------
# I2 — consistent surface load on a Q1 hex mesh
# ---------------------------------------------------------------------------


def gate_i2() -> dict:
    rows = []
    for n in (6, 12):
        grid = HexGrid3D((-1.5, -1.5, -1.5), (3.0, 3.0, 3.0), (n, n, n))
        rect = rectangle_surface((-0.5, -0.5, 0.25), (1, 0, 0), (0, 1, 0), 5, 5)
        t = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE).view(1, 3).repeat(len(rect), 1)
        nodal = grid.distribute_surface(rect, t)
        res = nodal.sum(0)
        mom = torch.cross(grid.nodes, nodal, dim=1).sum(0)
        ref_f = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE)  # t * A, A = 1
        rows.append({
            "n_cells": n,
            "resultant": res.tolist(),
            "resultant_rel": rel(float(res[2]), 2.0),
            "moment_norm": float(torch.linalg.norm(mom)),
            "moment_ref": 0.0,  # rectangle centred on the z axis -> moment free load
        })
    return {
        "rows": rows,
        "ref_force": ref_f.tolist(),
        "pass": all(r["resultant_rel"] < 1e-12 and r["moment_norm"] < 1e-12 for r in rows),
    }


# ---------------------------------------------------------------------------
# I3 — affine patch test (work conjugacy + moment)
# ---------------------------------------------------------------------------


def _affine_fields() -> tuple:
    t0 = torch.tensor([0.7, -1.3, 2.1], dtype=DTYPE)
    B = torch.tensor([[0.3, -0.2, 0.1], [0.05, 0.4, -0.15], [-0.25, 0.2, 0.6]], dtype=DTYPE)
    u0 = torch.tensor([0.11, -0.07, 0.23], dtype=DTYPE)
    omega = torch.tensor([0.4, -0.9, 0.6], dtype=DTYPE)
    x0 = torch.tensor([0.13, -0.21, 0.37], dtype=DTYPE)

    def traction(p: torch.Tensor) -> torch.Tensor:
        return t0.view(1, 3) + p @ B.T

    def velocity(p: torch.Tensor) -> torch.Tensor:
        return u0.view(1, 3) + torch.cross(omega.view(1, 3), p - x0.view(1, 3), dim=1)

    return traction, velocity


def gate_i3() -> dict:
    """Two distinct claims, tested separately (see module docstring I3).

    * **discrete work identity** — the transfer algebra is exact: summing the
      nodal forces against the *nodal* velocities reproduces the marker-surface
      integral ``Σ_m t_m·u(x_m) dA_m`` to machine precision, because ``u`` is
      affine and Q1 satisfies partition of unity + linear completeness.  This is
      the invariant the interface contract asserts.
    * **quadrature error** — the marker rule itself is a one-point rule per
      triangle, so it approximates the analytic surface integral of the
      *quadratic* work density with ``O(h²)``; reported against a 6×6
      Gauss-Legendre reference (not gated to machine precision, gated to order).
    """
    traction, velocity = _affine_fields()
    origin = (-0.5, -0.5, 0.25)
    e1, e2 = (1, 0, 0), (0, 1, 0)

    import numpy as np

    ref_w = 0.0
    gp, gw = np.polynomial.legendre.leggauss(6)
    gp = torch.tensor(gp, dtype=DTYPE)
    gw = torch.tensor(gw, dtype=DTYPE)
    o = torch.tensor(origin, dtype=DTYPE)
    a1 = torch.tensor(e1, dtype=DTYPE)
    a2 = torch.tensor(e2, dtype=DTYPE)
    jac = 0.25 * float(torch.linalg.norm(torch.cross(a1, a2, dim=0)))
    for xi, wi in zip(gp, gw):
        for eta, wj in zip(gp, gw):
            p = (o + a1 * (0.5 * (xi + 1)) + a2 * (0.5 * (eta + 1))).view(1, 3)
            ref_w += float(wi * wj) * float((traction(p) * velocity(p)).sum()) * jac

    rows = []
    for n_cells, n_mark in ((6, 4), (8, 8), (12, 16)):
        grid = HexGrid3D((-1.5, -1.5, -1.5), (3.0, 3.0, 3.0), (n_cells, n_cells, n_cells))
        rect = rectangle_surface(origin, e1, e2, n_mark, n_mark)
        trac = traction(rect.positions)
        u_mark = velocity(rect.positions)
        nodal = grid.distribute_surface(rect, trac)
        u_nodes = velocity(grid.nodes)
        w_nodal = float((nodal * u_nodes).sum())
        w_discrete = float(((trac * u_mark).sum(1) * rect.dA).sum())
        rows.append({
            "n_cells": n_cells, "n_markers": n_mark,
            "w_discrete_identity": w_nodal, "w_marker_integral": w_discrete,
            "identity_err": rel(w_nodal, w_discrete),
            "w_gauss_ref": ref_w, "quadrature_err": rel(w_nodal, ref_w),
        })
    for i in range(len(rows) - 1):
        rows[i]["quadrature_order"] = order(rows[i]["quadrature_err"],
                                            rows[i + 1]["quadrature_err"], 2.0)
    return {
        "rows": rows,
        "pass": all(r["identity_err"] < 1e-12 for r in rows)
                and rows[-1]["quadrature_err"] < 1e-3
                and rows[-2]["quadrature_order"] > 1.8,
    }


# ---------------------------------------------------------------------------
# I4 — DEM point load (consistent vs nearest-node control)
# ---------------------------------------------------------------------------


def gate_i4() -> dict:
    p = torch.tensor([0.13, -0.21, 0.37], dtype=DTYPE)
    f = torch.tensor([1.0, 2.0, -3.0], dtype=DTYPE)
    ref_m = torch.cross(p.view(1, 3), f.view(1, 3), dim=1)[0]
    rows = []
    for n in (6, 12, 24):
        grid = HexGrid3D((-1.5, -1.5, -1.5), (3.0, 3.0, 3.0), (n, n, n))
        cons = grid.distribute_point_consistent(p, f)
        lump = grid.distribute_point_lumped(p, f)
        rows.append({
            "n_cells": n,
            "h": 3.0 / n,
            "consistent_resultant_err": float(torch.linalg.norm(cons.sum(0) - f)) / float(torch.linalg.norm(f)),
            "consistent_moment_err": float(torch.linalg.norm(torch.cross(grid.nodes, cons, dim=1).sum(0) - ref_m)),
            "lumped_moment_err": float(torch.linalg.norm(torch.cross(grid.nodes, lump, dim=1).sum(0) - ref_m)),
        })
    for i in range(len(rows) - 1):
        rows[i]["lumped_order"] = order(rows[i]["lumped_moment_err"], rows[i + 1]["lumped_moment_err"], 2.0)
    return {
        "rows": rows,
        "consistent_moment_ref": 0.0,
        "pass": all(r["consistent_moment_err"] < 1e-12 and r["consistent_resultant_err"] < 1e-14
                    for r in rows),
    }


# ---------------------------------------------------------------------------
# I5 — added-mass stability
# ---------------------------------------------------------------------------


def _growth_fit(mu: float, n0: int = 300, n1: int = 800, omega_n_dt: float = 1e-2) -> float:
    """Least-squares slope of log‖state‖ over [n0, n1] -> growth per step.

    A least-squares slope (rather than an endpoint ratio) averages the O(1)
    oscillation of the undamped oscillator, so the measured number can be
    compared with the closed-form spectral radius.
    """
    hist = march_one_dof(mu, omega_n_dt=omega_n_dt, n_steps=n1)
    n = torch.arange(n0, n1 + 1, dtype=DTYPE)
    sub = hist[:, n0:]
    scale = sub.abs().amax(dim=0)                     # overflow-safe log-norm
    y = torch.log(scale) + torch.log(torch.linalg.norm(sub / scale, dim=0))
    nm = n - n.mean()
    slope = (nm * (y - y.mean())).sum() / (nm * nm).sum()
    return float(torch.exp(slope))


def gate_i5() -> dict:
    rows = []
    for mu in (0.5, 2.0):
        lam = staggered_spectral_radius(mu)
        growth = _growth_fit(mu)
        rows.append({
            "mu": mu,
            "lambda_closed_form": lam,
            "growth_measured": growth,
            "rel_err": rel(growth, lam),
            "stable_predicted": explicit_staggered_stable(mu),
            "observed_bounded": growth < 1.0 + 1e-3,
        })
    lo, hi = 0.5, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if staggered_spectral_radius(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    mu_crit = 0.5 * (lo + hi)
    relax_rows = []
    mu = 2.0
    for omega in (0.9 * relaxation_bound(mu), 1.1 * relaxation_bound(mu)):
        y = 1.0
        ys = [y]
        for _ in range(60):
            y = y + omega * (-mu * y - y)
            ys.append(y)
        measured = abs(ys[-1] / ys[-2])
        theory = abs(1.0 - omega * (1.0 + mu))
        relax_rows.append({"mu": mu, "omega": omega, "omega_bound": relaxation_bound(mu),
                           "measured_rate": measured, "theory_rate": theory,
                           "rel_err": rel(measured, theory),
                           "converged": abs(ys[-1]) < abs(ys[0])})
    pass_i5 = (
        all(r["rel_err"] < 5e-2 for r in rows)
        and rows[0]["stable_predicted"] and not rows[1]["stable_predicted"]
        and abs(mu_crit - 1.0) < 1e-2
        and relax_rows[0]["converged"] and not relax_rows[1]["converged"]
        and all(r["rel_err"] < 1e-10 for r in relax_rows)
    )
    return {
        "march_rows": rows,
        "mu_critical_bisection": mu_crit,
        "mu_critical_ref": 1.0,
        "relaxation_rows": relax_rows,
        "pass": pass_i5,
    }


# ---------------------------------------------------------------------------
# I6 — hull-girder leg fed through the 3-D surface interface
# ---------------------------------------------------------------------------


def gate_i6() -> dict:
    props = BeamProperties(youngs_modulus=2.1e11, density=7850.0, width=0.1, height=0.01,
                           length=1.0, damping_ratio=0.01, boundary_condition="cantilever")
    EI = props.flexural_rigidity
    n_el = 20

    # (a) static tip point load vs Euler-Bernoulli
    beam = EulerBernoulliBeamFEM(props, n_elements=n_el)
    F = 100.0
    load = beam.apply_point_load(props.length, F)
    w = torch.linalg.solve(beam.K_global, load)
    w_tip_ref = F * props.length ** 3 / (3.0 * EI)
    w_tip_got = float(w[2 * n_el])

    # (b) first natural frequency vs analytic cantilever root
    ev, _ = beam.compute_eigenmodes(n_modes=1)
    f_fem = float(torch.sqrt(ev[0])) / (2 * math.pi)
    f_ref = props.first_natural_frequency

    # (c) pressure on a hull panel, delivered through the 3-D marker interface
    p = 1000.0  # Pa
    q = p * props.width  # UDL N/m
    w_udl_ref = q * props.length ** 4 / (8.0 * EI)
    panel_rows = []
    for n_mark in (8, 16):
        panel = rectangle_surface((0.0, 0.0, 0.05), (1.0, 0.0, 0.0), (0.0, props.width, 0.0),
                                  n_mark, 1)
        traction = torch.stack([torch.zeros(len(panel), dtype=DTYPE),
                                torch.zeros(len(panel), dtype=DTYPE),
                                -p * torch.ones(len(panel), dtype=DTYPE)], 1)
        f_markers = traction * panel.dA.view(-1, 1)
        dof = project_to_beam(panel.positions[:, 0], f_markers[:, 2], props.length, n_el)
        w_panel = torch.linalg.solve(beam.K_global, dof)
        panel_rows.append({
            "n_markers_along_span": n_mark,
            "total_load": float(f_markers[:, 2].sum()),
            "total_load_ref": -q * props.length,
            "load_err": rel(float(f_markers[:, 2].sum()), -q * props.length),
            "w_tip": float(w_panel[2 * n_el]),
            "w_tip_ref": w_udl_ref,
            "w_err": rel(float(w_panel[2 * n_el]), -w_udl_ref),
        })
    for i in range(len(panel_rows) - 1):
        panel_rows[i]["convergence_order"] = order(panel_rows[i]["w_err"],
                                                   panel_rows[i + 1]["w_err"], 2.0)
    pass_i6 = (rel(w_tip_got, w_tip_ref) < 1e-6 and rel(f_fem, f_ref) < 1e-3
               and panel_rows[-1]["w_err"] < 5e-2
               and panel_rows[-1]["w_err"] < panel_rows[0]["w_err"])
    return {
        "tip_load": {"got": w_tip_got, "ref": w_tip_ref, "err": rel(w_tip_got, w_tip_ref)},
        "first_frequency": {"got": f_fem, "ref": f_ref, "err": rel(f_fem, f_ref)},
        "panel_udl": panel_rows,
        "pass": pass_i6,
    }


# ---------------------------------------------------------------------------
# I7 — interface work ledger
# ---------------------------------------------------------------------------


def gate_i7() -> dict:
    """Work-energy identity of the interface load path.

    A single DOF (mass ``m`` on a spring ``k``) whose only applied force is the
    interface force ``f = -k x``.  The work-energy theorem gives ``W = ΔKE``
    exactly in the continuum; the ledger integrates ``W`` with the trapezoid rule
    and the integrator is velocity-Verlet, so the residual is the O(dt²)
    difference between the trapezoid and the midpoint rule — second order, which
    is what the gate checks (two time steps, observed order > 1.8).
    """
    rows = []
    E0 = None
    for dt in (1.0e-2, 5.0e-3):
        m, k = 1.0, 4.0
        x, v = 1.0, 0.0
        ledger = InterfaceLedger()
        e0 = 0.5 * m * v * v + 0.5 * k * x * x
        E0 = e0
        ke0 = 0.5 * m * v * v
        f = -k * x
        n = int(2.0 / dt)
        for _ in range(n):
            a = f / m
            x_new = x + dt * v + 0.5 * dt * dt * a
            f_new = -k * x_new
            a_new = f_new / m
            v_new = v + 0.5 * dt * (a + a_new)
            ledger.add(torch.tensor([f], dtype=DTYPE), torch.tensor([x], dtype=DTYPE))
            ledger.add(torch.tensor([f_new], dtype=DTYPE), torch.tensor([x_new], dtype=DTYPE))
            x, v, f = x_new, v_new, f_new
        dke = 0.5 * m * v * v - ke0
        rows.append({"dt": dt, "W": ledger.work, "dKE": dke, "E0": e0,
                     "closure": abs(ledger.work - dke) / e0})
    for i in range(len(rows) - 1):
        rows[i]["convergence_order"] = order(rows[i]["closure"], rows[i + 1]["closure"], 2.0)
    return {"rows": rows, "pass": rows[-1]["closure"] < 1e-4 and rows[0]["convergence_order"] > 1.8}


# ---------------------------------------------------------------------------
# clock consistency (I4 in the module docstring)
# ---------------------------------------------------------------------------


def gate_clock() -> dict:
    good = CouplingClock(dt_fluid=1e-3, dt_struct=4e-3, n_sub=4, lag=1)
    good.check()
    bad = CouplingClock(dt_fluid=1e-3, dt_struct=1e-2, n_sub=4, lag=0)
    raised = False
    try:
        bad.check()
    except ValueError:
        raised = True
    mu_sphere = added_mass_ratio(m_s=1.0, rho_f=1000.0, displaced_volume=4.0 / 3.0 * math.pi * 0.05 ** 3)
    return {
        "consistent_mismatch": good.mismatch,
        "inconsistent_mismatch": bad.mismatch,
        "invalid_clock_raises": raised,
        "added_mass_ratio_sphere_r0.05_m1kg": mu_sphere,
        "pass": good.mismatch == 0.0 and raised,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(res: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out = []

    # fig 1 — point-load moment conservation: consistent vs nearest-node
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    rows = res["I4"]["rows"]
    h = [r["h"] for r in rows]
    ax.loglog(h, [r["lumped_moment_err"] for r in rows], "o-", label="nearest-node (control)")
    ax.loglog(h, [max(r["consistent_moment_err"], 1e-18) for r in rows], "s-",
              label="Q1 consistent (adopted)")
    ax.axhline(1e-15, ls=":", c="k", lw=0.8)
    ax.set_xlabel("hex edge length h (m)"); ax.set_ylabel("|M - r x F| (machine units)")
    ax.set_title("I4  point-load moment conservation vs resolution")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); p = FIG / "i4_moment_conservation.png"; fig.savefig(p, dpi=140); plt.close(fig)
    out.append(p)

    # fig 2 — open-hemisphere pressure quadrature convergence
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    rows = res["I1"]["hemisphere_rows"]
    n = [r["markers"] for r in rows]
    ax.loglog(n, [r["err"] for r in rows], "o-", label=r"$\int -p\,n\,dA$ vs $p\pi a^2$")
    ax.loglog(n, [r["area_err"] for r in rows], "s--", label=r"area vs $2\pi a^2$")
    ax.set_xlabel("markers on hemisphere"); ax.set_ylabel("relative error")
    ax.set_title("I1  Lagrangian surface quadrature convergence")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); p = FIG / "i1_surface_quadrature.png"; fig.savefig(p, dpi=140); plt.close(fig)
    out.append(p)

    # fig 3 — added-mass stability
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    mus = [0.05 * i for i in range(1, 61)]
    ax.plot(mus, [staggered_spectral_radius(m) for m in mus], "-", label=r"closed form $|\lambda|(\mu)$")
    ax.plot([r["mu"] for r in res["I5"]["march_rows"]],
            [r["growth_measured"] for r in res["I5"]["march_rows"]], "o",
            label="time-marched growth")
    ax.axvline(res["I5"]["mu_critical_bisection"], ls=":", c="r",
               label=r"$\mu_{crit}$ = %.3f" % res["I5"]["mu_critical_bisection"])
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.set_xlabel(r"added-mass ratio $\mu=m_a/m_s$"); ax.set_ylabel(r"growth per step $|\lambda|$")
    ax.set_title("I5  explicit staggered added-mass stability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = FIG / "i5_added_mass.png"; fig.savefig(p, dpi=140); plt.close(fig)
    out.append(p)

    # fig 4 — beam deflection: analytic vs FEM fed through the 3-D interface
    props = BeamProperties(youngs_modulus=2.1e11, density=7850.0, width=0.1, height=0.01,
                           length=1.0, damping_ratio=0.01, boundary_condition="cantilever")
    beam = EulerBernoulliBeamFEM(props, n_elements=20)
    p_press, wb = 1000.0, props.width
    panel = rectangle_surface((0.0, 0.0, 0.05), (1.0, 0.0, 0.0), (0.0, wb, 0.0), 32, 1)
    traction = torch.stack([torch.zeros(len(panel), dtype=DTYPE),
                            torch.zeros(len(panel), dtype=DTYPE),
                            -p_press * torch.ones(len(panel), dtype=DTYPE)], 1)
    f_m = traction * panel.dA.view(-1, 1)
    dof = project_to_beam(panel.positions[:, 0], f_m[:, 2], props.length, 20)
    w_fem = torch.linalg.solve(beam.K_global, dof)
    EI, q = props.flexural_rigidity, -p_press * props.width
    xs = torch.linspace(0, props.length, 200, dtype=DTYPE)
    w_exact = q * xs ** 2 * (6 * props.length ** 2 - 4 * props.length * xs + xs ** 2) / (24 * EI)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(xs, -w_exact, "k-", lw=1.6, label=r"Euler-Bernoulli $qL^4/8EI$ form")
    ax.plot(beam.nodes_x if hasattr(beam, "nodes_x") else torch.linspace(0, props.length, 21),
            [-float(w_fem[2 * i]) for i in range(21)], "o", ms=3.5,
            label="Hermite FEM, load via 3-D interface")
    ax.set_xlabel("x along hull girder (m)"); ax.set_ylabel("deflection w (m)")
    ax.set_title("I6  hull-girder deflection: theory vs interface-fed FEM")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = FIG / "i6_beam_deflection.png"; fig.savefig(p, dpi=140); plt.close(fig)
    out.append(p)

    # fig 5 — ledger closure convergence in dt
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    rows = res["I7"]["rows"]
    ax.loglog([r["dt"] for r in rows], [r["closure"] for r in rows], "o-")
    ax.set_xlabel("dt (s)"); ax.set_ylabel(r"$|W-\Delta E|/|\Delta E|$")
    ax.set_title("I7  interface work ledger closure")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); p = FIG / "i7_ledger.png"; fig.savefig(p, dpi=140); plt.close(fig)
    out.append(p)
    return out


def emit_tables(res: dict) -> Path:
    """Write the gate tables as markdown, straight from the result dict.

    This file — not a human transcription — is the source of every number quoted
    in ``docs/GAP-LBM-T5-ICE-COUPLING_round1.md`` and in the mission report, so a
    stale number is a regeneratable defect rather than an unverifiable claim.
    """
    out = ROOT / "docs" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    L = ["<!-- GENERATED by scripts/three_way_interface3d_gates.py — do not edit -->",
         "", "# GAP-LBM-T5-ICE-COUPLING round-1 gate tables", "",
         "| gate | verdict |", "|---|---|"]
    for k, v in res.get("summary", {}).get("gates", {}).items():
        L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")

    L += ["", "## I1 closed-surface quadrature", "",
          "| n_theta | n_phi | markers | got Fz | ref pi a^2 | rel err | area err | order |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["I1"]["hemisphere_rows"]:
        L.append(f"| {r['n_theta']} | {r['n_phi']} | {r['markers']} | {r['got']:.6f} | {r['ref']:.6f} "
                 f"| {r['err']:.3e} | {r['area_err']:.3e} | {r.get('convergence_order', float('nan')):.3f} |")
    L.append("")
    L.append(f"closed-surface `|sum n dA| = {res['I1']['closed_surface_sum_ndA']:.3e}` (ref 0).")

    L += ["", "## I2 consistent surface load on a Q1 hex mesh", "",
          "| n_cells | F_z got | F_z ref | rel err | |M| |", "|---|---|---|---|---|"]
    for r in res["I2"]["rows"]:
        L.append(f"| {r['n_cells']} | {r['resultant'][2]:.12f} | 2.0 | {r['resultant_rel']:.3e} | {r['moment_norm']:.3e} |")

    L += ["", "## I3 affine patch test", "",
          "| n_cells | n_markers | discrete identity err | W vs Gauss ref | quadrature err | order |",
          "|---|---|---|---|---|---|"]
    for r in res["I3"]["rows"]:
        L.append(f"| {r['n_cells']} | {r['n_markers']} | {r['identity_err']:.3e} | {r['w_gauss_ref']:.6f} "
                 f"| {r['quadrature_err']:.3e} | {r.get('quadrature_order', float('nan')):.3f} |")

    L += ["", "## I4 DEM point load: consistent vs nearest-node control", "",
          "| n_cells | h | resultant err | consistent moment err | lumped moment err | lumped order |",
          "|---|---|---|---|---|---|"]
    for r in res["I4"]["rows"]:
        L.append(f"| {r['n_cells']} | {r['h']:.4f} | {r['consistent_resultant_err']:.3e} "
                 f"| {r['consistent_moment_err']:.3e} | {r['lumped_moment_err']:.6f} "
                 f"| {r.get('lumped_order', float('nan')):.3f} |")

    L += ["", "## I5 added-mass stability", "",
          "| mu | lambda closed form | growth measured | rel err | predicted stable |",
          "|---|---|---|---|---|"]
    for r in res["I5"]["march_rows"]:
        L.append(f"| {r['mu']:.2f} | {r['lambda_closed_form']:.6f} | {r['growth_measured']:.6f} "
                 f"| {r['rel_err']:.3e} | {r['stable_predicted']} |")
    L.append("")
    L.append(f"bisection: mu_critical = {res['I5']['mu_critical_bisection']:.6f} (ref 1.0)")
    L += ["", "| mu | omega | omega bound | measured rate | theory rate | converged |", "|---|---|---|---|---|---|"]
    for r in res["I5"]["relaxation_rows"]:
        L.append(f"| {r['mu']:.2f} | {r['omega']:.4f} | {r['omega_bound']:.6f} | {r['measured_rate']:.6f} "
                 f"| {r['theory_rate']:.6f} | {r['converged']} |")

    L += ["", "## I6 hull-girder leg", "",
          f"tip point load: got {res['I6']['tip_load']['got']:.9f} m vs ref "
          f"{res['I6']['tip_load']['ref']:.9f} m (err {res['I6']['tip_load']['err']:.3e})", "",
          f"first natural frequency: got {res['I6']['first_frequency']['got']:.6f} Hz vs ref "
          f"{res['I6']['first_frequency']['ref']:.6f} Hz (err {res['I6']['first_frequency']['err']:.3e})", "",
          "| markers along span | total load | w_tip got | w_tip ref | rel err | order |",
          "|---|---|---|---|---|---|"]
    for r in res["I6"]["panel_udl"]:
        L.append(f"| {r['n_markers_along_span']} | {r['total_load']:.6f} | {r['w_tip']:.9f} "
                 f"| {r['w_tip_ref']:.9f} | {r['w_err']:.3e} | {r.get('convergence_order', float('nan')):.3f} |")

    L += ["", "## I7 interface work ledger", "",
          "| dt | W | dKE | closure | order |", "|---|---|---|---|---|"]
    for r in res["I7"]["rows"]:
        L.append(f"| {r['dt']} | {r['W']:.9f} | {r['dKE']:.9f} | {r['closure']:.3e} "
                 f"| {r.get('convergence_order', float('nan')):.3f} |")
    L += ["", f"runtime {res['meta']['runtime_s']:.1f} s; repro: `{res['meta']['repro_cmd']}`", ""]
    path = out / "GAP-LBM-T5-ICE-COUPLING_gate_tables.md"
    path.write_text("\n".join(L))
    return path


def main() -> int:
    t0 = time.time()
    res = {
        "I1": gate_i1(),
        "I2": gate_i2(),
        "I3": gate_i3(),
        "I4": gate_i4(),
        "I5": gate_i5(),
        "I6": gate_i6(),
        "I7": gate_i7(),
        "clock": gate_clock(),
    }
    figs = make_figures(res)
    fig_md = [{"file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in figs]
    res["meta"] = {
        "node": "GAP-LBM-T5-ICE-COUPLING",
        "round": 1,
        "scope": "3-D load-exchange interface: dependency audit + interface design + invariant gates",
        "runtime_s": time.time() - t0,
        "figures": fig_md,
        "torch": torch.__version__,
        "dtype": "float64",
        "entry": "scripts/three_way_interface3d_gates.py",
        "repro_cmd": "OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/three_way_interface3d_gates.py",
    }
    gates = {k: v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v}
    res["summary"] = {"gates": gates, "all_pass": all(gates.values())}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(res, indent=2))

    lines = ["gate,metric,ref,got,value,pass"]
    for gid, g in res.items():
        if not isinstance(g, dict) or "pass" not in g:
            continue
        if gid == "I1":
            for r in g["hemisphere_rows"]:
                lines.append(f"I1,hemisphere_pressure_force,p*pi*a^2={r['ref']:.6f},{r['got']:.6f},err={r['err']:.3e},{g['pass']}")
            lines.append(f"I1,closed_surface_sum_n_dA,0,{g['closed_surface_sum_ndA']:.3e},-,-")
        if gid == "I4":
            for r in g["rows"]:
                lines.append(f"I4,moment_err_consistent,n_cells={r['n_cells']},0,{r['consistent_moment_err']:.3e},{g['pass']}")
                lines.append(f"I4,moment_err_lumped,n_cells={r['n_cells']},0,{r['lumped_moment_err']:.3e},-")
        if gid == "I5":
            for r in g["march_rows"]:
                lines.append(f"I5,growth_per_step,lambda={r['lambda_closed_form']:.6f},{r['growth_measured']:.6f},err={r['rel_err']:.3e},{g['pass']}")
            lines.append(f"I5,mu_critical,1.0,{g['mu_critical_bisection']:.6f},-,-")
        if gid == "I6":
            lines.append(f"I6,tip_deflection,F*L^3/3EI={g['tip_load']['ref']:.6f},{g['tip_load']['got']:.6f},err={g['tip_load']['err']:.3e},{g['pass']}")
            lines.append(f"I6,first_freq_hz,{g['first_frequency']['ref']:.6f},{g['first_frequency']['got']:.6f},err={g['first_frequency']['err']:.3e},{g['pass']}")
            for r in g["panel_udl"]:
                lines.append(f"I6,panel_udl_tip_deflection,{r['w_tip_ref']:.6f},{r['w_tip']:.6f},err={r['w_err']:.3e},-")
        if gid == "I7":
            for r in g["rows"]:
                lines.append(f"I7,ledger_closure,dt={r['dt']},0,{r['closure']:.3e},{g['pass']}")
    (OUT / "gates.csv").write_text("\n".join(lines) + "\n")
    table_path = emit_tables(res)

    print(json.dumps(res["summary"], indent=1))
    print(f"\nruntime {res['meta']['runtime_s']:.1f}s -> {OUT/'gates.json'} + {table_path}")
    for gid, g in res.items():
        if isinstance(g, dict) and "pass" in g:
            print(f"  {gid}: {'PASS' if g['pass'] else 'FAIL'}")
    return 0 if res["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())