"""D3Q27 activation and accuracy-ladder validation (gaps G7/G8).

The D3Q27 lattice adds the 8 body-diagonal velocity directions
``(±1,±1,±1)`` that D3Q19 lacks.  Their absence breaks the isotropy of the
4th-order velocity moment of the discrete equilibrium — exactly the moment
that carries the deviatoric (shear) part of the stress tensor — so the
consequence is a lattice-orientation-dependent numerical viscosity and a
larger staircase-boundary error at corners.

This module provides the *measurable* ladder that separates the two lattices,
in three independent layers:

1. **Lattice algebra** (:func:`isotropy_moment_deviation`) — the 2nd/4th/6th
   order moment tensors ``Σ_i w_i c_i ⊗ … ⊗ c_i`` must equal the isotropic
   tensors built from ``c_s² = 1/3``.  D3Q19 is isotropic to 2nd order only;
   D3Q27 is isotropic to 4th order.  The residual is a pure lattice constant:
   exact, deterministic, cheap to assert in CI.
2. **Hydrodynamic isotropy** (:func:`run_shear_wave_case`) — an exact
   unidirectional shear wave ``u = A sin(k·x) ŝ`` (``ŝ ⊥ k̂``) decays as
   ``exp(-νk²t)`` for *any* orientation of ``k̂``.  Measuring the effective
   viscosity for ``k̂ = [100] / [110] / [111]`` isolates the lattice-induced
   anisotropy of the dissipation, which is the physical content of (1).
3. **Flow solution** (:func:`run_sphere_case`, :func:`run_taylor_green_case`) —
   the D3Q27 path of :class:`~tensorlbm.general_sim.GeneralSimEngine` is run
   against the same 3-D sphere benchmark (Re = 100 / 200) and the same
   Taylor–Green exact solution as the D3Q19 path, i.e. identical grid, initial
   state, boundary conditions and force estimator (wet-node momentum
   exchange, gaps G12).

Honesty notes (see ``docs/D3Q27_ACTIVATION.md``):

* the sphere drag reference at Re = 100 is ``Cd ≈ 1.09`` (Schiller–Naumann
  1.0909, Johnson–Patel 1999 DNS 1.087) — *not* 0.44, which is the Re ≈ 1000
  value.  The campaign text's 0.44 is inconsistent with every standard
  reference and is recorded as a known discrepancy rather than silently
  "hit";
* the Taylor–Green case used for the exact L2 reference is the exact NS
  solution embedded in 3-D (``∂/∂z = 0``, extruded 2-D mode).  The genuine
  3-D Taylor–Green vortex has no closed-form solution, so no exact L2
  reference exists for it; the cross-lattice field difference is reported
  instead (:func:`compare_lattices_sphere` does the same for the sphere
  pressure field).

Usage
-----
    from tensorlbm.d3q27_validation import run_validation_suite

    report = run_validation_suite(resolution=16, steps=3000)
    print(report["sphere"]["Re100"]["d3q27"]["cd_err_pct_vs_sn"])
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .d3q19 import C as C19
from .d3q19 import W as W19
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import collide_bgk27, equilibrium27, macroscopic27, stream27
from .general_sim import (
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
)

__all__ = [
    "CS2",
    "LATTICE_DATA",
    "SCHILLER_NAUMANN_NOTE",
    "clift_gauvin_cd",
    "compare_lattices_sphere",
    "isotropy_lattice_difference",
    "isotropy_moment_deviation",
    "literature_sphere_cd",
    "run_shear_wave_case",
    "run_sphere_case",
    "run_taylor_green_case",
    "run_taylor_green_convergence",
    "run_sphere_matrix",
    "run_validation_suite",
    "schiller_naumann_cd",
    "separation_angle_deg",
]

CS2 = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

SCHILLER_NAUMANN_NOTE = (
    "Cd_SN = 24/Re (1 + 0.15 Re^0.687): Re=100 -> 1.0909, Re=200 -> 0.8129. "
    "The campaign text quotes Cd=0.44 at Re=100, which corresponds to Re~1000; "
    "that number is NOT used as a reference here (recorded as a discrepancy)."
)


def schiller_naumann_cd(re: float) -> float:
    """Schiller–Naumann drag correlation for a rigid sphere."""
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def clift_gauvin_cd(re: float) -> float:
    """Clift–Gauvin (1971) drag correlation for a rigid sphere."""
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


#: Steady axisymmetric DNS values (Johnson & Patel 1999, JFM 381, 1–38):
#: Re = 100 → Cd = 1.087, Re = 200 → Cd = 0.772.  These are the primary
#: literature anchors; the correlations above are the cross-check.
_LITERATURE_CD: dict[float, float] = {100.0: 1.087, 200.0: 0.772}


def literature_sphere_cd(re: float) -> float | None:
    """DNS reference drag coefficient (Johnson & Patel 1999), if tabulated."""
    return _LITERATURE_CD.get(float(re))


#: Lattice velocity / weight tables.
LATTICE_DATA: dict[int, tuple[torch.Tensor, torch.Tensor]] = {19: (C19, W19), 27: (C27, W27)}


# ---------------------------------------------------------------------------
# Layer 1: lattice isotropy (exact, algebraic)
# ---------------------------------------------------------------------------


def _pairings(indices: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    """All distinct complete pairings of *indices* (each index used exactly once)."""
    if not indices:
        return [()]
    first, rest = indices[0], indices[1:]
    out: list[tuple[tuple[int, int], ...]] = []
    for j, other in enumerate(rest):
        remaining = rest[:j] + rest[j + 1 :]
        for tail in _pairings(remaining):
            out.append(((first, other), *tail))
    return out


def _isotropic_tensor(order: int, cs2: float = CS2) -> np.ndarray:
    """Isotropic tensor of even *order* built from ``cs2`` and Kronecker deltas.

    ``I_{i1..in} = cs2^{n/2} Σ_{distinct pairings} Π δ_{i_a i_b}`` — e.g. for
    ``n = 4`` this is the familiar ``δδ + δδ + δδ`` with three pairings and for
    ``n = 6`` the 15 pairings.
    """
    d = np.eye(3)
    out = np.zeros((3,) * order)
    for grouping in _pairings(tuple(range(order))):
        term = np.ones((3,) * order)
        for a, b in grouping:
            shape = [1] * order
            shape[a] = 3
            shape[b] = 3
            term = term * d.reshape(shape)
        out += term
    return out * cs2 ** (order // 2)


def _exact_weights(lattice: int, c: np.ndarray) -> np.ndarray:
    """Analytic (exact rational) weights as float64, keyed by the direction class.

    The library tables are stored in float32, so moment residuals computed from
    them bottom out at ~2.5e-9 (single-precision round-off of ``1/3``).  Using
    the rational weights makes the isotropy residuals a property of the lattice
    algebra itself (~1e-17).  The values are cross-checked against the library
    tables in :func:`_weight_table_ok`.
    """
    axis = np.abs(c).sum(axis=1)
    table = {
        19: {0: 1.0 / 3.0, 1: 1.0 / 18.0, 2: 1.0 / 36.0},
        27: {0: 8.0 / 27.0, 1: 2.0 / 27.0, 2: 1.0 / 54.0, 3: 1.0 / 216.0},
    }[lattice]
    w = np.array([table[int(k)] for k in axis], dtype=np.float64)
    # note: for D3Q19 the |c|_1 classes are 0,1,2 (rest, axis, edge); for D3Q27
    # they are 0,1,2,3 (rest, axis, edge, corner).
    return w


def _weight_table_ok(lattice: int, tol: float = 1e-6) -> bool:
    """True when the analytic weights reproduce the shipped float32 tables."""
    c_t, w_t = LATTICE_DATA[lattice]
    c = c_t.detach().cpu().numpy().astype(np.float64)
    w_lib = w_t.detach().cpu().numpy().astype(np.float64)
    return bool(np.abs(_exact_weights(lattice, c) - w_lib).max() < tol)


def isotropy_moment_deviation(lattice: int) -> dict[str, Any]:
    """Max |moment − isotropic target| for orders 2, 4 and 6.

    The isotropic tensors for sound speed ``c_s² = 1/3`` are

    * order 2: ``c_s² δ_ij``
    * order 4: ``c_s⁴ (δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk)``
    * order 6: ``c_s⁶ Σ_{15 pairings} δ…``

    Args:
        lattice: 19 or 27.

    Returns:
        dict with ``order2``/``order4``/``order6`` max absolute deviations,
        booleans marking which orders are isotropic at round-off, and the
        measured second moment (must be ``1/3`` for both lattices).
    """
    if lattice not in LATTICE_DATA:
        raise ValueError(f"lattice must be 19 or 27, got {lattice}")
    c_t = LATTICE_DATA[lattice][0]
    c = c_t.detach().cpu().numpy().astype(np.float64)
    w = _exact_weights(lattice, c)

    m2 = np.einsum("q,qi,qj->ij", w, c, c)
    m4 = np.einsum("q,qi,qj,qk,ql->ijkl", w, c, c, c, c)
    m6 = np.einsum("q,qi,qj,qk,ql,qm,qn->ijklmn", w, c, c, c, c, c, c)

    dev2 = float(np.abs(m2 - _isotropic_tensor(2)).max())
    dev4 = float(np.abs(m4 - _isotropic_tensor(4)).max())
    dev6 = float(np.abs(m6 - _isotropic_tensor(6)).max())
    return {
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "order2": dev2,
        "order4": dev4,
        "order6": dev6,
        "isotropic_order2": dev2 < 1e-12,
        "isotropic_order4": dev4 < 1e-12,
        "isotropic_order6": dev6 < 1e-12,
        "second_moment_xx": float(m2[0, 0]),
        "cs2_target": CS2,
        "weight_sum": float(w.sum()),
        "weights_match_library_table": _weight_table_ok(lattice),
    }


def isotropy_lattice_difference() -> dict[str, Any]:
    """Where D3Q19 and D3Q27 actually differ: rank-4 vs rank-6 moment tensors.

    Both lattices reproduce the isotropic rank-2 and rank-4 tensors exactly
    (see :func:`isotropy_moment_deviation`), so the popular shorthand "D3Q19
    lacks 4th-order isotropy" is *false* as stated.  The sets differ at rank 6:
    D3Q27 contains the 8 body-diagonal ``(±1,±1,±1)`` directions, D3Q19 does
    not.  This function quantifies that difference and also reports each
    lattice's rank-6 isotropy residual.
    """
    out: dict[str, Any] = {"rank4_max_abs_diff": 0.0, "rank6_max_abs_diff": 0.0}
    tensors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for q in (19, 27):
        c_t = LATTICE_DATA[q][0]
        c = c_t.detach().cpu().numpy().astype(np.float64)
        w = _exact_weights(q, c)
        tensors[q] = (
            np.einsum("q,qi,qj->ij", w, c, c),
            np.einsum("q,qi,qj,qk,ql->ijkl", w, c, c, c, c),
            np.einsum("q,qi,qj,qk,ql,qm,qn->ijklmn", w, c, c, c, c, c, c),
        )
    for rank, idx in ((4, 1), (6, 2)):
        diff = float(np.abs(tensors[27][idx] - tensors[19][idx]).max())
        out[f"rank{rank}_max_abs_diff"] = diff
        out[f"rank{rank}_identically_equal"] = diff < 1e-15
    out["rank6_residual_d3q19"] = float(
        np.abs(tensors[19][2] - _isotropic_tensor(6)).max()
    )
    out["rank6_residual_d3q27"] = float(
        np.abs(tensors[27][2] - _isotropic_tensor(6)).max()
    )
    out["interpretation"] = (
        "rank-2 and rank-4 moments are identical and isotropic for both lattices; "
        "the sets separate only at rank 6 (corner directions) and in the O(Ma^3) "
        "equilibrium truncation -> the accuracy gain must be measured, not assumed"
    )
    return out


# ---------------------------------------------------------------------------
# Lattice-agnostic kernel helpers
# ---------------------------------------------------------------------------


def _eq(lattice: int, rho, ux, uy, uz, device):
    if lattice == 27:
        return equilibrium27(rho, ux, uy, uz, device=device)
    return equilibrium3d(rho, ux, uy, uz, device=device)


def _macro(lattice: int, f):
    if lattice == 27:
        return macroscopic27(f)
    return macroscopic3d(f)


def _stream(lattice: int, f):
    if lattice == 27:
        return stream27(f)
    from .solver3d import stream3d

    return stream3d(f)


def _collide_bgk(lattice: int, f, tau: float):
    if lattice == 27:
        return collide_bgk27(f, tau)
    from .solver3d import collide_bgk3d

    return collide_bgk3d(f, tau)


def _unit(vec: tuple[float, float, float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Layer 2: hydrodynamic isotropy (exact shear wave)
# ---------------------------------------------------------------------------


def run_shear_wave_case(
    lattice: int,
    n: int = 32,
    nu: float = 0.02,
    amplitude: float = 0.01,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_vector: tuple[float, float, float] = (1.0, 0.0, 0.0),
    periods: int = 2,
) -> dict[str, Any]:
    """Exact shear-wave decay along an arbitrary lattice direction.

    ``u(x, t) = A sin(k·x) ŝ exp(-ν k² t)`` with ``ŝ ⊥ k`` solves the
    incompressible Navier–Stokes equations exactly (the convective term
    vanishes identically).  Fitting the measured decay gives the *effective*
    viscosity ``ν_eff``; ``(ν_eff − ν)/ν`` is the lattice's dissipation
    anisotropy for that orientation — the hydrodynamic counterpart of the
    algebraic defect from :func:`isotropy_moment_deviation`.

    Args:
        lattice: 19 or 27.
        n: cubic period (cells).
        nu: kinematic viscosity (lattice units).
        amplitude: wave amplitude (keep Ma ≲ 0.05).
        direction: polarisation ``ŝ`` (must be ⟂ ``wave_vector``).
        wave_vector: integer wave-vector direction (periodic on the box).
        periods: number of decay time scales ``1/(νk²)`` to integrate.

    Returns:
        dict with ``nu_eff``, ``nu_err_pct``, ``l2_rel_err`` and metadata.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    khat = _unit(wave_vector)
    shat = _unit(direction)
    if abs(float(khat @ shat)) > 1e-12:
        raise ValueError("polarisation must be orthogonal to the wave vector")

    coords = torch.arange(n, dtype=dtype, device=device) * (2.0 * math.pi / n)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    # coords already carry the 2*pi/n factor, so an integer wave vector m gives
    # phase = m·x (periodic on the box) and |k| = (2*pi/n)·|m|.
    iv = tuple(int(round(v)) for v in wave_vector)
    m = np.asarray(iv, dtype=np.float64)
    if float(np.abs(m - np.asarray(wave_vector, dtype=np.float64)).max()) > 1e-12:
        raise ValueError("wave_vector must have integer components (periodic box)")
    k_mag = 2.0 * math.pi / n * float(np.linalg.norm(m))
    phase = xx * iv[0] + yy * iv[1] + zz * iv[2]
    profile = torch.sin(phase)

    rho = torch.ones((n, n, n), dtype=dtype, device=device)
    ux = amplitude * float(shat[0]) * profile
    uy = amplitude * float(shat[1]) * profile
    uz = amplitude * float(shat[2]) * profile
    f = _eq(lattice, rho, ux, uy, uz, device)

    tau = 3.0 * nu + 0.5
    t_end = periods / (nu * k_mag**2)
    n_steps = max(20, int(round(t_end)))

    a0 = amplitude * math.sqrt(0.5)  # RMS of a sine wave
    t0 = time.time()
    for _ in range(n_steps):
        f = _collide_bgk(lattice, f, tau)
        f = _stream(lattice, f)
    wall = time.time() - t0

    _, vx, vy, vz = _macro(lattice, f)
    u = vx * float(shat[0]) + vy * float(shat[1]) + vz * float(shat[2])
    a1 = float(torch.sqrt((u**2).mean()).item())
    decay_exact = math.exp(-nu * k_mag**2 * n_steps)
    a1_exact = a0 * decay_exact
    nu_eff = -math.log(max(a1, 1e-300) / a0) / (k_mag**2 * n_steps)

    prof_exact = profile * decay_exact
    ex = amplitude * float(shat[0]) * prof_exact
    ey = amplitude * float(shat[1]) * prof_exact
    ez = amplitude * float(shat[2]) * prof_exact
    num = math.sqrt(float((((vx - ex) ** 2) + ((vy - ey) ** 2) + ((vz - ez) ** 2)).mean().item()))
    den = math.sqrt(float(((ex**2) + (ey**2) + (ez**2)).mean().item()))
    return {
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "n": n,
        "nu": nu,
        "tau": tau,
        "steps": n_steps,
        "wave_vector": list(wave_vector),
        "direction": list(direction),
        "amplitude": amplitude,
        "nu_eff": nu_eff,
        "nu_err_pct": (nu_eff - nu) / nu * 100.0,
        "amplitude_rms_end": a1,
        "amplitude_rms_exact": a1_exact,
        "amplitude_rel_err_pct": (a1 - a1_exact) / a1_exact * 100.0,
        "l2_rel_err": num / den if den > 0 else float("nan"),
        "wall_seconds": wall,
    }


def isotropy_dissipation_matrix(
    lattice: int,
    n: int = 32,
    nu: float = 0.02,
    amplitude: float = 0.01,
) -> dict[str, Any]:
    """Effective viscosity for three wave-vector orientations (isotropy error).

    Returns the per-orientation ``ν_eff`` plus the spread
    ``max|ν_eff − ν| / ν``, the single number that quantifies the lattice's
    hydrodynamic anisotropy.  The three wave vectors share the same modulus,
    so the spread is not contaminated by the O(k²) truncation error.
    """
    # All three wave vectors have the SAME integer norm |m| = 3, so |k| is
    # identical and the measured differences are pure lattice anisotropy
    # (a |k|-mismatch would mix in the O(k^2) truncation error, which is
    # common to both lattices and would mask the effect).
    cases = {
        "k_3x[100]": dict(wave_vector=(3.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0)),
        "k_2x[110]+[001]": dict(wave_vector=(2.0, 2.0, 1.0), direction=(1.0, -1.0, 0.0)),
        "k_[100]+2x[011]": dict(wave_vector=(1.0, 2.0, 2.0), direction=(0.0, 1.0, -1.0)),
    }
    rows = {}
    for key, kw in cases.items():
        rows[key] = run_shear_wave_case(
            lattice, n=n, nu=nu, amplitude=amplitude, periods=1, **kw
        )
    errs = [abs(r["nu_err_pct"]) for r in rows.values()]
    return {
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "n": n,
        "nu": nu,
        "cases": rows,
        "max_abs_nu_err_pct": max(errs),
        "spread_nu_err_pct": max(r["nu_err_pct"] for r in rows.values())
        - min(r["nu_err_pct"] for r in rows.values()),
    }


# ---------------------------------------------------------------------------
# Layer 3a: Taylor–Green exact solution
# ---------------------------------------------------------------------------


def run_taylor_green_case(
    lattice: int,
    n: int = 32,
    nu: float = 0.02,
    amplitude: float = 0.02,
    t_end: float | None = None,
    sample_every: int = 10,
) -> dict[str, Any]:
    """Taylor–Green exact solution in 3-D (extruded 2-D mode), L2 error.

    ``u = A sin(kx) cos(ky) e^{-ν(kx²+ky²)t}``,
    ``v = -A cos(kx) sin(ky) e^{-ν(kx²+ky²)t}``, ``w = 0`` on a periodic box
    with ``∂_z = 0`` — an exact solution of the incompressible Navier–Stokes
    equations (``p ∝ e^{-2ν(kx²+ky²)t}``).  With one wave per box and lattice
    spacing 1, ``kx = ky = 2π/N``.

    Note (honesty): the genuine 3-D Taylor–Green vortex (``w ≠ 0``) has no
    closed-form solution, so it cannot supply an exact L2 reference; this
    extruded mode is the closest 3-D configuration with an analytic answer.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    coords = torch.arange(n, dtype=dtype, device=device) * (2.0 * math.pi / n)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    # Lattice units: box = n cells, lattice spacing 1 -> the TG mode has
    # kx = ky = 2*pi/n and Laplacian eigenvalue -(kx^2 + ky^2), so the exact
    # decay is exp(-nu (kx^2+ky^2) t) (NOT exp(-2 nu t), which holds only when
    # the box is 2*pi long in *lattice* units).
    ksq = 2.0 * (2.0 * math.pi / n) ** 2
    decay_rate = nu * ksq
    if t_end is None:
        t_end = 0.5 / decay_rate  # half a viscous decay time
    n_steps = max(10, int(round(t_end)))

    rho = torch.ones((n, n, n), dtype=dtype, device=device)
    ux = amplitude * torch.sin(xx) * torch.cos(yy)
    uy = -amplitude * torch.cos(xx) * torch.sin(yy)
    uz = torch.zeros_like(ux)
    f = _eq(lattice, rho, ux, uy, uz, device)
    tau = 3.0 * nu + 0.5

    history: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = _collide_bgk(lattice, f, tau)
        f = _stream(lattice, f)
        if step % sample_every == 0 or step == n_steps:
            rho_t, vx, vy, vz = _macro(lattice, f)
            dec = math.exp(-decay_rate * step)
            ex = amplitude * torch.sin(xx) * torch.cos(yy) * dec
            ey = -amplitude * torch.cos(xx) * torch.sin(yy) * dec
            num = math.sqrt(
                float((((vx - ex) ** 2) + ((vy - ey) ** 2) + (vz**2)).mean().item())
            )
            den = math.sqrt(float(((ex**2) + (ey**2)).mean().item()))
            history.append(
                {
                    "step": float(step),
                    "l2_rel_err": num / den,
                    "rho_drift": float(rho_t.mean().item()) - 1.0,
                }
            )
    wall = time.time() - t0
    return {
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "n": n,
        "nu": nu,
        "tau": tau,
        "steps": n_steps,
        "amplitude": amplitude,
        "decay_rate": decay_rate,
        "l2_rel_err_final": history[-1]["l2_rel_err"] if history else float("nan"),
        "rho_drift_final": history[-1]["rho_drift"] if history else float("nan"),
        "history": history,
        "wall_seconds": wall,
    }


def run_taylor_green_convergence(
    lattice: int,
    sizes: tuple[int, ...] = (16, 32, 64),
    nu: float = 0.02,
    amplitude: float = 0.02,
) -> dict[str, Any]:
    """Observed spatial order of the Taylor–Green L2 error."""
    rows = []
    for n in sizes:
        res = run_taylor_green_case(lattice, n=n, nu=nu, amplitude=amplitude)
        res.pop("history", None)
        rows.append(res)
    errors = [r["l2_rel_err_final"] for r in rows]
    orders = [
        math.log(errors[i - 1] / errors[i]) / math.log(rows[i]["n"] / rows[i - 1]["n"])
        for i in range(1, len(rows))
    ]
    return {
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "sizes": list(sizes),
        "l2_rel_err": errors,
        "observed_order": orders,
        "nu": nu,
    }


# ---------------------------------------------------------------------------
# Layer 3b: sphere benchmark through GeneralSimEngine
# ---------------------------------------------------------------------------


def _sphere_config(
    lattice: int,
    re: float,
    resolution: int,
    steps: int,
    out_dir: str,
    collision: CollisionModel = CollisionModel.AUTO,
    force_sample_interval: int = 25,
) -> GeneralSimConfig:
    """Sphere benchmark config — identical physics for both lattices."""
    return GeneralSimConfig(
        name=f"sphere_re{int(re)}_d{resolution}_d3q{lattice}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            # Re = u L / nu is set by the *physics* triple, so nu must be derived
            # from the requested Re (a fixed nu silently ran every Re at 100).
            viscosity=1.0e-4 * 1.0 / float(re),
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q27 if lattice == 27 else LatticeModel.D3Q19,
            collision=collision,
            resolution=resolution,
            max_steps=steps,
            snapshot_interval=10**9,  # no per-step snapshots (memory)
            force_sample_interval=force_sample_interval,
            device="cpu",
            force_method=ForceMethod.BOTH,
            mem_variant="wet_node",  # G12 fix: wet-node MEM, no pressure extrapolation
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
        ),
        output=OutputConfig(
            directory=out_dir,
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def separation_angle_deg(
    ux: torch.Tensor,
    uy: torch.Tensor,
    solid: torch.Tensor,
    diameter_lu: float,
    band: float = 1.6,
) -> dict[str, Any]:
    """Azimuthal separation angle of the sphere wake (mid x-y plane).

    The angle is measured from the front stagnation point (``theta = 0``
    upstream, ``theta = 180`` downstream).  Every fluid cell of the mid-plane
    inside a one-cell band outside the voxelised surface contributes one sample
    ``u_theta = u · e_theta`` with ``e_theta = (sin theta, cos theta)``; samples
    are collected into 1-degree azimuth bins (mean per bin) and the separation
    angle is the first bin where ``u_theta`` crosses from attached (positive) to
    reversed (negative), linearly interpolated between bin centres.

    On a staircase surface this is a first-order-accurate geometric estimate;
    ``n_samples`` and the bin width are returned so the uncertainty is visible.
    """
    nz, ny, nx = solid.shape
    kz = nz // 2
    solid_mid = solid[kz]
    if not bool(solid_mid.any()):
        return {"separation_angle_deg": float("nan"), "n_samples": 0}

    idx = torch.nonzero(solid_mid)
    cy = float(idx[:, 0].float().mean().item())
    cx = float(idx[:, 1].float().mean().item())
    r_geo = diameter_lu / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    rad = torch.sqrt(dx * dx + dy * dy)
    fluid = ~solid_mid
    band_mask = fluid & (rad >= r_geo) & (rad <= r_geo + band) & (rad > 1e-6)
    if not bool(band_mask.any()):
        return {"separation_angle_deg": float("nan"), "n_samples": 0}

    theta = torch.atan2(dy[band_mask], -dx[band_mask]).abs() * (180.0 / math.pi)
    sin_t = torch.sin(theta * math.pi / 180.0)
    cos_t = torch.cos(theta * math.pi / 180.0)
    ut = ux[kz][band_mask] * sin_t + uy[kz][band_mask] * cos_t

    bins = torch.clamp(theta.floor().to(torch.long), 0, 179)
    sums = torch.zeros(180, dtype=torch.float64)
    counts = torch.zeros(180, dtype=torch.int64)
    for b, v in zip(bins.tolist(), ut.tolist()):
        sums[b] += v
        counts[b] += 1
    centres = torch.arange(180, dtype=torch.float64) + 0.5
    means = torch.where(
        counts > 0, sums / counts.clamp(min=1).to(torch.float64), torch.full_like(sums, float("nan"))
    )
    filled = counts > 0
    if int(filled.sum()) < 3:
        return {"separation_angle_deg": float("nan"), "n_samples": int(counts.sum().item())}

    th_f = centres[filled].tolist()
    ut_f = means[filled].tolist()
    sep = float("nan")
    for i in range(1, len(ut_f)):
        if ut_f[i - 1] > 0.0 >= ut_f[i]:
            w = ut_f[i - 1] / (ut_f[i - 1] - ut_f[i])
            sep = th_f[i - 1] + w * (th_f[i] - th_f[i - 1])
            break
    return {
        "separation_angle_deg": sep,
        "n_samples": int(counts.sum().item()),
        "theta_bin_deg": 1.0,
        "u_theta_max": float(max(ut_f)),
        "u_theta_min": float(min(ut_f)),
        "n_reversed_bins": int(sum(1 for v in ut_f if v < 0.0)),
        "centre_lu": [cx, cy],
    }


def _window_rel_std(entries: list[dict], key: str, last_frac: float = 0.25) -> float:
    """Relative standard deviation of the last ``last_frac`` force samples."""
    if not entries:
        return float("nan")
    n = max(2, int(len(entries) * last_frac))
    vals = np.array([e[key] for e in entries[-n:] if e.get(key) is not None], dtype=float)
    if vals.size < 2 or vals.mean() == 0:
        return float("nan")
    return float(vals.std() / abs(vals.mean()))


def _window_mean(entries: list[dict], key: str, last_frac: float = 0.25) -> float:
    """Mean of the last ``last_frac`` of the force samples (steady window)."""
    if not entries:
        return float("nan")
    n = max(1, int(len(entries) * last_frac))
    vals = [e[key] for e in entries[-n:] if e.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def run_sphere_case(
    re: float,
    lattice: int,
    resolution: int = 16,
    steps: int = 3000,
    out_dir: str = "/tmp/tensorlbm_d3q27_validation/sphere",
    collision: CollisionModel = CollisionModel.AUTO,
    force_sample_interval: int = 25,
    keep_fields: bool = True,
) -> dict[str, Any]:
    """Run one sphere case through the D3Q19/D3Q27 GeneralSimEngine path.

    Returns the drag catalogue (wet-node MEM — the G12 fix — and the legacy
    pressure+friction control surface), the pressure-field L2 norm, the
    separation angle and run diagnostics.  Both lattices use the *same*
    engine, domain, initial state, BC and force estimator, so the only
    difference in the numbers is the lattice.
    """
    cfg = _sphere_config(
        lattice=lattice,
        re=re,
        resolution=resolution,
        steps=steps,
        out_dir=out_dir,
        collision=collision,
        force_sample_interval=force_sample_interval,
    )
    engine = GeneralSimEngine(cfg)
    setup_info = engine.setup()
    t0 = time.time()
    run_info = engine.run(steps=steps)
    wall = time.time() - t0

    rho, ux, uy, uz = _macro(lattice, engine.f)
    if engine.solid is not None:
        ux = ux.masked_fill(engine.solid, 0.0)
        uy = uy.masked_fill(engine.solid, 0.0)
        uz = uz.masked_fill(engine.solid, 0.0)

    d_lu = float(2.0 * cfg.geometry.sphere_radius / (cfg.physics.reference_length / resolution))
    sep = separation_angle_deg(ux, uy, engine.solid, d_lu)

    cd_mem = _window_mean(engine.forces_log, "cd_mem")
    cd_pf = _window_mean(engine.forces_log, "cd_total")
    cd_p = _window_mean(engine.forces_log, "cd_pressure")
    cd_f = _window_mean(engine.forces_log, "cd_friction")
    cd_ref_sn = schiller_naumann_cd(re)
    cd_ref_lit = literature_sphere_cd(re)

    p = (rho - 1.0) * CS2  # p = c_s^2 (rho - 1) at rho_0 = 1
    fluid = ~engine.solid
    p_fluid = p[fluid]

    result: dict[str, Any] = {
        "re": re,
        "lattice": f"D3Q{lattice}",
        "q": lattice,
        "resolution": resolution,
        "diameter_lu": d_lu,
        "steps": steps,
        "domain_lu": list(setup_info["domain_lu"]),
        "total_cells": setup_info["total_cells"],
        "obstacle_cells": setup_info["obstacle_cells"],
        "tau": setup_info["tau"],
        "u_lb": setup_info["u_lb"],
        "ma": setup_info["Ma"],
        "auto_collision": setup_info["auto_collision"],
        "diverged": run_info["diverged"],
        "wall_seconds": wall,
        "ms_per_step": wall / max(run_info["steps"], 1) * 1000.0,
        "n_force_samples": len(engine.forces_log),
        "cd_mem_wet_node": cd_mem,
        "cd_pressure": cd_p,
        "cd_friction": cd_f,
        "cd_pressure_friction": cd_pf,
        "cd_ref_schiller_naumann": cd_ref_sn,
        "cd_ref_literature_jp1999": cd_ref_lit,
        "cd_err_pct_vs_sn": (cd_mem - cd_ref_sn) / cd_ref_sn * 100.0,
        "cd_err_pct_vs_literature": (
            None if cd_ref_lit is None else (cd_mem - cd_ref_lit) / cd_ref_lit * 100.0
        ),
        "cd_pf_err_pct_vs_sn": (cd_pf - cd_ref_sn) / cd_ref_sn * 100.0,
        "separation_angle_deg": sep["separation_angle_deg"],
        "separation": sep,
        "pressure_l2_norm": float(torch.sqrt((p_fluid**2).sum()).item()),
        "pressure_l2_rms": float(torch.sqrt((p_fluid**2).mean()).item()),
        "pressure_min": float(p_fluid.min().item()),
        "pressure_max": float(p_fluid.max().item()),
        "u_max": float(torch.sqrt(ux**2 + uy**2 + uz**2).max().item()),
        "mass_drift": float(rho.sum().item() - engine._initial_mass),
        "cd_window_rel_std": _window_rel_std(engine.forces_log, "cd_mem"),
        "cd_history": [
            {"step": e["step"], "cd_mem": e.get("cd_mem"), "cd_total": e.get("cd_total")}
            for e in engine.forces_log
        ],
    }
    # Stability / convergence verdict for this case.  A healthy incompressible
    # run keeps max|u| within a small multiple of the inlet velocity; a
    # collapsing Cd window together with a large max|u| means the case went
    # unstable and its Cd must not be compared against a reference.
    u_over_u_in = result["u_max"] / result["u_lb"] if result["u_lb"] else float("nan")
    result["u_max_over_u_in"] = u_over_u_in
    result["stable"] = bool((not run_info["diverged"]) and u_over_u_in < 3.0)
    result["converged"] = bool(result["stable"] and result.get("cd_window_rel_std", 1.0) < 0.05)

    if keep_fields:
        kz = engine.solid.shape[0] // 2
        result["_fields"] = {
            "midplane_u": torch.sqrt(ux[kz] ** 2 + uy[kz] ** 2).detach().cpu().numpy(),
            "midplane_p": p[kz].detach().cpu().numpy(),
            "midplane_solid": engine.solid[kz].detach().cpu().numpy(),
            "midplane_ux": ux[kz].detach().cpu().numpy(),
            "midplane_uy": uy[kz].detach().cpu().numpy(),
        }
    return result


def compare_lattices_sphere(re: float, res19: dict[str, Any], res27: dict[str, Any]) -> dict[str, Any]:
    """Quantify the D3Q19 → D3Q27 accuracy gain for one sphere case.

    Metrics:

    * ``cd_err_pct_*`` — |error| vs Schiller–Naumann and vs the
      Johnson–Patel DNS anchor, and the reduction (percentage points);
    * ``pressure_l2_rel_diff_19_vs_27`` — ``||p19 − p27|| / ||p27||`` over the
      fluid cells of the mid-plane (lattice-induced pressure-field error,
      D3Q27 as reference);
    * separation-angle difference.
    """
    p19 = res19.get("_fields", {}).get("midplane_p")
    p27 = res27.get("_fields", {}).get("midplane_p")
    press_l2 = float("nan")
    if p19 is not None and p27 is not None:
        p19t = torch.as_tensor(np.asarray(p19, dtype=np.float64))
        p27t = torch.as_tensor(np.asarray(p27, dtype=np.float64))
        mask = ~torch.as_tensor(np.asarray(res27["_fields"]["midplane_solid"], dtype=bool))
        num = float(torch.sqrt(((p27t[mask] - p19t[mask]) ** 2).sum()).item())
        den = float(torch.sqrt((p27t[mask] ** 2).sum()).item())
        press_l2 = num / den if den > 0 else float("nan")

    sep19 = res19.get("separation_angle_deg")
    sep27 = res27.get("separation_angle_deg")
    sep_delta = (
        None
        if sep19 is None or sep27 is None or math.isnan(sep19) or math.isnan(sep27)
        else sep27 - sep19
    )
    out = {
        "re": re,
        "cd_d3q19": res19["cd_mem_wet_node"],
        "cd_d3q27": res27["cd_mem_wet_node"],
        "cd_err_pct_d3q19_vs_sn": res19["cd_err_pct_vs_sn"],
        "cd_err_pct_d3q27_vs_sn": res27["cd_err_pct_vs_sn"],
        "cd_err_reduction_pct_points_vs_sn": abs(res19["cd_err_pct_vs_sn"])
        - abs(res27["cd_err_pct_vs_sn"]),
        "cd_err_pct_d3q19_vs_lit": res19.get("cd_err_pct_vs_literature"),
        "cd_err_pct_d3q27_vs_lit": res27.get("cd_err_pct_vs_literature"),
        "cd_rel_change_pct": (res27["cd_mem_wet_node"] - res19["cd_mem_wet_node"])
        / res19["cd_mem_wet_node"]
        * 100.0,
        "pressure_l2_rel_diff_19_vs_27": press_l2,
        "pressure_l2_norm_d3q19": res19["pressure_l2_norm"],
        "pressure_l2_norm_d3q27": res27["pressure_l2_norm"],
        "separation_angle_d3q19": sep19,
        "separation_angle_d3q27": sep27,
        "separation_angle_delta_deg": sep_delta,
        "stable_d3q19": res19.get("stable"),
        "stable_d3q27": res27.get("stable"),
        "both_stable": bool(res19.get("stable") and res27.get("stable")),
    }
    if out["cd_err_pct_d3q19_vs_lit"] is not None:
        out["cd_err_reduction_pct_points_vs_lit"] = abs(out["cd_err_pct_d3q19_vs_lit"]) - abs(
            out["cd_err_pct_d3q27_vs_lit"]
        )
    return out

# ---------------------------------------------------------------------------
# Parallel sphere workers (the sphere runs dominate the wall clock)
# ---------------------------------------------------------------------------


def _sphere_worker(job: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    """Process-pool entry point: run one sphere case and return its report."""
    import os

    threads = int(job.pop("threads", 0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    res = run_sphere_case(**job)
    return job_key(job), int(job["lattice"]), res


def job_key(job: dict[str, Any]) -> str:
    """Stable report key for one sphere job."""
    return f"Re{int(job['re'])}"


def run_sphere_matrix(
    re_values: tuple[float, ...],
    lattices: tuple[int, ...],
    resolution: int,
    steps: int,
    out_root: Path,
    workers: int = 1,
    threads_per_worker: int = 8,
    sphere_case_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the (Re × lattice) sphere matrix, optionally in parallel.

    Args:
        re_values: Reynolds numbers.
        lattices: lattices to run (19/27).
        resolution: sphere diameter in cells.
        steps: steps per case.
        out_root: root directory for the per-case output folders.
        workers: number of concurrent processes (>1 → ProcessPoolExecutor).
        threads_per_worker: torch intra-op threads inside each worker process.
        sphere_case_kwargs: extra kwargs forwarded to :func:`run_sphere_case`.
        sphere_workers: concurrent processes for the (Re × lattice) sphere matrix.
        threads_per_worker: torch threads inside each sphere worker.

    Returns:
        ``{Re100: {d3q19: result, d3q27: result}, ...}``
    """
    skw = dict(sphere_case_kwargs or {})
    jobs: list[dict[str, Any]] = []
    for re_v in re_values:
        for q in lattices:
            jobs.append(
                {
                    "re": re_v,
                    "lattice": q,
                    "resolution": resolution,
                    "steps": steps,
                    "out_dir": str(out_root / "sphere" / f"Re{int(re_v)}_d3q{q}"),
                    "threads": threads_per_worker,
                    **skw,
                }
            )

    results: dict[str, dict[str, dict[str, Any]]] = {}
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            job = dict(job)
            threads = int(job.pop("threads", 0) or 0)
            if threads > 0:
                torch.set_num_threads(threads)
            res = run_sphere_case(**job)
            results.setdefault(job_key(job), {})[f"d3q{job['lattice']}"] = res
        return results

    import concurrent.futures as _cf
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    with _cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_sphere_worker, dict(job)) for job in jobs]
        for fut in _cf.as_completed(futures):
            key, q, res = fut.result()
            results.setdefault(key, {})[f"d3q{q}"] = res
    # deterministic key order
    return {k: dict(sorted(v.items())) for k, v in sorted(results.items())}


# ---------------------------------------------------------------------------
# Report figures
# ---------------------------------------------------------------------------


def _figures(
    report: dict[str, Any],
    fig_dir: Path,
    lattices: tuple[int, ...] = (19, 27),
    re_values: tuple[float, ...] = (100.0, 200.0),
) -> list[dict[str, Any]]:
    """Render the validation figures; returns ``[{name, path, caption}]``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: list[dict[str, Any]] = []
    sphere = report.get("sphere", {})

    # 1. mid-plane |u| and p per (Re, lattice)
    panels: list[tuple[str, dict[str, Any]]] = []
    for re_v in re_values:
        key = f"Re{int(re_v)}"
        for lat in lattices:
            case = sphere.get(key, {}).get(f"d3q{lat}")
            if case and case.get("_fields") is not None:
                panels.append((f"{key} {case['lattice']}", case["_fields"]))
    if panels:
        fig, axes = plt.subplots(
            len(panels), 2, figsize=(9.5, 3.0 * len(panels)), squeeze=False
        )
        for i, (title, fd) in enumerate(panels):
            um = np.array(fd["midplane_u"], dtype=float, copy=True)
            um[fd["midplane_solid"]] = np.nan
            im0 = axes[i][0].imshow(um, origin="lower", cmap="viridis")
            axes[i][0].set_title(f"{title}: |u| (mid-z), lattice units")
            axes[i][0].set_xlabel("i (x)")
            axes[i][0].set_ylabel("j (y)")
            plt.colorbar(im0, ax=axes[i][0], fraction=0.046)
            pm = np.array(fd["midplane_p"], dtype=float, copy=True)
            pm[fd["midplane_solid"]] = np.nan
            lim = float(np.nanpercentile(np.abs(pm), 99)) or 1.0
            im1 = axes[i][1].imshow(pm, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
            axes[i][1].set_title(f"{title}: p (rho/3), mid-z")
            axes[i][1].set_xlabel("i (x)")
            axes[i][1].set_ylabel("j (y)")
            plt.colorbar(im1, ax=axes[i][1], fraction=0.046)
        fig.tight_layout()
        p = fig_dir / "sphere_midplane_fields.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(
            {
                "name": p.name,
                "path": str(p),
                "caption": "Mid-plane (z = nz/2) speed and pressure fields for the "
                "sphere cases; D3Q19 vs D3Q27 on identical grids.",
            }
        )

    # 2. Cd history
    fig, axes = plt.subplots(1, len(re_values), figsize=(5.0 * len(re_values), 3.6), squeeze=False)
    for j, re_v in enumerate(re_values):
        key = f"Re{int(re_v)}"
        ax = axes[0][j]
        for lat in lattices:
            case = sphere.get(key, {}).get(f"d3q{lat}")
            if not case:
                continue
            hist = case.get("cd_history") or []
            steps = [h["step"] for h in hist]
            cds = [h["cd_mem"] for h in hist]
            ax.plot(steps, cds, label=f"{case['lattice']}", lw=1.2)
        ref = schiller_naumann_cd(re_v)
        ax.axhline(ref, color="k", ls="--", lw=1.0, label=f"Schiller-Naumann {ref:.3f}")
        lit = literature_sphere_cd(re_v)
        if lit:
            ax.axhline(lit, color="tab:red", ls=":", lw=1.0, label=f"DNS (J&P 1999) {lit:.3f}")
        ax.set_title(f"Sphere Re={int(re_v)}: Cd history (wet-node MEM)")
        ax.set_xlabel("step")
        ax.set_ylabel("$C_D$")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "sphere_cd_history.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": p.name,
            "path": str(p),
            "caption": "Drag-coefficient history against the Schiller-Naumann "
            "correlation and the Johnson-Patel (1999) DNS anchor.",
        }
    )

    # 3. dissipation isotropy
    iso = report.get("isotropy_dissipation", {})
    if iso:
        labels = list(next(iter(iso.values()))["cases"].keys())
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        width = 0.8 / max(len(iso), 1)
        for i, (k, row) in enumerate(sorted(iso.items())):
            vals = [abs(row["cases"][lb]["nu_err_pct"]) for lb in labels]
            ax.bar(x + i * width, vals, width, label=k.upper())
        ax.set_xticks(x + width * (len(iso) - 1) / 2)
        ax.set_xticklabels([lb.replace("_", " ") for lb in labels], fontsize=8)
        ax.set_ylabel(r"$|\nu_{eff}-\nu|/\nu$  [%]")
        ax.set_title("Hydrodynamic isotropy: shear-wave dissipation error by orientation")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = fig_dir / "isotropy_dissipation.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(
            {
                "name": p.name,
                "path": str(p),
                "caption": "Effective-viscosity error of the exact shear-wave decay along "
                "[100]/[110]/[111] wave vectors (lower and flatter = more isotropic).",
            }
        )

    # 4. Taylor-Green convergence
    tg = report.get("taylor_green", {})
    if tg:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for k, row in sorted(tg.items()):
            ns = np.array(row["sizes"], dtype=float)
            errs = np.array(row["l2_rel_err"], dtype=float)
            ax.loglog(ns, errs, "o-", label=f"{k.upper()}")
            if len(ns) >= 2:
                # reference slope 2 anchored at the coarsest point
                ax.loglog(
                    ns,
                    errs[0] * (ns / ns[0]) ** -2.0,
                    "k:",
                    lw=0.8,
                    label="slope -2" if k == sorted(tg)[0] else None,
                )
        ax.set_xlabel("N (cells per 2*pi box)")
        ax.set_ylabel("relative L2 velocity error")
        ax.set_title("Taylor-Green (3-D exact solution) L2 convergence")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = fig_dir / "taylor_green_convergence.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(
            {
                "name": p.name,
                "path": str(p),
                "caption": "L2 velocity error of the Taylor-Green exact solution vs grid; "
                "dotted line = second-order reference slope.",
            }
        )

    # 5. Cd bar chart with references
    if sphere:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        keys = [f"Re{int(r)}" for r in re_values]
        x = np.arange(len(keys))
        width = 0.8 / (len(lattices) + 2)
        for i, lat in enumerate(lattices):
            vals = [
                sphere.get(k, {}).get(f"d3q{lat}", {}).get("cd_mem_wet_node", np.nan)
                for k in keys
            ]
            ax.bar(x + i * width, vals, width, label=f"D3Q{lat} (wet-node MEM)")
        off = len(lattices) * width
        ax.bar(
            x + off,
            [schiller_naumann_cd(r) for r in re_values],
            width,
            label="Schiller-Naumann",
            color="0.6",
        )
        ax.bar(
            x + off + width,
            [literature_sphere_cd(r) or np.nan for r in re_values],
            width,
            label="DNS (J&P 1999)",
            color="0.35",
        )
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(keys)
        ax.set_ylabel("$C_D$")
        ax.set_title("Sphere drag: lattice ladder vs reference")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = fig_dir / "sphere_cd_bars.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(
            {
                "name": p.name,
                "path": str(p),
                "caption": "Sphere drag coefficient per lattice against the correlation and "
                "DNS reference values.",
            }
        )
    return figs


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Recursively drop private (underscore) keys and convert numpy/torch types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def run_validation_suite(
    re_values: tuple[float, ...] = (100.0, 200.0),
    lattices: tuple[int, ...] = (19, 27),
    resolution: int = 16,
    steps: int = 3000,
    out_dir: str | Path = "/tmp/tensorlbm_d3q27_validation",
    tg_sizes: tuple[int, ...] = (16, 32, 64),
    isotropy_n: int = 16,
    make_figures: bool = True,
    sphere_case_kwargs: dict[str, Any] | None = None,
    sphere_workers: int = 1,
    threads_per_worker: int = 8,
) -> dict[str, Any]:
    """Run the full D3Q27 activation validation suite.

    Args:
        re_values: sphere Reynolds numbers.
        lattices: (19, 27) or a subset.
        resolution: sphere diameter in lattice cells.
        steps: steps per sphere case.
        out_dir: root output directory (JSON + figures + per-case dirs).
        tg_sizes: grid sizes for the Taylor–Green convergence study.
        isotropy_n: box size for the shear-wave isotropy study.
        make_figures: render PNG figures (requires matplotlib).
        sphere_case_kwargs: extra kwargs forwarded to :func:`run_sphere_case`.
        sphere_workers: concurrent processes for the (Re × lattice) sphere matrix.
        threads_per_worker: torch threads inside each sphere worker.

    Returns:
        The full report dict (JSON-serialisable apart from ``_fields`` which
        are used for figures and stripped from the dumped JSON).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    skw = dict(sphere_case_kwargs or {})

    report: dict[str, Any] = {
        "suite": "D3Q27 activation / accuracy ladder",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "sphere_workers": sphere_workers,
            "threads_per_worker": threads_per_worker,
            "re_values": list(re_values),
            "lattices": [f"D3Q{q}" for q in lattices],
            "resolution": resolution,
            "steps": steps,
            "tg_sizes": list(tg_sizes),
            "isotropy_n": isotropy_n,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "references": {
            "schiller_naumann": {f"Re{int(r)}": schiller_naumann_cd(r) for r in re_values},
            "clift_gauvin": {f"Re{int(r)}": clift_gauvin_cd(r) for r in re_values},
            "dns_johnson_patel_1999": {
                f"Re{int(r)}": literature_sphere_cd(r) for r in re_values
            },
            "note": SCHILLER_NAUMANN_NOTE,
        },
        "isotropy_moments": {},
        "isotropy_dissipation": {},
        "taylor_green": {},
        "sphere": {},
        "comparison": {},
    }

    for q in lattices:
        report["isotropy_moments"][f"d3q{q}"] = isotropy_moment_deviation(q)
        report["isotropy_dissipation"][f"d3q{q}"] = isotropy_dissipation_matrix(
            q, n=isotropy_n
        )
        report["taylor_green"][f"d3q{q}"] = run_taylor_green_convergence(q, sizes=tg_sizes)

    report["sphere"] = run_sphere_matrix(
        re_values=re_values,
        lattices=lattices,
        resolution=resolution,
        steps=steps,
        out_root=out_root,
        workers=sphere_workers,
        threads_per_worker=threads_per_worker,
        sphere_case_kwargs=skw,
    )
    for re_v in re_values:
        key = f"Re{int(re_v)}"
        if 19 in lattices and 27 in lattices:
            report["comparison"][key] = compare_lattices_sphere(
                re_v, report["sphere"][key]["d3q19"], report["sphere"][key]["d3q27"]
            )

    if make_figures:
        try:
            report["figures"] = _figures(report, fig_dir, lattices, re_values)
        except Exception as exc:  # pragma: no cover - figures are best effort
            report["figures_error"] = f"{type(exc).__name__}: {exc}"

    # Verdicts (machine-readable acceptance summary)
    verdicts: dict[str, Any] = {}
    for key, cmp_row in report["comparison"].items():
        verdicts[key] = {
            "d3q19_stable": cmp_row.get("stable_d3q19"),
            "d3q27_stable": cmp_row.get("stable_d3q27"),
            "cd_err_pct_d3q27_vs_sn": cmp_row["cd_err_pct_d3q27_vs_sn"],
            "d3q27_within_15pct_of_sn": abs(cmp_row["cd_err_pct_d3q27_vs_sn"]) < 15.0,
            "d3q27_within_3pct_of_sn": abs(cmp_row["cd_err_pct_d3q27_vs_sn"]) < 3.0,
            "cd_err_reduction_pct_points": cmp_row["cd_err_reduction_pct_points_vs_sn"],
            "d3q27_better_than_d3q19": (
                abs(cmp_row["cd_err_pct_d3q27_vs_sn"])
                < abs(cmp_row["cd_err_pct_d3q19_vs_sn"])
                if cmp_row.get("both_stable", True)
                else None
            ),
            "case_unstable": not cmp_row.get("both_stable", True),
        }
    iso = report["isotropy_moments"]
    if "d3q19" in iso and "d3q27" in iso:
        verdicts["isotropy"] = {
            "d3q19_order4_residual": iso["d3q19"]["order4"],
            "d3q27_order4_residual": iso["d3q27"]["order4"],
            "d3q27_isotropic_to_order4": iso["d3q27"]["isotropic_order4"],
            "d3q19_isotropic_to_order4": iso["d3q19"]["isotropic_order4"],
        }
    report["verdicts"] = verdicts

    json_path = out_root / "d3q27_validation_report.json"
    report["report_json"] = str(json_path)
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return report
