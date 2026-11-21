"""Fast gates for the acceptance harness of the AA-pattern fused kernel.

Cheap counterparts of ``benchmarks/xval/aa_fused_cavity_acceptance.py`` (whose
cavity sweep runs on the scheduler).  They exist because the 2026-11-13 XVAL
report quoted a cavity table produced by a kernel whose wall treatment was never
checked independently: every moving-wall "equivalence" test compared two code
paths that shared the same bounce-back sign, and the profile metric normalised
*after* subtracting the reference (inflating errors by ``1/U``).

Covered here
------------
* the moving-wall bounce-back reproduces the **exact** plane-Couette profile
  (the only test that can see a momentum-sign error);
* that gate has teeth: a sign-inverted wall term is rejected;
* the fused kernel is bit-identical to the canonical materialised step in the
  *cavity* geometry (non-uniform, wall-bounded, moving lid);
* fluid-node mass is conserved to round-off with a moving lid;
* populations on solid nodes are junk and must not leak into the fluid;
* the ``L2`` metric has a known half-cell floor and is linear in a uniform
  perturbation (neither a tautology nor a rubber stamp).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "benchmarks", "xval"))

from aa_fused_cavity_acceptance import (  # noqa: E402
    _couette_geometry,
    _fluid_mass,
    metric_floor,
    profile_error,
    task_couette,
    task_equality,
)
from aa_fused_stream_collide_ordering_audit import (  # noqa: E402
    GHIA_U,
    _cavity_field,
    _cavity_geometry,
)
from tensorlbm.aa_fused_stream_collide import (  # noqa: E402
    C,
    OPPOSITE,
    W,
    _equilibrium,
    _macroscopic,
    aa_fused_step,
)

torch.set_default_dtype(torch.float64)


def test_couette_matches_exact_solution():
    """Halfway bounce-back with a moving wall: ``u_x(j) = U (j-1/2)/L``."""
    r = task_couette(L=16, U=0.05, tau=0.8, steps=4000)
    assert r["rel_err_max"] < 1e-3, r["rel_err_max"]
    assert r["fluid_mass_drift_rel"] < 1e-10, r["fluid_mass_drift_rel"]
    prof = np.asarray(r["profile_numeric"])
    assert np.all(np.diff(prof) > 0.0)


def _sign_flipped_step(f, tau, solid, U):
    """Same fused pull as the library, but with the **wrong** wall-momentum sign."""
    rho, ux, uy = _macroscopic(f)
    post = f + (1.0 / tau) * (_equilibrium(rho, ux, uy) - f)
    fnew = torch.empty_like(f)
    for i in range(9):
        fnew[i] = torch.roll(post[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1))
    wall_u = torch.zeros((2,) + tuple(solid.shape))
    wall_u[0, -1, :] = U
    for i in range(1, 9):
        sh = (int(C[i, 1]), int(C[i, 0]))
        up = torch.roll(solid, shifts=sh, dims=(0, 1))
        j = int(OPPOSITE[i])
        val = post[j] - 6.0 * W[i] * rho * (int(C[i, 0]) * torch.roll(wall_u[0], shifts=sh, dims=(0, 1)))
        fnew[i] = torch.where(up, val, fnew[i])
    return fnew


def test_couette_gate_detects_sign_flip():
    """Negative control: the Couette gate must reject a sign-inverted wall term."""
    L, U, tau = 16, 0.05, 0.8
    n, solid, f = _couette_geometry(L)
    for _ in range(2000):
        f = _sign_flipped_step(f, tau, solid, U)
    _, ux, _ = _macroscopic(f)
    u_num = ux[1:L + 1, 0].numpy()
    u_ana = U * (np.arange(1, L + 1) - 0.5) / L
    err = float(np.abs(u_num - u_ana).max() / U)
    assert err > 0.1, f"sign-flipped kernel was not rejected (err={err})"


def test_fused_equals_materialised_in_cavity_geometry():
    r = task_equality(L=16, steps=300)
    assert r["bit_equal"], r["first_nonzero_diff"]


def test_fluid_mass_conserved_with_moving_lid():
    L, U = 16, 0.1
    tau = 0.5 + 3.0 * (U * L / 100.0)
    n, solid, _ = _cavity_geometry(L, U)
    fluid = ~solid
    f = _cavity_field(n)
    m0 = _fluid_mass(f, fluid)
    for k in range(1, 1501):
        f = aa_fused_step(f, tau, solid, u_lid=U * min(1.0, k / 500.0), lid_row=-1)
    assert abs(_fluid_mass(f, fluid) - m0) / m0 < 1e-10


def test_solid_node_junk_does_not_leak_into_fluid():
    """Poisoning the populations on solid nodes must not change the fluid state."""
    L, U = 16, 0.1
    tau = 0.5 + 3.0 * (U * L / 100.0)
    n, solid, _ = _cavity_geometry(L, U)
    a = _cavity_field(n)
    b = a.clone()
    b[:, solid] = 1e3
    for k in range(1, 501):
        u = U * min(1.0, k / 200.0)
        a = aa_fused_step(a, tau, solid, u_lid=u, lid_row=-1)
        b = aa_fused_step(b, tau, solid, u_lid=u, lid_row=-1)
    fluid = ~solid
    assert bool(torch.equal(a[:, fluid], b[:, fluid]))


def test_metric_floor_is_small_and_shrinks_with_the_grid():
    """The metric has a half-cell extrapolation floor (~O(1/L)), not zero."""
    f32, f64, f128 = (metric_floor(L)["floor_u"] for L in (32, 64, 128))
    assert f32 < 0.05 and f64 < 0.03 and f128 < 0.015, (f32, f64, f128)
    assert f64 < f32 and f128 < f64
    # interior (no y = 0/1 extrapolation) floor is an order smaller
    assert metric_floor(64)["floor_u_interior"] < 0.01


def test_metric_is_linear_in_perturbation():
    L, U = 64, 0.1
    ys = (np.arange(L) + 0.5) / L
    exact = np.interp(ys, GHIA_U[:, 0], GHIA_U[:, 1]) * U
    for amp in (0.03, 0.10):
        err = profile_error(exact + amp * U, ys, GHIA_U, U)
        assert err == pytest.approx(amp, rel=0.10), (amp, err)


def test_metric_normalises_before_subtracting():
    """Regression for the 1/U metric bug fixed in this mission.

    The retired formula ``sqrt(mean((u - u_ghia)^2))/U`` inflates the residual
    by ``~1/U``: on the same profile it must be ~1/U times larger than the
    correct ``sqrt(mean((u/U - u_ghia)^2))``.
    """
    L, U = 64, 0.1
    ys = (np.arange(L) + 0.5) / L
    exact = np.interp(ys, GHIA_U[:, 0], GHIA_U[:, 1]) * U
    good = profile_error(exact, ys, GHIA_U, U)
    vals = np.interp(GHIA_U[:, 0], ys, exact)
    buggy = float(np.sqrt(np.mean((vals - GHIA_U[:, 1]) ** 2)) / U)
    assert buggy > 5.0 * good, (buggy, good)
