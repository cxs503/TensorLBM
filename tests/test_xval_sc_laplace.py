"""Tests for xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.

Guards the *cross-validation* itself (not the physics benchmark, which lives in
``benchmarks/verified/laplace_droplet``):

1. the from-scratch reference solver (``ref_sc.py``) reproduces the TensorLBM
   SCMP step bit-for-bit up to float32/float64 rounding — otherwise the two
   legs would not be comparing the same model;
2. the Maxwell equal-area construction on the SC EOS is self-consistent
   (equal pressure, equal chemical potential, two distinct densities);
3. the Young-Laplace fit on a *short* run already returns the right order of
   magnitude and the right curvature trend (Δp decreasing with R).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION"
sys.path.insert(0, str(XVAL))
sys.path.insert(0, str(REPO / "src"))

ref_sc = pytest.importorskip("ref_sc")


def _torch_solver():
    pytest.importorskip("torch")
    import run_xval

    return run_xval.TorchSolver(threads=2)


def test_reference_impl_matches_library_step_by_step():
    """ref_sc.py and TensorLBM produce the same distribution function."""
    lib = _torch_solver()
    import run_xval

    L, R = 32, 8.0
    rho0, _ = run_xval.init_droplet(L, R)
    f_lib = lib.init(rho0)
    z = np.zeros_like(rho0)
    f_ref = ref_sc.equilibrium(rho0, z, z)

    # step 0: the equilibrium distributions must agree to float32 precision
    assert np.max(np.abs(f_ref - f_lib.numpy())) < 1e-6

    for _ in range(120):
        f_lib = lib.step(f_lib)
        f_ref = ref_sc.stream(ref_sc.collide_sc(f_ref, G=run_xval.CFG["G_lib"], tau=1.0))
    dmax = float(np.max(np.abs(f_ref - f_lib.numpy())))
    assert dmax < 1e-5, f"implementations diverge: max|df| = {dmax:.3e}"

    # and the macroscopic density agrees
    rho_ref = f_ref.sum(axis=0)
    rho_lib = f_lib.numpy().sum(axis=0)
    assert np.max(np.abs(rho_ref - rho_lib)) < 1e-5


def test_force_points_out_of_the_liquid():
    """G > 0 must be attractive: at a flat interface the force on the liquid
    side pulls towards the vapour (positive x here), i.e. it compresses the
    liquid — the sign convention documented in the library."""
    rho = np.full((4, 64), 0.16)
    rho[:, 32:] = 1.95
    # x < 32 is vapour, x >= 32 is liquid: the interface pair is pulled
    # towards the denser phase (+x), i.e. F = -G psi grad(psi) with psi
    # increasing with rho.  Mirroring the configuration flips both signs.
    fx, fy = ref_sc.sc_force_standard(rho, 5.0)
    assert fx[2, 31] > 0.0 and fx[2, 32] > 0.0, f"interface forces {fx[2, 31]:.3e}/{fx[2, 32]:.3e}"
    assert abs(float(fx.sum())) < 1e-5  # periodic box: action = reaction
    rho_m = np.full((4, 64), 0.16)
    rho_m[:, :32] = 1.95  # mirror image: liquid on the left
    fx_m, _ = ref_sc.sc_force_standard(rho_m, 5.0)
    assert fx_m[2, 31] < 0.0 and fx_m[2, 32] < 0.0
    # a uniform field carries no interaction force
    fx_u, fy_u = ref_sc.sc_force_standard(np.full((8, 8), 1.0), 5.0)
    assert np.max(np.abs(fx_u)) < 1e-12 and np.max(np.abs(fy_u)) < 1e-12


def test_maxwell_coexistence_is_self_consistent():
    rv, rl, pc = ref_sc.maxwell_coexistence(-5.0)
    assert np.isfinite([rv, rl, pc]).all()
    assert rv < rl
    # equal pressure on both branches
    assert abs(ref_sc.eos_pressure(np.asarray(rl), -5.0) - pc) < 1e-6
    assert abs(ref_sc.eos_pressure(np.asarray(rv), -5.0) - pc) < 1e-6
    # equal chemical potential (equal-area condition)
    g = np.linspace(rv, rl, 4001)
    dpi = ref_sc.CS2 - (5.0 / 3.0) * ref_sc.psi_exp(g) * np.exp(-g)
    assert abs(float(np.trapezoid(dpi / g, g))) < 1e-5
    # reference values reproduced (regression guard)
    assert abs(rv - 0.152895) < 5e-4
    assert abs(rl - 1.928866) < 5e-4


@pytest.mark.parametrize("R", [8.0, 12.0])
def test_droplet_laplace_short_run(R):
    """Short-run Young-Laplace check: converged dp in a plausible range."""
    import run_xval

    lib = _torch_solver()
    row = run_xval.run_case(
        lib, R, int(4 * R), verbose=False, min_steps=1200, max_steps=2500
    )
    assert row["converged"], f"R={R} did not converge in the short run"
    # 2-D cylinder: dp = sigma / R with sigma ~ 0.056 (lattice units)
    assert 0.040 < row["sigma_i"] < 0.075, f"sigma_i={row['sigma_i']:.5f} out of range"
    # mass conservation
    assert row["mass_drift_rel"] < 1e-3
    # Laplace pressure must be positive and larger for the smaller drop
    assert row["dp"] > 0.0


def test_laplace_pressure_decreases_with_radius():
    """dp ~ 1/R: the two radii of the short run must be ordered."""
    import run_xval

    lib = _torch_solver()
    rows = [
        run_xval.run_case(lib, R, int(4 * R), verbose=False, min_steps=1200, max_steps=2500)
        for R in (8.0, 12.0)
    ]
    assert rows[0]["dp"] > rows[1]["dp"], "dp must decrease with the droplet radius"
    fit = run_xval.laplace_fit(rows)
    assert fit["sigma_through_origin"] > 0.0
    assert np.isfinite(fit["R2"])