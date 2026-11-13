"""Cross-validation tests: TensorLBM 2:1 buffer-layer refinement vs haimingz/vivsim.

The reference is the *vendored upstream source* under
``benchmarks/xval_buffer_layer_2to1/oss_ref`` executed verbatim through a tiny
NumPy stand-in for ``jax.numpy`` (``.at[idx].set``).  The sha256 of the vendored
upstream file is pinned, so a silent edit of the reference is caught here.

These tests are the L1 evidence for ``XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT``:
they assert *invariants* (identity where the formulations must agree, a
predicted factor where they intentionally differ, conservation and finiteness in
the solver, and the <15 % cross-arm gate), not merely that code runs.
"""

from __future__ import annotations

import hashlib
import importlib

import numpy as np
import pytest

from tensorlbm import block_refinement_2to1 as blk
from tensorlbm.amr_cavity2d import CavityAMR2D, CavityAMR2DConfig, _equilibrium

OSS_REF = "benchmarks.xval_buffer_layer_2to1.oss_ref"


def _oss():
    return importlib.import_module(f"{OSS_REF}.vivsim_multigrid")


def _shim():
    return importlib.import_module(f"{OSS_REF}.np_jax_shim")


def _oss_pkg():
    return importlib.import_module(OSS_REF)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def test_vendored_oss_source_matches_pinned_sha256():
    pkg = _oss_pkg()
    got = hashlib.sha256(pkg.UPSTREAM_PATH.read_bytes()).hexdigest()
    assert got == pkg.UPSTREAM_SHA256
    assert len(pkg.UPSTREAM_COMMIT) == 40
    assert pkg.UPSTREAM_FILE == "vivsim/multigrid.py"
    assert pkg.UPSTREAM_LICENSE == "MIT"


def test_shim_import_is_the_only_change():
    """The shimmed body must equal upstream apart from the import block."""
    pkg = _oss_pkg()
    upstream = pkg.UPSTREAM_PATH.read_text().splitlines()
    shimmed = pkg.SHIMMED_PATH.read_text().splitlines()
    up_import = upstream.index("from .lbm.basic import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS")
    sh_import = shimmed.index("from .lattice_dirs import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS")
    # upstream the two imports are adjacent (jax.numpy, blank line, direction table)
    assert upstream[up_import - 2].startswith("import jax.numpy as jnp")
    assert shimmed[sh_import - 1].startswith("from . import np_jax_shim as jnp")
    # every line after the import block is byte-identical
    assert shimmed[sh_import + 1:] == upstream[up_import + 1:]


# ---------------------------------------------------------------------------
# relaxation ladder (must agree exactly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu0", [0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
@pytest.mark.parametrize("level", [-2, -1, 0, 1, 2])
def test_omega_matches_executed_oss_get_omega(nu0, level):
    oss = _oss()
    ref = float(oss.get_omega(nu0, level))
    # TensorLBM: the level-l lattice viscosity is nu0 * 2**level, tau = 3 nu + 1/2
    got = 1.0 / (3.0 * (nu0 * 2.0 ** level) + 0.5)
    assert got == pytest.approx(ref, rel=1e-12)
    # the ported helper must reproduce the executed reference as well
    assert blk.get_omega(nu0, level) == pytest.approx(ref, rel=1e-14)


# ---------------------------------------------------------------------------
# transfer index algebra (must agree bit-for-bit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_fine_to_coarse_face_is_identical_to_executed_oss(direction):
    oss, shim = _oss(), _shim()
    rng = np.random.default_rng(7)
    f_fine_up = rng.random((9, 8, 6)).astype(np.float32)      # upstream (q, nx, ny)
    f_coarse_up = rng.random((9, 4, 3)).astype(np.float32)
    got = blk.fine_to_coarse_face(
        np.transpose(f_fine_up, (0, 2, 1)), np.transpose(f_coarse_up, (0, 2, 1)), direction
    )
    ref = np.asarray(oss.fine_to_coarse(shim.asarray(f_fine_up), shim.asarray(f_coarse_up), direction))
    assert np.array_equal(got, np.transpose(ref, (0, 2, 1)))


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_coarse_to_fine_face_is_identical_to_executed_oss(direction):
    oss, shim = _oss(), _shim()
    rng = np.random.default_rng(11)
    f_fine_up = rng.random((9, 8, 6)).astype(np.float32)
    f_coarse_up = rng.random((9, 4, 3)).astype(np.float32)
    got = blk.coarse_to_fine_face(
        np.transpose(f_coarse_up, (0, 2, 1)), np.transpose(f_fine_up, (0, 2, 1)), direction
    )
    ref = np.asarray(oss.coarse_to_fine(shim.asarray(f_coarse_up), shim.asarray(f_fine_up), direction))
    assert np.array_equal(got, np.transpose(ref, (0, 2, 1)))


def test_face_operators_only_touch_the_crossing_populations():
    rng = np.random.default_rng(3)
    f_fine = rng.random((9, 8, 6)).astype(np.float32)
    f_coarse = rng.standard_normal((9, 4, 3)).astype(np.float32)
    for direction, dirs in (
        ("left", blk.LEFT_DIRS),
        ("right", blk.RIGHT_DIRS),
        ("up", blk.UP_DIRS),
        ("down", blk.DOWN_DIRS),
    ):
        out = blk.fine_to_coarse_face(f_fine, f_coarse, direction)
        mask = np.ones(9, dtype=bool)
        mask[dirs] = False
        # non-crossing populations are returned unchanged
        assert np.array_equal(out[mask], f_coarse[mask])


def test_shifted_ddf_convention_commutes_with_the_operators():
    """vivsim stores f_i = f_i^full - w_i; average and copy reproduce w exactly.

    Therefore the vendored (shifted) operators and the ported (full) operators
    transfer the same full-population field.
    """
    rng = np.random.default_rng(5)
    w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                 dtype=np.float32).reshape(9, 1, 1)
    f_full = rng.random((9, 8, 6)).astype(np.float32)
    coarse_full = rng.random((9, 4, 3)).astype(np.float32)
    shifted = f_full - w
    coarse_shifted = coarse_full - w
    for direction in ("left", "right", "up", "down"):
        got = blk.fine_to_coarse_face(f_full, coarse_full, direction)
        from_shifted = blk.fine_to_coarse_face(shifted, coarse_shifted, direction) + w
        assert np.allclose(got, from_shifted, atol=1e-6)
        got_c2f = blk.coarse_to_fine_face(coarse_full, f_full, direction)
        from_shifted_c2f = blk.coarse_to_fine_face(coarse_shifted, shifted, direction) + w
        assert np.allclose(got_c2f, from_shifted_c2f, atol=1e-6)


# ---------------------------------------------------------------------------
# the intended difference
# ---------------------------------------------------------------------------

def test_equilibrium_field_makes_both_schemes_identical():
    """FH rescaling is the identity on equilibrium data -> no scheme difference."""
    rng = np.random.default_rng(2)
    ny, nx = 8, 10
    rho = 1.0 + 0.01 * rng.standard_normal((ny, nx)).astype(np.float32)
    ux = 0.01 * rng.standard_normal((ny, nx)).astype(np.float32)
    uy = 0.01 * rng.standard_normal((ny, nx)).astype(np.float32)
    feq = _equilibrium(rho, ux, uy)[:, 3:4, 2:3]
    tau_c, tau_f = 0.6, 0.7
    fh = blk.block_repeat(feq + np.float32(tau_f / (2 * tau_c)) * (feq - feq))
    assert np.allclose(fh, blk.block_repeat(feq), atol=1e-7)


def test_non_equilibrium_difference_matches_the_fh_factor():
    """OSS copy keeps f_neq; FH rescales it by tau_f/(2 tau_c): ratio must be 2 tau_c/tau_f."""
    tau_c, tau_f = 0.518, 0.536
    ny, nx = 6, 6
    rho = np.ones((ny, nx), dtype=np.float32)
    ux = np.full((ny, nx), 0.02, dtype=np.float32)
    uy = np.zeros((ny, nx), dtype=np.float32)
    feq = _equilibrium(rho, ux, uy)
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32).reshape(9, 1, 1)
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32).reshape(9, 1, 1)
    w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                 dtype=np.float32).reshape(9, 1, 1)
    f = feq + 4.5 * w * (cx * cy) * np.float32(1e-3)
    fh = feq + np.float32(tau_f / (2 * tau_c)) * (f - feq)
    ratio = np.abs(f - feq).max() / np.abs(fh - feq).max()
    assert ratio == pytest.approx(2 * tau_c / tau_f, rel=1e-4)
    # and the two are materially different on the non-equilibrium part (>15 %)
    assert np.abs(f - fh).max() / np.abs(f - feq).max() > 0.15


# ---------------------------------------------------------------------------
# solver integration and conservation
# ---------------------------------------------------------------------------

def test_default_scheme_is_fh_and_invalid_scheme_is_rejected():
    assert CavityAMR2DConfig().interface_scheme == "fh"
    with pytest.raises(ValueError):
        CavityAMR2DConfig(interface_scheme="bogus")


@pytest.mark.parametrize("scheme", ["fh", "vivsim_copy"])
def test_amr_cavity_runs_conserves_mass_and_stays_finite(scheme):
    cfg = CavityAMR2DConfig(coarse=32, strip_coarse=8, re=100.0, u_lid=0.05,
                            reflux=True, interface_scheme=scheme)
    solver = CavityAMR2D(cfg)
    inv0 = solver.inventory()["total"]
    for _ in range(60):
        solver.step()
    inv1 = solver.inventory()["total"]
    prof = solver.centreline_profiles()
    assert np.isfinite(prof["u"]).all() and np.isfinite(prof["v"]).all()
    assert abs(inv1 - inv0) / inv0 < 1e-6
    assert np.abs(prof["u"]).max() <= cfg.u_lid * 1.5


def test_cross_arm_centreline_gate_under_15_percent():
    """The XVAL gate on the physical solution: FH vs OSS copy, same grid and steps."""
    from benchmarks.xval_buffer_layer_2to1.run_xval import CaseConfig, compare_profiles, run_arm

    cfg = CaseConfig(coarse=32, strip=8, re=100.0, u_lid=0.05, steps=600)
    fh = run_arm(cfg, "fh")
    oss = run_arm(cfg, "vivsim_copy")
    assert fh["finite"] and oss["finite"]
    cmp_ = compare_profiles(fh["profiles"], oss["profiles"], cfg.u_lid)
    assert cmp_["max_du_pct_of_ulid"] < 15.0
    assert cmp_["max_dv_pct_of_ulid"] < 15.0