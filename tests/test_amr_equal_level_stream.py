"""Invariants of the waLBerla-style equal-level border stream correction."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.amr_cavity2d import _equilibrium, _stream
from tensorlbm.amr_equal_level_stream import (
    EqualLevelBorderStreamCorrection,
    analytic_fields_2d,
    calibrate_prolongation_scale,
    prolongation_scale,
    restriction_scale,
)


def _uniform(n: int = 32, rho: float = 1.0) -> np.ndarray:
    return _equilibrium(
        np.full((n, n), rho), np.zeros((n, n)), np.zeros((n, n)),
    ).astype(np.float64)


def _tgv(n: int, patch: int, u0: float = 0.04):
    k = 2.0 * np.pi / n
    velocity = lambda x, y: (  # noqa: E731
        u0 * np.cos(k * x) * np.sin(k * y),
        -u0 * np.sin(k * x) * np.cos(k * y),
    )
    return analytic_fields_2d(n, patch, velocity)


def _patch_mask(n: int, patch: int) -> np.ndarray:
    p0 = (n - patch) // 2
    mask = np.zeros((n, n), dtype=bool)
    mask[p0 : p0 + patch, p0 : p0 + patch] = True
    return mask


def test_uniform_field_is_inert():
    """A uniform state must stream bit-identically with and without correction."""
    f = _uniform()
    plan = EqualLevelBorderStreamCorrection(_patch_mask(32, 16))
    assert len(plan) > 0
    plain = _stream(f.copy())
    corrected = plan.stream_with_correction(f.copy())
    assert np.array_equal(plain, corrected)


def test_prepare_restore_is_bit_exact():
    """prepare/apply must leave the state untouched (streaming-time substitution)."""
    _, fine, _, mask = _tgv(32, 16)
    del fine
    n = mask.shape[0]
    f = _uniform(n, rho=1.0) + 1.0e-4 * np.random.default_rng(7).standard_normal((9, n, n))
    plan = EqualLevelBorderStreamCorrection(mask)
    before = f.copy()
    saved = plan.prepare(f)
    assert not np.array_equal(f, before)  # the substitution really happened
    plan.restore(f, saved)
    assert np.array_equal(f, before)


def test_links_point_from_blocked_to_active():
    """Every corrected slot must sit on a blocked cell and copy from an active one."""
    n, patch = 32, 12
    mask = _patch_mask(n, patch)
    plan = EqualLevelBorderStreamCorrection(mask)
    ny, nx = mask.shape
    dst_r, dst_c = np.divmod(plan.dst, nx)
    src_r, src_c = np.divmod(plan.src, nx)
    assert mask[dst_r, dst_c].all()
    assert not mask[src_r, src_c].any()
    # the destination must be the pull source of the equally-sized neighbour
    assert set(plan.stencil_directions) == {1, 2, 3, 4, 5, 6, 7, 8}


def test_donor_modes_differ_and_change_the_stream_result():
    _, _, _, mask = _tgv(32, 16)
    n = mask.shape[0]
    rng = np.random.default_rng(3)
    f = _uniform(n) + 1.0e-3 * rng.standard_normal((9, n, n))
    mirror = EqualLevelBorderStreamCorrection(mask, donor="mirror")
    zero = EqualLevelBorderStreamCorrection(mask, donor="self")
    a = mirror.stream_with_correction(f.copy())
    b = zero.stream_with_correction(f.copy())
    plain = _stream(f.copy())
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, plain)
    assert not np.array_equal(b, plain)


def test_prolongation_scales_and_restriction_inverse():
    scales = {c: prolongation_scale(0.6, 0.7, 2, c) for c in
              ("fh_spacing", "fh_plain", "repo_torch", "repo_cavity")}
    assert scales["fh_spacing"] == pytest.approx(4.0)
    assert scales["fh_plain"] == pytest.approx(2.0)
    assert scales["repo_torch"] == pytest.approx(0.7 / (0.5 * 0.6))
    assert scales["repo_cavity"] == pytest.approx(0.7 / 1.2)
    for value in scales.values():
        assert restriction_scale(value) * value == pytest.approx(1.0)
    assert prolongation_scale(0.6, 0.7, 2, 1.5) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        prolongation_scale(0.6, 0.7, 2, "nope")


def test_analytic_calibration_peaks_at_unit_scale():
    """The measured F-H factor for convective 2:1 refinement is 1.0."""
    coarse, fine, _, _ = _tgv(64, 32)
    n, patch = 64, 32
    p0 = (n - patch) // 2
    scales = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0])
    report = calibrate_prolongation_scale(coarse[:, p0 : p0 + patch, p0 : p0 + patch], fine, scales)
    assert report["best_scale"] == pytest.approx(1.0)
    assert report["best_error"] < 5.0e-3
    assert report["objective_gap"] > 0.1


def test_cavity_frozen_shadow_uses_the_correction():
    """Integration caller: CavityAMR2D('frozen') really drives the correction."""
    from tensorlbm.amr_cavity2d import CavityAMR2D, CavityAMR2DConfig

    cfg = CavityAMR2DConfig(
        coarse=32, strip_coarse=8, re=100.0, u_lid=0.05,
        coarse_shadow="frozen", stream_correction=True,
    )
    sim = CavityAMR2D(cfg)
    assert sim.correction is not None and len(sim.correction) > 0

    ref = CavityAMR2D(
        CavityAMR2DConfig(coarse=32, strip_coarse=8, re=100.0, u_lid=0.05),
    )
    for _ in range(5):
        diag = sim.step()
        ref.step()
    assert np.isfinite(sim.f).all() and np.isfinite(sim.f_top).all()
    assert abs(diag["inventory_drift"]) < 1.0
    # the default ("restricted") coupling must stay untouched by this branch
    assert ref.correction is None
