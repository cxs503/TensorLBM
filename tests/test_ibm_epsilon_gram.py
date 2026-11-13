"""Unit tests for the LIFE epsilon-Gram IBM force-transfer transplant.

Anchor: XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION

All references are exact/analytic (no hand-copied numbers):

* the LIFE 3-point regularised delta is a 1-D partition of unity at every
  fractional offset, hence a separable 2-D partition of unity;
* for non-overlapping marker supports the Gram system ``A eps = 1`` makes the
  spread -> interpolate round trip the identity map, which gives an exact
  reference (2.0) to test against;
* the control arm (eps = 1) deposits exactly ``sum_m f_m ds_m`` -- exact
  bookkeeping that the corrected arm deliberately trades away.
"""
import math

import torch

from tensorlbm.ibm_epsilon_gram import (
    build_spread_matrix_2d,
    build_support_2d,
    compute_ds_2d,
    compute_epsilon_gram,
    interpolate_force_2d,
    interpolate_force_2d_vec,
    life_correction,
    life_delta_3pt,
    life_epsilon_gram_diagnostics,
    make_life_ibm_stencil,
    selftest,
    spread_force_2d,
    spread_force_2d_vec,
    spread_with_correction,
)

DT = torch.float64


def _ring(n: int, r: float = 10.0, c: float = 32.0, aligned: bool = False):
    th = torch.arange(n, dtype=DT) * (2.0 * math.pi / n)
    x = c + r * torch.cos(th)
    y = c + r * torch.sin(th)
    if aligned:
        x, y = torch.round(x), torch.round(y)
    return x, y


def test_delta_partition_of_unity_exact():
    """sum_i phi(|r - i|) == 1 for all fractional offsets (analytic)."""
    for frac in (0.0, 0.1, 0.25, 0.3, 0.5, 0.618, 0.75, 0.9):
        total = sum(
            float(life_delta_3pt(torch.tensor(abs(frac - i), dtype=DT)))
            for i in range(-5, 6)
        )
        assert abs(total - 1.0) < 1e-14, (frac, total)


def test_delta_support_and_moments():
    """Zero outside |r| <= 1.5, symmetric, finite at the kink."""
    r = torch.tensor([0.0, 0.5, 0.5000001, 1.5, 1.5000001, 2.0, -0.5], dtype=DT)
    phi = life_delta_3pt(r)
    assert float(phi[4]) == 0.0 and float(phi[5]) == 0.0
    assert abs(float(phi[1]) - float(phi[6])) < 1e-15  # symmetric
    assert abs(float(phi[2]) - float(phi[1])) < 1e-5  # continuous at r = 0.5
    # first moment vanishes by symmetry
    xs = torch.arange(-6, 7, dtype=DT)
    assert abs(float((life_delta_3pt(xs) * xs).sum())) < 1e-14


def test_support_weights_sum_to_one():
    """2-D support weight sum == 1 for arbitrary (non-aligned) markers."""
    for n, aligned in ((16, True), (16, False), (24, False)):
        x, y = _ring(n, aligned=aligned)
        diag = life_epsilon_gram_diagnostics(torch.stack([x, y], dim=1), (64, 64))
        assert abs(diag["dirac_sum_min"] - 1.0) < 1e-13
        assert abs(diag["dirac_sum_max"] - 1.0) < 1e-13


def test_support_does_not_wrap_periodically():
    """Markers near the boundary lose support nodes instead of wrapping."""
    x = torch.tensor([0.5, 32.0], dtype=DT)
    y = torch.tensor([0.5, 32.0], dtype=DT)
    supports = build_support_2d(x, y, 64, 64)
    assert len(supports[0]["w"]) < len(supports[1]["w"])
    assert int(supports[0]["ix"].min()) >= 0
    assert int(supports[0]["iy"].min()) >= 0


def test_gram_roundtrip_identity_nonoverlapping():
    """A1 (Gram) round trip is the identity; A2 (eps=1) is off by ~2.5%."""
    x, y = _ring(16)
    supports = build_support_2d(x, y, 64, 64)
    ds = compute_ds_2d(x, y)
    eps = compute_epsilon_gram(x, y, supports, ds)
    fx = torch.ones(16, dtype=DT)
    fy = torch.zeros(16, dtype=DT)
    errs = {}
    for arm, e in (("A1", eps), ("A2", torch.ones_like(ds))):
        gx, gy = spread_force_2d(fx, fy, supports, e, ds, 64, 64)
        bx, by = interpolate_force_2d(gx, gy, supports)
        errs[arm] = float((torch.cat([bx - fx, by - fy]).norm() / torch.cat([fx, fy]).norm()))
    assert errs["A1"] < 1e-14, errs
    assert errs["A2"] > 1e-2, errs


def test_deposit_bookkeeping_control_exact_and_cost_of_correction():
    """eps=1 deposits exactly sum f*ds; the correction inflates it by O(eps-1)."""
    x, y = _ring(24)
    supports = build_support_2d(x, y, 64, 64)
    ds = compute_ds_2d(x, y)
    eps = compute_epsilon_gram(x, y, supports, ds)
    fx = torch.ones(24, dtype=DT)
    fy = torch.zeros(24, dtype=DT)
    for e, tol in ((torch.ones_like(ds), 1e-14), (eps, None)):
        gx, _ = spread_force_2d(fx, fy, supports, e, ds, 64, 64)
        dev = abs(float(gx.sum()) - float(ds.sum())) / float(ds.sum())
        if tol is not None:
            assert dev < tol, dev
        else:
            # the trade-off is real and bounded by the epsilon spread
            assert 0.0 < dev < float(eps.max()) , (dev, float(eps.max()))


def test_vectorised_and_stencil_paths_match_loop():
    """Vectorised and IBMStencil paths reproduce the loop reference exactly."""
    x, y = _ring(9, r=6.0, c=32.0)
    ny, nx = 64, 64
    supports = build_support_2d(x, y, ny, nx)
    ds = compute_ds_2d(x, y)
    eps = compute_epsilon_gram(x, y, supports, ds)
    torch.manual_seed(0)
    fx = torch.randn(9, dtype=DT)
    fy = torch.randn(9, dtype=DT)

    gx1, gy1 = spread_force_2d(fx, fy, supports, eps, ds, ny, nx)
    S, _ = build_spread_matrix_2d(x, y, ny, nx)
    gx2, gy2 = spread_force_2d_vec(fx, fy, S, eps, ds, ny, nx)
    st = make_life_ibm_stencil(torch.stack([x, y], dim=1), (ny, nx))
    g3 = spread_with_correction(st, fx, fy, eps, ds)

    assert float((gx1 - gx2).abs().max()) < 1e-14
    assert float((gy1 - gy2).abs().max()) < 1e-14
    assert float((gx2 - g3[0]).abs().max()) < 1e-14
    assert float((gy2 - g3[1]).abs().max()) < 1e-14

    bx1, by1 = interpolate_force_2d(gx1, gy1, supports)
    bx2, by2 = interpolate_force_2d_vec(gx1, gy1, S)
    bi = st.interpolate(torch.stack([gx1, gy1]))
    assert float((bx1 - bx2).abs().max()) < 1e-14
    assert float((by1 - by2).abs().max()) < 1e-14
    assert float((bx2 - bi[0]).abs().max()) < 1e-14
    assert float((by2 - bi[1]).abs().max()) < 1e-14


def test_stencil_layout_and_marker_count():
    x, y = _ring(12, aligned=True)
    st = make_life_ibm_stencil(torch.stack([x, y], dim=1), (64, 64))
    assert st.n_markers == 12
    assert st.shape == (64, 64)
    assert st.kernel == "life3"
    assert int(st.idx.min()) >= 0 and int(st.idx.max()) < 64 * 64
    # zero padding keeps every marker's support rectangular
    assert st.w.numel() == 12 * (st.w.numel() // 12)


def test_life_correction_shapes():
    pts = torch.stack(_ring(10), dim=1)
    eps, ds = life_correction(pts, (64, 64))
    assert eps.shape == ds.shape == (10,)
    assert bool((ds > 0).all()) and bool((eps > 0).all())


def test_module_selftest_passes():
    res = selftest(verbose=False)
    bad = {k: v for k, v in res["pass"].items() if not v}
    assert not bad, bad