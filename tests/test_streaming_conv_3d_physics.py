"""3D physics checks for the conv-as-streaming transport (`XVAL-...-R4`).

CI-sized companions of ``benchmarks/xval_conv_as_streaming/run_3d_physics.py``.
They assert invariants, not just "it runs":

* the 3D wall operator (`streaming_conv.bounce_blend`) is *bit-wise* the explicit
  direction reversal, including on a mask that has corners (this is the gap that
  made ``bounce_blend`` unusable for 3D before round 4: it was hard-wired to
  ``conv2d`` and raised on a 4D field);
* the 2D wall operator is unchanged by that fix (regression guard);
* the 3D transport pair (gather vs fixed kernel) is bit-wise identical;
* the analytic square-duct judge reproduces the published ``f*Re = 56.908`` and
  satisfies the Poisson equation, i.e. the judge itself is falsifiable;
* a short 3D slit-channel run through both arms lands inside the owner gate
  against the exact parabola, and both arms agree bit-wise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_conv_as_streaming"))

import run_3d_physics as R3  # noqa: E402
from tensorlbm import streaming_conv as SC  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE as OP2  # noqa: E402
from tensorlbm.d3q19 import OPPOSITE as OP3  # noqa: E402


# --------------------------------------------------------------- wall operator
def test_bounce_blend_3d_matches_index_reversal_bitwise():
    torch.manual_seed(11)
    mask = torch.zeros(5, 6, 7, dtype=torch.float64)
    mask[0, :, :] = 1.0
    mask[-1, :, :] = 1.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0  # corners: a flat single wall never covers this
    pre = torch.rand(19, 5, 6, 7, dtype=torch.float64)
    post = torch.rand(19, 5, 6, 7, dtype=torch.float64)
    got = SC.bounce_blend(post, pre, mask, opposite=OP3)
    ref = post.clone()
    ref[:, mask > 0] = pre[OP3][:, mask > 0]
    assert got.shape == post.shape
    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_bounce_blend_3d_accepts_rank4_mask():
    torch.manual_seed(12)
    mask = torch.zeros(1, 4, 5, 6, dtype=torch.float64)
    mask[:, 0, :, :] = 1.0
    pre = torch.rand(19, 4, 5, 6, dtype=torch.float64)
    post = torch.rand(19, 4, 5, 6, dtype=torch.float64)
    assert torch.equal(
        SC.bounce_blend(post, pre, mask, opposite=OP3),
        SC.bounce_blend(post, pre, mask[0], opposite=OP3),
    )


def test_bounce_blend_2d_unchanged():
    torch.manual_seed(13)
    mask = torch.zeros(6, 7, dtype=torch.float64)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    pre = torch.rand(9, 6, 7, dtype=torch.float64)
    post = torch.rand(9, 6, 7, dtype=torch.float64)
    got = SC.bounce_blend(post, pre, mask, opposite=OP2)
    ref = post.clone()
    ref[:, mask > 0] = pre[OP2][:, mask > 0]
    assert torch.equal(got, ref)


def test_bounce_blend_rejects_mismatched_mask_rank():
    f = torch.zeros(19, 4, 5, 6, dtype=torch.float64)
    with pytest.raises(ValueError):
        SC.bounce_blend(f, f, torch.zeros(4, 5), opposite=OP3)  # 2D mask, 3D field
    with pytest.raises(ValueError):
        SC.bounce_blend(torch.zeros(4, dtype=torch.float64), f, f)


# -------------------------------------------------------------- transport pair
def test_3d_gather_vs_fixed_kernel_bitwise():
    torch.manual_seed(14)
    f = torch.rand(19, 5, 6, 7, dtype=torch.float64)
    assert torch.equal(R3.stream_gather_3d(f), SC.stream_conv_3d(f))


# ------------------------------------------------------------- analytic judges
def test_square_duct_series_reproduces_published_fRe():
    check = R3.duct_series_selfcheck()
    assert check["judge_ok"], check
    assert abs(check["fRe_series"] - 56.908) / 56.908 < 1e-3
    assert check["poisson_mean_rel_err_pct"] < 1.0
    # textbook max/mean velocity ratio of a square duct
    assert abs(check["centre_ratio_series"] - 2.096) < 5e-3


def test_stokes_mode_initial_condition_is_divergence_free_and_laplacian_eigenmode():
    n, amp = 16, 1e-3
    _, ux, uy, uz = R3.stokes_mode_initial(n, amp)
    k = 2.0 * np.pi / n
    # central differences on the periodic cube: the continuum field is divergence-free,
    # and the centred operator cancels the two cos*cos*cos factors exactly
    div = (
        0.5 * (np.roll(ux, -1, axis=2) - np.roll(ux, 1, axis=2))
        + 0.5 * (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        + 0.5 * (np.roll(uz, -1, axis=0) - np.roll(uz, 1, axis=0))
    )
    assert np.abs(div).max() < 1e-9 * amp * k
    lap = (
        np.roll(ux, 1, axis=2) - 2 * ux + np.roll(ux, -1, axis=2)
        + np.roll(ux, 1, axis=1) - 2 * ux + np.roll(ux, -1, axis=1)
        + np.roll(ux, 1, axis=0) - 2 * ux + np.roll(ux, -1, axis=0)
    )
    # discrete Laplacian eigenvalue of a centred-mode field is -4 sin^2(k/2) summed over axes
    lam = -4.0 * (3.0 * np.sin(k / 2.0) ** 2)
    mask_inner = (slice(1, -1),) * 3
    rel = np.abs(lap[mask_inner] / (lam * ux[mask_inner]) - 1.0).max()
    assert rel < 1e-10


# --------------------------------------------------------------- physics smoke
def test_slit3d_arms_agree_and_hit_the_parabola():
    runs = [
        R3.run_slit3d(10, 2, 2, 0.8, 0.02, 4000, arm, sample_every=1000)
        for arm in R3.ARMS
    ]
    a, b = runs
    assert a["f_sha256"] == b["f_sha256"]
    assert a["finite"] and b["finite"]
    for r in runs:
        assert r["profile_l2_pct"] < 15.0, r["profile_l2_pct"]
        assert r["half_run_drift"] < 5e-2, r["half_run_drift"]