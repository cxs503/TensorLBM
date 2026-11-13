"""XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF -- regression gate.

Pins the four claims the XVAL report rests on:

1. the 1-D Peskin 4-point kernel is *identical* in TensorLBM and in the vendored
   haimingz/vivsim reference (``oss_ref/kernels.py``),
2. the tensor-product stencils are identical node-by-node in 2-D and 3-D,
3. TensorLBM's new ``direct_forcing_mdf`` reproduces the OSS ``multi_direct_forcing``
   converged marker force (the port), and
4. the pre-existing pointwise ``direct_forcing`` does *not* reach the no-slip
   interface condition that the OSS MDF reaches -- i.e. the defect the port fixes
   is real, so this test fails if somebody silently "fixes" it by loosening the
   MDF port instead.

Run: ``PYTHONPATH=src pytest -q tests/test_xval_peskin_tensor_product_stencil_mdf.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

XVAL_DIR = Path(__file__).resolve().parents[1] / "xval" / "XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF"
sys.path.insert(0, str(XVAL_DIR))

import oss_ref  # noqa: E402
from oss_ref import np_jax_shim as shim  # noqa: E402

from tensorlbm.ibm import ibm_delta_4pt  # noqa: E402
from tensorlbm.ibm_force_transfer import build_stencil, direct_forcing, direct_forcing_mdf  # noqa: E402

RADIUS = 8.0
NY = NX = 64
UX = 0.05


def circle_markers(h: float) -> np.ndarray:
    n = max(8, int(round(2.0 * math.pi * RADIUS / h)))
    t = np.arange(n) * (2.0 * math.pi / n)
    return np.stack([32.13 + RADIUS * np.cos(t), 31.87 + RADIUS * np.sin(t)], axis=1)


def oss_mdf(pts: np.ndarray, n_iter: int):
    """Run the vendored OSS stencil + MDF on a uniform u=(UX, 0) field."""
    w, idx = oss_ref.get_ib_stencil(shim.asarray(pts[:, 0]), shim.asarray(pts[:, 1]), NY)
    ux = np.full((NY, NX), UX)
    grid_u = shim.asarray(np.stack([ux.T, np.zeros((NY, NX)).T]))
    u_target = shim.asarray(np.zeros((len(pts), 2)))
    ds = np.asarray(oss_ref.get_ds(shim.asarray(pts)))
    gf, _reaction = oss_ref.multi_direct_forcing(grid_u, w, idx, u_target, shim.asarray(ds), n_iter=n_iter)
    return np.asarray(gf)  # (2, nx, ny)


def tl_forcing(pts: np.ndarray, n: int, scheme: str):
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), (NY, NX), "4pt")
    uy = torch.zeros((NY, NX), dtype=torch.float64)
    ux = torch.full((NY, NX), UX, dtype=torch.float64)
    u_target = torch.zeros((len(pts), 2), dtype=torch.float64)
    if scheme == "pointwise":
        f_m, f_grid, _, _ = direct_forcing(st, torch.stack([uy, ux]), u_target, None, n_correctors=n)
    else:
        f_m, f_grid, _, _ = direct_forcing_mdf(st, torch.stack([uy, ux]), u_target, None, n_mdf=n)
    # interface residual of the Guo half-force condition
    u_after = st.interpolate(torch.stack([uy, ux])).T + 0.5 * st.interpolate(f_grid).T
    return f_m, f_grid, float(u_after.abs().max()) / UX


def test_peskin_4pt_kernel_identical():
    r = np.linspace(-2.5, 2.5, 2001)
    assert np.abs(np.asarray(oss_ref.kernel_peskin_4pt(shim.asarray(r))) - ibm_delta_4pt(torch.tensor(r)).numpy()).max() < 1e-14


def test_stencil_identity_2d():
    pts = circle_markers(0.9)
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), (NY, NX), "4pt")
    w = st.w.numpy().reshape(st.n_markers, -1)
    ws = np.sort(w, axis=1)
    oss_w, _ = oss_ref.get_ib_stencil(shim.asarray(pts[:, 0]), shim.asarray(pts[:, 1]), NY)
    oss_sorted = np.sort(np.asarray(oss_w), axis=1)
    assert np.abs(ws - oss_sorted).max() < 1e-14
    assert st.peak_normalization() == pytest.approx(1.0, abs=1e-14)


def test_stencil_identity_3d():
    import oss_ref.stencil3d as s3d

    n = 40
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rxy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    pts = np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], axis=1) * 5.0 + 16.3
    shape = (32, 32, 32)
    st = build_stencil(torch.tensor(pts, dtype=torch.float64), shape, "4pt")
    w = np.sort(st.w.numpy().reshape(st.n_markers, -1), axis=1)
    oss_w, _ = s3d.get_ib_stencil(shim.asarray(pts), shape)
    assert np.abs(w - np.sort(np.asarray(oss_w), axis=1)).max() < 1e-13


def test_mdf_port_matches_oss_converged_force():
    pts = circle_markers(0.9)
    oss = oss_mdf(pts, 40)[0].sum()
    _fm, fg, slip = tl_forcing(pts, 20, "mdf")
    assert abs(float(fg[1].sum()) - float(oss)) / abs(float(oss)) < 1e-4
    assert slip < 0.02


def test_pointwise_forcing_does_not_reach_no_slip():
    """The defect the port fixes: pointwise direct_forcing leaves >15 % slip."""
    pts = circle_markers(0.9)
    _fm, fg_point, slip_point = tl_forcing(pts, 10, "pointwise")
    _fm2, fg_mdf, slip_mdf = tl_forcing(pts, 10, "mdf")
    assert slip_point > 0.15
    assert slip_mdf < 0.02
    # and the marker force magnitude differs accordingly
    assert abs(float(fg_point[1].sum())) < 0.7 * abs(float(fg_mdf[1].sum()))