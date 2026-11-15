"""Analytic identity tests for the 3-D Peskin tensor-product stencil + MDF.

These lock the *closed-form* properties the immersed-boundary coupling must
satisfy (xval ``XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF``):

* partition of unity of the tensor-product stencil,
* zero first lattice moment (linear consistency),
* exact reproduction of a linear velocity field,
* Lagrangian/Eulerian adjointness of spread and interpolate,
* exact total-momentum conservation of the spreading operator,
* multi-direct forcing: reaction/grid-force closure is exact for every
  ``n_iter`` and the slip residual contracts monotonically.

The second-moment (grid-anisotropy) spread is a *characterisation* assertion:
no 4-point kernel makes ``sum_s w (x_i - x_m)^2`` independent of the marker
offset, which is why the coupling is second order but not exact.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import ibm3d_mdf as mdf

torch.set_grad_enabled(False)

DTYPE = torch.float64
SHAPE = (16, 18, 20)


def _markers(n: int = 256, seed: int = 11) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    nz, ny, nx = SHAPE
    lo = torch.tensor([3.0, 3.0, 3.0], dtype=DTYPE)
    hi = torch.tensor([nx - 3.0, ny - 3.0, nz - 3.0], dtype=DTYPE)
    return lo + torch.rand((n, 3), generator=gen, dtype=DTYPE) * (hi - lo)


def _fields():
    nz, ny, nx = SHAPE
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=DTYPE),
        torch.arange(ny, dtype=DTYPE),
        torch.arange(nx, dtype=DTYPE),
        indexing="ij",
    )
    return zz, yy, xx


@pytest.mark.parametrize("kernel", ["peskin4pt", "peskin3pt", "cosine4pt"])
def test_partition_of_unity(kernel: str) -> None:
    markers = _markers()
    w, _ = mdf.build_stencil(markers, SHAPE, kernel=kernel)
    assert (w.sum(dim=1) - 1.0).abs().max() < 1e-12


@pytest.mark.parametrize("kernel,tol", [("peskin4pt", 1e-12), ("peskin3pt", 1e-12),
                                        ("cosine4pt", 5e-2)])
def test_first_lattice_moment(kernel: str, tol: float) -> None:
    """Peskin's kernels satisfy ``sum_s w (x_s - x_m) = 0`` exactly.

    The plain 4-point cosine kernel does *not* (measured 2.1e-2 here): it is a
    partition of unity but not first-moment consistent, which is why the
    peskin kernels are the default for this coupling.
    """
    markers = _markers()
    w, idx = mdf.build_stencil(markers, SHAPE, kernel=kernel)
    nz, ny, nx = SHAPE
    iz = idx // (ny * nx)
    iy = (idx % (ny * nx)) // nx
    ix = idx % nx
    for axis, index in enumerate((ix, iy, iz)):
        dx = index.to(DTYPE) - markers[:, axis: axis + 1]
        assert (w * dx).sum(dim=1).abs().max() < tol


def test_linear_field_is_reproduced_exactly() -> None:
    markers = _markers()
    w, idx = mdf.build_stencil(markers, SHAPE, kernel="peskin4pt")
    zz, yy, xx = _fields()
    lin = torch.stack(
        [
            1.0 + 0.5 * xx + 2.0 * yy - 0.25 * zz,
            -3.0 + 0.7 * xx - 1.1 * yy + 0.9 * zz,
            0.25 * xx + 0.3 * yy + 4.0,
        ]
    )
    got = mdf.interpolate(lin, w, idx)
    ref = torch.stack(
        [
            1.0 + 0.5 * markers[:, 0] + 2.0 * markers[:, 1] - 0.25 * markers[:, 2],
            -3.0 + 0.7 * markers[:, 0] - 1.1 * markers[:, 1] + 0.9 * markers[:, 2],
            0.25 * markers[:, 0] + 0.3 * markers[:, 1] + 4.0,
        ],
        dim=1,
    )
    assert (got - ref).abs().max() < 1e-12


def test_spread_interpolate_adjointness_and_momentum_conservation() -> None:
    markers = _markers()
    w, idx = mdf.build_stencil(markers, SHAPE, kernel="peskin4pt")
    _, _, xx = _fields()
    u = torch.stack([torch.sin(0.3 * xx).expand(SHAPE).clone() for _ in range(3)])
    gen = torch.Generator().manual_seed(5)
    q = torch.randn((markers.shape[0], 3), generator=gen, dtype=DTYPE)

    spread_q = mdf.spread(q, w, idx, SHAPE)
    interp_u = mdf.interpolate(u, w, idx)
    lhs = float((spread_q * u).sum())
    rhs = float((q * interp_u).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-12

    # the spreading operator conserves total momentum component-wise
    for c in range(3):
        assert abs(float(spread_q[c].sum()) - float(q[:, c].sum())) < 1e-10 * max(
            1.0, abs(float(q[:, c].sum()))
        )


@pytest.mark.parametrize("kernel", ["peskin4pt", "peskin3pt", "cosine4pt"])
def test_second_moment_is_offset_dependent_but_bounded(kernel: str) -> None:
    """Characterisation: the 4-point kernels are not second-moment invariant.

    ``sum_s w (x_i - x_m)^2`` varies with the fractional marker offset, which
    bounds the coupling to second-order accuracy (grid anisotropy).  The test
    pins the measured spread so a kernel change cannot silently make it worse.
    """
    markers = _markers(n=512, seed=3)
    w, idx = mdf.build_stencil(markers, SHAPE, kernel=kernel)
    nz, ny, nx = SHAPE
    ix = idx % nx
    dx = ix.to(DTYPE) - markers[:, 0:1]
    m2 = (w * dx * dx).sum(dim=1)
    spread = float(m2.max() - m2.min())
    assert 0.0 < spread < 0.12


def test_mdf_closure_and_contraction() -> None:
    from tensorlbm.ibm3d_mdf import multi_direct_forcing

    centre = (SHAPE[2] / 2.0, SHAPE[1] / 2.0, SHAPE[0] / 2.0)
    verts, faces = mdf.sphere_surface(centre, 3.0, n_theta=16, n_phi=10)
    ds = mdf.surface_weights(verts, faces)
    w, idx = mdf.build_stencil(verts, SHAPE, kernel="peskin4pt")
    _, _, xx = _fields()
    grid_u = torch.stack([0.01 * xx, 0.02 * torch.sin(0.4 * xx), 0.0 * xx])
    u_target = torch.zeros((verts.shape[0], 3), dtype=DTYPE)

    previous = None
    for n_iter in (1, 2, 3, 5, 8):
        res = multi_direct_forcing(grid_u, w, idx, u_target, ds, n_iter=n_iter)
        # reaction and grid force are exact negatives for every iteration count
        gf = float(res.grid_force[0].sum())
        mr = float(res.marker_reaction[:, 0].sum())
        assert abs(gf + mr) <= 1e-10 * max(1.0, abs(mr))
        if previous is not None:
            assert res.residuals[-1] < previous
        previous = res.residuals[-1]
        assert math.isfinite(previous)


def test_mdf_fixed_point_removes_the_slip() -> None:
    from tensorlbm.ibm3d_mdf import multi_direct_forcing

    centre = (SHAPE[2] / 2.0, SHAPE[1] / 2.0, SHAPE[0] / 2.0)
    verts, faces = mdf.sphere_surface(centre, 3.0, n_theta=16, n_phi=10)
    ds = mdf.surface_weights(verts, faces)
    w, idx = mdf.build_stencil(verts, SHAPE, kernel="peskin4pt")
    _, _, xx = _fields()
    grid_u = torch.stack([0.01 * xx, 0.0 * xx, 0.0 * xx])
    u_target = torch.zeros((verts.shape[0], 3), dtype=DTYPE)

    res = multi_direct_forcing(grid_u, w, idx, u_target, ds, n_iter=40)
    # The iteration contracts geometrically (ratio ~0.75) and then stalls at a
    # grid-limited floor ~4e-3: the discrete spread/interp pair has a null space,
    # so MDF cannot enforce no-slip below the resolution of the Eulerian grid.
    assert res.residuals[-1] < 1e-2
    assert res.residuals[5] > res.residuals[-1]  # still contracting at n_iter=6