"""Regression tests for the curvilinear prism-band DG-LBM (tensorlbm.dg_curv).

These guard the exact properties that broke during the sphere-drag case study:
  * T1  affine operator reproduces the Cartesian band operator (identity geometry)
  * T2  DG advection is mass-conserving on a fully-periodic band
  * T3  the sphere prism topology is geometrically sane (wall normals radial, finite RHS)
  * T4  (CRITICAL) the curvilinear operator is conservative: a *uniform* field must
        give RHS ~ 0.  This is the property that was violated by the per-face
        area scaling bug and caused the coupled run to blow up.
"""
import torch
import pytest

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_band import build_band_topology, dg_rhs_band
from tensorlbm.dg_curv import (
    PrismGeometry,
    dg_rhs_band_geo,
    make_sphere_prism_topology,
)


@pytest.fixture
def dg_ops():
    return get_ops(degree=1, dx=1.0, dtype=torch.float64)


def _cart_topo():
    nz, ny, nx = 8, 16, 16
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[2:6, 6:10, 6:10] = True
    band = torch.zeros_like(solid)
    for ax in (0, 1, 2):
        band |= torch.roll(solid, 1, dims=ax) & ~solid
        band |= torch.roll(solid, -1, dims=ax) & ~solid
    band &= ~solid
    return build_band_topology(band, solid_mask=solid, periodic=False), (nz, ny, nx)


def test_t1_affine_reproduces_cartesian(dg_ops):
    topo, _ = _cart_topo()
    torch.manual_seed(0)
    f_dg = torch.rand(19, topo.n_band, 2, 2, 2, dtype=torch.float64)
    ext = torch.rand(19, 8 * 16 * 16, dtype=torch.float64)
    ref = dg_rhs_band(f_dg, C3D, dg_ops, topo, ext, opposite=OPP3D)

    n_b = topo.n_band
    contrav = torch.eye(3, dtype=torch.float64).flip(1).expand(n_b, 3, 3).clone()
    face_J = torch.ones(3, 2, n_b)
    n_phys = torch.zeros(3, 2, n_b, 3)
    for a in range(3):
        n_phys[a, 1, :, a] = 1.0
        n_phys[a, 0, :, a] = -1.0
    specular = OPP3D.unsqueeze(0).expand(n_b, 19).clone()
    geo = PrismGeometry(contrav=contrav, face_J=face_J, n_phys=n_phys,
                        specular=specular, detJ=torch.ones(n_b))
    got = dg_rhs_band_geo(f_dg, C3D, dg_ops, topo, geo, ext_field=ext)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, f"curvilinear operator must reproduce Cartesian, got {err}"


def test_t2_mass_conservation_periodic(dg_ops):
    nz, ny, nx = 8, 16, 16
    topo = build_band_topology(torch.ones(nz, ny, nx, dtype=torch.bool),
                               solid_mask=None, periodic=True)
    torch.manual_seed(0)
    f_dg = torch.rand(19, topo.n_band, 2, 2, 2, dtype=torch.float64) + 0.1
    geo = PrismGeometry(
        contrav=torch.eye(3, dtype=torch.float64).flip(1).expand(topo.n_band, 3, 3).clone(),
        face_J=torch.ones(3, 2, topo.n_band),
        n_phys=torch.zeros(3, 2, topo.n_band, 3),
        specular=OPP3D.unsqueeze(0).expand(topo.n_band, 19).clone(),
        detJ=torch.ones(topo.n_band),
    )
    rhs = dg_rhs_band_geo(f_dg, C3D, dg_ops, topo, geo, ext_field=None)
    s = rhs.sum().item()
    assert abs(s) < 1e-4, f"advection must conserve mass on periodic band, got {s}"


def test_t3_sphere_topology_geometry(dg_ops):
    nz, ny, nx = 32, 32, 64
    radius = 5.0
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    r = torch.sqrt(((torch.arange(nx) - cx) ** 2).reshape(1, 1, nx) +
                   ((torch.arange(ny) - cy) ** 2).reshape(1, ny, 1) +
                   ((torch.arange(nz) - cz) ** 2).reshape(nz, 1, 1))
    solid = r <= radius
    topo, geo, meta = make_sphere_prism_topology(
        solid, center=(cx, cy, cz), R=radius, n_layers=3, first_height=0.5,
        n_az=16, n_stream=12, polar_cap=0.985, vel=C3D,
        dtype=torch.float64, device="cpu")
    assert topo.n_band > 0
    assert torch.isfinite(geo.contrav).all()
    assert (geo.detJ > 0).all()
    assert (geo.face_J > 0).all()

    # wall normal must align with sphere radial direction
    cx_t = torch.tensor([cx, cy, cz], dtype=torch.float64)
    nrm = geo.n_phys[0, 1]  # radial + face normal
    center_b = (topo.band_coords[:, [2, 1, 0]].double() + 0.0)
    radial = (center_b - cx_t).double()
    radial = radial / (radial.norm(dim=-1, keepdim=True) + 1e-30)
    align = (nrm * radial).sum(dim=-1).abs().min().item()
    assert align > 0.9, f"wall normals must align with sphere radial, got {align}"

    f_dg = torch.rand(19, topo.n_band, 2, 2, 2, dtype=torch.float64) + 0.1
    ext = torch.rand(19, nz * ny * nx, dtype=torch.float64)
    rhs = dg_rhs_band_geo(f_dg, C3D, dg_ops, topo, geo, ext_field=ext)
    assert torch.isfinite(rhs).all(), "curvilinear sphere RHS must be finite"


def test_t4_uniform_field_conservative(dg_ops):
    """CRITICAL: a uniform field must give RHS ~ 0 (divergence-theorem balance).

    Regression guard for the per-face area scaling bug that injected energy into a
    uniform field and blew up the coupled run.
    """
    nz, ny, nx = 32, 32, 64
    radius = 5.0
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    r = torch.sqrt(((torch.arange(nx) - cx) ** 2).reshape(1, 1, nx) +
                   ((torch.arange(ny) - cy) ** 2).reshape(1, ny, 1) +
                   ((torch.arange(nz) - cz) ** 2).reshape(nz, 1, 1))
    solid = r <= radius
    topo, geo, _ = make_sphere_prism_topology(
        solid, center=(cx, cy, cz), R=radius, n_layers=1, first_height=1.0,
        n_az=24, n_stream=16, polar_cap=0.985, vel=C3D,
        dtype=torch.float64, device="cpu")
    f_dg = torch.full((19, topo.n_band, 2, 2, 2), 0.1, dtype=torch.float64)
    ext = torch.full((19, nz * ny * nx), 0.1, dtype=torch.float64)
    rhs = dg_rhs_band_geo(f_dg, C3D, dg_ops, topo, geo, ext_field=ext)
    err = float(rhs.abs().max().item())
    assert err < 1e-9, f"uniform field RHS must be ~0 for a conservative operator, got {err}"
