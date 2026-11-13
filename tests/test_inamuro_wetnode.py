"""Tests for the Inamuro wet-node / anti-bounce-back closures (D2Q9).

These are *property* tests of the operator (the thing the XVAL validates against
the OSS binary); the numeric cross-validation itself lives in
``benchmarks/verified/inamuro_wetnode_xval/run.py``.
"""

from __future__ import annotations

import torch

from tensorlbm.boundary.d2q9 import C2Q9, OPPOSITE_2Q9, W2Q9
from tensorlbm.boundary.inamuro_wetnode import (
    inamuro_corner_average,
    inamuro_wetnode_scalar,
    inamuro_wetnode_velocity,
    inamuro_zero_gradient,
    unknown_directions,
)
from tensorlbm.boundary.solver2d import equilibrium2d, macroscopic2d

DT = torch.float64


def _feq(rho, ux, uy) -> torch.Tensor:
    return equilibrium2d(rho.to(DT), ux.to(DT), uy.to(DT))


def _uniform(ny: int = 6, nx: int = 5, rho: float = 1.0):
    r = torch.full((ny, nx), rho, dtype=DT)
    z = torch.zeros((ny, nx), dtype=DT)
    return _feq(r, z, z)


# ---------------------------------------------------------------------------
# direction sets
# ---------------------------------------------------------------------------


def test_unknown_directions_follow_the_face_normal():
    # TLBM D2Q9: 1=E 2=W 3=N 4=S 5=NE 6=NW 7=SE 8=SW
    assert unknown_directions("left").tolist() == [1, 5, 7]
    assert unknown_directions("right").tolist() == [2, 6, 8]
    assert unknown_directions("bottom").tolist() == [3, 5, 6]
    assert unknown_directions("top").tolist() == [4, 7, 8]


def test_invalid_face_rejected():
    f = _uniform()
    for call in (lambda: inamuro_wetnode_scalar(f, "sideways", 1.0),
                 lambda: inamuro_zero_gradient(f, "sideways")):
        try:
            call()
        except ValueError as exc:
            assert "face" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for an unknown face")
    try:
        unknown_directions("up")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# scalar Dirichlet / zero-gradient closures
# ---------------------------------------------------------------------------


def test_scalar_dirichlet_is_mass_consistent():
    """Node value becomes exactly the target (the upstream executed closure)."""
    f = _uniform(rho=1.4)
    for face in ("left", "right", "bottom", "top"):
        g = inamuro_wetnode_scalar(f, face, 0.25)
        sl = {"left": (slice(None), 0), "right": (slice(None), -1),
              "bottom": (0, slice(None)), "top": (-1, slice(None))}[face]
        vals = g[:, sl[0], sl[1]].sum(dim=0)
        assert torch.allclose(vals, torch.full_like(vals, 0.25), atol=1e-14)


def test_scalar_dirichlet_rest_excluded_variant_leaves_rest_population():
    """The variant in upstream's dead ``update_bounce_back()`` keeps f0."""
    f = _uniform()
    g = inamuro_wetnode_scalar(f, "left", 0.5, known="all_but_rest")
    f0 = f[0, :, 0]
    vals = g[:, :, 0].sum(dim=0)
    assert torch.allclose(vals - f0, torch.full_like(vals, 0.5), atol=1e-12)
    # ... and it is measurably not the Dirichlet value
    assert not torch.allclose(vals, torch.full_like(vals, 0.5))


def test_zero_gradient_leaves_a_uniform_field_unchanged():
    f = _uniform(rho=0.7)
    for face in ("left", "right", "bottom", "top"):
        g = inamuro_zero_gradient(f, face)
        assert torch.allclose(g, f, atol=1e-14), face


def test_closure_only_touches_its_own_face():
    f = _uniform()
    g = inamuro_wetnode_scalar(f, "left", 3.0)
    assert torch.equal(g[:, :, 1:], f[:, :, 1:])


# ---------------------------------------------------------------------------
# velocity anti-bounce-back
# ---------------------------------------------------------------------------


def test_abb_leaves_a_wall_consistent_equilibrium_unchanged():
    """f = f^eq(rho_w, u_w) on the wall ⇒ the ABB closure must be the identity.

    This is the defining exactness property of the anti-bounce-back rule:
    f_ī = f_ī^eq + f_i^eq − f_i collapses to f_ī^eq when f = f^eq at the wall
    state.  Checked for all faces and for both the quadratic and the linear
    (OSS MATLAB) equilibrium.
    """
    ny, nx = 5, 6
    for face in ("left", "right", "bottom", "top"):
        for (uxw, uyw) in ((0.0, 0.0), (0.1, 0.0), (0.0, 0.07), (0.03, -0.02)):
            for eq in ("quadratic", "linear"):
                r = torch.ones(ny, nx, dtype=DT)
                ux = torch.full((ny, nx), uxw, dtype=DT)
                uy = torch.full((ny, nx), uyw, dtype=DT)
                if eq == "quadratic":
                    f = _feq(r, ux, uy)
                else:  # the OSS MATLAB scripts' linear incompressible equilibrium
                    cu = (C2Q9[:, 0].to(DT).view(-1, 1, 1) * ux.unsqueeze(0)
                          + C2Q9[:, 1].to(DT).view(-1, 1, 1) * uy.unsqueeze(0))
                    f = W2Q9.to(DT).view(-1, 1, 1) * r.unsqueeze(0) * (1.0 + 3.0 * cu)
                g = inamuro_wetnode_velocity(
                    f, face, (uxw, uyw), rho_w=1.0, equilibrium=eq
                )
                assert torch.allclose(g, f, atol=1e-12), (face, uxw, uyw, eq)


def test_abb_on_a_rest_field_with_a_rest_wall_is_the_identity():
    f = _uniform()
    for face in ("left", "right", "bottom", "top"):
        g = inamuro_wetnode_velocity(f, face, 0.0)
        assert torch.allclose(g, f, atol=1e-13), face


def test_abb_imposes_the_wall_velocity_on_the_boundary_node():
    """A converged-looking state: reconstructing the wall row gives u = u_wall.

    Consists of a field that is already the equilibrium of a shear flow; at the
    wall row the closure must return the wall velocity (within the O(u^3)
    compressibility of the quadratic equilibrium).
    """
    ny, nx = 7, 3
    y = torch.arange(ny, dtype=DT) / (ny - 1)
    ux = 0.1 * y.view(-1, 1).expand(ny, nx).contiguous()
    f = _feq(torch.ones(ny, nx, dtype=DT), ux, torch.zeros(ny, nx, dtype=DT))
    g = inamuro_wetnode_velocity(f, "top", (0.1, 0.0), rho_w=1.0)
    rho, gx, _ = macroscopic2d(g)
    assert abs(float(gx[-1, 0]) - 0.1) < 5e-3
    g2 = inamuro_wetnode_velocity(f, "bottom", (0.0, 0.0), rho_w=1.0)
    _, gx2, _ = macroscopic2d(g2)
    assert abs(float(gx2[0, 0])) < 5e-3


def test_abb_rho_extrapolation_matches_fixed_rho_on_a_uniform_field():
    f = _uniform(rho=1.0)
    a = inamuro_wetnode_velocity(f, "top", (0.05, 0.0), rho_w=None)
    b = inamuro_wetnode_velocity(f, "top", (0.05, 0.0), rho_w=1.0)
    assert torch.allclose(a, b, atol=1e-13)


# ---------------------------------------------------------------------------
# corner average / bookkeeping
# ---------------------------------------------------------------------------


def test_corner_average_matches_the_upstream_block():
    ny, nx = 4, 5
    f = torch.arange(9 * ny * nx, dtype=DT).reshape(9, ny, nx)
    g = inamuro_corner_average(f, "bl")
    assert torch.allclose(g[:, 0, 0], 0.5 * (f[:, 0, 1] + f[:, 1, 0]))
    g = inamuro_corner_average(f, "tr")
    assert torch.allclose(g[:, ny - 1, nx - 1], 0.5 * (f[:, ny - 1, nx - 2] + f[:, ny - 2, nx - 1]))


def test_reports_are_populated():
    f = _uniform()
    _, rep = inamuro_wetnode_scalar(f, "left", 0.5, return_report=True)
    assert rep.directions == (1, 5, 7)
    assert abs(rep.value_after - 0.5) < 1e-13
    assert rep.closure.startswith("scalar-dirichlet")
    _, rep2 = inamuro_wetnode_velocity(f, "bottom", (0.02, 0.0), return_report=True)
    assert rep2.directions == (3, 5, 6)
    assert rep2.closure == "velocity-abb-quadratic"


def test_inplace_semantics():
    f = _uniform()
    out = inamuro_wetnode_scalar(f, "top", 2.0, inplace=True)
    assert out.data_ptr() == f.data_ptr()
    g = f.clone()
    out2 = inamuro_wetnode_scalar(g, "top", 2.0)
    assert out2.data_ptr() != g.data_ptr()


def test_lattice_pairing_is_consistent():
    """The opposite map used by the closures is an involution on the lattice."""
    for k in range(9):
        j = int(OPPOSITE_2Q9[k])
        assert int(OPPOSITE_2Q9[j]) == k
        assert torch.equal(C2Q9[j], -C2Q9[k])
        assert abs(float(W2Q9[j] - W2Q9[k])) < 1e-15