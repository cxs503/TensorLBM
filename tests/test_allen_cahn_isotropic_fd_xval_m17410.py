"""XVAL m17410 — Allen-Cahn / CSF force chain vs lbmpy (D3Q19 isotropic operators).

Two layers of evidence:

* **Operator parity** — the D3Q19-isotropic Laplacian/gradient ported into
  ``tensorlbm.phasefield.isotropic_fd`` must reproduce, weight for weight, the
  stencils that lbmpy's ``phasefield_allen_cahn.derivatives`` derives.  The
  lbmpy assertions are skipped when lbmpy is not importable.
* **Force-chain parity** — on an identical phase field the TensorLBM chain
  ``mu grad(phi)`` must equal lbmpy's ``surface_tension_force``, and the
  seven-point family must *not* (that difference is the finding, so it is
  pinned as a regression bound rather than a pass/fail on exact equality).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

from tensorlbm.allen_cahn_lbm import free_energy_parameters
from tensorlbm.phasefield.free_energy import DoubleWellFreeEnergy, force_mu_grad_phi
from tensorlbm.phasefield.isotropic_fd import (
    FACE_DIAGONAL_OFFSETS,
    isotropic_gradient_3d,
    isotropic_laplacian_3d,
)
from tensorlbm.phasefield.operators import central_gradient_3d, laplacian_3d

INTERIOR = (slice(2, -2),) * 3


def _grid(n: int = 12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, y, x = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for _ in range(3)), indexing="ij"
    )
    return z, y, x


def test_face_diagonal_offsets_are_the_twelve_d3q19_diagonals():
    assert len(FACE_DIAGONAL_OFFSETS) == 12
    assert len(set(FACE_DIAGONAL_OFFSETS)) == 12
    for off in FACE_DIAGONAL_OFFSETS:
        assert sorted(abs(c) for c in off) == [0, 1, 1]


def test_isotropic_gradient_is_exact_on_a_linear_field():
    z, y, x = _grid()
    field = 2.0 * x + 3.0 * y - 5.0 * z
    gx, gy, gz = isotropic_gradient_3d(field, boundary="periodic")
    assert torch.allclose(gx[INTERIOR], torch.full_like(gx[INTERIOR], 2.0), atol=1e-12)
    assert torch.allclose(gy[INTERIOR], torch.full_like(gy[INTERIOR], 3.0), atol=1e-12)
    assert torch.allclose(gz[INTERIOR], torch.full_like(gz[INTERIOR], -5.0), atol=1e-12)


def test_isotropic_laplacian_is_exact_on_a_quadratic_field():
    z, y, x = _grid()
    lap = isotropic_laplacian_3d(x * x + y * y + z * z, boundary="periodic")
    assert torch.allclose(lap[INTERIOR], torch.full_like(lap[INTERIOR], 6.0), atol=1e-11)


def test_isotropic_laplacian_annihilates_constants_and_has_zero_weight_sum():
    field = torch.full((8, 8, 8), 0.37, dtype=torch.float64)
    lap = isotropic_laplacian_3d(field, boundary="periodic")
    assert float(lap.abs().max()) < 1e-13


def test_no_flux_boundary_mirrors_instead_of_wrapping():
    field = torch.zeros((6, 6, 6), dtype=torch.float64)
    field[-1, :, :] = 1.0
    periodic = isotropic_laplacian_3d(field, boundary="periodic")
    mirrored = isotropic_laplacian_3d(field, boundary="no_flux")
    # the periodic roll leaks the slab to the first plane; the mirrored one does not
    assert float(periodic[0].abs().max()) > 0.1
    assert float(mirrored[0].abs().max()) < 1e-12


def test_operators_coincide_on_a_one_dimensional_field():
    """A one-dimensionally varying field is stencil-family agnostic (measured, C1)."""
    x = torch.arange(32, dtype=torch.float64)
    phi_1d = 0.5 * (1.0 + torch.tanh(2.0 * (x - 16.0) / 4.0))
    field = phi_1d.view(1, 1, -1).expand(4, 4, 32).contiguous()
    lap7 = laplacian_3d(field, boundary="periodic")
    lap_iso = isotropic_laplacian_3d(field, boundary="periodic")
    assert float((lap7 - lap_iso).abs().max()) < 1e-12
    g7 = central_gradient_3d(field, boundary="periodic")
    g_iso = isotropic_gradient_3d(field, boundary="periodic")
    for a, b in zip(g7, g_iso):
        assert float((a - b).abs().max()) < 1e-12


def test_default_operator_is_unchanged_central7():
    phi = torch.linspace(-1.0, 1.0, steps=5, dtype=torch.float64).view(1, 1, -1)
    phi = phi.expand(6, 6, 5).contiguous()
    params = free_energy_parameters(sigma=1e-2, W=2.0)
    fe = DoubleWellFreeEnergy(A=params.beta, B=params.beta, kappa=params.kappa)
    explicit = fe.chemical_potential(phi, boundary="periodic", operator="central7")
    default = fe.chemical_potential(phi, boundary="periodic")
    assert torch.equal(explicit, default)
    fx_explicit, _, _ = force_mu_grad_phi(phi, explicit, boundary="periodic", operator="central7")
    fx_default, _, _ = force_mu_grad_phi(phi, default, boundary="periodic")
    assert torch.equal(fx_explicit, fx_default)


# --------------------------------------------------------------------------------------
# lbmpy parity (skipped when the reference implementation is unavailable)
# --------------------------------------------------------------------------------------
lbmpy = pytest.importorskip("lbmpy", reason="reference implementation (lbmpy) not installed")


def _lbmpy_operators(box: int = 5):
    import sympy as sp
    from lbmpy.phasefield_allen_cahn.derivatives import (
        isotropic_gradient_symbolic,
        laplacian_symbolic,
    )
    from lbmpy.stencils import LBStencil
    from pystencils import Field

    field = Field.create_generic("phi", spatial_dimensions=3)
    stencil = LBStencil("D3Q19")
    sym2off = {field.center.get_shifted(*off): tuple(off) for off in stencil.stencil_entries}

    def evaluator(expr):
        syms = sorted(expr.free_symbols, key=lambda s: sym2off[s])
        fn = sp.lambdify(syms, expr, "numpy")

        def evaluate(arr):
            args = []
            for sym in syms:
                ox, oy, oz = sym2off[sym]
                args.append(arr if (ox, oy, oz) == (0, 0, 0)
                            else np.roll(arr, (-oz, -oy, -ox), axis=(0, 1, 2)))
            return np.asarray(fn(*args), dtype=np.float64)

        return evaluate

    delta = np.zeros((box, box, box))
    delta[(2, 2, 2)] = 1.0
    lap = evaluator(laplacian_symbolic(field, stencil))(delta)
    grad = evaluator(isotropic_gradient_symbolic(field, stencil)[0])(delta)
    return lap, grad, delta


def test_isotropic_weights_match_lbmpy_exactly():
    lap_lb, grad_lb, delta = _lbmpy_operators()
    phi = torch.as_tensor(delta)
    lap_tl = isotropic_laplacian_3d(phi, boundary="periodic").numpy()
    grad_tl = isotropic_gradient_3d(phi, boundary="periodic")[0].numpy()
    assert np.allclose(lap_lb, lap_tl, atol=1e-15)
    assert np.allclose(grad_lb, grad_tl, atol=1e-15)
    # the reference stencil is (centre, axis, face diagonal) = (-4, 1/3, 1/6)
    assert lap_lb[2, 2, 2] == pytest.approx(-4.0)
    assert lap_lb[2, 2, 3] == pytest.approx(1.0 / 3.0)
    assert lap_lb[2, 3, 3] == pytest.approx(1.0 / 6.0)


def test_force_chain_matches_lbmpy_on_a_droplet():
    """XVAL decision metric: rel-L2 of the CSF force vs lbmpy, isotropic port."""

    from lbmpy.phasefield_allen_cahn.kernel_equations import (
        chemical_potential_symbolic,
        surface_tension_force,
    )
    from lbmpy.phasefield_allen_cahn.parameter_calculation import AllenCahnParameters
    from lbmpy.stencils import LBStencil
    from pystencils import Field
    import sympy as sp

    box, radius, W_lb, sigma = 40, 10.0, 4.0, 1e-2
    zz, yy, xx = np.meshgrid(
        *(np.arange(box, dtype=np.float64) - 0.5 * (box - 1) for _ in range(3)), indexing="ij"
    )
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    phi_lb = 0.5 * (1.0 - np.tanh(2.0 * (r - radius) / W_lb))

    params = AllenCahnParameters(
        density_heavy=1.0, density_light=0.1, dynamic_viscosity_heavy=0.1,
        dynamic_viscosity_light=0.01, surface_tension=sigma, interface_thickness=int(W_lb),
    )
    field = Field.create_generic("phi", spatial_dimensions=3)
    stencil = LBStencil("D3Q19")
    sym2off = {field.center.get_shifted(*off): tuple(off) for off in stencil.stencil_entries}
    subs = params.parameter_map()

    def evaluator(expr):
        expr = sp.sympify(expr).subs(subs)
        syms = sorted(expr.free_symbols, key=lambda s: sym2off[s])
        fn = sp.lambdify(syms, expr, "numpy")

        def evaluate(arr):
            args = []
            for sym in syms:
                ox, oy, oz = sym2off[sym]
                args.append(arr if (ox, oy, oz) == (0, 0, 0)
                            else np.roll(arr, (-oz, -oy, -ox), axis=(0, 1, 2)))
            return np.asarray(fn(*args), dtype=np.float64)

        return evaluate

    f_lb = [evaluator(e)(phi_lb) for e in surface_tension_force(field, stencil, params)]

    phi_tl = torch.as_tensor(2.0 * phi_lb - 1.0)
    chain = free_energy_parameters(sigma=sigma, W=W_lb / 2.0)  # W_TL = W_lb / 2
    fe = DoubleWellFreeEnergy(A=chain.beta, B=chain.beta, kappa=chain.kappa)
    mu = fe.chemical_potential(phi_tl, boundary="periodic", operator="isotropic_d3q19")
    fx, fy, fz = force_mu_grad_phi(phi_tl, mu, boundary="periodic", operator="isotropic_d3q19")
    f_tl = np.stack([fz.numpy(), fy.numpy(), fx.numpy()])
    # lbmpy returns (fx, fy, fz); the arrays above are laid out (z, y, x)
    f_ref = np.stack([f_lb[2], f_lb[1], f_lb[0]])

    rel = np.linalg.norm(f_tl - f_ref) / np.linalg.norm(f_ref)
    assert rel < 1e-12, f"isotropic force chain deviates from lbmpy by {rel:.3e}"

    # the seven-point family does deviate: this is the finding, not a failure mode
    mu7 = fe.chemical_potential(phi_tl, boundary="periodic", operator="central7")
    fx7, fy7, fz7 = force_mu_grad_phi(phi_tl, mu7, boundary="periodic", operator="central7")
    f7 = np.stack([fz7.numpy(), fy7.numpy(), fx7.numpy()])
    rel7 = np.linalg.norm(f7 - f_ref) / np.linalg.norm(f_ref)
    assert 0.01 < rel7 < 0.30, f"expected an order-10% anisotropy error, got {rel7:.3e}"


def test_chemical_potential_scales_by_four_under_the_convention_map():
    """mu_TL = mu_lb / 2 and grad_TL = 2 grad_lb, so beta_TL = beta_lb/4, kappa_TL = kappa_lb/4."""
    sigma, W_lb = 1e-2, 4.0
    tl = free_energy_parameters(sigma=sigma, W=W_lb / 2.0)
    from lbmpy.phasefield_allen_cahn.parameter_calculation import AllenCahnParameters

    params = AllenCahnParameters(
        density_heavy=1.0, density_light=0.1, dynamic_viscosity_heavy=0.1,
        dynamic_viscosity_light=0.01, surface_tension=sigma, interface_thickness=int(W_lb),
    )
    subs = params.parameter_map()
    assert tl.beta == pytest.approx(float(sp.sympify(params.beta).subs(subs)) / 4.0, rel=1e-12)
    assert tl.kappa == pytest.approx(float(sp.sympify(params.kappa).subs(subs)) / 4.0, rel=1e-12)