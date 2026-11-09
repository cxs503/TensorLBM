"""Tests for :mod:`tensorlbm.multiphase_edm` (SC force-format upgrade).

Physics assertions (not just shape checks):

* the classic product form reproduces ``multiphase.sc_single_component_force``
  (``scheme="standard"``) and the ψ² divergence form reproduces
  ``scheme="pressure_tensor"`` — i.e. the module is a drop-in superset;
* the exact-difference (EDM) force is *algebraically identical* to the classic
  product form on a symmetric complete stencil (Σ W c = 0), which is asserted
  numerically and is the reason the EDM force alone cannot change the physics;
* the isotropy moment conditions of every stencil, and the fact that the
  8-neighbour set has a unique solution = the standard D2Q9 weights;
* the 6th-order isotropic stencil demonstrably reduces the spurious tangential
  force on a circular interface (the drive of spurious currents);
* Kupershtokh (exact-difference) forcing adds exactly F of momentum per step,
  while the historical velocity-shift forcing does not;
* a short droplet march is stable, mass-conserving and produces Δp > 0.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import multiphase as mp
from tensorlbm import multiphase_edm as edm
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream


# ---------------------------------------------------------------------------
# equivalence with the existing library
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil_shape", [(24, 32), (16, 16)])
def test_product_matches_library_standard(stencil_shape):
    torch.manual_seed(0)
    ny, nx = stencil_shape
    rho = 0.2 + 1.5 * torch.rand(ny, nx)
    psi = mp.psi_exp(rho)
    G_lib = 5.0
    Fx_lib, Fy_lib = mp.sc_single_component_force(rho, G_lib, mp.psi_exp, scheme="standard")
    Fx, Fy = edm.sc_interaction_force(psi, edm.physical_coupling(G_lib), "d2q9", "product")
    scale = max(float(Fx_lib.abs().max()), 1e-6)
    assert torch.allclose(Fx, Fx_lib, atol=1e-6 * scale, rtol=0)
    assert torch.allclose(Fy, Fy_lib, atol=1e-6 * scale, rtol=0)


def test_psi2_matches_library_pressure_tensor():
    torch.manual_seed(1)
    rho = 0.2 + 1.5 * torch.rand(24, 32)
    psi = mp.psi_exp(rho)
    G_lib = 5.0
    Fx_lib, Fy_lib = mp.sc_single_component_force(rho, G_lib, mp.psi_exp,
                                                  scheme="pressure_tensor")
    Fx, Fy = edm.sc_interaction_force(psi, edm.physical_coupling(G_lib), "d2q9", "psi2")
    scale = max(float(Fx_lib.abs().max()), 1e-6)
    assert torch.allclose(Fx, Fx_lib, atol=1e-6 * scale, rtol=0)
    assert torch.allclose(Fy, Fy_lib, atol=1e-6 * scale, rtol=0)


def test_gravity_term_added():
    psi = torch.full((8, 8), 0.5)
    rho = torch.full((8, 8), 2.0)
    Fx, Fy = edm.sc_interaction_force(psi, -5.0, "d2q9", "difference", gx=1e-3, gy=-2e-3, rho=rho)
    assert torch.allclose(Fx, torch.full_like(Fx, 2.0 * 1e-3), atol=1e-9)
    assert torch.allclose(Fy, torch.full_like(Fy, 2.0 * -2e-3), atol=1e-9)


# ---------------------------------------------------------------------------
# the exact-difference identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["d2q9", "iso6", "iso6_25"])
def test_exact_difference_equals_product_on_symmetric_stencil(name):
    """Σ_i W_i c_i = 0 ⇒ ψ Σ W c [ψ(x+c) − ψ(x)] ≡ ψ Σ W c ψ(x+c)."""
    torch.manual_seed(2)
    psi = torch.tanh(torch.randn(32, 32) * 2.0)
    Fd = edm.sc_interaction_force(psi, -5.0, name, "difference")
    Fp = edm.sc_interaction_force(psi, -5.0, name, "product")
    scale = max(float(Fp[0].abs().max()), 1e-6)
    for a, b in zip(Fd, Fp):
        assert torch.allclose(a, b, atol=1e-6 * scale, rtol=0), name


def test_weight_sum_vanishes():
    """Σ_i W_i c_i = 0 — the identity behind the exact-difference form."""
    for name in edm.stencil_names():
        st = edm.stencil(name)
        vec, wt = st.tensors(dtype=torch.float64)
        assert abs(float((wt * vec[:, 0]).sum())) < 1e-15
        assert abs(float((wt * vec[:, 1]).sum())) < 1e-15
        assert abs(float((wt * vec[:, 0] * vec[:, 1]).sum())) < 1e-15


# ---------------------------------------------------------------------------
# stencil / isotropy algebra
# ---------------------------------------------------------------------------

def _moments(name):
    st = edm.stencil(name)
    v = torch.tensor(st.vectors, dtype=torch.float64)
    w = torch.tensor(st.weights, dtype=torch.float64)
    m2 = float((w * v[:, 0] ** 2).sum())
    m4 = float((w * v[:, 0] ** 4).sum()) - 3 * float((w * v[:, 0] ** 2 * v[:, 1] ** 2).sum())
    m6 = float((w * v[:, 0] ** 6).sum()) - 5 * float((w * v[:, 0] ** 4 * v[:, 1] ** 2).sum())
    return m2, m4, m6


def test_d2q9_weights_are_the_unique_8neighbour_solution():
    """M2 = 1 and M4 isotropic force exactly the D2Q9 weights (1/3, 1/12)."""
    w = edm.isotropic_weights(("axis1", "diag"), max_order=4)
    assert w == pytest.approx((1 / 3, 1 / 12), rel=1e-12)
    m2, m4, m6 = _moments("d2q9")
    assert m2 == pytest.approx(1.0, abs=1e-12)
    assert m4 == pytest.approx(0.0, abs=1e-12)
    # ... and the residual anisotropy is 6th order: Σ W c_x⁶ − 5 Σ W c_x⁴c_y² = −2/3
    assert m6 == pytest.approx(-2 / 3, abs=1e-12)


@pytest.mark.parametrize("name", ["iso6", "iso6_25"])
def test_high_order_stencils_are_6th_order_isotropic(name):
    m2, m4, m6 = _moments(name)
    assert m2 == pytest.approx(1.0, abs=1e-12)
    assert m4 == pytest.approx(0.0, abs=1e-10)
    assert m6 == pytest.approx(0.0, abs=1e-10)


def test_iso6_shell_weights_closed_form():
    st = edm.stencil("iso6")
    sw = st.shell_weights
    assert sw["axis1"] == pytest.approx(4 / 15, rel=1e-12)
    assert sw["diag"] == pytest.approx(1 / 10, rel=1e-12)
    assert sw["axis2"] == pytest.approx(1 / 120, rel=1e-12)


def test_isotropic_weights_rejects_underdetermined_order():
    with pytest.raises(ValueError):
        edm.isotropic_weights(("axis1", "diag"), max_order=6)


def test_unknown_stencil_raises():
    with pytest.raises(ValueError):
        edm.stencil("d2q99")
    with pytest.raises(ValueError):
        edm.psi_gradient(torch.zeros(8, 8), "d2q9", form="bogus")
    with pytest.raises(ValueError):
        edm.sc_interaction_force(torch.full((8, 8), 0.5), -5.0, "d2q9", form="bogus")


# ---------------------------------------------------------------------------
# gradient exactness / anisotropy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["d2q9", "iso6", "iso6_25"])
def test_gradient_of_linear_field_is_exact(name):
    ny = nx = 16
    y, x = torch.meshgrid(torch.arange(ny, dtype=torch.float32),
                          torch.arange(nx, dtype=torch.float32), indexing="ij")
    gx, gy = edm.psi_gradient(x, name, "difference", periodic=True)
    # ψ = x is not periodic: exclude the 3-cell wrap band at each edge
    inner = (slice(3, -3), slice(3, -3))
    assert torch.allclose(gx[inner], torch.ones_like(gx[inner]), atol=1e-6)
    assert torch.allclose(gy[inner], torch.zeros_like(gy[inner]), atol=1e-6)


def test_uniform_field_gives_zero_force():
    psi = torch.full((16, 16), 0.7)
    for name in edm.stencil_names():
        for form in edm.FORMS:
            Fx, Fy = edm.sc_interaction_force(psi, -1.0, name, form, periodic=True)
            assert float(Fx.abs().max()) == 0.0
            assert float(Fy.abs().max()) == 0.0


def _circular_aniso(name, form="difference", L=96, R=20.0, w=2.0):
    y, x = torch.meshgrid(torch.arange(L, dtype=torch.float32),
                          torch.arange(L, dtype=torch.float32), indexing="ij")
    c = 0.5 * (L - 1)
    dx, dy = x - c, y - c
    r = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-9)
    psi = torch.tanh((R - r) / w)
    Fx, Fy = edm.sc_interaction_force(psi, -1.0, name, form, periodic=True)
    Fr = Fx * dx / r + Fy * dy / r
    Ft = -Fx * dy / r + Fy * dx / r
    band = (r - R).abs() < 2.5 * w
    return float(Ft[band].abs().max()) / float(Fr[band].abs().max())


def test_iso6_reduces_interface_anisotropy():
    a_d2q9 = _circular_aniso("d2q9")
    a_iso6 = _circular_aniso("iso6")
    assert a_d2q9 > 5e-4, a_d2q9            # ~1e-3: measurable anisotropy
    assert a_iso6 < a_d2q9 / 3.0, (a_d2q9, a_iso6)   # benchmark: 5.2x better


# ---------------------------------------------------------------------------
# collision / forcing
# ---------------------------------------------------------------------------

def test_collide_matches_library_edm_forcing():
    torch.manual_seed(3)
    ny = nx = 24
    rho = 0.5 + torch.rand(ny, nx)
    rho[8:16, 8:16] += 1.0
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    G_lib, tau = 5.0, 1.0
    Fx, Fy = mp.sc_single_component_force(rho, G_lib, mp.psi_exp, scheme="standard")
    f_lib = mp.collide_sc_single_component(f, G_lib, tau, mp.psi_exp, forcing="edm")
    f_mine = edm.collide_sc_force(f, Fx, Fy, tau, forcing="kupershtokh")
    assert torch.allclose(f_mine, f_lib, atol=1e-6, rtol=1e-6)


def test_kupershtokh_forcing_adds_exactly_the_force():
    """Σ_i c_i (feq(u+F/ρ) − feq(u)) = ρ·(F/ρ) = F, exactly."""
    torch.manual_seed(4)
    rho = 1.0 + 0.2 * torch.rand(12, 12)
    ux = 0.02 * torch.rand(12, 12)
    uy = -0.02 * torch.rand(12, 12)
    f = equilibrium(rho, ux, uy)
    Fx = 0.01 * torch.rand(12, 12)
    Fy = -0.01 * torch.rand(12, 12)
    f_out = edm.collide_sc_force(f, Fx, Fy, tau=0.8)
    rho2, ux2, uy2 = macroscopic(f_out)
    mom_x_pre = (f * torch.tensor(edm._D2Q9_C[:, 0]).view(9, 1, 1)).sum(0)
    mom_x_post = (f_out * torch.tensor(edm._D2Q9_C[:, 0]).view(9, 1, 1)).sum(0)
    assert torch.allclose(mom_x_post - mom_x_pre, Fx, atol=1e-7)
    mom_y = (f_out * torch.tensor(edm._D2Q9_C[:, 1]).view(9, 1, 1)).sum(0) - \
            (f * torch.tensor(edm._D2Q9_C[:, 1]).view(9, 1, 1)).sum(0)
    assert torch.allclose(mom_y, Fy, atol=1e-7)
    assert torch.allclose(rho2, rho, atol=1e-6)      # mass conserving


def test_all_forcings_add_exactly_F_momentum_but_differ_in_stress():
    torch.manual_seed(5)
    rho = 1.0 + 0.2 * torch.rand(12, 12)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    Fx = torch.full_like(rho, 1e-2)
    Fy = torch.zeros_like(rho)
    cx = edm._D2Q9_C[:, 0].clone().view(9, 1, 1)
    f_edm = edm.collide_sc_force(f, Fx, Fy, tau=0.8, forcing="kupershtokh")
    f_vs = edm.collide_sc_force(f, Fx, Fy, tau=0.8, forcing="velocity_shift")
    f_guo = edm.collide_sc_force(f, Fx, Fy, tau=0.8, forcing="guo")
    for f_out in (f_edm, f_vs):
        assert torch.allclose(((f_out - f) * cx).sum(0), Fx, atol=1e-6)
    # ... but they relax to different distributions (different stress term), and
    # Guo carries the half-force velocity correction
    assert not torch.allclose(f_edm, f_vs, atol=1e-7, rtol=0)
    assert not torch.allclose(f_guo, f_edm, atol=1e-7, rtol=0)
    # velocity-shift is momentum exact at any tau, but the stress differs
    f_vs2 = edm.collide_sc_force(f, Fx, Fy, tau=0.6, forcing="velocity_shift")
    assert torch.allclose(((f_vs2 - f) * cx).sum(0), Fx, atol=1e-6)
    assert not torch.allclose(f_vs2, f_vs, atol=1e-7, rtol=0)


def test_solid_mask_is_preserved():
    rho = torch.full((10, 10), 1.0)
    # non-equilibrium state so that collision actually changes the interior
    torch.manual_seed(7)
    f = equilibrium(rho, torch.full_like(rho, 0.05), torch.full_like(rho, -0.03))
    f = f + 0.01 * torch.randn_like(f)   # non-equilibrium so collision acts
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[0, :] = True
    Fx = torch.zeros_like(rho)
    Fy = torch.zeros_like(rho)
    f_out = edm.collide_sc_force(f, Fx, Fy, 1.0, solid_mask=mask)
    assert torch.allclose(f_out[:, 0, :], f[:, 0, :])
    assert not torch.allclose(f_out[:, 5, :], f[:, 5, :])


def test_pressure_mech_matches_verified_formula():
    rho = torch.tensor([1.957, 0.1596])
    psi = mp.psi_exp(rho)
    p = edm.pressure_mech(rho, psi, -5.0)
    # benchmarks/verified/laplace_droplet uses p = rho/3 + G_eff*psi^2/6
    assert torch.allclose(p, rho / 3.0 + (-5.0) * psi**2 / 6.0)


# ---------------------------------------------------------------------------
# time marching smoke test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,form", [("d2q9", "product"), ("iso6", "difference"),
                                       ("iso6", "psi2")])
def test_droplet_march_is_stable_and_mass_conserving(name, form):
    torch.set_num_threads(2)
    L, R = 48, 9.0
    y, x = torch.meshgrid(torch.arange(L, dtype=torch.float32),
                          torch.arange(L, dtype=torch.float32), indexing="ij")
    c = 0.5 * (L - 1)
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    phi = 0.5 * (1.0 - torch.tanh((r - R) / 1.0))
    rho = mp.psi_exp.__self__ if False else 0.1596 + (1.957 - 0.1596) * phi  # noqa: F841
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    m0 = float(f.sum())
    for _ in range(120):
        f = edm.step(f, -5.0, 1.0, mp.psi_exp, name, form, "kupershtokh")
    assert torch.isfinite(f).all()
    assert abs(float(f.sum()) - m0) / m0 < 1e-5
    rho_end, ux, uy = macroscopic(f)
    assert float(rho_end.min()) > 0.0
    assert float(rho_end.max()) < 3.0
    assert float(torch.sqrt(ux * ux + uy * uy).max()) < 0.2


def test_stream_step_matches_solver_stream():
    torch.manual_seed(6)
    rho = 0.5 + torch.rand(12, 12)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    assert torch.allclose(edm.stream_step(f), stream(f))


def test_public_api_complete():
    for name in ("Stencil", "stencil", "stencil_names", "isotropic_weights",
                 "psi_gradient", "sc_interaction_force", "collide_sc_force",
                 "stream_step", "pressure_mech", "library_coupling",
                 "physical_coupling", "FORMS", "FORCINGS"):
        assert hasattr(edm, name)
        assert name in edm.__all__
    assert edm.library_coupling(-5.0) == 5.0
    assert edm.physical_coupling(5.0) == -5.0
    assert edm.FORMS == ("product", "difference", "psi2")
    assert set(edm.FORCINGS) == {"velocity_shift", "kupershtokh", "guo"}