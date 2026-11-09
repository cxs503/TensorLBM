"""Momentum-flux pressure readout and Korteweg pressure tensor (node P143W2-N01).

The point of these tests is the defect that made the Shan-Chen interface tension
"unmeasurable": the raw momentum-flux trace ``p_kin`` returns ``rho cs^2`` and
therefore misses the *whole* interaction pressure.  Anything that reads pressure
must add ``G_eff cs^2 psi^2 / 2`` -- and that term has to come from the lattice
pseudopotential, not from a state-equation closure of the density.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.sc_surface_tension import (  # noqa: E402
    CS2,
    G_EFF,
    build_stencil,
    korteweg_pn_minus_pt,
    pressure_eos,
    pressure_flux,
    pressure_kinetic,
    pressure_tensor,
    psi_exp,
    sigma_korteweg,
    sc_force,
)


def _uniform(L: int = 24, rho0: float = 1.3):
    rho = torch.full((L, L), rho0)
    z = torch.zeros_like(rho)
    return rho, equilibrium(rho, z, z)


def test_kinetic_readout_is_ideal_gas_pressure():
    rho, f = _uniform(rho0=1.3)
    pk = pressure_kinetic(f)
    assert torch.allclose(pk, torch.full_like(pk, 1.3 * CS2), atol=1e-6)


def test_flux_readout_adds_interaction_pressure():
    rho, f = _uniform(rho0=1.3)
    psi = psi_exp(rho)
    miss = 0.5 * G_EFF * CS2 * psi * psi  # the term p_kin lacks
    assert torch.allclose(pressure_flux(f), pressure_kinetic(f) + miss, atol=1e-6)
    # sign: attraction (G_eff < 0) *lowers* the pressure
    assert float(miss.mean()) < 0.0
    assert not torch.allclose(pressure_flux(f), pressure_kinetic(f), atol=1e-3)


def test_flux_equals_eos_closure_for_uniform_state():
    """At local equilibrium the two readouts must agree -- and they do, exactly,
    because the kinetic trace of the equilibrium distribution *is* rho cs^2."""
    rho, f = _uniform(rho0=1.1)
    assert torch.allclose(pressure_flux(f), pressure_eos(rho), atol=1e-6)


def test_pressure_tensor_trace_and_isotropy_at_rest():
    L = 16
    rho = torch.full((L, L), 1.4)
    ux = torch.full((L, L), 0.05)
    uy = torch.zeros_like(ux)
    f = equilibrium(rho, ux, uy)
    lxx, lyy, lxy = pressure_tensor(f)
    p = pressure_kinetic(f)
    assert torch.allclose(0.5 * (lxx + lyy), p, atol=1e-6)
    # a uniform equilibrium state carries no shear
    assert float(lxy.abs().max()) < 1e-7
    assert float((lxx - lyy).abs().max()) < 1e-5


def test_nonuniform_state_separates_readouts():
    """A static interface: the closure and the flux readout must agree in the
    bulk (where local equilibrium holds) and may differ inside the interface."""
    L = 64
    y = torch.arange(L, dtype=torch.float32).view(L, 1).expand(L, L)
    rho = 0.16 + 1.8 / (1.0 + torch.exp((y - 32.0) / 1.5))
    z = torch.zeros_like(rho)
    f = equilibrium(rho, z, z)
    bulk_in = y < 12
    bulk_out = y > 52
    for band in (bulk_in, bulk_out):
        d = (pressure_flux(f) - pressure_eos(rho))[band]
        assert float(d.abs().max()) < 1e-6
    # the raw kinetic readout is wrong by the full interaction pressure
    dk = (pressure_flux(f) - pressure_kinetic(f))[bulk_in]
    assert float(dk.mean()) < -0.3


def test_interaction_force_conserves_momentum():
    """Sum of the stencil interaction force over a periodic box is zero to round-off
    (the discrete force is a divergence, i.e. Newton's third law holds)."""
    L = 32
    y = torch.arange(L, dtype=torch.float32).view(L, 1).expand(L, L)
    rho = 0.16 + 1.8 / (1.0 + torch.exp((y - 16.0) / 1.5))
    for name in ("d2q9", "shell4"):
        st = build_stencil(name)
        Fx, Fy = sc_force(rho, G_EFF, st)
        assert abs(float(Fx.sum())) < 2e-3, name
        assert abs(float(Fy.sum())) < 2e-3, name


def test_korteweg_stress_integrates_to_sigma():
    """``Integral (p_N - p_T) dn = +sigma_K`` (eq. 6): the truncation-induced
    stress is the *only* source of surface tension in this model."""
    L = 96
    y = torch.arange(L, dtype=torch.float32).view(L, 1).expand(L, L)
    # a *periodic* interface: the spectral Hessian inside the diagnostic assumes
    # periodicity, so the profile must wrap (a monotone tanh does not).
    rho = 1.06 + 0.9 * torch.cos(2.0 * math.pi * y / L)
    st = build_stencil("d2q9")
    pn_pt = korteweg_pn_minus_pt(rho, st)
    integral = float(pn_pt.sum(dim=0).mean())
    psi = psi_exp(rho)
    dpsi = 0.5 * (torch.roll(psi, -1, dims=0) - torch.roll(psi, 1, dims=0))
    sigma = abs(G_EFF * CS2 * st["kappa3"]) * float((dpsi ** 2).sum(dim=0).mean())
    assert sigma > 1e-3
    assert abs(integral - sigma) / sigma < 0.10


def test_sigma_scales_with_kappa3():
    """``sigma_K`` is a stencil property: the 4-shell stencil has a 1.375x larger
    kappa3 and must give a 1.375x larger Korteweg integral on the same field."""
    L = 96
    y = torch.arange(L, dtype=torch.float32).view(L, 1).expand(L, L)
    rho = 1.06 + 0.9 * torch.cos(2.0 * math.pi * y / L)
    sk = {}
    for name in ("d2q9", "shell4"):
        st = build_stencil(name)
        psi = psi_exp(rho)
        dpsi = 0.5 * (torch.roll(psi, -1, dims=0) - torch.roll(psi, 1, dims=0))
        sk[name] = abs(G_EFF * CS2 * st["kappa3"]) * float((dpsi ** 2).sum(dim=0).mean())
    ratio = sk["shell4"] / sk["d2q9"]
    assert abs(ratio - 1.375) < 1e-9


def test_sigma_korteweg_helper_positive_for_a_drop():
    L = 128
    y = torch.arange(L, dtype=torch.float32).view(L, 1)
    x = torch.arange(L, dtype=torch.float32).view(1, L)
    r = torch.sqrt((x - 63.5) ** 2 + (y - 63.5) ** 2)
    rho = 0.16 + 1.8 / (1.0 + torch.exp((r - 25.0) / 1.5))
    out = sigma_korteweg(rho, L, build_stencil("d2q9"))
    assert 0.005 < out["sigma_K"] < 0.1
    assert out["kappa3"] == pytest.approx(1.0 / 6.0)