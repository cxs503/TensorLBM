"""Fast CPU discriminators for the corrected-sign EOS pseudopotentials (M4).

Locks three contracts of ``tensorlbm.multiphase``:

1. the corrected-sign PR pseudopotential reconstructs the EOS mechanical
   pressure p_mech = rho*cs^2 - (G/2)*cs^2*psi^2 (G = +1) at the Maxwell
   coexistence pair, against equal-area construction lock values;
2. the ``forcing="velocity_shift"`` branch (and the default) of
   ``collide_sc_single_component`` is bit-identical to the pre-patch
   implementation inlined below -- the opt-in is purely additive;
3. the sign convention: with the corrected psi, G = +1 is attractive
   (interfacial force points toward the dense phase) and the mechanical
   pressure has a non-monotone (van der Waals loop) subcritical isotherm,
   while the opposite sign yields a monotone p_eff -- no coexistence
   possible (Phase-0 probe P2 discrimination).
"""

from __future__ import annotations

import math

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import (
    collide_sc_single_component,
    make_psi_eos,
    psi_eos_peng_robinson,
    sc_single_component_force,
)

CS2 = 1.0 / 3.0

# Equal-area Maxwell construction lock values for PR at T_r = 0.55
# (source: campaign archive phase0/out/pr_maxwell.json, Wave-10 M4 staging;
# the archived area residuals are ~1e-17, p_match ~1e-16).
PR_LOCK = {
    "T_r": 0.55,
    "T": 0.04010547044515562,
    "rho_l": 7.997835999814006,
    "rho_v": 0.0649052793822428,
    "ratio": 123.22319657100351,
    "p_sat": 0.002449394170218779,
    "rho_cross": 9.708178500398901,
}


def _p_mech(rho: float, psi: torch.Tensor, G: float = 1.0) -> torch.Tensor:
    """Mechanical pressure of the lattice scheme, p = rho cs^2 - (G/2) cs^2 psi^2."""
    return rho * CS2 - 0.5 * G * CS2 * psi * psi


def test_corrected_psi_reconstructs_maxwell_pressure_lock() -> None:
    """p_mech(G=+1) equals p_EOS = p_sat at BOTH coexistence densities."""
    t = PR_LOCK["T"]
    psi_l = psi_eos_peng_robinson(torch.tensor([PR_LOCK["rho_l"]], dtype=torch.float64), t)
    psi_v = psi_eos_peng_robinson(torch.tensor([PR_LOCK["rho_v"]], dtype=torch.float64), t)
    p_l = float(_p_mech(PR_LOCK["rho_l"], psi_l))
    p_v = float(_p_mech(PR_LOCK["rho_v"], psi_v))
    assert math.isclose(p_l, PR_LOCK["p_sat"], rel_tol=1e-9)
    assert math.isclose(p_v, PR_LOCK["p_sat"], rel_tol=1e-9)
    # equal mechanical pressure at the pair (coexistence requirement)
    assert math.isclose(p_l, p_v, rel_tol=1e-9)
    # closure agrees with the direct function
    closure = make_psi_eos("pr", t)
    rho = torch.tensor([0.1, 1.0, PR_LOCK["rho_l"]], dtype=torch.float64)
    assert torch.equal(closure(rho), psi_eos_peng_robinson(rho, t))
    # psi-domain: above rho_cross the argument clamps to zero (psi = 0),
    # so the scheme falls back to p_mech = rho*cs^2 on the dense branch
    above = torch.tensor([PR_LOCK["rho_cross"] + 0.1], dtype=torch.float64)
    psi_above = psi_eos_peng_robinson(above, t)
    assert float(psi_above) == 0.0
    assert torch.allclose(_p_mech(float(above), psi_above), above * CS2)


def _collide_sc_legacy(f, G, tau, psi_fn):
    """Pre-patch implementation of collide_sc_single_component (verbatim body)."""
    rho, ux, uy = macroscopic(f)
    fx, fy = sc_single_component_force(rho, G, psi_fn, 0.0, 0.0, None)
    rho_s = torch.clamp(rho, min=1e-12)
    feq = equilibrium(rho, ux + tau * fx / rho_s, uy + tau * fy / rho_s)
    return f - (f - feq) / tau


def test_forcing_velocity_shift_branch_is_bit_identical_to_legacy() -> None:
    """forcing="velocity_shift" (and the default call) reproduce the old path bitwise."""
    torch.manual_seed(20261001)
    L = 48
    rho = 0.05 + 7.9 * torch.rand(L, L)
    ux = 0.02 * torch.randn(L, L)
    uy = 0.02 * torch.randn(L, L)
    f = equilibrium(rho, ux, uy)
    psi_fn = make_psi_eos("pr", PR_LOCK["T"])
    for G, tau in ((1.0, 1.0), (-1.0, 0.7)):
        legacy = _collide_sc_legacy(f, G, tau, psi_fn)
        out_sc = collide_sc_single_component(
            f, G=G, tau=tau, psi_fn=psi_fn, forcing="velocity_shift"
        )
        out_default = collide_sc_single_component(f, G=G, tau=tau, psi_fn=psi_fn)
        assert torch.equal(out_sc, legacy)
        assert torch.equal(out_default, legacy)
    # the opt-in selector is live: edm differs from velocity_shift
    out_edm = collide_sc_single_component(f, G=1.0, tau=1.0, psi_fn=psi_fn, forcing="edm")
    assert not torch.equal(out_edm, _collide_sc_legacy(f, 1.0, 1.0, psi_fn))


def test_corrected_sign_attractive_and_isotherm_loops() -> None:
    """G = +1 pulls vapour toward the liquid; p_eff loops (coexistence-capable).

    The opposite sign gives a monotone p_eff = max(rho*cs^2, p_EOS) at any
    subcritical T (no liquid/vapour coexistence) -- the P2 discrimination.
    """
    psi_fn = make_psi_eos("pr", PR_LOCK["T"])
    # 1-D tanh interface, LIQUID at small x
    xs = torch.arange(64, dtype=torch.float32)
    prof = PR_LOCK["rho_v"] + 0.5 * (PR_LOCK["rho_l"] - PR_LOCK["rho_v"]) * (
        1.0 + torch.tanh((32.0 - xs) / 4.0)
    )
    field = prof.view(1, 64).expand(64, 64).contiguous()
    fx_pos, _ = sc_single_component_force(field, +1.0, psi_fn, 0.0, 0.0, None)
    fx_neg, _ = sc_single_component_force(field, -1.0, psi_fn, 0.0, 0.0, None)
    vapour_band = slice(36, 44)  # vapour side of the interface (interface near x=32)
    # toward the liquid = negative x direction => attractive for G = +1
    assert bool((fx_pos[:, vapour_band] < 0).all())
    assert float(fx_neg[:, vapour_band].mean()) > 0.0

    # isotherm discrimination, float64, crossing rho_cross
    a, b, R = 2.0 / 49.0, 2.0 / 21.0, 1.0
    t = PR_LOCK["T"]

    def p_eos(r):
        return r * R * t / (1.0 - b * r) - a * r * r / (1.0 + 2.0 * b * r - b * b * r * r)

    grid = torch.linspace(0.05, 10.3, 601, dtype=torch.float64)
    psi_corr = psi_fn(grid)
    p_eff_corr = grid * CS2 - 0.5 * CS2 * psi_corr * psi_corr  # = min(rho cs2, p_EOS)
    assert bool((torch.diff(p_eff_corr) < -1e-4).any())  # van der Waals loop present

    psi_lib = torch.sqrt(torch.clamp(2.0 * (p_eos(grid) - grid * CS2) / CS2, min=0.0))
    p_eff_lib = grid * CS2 + 0.5 * CS2 * psi_lib * psi_lib  # = max(rho cs2, p_EOS)
    assert bool((torch.diff(p_eff_lib) > -1e-9).all())  # monotone: no coexistence
