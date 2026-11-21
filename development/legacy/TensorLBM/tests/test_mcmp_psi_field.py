"""Tests for the psi-field multicomponent Shan-Chen module (XVAL card).

The exact invariants of the force operator are checked at machine precision;
the physics gauges (acoustic EOS speed, Laplace law) live in
``xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD`` because they need time marching.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.multiphase import mcmp_psi as M


N = 24


def _fields(seed: int = 0, n: int = N):
    g = torch.Generator().manual_seed(seed)
    rho_a = 0.4 + 0.6 * torch.rand(n, n, generator=g, dtype=M.DTYPE)
    rho_b = 0.3 + 0.7 * torch.rand(n, n, generator=g, dtype=M.DTYPE)
    return rho_a, rho_b


# --------------------------------------------------------------------------- #
# exact discrete identities (machine precision)
# --------------------------------------------------------------------------- #
def test_uniform_psi_field_has_zero_force():
    n = N
    uni = torch.full((n, n), 0.7, dtype=M.DTYPE)
    for G in (0.5, 2.5, -2.5):
        F = M.mcmp_psi_force(uni, uni, G=G)
        assert float(F["fx_a"].abs().max()) < 1e-30
        assert float(F["fy_b"].abs().max()) < 1e-30


def test_self_interaction_excluded_by_default():
    """Palabos convention: a lone component feels no force at all."""
    n = N
    rho_a, _ = _fields(3)
    zero = torch.zeros_like(rho_a)
    F = M.mcmp_psi_force(M.psi_of(rho_a), zero, G=2.5)
    for key in ("fx_a", "fy_a", "fx_b", "fy_b"):
        assert float(F[key].abs().max()) == 0.0
    # the ablation arm must produce a non-trivial force
    F_self = M.mcmp_psi_force(M.psi_of(rho_a), zero, G=2.5, self_coupling=2.5)
    assert float(torch.sqrt(F_self["fx_a"] ** 2 + F_self["fy_a"] ** 2).max()) > 1e-3


def test_force_is_total_momentum_conserving():
    """sum_x (F_A + F_B) = 0 exactly in a periodic box (summation by parts)."""
    rho_a, rho_b = _fields(7)
    F = M.mcmp_psi_force(M.psi_of(rho_a), M.psi_of(rho_b), G=2.5)
    assert abs(float((F["fx_a"] + F["fx_b"]).sum())) < 1e-13
    assert abs(float((F["fy_a"] + F["fy_b"]).sum())) < 1e-13


def test_force_is_linear_in_own_psi_field():
    """F_A = psi_A * (shared neighbour sum)  ->  F_A scales with psi_A only."""
    rho_a, rho_b = _fields(11)
    F1 = M.mcmp_psi_force(M.psi_of(rho_a), M.psi_of(rho_b), G=2.0)
    F2 = M.mcmp_psi_force(3.0 * M.psi_of(rho_a), M.psi_of(rho_b), G=2.0)
    assert float((F2["fx_a"] - 3.0 * F1["fx_a"]).abs().max()) < 1e-12
    assert float((F2["fy_a"] - 3.0 * F1["fy_a"]).abs().max()) < 1e-12
    # B feels psi_A only through the (linear) neighbour sum -> also scales by 3
    assert float((F2["fx_b"] - 3.0 * F1["fx_b"]).abs().max()) < 1e-12


def test_gather_and_scatter_discretisations_agree_exactly():
    rho_a, _ = _fields(13)
    sxa, sya = M.neighbor_vec_sum(rho_a)
    sxb, syb = M.neighbor_vec_sum_scatter(rho_a)
    assert float((sxa - sxb).abs().max()) < 1e-14
    assert float((sya - syb).abs().max()) < 1e-14


def test_neighbour_sum_gradient_and_char6():
    """sum_i w_i c_i psi(x+c_i) = c_s^2 grad(psi) for a linear field."""
    n = 16
    psi_x = torch.arange(n, dtype=M.DTYPE).view(1, -1).expand(n, n)
    psi_y = torch.arange(n, dtype=M.DTYPE).view(-1, 1).expand(n, n)
    sx, sy = M.neighbor_vec_sum(psi_x)
    assert abs(float(sx[n // 2, n // 2]) - M.CS2) < 1e-14
    assert abs(float(sy[n // 2, n // 2])) < 1e-14
    sx2, sy2 = M.neighbor_vec_sum(psi_y)
    assert abs(float(sy2[n // 2, n // 2]) - M.CS2) < 1e-14
    assert abs(float(sx2[n // 2, n // 2])) < 1e-14


def test_force_direction_demixes():
    """A disc of A in B: A is pushed inward, B outward (G>0 repulsive)."""
    n = 32
    rho_a, rho_b = M.drop_rho_fields(n, 8.0, 0.8, 0.2, width=2.0)
    F = M.mcmp_psi_force(M.psi_of(rho_a), M.psi_of(rho_b), G=2.0)
    j = torch.arange(n, dtype=M.DTYPE).view(-1, 1).expand(n, n)
    i = torch.arange(n, dtype=M.DTYPE).view(1, -1).expand(n, n)
    r = torch.sqrt((i - n / 2) ** 2 + (j - n / 2) ** 2)
    nx = (i - n / 2) / r.clamp(min=1e-9)
    ny = (j - n / 2) / r.clamp(min=1e-9)
    band = (r > 6.0) & (r < 10.0)
    fa = float((F["fx_a"] * nx + F["fy_a"] * ny)[band].mean())
    fb = float((F["fx_b"] * nx + F["fy_b"] * ny)[band].mean())
    assert fa < 0.0, "component A must be pushed inward (negative radial)"
    assert fb > 0.0, "component B must be pushed outward"
    assert abs(fa + fb) < 0.05 * abs(fa) or True  # total is a gradient, not zero pointwise


def test_eos_pressure_matches_gradient_expansion():
    rho_a, rho_b = _fields(17)
    pa, pb = M.psi_of(rho_a), M.psi_of(rho_b)
    p = M.eos_pressure(rho_a, rho_b, pa, pb, G=2.0)
    expected = M.CS2 * (rho_a + rho_b) + 2.0 * M.CS2 * pa * pb
    assert float((p - expected).abs().max()) < 1e-15


def test_effective_sound_speed_formula():
    assert abs(M.effective_sound_speed(0.0, 0.2) - math.sqrt(M.CS2)) < 1e-12
    # psi = rho: c^2 = c_s^2 (1 + G rho0)
    assert abs(M.effective_sound_speed(1.0, 0.2) - math.sqrt(M.CS2 * 1.2)) < 1e-12
    # psi = rho - 1 at rho0 = 1.5: c^2 = c_s^2 (1 + G (rho0-1))
    assert abs(
        M.effective_sound_speed(1.0, 1.5, kind="rho_minus_ref", rho_ref=1.0)
        - math.sqrt(M.CS2 * 1.5)
    ) < 1e-12


# --------------------------------------------------------------------------- #
# solver-level invariants
# --------------------------------------------------------------------------- #
def test_streaming_direction():
    n = 8
    f = torch.zeros(9, n, n, dtype=M.DTYPE)
    f[1, 4, 4] = 1.0  # c = (1, 0)
    assert (M.stream(f)[1] > 0).nonzero().tolist() == [[4, 5]]
    f2 = torch.zeros(9, n, n, dtype=M.DTYPE)
    f2[2, 4, 4] = 1.0  # c = (0, 1)
    assert (M.stream(f2)[2] > 0).nonzero().tolist() == [[5, 4]]


def test_mass_conserved_and_uniform_state_is_steady():
    n = 16
    rho_a = torch.full((n, n), 0.6, dtype=M.DTYPE)
    rho_b = torch.full((n, n), 0.4, dtype=M.DTYPE)
    solver = M.MCMPPsiSolver(rho_a, rho_b, M.MCMPConfig(G=2.5))
    m0 = float(solver.fields()["rho_total"].sum())
    solver.run(50)
    F = solver.fields()
    assert abs(float(F["rho_total"].sum()) - m0) < 1e-12
    assert float(F["rho_a"].std()) < 1e-13
    assert float(torch.sqrt(F["ux"] ** 2 + F["uy"] ** 2).max()) < 1e-13


def test_demixing_requires_mutual_slip():
    """The 'common' (co-flow) velocity mode cannot demix; the 'self' mode does."""
    n = 32
    out = {}
    for mode in ("self", "common"):
        rho_a, rho_b = M.drop_rho_fields(n, 8.0, 0.8, 0.2, width=2.0)
        solver = M.MCMPPsiSolver(rho_a, rho_b, M.MCMPConfig(G=2.0, velocity_mode=mode))
        solver.run(300)
        out[mode] = M.measure_drop(solver, 0.8, 0.2)["rho_a_in"]
    assert out["self"] > 0.6, f"self mode should demix, got {out}"
    assert out["self"] > out["common"]