"""Tests for the AA-pattern fused stream-collide module.

These replace the 2026-11-13 suite, whose equivalence tests were **degenerate**:
they compared two operators on a *spatially uniform* state, where every stencil
(``torch.roll``) is the identity, so the assertion ``max|f_aa - f_sep| < 1e-14``
held for *any* pair of operators -- including one that is not equivalent at all.
Every equivalence test below therefore runs on a **non-uniform** state.

Guards
------
* ``test_fused_equals_canonical_*``  -- the AA fused pull is bit-identical to
  collide-then-stream (the property the pattern is supposed to have).
* ``test_full_step_equivalence_with_walls`` -- same, with halfway bounce-back
  and a moving lid.
* ``test_uniform_state_is_degenerate`` -- pins the trap that made the old suite
  pass: on a uniform state all three orderings agree exactly, so such a test
  proves nothing.
* ``test_legacy_stream_then_collide_differs`` -- negative control: the retired
  ``C[S[f]]`` operator must stay measurably different from the canonical one.
* ``test_shear_wave_viscosity_matches_analytic`` -- physical anchor: the fused
  kernel reproduces the exact viscous decay rate of a shear wave.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.aa_fused_push_pull import fused_push_pull_step
from tensorlbm.aa_fused_stream_collide import (
    aa_fused_stream_collide,
    separate_stream_collide,
    stream_then_collide_legacy,
    aa_fused_step,
    separate_step,
    _macroscopic,
    _equilibrium,
    C,
    W,
)

torch.set_default_dtype(torch.float64)


def _shear_state(ny: int = 32, nx: int = 8, A: float = 0.05) -> torch.Tensor:
    """Spatially non-uniform, MAC-clean state: u = A sin(2 pi y / ny) e_x, rho = 1."""
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1).expand(ny, nx)
    ux = A * torch.sin(2.0 * math.pi * y / ny)
    uy = torch.zeros_like(ux)
    feq = _equilibrium(torch.ones_like(ux), ux, uy)
    return feq + 1e-3 * (feq - feq.mean(0, keepdim=True))


def _uniform_state(n: int = 32) -> torch.Tensor:
    f = torch.ones((9, n, n)) * 0.1
    f[0] = 0.5
    return f


class TestFusedEquivalence:
    """AA fused pull must equal the canonical collide-then-stream operator."""

    def test_fused_equals_canonical_nonuniform(self):
        f = _shear_state()
        tau = 0.8
        a = aa_fused_stream_collide(f.clone(), tau)
        b = separate_stream_collide(f.clone(), tau)
        c = fused_push_pull_step(f.clone(), tau)  # independent, verified kernel
        assert torch.equal(a, b), f"max diff {(a - b).abs().max().item():.3e}"
        assert (c - b).abs().max().item() < 1e-14

    def test_fused_equals_canonical_multi_step_nonuniform(self):
        f_a, f_b = _shear_state(), _shear_state()
        tau = 0.8
        for _ in range(200):
            f_a = aa_fused_stream_collide(f_a, tau)
            f_b = separate_stream_collide(f_b, tau)
        assert torch.equal(f_a, f_b)

    def test_uniform_state_is_degenerate(self):
        """Why the old suite was vacuous: on a uniform state every ordering matches."""
        f = _uniform_state()
        tau = 1.0
        a = aa_fused_stream_collide(f.clone(), tau)
        b = separate_stream_collide(f.clone(), tau)
        c = stream_then_collide_legacy(f.clone(), tau)
        assert torch.equal(a, b) and torch.equal(a, c)

    def test_legacy_stream_then_collide_differs(self):
        """Negative control: the retired C[S[f]] operator is not equivalent."""
        f = _shear_state()
        tau = 0.8
        canon = separate_stream_collide(f.clone(), tau)
        leg = stream_then_collide_legacy(f.clone(), tau)
        rel = (leg - canon).abs().max().item() / canon.abs().max().item()
        assert rel > 1e-4, f"legacy operator unexpectedly equivalent (rel {rel:.3e})"


class TestFullStepWithWalls:
    """Fused and materialised full steps share the boundary algebra exactly."""

    def _cavity(self, L: int = 16, U: float = 0.1):
        n = L + 2
        solid = torch.zeros((n, n), dtype=torch.bool)
        solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
        rho = torch.ones((n, n))
        feq = _equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        return feq, solid, U

    def test_full_step_equivalence_with_walls(self):
        f0, solid, U = self._cavity()
        tau = 0.55
        f_a = aa_fused_step(f0.clone(), tau, solid, u_lid=U, lid_row=-1)
        f_b = separate_step(f0.clone(), tau, solid, u_lid=U, lid_row=-1)
        assert torch.equal(f_a, f_b)

    def test_full_step_equivalence_multi_step_with_walls(self):
        f_a, solid, U = self._cavity()
        f_b = f_a.clone()
        tau = 0.55
        for _ in range(50):
            f_a = aa_fused_step(f_a, tau, solid, u_lid=U, lid_row=-1)
            f_b = separate_step(f_b, tau, solid, u_lid=U, lid_row=-1)
        assert torch.equal(f_a, f_b)

    def test_moving_lid_drives_flow(self):
        f, solid, U = self._cavity(L=16, U=0.1)
        tau = 0.55
        for _ in range(300):
            f = aa_fused_step(f, tau, solid, u_lid=U, lid_row=-1)
        _, ux, _ = _macroscopic(f)
        interior = ux[1:-1, 1:-1]
        assert interior.max() > 0.0 and torch.isfinite(f).all()


class TestMassConservation:
    def test_mass_conserved_periodic(self):
        f = _shear_state()
        m0 = f.sum().item()
        f_out = aa_fused_stream_collide(f, 1.0)
        assert abs(f_out.sum().item() - m0) / m0 < 1e-12


class TestPhysics:
    """Analytic anchor: shear-wave viscous decay u_x = A sin(ky) exp(-nu k^2 t)."""

    def test_shear_wave_viscosity_matches_analytic(self):
        ny, nx, A = 64, 8, 0.02
        nu = 0.05
        tau = 0.5 + 3.0 * nu
        k = 2.0 * math.pi / ny
        f = _shear_state(ny, nx, A)

        def fundamental(g):
            _, ux, _ = _macroscopic(g)
            prof = ux[:, 0].numpy()
            return float(2.0 / ny * np.dot(prof, np.sin(2 * np.pi * np.arange(ny) / ny)))

        a0 = fundamental(f)
        steps = 4000
        for _ in range(steps):
            f = aa_fused_stream_collide(f, tau)
        a_end = fundamental(f)
        a_exact = a0 * math.exp(-nu * k**2 * steps)
        rel_err = abs(a_end - a_exact) / a_exact
        assert rel_err < 0.01, f"shear-wave amplitude error {rel_err:.3%} >= 1%"


class TestEquilibrium:
    def test_equilibrium_fixed_point(self):
        n = 33
        rho = torch.ones((n, n))
        ux = torch.zeros((n, n))
        uy = torch.zeros((n, n))
        feq = _equilibrium(rho, ux, uy)
        f_out = aa_fused_stream_collide(feq, 1.0)
        rho_out, ux_out, uy_out = _macroscopic(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-12)
        assert torch.allclose(ux_out, ux, atol=1e-12)
        assert torch.allclose(uy_out, uy, atol=1e-12)


class TestStability:
    def test_finite_values_after_steps(self):
        f = _shear_state()
        for _ in range(500):
            f = aa_fused_stream_collide(f, 1.0)
        assert torch.all(torch.isfinite(f))


class TestLatticeConstants:
    def test_d2q9_velocities(self):
        assert C[:, 0].sum().item() == 0
        assert C[:, 1].sum().item() == 0

    def test_d2q9_weights(self):
        expected = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
        for i, w in enumerate(expected):
            assert abs(W[i].item() - w) < 1e-15
        assert abs(W.sum().item() - 1.0) < 1e-15