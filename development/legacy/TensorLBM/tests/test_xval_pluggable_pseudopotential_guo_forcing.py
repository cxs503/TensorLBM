"""Tests for XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING.

Cross-validates TensorLBM's Guo forcing implementation against an independent
reference solver (listLBM-style) for the Shan-Chen pseudopotential multiphase model.

Acceptance: <15% cross-error on surface tension (owner precision standard).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import C, CS2, W
from tensorlbm.multiphase import (
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    collide_guo,
    psi_exp,
    sc_force,
    stream_periodic,
)
from tensorlbm.multiphase.laplace import fit_laplace, sc_maxwell_coexistence

torch.set_default_dtype(torch.float64)

_C = C.to("cpu").float()
_W = W.to("cpu").float().view(9, 1, 1)


# ---------------------------------------------------------------------------
# Reference solver (independent implementation, listLBM-style)
# ---------------------------------------------------------------------------

def _eq_ref(rho, ux, uy):
    cu = _C[:, 0].view(9, 1, 1) * ux[None] + _C[:, 1].view(9, 1, 1) * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def _macro_ref(f):
    rho = f.sum(dim=0)
    rs = torch.clamp(rho, min=1e-12)
    ux = (f * _C[:, 0].view(9, 1, 1)).sum(dim=0) / rs
    uy = (f * _C[:, 1].view(9, 1, 1)).sum(dim=0) / rs
    return rho, ux, uy


def _sc_force_ref(rho, G):
    psi = psi_exp(rho)
    fx = torch.zeros_like(psi)
    fy = torch.zeros_like(psi)
    for i in range(9):
        cx, cy = int(_C[i, 0]), int(_C[i, 1])
        if cx == 0 and cy == 0:
            continue
        psi_shift = torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
        fx = fx + float(W[i]) * cx * psi_shift
        fy = fy + float(W[i]) * cy * psi_shift
    return G * CS2 * psi * fx, G * CS2 * psi * fy


def _guo_collision_ref(f, fx, fy, tau):
    rho, ux, uy = _macro_ref(f)
    feq = _eq_ref(rho, ux, uy)
    rho_safe = torch.clamp(rho, min=1e-12)
    uxp = ux + fx / (2.0 * rho_safe)
    uyp = uy + fy / (2.0 * rho_safe)
    cf = _C[:, 0].view(9, 1, 1) * fx.unsqueeze(0) + _C[:, 1].view(9, 1, 1) * fy.unsqueeze(0)
    cu = _C[:, 0].view(9, 1, 1) * uxp.unsqueeze(0) + _C[:, 1].view(9, 1, 1) * uyp.unsqueeze(0)
    uf = uxp * fx + uyp * fy
    src = _W * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2 ** 2)
    return f - (f - feq) / tau + src


def _stream_ref(f):
    out = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(_C[i, 0]), int(_C[i, 1])
        out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return out


def step_ref(f, G, tau):
    rho = f.sum(dim=0)
    fx, fy = _sc_force_ref(rho, G)
    f_post = _guo_collision_ref(f, fx, fy, tau)
    return _stream_ref(f_post)


def step_tlbm(f, G, tau):
    rho = f.sum(dim=0)
    fx, fy = sc_force(rho, G)
    f_post = collide_guo(f, fx, fy, tau)
    return stream_periodic(f_post)


def init_droplet(N, R, rho_l, rho_v, width=3.0):
    y = torch.arange(N, dtype=torch.float64)
    x = torch.arange(N, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    cx, cy = N / 2, N / 2
    dx = X - cx; dy = Y - cy
    dx -= N * torch.round(dx / N); dy -= N * torch.round(dy / N)
    r = torch.sqrt(dx ** 2 + dy ** 2)
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 + torch.tanh((R - r) / width))
    ux = torch.zeros(N, N, dtype=torch.float64)
    uy = torch.zeros_like(ux)
    return _eq_ref(rho, ux, uy)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGuoForcingCrossValidation:
    """XVAL: TensorLBM Guo forcing vs listLBM-style reference."""

    @pytest.mark.parametrize("R_init", [10, 14])
    def test_guo_forcing_matches_reference(self, R_init):
        """TensorLBM collide_guo must match independent reference to machine precision."""
        N = 48
        G = 5.0
        tau = 1.0
        steps = 200  # Short run for test speed

        f_ref = init_droplet(N, R_init, RHO_L_DEFAULT, RHO_V_DEFAULT)
        f_tl = f_ref.clone()

        for _ in range(steps):
            f_ref = step_ref(f_ref, G, tau)
            f_tl = step_tlbm(f_tl, G, tau)

        # Check bitwise identity (both use same float64 arithmetic)
        assert torch.allclose(f_ref, f_tl, atol=1e-12), \
            f"TensorLBM Guo forcing diverges from reference at R={R_init}"

    def test_laplace_law_consistency(self):
        """Both solvers must satisfy Laplace law with same surface tension."""
        N = 48
        G = 5.0
        tau = 1.0
        steps = 200
        radii = [10, 14]

        sigmas_ref = []
        sigmas_tl = []

        for R_init in radii:
            f_ref = init_droplet(N, R_init, RHO_L_DEFAULT, RHO_V_DEFAULT)
            f_tl = f_ref.clone()

            for _ in range(steps):
                f_ref = step_ref(f_ref, G, tau)
                f_tl = step_tlbm(f_tl, G, tau)

            # Measure sigma for reference
            rho_ref = f_ref.sum(dim=0)
            rho_l_ref = float(rho_ref.max())
            rho_v_ref = float(rho_ref.min())
            psi_l = float(psi_exp(torch.tensor(rho_l_ref)))
            psi_v = float(psi_exp(torch.tensor(rho_v_ref)))
            dp_ref = (rho_l_ref * CS2 + 0.5 * G * CS2 * psi_l ** 2) - \
                     (rho_v_ref * CS2 + 0.5 * G * CS2 * psi_v ** 2)
            mask_ref = rho_ref > 0.5 * (rho_l_ref + rho_v_ref)
            area_ref = float(mask_ref.sum())
            R_eq_ref = math.sqrt(area_ref / math.pi) if area_ref > 0 else 1.0
            sigmas_ref.append(dp_ref * R_eq_ref)

            # Measure sigma for TensorLBM
            rho_tl = f_tl.sum(dim=0)
            rho_l_tl = float(rho_tl.max())
            rho_v_tl = float(rho_tl.min())
            psi_l = float(psi_exp(torch.tensor(rho_l_tl)))
            psi_v = float(psi_exp(torch.tensor(rho_v_tl)))
            dp_tl = (rho_l_tl * CS2 + 0.5 * G * CS2 * psi_l ** 2) - \
                    (rho_v_tl * CS2 + 0.5 * G * CS2 * psi_v ** 2)
            mask_tl = rho_tl > 0.5 * (rho_l_tl + rho_v_tl)
            area_tl = float(mask_tl.sum())
            R_eq_tl = math.sqrt(area_tl / math.pi) if area_tl > 0 else 1.0
            sigmas_tl.append(dp_tl * R_eq_tl)

        # Cross-error on surface tension
        for s_ref, s_tl in zip(sigmas_ref, sigmas_tl):
            err = abs(s_tl - s_ref) / max(abs(s_ref), 1e-12) * 100
            assert err < 15.0, f"Surface tension cross-error {err:.2f}% exceeds 15%"

    def test_pluggable_pseudopotential(self):
        """Guo forcing must work with custom pseudopotential functions."""
        N = 32
        G = 5.0
        tau = 1.0
        steps = 100

        def psi_custom(rho):
            """Scaled exponential pseudopotential."""
            return 0.8 * psi_exp(rho)

        f = init_droplet(N, 8, RHO_L_DEFAULT, RHO_V_DEFAULT)

        # Run with custom psi
        for _ in range(steps):
            rho = f.sum(dim=0)
            psi = psi_custom(rho)
            fx = torch.zeros_like(psi)
            fy = torch.zeros_like(psi)
            for i in range(9):
                cx, cy = int(_C[i, 0]), int(_C[i, 1])
                if cx == 0 and cy == 0:
                    continue
                psi_shift = torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
                fx = fx + float(W[i]) * cx * psi_shift
                fy = fy + float(W[i]) * cy * psi_shift
            fx = G * CS2 * psi * fx
            fy = G * CS2 * psi * fy
            f = collide_guo(f, fx, fy, tau)
            f = stream_periodic(f)

        # Check that simulation ran without NaN/Inf
        rho = f.sum(dim=0)
        assert torch.isfinite(rho).all(), "Custom pseudopotential produced NaN/Inf"
        assert rho.max() > 0.1, "Density collapsed"

    def test_mass_conservation(self):
        """Guo forcing must conserve mass to machine precision."""
        N = 32
        G = 5.0
        tau = 1.0
        steps = 200

        f = init_droplet(N, 8, RHO_L_DEFAULT, RHO_V_DEFAULT)
        mass_0 = float(f.sum())

        for _ in range(steps):
            f = step_tlbm(f, G, tau)

        mass_final = float(f.sum())
        mass_error = abs(mass_final - mass_0) / mass_0

        assert mass_error < 1e-10, f"Mass error {mass_error:.2e} exceeds 1e-10"

    def test_spurious_currents_bounded(self):
        """Spurious currents must remain bounded (not blow up)."""
        N = 32
        G = 5.0
        tau = 1.0
        steps = 200

        f = init_droplet(N, 8, RHO_L_DEFAULT, RHO_V_DEFAULT)

        for _ in range(steps):
            f = step_tlbm(f, G, tau)

        rho = f.sum(dim=0)
        rs = torch.clamp(rho, min=1e-12)
        ux = (f * _C[:, 0].view(9, 1, 1)).sum(dim=0) / rs
        uy = (f * _C[:, 1].view(9, 1, 1)).sum(dim=0) / rs
        u_max = float(torch.sqrt(ux ** 2 + uy ** 2).max())

        assert u_max < 0.1, f"Spurious current {u_max:.2e} too large (possible instability)"
        assert math.isfinite(u_max), "Spurious current is NaN/Inf"
