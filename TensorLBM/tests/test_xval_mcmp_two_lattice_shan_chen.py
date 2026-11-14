"""Test: XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN cross-validation.

Validates that TensorLBM's two-component Shan-Chen (MCMP) implementation
produces results consistent with the MPLBM-UT reference algorithm.

The test implements a minimal MPLBM-UT convention solver (forward-gather force,
G > 0 = repulsive) and compares it against TensorLBM's implementation
(backward-gather force, G < 0 = repulsive).  The two should be mathematically
equivalent when G_TL = -G_MPLBM.

Acceptance: σ cross-error < 15% (owner precision standard).
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------

CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
CS2 = 1.0 / 3.0

CX_T = torch.tensor(CX, dtype=torch.float64)
CY_T = torch.tensor(CY, dtype=torch.float64)
W_T = torch.tensor(W, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Reference solver: MPLBM-UT convention
# ---------------------------------------------------------------------------

def _eq(rho, ux, uy):
    cu = CX_T[:, None, None] * ux[None] + CY_T[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W_T[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def _macro(f):
    rho = f.sum(dim=0)
    rs = torch.clamp(rho, min=1e-12)
    ux = (f * CX_T[:, None, None]).sum(dim=0) / rs
    uy = (f * CY_T[:, None, None]).sum(dim=0) / rs
    return rho, ux, uy


def _force_mplbm(rho1, rho2, G12):
    """MPLBM-UT force: forward gather, G12 > 0 = repulsive."""
    gx2 = torch.zeros_like(rho2)
    gy2 = torch.zeros_like(rho2)
    gx1 = torch.zeros_like(rho1)
    gy1 = torch.zeros_like(rho1)
    for i in range(9):
        if CX[i] == 0 and CY[i] == 0:
            continue
        r2s = torch.roll(rho2, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        r1s = torch.roll(rho1, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        gx2 += W[i] * CX[i] * r2s
        gy2 += W[i] * CY[i] * r2s
        gx1 += W[i] * CX[i] * r1s
        gy1 += W[i] * CY[i] * r1s
    return -G12 * rho1 * gx2, -G12 * rho1 * gy2, -G12 * rho2 * gx1, -G12 * rho2 * gy1


def _stream(f):
    out = torch.empty_like(f)
    for i in range(9):
        out[i] = torch.roll(f[i], shifts=(int(CY[i]), int(CX[i])), dims=(0, 1))
    return out


def step_mplbm(f1, f2, G12, tau1, tau2):
    """One full step: collide (velocity-shift) + stream (MPLBM-UT convention)."""
    rho1, ux1, uy1 = _macro(f1)
    rho2, ux2, uy2 = _macro(f2)
    Fx1, Fy1, Fx2, Fy2 = _force_mplbm(rho1, rho2, G12)
    rs1 = torch.clamp(rho1, min=1e-12)
    rs2 = torch.clamp(rho2, min=1e-12)
    feq1 = _eq(rho1, ux1 + tau1 * Fx1 / rs1, uy1 + tau1 * Fy1 / rs1)
    feq2 = _eq(rho2, ux2 + tau2 * Fx2 / rs2, uy2 + tau2 * Fy2 / rs2)
    fp1 = f1 - (f1 - feq1) / tau1
    fp2 = f2 - (f2 - feq2) / tau2
    return _stream(fp1), _stream(fp2)


# ---------------------------------------------------------------------------
# TensorLBM solver
# ---------------------------------------------------------------------------

def _get_tl_step():
    from tensorlbm.multiphase import collide_sc_two_component
    from tensorlbm.solver import stream as tl_stream

    def step_tl(f1, f2, G12, tau1, tau2):
        fp1, fp2 = collide_sc_two_component(
            f1, f2, G_12=G12, tau1=tau1, tau2=tau2,
            u_eq="self", force_scheme="shift", psi=None,
        )
        return tl_stream(fp1), tl_stream(fp2)

    return step_tl


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _measure(f1, f2, G12_sign, G12):
    rho1, ux1, uy1 = _macro(f1)
    rho2, ux2, uy2 = _macro(f2)
    rho_tot = rho1 + rho2
    rs = torch.clamp(rho_tot, min=1e-12)
    ux = (rho1 * ux1 + rho2 * ux2) / rs
    uy = (rho1 * uy1 + rho2 * uy2) / rs

    if G12_sign == "mplbm":
        p = CS2 * (rho1 + rho2) - G12 * CS2 * rho1 * rho2
    else:
        p = CS2 * (rho1 + rho2) + G12 * CS2 * rho1 * rho2

    n = rho1.shape[0]
    c = n // 2
    k = max(4, n // 16)
    p_core = float(p[c - k:c + k, c - k:c + k].mean())
    p_far = float(p[:k, :k].mean())
    dp = p_core - p_far

    rho1_mean = float(rho1.mean())
    mask = rho1 > rho1_mean
    area = float(mask.sum())
    R_eq = math.sqrt(area / math.pi) if area > 0 else 0.0
    sigma = dp * R_eq if R_eq > 0 else 0.0

    u_max = float(torch.sqrt(ux ** 2 + uy ** 2).max())
    return dp, R_eq, sigma, u_max


def _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas, width=2.0):
    y = torch.arange(n, dtype=torch.float64)
    x = torch.arange(n, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    centre = (n - 1) / 2.0
    r = torch.sqrt((X - centre) ** 2 + (Y - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / max(width, 1e-6)))
    rho1 = rho1_gas + s * (rho1_liq - rho1_gas)
    rho2 = rho2_gas + s * (rho2_liq - rho2_gas)
    zero = torch.zeros_like(rho1)
    return _eq(rho1, zero, zero), _eq(rho2, zero, zero)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMCMPTwoLatticeShanChen:
    """Cross-validate TensorLBM MCMP against MPLBM-UT reference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.set_num_threads(1)
        self.step_tl = _get_tl_step()

    def test_force_equivalence(self):
        """MPLBM-UT forward-gather force equals TensorLBM backward-gather with sign flip."""
        n = 16
        torch.manual_seed(42)
        rho1 = torch.rand(n, n, dtype=torch.float64) * 0.5 + 0.5
        rho2 = torch.rand(n, n, dtype=torch.float64) * 0.5 + 0.5
        G_mplbm = 2.5
        G_tl = -G_mplbm

        # MPLBM-UT force (forward gather, G > 0 repulsive)
        Fx1_m, Fy1_m, Fx2_m, Fy2_m = _force_mplbm(rho1, rho2, G_mplbm)

        # TensorLBM force (backward gather, G < 0 repulsive)
        from tensorlbm.multiphase import sc_two_component_force
        Fx1_t, Fy1_t, Fx2_t, Fy2_t = sc_two_component_force(
            rho1, rho2, G_12=G_tl, psi=None,
        )

        # The forces should be identical (sign flip + direction flip cancel)
        torch.testing.assert_close(Fx1_m, Fx1_t, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Fy1_m, Fy1_t, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Fx2_m, Fx2_t, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Fy2_m, Fy2_t, atol=1e-12, rtol=1e-12)

    def test_single_step_equivalence(self):
        """One collide+stream step produces identical results."""
        n = 16
        G_mplbm = 2.5
        G_tl = -G_mplbm
        tau = 1.0

        rho1 = torch.full((n, n), 0.5, dtype=torch.float64)
        rho2 = torch.full((n, n), 0.5, dtype=torch.float64)
        zero = torch.zeros_like(rho1)
        f1 = _eq(rho1, zero, zero)
        f2 = _eq(rho2, zero, zero)

        f1_m, f2_m = step_mplbm(f1.clone(), f2.clone(), G_mplbm, tau, tau)
        f1_t, f2_t = self.step_tl(f1.clone(), f2.clone(), G_tl, tau, tau)

        torch.testing.assert_close(f1_m, f1_t, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(f2_m, f2_t, atol=1e-12, rtol=1e-12)

    def test_droplet_sigma_cross_error_below_15pct(self):
        """Static droplet: σ cross-error between MPLBM-UT and TensorLBM < 15%."""
        n = 32
        G_mplbm = 2.5
        G_tl = -G_mplbm
        tau = 1.0
        steps = 1500
        radius = 10.0

        rho1_liq, rho1_gas = 0.92, 0.08
        rho2_liq, rho2_gas = 0.08, 0.92

        f1_m, f2_m = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)
        f1_t, f2_t = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)

        for _ in range(steps):
            f1_m, f2_m = step_mplbm(f1_m, f2_m, G_mplbm, tau, tau)
            f1_t, f2_t = self.step_tl(f1_t, f2_t, G_tl, tau, tau)

        assert not torch.isnan(f1_m).any(), "MPLBM-UT result is NaN"
        assert not torch.isnan(f1_t).any(), "TensorLBM result is NaN"

        dp_m, R_m, sig_m, u_m = _measure(f1_m, f2_m, "mplbm", G_mplbm)
        dp_t, R_t, sig_t, u_t = _measure(f1_t, f2_t, "tensorlbm", G_tl)

        sig_err = abs(sig_t - sig_m) / max(abs(sig_m), 1e-12) * 100
        assert sig_err < 15.0, f"σ cross-error {sig_err:.2f}% exceeds 15%"

    def test_droplet_dp_cross_error_below_15pct(self):
        """Static droplet: Δp cross-error < 15%."""
        n = 32
        G_mplbm = 2.5
        G_tl = -G_mplbm
        tau = 1.0
        steps = 1500
        radius = 10.0

        rho1_liq, rho1_gas = 0.92, 0.08
        rho2_liq, rho2_gas = 0.08, 0.92

        f1_m, f2_m = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)
        f1_t, f2_t = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)

        for _ in range(steps):
            f1_m, f2_m = step_mplbm(f1_m, f2_m, G_mplbm, tau, tau)
            f1_t, f2_t = self.step_tl(f1_t, f2_t, G_tl, tau, tau)

        dp_m, R_m, sig_m, u_m = _measure(f1_m, f2_m, "mplbm", G_mplbm)
        dp_t, R_t, sig_t, u_t = _measure(f1_t, f2_t, "tensorlbm", G_tl)

        dp_err = abs(dp_t - dp_m) / max(abs(dp_m), 1e-12) * 100
        assert dp_err < 15.0, f"Δp cross-error {dp_err:.2f}% exceeds 15%"

    def test_no_nan_at_moderate_coupling(self):
        """Both solvers remain stable at G = 3.0 for 1000 steps."""
        n = 32
        G_mplbm = 3.0
        G_tl = -G_mplbm
        tau = 1.0
        steps = 1000
        radius = 10.0

        rho1_liq, rho1_gas = 0.92, 0.08
        rho2_liq, rho2_gas = 0.08, 0.92

        f1_m, f2_m = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)
        f1_t, f2_t = _init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas)

        for _ in range(steps):
            f1_m, f2_m = step_mplbm(f1_m, f2_m, G_mplbm, tau, tau)
            f1_t, f2_t = self.step_tl(f1_t, f2_t, G_tl, tau, tau)

        assert not torch.isnan(f1_m).any(), "MPLBM-UT NaN at G=3.0"
        assert not torch.isnan(f1_t).any(), "TensorLBM NaN at G=3.0"
