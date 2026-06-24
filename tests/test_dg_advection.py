"""Correctness tests for the real nodal-DG advection kernel (dg_advection.py).

These are the mathematical gate for the DG-LBM: they verify the DG operator
itself, in isolation, before any LBM collision or hybrid coupling is layered on.
Run CPU-only so they never contend with GPU jobs:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_dg_advection.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.dg_advection import (
    cell_means_from_nodal,
    collide_bgk_dg,
    dg_advect,
    dg_lbm_step,
    dg_rhs,
    equilibrium_dg,
    get_ops,
    lobatto_nodes,
    macroscopic_dg,
    nodal_from_mean,
)
from tensorlbm.d2q9 import C as C2D, W as W2D, equilibrium, macroscopic
from tensorlbm.d3q19 import C as C3D, W as W3D
from tensorlbm.solver import stream

DT = torch.float64


# ---------------------------------------------------------------------------
# 1. Reference-operator construction sanity (p=1, dx=1)
# ---------------------------------------------------------------------------


class TestOperators:
    def test_lobatto_nodes(self) -> None:
        assert np.allclose(lobatto_nodes(1), [-1.0, 1.0])
        assert np.allclose(lobatto_nodes(2), [-1.0, 0.0, 1.0])
        # p=3: ±1 plus ±sqrt(1/5)
        n3 = lobatto_nodes(3)
        assert np.allclose(n3, [-1.0, -math.sqrt(1 / 5), math.sqrt(1 / 5), 1.0])

    def test_p1_operators_match_hand_calc(self) -> None:
        """For P1 Lobatto, dx=1:

        M = [[2/3, 1/3],[1/3, 2/3]],  G = [[-1/2,-1/2],[1/2,1/2]],
        Ax  = (2/dx) M⁻¹ G   = [[-3,-3],[3,3]],
        face_lift = (2/dx) M⁻¹ S = [[-4,-2],[2,4]]   (S=[[-1,0],[0,1]]).
        """
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        Ax_expected = torch.tensor([[-3.0, -3.0], [3.0, 3.0]], dtype=DT)
        fl_expected = torch.tensor([[-4.0, -2.0], [2.0, 4.0]], dtype=DT)
        assert torch.allclose(ops.Ax, Ax_expected, atol=1e-9)
        assert torch.allclose(ops.face_lift, fl_expected, atol=1e-9)

    def test_operators_scale_with_dx(self) -> None:
        """Operators scale as 1/dx."""
        a = get_ops(degree=1, dx=1.0, dtype=DT)
        b = get_ops(degree=1, dx=2.0, dtype=DT)
        assert torch.allclose(a.Ax, 2.0 * b.Ax, atol=1e-9)
        assert torch.allclose(a.face_lift, 2.0 * b.face_lift, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. p=0 (P0) with Δt=Δx=1 == exact LBM shift for axis-aligned velocities
# ---------------------------------------------------------------------------


class TestP0Equivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_axis_aligned_equals_stream(self, seed: int) -> None:
        """For axis-aligned lattice velocities, P0 DG + one Euler step at
        Δt=Δx=1 reproduces the exact LBM stream to machine precision."""
        torch.manual_seed(seed)
        nx, ny = 24, 16
        rho = 1.0 + 0.1 * torch.rand(ny, nx, dtype=DT)
        ux = 0.08 + 0.03 * torch.rand(ny, nx, dtype=DT)
        uy = 0.05 * torch.rand(ny, nx, dtype=DT)
        f_lbm = equilibrium(rho, ux, uy, device="cpu").to(DT)

        ops = get_ops(degree=0, dx=1.0, dtype=DT)
        node_axes = (3, 4)
        f_dg = nodal_from_mean(f_lbm, ops, node_axes)

        axis_aligned = [1, 2, 3, 4]  # D2Q9 dirs (1,0),(0,1),(-1,0),(0,-1)
        vel = C2D.to(DT)
        advected = dg_advect(
            f_dg, vel, ops, ndim_spatial=2, dt=1.0, n_substeps=1, scheme="euler"
        )
        got = cell_means_from_nodal(advected, node_axes)

        expected = stream(f_lbm)
        # Compare only the axis-aligned populations; the rest dir is trivial.
        for q in axis_aligned:
            assert torch.allclose(got[q], expected[q], atol=1e-9), (
                f"P0 dir {q} (c={C2D[q].tolist()}) != stream"
            )

    def test_rest_population_unchanged(self) -> None:
        nx, ny = 12, 8
        f_lbm = equilibrium(
            torch.ones(ny, nx, dtype=DT),
            torch.full((ny, nx), 0.1, dtype=DT),
            torch.zeros(ny, nx, dtype=DT),
        )
        ops = get_ops(degree=0, dx=1.0, dtype=DT)
        f_dg = nodal_from_mean(f_lbm, ops, (3, 4))
        advected = dg_advect(
            f_dg, C2D.to(DT), ops, ndim_spatial=2, dt=1.0, n_substeps=1, scheme="euler"
        )
        got = cell_means_from_nodal(advected, (3, 4))
        assert torch.allclose(got[0], f_lbm[0], atol=1e-9)


# ---------------------------------------------------------------------------
# 3. MMS: pure periodic advection, P1 must converge at O(Δx²)
# ---------------------------------------------------------------------------


def _advect_periodic_1d(nx: int, degree: int, n_periods: float = 1.0) -> float:
    """Advect sin(x) once around [0, 2π); return L2 error (Lobatto quadrature)."""
    L = 2.0 * math.pi
    dx = L / nx
    ops = get_ops(degree=degree, dx=dx, dtype=DT)
    # Node positions: cell i, node k at x = i*dx + dx/2*(1+r_k).
    r = torch.tensor(lobatto_nodes(degree), dtype=DT)
    i_idx = torch.arange(nx, dtype=DT).view(nx, 1)
    x_node = i_idx * dx + (dx / 2.0) * (1.0 + r.view(1, -1))     # (nx, n_node)
    x_node = x_node % L
    f = torch.sin(x_node).unsqueeze(0)                            # (1, nx, n_node)

    vel = torch.tensor([[1.0]], dtype=DT)                        # scalar, c=1
    cfl = 0.2
    dt_step = cfl * dx
    T = n_periods * L
    n_steps = int(round(T / dt_step))
    dt_step = T / n_steps                                         # exact total time
    for _ in range(n_steps):
        f = dg_advect(f, vel, ops, ndim_spatial=1, dt=dt_step, n_substeps=1, scheme="rk3")

    exact = torch.sin(x_node)
    # Lobatto L2 error per cell: (dx/2) Σ_k w_k (f_k - exact_k)².
    wlob = _lobatto_weights(degree)
    err2 = ((f.squeeze(0) - exact) ** 2 * torch.tensor(wlob, dtype=DT)).sum().item()
    err2 *= dx / 2.0
    return math.sqrt(err2 / L)


def _lobatto_weights(degree: int) -> np.ndarray:
    """Lobatto quadrature weights on [-1,1] for (degree+1) points."""
    n = degree + 1
    nodes = lobatto_nodes(degree)
    w = np.zeros(n)
    for k in range(n):
        # w_k = 2/(n(n-1)) * 1/[P_{n-1}(x_k)]²
        pk = np.polynomial.legendre.Legendre.basis(n - 1)(nodes[k])
        w[k] = 2.0 / (n * (n - 1)) / (pk * pk)
    return w


class TestMMSConvergence:
    def test_p1_is_second_order(self) -> None:
        nxs = [16, 32, 64, 128]
        errs = [_advect_periodic_1d(nx, degree=1) for nx in nxs]
        rates = [math.log(errs[i] / errs[i + 1], 2) for i in range(len(errs) - 1)]
        # P1 DG ⇒ O(Δx²). Accept rate in [1.7, 2.3] (temporal error perturbs it).
        assert min(rates[1:]) > 1.7, f"P1 rates {rates} not ~2"
        # And the error should actually be decreasing.
        assert errs[-1] < errs[0], f"error not decreasing: {errs}"

    def test_p2_is_third_order(self) -> None:
        nxs = [16, 32, 64]
        errs = [_advect_periodic_1d(nx, degree=2) for nx in nxs]
        rates = [math.log(errs[i] / errs[i + 1], 2) for i in range(len(errs) - 1)]
        # P2 DG ⇒ O(Δx³).
        assert min(rates) > 2.5, f"P2 rates {rates} not ~3"


# ---------------------------------------------------------------------------
# 4. Conservation: periodic advection preserves total mass per velocity
# ---------------------------------------------------------------------------


class TestConservation:
    def test_mass_per_velocity_conserved(self) -> None:
        """On a periodic domain the upwind-flux DG advection conserves each
        population's total mass exactly (to round-off)."""
        torch.manual_seed(3)
        nx, ny = 20, 12
        rho = 1.0 + 0.2 * torch.rand(ny, nx, dtype=DT)
        ux = 0.1 * torch.rand(ny, nx, dtype=DT)
        uy = 0.1 * torch.rand(ny, nx, dtype=DT)
        f_lbm = equilibrium(rho, ux, uy).to(DT)

        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        f_dg = nodal_from_mean(f_lbm, ops, (3, 4))
        mass0 = f_dg.sum(dim=(1, 2, 3, 4))     # per-velocity total

        vel = C2D.to(DT)
        for _ in range(40):
            f_dg = dg_advect(
                f_dg, vel, ops, ndim_spatial=2, dt=0.25, n_substeps=1, scheme="rk3"
            )
        mass1 = f_dg.sum(dim=(1, 2, 3, 4))
        drift = ((mass1 - mass0).abs() / mass0.abs().clamp(min=1e-30)).max().item()
        # SSP-RK3 conserves a conserved quantity exactly in exact arithmetic;
        # this is float64 round-off accumulated over 40×3 RK stages (a real
        # conservation bug gives O(1) drift, cf. the pre-fix 1e14 blow-up).
        assert drift < 1e-7, f"relative mass drift {drift:.2e} exceeds round-off"

    def test_momentum_roughly_conserved_no_collision(self) -> None:
        """Without collision, the conserved momentum Σ c_i f_i is preserved by
        pure advection (each velocity advected independently)."""
        torch.manual_seed(4)
        nx, ny = 16, 12
        rho = 1.0 + 0.2 * torch.rand(ny, nx, dtype=DT)
        ux = 0.1 * torch.rand(ny, nx, dtype=DT)
        uy = 0.1 * torch.rand(ny, nx, dtype=DT)
        f_lbm = equilibrium(rho, ux, uy).to(DT)
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        f_dg = nodal_from_mean(f_lbm, ops, (3, 4))

        c = C2D.to(DT)
        def momentum(fd: torch.Tensor) -> torch.Tensor:
            flat = fd.sum(dim=(1, 2, 3, 4))           # per-velocity total mass
            return (c * flat.unsqueeze(1)).sum(dim=0)

        p0 = momentum(f_dg)
        for _ in range(30):
            f_dg = dg_advect(f_dg, c, ops, ndim_spatial=2, dt=0.25, scheme="rk3")
        p1 = momentum(f_dg)
        drift = (p1 - p0).abs().max().item()
        assert drift < 1e-9, f"momentum drift {drift:.2e}"


# ---------------------------------------------------------------------------
# 5. DG-LBM physics gate: shear-wave decay recovers the DVBE viscosity ν = τ/3
# ---------------------------------------------------------------------------


class TestShearWaveViscosity:
    """The fundamental "does my DG-LBM recover Navier–Stokes viscosity" check.

    A periodic shear wave u_x = U₀ sin(k y) decays as exp(−ν k² t).  The
    method-of-lines DG-LBM (advection + collision in one RK RHS) solves the
    *continuous* discrete-velocity Boltzmann equation, so it recovers
    ν = τ c_s² = τ/3 — NOT the discrete-LBM (τ − ½)/3 (the −½ is a time-split
    artefact of exact-shift streaming that method-of-lines does not have).
    To later match a discrete-LBM exterior in the hybrid, the coupler uses
    τ_dg = τ_lbm − ½.
    """

    @pytest.mark.parametrize("tau", [0.7, 0.9, 1.2])
    def test_recovers_dvbe_viscosity(self, tau: float) -> None:
        ny, nx = 32, 32
        n_steps = 40
        U0 = 0.01
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        r = torch.tensor(lobatto_nodes(1), dtype=DT)
        j = torch.arange(ny, dtype=DT).view(ny, 1)
        y_node = (j + 0.5 * (1.0 + r.view(1, -1))) % ny          # (ny, 2)
        ux_node = U0 * torch.sin(2.0 * math.pi * y_node / ny)
        rho = torch.ones(ny, nx, 2, 2, dtype=DT)
        ux = ux_node.view(ny, 1, 2, 1).expand(ny, nx, 2, 2)
        uy = torch.zeros_like(ux)
        f_dg = equilibrium_dg(rho, [ux, uy], C2D.to(DT), W2D.to(DT))

        vel = C2D.to(DT)
        for _ in range(n_steps):
            f_dg = dg_lbm_step(
                f_dg, vel, W2D.to(DT), ops, tau=tau,
                ndim_spatial=2, dt=1.0, n_substeps=6, scheme="rk3",
            )

        rho_o, us = macroscopic_dg(f_dg, vel)
        ux_mean = us[0].mean(dim=(2, 3))                       # (ny, nx)
        yc = (torch.arange(ny, dtype=DT) + 0.5)
        amp = (ux_mean.mean(dim=1) * torch.sin(2 * math.pi * yc / ny)).sum().item() * (2.0 / ny)

        assert amp > 0, f"τ={tau}: unstable (amp={amp}) — scheme diverged"
        k = 2.0 * math.pi / ny
        nu_eff = -math.log(amp / U0) / (k * k * n_steps)
        nu_exact = tau / 3.0
        rel = abs(nu_eff - nu_exact) / nu_exact
        assert rel < 0.06, f"τ={tau}: ν_eff={nu_eff:.4f} vs DVBE {nu_exact:.4f} (rel {rel:.2%})"


# ---------------------------------------------------------------------------
# 6. 3D (D3Q19) — advection convergence + DG-LBM viscosity
# ---------------------------------------------------------------------------


def _advect_periodic_3d(n: int, degree: int, vel: tuple[float, float, float]) -> float:
    """Advect sin(x)sin(y)sin(z) on [0, 2π)³ by *vel* for one period (T=2π);
    return the L2 error (Lobatto quadrature).  Exercises all three axes."""
    L = 2.0 * math.pi
    dx = L / n
    ops = get_ops(degree=degree, dx=dx, dtype=DT)
    r = torch.tensor(lobatto_nodes(degree), dtype=DT)
    i = torch.arange(n, dtype=DT).view(n, 1)
    pos = (i * dx + (dx / 2.0) * (1.0 + r.view(1, -1))) % L          # (n, n_node)
    nn = degree + 1
    xb = pos.view(1, 1, n, 1, 1, nn).expand(n, n, n, nn, nn, nn)
    yb = pos.view(1, n, 1, 1, nn, 1).expand(n, n, n, nn, nn, nn)
    zb = pos.view(n, 1, 1, nn, 1, 1).expand(n, n, n, nn, nn, nn)
    f = (torch.sin(xb) * torch.sin(yb) * torch.sin(zb)).unsqueeze(0)  # (1,nz,ny,nx,pz,py,px)
    v = torch.tensor([list(vel)], dtype=DT)                          # (1,3)
    # Time step: one full period along the dominant axis (c=1 ⇒ T=2π).
    # 3D dimension-by-dimension DG has a CFL bound ~1/3 of the 1D limit, so use
    # a conservative CFL well inside the stability region.
    T = L
    cfl = 0.08
    dt = cfl * dx
    nsteps = max(1, int(round(T / dt)))
    dt = T / nsteps
    for _ in range(nsteps):
        f = dg_advect(f, v, ops, ndim_spatial=3, dt=dt, n_substeps=1, scheme="rk3")
    exact = (torch.sin(xb) * torch.sin(yb) * torch.sin(zb))
    wlob = torch.tensor(_lobatto_weights(degree), dtype=DT)
    w3 = wlob[:, None, None] * wlob[None, :, None] * wlob[None, None, :]   # (nn,nn,nn)
    err2 = (((f.squeeze(0) - exact) ** 2) * w3.view(1, 1, 1, nn, nn, nn)).sum().item()
    err2 *= (dx / 2.0) ** 3
    return math.sqrt(err2 / (L ** 3))


class Test3DAdvection:
    def test_p1_second_order_diagonal(self) -> None:
        nxs = [8, 12, 16]
        errs = [_advect_periodic_3d(n, degree=1, vel=(1.0, 1.0, 1.0)) for n in nxs]
        # n does not double across the (small, 3-D) grid sequence, so use the
        # actual grid ratios rather than a base-2 log.
        rates = [
            math.log(errs[i] / errs[i + 1]) / math.log(nxs[i + 1] / nxs[i])
            for i in range(len(errs) - 1)
        ]
        assert min(rates) > 1.7, f"3D P1 rates {rates} not ~2"
        assert errs[-1] < errs[0]

    def test_mass_conserved_3d(self) -> None:
        torch.manual_seed(7)
        n = 10
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        f = (0.5 + 0.4 * torch.rand(1, n, n, n, 2, 2, 2, dtype=DT))
        v = torch.tensor([[1.0, 0.5, -0.5]], dtype=DT)
        m0 = f.sum().item()
        for _ in range(20):
            f = dg_advect(f, v, ops, ndim_spatial=3, dt=0.15, scheme="rk3")
        rel = abs(f.sum().item() - m0) / abs(m0)
        assert rel < 1e-7, f"3D mass drift {rel:.2e}"


class Test3DDGLBMViscosity:
    """3D D3Q19 DG-LBM recovers ν = τ/3 on a 3D shear wave u_x = U₀ sin(k y)."""

    @pytest.mark.parametrize("tau", [0.9, 1.2])
    def test_recovers_dvbe_viscosity_3d(self, tau: float) -> None:
        n = 24
        n_steps = 20
        U0 = 0.01
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        r = torch.tensor(lobatto_nodes(1), dtype=DT)
        j = torch.arange(n, dtype=DT).view(n, 1)
        y_node = (j + 0.5 * (1.0 + r.view(1, -1))) % n               # (n, 2)
        uxn = U0 * torch.sin(2.0 * math.pi * y_node / n)             # (n, 2)
        rho = torch.ones(n, n, n, 2, 2, 2, dtype=DT)
        ux = uxn.view(1, n, 1, 2, 1, 1).expand(n, n, n, 2, 2, 2)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        f = equilibrium_dg(rho, [ux, uy, uz], C3D.to(DT), W3D.to(DT))

        for _ in range(n_steps):
            f = dg_lbm_step(
                f, C3D.to(DT), W3D.to(DT), ops, tau=tau,
                ndim_spatial=3, dt=1.0, n_substeps=8, scheme="rk3",
            )
        rho_o, us = macroscopic_dg(f, C3D.to(DT))
        ux_mean = us[0].mean(dim=(0, 2, 3, 4, 5))                    # keep ny (avg nz,nx,pz,py,px)
        yc = (torch.arange(n, dtype=DT) + 0.5)
        amp = (ux_mean * torch.sin(2 * math.pi * yc / n)).sum().item() * (2.0 / n)
        assert amp > 0, f"τ={tau}: unstable (amp={amp})"
        k = 2.0 * math.pi / n
        nu_eff = -math.log(amp / U0) / (k * k * n_steps)
        rel = abs(nu_eff - tau / 3.0) / (tau / 3.0)
        assert rel < 0.10, f"τ={tau}: 3D ν_eff={nu_eff:.4f} vs τ/3={tau/3.0:.4f}"



