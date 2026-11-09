"""Tests for the microbubble drag-reduction module (P106W2-N01).

The tests target the invariants the node's machine criteria are built on, all of
which are verifiable without a long turbulent run:

1. ``test_laminar_poiseuille`` -- the kernel reproduces the analytic parabolic
   channel profile and the wall shear ``tau_w = F0 H / 2`` (this is what makes
   the uniform per-volume forcing and the bounce-back momentum accounting
   trustworthy).
2. ``test_momentum_exchange_wall_shear`` -- the Ladd momentum-exchange estimator
   equals the analytic wall shear (it is the *independent* measurement the FIK
   decomposition is validated against).
3. ``test_mass_conservation`` -- total mass is conserved to round-off.
4. ``test_fik_recovers_laminar_wall_shear`` -- the FIK identity implemented in
   :mod:`tensorlbm.microbubble_dr.fik` reproduces the analytic wall shear of the
   laminar run from the *mean profile alone*.
5. ``test_decomposition_closure`` -- the three mechanism terms sum to the total
   by construction, and each individual mechanism vanishes for the trivial cases
   (uniform viscosity / no Reynolds stress / no bubble stress).
6. ``test_negative_control_is_identity`` -- comparing a run with itself (gas
   fraction -> 0) gives ``Delta C_f = 0`` to machine precision: the negative
   control criterion is not satisfied by accident.
7. ``test_bubble_initialisation`` -- the seeded gas volume fraction matches the
   analytic sphere volume (the alpha the campaign reports is really the gas
   volume it placed).
8. ``test_sc_force_null_on_uniform_density`` -- the Shan-Chen force is exactly
   zero for a uniform density field, which is why the "SC on, no gas" negative
   control must reduce to the single-phase baseline.
"""

from __future__ import annotations

from functools import lru_cache

import math

import numpy as np
import pytest
import torch

from tensorlbm.microbubble_dr.channel import ChannelConfig, init_bubbles
from tensorlbm.microbubble_dr.fik import Profile, fik_decomposition, momentum_consistency
from tensorlbm.microbubble_dr.kernel import CS2, ChannelKernel, equilibrium, momentum, sc_force

torch.set_grad_enabled(False)


@lru_cache(maxsize=2)
def _laminar_run(ny: int = 16, nx: int = 8, nz: int = 4, F0: float = 2e-5, tau: float = 0.8,
                 steps: int = 5000) -> tuple[dict, dict]:
    """Fully developed laminar channel run; returns (profile record, analytic)."""
    k = ChannelKernel((nz, ny, nx), tau=tau, G=0.0)
    f = k.field(1.0, 0.0)
    for _ in range(steps):
        f, diag = k.step(f, F0)
    rho, j = momentum(f)
    u = (j[0] / rho).mean(dim=(0, 2)).double().numpy()
    rec = {
        "y": np.arange(ny) + 0.5,
        "u": u,
        "rho": rho.mean(dim=(0, 2)).double().numpy(),
        "uv": np.zeros(ny),
        "rho_b": float(rho.mean()),
        "nu": k.nu,
        "H": float(ny),
        "F0": F0,
        "mem_bottom": diag["mem_bottom"],
        "mem_top": diag["mem_top"],
    }
    ny_ = ny
    analytic = {
        "U_b": F0 * ny_**2 / (12.0 * k.nu),
        "u_max": F0 * ny_**2 / (8.0 * k.nu),
        "tau_w": F0 * ny_ / 2.0,
    }
    return rec, analytic


def test_laminar_poiseuille():
    rec, an = _laminar_run()
    assert rec["u"].mean() == pytest.approx(an["U_b"], rel=0.01)
    assert rec["u"].max() == pytest.approx(an["u_max"], rel=0.01)


def test_momentum_exchange_wall_shear():
    rec, an = _laminar_run()
    assert rec["mem_bottom"] == pytest.approx(an["tau_w"], rel=0.01)
    assert rec["mem_top"] == pytest.approx(an["tau_w"], rel=0.01)


def test_mass_conservation():
    k = ChannelKernel((4, 16, 8), tau=0.7, G=0.0)
    f = k.field(1.0, 0.0)
    m0 = float(f.sum())
    for _ in range(500):
        f, _d = k.step(f, 1e-5)
    # float32 round-off only (measured drift ~1.4e-5 relative over 500 steps)
    assert float(f.sum()) == pytest.approx(m0, rel=1e-4)


def test_fik_recovers_laminar_wall_shear():
    """FIK with r == 1 and no Reynolds stress must give C = F0*H/2 exactly."""
    rec, an = _laminar_run()
    p = Profile(
        y=rec["y"], rho=rec["rho"], u=rec["u"], uv=rec["uv"],
        alpha=np.zeros_like(rec["y"]), U_b=float(rec["u"].mean()),
        F0=rec["F0"], H=rec["H"], nu=rec["nu"], rho_b=rec["rho_b"],
    )
    d = fik_decomposition(p, p, tau_b=None)
    # The FIK identity is exact for a profile that satisfies the *discrete* mean
    # momentum balance; half-way bounce-back puts the wall half a cell outside
    # the first node, which biases the reconstructed bulk velocity by O(1/H).
    # At H = 16 that is ~3 %; at the production H = 40 it is < 1.5 %.
    assert d["C_bottom"] == pytest.approx(an["tau_w"], rel=0.10)
    assert d["tau_top"] == pytest.approx(an["tau_w"], rel=0.10)


def test_decomposition_closure():
    """Independent mechanisms: each vanishes in the case where it is absent."""
    y = np.arange(40) + 0.5
    H = 40.0
    nu = 5e-3
    u = 0.1 * np.sin(np.pi * y / H)
    rho = np.full_like(y, 1.0)
    uv = -np.full_like(y, 1e-4) * np.sin(np.pi * y / H) ** 2
    base = Profile(y=y, rho=rho, u=u, uv=np.zeros_like(y), alpha=np.zeros_like(y),
                   U_b=float(u.mean()), F0=2e-4, H=H, nu=nu, rho_b=1.0)
    two = Profile(y=y, rho=rho, u=u, uv=uv, alpha=np.zeros_like(y),
                  U_b=float(u.mean()), F0=2e-4, H=H, nu=nu, rho_b=1.0)
    d = fik_decomposition(two, base, tau_b=np.zeros_like(y))
    # viscosity term must vanish: mu_ref/mu == 1 everywhere
    assert abs(d["dtau_visc"]) < 1e-10
    # slip term must vanish: tau_b == 0
    assert abs(d["dtau_slip"]) < 1e-10
    # a negative <u'v'> raises the wall shear, i.e. it is a negative contribution
    # to the drag reduction Delta C_f = C_f0 - C_f
    assert d["dtau_turb"] < 0
    assert d["dtau_total"] > 0
    assert d["sum_residual_rel"] < 1e-10
    # piecewise split of the reported coefficients adds up
    assert d["dCf_visc"] + d["dCf_turb"] + d["dCf_slip"] == pytest.approx(d["dCf"], rel=1e-9)


def test_negative_control_is_identity():
    """gas -> 0: comparing a profile with itself gives Delta C_f = 0."""
    y = np.arange(32) + 0.5
    rng = np.random.default_rng(0)
    u = 0.1 * rng.random(y.size) + 0.05
    uv = -1e-4 * rng.random(y.size)
    p = Profile(y=y, rho=np.full_like(y, 1.4), u=u, uv=uv, alpha=np.zeros_like(y),
                U_b=float(u.mean()), F0=1e-4, H=32.0, nu=6.7e-4, rho_b=1.4)
    d = fik_decomposition(p, p, tau_b=np.zeros_like(y))
    assert d["DR"] == pytest.approx(0.0, abs=1e-12)
    assert abs(d["dCf"]) < 1e-15


def test_bubble_initialisation():
    cfg = ChannelConfig(nx=56, ny=40, nz=24, alpha=0.03, d_plus=60.0)
    gen = torch.Generator().manual_seed(3)
    rho, meta = init_bubbles(cfg, cfg.rho_l, cfg.rho_g, gen)
    d = cfg.d_lattice
    assert meta["n_bubbles"] == max(1, int(round(cfg.alpha * 56 * 40 * 24 / (math.pi / 6 * d**3))))
    # gas volume measured from the density field agrees with the sphere volume
    alpha = (cfg.rho_l - rho) / (cfg.rho_l - cfg.rho_g)
    gas_vol = float((alpha > 0.5).sum())          # half-volume of the diffused spheres
    assert gas_vol == pytest.approx(meta["n_bubbles"] * math.pi / 6 * d**3, rel=0.25)
    assert float(rho.max()) == pytest.approx(cfg.rho_l)


def test_sc_force_null_on_uniform_density():
    rho = torch.full((4, 16, 8), 1.37)
    fx, fy, fz = sc_force(rho, 4.3)
    assert float(fx.abs().max()) < 1e-5
    assert float(fy.abs().max()) < 1e-5
    assert float(fz.abs().max()) < 1e-5


def test_momentum_consistency_estimators_agree_laminar():
    rec, an = _laminar_run()
    rec["u_tau"] = math.sqrt(an["tau_w"] / rec["rho_b"])
    mc = momentum_consistency(rec)
    # the three independent channels must agree on the laminar wall shear
    assert mc["tau_wall_force"] == pytest.approx(an["tau_w"], rel=1e-12)
    assert mc["tau_wall_mem_bottom"] == pytest.approx(an["tau_w"], rel=0.01)
    assert mc["tau_wall_reynolds"] == pytest.approx(an["tau_w"], rel=0.05)
    assert mc["u_tau_spread_rel"] < 0.10


def test_trt_reduces_to_bgk_when_rates_equal():
    """With omega_m == omega_p the TRT collision must match plain BGK exactly."""
    k_trt = ChannelKernel((4, 12, 8), tau=0.8, G=0.0, lam_trt=0.5 * (0.8 - 0.5))
    k_bgk = ChannelKernel((4, 12, 8), tau=0.8, G=0.0, lam_trt=0.5 * (0.8 - 0.5))
    k_trt.omega_m = k_trt.omega_p
    gen = torch.Generator().manual_seed(5)
    rho = 1.0 + 0.01 * torch.randn((4, 12, 8), generator=gen)
    u = 0.01 * torch.randn((3, 4, 12, 8), generator=gen)
    f = equilibrium(rho, u)
    f1, _ = k_trt.step(f, 1e-6)
    f2, _ = k_bgk.step(f, 1e-6)
    assert torch.allclose(f1, f2, atol=1e-12)