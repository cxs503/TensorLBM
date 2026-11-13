"""Acceptance gate for XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY.

Checks the two things that can be fast and are load-bearing:

* the library's ``u_eq="viscous"`` collision imposes exactly the REF-OSS
  ``vel_mix = Σ(Vel·Rho/τ)/Σ(Rho/τ)`` (black-box momentum inversion);
* the τ-mixed coupling conserves total momentum for unequal τ, while the
  barycentric ``u_eq="mixture"`` does not;
* the analytic channel reference used by the cross-validation is consistent
  with its own continuum limit (parabola with ν_eff, walls at node centres).

The expensive channel/separation arms live in
``xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/run_xval.py``; their JSON is
checked here when present, so a stale or failing artifact cannot masquerade as
a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.mcmp_tau_mixed import (
    CS2,
    channel_poiseuille_profile,
    collision_momentum_probe,
    mixture_effective_viscosity,
    mixture_effective_tau,
    oss_vel_mix,
    tau_mixed_velocity,
)
from tensorlbm.multiphase import collide_sc_two_component
from tensorlbm.solver import stream

XVAL_DIR = Path(__file__).resolve().parents[1] / "xval" / "XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY"


def _random_state(seed: int, rho=(1.0, 1.0), u=(0.05, -0.02), tau=(1.0, 0.7), dtype=torch.float64):
    torch.manual_seed(seed)
    ny, nx = 8, 6
    rho1 = rho[0] * (1.0 + 0.1 * torch.rand((ny, nx), dtype=dtype))
    rho2 = rho[1] * (1.0 + 0.1 * torch.rand((ny, nx), dtype=dtype))
    ux1 = u[0] * torch.rand((ny, nx), dtype=dtype)
    uy1 = 0.3 * u[0] * torch.rand((ny, nx), dtype=dtype)
    ux2 = u[1] * torch.rand((ny, nx), dtype=dtype)
    uy2 = 0.3 * u[1] * torch.rand((ny, nx), dtype=dtype)
    f1 = equilibrium(rho1, ux1, uy1)
    f2 = equilibrium(rho2, ux2, uy2)
    return f1, f2, tau


def test_oss_vel_mix_matches_stacked_formulation():
    rho1 = torch.rand(4, 5, dtype=torch.float64) + 0.5
    rho2 = torch.rand(4, 5, dtype=torch.float64) + 0.5
    ux1 = torch.rand(4, 5, dtype=torch.float64)
    ux2 = torch.rand(4, 5, dtype=torch.float64)
    a = oss_vel_mix([rho1, rho2], [ux1, ux2], [1.0, 0.7])
    b = oss_vel_mix(torch.stack([rho1, rho2]), torch.stack([ux1, ux2]), [1.0, 0.7])
    torch.testing.assert_close(a, b)
    # equal tau degenerates to the barycentric velocity
    bary = (rho1 * ux1 + rho2 * ux2) / (rho1 + rho2)
    torch.testing.assert_close(oss_vel_mix([rho1, rho2], [ux1, ux2], [1.0, 1.0]), bary)


def test_tau_mixed_velocity_two_component_helper():
    rho1 = torch.full((3, 3), 2.0, dtype=torch.float64)
    rho2 = torch.full((3, 3), 1.0, dtype=torch.float64)
    ux1 = torch.full((3, 3), 0.03, dtype=torch.float64)
    ux2 = torch.full((3, 3), -0.03, dtype=torch.float64)
    zero = torch.zeros_like(ux1)
    ux, uy = tau_mixed_velocity(rho1, ux1, zero, rho2, ux2, zero, 1.0, 0.5)
    w1, w2 = 2.0 / 1.0, 1.0 / 0.5
    assert ux[0, 0].item() == pytest.approx((w1 * 0.03 + w2 * -0.03) / (w1 + w2), rel=1e-14)
    assert torch.allclose(uy, zero)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_library_viscous_collision_imposes_ref_oss_velocity(seed):
    f1, f2, (t1, t2) = _random_state(seed)
    probe = collision_momentum_probe(f1, f2, t1, t2)
    assert probe["max_rel_dev"] < 1e-10, probe


def test_mixture_viscosity_continuum_limit():
    # equal tau: the mixture is a single-tau fluid
    assert mixture_effective_viscosity(1.0, 1.0) == pytest.approx(CS2 * 0.5)
    # rho-weighted arithmetic mean, not the tau mean nor the harmonic mean
    nu1, nu2 = CS2 * 0.5, CS2 * 0.2
    assert mixture_effective_viscosity(1.0, 0.7, 1.0, 1.0) == pytest.approx(0.5 * (nu1 + nu2))
    assert mixture_effective_viscosity(1.0, 0.7, 3.0, 1.0) == pytest.approx((3 * nu1 + nu2) / 4)
    assert mixture_effective_tau(1.0, 0.7) == pytest.approx(1.0 - 0.15)
    with pytest.raises(ValueError):
        mixture_effective_viscosity(0.4, 1.0)


def test_channel_reference_is_self_consistent():
    ny, gx, nu = 33, 1e-6, mixture_effective_viscosity(1.0, 0.7)
    u = channel_poiseuille_profile(ny, gx, nu)
    assert u.shape == (33, 1)
    assert float(u[0]) == 0.0 and float(u[-1]) == 0.0
    assert float(u.max()) == pytest.approx(gx * (ny - 1) ** 2 / (8 * nu), rel=1e-12)
    # the wall planes implied by the profile are the node centres
    nonzero = u[1:-1, 0]
    assert float(nonzero.min()) > 0.0


@pytest.mark.parametrize("mode,expect_conserved", [("viscous", True), ("mixture", False), ("self", True)])
def test_momentum_conservation_of_coupling_modes(mode, expect_conserved):
    n, steps, tau1, tau2 = 12, 60, 1.0, 0.7
    rho = torch.ones((n, n), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f1 = equilibrium(rho, torch.full_like(rho, 0.02), zero)
    f2 = equilibrium(rho, torch.full_like(rho, -0.01), zero)
    j0 = float((0.02 * rho).sum() - (0.01 * rho).sum())
    for _ in range(steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=0.0, tau1=tau1, tau2=tau2, u_eq=mode, force_scheme="shift"
        )
        f1 = stream(f1)
        f2 = stream(f2)
    r1, ux1, _ = macroscopic(f1)
    r2, ux2, _ = macroscopic(f2)
    drift = abs(float((r1 * ux1 + r2 * ux2).sum()) - j0) / abs(j0)
    if expect_conserved:
        assert drift < 1e-12, (mode, drift)
    else:
        assert drift > 1e-3, (mode, drift)


def test_xval_artifact_is_consistent_when_present():
    path = XVAL_DIR / "results" / "xval_results.json"
    if not path.exists():
        pytest.skip("xval results not generated in this checkout")
    res = json.loads(path.read_text())
    decl = res["predeclared"]["predeclared_thresholds"]
    assert res["parity"]["worst_rel_dev_float64"] <= decl["parity"]["max_rel_dev_float64"]
    for case in res["channel"]["cases"]:
        assert case["rel_err_max"] <= decl["channel"]["rel_err_max"], case["tag"]
    assert res["overall"] == "PASS", res["verdicts"]