"""Invariant tests for :mod:`tensorlbm.lbm_config` (single-point configuration).

Every expected number below is re-derived here from first principles (lattice
Boltzmann definitions in the LBM literature) rather than read back from the
module under test, so the tests assert *invariants*, not "the code ran".
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import d2q9
from tensorlbm.lbm_config import CS2, DEFAULT_MRT_RATES, LBMConfig

C = d2q9.C.to(torch.float64)          # (9, 2)
W = torch.tensor(d2q9.W_EXACT, dtype=torch.float64).view(9, 1, 1)
OPP = d2q9.OPPOSITE
CX = C[:, 0].view(9, 1, 1)
CY = C[:, 1].view(9, 1, 1)


def _field(ny: int = 6, nx: int = 5, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    f = torch.full((9, ny, nx), 1.0 / 9.0, dtype=torch.float64)
    f = f + 1e-3 * torch.randn((9, ny, nx), generator=g, dtype=torch.float64)
    f[0] = 1.0 - f[1:].sum(dim=0)  # keep density near 1
    return f


# ---------------------------------------------------------------- viscosity
@pytest.mark.parametrize("omega", [0.4, 1.0, 1.5, 1.9])
def test_relaxation_and_viscosity_follow_closed_form(omega: float) -> None:
    cfg = LBMConfig(relaxation_rate=omega)
    tau = 1.0 / omega
    assert cfg.tau == pytest.approx(tau, rel=0, abs=1e-15)
    # nu = cs^2 (tau - 1/2), cs^2 = 1/3
    assert cfg.nu == pytest.approx((tau - 0.5) / 3.0, rel=0, abs=1e-15)
    assert CS2 == pytest.approx(1.0 / 3.0, rel=0, abs=0.0)


def test_viscosity_from_momentum_diffusion_signal() -> None:
    """Transverse (shear) mode must decay as exp(-nu k^2 t), nu from the config.

    Polarization along x with wavevector along y is the solenoidal eigenmode of
    the D2Q9 BGK operator: it decays at the *shear* rate with only O(k^4)
    lattice dispersion, so the measured decay cross-checks the config-derived
    viscosity against physics rather than against the code under test.
    """
    omega = 1.2
    cfg = LBMConfig(relaxation_rate=omega)
    ny, nx, steps = 32, 4, 200
    k = 2.0 * math.pi / ny
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    amp = 1e-6
    ux0 = (amp * torch.sin(k * y)).expand(ny, nx).contiguous()
    f = cfg.equilibrium(torch.ones((ny, nx), dtype=torch.float64), ux0, torch.zeros_like(ux0))

    def mode_amplitude(state: torch.Tensor) -> float:
        _, ux, _ = cfg.velocity(state)
        return float(2.0 * (ux * torch.sin(k * y)).mean())

    a0 = mode_amplitude(f)
    for _ in range(steps):
        f = cfg.step(f)
    measured = mode_amplitude(f) / a0
    expected = math.exp(-cfg.nu * k * k * steps)
    assert measured == pytest.approx(expected, rel=3e-2), (measured, expected)


# ------------------------------------------------------------------- TRT
@pytest.mark.parametrize("omega", [0.8, 1.5])
def test_trt_magic_parameter_relations(omega: float) -> None:
    lam = 3.0 / 16.0
    cfg = LBMConfig(method="TRT", relaxation_rate=omega, lambda_trt=lam)
    tau_plus = 1.0 / omega
    assert cfg.tau_minus == pytest.approx(0.5 + lam / (tau_plus - 0.5), abs=1e-15)
    assert cfg.omega_minus == pytest.approx(1.0 / cfg.tau_minus, rel=1e-15)
    # the magic parameter is the product of the two half-relaxation times
    assert (tau_plus - 0.5) * (cfg.tau_minus - 0.5) == pytest.approx(lam, abs=1e-15)
    # the physical viscosity must not depend on the anti-symmetric rate
    assert cfg.nu == pytest.approx((tau_plus - 0.5) / 3.0, abs=1e-15)


def test_trt_and_srt_agree_on_conserved_moments() -> None:
    f = _field()
    srt = LBMConfig(method="SRT", relaxation_rate=1.4).collide(f)
    trt = LBMConfig(method="TRT", relaxation_rate=1.4).collide(f)
    for kicked in (srt, trt):
        assert float(kicked.sum()) == pytest.approx(float(f.sum()), abs=1e-13)


# -------------------------------------------------------------- equilibrium
def test_equilibrium_reproduces_the_three_lowest_moments() -> None:
    cfg = LBMConfig()
    g = torch.Generator().manual_seed(11)
    rho = 1.0 + 1e-3 * torch.randn((3, 4), generator=g, dtype=torch.float64)
    ux = 1e-3 * torch.randn((3, 4), generator=g, dtype=torch.float64)
    uy = 1e-3 * torch.randn((3, 4), generator=g, dtype=torch.float64)
    feq = cfg.equilibrium(rho, ux, uy)

    assert torch.allclose(feq.sum(dim=0), rho, atol=1e-15, rtol=0)
    assert torch.allclose((feq * CX).sum(dim=0), rho * ux, atol=1e-16, rtol=0)
    assert torch.allclose((feq * CY).sum(dim=0), rho * uy, atol=1e-16, rtol=0)
    # second moment: cs^2 rho delta + rho u u  (exact to O(u^3), truncated)
    pixx = (feq * CX * CX).sum(dim=0)
    piyy = (feq * CY * CY).sum(dim=0)
    pixy = (feq * CX * CY).sum(dim=0)
    assert torch.allclose(pixx, CS2 * rho + rho * ux * ux, atol=1e-16, rtol=0)
    assert torch.allclose(piyy, CS2 * rho + rho * uy * uy, atol=1e-16, rtol=0)
    assert torch.allclose(pixy, rho * ux * uy, atol=1e-16, rtol=0)


@pytest.mark.parametrize("method", ["SRT", "TRT", "MRT"])
@pytest.mark.parametrize("omega", [0.9, 1.5, 1.85])
def test_equilibrium_is_a_fixed_point_of_every_collision(method: str, omega: float) -> None:
    cfg = LBMConfig(method=method, relaxation_rate=omega)
    g = torch.Generator().manual_seed(3)
    rho = 1.0 + 1e-4 * torch.randn((4, 4), generator=g, dtype=torch.float64)
    ux = 1e-4 * torch.randn((4, 4), generator=g, dtype=torch.float64)
    uy = 1e-4 * torch.randn((4, 4), generator=g, dtype=torch.float64)
    feq = cfg.equilibrium(rho, ux, uy)
    assert torch.allclose(cfg.relax(feq, feq), feq, atol=1e-14, rtol=0)


def test_srt_relax_scales_the_non_equilibrium_part_by_one_minus_omega() -> None:
    omega = 1.3
    cfg = LBMConfig(relaxation_rate=omega)
    g = torch.Generator().manual_seed(5)
    rho = torch.ones((2, 2), dtype=torch.float64)
    ux = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    uy = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    feq = cfg.equilibrium(rho, ux, uy)
    delta = 1e-6 * torch.randn((9, 2, 2), generator=g, dtype=torch.float64)
    out = cfg.relax(feq + delta, feq)
    assert torch.allclose(out - feq, (1.0 - omega) * delta, atol=1e-15, rtol=0)


def test_trt_relaxes_the_two_eigen_subspaces_at_tau_plus_and_tau_minus() -> None:
    omega, lam = 1.3, 3.0 / 16.0
    cfg = LBMConfig(method="TRT", relaxation_rate=omega, lambda_trt=lam)
    g = torch.Generator().manual_seed(6)
    rho = torch.ones((2, 2), dtype=torch.float64)
    ux = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    uy = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    feq = cfg.equilibrium(rho, ux, uy)
    delta = 1e-6 * torch.randn((9, 2, 2), generator=g, dtype=torch.float64)
    out = cfg.relax(feq + delta, feq) - feq
    d_plus = 0.5 * (delta + delta[OPP])
    d_minus = 0.5 * (delta - delta[OPP])
    expected = (1.0 - omega) * d_plus + (1.0 - cfg.omega_minus) * d_minus
    assert torch.allclose(out, expected, atol=1e-16, rtol=0)


def test_mrt_relax_applies_each_moment_its_own_rate() -> None:
    omega = 1.3
    rates = {"s_e": 1.1, "s_eps": 1.2, "s_q": 1.4}
    cfg = LBMConfig(method="MRT", relaxation_rate=omega, mrt_rates=rates)
    from tensorlbm.solver import _get_d2q9_mrt_matrices

    matrix, _ = _get_d2q9_mrt_matrices(torch.device("cpu"), torch.float64)
    g = torch.Generator().manual_seed(8)
    rho = torch.ones((2, 2), dtype=torch.float64)
    ux = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    uy = 1e-4 * torch.randn((2, 2), generator=g, dtype=torch.float64)
    feq = cfg.equilibrium(rho, ux, uy)
    delta = 1e-6 * torch.randn((9, 2, 2), generator=g, dtype=torch.float64)
    out = cfg.relax(feq + delta, feq) - feq
    m_delta = matrix @ (out.reshape(9, -1))
    m_ref = matrix @ (delta.reshape(9, -1))
    s_vec = torch.tensor(
        [0.0, rates["s_e"], rates["s_eps"], 0.0, rates["s_q"], 0.0,
         rates["s_q"], omega, omega],
        dtype=torch.float64,
    ).unsqueeze(1)
    assert torch.allclose(m_delta, (1.0 - s_vec) * m_ref, atol=1e-14, rtol=1e-10)


# ------------------------------------------------------------------- force
def test_guo_source_first_moment_is_the_corrected_force() -> None:
    omega = 1.25
    fx, fy = 2.0e-5, -3.0e-5
    cfg = LBMConfig(relaxation_rate=omega, force=(fx, fy), force_model="guo")
    f = _field()
    s = cfg.force_source(f)
    # mass must be conserved by the source term
    assert float(s.sum()) == pytest.approx(0.0, abs=1e-18)
    # first moment of the textbook Guo source = (1 - omega/2) F (Guo et al. 2002)
    scale = 1.0 - omega / 2.0
    n = f.shape[1] * f.shape[2]
    assert float((s * CX).sum()) == pytest.approx(scale * fx * n, rel=1e-12, abs=0.0)
    assert float((s * CY).sum()) == pytest.approx(scale * fy * n, rel=1e-12, abs=0.0)


def test_guo_split_source_first_moment_is_the_full_force() -> None:
    fx = 1e-5
    cfg = LBMConfig(relaxation_rate=1.7, force=(fx, 0.0), force_model="guo_split")
    f = _field()
    s = cfg.force_source(f)
    n = f.shape[1] * f.shape[2]
    assert float((s * CX).sum()) == pytest.approx(fx * n, rel=1e-12)
    assert float(s.sum()) == pytest.approx(0.0, abs=1e-18)


def test_simple_force_model_injects_full_force() -> None:
    fx = 1e-5
    cfg = LBMConfig(relaxation_rate=1.7, force=(fx, 0.0), force_model="simple")
    f = _field()
    s = cfg.force_source(f)
    n = f.shape[1] * f.shape[2]
    assert s.shape == f.shape
    assert float((s * CX).sum()) == pytest.approx(fx * n, rel=1e-12)


@pytest.mark.parametrize(
    "method,model",
    [
        ("SRT", "guo"),
        ("TRT", "guo"),
        ("SRT", "guo_split"),
        ("TRT", "guo_split"),
        ("MRT", "guo_split"),
        ("SRT", "simple"),
    ],
)
def test_total_momentum_injection_per_step_is_exactly_the_force(
    method: str, model: str
) -> None:
    """Collision + source must inject exactly F per node and step.

    This is the free-of-charge invariant that makes the two halves of the
    force bookkeeping consistent; violating it (e.g. corrected source with a
    raw-velocity equilibrium) injects only (1 - omega/2) F.
    """
    omega = 1.35
    fx, fy = 3e-6, -2e-6
    cfg = LBMConfig(method=method, relaxation_rate=omega, force=(fx, fy), force_model=model)
    f = _field(ny=8, nx=8)
    n = f.shape[1] * f.shape[2]
    steps = 4
    j0 = float((f * CX).sum()), float((f * CY).sum())
    m0 = float(f.sum())
    for _ in range(steps):
        # collision + source only (streaming conserves momentum exactly)
        f = cfg.collide(f) + cfg.force_source(f)
    j1 = float((f * CX).sum()), float((f * CY).sum())
    assert float(f.sum()) == pytest.approx(m0, rel=1e-13)
    assert j1[0] - j0[0] == pytest.approx(steps * fx * n, rel=1e-11)
    assert j1[1] - j0[1] == pytest.approx(steps * fy * n, rel=1e-11)


def test_velocity_output_follows_the_force_bookkeeping() -> None:
    fx = 4e-5
    f = _field()
    # textbook Guo: output carries the half-step force
    guo = LBMConfig(relaxation_rate=1.6, force=(fx, 0.0), force_model="guo")
    assert guo.velocity_output_includes_force is True
    _, ux_half, _ = guo.velocity(f)
    _, ux_raw, _ = guo.velocity(f, include_force=False)
    rho, _, _ = guo.momentum(f)
    assert torch.allclose(ux_half - ux_raw, 0.5 * fx / rho, atol=1e-18, rtol=0)
    # split/naive schemes: raw velocity is the physical one
    for model in ("guo_split", "simple"):
        cfg = LBMConfig(relaxation_rate=1.6, force=(fx, 0.0), force_model=model)
        assert cfg.velocity_output_includes_force is False


def test_force_source_vanishes_without_a_force_model() -> None:
    cfg = LBMConfig(relaxation_rate=1.3)
    f = _field()
    assert torch.count_nonzero(cfg.force_source(f)) == 0


def test_step_conserves_mass_on_a_periodic_domain() -> None:
    cfg = LBMConfig(relaxation_rate=1.1, force=(1e-5, 0.0), force_model="guo")
    f = _field(ny=8, nx=8)
    m0 = float(f.sum())
    for _ in range(5):
        f = cfg.step(f)
    assert float(f.sum()) == pytest.approx(m0, rel=1e-14)


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("omega", [0.0, -0.5, 2.0, 2.5])
def test_relaxation_rate_outside_stability_window_is_rejected(omega: float) -> None:
    with pytest.raises(ValueError, match="relaxation_rate"):
        LBMConfig(relaxation_rate=omega)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "BGK"},
        {"stencil": "D3Q19"},
        {"force_model": "shan-chen"},
        {"lambda_trt": 0.0},
        {"force": (1e-6, 0.0)},                       # force without a model
        {"force_model": "guo", "force": (float("nan"), 0.0)},
        {"mrt_rates": {"s_bogus": 1.0}},
        # the defect class the single point exists to prevent:
        {"force_model": "guo", "velocity_output_includes_force": False},
        {"force_model": "guo_split", "velocity_output_includes_force": True},
        {"force_model": "guo", "method": "MRT"},
    ],
)
def test_invalid_declarations_fail_at_the_single_point(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        LBMConfig(**kwargs)


def test_inconsistent_mix_is_named_in_the_error_message() -> None:
    with pytest.raises(ValueError, match="inconsistent force bookkeeping"):
        LBMConfig(force_model="guo", velocity_output_includes_force=False)
    with pytest.raises(ValueError, match="MRT"):
        LBMConfig(force_model="guo", method="MRT")


def test_config_is_frozen_and_copyable() -> None:
    cfg = LBMConfig(relaxation_rate=1.2)
    with pytest.raises(Exception):
        cfg.relaxation_rate = 1.5  # type: ignore[misc]
    other = cfg.with_(method="TRT")
    assert cfg.method == "SRT" and other.method == "TRT"
    assert other.nu == pytest.approx(cfg.nu, rel=0, abs=1e-15)
    # switching force model re-resolves the velocity convention
    forced = cfg.with_(force=(1e-6, 0.0), force_model="guo")
    assert forced.velocity_output_includes_force is True
    assert cfg.with_(force_model="guo_split").velocity_output_includes_force is False


def test_json_round_trip_preserves_the_scheme() -> None:
    cfg = LBMConfig(
        method="TRT",
        relaxation_rate=1.35,
        force=(3e-5, -1e-6),
        force_model="guo_split",
    )
    back = LBMConfig.from_dict(cfg.as_dict())
    assert back == cfg
    assert back.scheme() == cfg.scheme()


def test_scheme_dict_reports_derived_quantities() -> None:
    cfg = LBMConfig(method="TRT", relaxation_rate=1.5, force=(1e-5, 0.0), force_model="guo")
    s = cfg.scheme()
    assert s["tau"] == pytest.approx(2.0 / 3.0, abs=1e-15)
    assert s["nu"] == pytest.approx((2.0 / 3.0 - 0.5) / 3.0, abs=1e-15)
    assert s["tau_minus"] == pytest.approx(0.5 + (3.0 / 16.0) / (2.0 / 3.0 - 0.5), abs=1e-15)
    tau_minus = 0.5 + (3.0 / 16.0) / (2.0 / 3.0 - 0.5)
    assert s["momentum_relaxation_rate"] == pytest.approx(1.0 / tau_minus, abs=1e-15)
    assert s["force_source_scale"] == pytest.approx(1.0 - 0.5 / tau_minus, abs=1e-15)
    assert s["velocity_output_includes_force"] is True
    assert s["mrt_rates"] is None


def test_mrt_defaults_are_used_and_overridable() -> None:
    cfg = LBMConfig(method="MRT", relaxation_rate=1.4)
    assert cfg.scheme()["mrt_rates"] == dict(DEFAULT_MRT_RATES)
    custom = cfg.with_(mrt_rates={"s_e": 1.2, "s_eps": 1.2, "s_q": 1.2})
    f = _field()
    same = LBMConfig(
        method="MRT", relaxation_rate=1.4, mrt_rates={"s_e": 1.2, "s_eps": 1.2, "s_q": 1.2}
    )
    assert torch.allclose(custom.collide(f), same.collide(f))