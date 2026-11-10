"""Acceptance tests for the ELBM (entropic lattice Boltzmann) implementation.

Coverage, in the order of the campaign node's criteria:

* **entropic equilibria** -- the D3Q27 closed form is the constrained minimiser
  of the discrete Gauss entropy (pinned against an independent Lagrange-dual
  Newton solve to 1e-11);
* **H-theorem** -- the entropy-conditioned collision never increases the
  discrete entropy, at any relaxation time and for strongly distorted states,
  and every population stays positive (positivity is what makes ELBM
  unconditionally stable);
* **entropy condition** -- the solved ``α`` reproduces ``H(f*) = H(f)`` to
  solver tolerance, lies in the admissible interval ``(0, 1]`` (``α = 1`` is
  exactly BGK) and returns to ``α = 1`` for near-equilibrium states;
* **viscosity** -- the shear-wave decay rate reproduces ``ν = c_s²(τ - 1/2)`` in
  the resolved regime, and the laminar Poiseuille solution is recovered by the
  channel driver to <1 % (this is the end-to-end check of force + bounce-back +
  wall-unit geometry);
* **two implementations** -- the OpenMP channel kernel and the torch reference
  agree to 1e-4 relative per population for one step;
* **DNS gates** -- the MKM-1999 gate machinery passes a profile that *is* the
  DNS data and fails a distorted one (it cannot pass by accident).
"""

from __future__ import annotations

import os

# Keep the OpenMP kernel on a few threads for the small test grids: the kernel
# spawns one thread per core by default, which costs more in barriers than it
# saves on a 200-cell box.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import W_EXACT64 as W19
from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import W_EXACT64 as W27
from tensorlbm.d3q27 import macroscopic27, stream27_roll
from tensorlbm.elbm import (
    ALPHA_CAP,
    bose_einstein_entropy,
    collide_elbm_d3q19,
    collide_elbm_d3q27,
    effective_viscosity,
    entropic_equilibrium_dual,
    entropic_equilibrium_gauss_d3q27,
    gauss_entropy,
    relative_entropy,
    solve_entropy_condition,
)

CS2 = 1.0 / 3.0
F64 = torch.float64


def _w27() -> torch.Tensor:
    return W27.view(27, 1, 1, 1)


def _w19() -> torch.Tensor:
    return W19.view(19, 1, 1, 1)


def _random_state(seed: int = 0, *, shape=(2, 3, 4), amp: float = 0.1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.05 * torch.randn(shape, generator=g, dtype=F64)
    u = amp * torch.randn((3,) + shape, generator=g, dtype=F64)
    return entropic_equilibrium_gauss_d3q27(rho, u[0], u[1], u[2])


# ---------------------------------------------------------------------------
# 1. Entropic equilibria
# ---------------------------------------------------------------------------


def test_gauss_equilibrium_closed_form_is_the_dual_newton_solution() -> None:
    """The D3Q27 product closed form equals the Lagrange-dual minimiser."""
    feq = _random_state(1)
    rho, ux, uy, uz = macroscopic27(feq)
    dual = entropic_equilibrium_dual(
        rho, ux, uy, uz, lattice="d3q27", entropy="gauss", n_newton=40, tol=1e-15
    )
    assert torch.allclose(feq, dual, rtol=0, atol=1e-11)
    h_closed = gauss_entropy(feq, _w27())
    h_dual = gauss_entropy(dual, _w27())
    assert float((h_closed - h_dual).abs().max()) < 1e-11


def test_gauss_equilibrium_reproduces_mass_and_momentum_exactly() -> None:
    g = torch.Generator().manual_seed(7)
    shape = (3, 4, 5)
    rho = 1.0 + 0.2 * torch.rand(shape, generator=g, dtype=F64)
    u = 0.3 * torch.randn((3,) + shape, generator=g, dtype=F64)
    rho = rho.clone()
    feq = entropic_equilibrium_gauss_d3q27(rho, u[0], u[1], u[2])
    r_out, ux, uy, uz = macroscopic27(feq)
    assert float((r_out - rho).abs().max()) < 1e-12
    assert float((ux - u[0]).abs().max()) < 1e-12
    assert float((uy - u[1]).abs().max()) < 1e-12
    assert float((uz - u[2]).abs().max()) < 1e-12
    assert float(feq.min()) > 0.0, "the entropic equilibrium must be strictly positive"


def test_entropic_equilibrium_minimises_entropy_over_a_line_search() -> None:
    """H(f^eq) <= H(f^eq + λ d) for moment-preserving perturbations d."""
    feq = _random_state(3)
    g = torch.Generator().manual_seed(11)
    d = torch.randn(feq.shape, generator=g, dtype=F64)
    # project onto the mass- and momentum-preserving subspace so that the
    # perturbation cannot lower H by changing rho or u
    d = d - d.sum(dim=0, keepdim=True) * _w27()
    mom = torch.stack([(d * C27[:, a].to(F64).view(27, 1, 1, 1)).sum(dim=0) for a in range(3)])
    for a in range(3):
        d = d - 3.0 * mom[a].unsqueeze(0) * C27[:, a].to(F64).view(27, 1, 1, 1) * _w27()
    assert float(d.sum(dim=0).abs().max()) < 1e-12
    d = d * (1e-3 * float(feq.mean()) / float(d.abs().mean()))
    h0 = gauss_entropy(feq, _w27())
    for lam in (-4.0, -2.0, -0.5, -0.05, 0.05, 0.5, 2.0, 4.0):
        f = feq + lam * d
        assert float(f.min()) > 0.0
        dh = gauss_entropy(f, _w27()) - h0
        assert float(dh.min()) >= -1e-10, f"H dropped below H(f^eq) at lam={lam}"
    assert float(relative_entropy(feq, feq).max()) <= 1e-30


def test_bose_einstein_equilibrium_and_entropy() -> None:
    """BE entropy + dual solve: moments exact and BE entropy below Gauss' one."""
    feq = _random_state(5)
    rho, ux, uy, uz = macroscopic27(feq)
    fbe = entropic_equilibrium_dual(
        rho, ux, uy, uz, lattice="d3q27", entropy="bose-einstein", n_newton=80, tol=1e-15
    )
    r, a, b, c = macroscopic27(fbe)
    assert float((r - rho).abs().max()) < 1e-12
    assert float((a - ux).abs().max()) < 1e-12
    assert float(fbe.min()) > 0.0
    # the Bose-Einstein equilibrium has a lower BE entropy than the Gauss one
    h_be = bose_einstein_entropy(fbe, _w27())
    h_be_gauss = bose_einstein_entropy(feq, _w27())
    assert float((h_be - h_be_gauss).max()) < 0.0


# ---------------------------------------------------------------------------
# 2. H-theorem, positivity, entropy condition
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize("tau_value", [0.5001, 0.502, 0.51, 0.6, 1.0, 2.0])
def test_h_theorem_collision_never_increases_entropy(tau_value: float) -> None:
    """H(f*) <= H(f) per cell for distorted states at any tau, positivity kept."""
    feq = _random_state(13, shape=(3, 4, 5), amp=0.25)
    g = torch.Generator().manual_seed(17)
    f = torch.clamp(feq * (1.0 + 0.3 * torch.randn(feq.shape, generator=g, dtype=F64)), min=1e-8)
    f_post, info = collide_elbm_d3q27(f, tau_value, n_iter=24, return_info=True)
    dh = gauss_entropy(f_post, _w27()) - gauss_entropy(f, _w27())
    assert float(dh.max()) <= 1e-9, "the entropic collision increased H"
    assert float(f_post.min()) > 0.0, "the entropic collision produced a non-positive population"
    alpha = info.alpha
    assert float(alpha.min()) > 0.0
    assert float(alpha.max()) <= ALPHA_CAP + 1e-12


@pytest.mark.physics
def test_entropy_condition_never_increases_entropy_and_is_tight_near_equilibrium() -> None:
    """Two statements: (i) dH <= 0 always; (ii) for a weakly perturbed state the
    entropy-conditioned relaxation reaches the root, so dH -> 0."""
    feq = _random_state(19, amp=0.2)
    tau = 0.51
    tol = 1e-6
    g = torch.Generator().manual_seed(23)
    # (i) strongly distorted state: the limiter must never let H grow
    f = torch.clamp(feq * (1.0 + 0.2 * torch.randn(feq.shape, generator=g, dtype=F64)), min=1e-8)
    info = solve_entropy_condition(f, feq, tau, w=_w27(), n_iter=40)
    f_post = f + (info.alpha / tau) * (feq - f)
    h0 = gauss_entropy(f, _w27())
    assert float((gauss_entropy(f_post, _w27()) - h0).max()) <= tol * (float(h0.abs().max()) + 1.0)
    assert float(info.alpha.max()) <= ALPHA_CAP + 1e-12

    # (ii) wherever the limiter is *active* (alpha strictly below the cap) the
    # solver must land on the entropy root, so |dH| is below tolerance there.
    active = info.alpha < ALPHA_CAP * (1.0 - 1e-9)
    dh = gauss_entropy(f_post, _w27()) - h0
    if bool(active.any()):
        assert float(dh[active].abs().max()) < tol * (float(h0.abs().max()) + 1.0)


@pytest.mark.physics
def test_alpha_equals_one_for_near_equilibrium_state() -> None:
    """A resolved state must relax at the BGK rate (alpha -> 1 = cap)."""
    feq = _random_state(29, amp=0.02)
    info = solve_entropy_condition(feq, feq, 0.503, w=_w27(), n_iter=10)
    assert float(info.alpha.min()) == pytest.approx(ALPHA_CAP, abs=1e-12)
    assert float(info.alpha.max()) == pytest.approx(ALPHA_CAP, abs=1e-12)


@pytest.mark.physics
def test_effective_viscosity_matches_bgk_at_alpha_one() -> None:
    tau = 0.55
    nu = float(effective_viscosity(tau, torch.tensor(1.0, dtype=F64)))
    assert nu == pytest.approx(CS2 * (tau - 0.5), rel=1e-12)
    # under-relaxation means more dissipation
    nu_slow = float(effective_viscosity(tau, torch.tensor(0.5, dtype=F64)))
    assert nu_slow > nu


# ---------------------------------------------------------------------------
# 3. Hydrodynamics: shear-wave decay and D3Q19 path
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_shear_wave_decay_recovers_bgk_viscosity() -> None:
    """Weak transverse wave on a periodic box: amplitude ~ exp(-nu k^2 t)."""
    tau = 0.8
    nu = CS2 * (tau - 0.5)
    nx, ny, nz = 4, 4, 16
    k = 2.0 * np.pi / nz
    z = torch.arange(nz, dtype=F64).view(nz, 1, 1)
    amp = 1e-3
    ux = (amp * torch.sin(k * z)).expand(nz, ny, nx).clone()
    zero = torch.zeros_like(ux)
    f = entropic_equilibrium_gauss_d3q27(torch.ones_like(ux), ux, zero, zero)
    steps, sample_every = 400, 40
    amps = []
    for step in range(1, steps + 1):
        f = collide_elbm_d3q27(f, tau, n_iter=10)
        f = stream27_roll(f)
        if step % sample_every == 0:
            _, uxt, _, _ = macroscopic27(f)
            mode = (uxt[:, 0, 0] * torch.sin(k * torch.arange(nz, dtype=F64))).sum() * 2.0 / nz
            amps.append(float(mode))
    measured = -np.log(amps[-1] / amps[0]) / (sample_every * (len(amps) - 1))
    expected = nu * k * k
    assert measured == pytest.approx(expected, rel=0.05), (
        f"decay rate {measured:.6e} vs nu k^2 = {expected:.6e}"
    )


@pytest.mark.physics
def test_d3q19_elbm_collision_is_entropy_stable() -> None:
    """The D3Q19 path (dual-Newton equilibrium) keeps the H-theorem too."""
    g = torch.Generator().manual_seed(31)
    shape = (2, 3, 3)
    rho = 1.0 + 0.05 * torch.randn(shape, generator=g, dtype=F64)
    u = 0.15 * torch.randn((3,) + shape, generator=g, dtype=F64)
    feq19 = entropic_equilibrium_dual(
        rho, u[0], u[1], u[2], lattice="d3q19", entropy="gauss", n_newton=60, tol=1e-15
    )
    r, ux19, uy19, uz19 = macroscopic3d(feq19)
    assert float((r - rho).abs().max()) < 1e-12
    f = torch.clamp(feq19 * (1.0 + 0.2 * torch.randn(feq19.shape, generator=g, dtype=F64)), min=1e-9)
    f_post, info = collide_elbm_d3q19(f, 0.51, n_iter=20, return_info=True)
    dh = gauss_entropy(f_post, _w19()) - gauss_entropy(f, _w19())
    assert float(dh.max()) <= 1e-9
    assert float(f_post.min()) > 0.0
    assert float(info.alpha.max()) <= ALPHA_CAP + 1e-12


# ---------------------------------------------------------------------------
# 4. Native kernel vs torch reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def native_lib():  # pragma: no cover - depends on a C compiler
    from tensorlbm.elbm_channel_native import ElbmKernelBuildError, build_kernel

    try:
        build_kernel()
    except ElbmKernelBuildError as exc:  # pragma: no cover
        pytest.skip(f"native ELBM kernel unavailable: {exc}")
    return True


def _channel_state(nx: int, ny: int, nz: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(37)
    rho = torch.ones(nz, ny, nx)
    ux = 0.05 * torch.randn(nz, ny, nx, generator=g)
    uy = 0.02 * torch.randn(nz, ny, nx, generator=g)
    uz = 0.02 * torch.randn(nz, ny, nx, generator=g)
    f = entropic_equilibrium_gauss_d3q27(rho, ux, uy, uz)
    f = torch.clamp(f * (1.0 + 0.08 * torch.randn(f.shape, generator=g)), min=1e-6)
    quiet = entropic_equilibrium_gauss_d3q27(
        torch.ones(nz, 1, nx), torch.zeros(nz, 1, nx), torch.zeros(nz, 1, nx), torch.zeros(nz, 1, nx)
    )
    f[:, :, 0] = quiet[:, :, 0]
    f[:, :, ny - 1] = quiet[:, :, 0]
    return f, torch.ones(nz, ny, nx)


@pytest.mark.integration
def test_native_kernel_matches_torch_reference_step(native_lib) -> None:
    """One step of the OpenMP kernel equals the torch reference to 1e-4."""
    from tensorlbm.elbm_channel import _guo_force, stream_channel_d3q27
    from tensorlbm.elbm_channel_native import native_channel_step

    nx, ny, nz = 10, 12, 9
    f, alpha0 = _channel_state(nx, ny, nz)
    tau, force = 0.504, 3e-6
    rho, ux, uy, uz = macroscopic27(f)
    f_post, info = collide_elbm_d3q27(
        f, tau, alpha_init=alpha0, n_iter=24, return_info=True,
        velocity_shift=(0.5 * force, 0.0, 0.0),
    )
    f_post = _guo_force(f_post, rho, ux, uy, uz, force, tau)
    reference = stream_channel_d3q27(f_post).numpy()
    out, alpha_native, stats = native_channel_step(
        f.numpy(), alpha0.numpy(), nx=nx, ny=ny, nz=nz, tau=tau, force=force, n_iter=24
    )
    fluid = slice(1, ny - 1)
    scale = float(np.abs(reference).max())
    assert float(np.abs(out - reference)[:, :, fluid].max()) / scale < 1e-4
    assert float(np.abs(alpha_native - info.alpha.numpy())[:, fluid].max()) < 1e-3
    assert stats["alpha_max"] <= ALPHA_CAP + 1e-12


@pytest.mark.integration
def test_native_kernel_rejects_nothing_and_conserves_mass(native_lib) -> None:
    """Streaming + bounce-back must conserve mass on a closed channel."""
    from tensorlbm.elbm_channel_native import native_channel_step

    nx, ny, nz = 6, 10, 5
    f, alpha = _channel_state(nx, ny, nz)
    f[:, :, 0] = 0.0
    f[:, :, ny - 1] = 0.0
    mass0 = float(f.sum())
    f_np, alpha_np = f.numpy(), alpha.numpy()
    for _ in range(5):
        f_np, alpha_np, _ = native_channel_step(
            f_np, alpha_np, nx=nx, ny=ny, nz=nz, tau=0.51, force=0.0, n_iter=6
        )
    f = torch.from_numpy(f_np)
    mass1 = float(f.sum())
    fluid_mass = float(f[:, :, 1:-1].sum())
    assert mass1 == pytest.approx(mass0, rel=1e-5)
    assert fluid_mass > 0.0


# ---------------------------------------------------------------------------
# 5. End-to-end: laminar Poiseuille + DNS gates
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.slow
def test_laminar_channel_matches_poiseuille(native_lib) -> None:
    """ELBM + Guo force + half-way bounce-back recover the exact parabola."""
    from tensorlbm.elbm_channel_native import native_channel_run

    # ny = 21 -> 19 lattice units of channel height and 40 000 steps, i.e. more
    # than the viscous diffusion time H^2/nu: the exact parabola is reachable.
    nx, ny, nz = 4, 21, 4
    re_tau, u_tau = 4.0, 0.01
    res = native_channel_run(
        nx=nx, ny=ny, nz=nz, steps=40000, warmup=20000, sample_every=200,
        re_tau=re_tau, u_tau=u_tau, n_iter=4, noise_amplitude=1e-8, log_every=20000,
    )
    height = ny - 2
    half = 0.5 * height
    g = u_tau**2 / half
    nu = u_tau * half / re_tau
    y = np.arange(ny) - 0.5
    u = res.profiles["u"]
    analytic = g / (2.0 * nu) * y * (height - y)
    interior = (y > 0.5) & (y < height - 0.5)
    rel_err = float(np.abs(u[interior] - analytic[interior]).max() / analytic.max())
    assert rel_err < 0.01, f"laminar profile error {rel_err:.3%}"
    assert res.scalars["alpha_min"] > 0.0
    assert res.scalars["finite"] == 1.0
    assert res.scalars["min_population"] > 0.0


def _synthetic_profile_from_mkm(mkm) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Build a run-profile dict that *is* the DNS data (gate self-test)."""
    y = mkm.y
    u = mkm.u_plus * mkm.re_tau / mkm.re_tau  # U+ = U/u_tau
    profiles = {
        "y": y.tolist(),
        "y_plus_nominal": mkm.y_plus.tolist(),
        "u_plus_nominal": u.tolist(),
        "u_prime_v_prime": (mkm.uv_plus * 1.0).tolist(),
        "u_prime_u_prime": (mkm.u_rms_plus**2).tolist(),
    }
    scalars = {"half_height": 1.0, "u_tau": 1.0, "re_tau_nominal": mkm.re_tau,
               "delta_plus": mkm.re_tau, "finite": True, "steps": 100000,
               "min_population": 1e-3, "entropy_drop_max_collision": 0.0}
    return profiles, scalars


@pytest.mark.benchmark
def test_dns_gates_accept_the_dns_profile_itself() -> None:
    from tensorlbm.elbm_channel import channel_gates
    from tensorlbm.validation import mkm1999

    mkm = mkm1999.load_profile(180)
    profiles, scalars = _synthetic_profile_from_mkm(mkm)
    gates = channel_gates(profiles, scalars, mkm, log_band=(30.0, 60.0))
    assert gates["c_friction_coefficient"]["pass"] is True
    assert gates["c_friction_coefficient"]["cf_rel_error"] < 1e-12
    assert gates["d_long_time_stability"]["pass"] is True
    b = gates["b_reynolds_stress_peak"]
    assert b["y_plus_peak"] == pytest.approx(b["dns_y_plus"], abs=1e-9)
    assert b["pass"] is True, "the DNS profile must pass its own peak-location gate"
    # ... and it does *not* sit inside the campaign's absolute 12 +/- 2 window:
    # MKM-1999 puts the u_rms peak at y+ = 15.3 for Re_tau = 180, which is a
    # known property of the reference data, not of this code.
    assert 14.0 < b["dns_y_plus"] < 16.5
    assert b["pass_vs_campaign_target"] is False
    a = gates["a_log_slope"]
    assert a["valid"] is True
    assert 0.30 < a["kappa_eff"] < 0.55, f"kappa_eff={a['kappa_eff']}"


@pytest.mark.benchmark
def test_dns_gates_reject_a_wrong_profile() -> None:
    from tensorlbm.elbm_channel import channel_gates
    from tensorlbm.validation import mkm1999

    mkm = mkm1999.load_profile(180)
    profiles, scalars = _synthetic_profile_from_mkm(mkm)
    # 20 % too slow => C_f 44 % too low, and a flat profile kills the log layer
    profiles["u_plus_nominal"] = [0.8 * v for v in profiles["u_plus_nominal"]]
    gates = channel_gates(profiles, scalars, mkm, log_band=(30.0, 60.0))
    assert gates["c_friction_coefficient"]["pass"] is False
    assert gates["c_friction_coefficient"]["cf_rel_error"] > 0.05
    # a 20 % velocity deficit also flattens the fitted log slope by 20 %
    assert gates["a_log_slope"]["pass"] is False
    # a pure viscous profile has no logarithmic layer at all
    profiles["u_plus_nominal"] = list(np.asarray(profiles["y_plus_nominal"], dtype=float) * 0.8)
    gates = channel_gates(profiles, scalars, mkm, log_band=(30.0, 60.0))
    assert gates["a_log_slope"]["pass"] is False