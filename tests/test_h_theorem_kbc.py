"""H-theorem verification face for the KBC collision operator (node P67-N02).

Covers criteria C1 (entropy monotonicity), C2 (entropy production = dissipation
identities), C3 (equilibrium recovery), C4 (high-Reynolds KBC stability vs BGK
divergence), C5 (adaptive ``alpha`` in [0, 1]) and C7 (``alpha = 0``
degeneration control), plus cross-validation of the generic kernels against the
shipped :mod:`tensorlbm.entropic_kbc` implementation.

These tests are deliberately the *fast* subset of the campaign that is archived
in ``artifacts/h_theorem_kbc/campaign.json`` (produced by
``scripts/run_h_theorem_kbc_face.py``); the long runs live there.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import entropic_kbc as ek
from tensorlbm import h_theorem as hl

DTYPE = torch.float64
LATTICES = ("d2q9", "d3q19", "d3q27")


def _shapes(name: str, nx: int = 8, ny: int = 6, nz: int = 4) -> tuple[int, ...]:
    spec = hl.lattice_spec(name)
    return (ny, nx) if spec.dim == 2 else (nz, ny, nx)


def _random_state(
    name: str, *, seed: int = 0, amplitude: float = 0.02
) -> tuple[torch.Tensor, hl.LatticeSpec]:
    """Random *strictly positive* state: ``f = f^eq (1 + amplitude * p)``.

    The multiplicative form keeps every population positive (``amplitude < 1``),
    which the log-based entropy identities require; a purely additive random
    perturbation would drive some of the small-weight populations negative and
    make ``f ln f`` meaningless.
    """
    spec = hl.lattice_spec(name, dtype=DTYPE)
    shape = _shapes(name)
    g = torch.Generator().manual_seed(seed)
    rho = torch.ones(shape, dtype=DTYPE)
    u = 0.01 * torch.randn((spec.dim, *shape), generator=g, dtype=DTYPE)
    feq = hl.equilibrium(rho, u, spec)
    p = torch.tanh(torch.randn((spec.q, *shape), generator=g, dtype=DTYPE))
    return feq * (1.0 + amplitude * p), spec


# ---------------------------------------------------------------------------
# registry / functionals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATTICES)
def test_lattice_registry_and_weights(name: str) -> None:
    spec = hl.lattice_spec(name)
    assert spec.q == spec.c.shape[0] == spec.w.numel()
    assert spec.c.shape[1] == spec.dim
    assert math.isclose(float(spec.w.sum()), 1.0, rel_tol=1e-15)
    # discrete moments of the weights: sum w = 1, sum w c_a c_b = cs^2 delta_ab
    for a in range(spec.dim):
        assert math.isclose(float((spec.w * spec.c[:, a]).sum()), 0.0, abs_tol=1e-15)
        for b in range(spec.dim):
            expected = 1.0 / 3.0 if a == b else 0.0
            assert math.isclose(float((spec.w * spec.c[:, a] * spec.c[:, b]).sum()), expected, abs_tol=1e-15)


@pytest.mark.parametrize("name", LATTICES)
def test_entropy_functionals_at_equilibrium(name: str) -> None:
    spec = hl.lattice_spec(name)
    shape = _shapes(name)
    u = torch.zeros((spec.dim, *shape), dtype=DTYPE)
    u[0] = 0.03
    feq = hl.equilibrium(torch.ones(shape, dtype=DTYPE), u, spec)
    assert hl.entropy_relative(feq, feq) == pytest.approx(0.0, abs=1e-18)
    assert hl.entropy_excess_quadratic(feq, feq) == pytest.approx(0.0, abs=1e-18)
    # KL divergence is non-negative away from equilibrium
    spec2 = spec
    rho, uu = hl.macroscopic(feq, spec2)
    perturbed = feq * (1.0 + 0.05 * torch.cos(torch.arange(shape[-1], dtype=DTYPE)).view(
        *([1] * (spec.dim - 1)), shape[-1]
    ))
    assert hl.entropy_relative(perturbed, feq) > 0.0


@pytest.mark.parametrize("name", LATTICES)
def test_entropy_is_streaming_invariant(name: str) -> None:
    """Periodic streaming is an index permutation: H must be unchanged."""
    f, spec = _random_state(name, amplitude=0.05)
    f = hl.stream(f, spec)
    assert hl.entropy_boltzmann(f, spec) == pytest.approx(hl.entropy_boltzmann(hl.stream(f, spec), spec), rel=1e-14)
    # streaming twice is still a permutation -> same entropy
    assert hl.entropy_scale(f, spec) == pytest.approx(hl.entropy_scale(hl.stream(f, spec), spec), rel=1e-14)


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATTICES)
def test_kbc_decomposition_is_exact_and_shear_consistent(name: str) -> None:
    f, spec = _random_state(name, seed=3)
    rho, u = hl.macroscopic(f, spec)
    feq = hl.equilibrium(rho, u, spec)
    s, k, h, diag = hl.kbc_decompose(f, spec, feq)
    delta = f - feq

    assert diag["decomposition_residual"] < 1e-14
    assert torch.allclose(s + k + h, delta, atol=1e-14, rtol=0.0)
    # the deviatoric projection reproduces the deviatoric stress exactly
    assert diag["shear_stress_residual"] < 1e-12
    # s carries no mass / momentum, h carries no second-order moment
    for a in range(spec.dim):
        assert abs(float(s.sum())) < 1e-12
        assert float((s * spec.c_view(a, f.ndim)).sum().abs()) < 1e-12
    pi_h = hl.second_order_moment(h, spec)
    assert float(pi_h.abs().max()) < 1e-12


def test_alpha_solver_newton_matches_bisection() -> None:
    spec = hl.lattice_spec("d2q9")
    # small amplitude: the positivity domain does not clip the entropy minimum,
    # so the minimiser property is meaningful (at large amplitude the populations
    # hit the positivity boundary and the constrained minimum sits on the bound)
    f, spec = _random_state("d2q9", seed=7, amplitude=1e-3)
    rho, u = hl.macroscopic(f, spec)
    feq = hl.equilibrium(rho, u, spec)
    s, _k, h, _ = hl.kbc_decompose(f, spec, feq)
    tau = 0.8
    sigma = 1.0 - 1.0 / tau
    f_base = feq + sigma * s
    init = torch.full_like(rho, min(max(sigma, 0.0), 1.0))
    w = spec.weight_view(f.ndim)
    a_newton = hl._solve_alpha_entropy(f_base, h, w, init, method="newton")
    a_bisect = hl._solve_alpha_entropy(f_base, h, w, init, method="bisection", max_iter=80, tol=1e-14)
    assert float((a_newton - a_bisect).abs().max()) < 1e-10
    # and the entropy really is lower than at the bracket end points
    for alpha in (0.0, 1.0):
        H_alpha = float(
            (f_base + alpha * h).clamp_min(1e-300).mul(torch.log((f_base + alpha * h).clamp_min(1e-300) / w)).sum()
        )
        H_min = float(
            (f_base + a_newton.unsqueeze(0) * h)
            .clamp_min(1e-300)
            .mul(torch.log((f_base + a_newton.unsqueeze(0) * h).clamp_min(1e-300) / w))
            .sum()
        )
        assert H_min <= H_alpha + 1e-12


# ---------------------------------------------------------------------------
# C2: exact identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATTICES)
def test_c2_telescoping_and_kbc_ledger(name: str) -> None:
    f, spec = _random_state(name, seed=11, amplitude=0.05)
    tau = 0.7
    f_post, alpha, internals = hl.collide_kbc(f, tau, spec, return_internals=True)

    bal = hl.entropy_balance(f, f_post, spec)
    # normalise by the L1 scale of the summed terms (the entropy itself may pass
    # through zero, so |H| is not a stable reference)
    assert abs(bal.identity_residual) / hl.entropy_scale(f, spec) < 1e-13

    ledger = hl.kbc_production_identity(f, f_post, tau, spec, alpha, internals)
    assert ledger.identity_rel_residual < 1e-13
    assert ledger.operator_residual < 1e-15
    assert ledger.moment_residual < 1e-13
    # the operator really is the claimed closed form
    assert ledger.alpha_min >= -1e-12 and ledger.alpha_max <= 1.0 + 1e-12


@pytest.mark.parametrize("name", LATTICES)
def test_c2_bgk_sigma_squared_identity(name: str) -> None:
    """BGK relaxes the whole deviation: H2(f*) = sigma^2 H2(f) exactly."""
    f, spec = _random_state(name, seed=13, amplitude=0.05)
    tau = 0.63
    f_post = hl.collide_bgk(f, tau, spec)
    diss = hl.dissipation_identity(f, f_post, tau, spec)
    assert abs(diss.identity_residual) / max(abs(diss.H2_pre), 1e-30) < 1e-13
    assert diss.H2_post == pytest.approx(diss.sigma**2 * diss.H2_pre, rel=1e-13)
    # production is positive for a normal relaxation (tau > 0.5)
    assert diss.production > 0.0


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATTICES)
def test_kbc_and_bgk_conserve_mass_and_momentum(name: str) -> None:
    f, spec = _random_state(name, seed=17, amplitude=0.05)
    rho0, u0 = hl.macroscopic(f, spec)
    for collision in ("kbc", "bgk"):
        f_post = hl.collide_kbc(f, 0.8, spec)[0] if collision == "kbc" else hl.collide_bgk(f, 0.8, spec)
        rho1, u1 = hl.macroscopic(f_post, spec)
        assert float((rho1 - rho0).abs().max()) < 1e-13
        for a in range(spec.dim):
            assert float(((u1[a] - u0[a]) * rho0).abs().max()) < 1e-13


# ---------------------------------------------------------------------------
# cross-validation against the shipped production kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kernel", [("d3q19", ek.collide_kbc_d3q19), ("d3q27", ek.collide_kbc_d3q27)])
def test_generic_kbc_matches_production_kernel(name: str, kernel) -> None:
    """The verification face must certify the shipped operator, not a clone."""
    f, spec = _random_state(name, seed=23, amplitude=0.05)
    tau = 0.7
    mine, alpha = hl.collide_kbc(f, tau, spec)
    theirs = kernel(f, tau)
    rho, u = hl.macroscopic(f, spec)
    feq = hl.equilibrium(rho, u, spec)

    # both states: same mass/momentum, same entropy direction, near-identical H
    rho1, u1 = hl.macroscopic(mine, spec)
    rho2, u2 = hl.macroscopic(theirs, spec)
    assert float((rho1 - rho2).abs().max()) < 1e-12
    assert float((u1 - u2).abs().max()) < 1e-12
    H_mine = hl.entropy_boltzmann(mine, spec)
    H_theirs = hl.entropy_boltzmann(theirs, spec)
    H_pre = hl.entropy_boltzmann(f, spec)
    assert H_mine < H_pre + 1e-12
    assert H_theirs < H_pre + 1e-12
    assert abs(H_mine - H_theirs) <= 1e-6 * abs(H_pre)
    # shear stress (hence viscosity) is identical by construction: exact-tau retention
    pi_mine = hl.second_order_moment(mine - feq, spec)
    pi_theirs = hl.second_order_moment(theirs - feq, spec)
    assert float((pi_mine - pi_theirs).abs().max()) < 1e-10 * max(1.0, float(pi_mine.abs().max()))
    assert float(alpha.min()) >= -1e-12 and float(alpha.max()) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# C1 / C4 / C5 / C7 on the campaign configuration (short horizon)
# ---------------------------------------------------------------------------

CFG_HIGH_RE = dict(
    lattice="d2q9",
    shape=(32, 32),
    ic="shear_wave",
    u0=0.15,
    wavelength=4,
    tau=0.501,
    steps_kbc=300,
    steps_bgk_max=30,
    sample=30,
    eq_tau=1.0,
    eq_steps=800,
    eq_u0=0.01,
    eq_wavelength=16,
)


@pytest.fixture(scope="module")
def campaign() -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return hl.run_campaign(hl.CampaignConfig(**CFG_HIGH_RE))


def test_c1_entropy_monotone(campaign: dict) -> None:
    c1 = campaign["c1"]
    assert c1["violations"] == 0, f"{c1['violations']} steps increased H (max {c1['max_positive_excess_abs']:.3e})"
    # and the functional actually decreases over the run (not a no-op)
    assert c1["H_change_over_run"] < 0.0


def test_c2_pass_and_level(campaign: dict) -> None:
    c2 = campaign["c2"]
    assert c2["pass"], c2
    assert c2["telescoping_rel"] <= 1e-10
    assert c2["kbc_ledger_rel"] <= 1e-10
    assert c2["bgk_sigma2_rel"] <= 1e-10


def test_c3_equilibrium_recovery(campaign: dict) -> None:
    c3 = campaign["c3"]
    assert c3["pass"], c3
    assert c3["H_excess_ratio"] <= 1e-6
    assert c3["u_max_final"] < 1e-3


def test_c4_c6_high_re_stability(campaign: dict) -> None:
    assert campaign["c4_kbc"]["survived"], "KBC diverged on the high-Re case"
    assert campaign["c6_bgk"]["diverged"], "BGK did not diverge inside the control window"
    assert campaign["c6_bgk"]["diverged_at"] <= 100
    assert campaign["c4"]["pass"]


def test_c5_alpha_in_bounds(campaign: dict) -> None:
    c5 = campaign["c5"]
    assert c5["pass"], c5
    assert c5["alpha_min"] >= 0.0
    assert c5["alpha_max"] <= 1.0
    # the entropy condition must be active, i.e. alpha is not pinned at a constant
    assert c5["alpha_max"] - c5["alpha_min"] > 1e-6


def test_c7_alpha_zero_degeneration(campaign: dict) -> None:
    c7 = campaign["c7"]
    assert c7["pass"]
    assert c7["post_collision_diff_entropic_vs_alpha0"] > 0.0
    # alpha = 0 is the "ghost mode fully relaxed" operator: deterministic and reproducible
    f, spec = _random_state("d2q9", seed=31, amplitude=0.05)
    a = hl.collide_kbc_fixed_alpha(f, 0.501, spec, 0.0)
    b = hl.collide_kbc_fixed_alpha(f, 0.501, spec, 0.0)
    assert torch.equal(a, b)


def test_campaign_verdict_keys(campaign: dict) -> None:
    verdict = campaign["verdict"]
    for key in (
        "C1_H_monotone",
        "C2_production_dissipation_identity",
        "C3_equilibrium_recovery",
        "C4_high_re_kbc_stable_bgk_diverges",
        "C5_alpha_in_bounds",
        "C6_bgk_negative_control",
        "C7_alpha0_degeneration",
    ):
        assert key in verdict
    assert verdict["C1_H_monotone"] and verdict["C2_production_dissipation_identity"]
    assert verdict["C4_high_re_kbc_stable_bgk_diverges"] and verdict["C5_alpha_in_bounds"]


# ---------------------------------------------------------------------------
# viscosity calibration against the analytical reference
# ---------------------------------------------------------------------------


def test_kbc_viscosity_matches_analytic_reference() -> None:
    rows = hl.shear_wave_viscosity_study(
        taus=(0.7, 0.8, 1.0),
        u0=0.01,
        wavelengths=(16, 32),
        steps=600,
        sample=30,
    )
    for row in rows:
        assert abs(row["rel_error"]) < 0.03, row
    # error shrinks with resolution: the residual is O(k^2) spatial truncation
    for tau in (0.7, 0.8, 1.0):
        e16 = next(r for r in rows if r["tau"] == tau and r["wavelength"] == 16)["rel_error"]
        e32 = next(r for r in rows if r["tau"] == tau and r["wavelength"] == 32)["rel_error"]
        assert abs(e32) < abs(e16), (tau, e16, e32)
    # the viscosity is monotone in tau: nu = cs^2 (tau - 1/2)
    nus = [r["nu_measured"] for r in rows if r["wavelength"] == 32]
    assert nus[0] < nus[1] < nus[2]


def test_stability_sweep_envelope() -> None:
    """BGK dies at grid-scale amplitude, KBC does not, on the same initial data."""
    cfg = hl.CampaignConfig(lattice="d2q9", shape=(32, 32), ic="shear_wave", u0=0.15, wavelength=4, tau=0.501)
    rows = hl.run_stability_sweep(cfg, u0_list=(0.02, 0.15), kbc_steps=120, bgk_steps=40)
    low, high = rows
    assert low["bgk_diverged_at"] is None  # smooth/weak: BGK is fine
    assert high["bgk_diverged_at"] is not None and high["bgk_diverged_at"] <= 40
    assert high["kbc_diverged_at"] is None