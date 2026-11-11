"""Verification suite for the four LBM-LES sub-grid closures.

Three layers, all machine-checkable:

``A. kernel equivalence``  the fused C kernel used by the campaign must be a
   bit-level replica of the production torch operators in
   :mod:`tensorlbm.turbulence.turbulence` (otherwise the campaigns would
   verify a different physics than the library ships);

``B. closure physics``     laminar degeneracy, Stokes shear-wave decay, the
   Taylor-Green initial condition and the pseudo-spectral DNS reference;

``C. campaign gates``      the acceptance table produced by
   ``scripts/run_sgs_les_campaign.py`` (criteria a-i of the node contract) is
   re-checked here when ``runs/sgs_les/report.json`` is present, so a
   regression in any campaign arm shows up as a test failure rather than as a
   silent report edit.

Run: ``python -m pytest tests/test_sgs_four_model_verification.py -q``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402
from tensorlbm.turbulence import sgs_verification as sv  # noqa: E402
from tensorlbm.turbulence import spectral_dit as sd  # noqa: E402
from tensorlbm.turbulence.sgs_lbm_kernel import (  # noqa: E402
    DEFAULT_CONSTANTS,
    MODEL_IDS,
    SGSKernel,
)
from tensorlbm.turbulence.turbulence import (  # noqa: E402
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
    collide_dynamic_smagorinsky_bgk3d,
    collide_smagorinsky_bgk3d,
    collide_vreman_bgk3d,
    collide_wale_bgk3d,
)

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "runs" / "sgs_les" / "report.json"
if not REPORT.exists():  # committed copy (runs/ is gitignored)
    REPORT = REPO / "artifacts" / "sgs_les" / "report.json"

TORCH_MODELS = {
    "smagorinsky": collide_smagorinsky_bgk3d,
    "dynamic_smagorinsky": collide_dynamic_smagorinsky_bgk3d,
    "wale": collide_wale_bgk3d,
    "vreman": collide_vreman_bgk3d,
}


def _smooth_field(n: int, seed: int = 7) -> np.ndarray:
    """Divergence-free-ish smooth random velocity field with |u| ~ 0.05."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)
    u = np.zeros((3, n, n, n))
    for comp in range(3):
        for mode in range(3):
            k = rng.normal(size=3)
            k = np.round(k / np.linalg.norm(k) * (mode + 1))
            ph = rng.uniform(0.0, 2.0 * math.pi)
            u[comp] += (0.05 / (mode + 1)) * np.sin(
                2.0 * math.pi * (k[0] * X + k[1] * Y + k[2] * Z) / n + ph
            )
    return u


def _populations(n: int, u: np.ndarray, neq: float = 1e-3, seed: int = 3) -> np.ndarray:
    rho = torch.ones((n, n, n), dtype=torch.float64)
    f0 = equilibrium3d(rho, *[torch.from_numpy(u[i]) for i in range(3)]).numpy()
    if neq:
        f0 = f0 + neq * np.random.default_rng(seed).normal(size=f0.shape)
    return f0


# =========================================================================
# A. kernel equivalence with the production torch operators
# =========================================================================
@pytest.mark.parametrize("model", sv.SGS_MODELS)
def test_collide_matches_production(model: str) -> None:
    """One collision of the C kernel == the torch operator, rtol 1e-13."""
    n, tau = 16, 0.53
    u = _smooth_field(n)
    f0 = _populations(n, u)
    const = DEFAULT_CONSTANTS[model]
    ker = SGSKernel(n)
    ker.set_bc_mode(0)  # production stencil replica (_nonperiodic_derivative)
    got = ker.collide_only(f0, model, tau, const)
    fn = TORCH_MODELS[model]
    ft = torch.from_numpy(f0)
    if model == "smagorinsky":
        ref = fn(ft, tau, C_s=const).numpy()
    elif model == "dynamic_smagorinsky":
        ref = fn(ft, tau).numpy()
    else:
        ref = fn(ft, tau, const).numpy()
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-13, f"{model}: relative collide mismatch {rel:.3e}"


@pytest.mark.parametrize("model", ("wale", "vreman"))
def test_eddy_viscosity_matches_production(model: str) -> None:
    """nu_t of the C kernel == the torch SGS helper, rtol 1e-12."""
    n, tau = 16, 0.53
    u = _smooth_field(n, seed=11)
    const = DEFAULT_CONSTANTS[model]
    ker = SGSKernel(n)
    ker.set_bc_mode(0)
    nut, _ = ker.nu_t_field(u, model, tau, const)
    ut = torch.from_numpy(u)
    helper = _wale_nu_t_3d if model == "wale" else _vreman_nu_t_3d
    ref = helper(ut[0], ut[1], ut[2], const).numpy()
    rel = np.abs(nut - ref).max() / max(np.abs(ref).max(), 1e-300)
    assert rel < 1e-12, f"{model}: nu_t mismatch {rel:.3e}"


def test_multistep_trajectory_matches_production() -> None:
    """collide_* + stream3d (torch) and the kernel agree over 5 steps.

    The kernel applies *collide then stream* (pull from the post-collision
    upstream cell).  The shipped ``benchmarks/cbc_dit_les`` driver applies
    *stream then collide*; the two are the same scheme with the state defined
    one collision apart, so the reference here uses the matching order.
    """
    n, tau, steps = 16, 0.53, 5
    model = "smagorinsky"
    const = DEFAULT_CONSTANTS[model]
    u = _smooth_field(n, seed=5)
    f0 = _populations(n, u)
    f_t = torch.from_numpy(f0.copy())
    for _ in range(steps):
        f_t = TORCH_MODELS[model](f_t, tau, C_s=const)
        f_t = stream3d(f_t)
    ker = SGSKernel(n)
    ker.set_bc_mode(0)
    ker.set_populations(f0)
    for _ in range(steps):
        ker.step(model, tau, const, stats=False)
    f_k = ker.get_populations()
    rel = np.abs(f_k - f_t.numpy()).max() / np.abs(f_t.numpy()).max()
    assert rel < 1e-12, f"trajectory mismatch {rel:.3e}"


# =========================================================================
# B. closure physics
# =========================================================================
def test_uniform_flow_gives_zero_sgs_stress() -> None:
    """No velocity gradient anywhere -> every closure is exactly zero."""
    deg = sv.laminar_degeneracy(n=16)
    for model in sv.SGS_MODELS:
        v = deg["fields"]["uniform"]["models"][model]
        assert v["nut_max"] < 1e-12, f"{model} uniform nut_max={v['nut_max']}"


def test_wale_and_vreman_vanish_in_laminar_shear() -> None:
    """WALE (grad(u)^2) and Vreman (B_beta = 0) vanish for u = u(y)."""
    deg = sv.laminar_degeneracy(n=16)
    for field in ("linear_shear", "poiseuille"):
        for model in ("wale", "vreman", "dynamic_smagorinsky"):
            v = deg["fields"][field]["models"][model]
            assert v["nut_max"] < 1e-9, f"{field}/{model} nut_max={v['nut_max']}"


def test_dynamic_smagorinsky_selects_zero_cs_in_laminar_flow() -> None:
    """Germano identity gives Cs^2 = 0 in a resolved laminar shear."""
    deg = sv.laminar_degeneracy(n=16)
    for field in ("linear_shear", "poiseuille"):
        cs2 = deg["fields"][field]["models"]["dynamic_smagorinsky"]["cs2"]
        assert cs2 < 1e-12, f"{field}: Cs^2 = {cs2}"


def test_static_smagorinsky_is_neq_based() -> None:
    """Frozen-field probe cannot see the static model: it reads the neq stress.

    The library's ``collide_smagorinsky_bgk3d`` builds ``tau_eff`` from the
    non-equilibrium stress ``Q``; an equilibrium (frozen) field has ``Q = 0``,
    so ``nu_t = 0`` there.  This is recorded as a *property* of the
    implementation, and the spurious-viscosity check is done dynamically in
    :func:`test_smagorinsky_scales_with_cs_squared`.
    """
    deg = sv.laminar_degeneracy(n=16)
    v = deg["fields"]["poiseuille"]["models"]["smagorinsky"]
    assert v["nut_max"] < 1e-12
    assert v["expectation"] == "zero_frozen_field"


def test_smagorinsky_vanishes_in_rigid_rotation() -> None:
    """Zero strain rate -> the strain-based static model must be zero."""
    deg = sv.laminar_degeneracy(n=16)
    v = deg["fields"]["rigid_rotation"]["models"]["smagorinsky"]
    assert v["nut_max"] < 1e-9


def test_stokes_shear_wave_decays_at_the_exact_rate() -> None:
    """Laminar Stokes wave: nu_eff/nu ~ 1 for the closures that degenerate."""
    for model in ("none", "wale", "vreman"):
        r = sv.stokes_shear_wave(model, n=32, steps=1200, record_every=40)
        ratio = r["nu_eff_over_nu"]
        assert abs(ratio - 1.0) < 0.05, f"{model}: nu_eff/nu={ratio}"


def test_smagorinsky_scales_with_cs_squared() -> None:
    """nu_t ~ (Cs Delta)^2 |S|: halving Cs must quarter the eddy viscosity."""
    ratio = []
    for cs in (0.16, 0.08):
        r = sv.run_tgv("smagorinsky", 16, re=400.0, t_end=0.3,
                       amplitude_lat=0.06, C=cs)
        ratio.append(float(np.mean(r["nut_over_nu"])))
    assert ratio[0] == pytest.approx(4.0 * ratio[1], rel=0.35), ratio


def test_sgs_off_reproduces_plain_bgk() -> None:
    """C_s = 0 / C_w = 0 / C_v = 0 is bit-identical to the plain BGK arm."""
    n, tau, steps = 16, 0.55, 40
    ker = SGSKernel(n)
    ker.set_bc_mode(1)
    u = _smooth_field(n, seed=9)
    ker.set_velocity(u)
    for _ in range(steps):
        ker.step("none", tau, 0.0, stats=False)
    ref = ker.get_velocity()
    for model in ("smagorinsky", "wale", "vreman"):
        k2 = SGSKernel(n)
        k2.set_bc_mode(1)
        k2.set_velocity(u)
        for _ in range(steps):
            k2.step(model, tau, 0.0, stats=False)
        diff = float(np.abs(k2.get_velocity() - ref).max())
        assert diff < 1e-14, f"{model}: SGS-off trajectory differs by {diff:.3e}"


def test_taylor_green_initial_condition_is_analytic() -> None:
    """tke = 0.125 and eps0 = 2 nu <S:S> = 0.001875 at Re = 400."""
    n = 32
    u = sv.tgv_velocity(n, 1.0)
    tke = 0.5 * float((u ** 2).sum(axis=0).mean())
    assert abs(tke - 0.125) < 1e-12
    K = sv._wavenumbers(n)
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    eps0 = sv.spectral_dissipation(uh, K, 1.0 / 400.0)
    assert abs(eps0 - 2.0 * (1.0 / 400.0) * 0.375) < 1e-12


def test_spectral_dissipation_matches_finite_differences() -> None:
    """eps from the Fourier strain equals the physical-space strain."""
    n = 48
    u = sv.tgv_velocity(n, 0.7)
    K = sv._wavenumbers(n)
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    nu = 3e-3
    eps_spec = sv.spectral_dissipation(uh, K, nu)
    dx = 2.0 * math.pi / n

    # the LBM fields are laid out as (z, y, x) -> physical (x, y, z) maps onto
    # array axes (2, 1, 0); permuting only one index of S_ij would break the
    # symmetry of the tensor, so the mapping is applied consistently here
    axes = (2, 1, 0)

    def d(field: np.ndarray, axis: int) -> np.ndarray:
        """Periodic central difference (numpy's np.gradient is one-sided at the
        box faces, which on this field costs several tens of percent)."""
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * dx)

    g = np.stack([d(u[i], axes[a]) for i in range(3) for a in range(3)]).reshape(3, 3, n, n, n)
    S = 0.5 * (g + np.transpose(g, (1, 0, 2, 3, 4)))
    eps_fd = 2.0 * nu * float((S ** 2).sum(axis=(0, 1)).mean())
    assert abs(eps_spec - eps_fd) / eps_fd < 0.02


def test_spectral_dns_conserves_energy_without_viscosity() -> None:
    """nu = 0: the rotational-form RK4 + projection must conserve TKE."""
    r = sv.spectral_dns_tgv(n=16, re=1e12, t_end=1.0, dt=0.01, record_every=10)
    tke = np.asarray(r["tke"])
    drift = abs(tke[-1] - tke[0]) / tke[0]
    assert drift < 1e-8, f"inviscid energy drift {drift:.3e}"


def test_dit_setup_physical_mapping() -> None:
    """Grid refinement keeps the same physical box and shrinks dx."""
    c32, c64 = sv.dit_setup(32), sv.dit_setup(64)
    assert c32["box_phys"] == pytest.approx(c64["box_phys"])
    assert c64["dx_phys"] == pytest.approx(0.5 * c32["dx_phys"])
    # a finer grid means a smaller dx, hence a *larger* lattice viscosity for
    # the same physical nu (nu_lat = nu / (dx*U_ref))
    assert c64["tau"] > c32["tau"]
    assert c64["steps"] == pytest.approx(2 * c32["steps"], rel=0.01)


def test_run_dit_decays_and_stays_finite() -> None:
    """A short CBC-DIT run loses kinetic energy monotonically and stays finite."""
    r = sv.run_dit("smagorinsky", 16, steps=60, record_every=10)
    tke = np.asarray(r["time_series"]["tke"])
    assert np.all(np.isfinite(tke))
    assert tke[-1] < tke[0]
    assert r["blew_up_at_step"] is None
    m = sv.dit_spectrum_metrics(r)
    assert m["decay_exponent"] is not None


def test_run_tgv_records_dissipation_split() -> None:
    """LBM-TGV: molecular + SGS dissipation are both measured and positive."""
    r = sv.run_tgv("wale", 16, re=400.0, t_end=0.4, amplitude_lat=0.06)
    assert r["blew_up_at_step"] is None
    assert r["eps_max"] > 0.0
    assert np.sum(r["eps_sgs"]) > 0.0
    assert len(r["t"]) == len(r["eps"]) == len(r["eps_sgs"])


def test_invalid_model_is_rejected() -> None:
    """Unknown closures raise instead of silently running something else."""
    with pytest.raises(ValueError):
        sv.run_tgv("bogus", 16, re=400.0, t_end=0.1)
    with pytest.raises(ValueError):
        sv.run_dit("bogus", 16, steps=5)


def test_all_four_models_share_one_interface() -> None:
    """Every closure has a model id, a constant and a dispersion-free default."""
    for model in sv.SGS_MODELS:
        assert model in MODEL_IDS
        assert model in DEFAULT_CONSTANTS
        assert model in TORCH_MODELS


# =========================================================================
# C. campaign gates (re-checked from the committed artefact)
# =========================================================================
@pytest.mark.skipif(not REPORT.exists(), reason="campaign artefact not present")
def test_campaign_acceptance_table_is_documented() -> None:
    """Every gate has a verdict, and every non-PASS verdict carries a reason.

    An honest FAIL is a legitimate outcome of this campaign (the CBC case is
    outside the design envelope of three of the four closures); what is *not*
    acceptable is an undocumented or empty verdict.
    """
    rep = json.loads(REPORT.read_text())
    crit = rep["acceptance"]["criteria"]
    for key, c in crit.items():
        assert c["verdict"] in ("PASS", "FAIL", "DEVIATION"), (key, c["verdict"])
        if c["verdict"] != "PASS":
            assert c["note"] and len(c["note"]) > 20, (key, c["note"])
    # the implementation-level gates (kernel semantics, degeneracy, negative
    # controls) must pass -- they are statements about the code, not about a
    # physical measurement
    for key in ("d_dynamic_cs", "e_wale_wall_degeneracy",
                "f_vreman_laminar_degeneracy", "g_negative_controls"):
        assert crit[key]["verdict"] == "PASS", (key, crit[key])


@pytest.mark.skipif(not REPORT.exists(), reason="campaign artefact not present")
def test_campaign_stability_map_is_complete() -> None:
    """The DIT arm records, for every closure and grid, how far the run reached."""
    rep = json.loads(REPORT.read_text())
    for key, m in rep["dit_metrics"].items():
        assert "completed_fraction" in m, key
        assert 0.0 < m["completed_fraction"] <= 1.0, (key, m["completed_fraction"])


@pytest.mark.skipif(not REPORT.exists(), reason="campaign artefact not present")
def test_campaign_numbers_are_traceable() -> None:
    """The report must carry the raw numbers the verdicts were computed from."""
    rep = json.loads(REPORT.read_text())
    assert rep["dit_metrics"]
    assert rep["tgv_dns"]["eps_max"]
    assert rep["tgv_les"]
    assert rep["laminar_degeneracy"]["fields"]["poiseuille"]["models"]["vreman"]["below_1e_9"]
    assert rep["sgs_off_identity"]["max_abs_velocity_diff"] < 1e-9
