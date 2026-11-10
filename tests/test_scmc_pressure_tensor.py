"""Verification-debt tests for the Shan-Chen two-component static-droplet axis.

Context: ``multiphase_benchmarks.run_static_droplet`` measures ΔP with the
ideal-gas density proxy ``cs²ρ`` and labels the physics ``NOT_VALIDATED``.
These tests pin *why*, with numbers, and pin the new mechanical pressure tensor
(:mod:`tensorlbm.scmc_pressure_tensor`) that the labels would need:

* the tensor's isotropic limit is exact (analytic check);
* the shipped SCMC default coupling ``G12 = -0.9`` is in the **mixing** regime:
  the two components interdiffuse, the interface disappears and there is no
  surface tension to measure (this is the mechanistic reason the SCMC droplet
  ΔP collapses to ~1e-5 while the benchmark still reports a ``sigma_eff``);
* a real interface exists only for ``|G12| >= ~2.4``, and the model diverges for
  ``|G12| >= 4`` — a narrow, previously undocumented parameter window;
* inside that window the discrete normal stress ``P_NN`` is **not** constant
  across a flat static interface (36–71 % residual), so an absolute surface
  tension cannot be certified at the working resolution (honest blocker);
* the color-gradient (CG) arm does hold a linear Young-Laplace law
  (``ΔP·R`` spread < 5 % over the radii tested).

All runs are small (<= 48², <= 3000 steps, single-threaded).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.multiphase_benchmarks import StaticDropletConfig, run_static_droplet
from tensorlbm.scmc_pressure_tensor import (
    CS2,
    SCMCPressureConfig,
    droplet_pressure_jump,
    flat_interface_pressure_profile,
    laplace_sweep,
    mechanical_pressure,
    scmc_pressure_tensor,
    simulate_flat_interface,
    simulate_scmc_droplet,
)
from tensorlbm.d2q9 import equilibrium


@pytest.fixture(autouse=True)
def _single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _uniform_pair(rho1: float, rho2: float, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros((n, n))
    return (
        equilibrium(torch.full((n, n), rho1), zero, zero),
        equilibrium(torch.full((n, n), rho2), zero, zero),
    )


# ---------------------------------------------------------------------------
# 1. The pressure tensor itself
# ---------------------------------------------------------------------------


def test_pressure_tensor_isotropic_limit_is_analytic() -> None:
    """A uniform two-component state has P_ab = (cs²Σρ + G cs² ρ₁ρ₂) δ_ab exactly."""
    G = -0.9
    rho1, rho2 = 0.7, 0.3
    f1, f2 = _uniform_pair(rho1, rho2)
    Pxx, Pyy, Pxy = scmc_pressure_tensor(f1, f2, G, "sym")
    expected = CS2 * (rho1 + rho2) + G * CS2 * rho1 * rho2
    assert Pxx.mean().item() == pytest.approx(expected, rel=1e-6)
    assert Pyy.mean().item() == pytest.approx(expected, rel=1e-6)
    assert Pxy.abs().max().item() == pytest.approx(0.0, abs=1e-12)


def test_pressure_tensor_matches_proxy_when_interaction_is_off() -> None:
    """G → 0 must recover the ideal-gas pressure (regression against a coding slip)."""
    f1, f2 = _uniform_pair(0.6, 0.4)
    Pxx, _, _ = scmc_pressure_tensor(f1, f2, 0.0, "sym")
    assert Pxx.mean().item() == pytest.approx(CS2 * 1.0, rel=1e-7)
    assert mechanical_pressure(f1, f2, 0.0, "sym").mean().item() == pytest.approx(CS2, rel=1e-7)


def test_flat_interface_carries_a_spurious_shear_stress() -> None:
    """Documented limitation: the two-point SC tensor is NOT shear-free at a flat interface.

    For x-uniform data the diagonal stencil pairs ``(1,±1)`` sample ``q(y±1)`` with
    ``c_ix c_iy = ±1``, so ``P_xy = (G/2)(1/36)[q(y+1) − q(y−1)]`` does **not**
    cancel. Measured at ``G12 = −3.0``: ``max|P_xy| ≈ 1.2e−3`` (~0.5 % of the mean
    normal stress), confined to the ~1-cell interface. This is a property of the
    discrete Shan-Chen stress (one more reason an absolute σ cannot be certified
    at this resolution), so the test pins it instead of pretending it is zero.
    """
    cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=800, G12=-3.0)
    f1, f2 = simulate_flat_interface(cfg)
    _, Pyy, Pxy = scmc_pressure_tensor(f1, f2, cfg.G12, "sym")
    ratio = Pxy.abs().max().item() / abs(Pyy.mean().item())
    assert 0.0 < ratio < 0.01, f"spurious shear ratio {ratio:.3e} outside the documented band"


# ---------------------------------------------------------------------------
# 2. Regime map of the SCMC model (the ship-blocking finding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("G", [-0.9, -2.0])
def test_default_couplings_are_in_the_mixing_regime(G: float) -> None:
    """|G12| <= 2.0: the components interdiffuse, no interface survives."""
    cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=2000, G12=G)
    f1, f2 = simulate_flat_interface(cfg)
    contrast = (f1.sum(0) - f2.sum(0)).abs().max().item()
    assert contrast < 0.05, f"G={G} still has an interface (contrast={contrast:.4f})"


@pytest.mark.parametrize("G", [-2.4, -3.0])
def test_interface_window_sustains_a_phase_boundary(G: float) -> None:
    """2.4 <= |G12| <= 3.5: a real interface exists (contrast > 0.5)."""
    cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=2000, G12=G)
    f1, f2 = simulate_flat_interface(cfg)
    assert torch.isfinite(f1).all() and torch.isfinite(f2).all()
    contrast = (f1.sum(0) - f2.sum(0)).abs().max().item()
    assert contrast > 0.5, f"G={G} lost the interface (contrast={contrast:.4f})"


def test_model_diverges_for_strong_coupling() -> None:
    """|G12| = 4 blows up at tau = 1 on the finer grid — an undocumented stability limit.

    The divergence is resolution dependent (NaN at ny=48/2000 steps, still finite
    but unrelaxed at ny=32/3000 steps: see ``artifacts/scmc_laplace_20261108``).
    """
    cfg = SCMCPressureConfig(ny=48, nx=8, n_steps=2000, G12=-4.0)
    f1, f2 = simulate_flat_interface(cfg)
    assert not torch.isfinite(f1).all() or not torch.isfinite(f2).all()


# ---------------------------------------------------------------------------
# 3. Why an absolute σ cannot be certified yet (honest blocker)
# ---------------------------------------------------------------------------


def test_normal_stress_is_not_flat_inside_the_interface_window() -> None:
    """Static equilibrium requires ∂_n P_NN = 0; the discrete SC tensor misses it by >10 %.

    This is the *documented blocker*: until the interface is resolved over
    several cells, the mechanical σ carries an O(10 %) discretisation error and
    no threshold tighter than 15 % can be claimed.
    """
    cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=2000, G12=-3.0)
    f1, f2 = simulate_flat_interface(cfg)
    prof = flat_interface_pressure_profile(f1, f2, cfg.G12, "sym")
    assert prof["flatness"] > 0.10  # type: ignore[operator]


def test_proxy_pressure_differs_from_mechanical_where_it_matters() -> None:
    """cs²ρ ≠ mechanical pressure inside the interface (the legacy proxy is not ΔP)."""
    cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=2000, G12=-3.0)
    f1, f2 = simulate_flat_interface(cfg)
    prof = flat_interface_pressure_profile(f1, f2, cfg.G12, "sym")
    p_n = prof["p_normal"]
    proxy = prof["proxy_cs2rho"]
    dev = (p_n - proxy).abs().max().item() / abs(proxy.mean().item())  # type: ignore[union-attr]
    assert dev > 0.05, f"proxy/mechanical deviation only {dev:.3%}"


# ---------------------------------------------------------------------------
# 4. The shipped droplet benchmark at its own defaults
# ---------------------------------------------------------------------------


def test_scmc_default_droplet_has_no_capillary_pressure() -> None:
    """G12=-0.9 (shipped default): ΔP ~ 3e-5, i.e. no measurable capillary pressure.

    The droplet keeps a *partial* component contrast (mean |ρ₁−ρ₂| ≈ 0.36 inside
    R/2 vs 0.40 initially) while the mechanical pressure jump collapses to ~1e-5:
    the SC coupling is too weak to build an interface with surface tension, so
    ``σ_eff`` from this arm is not a surface tension (see the artifact table).
    """
    cfg = SCMCPressureConfig(G12=-0.9, tau=1.0, rho_heavy=0.7, rho_light=0.3)
    f1, f2 = simulate_scmc_droplet(8.0, cfg, nx=48, ny=48, n_steps=3000)
    row = droplet_pressure_jump(f1, f2, 8.0, cfg.G12)
    assert abs(row["delta_p"]) < 1e-3
    assert abs(row["sigma_laplace"]) < 1e-2
    assert row["max_spurious_u"] < 1e-4


def test_shipped_benchmark_still_labels_physics_as_not_validated() -> None:
    """The honest label must survive this work until an absolute σ is certified."""
    rep = run_static_droplet(
        StaticDropletConfig(
            nx=32, ny=32, radii=(6.0,), n_steps=400, output_root="/tmp/_scmc_label",
            overwrite=True, device="cpu",
        )
    )
    for model in ("scmc", "cg"):
        assert rep["results"][model]["laplace_physical_validation"] == "NOT_VALIDATED"  # type: ignore[index]
        assert rep["results"][model]["pressure_diagnostic"] == "ideal_gas_density_proxy"  # type: ignore[index]


# ---------------------------------------------------------------------------
# 5. The color-gradient arm does hold the Laplace law (positive evidence)
# ---------------------------------------------------------------------------


def test_cg_arm_holds_laplace_linearity_within_5_percent() -> None:
    """CG: ΔP·R is constant across radii ⇒ ΔP ∝ 1/R (σ consistent to a few %)."""
    rep = run_static_droplet(
        StaticDropletConfig(
            nx=64, ny=64, radii=(8.0, 12.0), n_steps=4000,
            output_root="/tmp/_cg_laplace", overwrite=True, device="cpu",
        )
    )
    per = rep["results"]["cg"]["per_radius"]  # type: ignore[index]
    sigma = [float(d["sigma_eff"]) for d in per]  # type: ignore[index]
    spread = (max(sigma) - min(sigma)) / abs(sum(sigma) / len(sigma))
    assert all(s > 0 for s in sigma), f"non-positive σ_eff: {sigma}"
    assert spread < 0.05, f"ΔP·R spread {spread:.2%} over radii {[d['r'] for d in per]}"  # type: ignore[index]


def test_laplace_sweep_reports_a_finite_linearity_metric() -> None:
    """``laplace_sweep`` API contract (used by the artifact script)."""
    cfg = SCMCPressureConfig(ny=24, nx=8, n_steps=200, G12=-0.9)
    out = laplace_sweep(radii=(6.0, 8.0), config=cfg, n_steps=200)
    assert math.isfinite(float(out["laplace_linearity"]))
    assert len(out["per_radius"]) == 2  # type: ignore[arg-type]
    assert out["pairing"] == "sym"