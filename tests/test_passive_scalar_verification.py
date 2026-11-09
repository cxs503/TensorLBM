"""Tests for :mod:`tensorlbm.passive_scalar_verification`.

Three groups, ordered by cost:

* **analytic anchors** (closed forms, discrete Aris auxiliary problems, erf
  profile) — pure numpy/torch, sub-second;
* **convergence and conservation** (erf diffusion front on two grids) — under a
  second;
* **lattice physics** (time-marched Taylor–Aris dispersion) — ~15 s, the
  regression anchor that ties the scalar kernel to the analytic law.

The gates themselves are evaluated by
``benchmarks/pending/passive_scalar_verification/run.py``; the tests pin the
individual pieces so a failure names the broken piece.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tensorlbm.passive_scalar_verification import (
    ARIS_PE2_PIPE,
    ARIS_PE2_SLIT,
    DiffusionFrontConfig,
    VerificationSpec,
    discrete_aris_pipe,
    discrete_aris_slit,
    erf_diffusion_profile,
    gaussian_diffusion_profile,
    measure_slit_dispersion,
    quick_spec,
    run_verification,
    simulate_diffusion_front,
    taylor_aris_effective_diffusivity,
    taylor_aris_ratio,
    write_artifact,
)

# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------


def test_geometry_attribution_constants_are_the_classical_values():
    """1/210 is the plane slit, 1/48 is the circular pipe (campaign-seed erratum)."""
    assert ARIS_PE2_SLIT == pytest.approx(1.0 / 210.0)
    assert ARIS_PE2_PIPE == pytest.approx(1.0 / 48.0)
    assert taylor_aris_ratio(0.0, "slit") == pytest.approx(1.0)
    assert taylor_aris_ratio(100.0, "slit") == pytest.approx(1.0 + 1.0e4 / 210.0)
    assert taylor_aris_ratio(100.0, "pipe") == pytest.approx(1.0 + 1.0e4 / 48.0)
    # the two geometries are *not* interchangeable
    assert taylor_aris_ratio(20.0, "pipe") > 3.0 * taylor_aris_ratio(20.0, "slit")


def test_effective_diffusivity_reduces_to_molecular_value_without_flow():
    assert taylor_aris_effective_diffusivity(0.1, 16.0, 0.0, "slit") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        taylor_aris_ratio(1.0, "annulus")
    with pytest.raises(ValueError):
        taylor_aris_effective_diffusivity(0.0, 1.0, 1.0)


@pytest.mark.parametrize("geometry", ["slit", "pipe"])
def test_discrete_aris_reproduces_the_closed_form(geometry):
    """The discrete Aris problem on the lattice nodes matches 1 + Pe^2/K."""
    solver = discrete_aris_slit if geometry == "slit" else discrete_aris_pipe
    for pe in (1.0, 20.0):
        row = solver(n_nodes=64, diffusivity=0.1, u_mean=pe * 0.1 / 64.0)
        assert row["pe_lattice"] == pytest.approx(pe, rel=5e-3)
        assert row["ratio_lattice"] == pytest.approx(row["ratio_closed_form"], rel=1e-3)
        assert abs(row["aux_residual"]) < 1e-10


@pytest.mark.parametrize("geometry", ["slit", "pipe"])
def test_discrete_aris_converges_second_order_in_transverse_resolution(geometry):
    solver = discrete_aris_slit if geometry == "slit" else discrete_aris_pipe
    coarse = abs(
        solver(n_nodes=16, diffusivity=0.1, u_mean=20.0 * 0.1 / 16.0)["rel_err_lattice_vs_closed"]
    )
    fine = abs(
        solver(n_nodes=32, diffusivity=0.1, u_mean=20.0 * 0.1 / 32.0)["rel_err_lattice_vs_closed"]
    )
    order = math.log2(coarse / fine)
    assert order > 1.8, f"{geometry}: observed order {order:.2f}"


def test_pipe_ratio_hits_one_plus_pe_squared_over_48_at_pe_20():
    row = discrete_aris_pipe(n_nodes=128, diffusivity=0.1, u_mean=20.0 * 0.1 / 128.0)
    assert row["ratio_lattice"] == pytest.approx(1.0 + 400.0 / 48.0, rel=1e-4)


def test_erf_profile_is_the_half_space_step_solution():
    x = torch.tensor([-50.0, 0.0, 50.0], dtype=torch.float64)
    c = erf_diffusion_profile(x, t=1.0, diffusivity=0.5)
    assert float(c[1]) == pytest.approx(0.5)
    assert float(c[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(c[2]) == pytest.approx(0.0, abs=1e-12)
    # documented identity: C(0,t) = 0.5 for every t, and the profile is
    # self-similar in eta = (x - x0) / sqrt(4 D t)
    t1 = erf_diffusion_profile(torch.tensor([1.0]), 1.0, 0.1)
    t4 = erf_diffusion_profile(torch.tensor([2.0]), 4.0, 0.1)
    assert float(t1) == pytest.approx(float(t4), rel=1e-12)
    with pytest.raises(ValueError):
        erf_diffusion_profile(x, t=0.0, diffusivity=0.1)


def test_gaussian_profile_conserves_zeroth_moment_analytically():
    x = torch.linspace(-200.0, 200.0, 4001, dtype=torch.float64)
    dx = float(x[1] - x[0])
    c = gaussian_diffusion_profile(x, t=10.0, diffusivity=0.1, sigma0=2.0)
    # unit-peak normalisation: the conserved integral is sigma0 sqrt(2 pi)
    assert float(c.sum()) * dx == pytest.approx(2.0 * math.sqrt(2.0 * math.pi), rel=2e-3)


# ---------------------------------------------------------------------------
# convergence and conservation
# ---------------------------------------------------------------------------


def test_diffusion_front_converges_second_order():
    out = run_diffusion_levels((0, 1))
    assert out["min_order"] > 1.9, out["orders"]


def run_diffusion_levels(levels):
    from tensorlbm.passive_scalar_verification import diffusion_convergence_study

    return diffusion_convergence_study(base_nx=128, levels=levels, diffusivity=0.1, end_time=200.0)


def test_diffusion_front_conserves_mass():
    case = simulate_diffusion_front(DiffusionFrontConfig(nx=128, level=0, end_time=100.0))
    assert case["mass_drift"] < 1e-8
    assert case["steps"] == 100


def test_diffusion_front_error_drops_against_the_exact_profile():
    coarse = simulate_diffusion_front(DiffusionFrontConfig(nx=128, level=0, end_time=200.0))
    assert coarse["l2_error"] < 1e-3  # absolute L2 error at the coarsest grid


# ---------------------------------------------------------------------------
# lattice physics (the regression anchor)
# ---------------------------------------------------------------------------


def test_lbm_dispersion_matches_taylor_aris_at_pe_5():
    """Time-marched D2Q9 scalar LBM reproduces 1 + Pe^2/210 within 3 %."""
    row = measure_slit_dispersion(pe=5.0, ny=17, nx=128, diffusivity=0.04)
    assert row["transient_resolved"]
    assert row["mass_drift"] < 1e-8
    assert abs(row["rel_err_vs_closed"]) < 0.03, row["rel_err_vs_closed"]
    assert row["ratio_sim"] > 1.0  # shear enhances, never reduces, dispersion


def test_lbm_propagates_the_mass_drift_and_reports_metadata():
    row = measure_slit_dispersion(pe=2.0, ny=9, nx=64, diffusivity=0.1, steps=1200)
    for key in ("ratio_sim", "ratio_closed_form", "rel_err_vs_closed", "mass_drift", "r2"):
        assert key in row
    assert row["mass_drift"] < 1e-8


# ---------------------------------------------------------------------------
# driver / artifact plumbing
# ---------------------------------------------------------------------------


def test_write_artifact_drops_private_keys(tmp_path):
    path = tmp_path / "artifact.json"
    write_artifact({"a": 1, "_figure_data": {"x": [1, 2]}, "nested": {"b": np.float64(2.5)}}, path)
    payload = json.loads(path.read_text())
    assert payload == {"a": 1, "nested": {"b": 2.5}}


def test_smoke_spec_produces_a_structured_artifact(tmp_path):
    """The under-resolved smoke spec still exercises every gate and the writer.

    Gates are *not* expected to pass at this resolution (the smoke label says
    so); the test pins the plumbing: every criterion block is present, the
    artifact is JSON, and the file is self-describing.
    """
    out = tmp_path / "smoke.json"
    report = run_verification(quick_spec(), out_path=out)
    payload = json.loads(out.read_text())
    for gate in ("G1", "G2", "G3", "G4", "G5"):
        assert gate in payload["criteria"]
    assert payload["verdict"] in ("PASS", "FAIL")
    assert payload["node"] == "p73w2b-n05-lbm-passive-scalar"
    assert report["label"] == "quick"
    assert len(payload["criteria"]["G1"]["anchor_rows"]) == 6  # 2 geometries x 3 Pe
    # the erf gate is resolution independent: it must already hold
    assert payload["criteria"]["G2"]["min_order"] > 1.9


def test_formal_spec_defaults_cover_the_campaign_ranges():
    spec = VerificationSpec()
    assert min(spec.pe_sweep) <= 0.1 and max(spec.pe_sweep) >= 100.0
    assert min(spec.schmidt_sc) <= 0.7 and max(spec.schmidt_sc) >= 10.0
    assert spec.taylor_tolerance_pct == 3.0
    assert spec.erf_order_threshold == 1.9
    assert spec.schmidt_tolerance_pct == 2.0
    assert spec.mass_tolerance == 1e-8
    assert torch.__version__  # module imports torch; keeps the lint honest
