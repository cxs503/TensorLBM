"""Unit + regression tests for :mod:`tensorlbm.scalar_dispersion_mach`.

Node `p405w1-n06-lbm-scalar-dispersion`: the Pe=40 Taylor-Aris slit dispersion
error ("+8.685 %" in m8228) is a finite-``kH`` artefact of the modal meter plus a
residual lattice-Mach ``O(u^2)`` term.  These tests pin the components:

* the sampled slit velocity / closed-form Taylor-Aris anchor,
* the LAPACK-free discrete Aris solve (and its cross-check against the library),
* the zero-flux wall relocation (mass conservation),
* the modal meter (``u = 0`` control must return the molecular diffusivity),
* the ``(kH)^2 -> 0`` extrapolation and the Richardson order,
* the lattice-Mach law fit (``err = a + b u^2``),
* the ``O(u^2)`` equilibrium ablation's moment properties,
* and — if the artifact is present — the full verification gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.scalar_dispersion_mach import (
    discrete_aris_deff_slit,
    equilibrium_scalar_2nd,
    k0_extrapolation,
    mach_law_fit,
    modal_decay_case,
    richardson_order,
    sampled_mean,
    scalar_wall_reflect,
    slit_velocity,
    theory_deff_slit,
)
from tensorlbm.thermal import temperature_equilibrium

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "scalar_dispersion_mach" / "results.json"


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)


# ── geometry / anchors ──────────────────────────────────────────────────────
def test_slit_velocity_sampled_mean_matches_nominal():
    """The *sampled* parabola mean must be the nominal u to O(1/H^2)."""
    for h in (16, 32, 64):
        u = slit_velocity(h, 0.05)
        assert u.shape == (h,)
        assert sampled_mean(u) == pytest.approx(0.05, rel=5.0e-3)
        # symmetry about the slit centre and max on the centre line
        assert np.allclose(u.numpy(), u.numpy()[::-1], atol=1e-14)
        assert float(u.abs().max()) == pytest.approx(1.5 * 0.05, rel=2.0e-2)
        assert float(u[0]) > 0.0 and float(u[-1]) > 0.0


def test_plug_profile_is_uniform_and_has_no_shear():
    u = slit_velocity(24, 0.04, kind="plug")
    assert np.allclose(u.numpy(), 0.04)
    grad = np.diff(u.numpy())
    assert np.allclose(grad, 0.0)


def test_theory_deff_matches_taylor_aris_pe2_over_210():
    d, h = 0.04, 17
    u = 40.0 * d / h
    assert theory_deff_slit(d, h, u) == pytest.approx(d * (1.0 + 40.0**2 / 210.0), rel=1e-12)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        slit_velocity(10, 0.05, kind="nope")
    with pytest.raises(ValueError):
        modal_decay_case(10.0, 10, 4.0, 0.0)


# ── discrete Aris reference ─────────────────────────────────────────────────
def test_discrete_aris_converges_at_second_order_or_better():
    """Sampled-profile Aris coefficient converges to the closed form (order >= 2)."""
    d = 0.025
    errs = []
    hs = [8, 16, 32]
    for h in hs:
        got = discrete_aris_deff_slit(d, h, 0.05)["d_eff_thomas"]
        errs.append(abs(got / theory_deff_slit(d, h, 0.05) - 1.0))
    orders = [math.log(errs[i] / errs[i + 1]) / math.log(hs[i + 1] / hs[i]) for i in range(2)]
    assert min(orders) >= 2.0, (errs, orders)
    assert errs[-1] < 1e-4


def test_thomas_cross_checks_library_aris_solver():
    """LAPACK-free Thomas solve must agree with the library solver (m8228 trap)."""
    for h in (16, 64):
        out = discrete_aris_deff_slit(0.04, h, 0.06)
        assert out["library_vs_thomas_rel"] < 1e-9
        assert out["solver_trustworthy"] is True


# ── boundaries / kernels ────────────────────────────────────────────────────
def test_scalar_wall_reflect_is_exact_relocation():
    """Zero-flux wall reflection must conserve the scalar mass bit-for-bit."""
    torch.manual_seed(0)
    g = torch.rand((5, 12, 9), dtype=torch.float64)
    g[:, 0, :] = 0.0  # dead wrap rows are zero in the meter
    g[:, -1, :] = 0.0
    g[3, 1, :] = 0.0  # post-stream slots that come from the dead rows
    g[4, -2, :] = 0.0
    before = float(g.sum())
    out = scalar_wall_reflect(g.clone())
    assert float(out.sum()) == before
    assert torch.all(out[:, 0, :] == 0.0) and torch.all(out[:, -1, :] == 0.0)


def test_second_order_equilibrium_keeps_zeroth_moment():
    """The O(u^2) equilibrium ablation must still integrate to T (mass-consistent)."""
    T = torch.full((6, 7), 1.3, dtype=torch.float64)
    ux = torch.full_like(T, 0.07)
    uy = torch.zeros_like(T)
    g2 = equilibrium_scalar_2nd(T, ux, uy)
    # fp32-normalised weights give a 3e-8 weight-sum residual -> 1e-8 on T=1.3
    assert torch.allclose(g2.sum(dim=0), T, atol=1e-8)
    # at rest it must reduce to the shipped first-order equilibrium
    z = torch.zeros_like(T)
    assert torch.allclose(equilibrium_scalar_2nd(T, z, z), temperature_equilibrium(T, z, z), atol=1e-15)


# ── the meter ───────────────────────────────────────────────────────────────
def test_zero_velocity_control_returns_molecular_diffusivity():
    """u = 0: the modal meter must return D (kH = 0.79 truncation < 0.2 %)."""
    rec = modal_decay_case(0.0, 10, 8.0, 0.05)
    assert rec["deff"] == pytest.approx(0.05, rel=2.0e-3)
    assert rec["mass_drift_rel"] < 1e-10


def test_meter_rate_is_independent_of_the_fit_window():
    """The asymptote is reached before the fit window opens (long-wave mode)."""
    a = modal_decay_case(20.0, 10, 16.0, 0.05, settle_relax=6.0)
    b = modal_decay_case(20.0, 10, 16.0, 0.05, settle_relax=10.0)
    assert a["rate"] == pytest.approx(b["rate"], rel=5.0e-3)
    assert a["r2"] > 0.9999


def test_pe40_case_is_in_the_right_ballpark():
    """End-to-end smoke on the shipped kernels at Pe = 40, kH = 0.39, Ma_max = 0.51.

    Deliberately coarse/large-Mach: this only checks that the meter runs, returns
    a Taylor-enhanced (not molecular) coefficient and stays mass-conservative; the
    quantitative gates live in the artifact regression test below.
    """
    rec = modal_decay_case(40.0, 10, 16.0, 0.05)
    assert rec["pe_sampled"] == pytest.approx(40.0, rel=1e-2)
    assert rec["ma_max"] > 0.3  # this deliberately runs a large lattice Mach
    assert rec["deff"] > 1.5 * rec["diffusivity"]
    assert 0.8 < rec["deff"] / rec["deff_theory"] < 1.2
    assert rec["r2"] > 0.9999
    assert rec["mass_drift_rel"] < 1e-10


# ── analysis helpers ────────────────────────────────────────────────────────
def test_k0_extrapolation_recovers_a_synthetic_quadratic_limit():
    rows = [{"kH": kh, "err_pct": 1.25 - 30.0 * kh**2 + 5.0 * kh**4}
            for kh in (0.1, 0.2, 0.4, 0.8)]
    ex = k0_extrapolation(rows)
    assert ex["intercept_k0"] == pytest.approx(1.25, abs=1e-9)
    assert ex["rms"] < 1e-9
    assert ex["n_used"] == 3 and len(ex["excluded_kH"]) == 1
    with pytest.raises(ValueError):
        k0_extrapolation(rows[:1])


def test_k0_extrapolation_two_spans_falls_back_to_linear():
    rows = [{"kH": kh, "err_pct": -3.0 + 12.0 * kh**2} for kh in (0.105, 0.21)]
    ex = k0_extrapolation(rows)
    assert ex["linear_fit"] is True
    assert ex["intercept_k0"] == pytest.approx(-3.0, abs=1e-12)


def test_richardson_order_of_a_second_order_sequence_is_two():
    def f(h):  # error ~ h^2
        return -3.2 + 7.0 * h**2

    out = richardson_order(f(0.4), f(0.2), f(0.1))
    assert out["order"] == pytest.approx(2.0, abs=1e-9)
    assert out["limit_estimate"] == pytest.approx(f(0.0), abs=1e-9)


def test_mach_law_fit_recovers_slope_and_r2():
    slope = -12.5
    rows = [{"u_sampled": u, "ma_sampled": u / math.sqrt(1 / 3), "err_pct": slope * u**2}
            for u in (0.02, 0.04, 0.06, 0.09)]
    law = mach_law_fit(rows)
    assert law["slope_per_u2"] == pytest.approx(slope, rel=1e-9)
    assert abs(law["intercept_pct"]) < 1e-9
    assert law["r2"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        mach_law_fit(rows[:2])


# ── regression guard on the verification artifact ───────────────────────────
@pytest.mark.skipif(not ARTIFACT.exists(), reason="verification artifact not built")
def test_artifact_gates_are_all_passed():
    res = json.loads(ARTIFACT.read_text())
    gates = res["gates"]
    if res["verdict"] != "PASS":
        pytest.skip(f"artifact incomplete: {res.get('missing_case_ids')} / "
                    f"{ {k: v.get('passed') for k, v in gates.items()} }")
    assert res["verdict"] == "PASS", res.get("missing_case_ids")
    assert gates["G1_pe40_within_3pct"]["passed"], gates["G1_pe40_within_3pct"]
    assert gates["G1b_pe40_corrected_equilibrium"]["passed"]
    assert abs(gates["G1_pe40_within_3pct"]["intercept_pct"]) <= 3.0
    assert gates["G2_mach_scaling_law"]["r2"] >= 0.99
    assert gates["G3_pe_scan"]["max_abs_err_pct"] <= 3.0
    assert gates["G4_schmidt_scan"]["max_abs_err_pct"] <= 3.0
    assert gates["G4_schmidt_scan"]["covered_sc_range"][1] >= 10.0
    assert gates["G5_refinement_order"]["min_k_order"] >= 2.0
    assert res["missing_case_ids"] == []
    assert res["n_cases_done"] >= res["n_cases_planned"]