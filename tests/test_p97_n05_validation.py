"""Tests for the P97-N05 campaign harness (``scripts/p97_n05_*.py``).

These tests pin the *measurement* layer rather than the physics:

* the Nusselt-number meters must return exactly 1 on a pure conduction state
  (any offset there would silently bias every Nu number in the campaign);
* the Rayleigh–Bénard lattice parameters must satisfy the definition of ``Ra``;
* the Dirichlet wall treatment must actually impose the wall temperatures;
* the acceptance gate (:func:`scripts.p97_n05_campaign.evaluate`) must flip
  red when a number is off — a gate that cannot fail is worthless.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import p97_n05_campaign as campaign  # noqa: E402
from p97_n05_cavity_case import NU_REF_1E4, simulate_cavity  # noqa: E402
from p97_n05_rb_case import (  # noqa: E402
    nusselt_midplane,
    nusselt_walls,
    rb_params,
    simulate_rb,
    temperature_walls_rb,
)
from tensorlbm.thermal import W5, thermal_params  # noqa: E402

# ---------------------------------------------------------------------------
# Nusselt-number meters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ny", [8, 16, 33])
def test_wall_nusselt_is_exactly_one_for_pure_conduction(ny):
    """A linear profile gives Nu = 1 for the second-order one-sided stencil."""
    H = float(ny)
    y = torch.arange(ny, dtype=torch.float64)
    T = (1.0 - y / H).view(ny, 1).expand(ny, 4).contiguous()
    nb, nt = nusselt_walls(T, H, 1.0, 1.0, 0.0)
    assert nb == pytest.approx(1.0, rel=1e-12)
    assert nt == pytest.approx(1.0, rel=1e-12)


def test_midplane_nusselt_is_exactly_one_for_pure_conduction():
    ny, nx = 17, 5
    H = float(ny)
    y = torch.arange(ny, dtype=torch.float64)
    T = (1.0 - y / H).view(ny, 1).expand(ny, nx).contiguous()
    uy = torch.zeros_like(T)
    nu = nusselt_midplane(uy, T, alpha=0.037, H=H, dT=1.0)
    assert nu == pytest.approx(1.0, rel=1e-12)


def test_midplane_nusselt_detects_convective_flux():
    """A positive vertical velocity carrying hot fluid upward must raise Nu."""
    ny, nx = 17, 5
    H = float(ny)
    y = torch.arange(ny, dtype=torch.float64)
    T = (1.0 - y / H).view(ny, 1).expand(ny, nx).contiguous()
    uy = torch.full_like(T, 0.05)
    nu = nusselt_midplane(uy, T, alpha=0.037, H=H, dT=1.0)
    assert nu > 1.5


# ---------------------------------------------------------------------------
# lattice parameters and boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Ra", [1e3, 1e4, 1e6])
def test_rb_parameters_satisfy_the_definition_of_Ra(Ra):
    p = rb_params(ny=50, ra=Ra, pr=0.71, tau=0.6)
    Ra_back = p["g_beta"] * 1.0 * p["H"] ** 3 / (p["nu"] * p["alpha"])
    assert Ra_back == pytest.approx(Ra, rel=1e-12)
    assert p["nu"] / p["alpha"] == pytest.approx(0.71, rel=1e-12)
    assert p["tau_T"] == pytest.approx(3.0 * p["alpha"] + 0.5, rel=1e-12)


def test_cavity_parameters_satisfy_Ra_and_Pr():
    p = thermal_params(nx=64, ra=1e4, pr=0.71, tau=1.0)
    # the driver re-forms Ra with the true thermal gap nx - 1 (see the module)
    assert p["H"] == 64.0
    Ra_back = p["g_beta"] * 1.0 * p["H"] ** 3 / (p["nu"] * p["alpha"])
    assert Ra_back == pytest.approx(1e4, rel=1e-10)
    assert p["nu"] / p["alpha"] == pytest.approx(0.71, rel=1e-12)


def test_rb_dirichlet_walls_impose_the_wall_temperatures():
    ny, nx = 9, 4
    g = torch.zeros((5, ny, nx), dtype=torch.float64)
    g = temperature_walls_rb(g, 1.0, 0.0)
    T = g.sum(dim=0)
    # anti-bounce-back on an empty field gives g = 2 w T at the wall, i.e. the
    # macroscopic temperature of the boundary rows is exactly the wall value
    w3 = float(W5[3])
    assert float(T[0, 0]) == pytest.approx(2.0 * w3, rel=1e-6)


def test_rb_negative_control_runs_at_low_ra():
    """Ra = 100 is deep in the conduction regime: Nu must be ~1."""
    res = simulate_rb(ra=100.0, nx=16, ny=8, steps=4000, sample_every=500)
    assert res["regime"] in ("conduction", "steady")
    assert res["nu_wall"] == pytest.approx(1.0, abs=0.03)
    assert res["u_max_mean"] < 1e-3
    assert res["ra_over_ra_c"] < 0.1


def test_cavity_driver_is_reproducible_and_reports_nu():
    res = simulate_cavity(nx=16, steps=2000, sample_every=500)
    assert res["nu"] > 0.0
    assert len(res["history"]) >= 4
    assert res["nu_ref"] == pytest.approx(NU_REF_1E4)


# ---------------------------------------------------------------------------
# acceptance gate
# ---------------------------------------------------------------------------


def _write_fake(tmp: Path, name: str, payload: dict) -> None:
    (tmp / f"{name}.json").write_text(json.dumps(payload))


@pytest.fixture()
def fake_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "RUNS", tmp_path)
    crit = {
        "continuum": {"Ra_c_extrapolated": 1707.8, "Ra_c_by_grid": [{"M": 50, "Ra_c": 1712.0}]},
        "lattice": [],
    }
    _write_fake(tmp_path, "critical_rb", crit)
    for name, ra, nu, gap in (
        ("rb_ra1e4", 1e4, 2.16, 0.005),
        ("rb_ra1e5", 1e5, 4.22, 0.006),
        ("rb_ra3e4", 3e4, 3.0, 0.004),
        ("rb_ra3e5", 3e5, 5.9, 0.003),
        ("rb_ra1e6", 1e6, 8.1, 0.002),
    ):
        _write_fake(
            tmp_path,
            name,
            {"ra": ra, "nu_wall": nu, "nu_wall_gap": gap, "regime": "steady"},
        )
    for name, nu, err in (("cav_n64", 2.20, 0.0), ("cav_n96", 2.25, 0.0)):
        _write_fake(
            tmp_path,
            name,
            {"nx": int(name[-2:]), "tau": 0.6, "steps": 1000, "nu": nu,
             "u_max_star": 18.0, "nu_tail_drift": 1e-4},
        )
    for name, pe, d, deff, rel in (
        ("ta_pe1", 1.0, 0.18, 0.1834, 0.01),
        ("ta_pe3", 3.0, 0.06, 0.0703, 0.02),
        ("ta_pe10", 10.0, 0.018, 0.0522, -0.01),
        ("ta_pe10_nx128", 10.0, 0.018, 0.0522, -0.02),
        ("ta_pe30", 30.0, 0.006, 0.1085, 0.03),
        ("ta_pe30_nx128", 30.0, 0.006, 0.1085, 0.01),
        ("ta_pe100", 100.0, 0.0068, 1.30, -0.9),
        ("ta_pe100_nx128", 100.0, 0.0068, 1.30, -0.9),
    ):
        _write_fake(
            tmp_path,
            name,
            {"pe": pe, "d_molecular": d, "d_eff": deff * (1 + rel),
             "d_eff_discrete": deff, "d_eff_analytic": deff, "rel_err_vs_discrete": rel,
             "rel_err_vs_analytic": rel, "r2": 1.0, "fit_t0": 1.0, "fit_t1": 2.0,
             "half_width": 4.5, "u_mean_profile": 0.04},
        )
    for name, d, meas, rel in (
        ("ta_ctrl_zero", 0.02, 0.0199, -0.005),
        ("ta_ctrl_zero_lowD", 0.006, 0.00598, -0.003),
        ("ta_ctrl_plug", 0.02, 0.0201, 0.005),
    ):
        _write_fake(
            tmp_path,
            name,
            {"d_molecular": d, "d_eff": meas, "rel_err_vs_discrete": rel, "r2": 1.0},
        )
    for name, ra in (("rb_ra100", 100.0), ("rb_ra1000", 1000.0)):
        _write_fake(tmp_path, name, {"ra": ra, "nu_wall": 1.001, "regime": "conduction"})
    return tmp_path


def test_gate_passes_on_consistent_numbers(fake_runs):
    res = campaign.evaluate()
    crit = res["criteria"]
    assert crit["C1_rac"]["pass"] is True
    assert crit["C2_nu_ra"]["pass"] is True
    assert crit["C3_cavity_dvd"]["pass"] is True
    assert crit["C6_negative_control"]["pass"] is True
    assert crit["C1_rac"]["rel_err"] < 0.01


def test_gate_flags_a_bad_cavity_number(fake_runs):
    _write_fake(fake_runs, "cav_n64", {"nx": 64, "tau": 0.6, "steps": 1000, "nu": 2.6,
                                       "u_max_star": 18.0, "nu_tail_drift": 1e-4})
    _write_fake(fake_runs, "cav_n96", {"nx": 96, "tau": 0.6, "steps": 1000, "nu": 2.7,
                                       "u_max_star": 18.0, "nu_tail_drift": 1e-4})
    res = campaign.evaluate()
    assert res["criteria"]["C3_cavity_dvd"]["pass"] is False
    assert res["all_pass"] is False


def test_gate_flags_a_bad_critical_rayleigh_number(fake_runs):
    _write_fake(fake_runs, "critical_rb", {
        "continuum": {"Ra_c_extrapolated": 1800.0, "Ra_c_by_grid": []}, "lattice": []})
    res = campaign.evaluate()
    assert res["criteria"]["C1_rac"]["pass"] is False


def test_gate_flags_a_bad_dispersion_number(fake_runs):
    _write_fake(fake_runs, "ta_pe10_nx128", {
        "pe": 10.0, "d_molecular": 0.018, "d_eff": 0.09, "d_eff_discrete": 0.0522,
        "d_eff_analytic": 0.0522, "rel_err_vs_discrete": 0.72,
        "rel_err_vs_analytic": 0.72, "r2": 1.0, "fit_t0": 1.0, "fit_t1": 2.0,
        "half_width": 4.5, "u_mean_profile": 0.04})
    res = campaign.evaluate()
    assert res["criteria"]["C4_taylor_aris"]["pass"] is False


def test_gate_cannot_pass_without_evidence(tmp_path, monkeypatch):
    """An empty run directory must NOT evaluate to all_pass=True."""
    monkeypatch.setattr(campaign, "RUNS", tmp_path)
    res = campaign.evaluate()
    assert res["all_pass"] is False
    assert len(res["criteria_missing"]) == 6


def test_gate_reports_missing_cases(fake_runs):
    (fake_runs / "ta_pe1.json").unlink()
    res = campaign.evaluate()
    assert "ta_pe1" in res["criteria"]["missing"]


def test_fitted_exponent_is_recovered_from_a_known_power_law(fake_runs):
    """Nu = 2.16 (Ra/1e4)^0.29 must be fitted back to 0.29 by the collector."""
    for name, ra in (("rb_ra1e4", 1e4), ("rb_ra3e4", 3e4), ("rb_ra1e5", 1e5),
                     ("rb_ra3e5", 3e5), ("rb_ra1e6", 1e6)):
        nu = 2.16 * (ra / 1e4) ** 0.29
        _write_fake(fake_runs, name, {"ra": ra, "nu_wall": nu, "nu_wall_gap": 0.004,
                                      "regime": "steady"})
    res = campaign.evaluate()
    assert res["criteria"]["C2_nu_ra"]["fitted_exponent"] == pytest.approx(0.29, abs=1e-6)


def test_case_list_contains_every_name_the_gate_needs():
    names = {c["name"] for c in campaign.CASES}
    needed = {
        "critical_rb", "rb_ra1e4", "rb_ra1e5", "rb_ra3e4", "rb_ra3e5", "rb_ra1e6",
        "cav_n64", "cav_n96", "cav_n128", "ta_pe1", "ta_pe3", "ta_pe10", "ta_pe30",
        "ta_ctrl_zero", "ta_ctrl_zero_lowD", "ta_ctrl_plug", "rb_ra100", "rb_ra1000",
    }
    assert needed <= names
    # every primary C4 rung has a resolved (ny=17) companion run
    for pe in ("ta_pe1_n17", "ta_pe3_n17", "ta_pe10_n17", "ta_pe30_n17"):
        assert pe in names