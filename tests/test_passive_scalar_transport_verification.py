"""Tests for :mod:`tensorlbm.passive_scalar_transport_verification` (P74-N10).

Groups, ordered by cost:

* **case planning** — the ``(Pe_r, N_y)`` ladder must keep the Mach number
  fixed and the run length tied to ``h^2/D`` (no hand-picked run lengths);
* **gate arithmetic** — threshold logic on synthetic measurements, so a gate
  regression names the broken comparison;
* **lattice physics** — the erf order, the ``u = 0`` control, the ``Pe_r -> 0``
  limit, the mass budget (fp64 vs fp32) and the Schmidt sweep, all measured on
  the quick spec;
* **surface integration** — the eight-gate verdict over one quick run.

The heavy ladder of the published artifact is
``benchmarks/pending/passive_scalar_verification/run_p74n10.py``; these tests
pin the individual pieces so a failure names the broken piece.
"""

from __future__ import annotations

import json

import pytest

from tensorlbm.passive_scalar_transport_verification import (
    TransportSpec,
    meter_rate_ratio,
    evaluate_gates,
    erf_gate,
    mass_budget_gate,
    negative_control_gate,
    pe_limit_gate,
    quick_transport_spec,
    run_surface,
    schmidt_gate,
    surface_cases,
)


# ---------------------------------------------------------------------------
# case planning
# ---------------------------------------------------------------------------


def test_ladder_holds_mach_fixed_and_spaces_runs_by_transverse_diffusion_time():
    """Every ladder cell sees the same |u|; the run length follows ``h^2/D``."""
    spec = TransportSpec()
    cells = [c for c in surface_cases(spec) if c["kind"] == "dispersion"]
    assert len(cells) == len(spec.pe_nodes) * len(spec.ny_ladder)
    for cell in cells:
        assert cell["u_mean"] == pytest.approx(spec.u_ref)
        # Pe_r = u H / D  ->  D = u N_y / Pe_r
        assert cell["diffusivity"] == pytest.approx(
            spec.u_ref * cell["ny"] / cell["pe"]
        )
        assert abs(cell["u_mean"]) <= 0.1  # lattice Mach budget
        planned = spec.steps_factor * (0.5 * cell["ny"]) ** 2 / cell["diffusivity"]
        assert cell["steps"] >= min(planned, spec.min_steps)


def test_domain_length_rule_decouples_transverse_and_longitudinal_decay():
    """``nx`` is chosen from the meter precondition, not from a fixed default."""
    spec = TransportSpec()
    cells = {(c["pe"], c["ny"]): c for c in surface_cases(spec) if c["kind"] == "dispersion"}
    for (pe, ny), cell in cells.items():
        rate_ratio = meter_rate_ratio(pe, ny, cell["nx"])
        assert rate_ratio <= spec.rate_ratio_max, (pe, ny, cell["nx"], rate_ratio)
        assert spec.nx_min <= cell["nx"] <= spec.nx_max
        # the shortest admissible domain is a power of two
        assert cell["nx"] & (cell["nx"] - 1) == 0
    # high Pe_r / coarse N_y demands a longer domain than the low-Pe_r cell
    assert cells[(40.0, 33)]["nx"] > cells[(5.0, 9)]["nx"]


def test_schmidt_sweep_moves_nu_and_holds_the_lattice_diffusivity():
    spec = TransportSpec()
    cells = [c for c in surface_cases(spec) if c["kind"] == "schmidt"]
    assert {c["sc"] for c in cells} == set(spec.sc_list)
    assert len({c["diffusivity"] for c in cells}) == 1  # tau_d held fixed
    taus = {c["tau_f"] for c in cells}
    assert len(taus) == len(cells)  # tau_f actually swept
    for cell in cells:
        assert cell["viscosity"] == pytest.approx(cell["sc"] * cell["diffusivity"])
        assert cell["tau_f"] == pytest.approx(3.0 * cell["viscosity"] + 0.5)


def test_surface_covers_every_gate_kind():
    kinds = {c["kind"] for c in surface_cases(TransportSpec())}
    assert {
        "erf",
        "dispersion",
        "pe_zero",
        "negative_control",
        "schmidt",
        "probe",
        "probe_naive_sc",
        "probe_diffusion",
    } <= kinds


# ---------------------------------------------------------------------------
# gate arithmetic (synthetic measurements)
# ---------------------------------------------------------------------------


def _disp(pe, ny, ratio, closed=None, r2=1.0, steps=4000, drift=1e-14, nx=512):
    closed = closed if closed is not None else 1.0 + pe**2 / 210.0
    return {
        "name": f"disp_pe{pe:g}_ny{ny}",
        "kind": "dispersion",
        "pe_target": pe,
        "ny": ny,
        "nx": nx,
        "ratio_sim": ratio,
        "ratio_closed_form": closed,
        "ratio_discrete_lattice": closed,
        "rel_err_vs_closed": ratio / closed - 1.0,
        "rel_err_vs_discrete": ratio / closed - 1.0,
        "r2": r2,
        "steps": steps,
        "mass_drift": drift,
        "diffusivity": 0.1,
        "u_mean": 0.05,
        "tau_d": 0.8,
    }


def test_gate_b_accepts_five_percent_and_rejects_above():
    spec = TransportSpec(pe_nodes=(10.0,), ny_ladder=(9, 17))
    closed = 1.0 + 100.0 / 210.0
    ok = [_disp(10.0, 9, closed * 1.049), _disp(10.0, 17, closed * 1.001)]
    bad = [_disp(10.0, 9, closed * 1.09), _disp(10.0, 17, closed * 1.06)]
    assert evaluate_gates(ok, spec)["gate_pass"]["B_taylor_aris_pe_ladder"] is True
    assert evaluate_gates(bad, spec)["gate_pass"]["B_taylor_aris_pe_ladder"] is False


def test_gate_e_normalizes_drift_to_a_fixed_step_count():
    """A short run must not look cleaner than a long one with the same rate."""
    spec = TransportSpec()
    short = _disp(10.0, 9, 1.0, steps=1000, drift=1e-13)  # rate 1e-12 / 1e4
    long = _disp(10.0, 9, 1.0, steps=20000, drift=1e-12)  # rate 5e-13 / 1e4
    gate = mass_budget_gate([short, long], spec)
    assert gate["rates"][short["name"]] == pytest.approx(1e-12)
    assert gate["rates"][long["name"]] == pytest.approx(5e-13)
    assert gate["pass"] is True


def test_gate_f_band_is_two_sided():
    spec = TransportSpec()
    inside = {
        "name": "negative_control_u0",
        "kind": "negative_control",
        "ratio_sim": 0.97,
        "ratio_discrete_lattice": 1.0,
        "diffusivity": 0.05,
        "tau_d": 0.65,
        "steps": 1000,
        "r2": 1.0,
    }
    assert negative_control_gate([inside], spec)["pass"] is True
    assert negative_control_gate([{**inside, "ratio_sim": 1.06}], spec)["pass"] is False
    assert negative_control_gate([{**inside, "ratio_sim": 0.94}], spec)["pass"] is False


def test_gate_c_and_d_thresholds():
    spec = TransportSpec(pe_zero_list=(0.5,), sc_list=(0.1, 1.0, 10.0))
    closed = 1.0 + 0.25 / 210.0
    pe0 = pe_limit_gate(
        [
            {
                "name": "pezero_pe0.5",
                "kind": "pe_zero",
                "pe_target": 0.5,
                "diffusivity": 0.05,
                "u_mean": 0.0015,
                "ratio_sim": 0.985,
                "ratio_closed_form": closed,
                "ratio_discrete_lattice": 0.99,
                "r2": 1.0,
            }
        ],
        spec,
    )
    assert pe0["pass"] is True
    assert pe_limit_gate(
        [
            {
                "name": "pezero_pe0.5",
                "kind": "pe_zero",
                "pe_target": 0.5,
                "diffusivity": 0.05,
                "u_mean": 0.0015,
                "ratio_sim": 0.975,
                "ratio_closed_form": closed,
                "ratio_discrete_lattice": 0.99,
                "r2": 1.0,
            }
        ],
        spec,
    )["pass"] is False

    def _sc(sc, ratio):
        return {
            "name": f"schmidt_sc{sc:g}",
            "kind": "schmidt",
            "sc": sc,
            "diffusivity": 0.1 / sc,
            "tau_d": 0.5 + 3 * 0.1 / sc,
            "u_mean": 0.01,
            "steps": 4000,
            "ratio_sim": ratio,
            "ratio_discrete_lattice": ratio,
            "rel_err_vs_closed": 0.0,
            "rel_err_vs_discrete": 0.0,
            "r2": 1.0,
        }

    tight = [_sc(0.1, 1.100), _sc(1.0, 1.101), _sc(10.0, 1.102)]
    wide = [_sc(0.1, 1.10), _sc(1.0, 1.101), _sc(10.0, 1.20)]
    assert schmidt_gate(tight, spec)["pass"] is True
    assert schmidt_gate(wide, spec)["pass"] is False


# ---------------------------------------------------------------------------
# lattice physics on the quick spec
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quick_cases(tmp_path_factory):
    spec = quick_transport_spec()
    case_dir = tmp_path_factory.mktemp("p74n10_cases")
    return spec, run_surface(spec, case_dir=case_dir, verbose=False)


def test_quick_spec_reproduces_erf_second_order(quick_cases):
    spec, cases = quick_cases
    gate = erf_gate(cases, spec)
    assert gate["convergence"]["monotone_error_decay"] is True
    assert gate["min_order"] >= spec.erf_order_min
    assert gate["finest_grid_rel_pct"] <= spec.erf_l2_rel_tol_pct


def test_quick_spec_mass_budget_is_met_in_float64(quick_cases):
    spec, cases = quick_cases
    gate = mass_budget_gate(cases, spec)
    assert gate["max_rate"] <= spec.mass_tol_per_1e4


def test_float32_breaks_the_mass_budget(quick_cases):
    """Why the surface runs in fp64: the D2Q5 weight sum is 1+2.98e-8 in fp32.

    The same erf case in fp32 drifts by orders of magnitude more per step than
    the ``1e-10`` per ``1e4`` steps gate allows, so the gate is not a
    formality — it pins the working precision.
    """
    from tensorlbm.passive_scalar_transport_verification import run_case

    spec = quick_transport_spec()
    case = {
        "name": "erf_L0",
        "kind": "erf",
        "level": 0,
        "nx": 64,
    }
    fp64 = run_case(case, spec)
    fp32 = run_case(case, TransportSpec(**{**spec.__dict__, "dtype": "float32"}))
    rate64 = fp64["mass_drift"] * 1e4 / fp64["steps"]
    rate32 = fp32["mass_drift"] * 1e4 / fp32["steps"]
    assert rate64 <= 1e-10
    assert rate32 > 1e-10
    assert rate32 > 100.0 * rate64


def test_quick_spec_negative_control_and_pe_limit(quick_cases):
    spec, cases = quick_cases
    ctrl = negative_control_gate(cases, spec)
    assert ctrl["band"][0] <= ctrl["ratio_sim"] <= ctrl["band"][1]
    pe0 = pe_limit_gate(cases, spec)
    for row in pe0["table"]:
        assert abs(row["dev_from_unity_pct"]) <= spec.pe_zero_tol_pct


def test_quick_spec_schmidt_spread_is_small(quick_cases):
    spec, cases = quick_cases
    gate = schmidt_gate(cases, spec)
    assert gate["spread_pct"] <= spec.sc_tol_pct
    for row in gate["table"]:
        assert row["r2"] >= spec.r2_min


def test_quick_spec_verdict_is_assembled(quick_cases):
    spec, cases = quick_cases
    report = evaluate_gates(cases, spec)
    assert set(report["gate_pass"]) == {
        "A_erf",
        "B_taylor_aris_pe_ladder",
        "C_pe_to_zero",
        "D_schmidt_independence",
        "E_mass_budget",
        "F_negative_control",
        "G_convergence_quality",
        "H_accuracy_domain_table",
    }
    assert report["verdict"] in {"PASS", "FAIL"}
    assert report["verdict"] == "PASS" if all(report["gate_pass"].values()) else "FAIL"
    assert report["gates"]["H"]["complete"] is True
    # the artifact payload must be JSON serialisable end to end
    json.dumps({k: v for k, v in report.items() if not k.startswith("_")}, default=str)