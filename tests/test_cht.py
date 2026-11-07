"""CHT verification ladder gates (campaign node IN49-W3-N04, TensorLBM).

Every acceptance criterion of the conjugate-heat-transfer verification ladder is
asserted here.  The heavy anchors (Graetz eigenpairs, 2-D annulus, 3-D shell,
3-D lumped cube, D2Q5 Graetz LBM) are computed once by a module-scoped fixture
via :func:`tensorlbm.cht_ladder.run_ladder`, so the whole file costs ~30 s.

Ladder covered
--------------
* solid conduction in 1-D (composite wall), 2-D (polar annulus) and 3-D
  (spherical shell),
* fluid-solid interface: ``q_f = q_s`` and ``T_f = T_s``,
* four analytical anchors: series resistance, erf thermal penetration, Graetz
  (Shah-London) Nusselt number, lumped-capacitance time constant,
* negative control: interface conductance ``h_int = 0`` gives an adiabatic
  interface.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorlbm.cht_ladder import GATES, evaluate_gates, run_ladder, write_artifacts

# ---------------------------------------------------------------------------
# module-scoped ladder run (single expensive computation shared by the gates)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ladder():
    metrics = run_ladder(verbose=False)
    gates = evaluate_gates(metrics)
    return metrics, gates


# ---------------------------------------------------------------------------
# gate-level assertions (one test per gate id)
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.parametrize("gate_id", sorted(GATES))
def test_gate(ladder, gate_id):
    """Every declared gate must satisfy its threshold."""
    _metrics, gates = ladder
    g = gates[gate_id]
    assert g["evidence"] == "REAL"
    assert g["value"] <= g["threshold"], (
        f"{gate_id}: {g['quantity']} = {g['value']:.6g} exceeds threshold "
        f"{g['threshold']:.6g} ({g['reference']})"
    )


@pytest.mark.physics
def test_all_gates_pass(ladder):
    _metrics, gates = ladder
    assert gates["all_pass"] is True
    assert gates["n_pass"] == gates["n_gates"] == len(GATES)


# ---------------------------------------------------------------------------
# anchor 1: 1-D composite wall, series resistance R = sum(delta_i/k_i)
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_composite_wall_series_resistance():
    from tensorlbm.cht_verification import composite_wall

    layers = [(0.02, 0.5), (0.05, 2.0), (0.03, 0.1)]
    res = composite_wall(layers, T_left=1.0, T_right=0.0, cells=240)
    assert res.rel_err_R < 0.01
    # the 1-D series network is the exact discretisation of the resistance chain
    assert res.rel_err_R < 1e-10
    assert res.max_err_T < 1e-10
    # flux uniformity is the discrete statement of steady state
    assert res.extra["q_spread"] < 1e-9 * abs(res.q)
    # analytical flux must equal dT/R
    assert abs(res.q - 1.0 / 0.365) < 1e-9


@pytest.mark.physics
def test_composite_wall_rejects_wrong_series_order():
    """Swapping two layers must change R (guards against a stub returning a constant)."""
    from tensorlbm.cht_verification import composite_wall

    a = composite_wall([(0.02, 0.5), (0.05, 2.0)], cells=140)
    b = composite_wall([(0.05, 2.0), (0.02, 0.5)], cells=140)
    assert abs(a.resistance - b.resistance) < 1e-12  # series resistance is order-free
    assert abs(a.resistance - (0.02 / 0.5 + 0.05 / 2.0)) < 1e-12


# ---------------------------------------------------------------------------
# solid conduction in 2-D and 3-D (grid convergence + analytical profiles)
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_cylinder_annulus_2d_profile_and_convergence():
    from tensorlbm.cht_verification import cylinder_annulus_2d

    coarse = cylinder_annulus_2d(n_r=60, n_theta=24, theta_amplitude=0.3)
    fine = cylinder_annulus_2d(n_r=140, n_theta=48, theta_amplitude=0.3)
    assert fine.max_err_T < 0.01
    assert fine.max_err_T < coarse.max_err_T  # genuine discretisation -> converges
    assert fine.rel_err_R < 0.01  # radial heat rate vs 2 pi k dT/ln(r2/r1)
    assert fine.extra["linear_residual"] < 1e-9


@pytest.mark.physics
def test_sphere_shell_3d_profile_and_heat_rate():
    from tensorlbm.cht_verification import sphere_shell

    shell = sphere_shell(r1=0.3, r2=1.0, n_r=600)
    assert shell.max_err_T < 0.01
    assert shell.rel_err_R < 0.01
    assert shell.extra["Q_spread"] < 1e-9 * abs(shell.q)
    # composite shell: radial resistances add in series
    layered = sphere_shell(layers=[(0.6, 3.0), (1.0, 0.5)], r1=0.3, n_r=1400)
    expected = (1 / 0.3 - 1 / 0.6) / (4 * np.pi * 3.0) + (1 / 0.6 - 1 / 1.0) / (4 * np.pi * 0.5)
    assert abs(layered.resistance - expected) / expected < 1e-3


# ---------------------------------------------------------------------------
# fluid-solid interface: flux and temperature continuity
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_interface_flux_and_temperature_continuity():
    from tensorlbm.cht_verification import interface_fluxes

    T_f = np.array([1.0, 0.85, 0.62, 0.4])
    T_s = np.array([0.55, 0.42, 0.3, 0.18])
    res = interface_fluxes(T_f, T_s, k_fluid=1.0, k_solid=5.0)
    assert res.flux_residual < 1e-6
    assert res.temperature_residual < 1e-6
    # series conductance value: q = dT /(dx_f/k_f + dx_s/k_s)
    expected = (T_f - T_s) / (0.5 / 1.0 + 0.5 / 5.0)
    assert np.allclose(res.q_fluid, expected, rtol=1e-12)


@pytest.mark.physics
def test_interface_metric_is_not_vacuous():
    """The arithmetic-mean interface must violate flux continuity by O(1)."""
    from tensorlbm.cht_verification import interface_fluxes

    T_f = np.array([1.0, 0.5])
    T_s = np.array([0.4, 0.0])
    naive = interface_fluxes(T_f, T_s, 1.0, 5.0, interface_weighting="arithmetic")
    harmonic = interface_fluxes(T_f, T_s, 1.0, 5.0)
    assert naive.flux_residual > 1e-2
    assert harmonic.flux_residual < 1e-12


@pytest.mark.physics
def test_two_domain_conservative_fv_interface():
    from tensorlbm.cht_verification import conservative_two_domain

    res = conservative_two_domain(k_fluid=1.0, k_solid=5.0, L_fluid=2.0, L_solid=1.0)
    assert res["flux_residual"] == 0.0  # one shared face -> single flux
    assert res["temperature_residual"] < 1e-12
    assert res["rel_err_q"] < 1e-10


# ---------------------------------------------------------------------------
# negative control: h_int = 0 -> adiabatic interface
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_adiabatic_interface_negative_control():
    from tensorlbm.cht_transient import adiabatic_interface_control
    from tensorlbm.cht_verification import interface_fluxes

    out = adiabatic_interface_control()
    assert out["max_gradient"] < 1e-10
    assert out["flux_interface"] == 0.0
    # each slab relaxes to its own uniform temperature
    assert out["temperature_range_left"] < 1e-10
    assert out["temperature_range_right"] < 1e-10

    zero = interface_fluxes(np.array([1.0, 0.5]), np.array([0.0, 0.2]), 1.0, 5.0,
                            contact_conductance=0.0)
    assert np.all(zero.q_fluid == 0.0)
    assert np.all(zero.q_solid == 0.0)


# ---------------------------------------------------------------------------
# anchor 2: erf thermal penetration
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_erf_penetration_profile():
    from tensorlbm.cht_transient import erf_penetration

    out = erf_penetration()
    assert out["max_err"] < 0.02
    assert out["max_err"] < 1e-3  # actual accuracy, well inside the gate
    assert out["l2_err"] < 1e-3
    # the similarity variable must collapse the profile: check one point
    import math

    z = 0.5
    i = int(np.argmin(np.abs(out["z"] - z)))
    assert abs(out["T"][i] - (1.0 - math.erf(z))) < 0.01


# ---------------------------------------------------------------------------
# anchor 3: Graetz / Shah-London
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_graetz_series_matches_literature():
    """Graetz series vs the classical Shah-London / textbook tabulation."""
    from tensorlbm.graetz import NU_TUBE_DEVELOPED, graetz_eigenpair_table, graetz_exact

    table = graetz_eigenpair_table(n_modes=8)
    # first eigenvalue in the library convention beta = 2 * lambda_1^2,
    # lambda_1^2 = 7.3126 (classical Graetz eigenvalue) -> beta_1 = 14.625
    assert abs(table["beta"][0] - 2 * 7.3126) / (2 * 7.3126) < 1e-2
    res = graetz_exact([0.002, 0.005, 0.01, 50.0], table=table)
    nu = res["Nu"]
    assert abs(nu[-1] - NU_TUBE_DEVELOPED) / NU_TUBE_DEVELOPED < 1e-3  # developed limit
    # Shah-London / Incropera Table 8.1 tabulated local Nusselt numbers
    assert abs(nu[0] - 8.03) / 8.03 < 0.02   # x* = 0.002
    assert abs(nu[1] - 6.00) / 6.00 < 0.02   # x* = 0.005
    assert abs(nu[2] - 5.14) / 5.14 < 0.05   # x* = 0.01 (coarser tabulation)


@pytest.mark.physics
def test_graetz_marching_matches_series():
    from tensorlbm.cht_ladder import run_ladder  # noqa: F401  (fixture reuse)
    from tensorlbm.graetz import graetz_eigenpair_table, graetz_exact, graetz_marching

    table = graetz_eigenpair_table(n_modes=10)
    xs = np.array([0.005, 0.01, 0.02, 0.05])
    ex = graetz_exact(xs, table=table)
    mg = graetz_marching(x_star_stations=xs)
    rel = np.abs(mg["Nu"] - ex["Nu"]) / ex["Nu"]
    assert np.max(rel) < 0.03
    assert np.max(rel) < 5e-3


@pytest.mark.physics
def test_graetz_lbm_kernel_matches_library():
    """The NumPy D2Q5 mirror used by ``graetz_lbm`` is the library kernel."""
    import torch

    from tensorlbm.thermal import W5, collide_thermal_bgk, temperature_equilibrium

    rng = np.random.default_rng(0)
    T = rng.random((6, 9))
    ux = 0.05 * rng.random((6, 9))
    uy = 0.05 * rng.random((6, 9))
    w = np.asarray(W5, dtype=np.float64)
    c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
    cu = c[:, 0].reshape(5, 1, 1) * ux[None] + c[:, 1].reshape(5, 1, 1) * uy[None]
    geq_np = w.reshape(5, 1, 1) * T[None] * (1.0 + 3.0 * cu)

    geq_t = temperature_equilibrium(
        torch.tensor(T), torch.tensor(ux), torch.tensor(uy)
    ).numpy()
    assert np.allclose(geq_np, geq_t, rtol=1e-13, atol=1e-15)

    tau = 0.59
    g = rng.random((5, 6, 9))
    coll_np = g - (g - geq_np) / tau
    coll_t = collide_thermal_bgk(
        torch.tensor(g), torch.tensor(T), torch.tensor(ux), torch.tensor(uy), tau
    ).numpy()
    assert np.allclose(coll_np, coll_t, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# anchor 4: lumped capacitance
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_lumped_capacitance_time_constant():
    from tensorlbm.cht_transient import lumped_cooling_cube

    out = lumped_cooling_cube(n=16, biot=0.02)
    assert out["rel_err_tau"] < 0.02
    assert out["meta"]["discrete_volume"] == pytest.approx(1.0, rel=1e-12)
    assert out["meta"]["discrete_area"] == pytest.approx(6.0, rel=1e-12)


@pytest.mark.physics
def test_lumped_error_scales_with_biot():
    """The residual is the finite-Biot correction: it must grow ~linearly in Bi."""
    from tensorlbm.cht_transient import lumped_cooling_cube

    low = lumped_cooling_cube(n=12, biot=0.005)
    mid = lumped_cooling_cube(n=12, biot=0.02)
    high = lumped_cooling_cube(n=12, biot=0.05)
    assert low["rel_err_tau"] < mid["rel_err_tau"] < high["rel_err_tau"]
    assert low["rel_err_tau"] < 0.01
    # tau_sim / tau_lumped - 1 ~ 0.85 * Bi
    ratio = (mid["rel_err_tau"] / mid["meta"]["biot"])
    assert 0.4 < ratio < 1.4


# ---------------------------------------------------------------------------
# artifact contract
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_artifact_contract(tmp_path, ladder):
    metrics, gates = ladder
    paths = write_artifacts(metrics, gates, tmp_path / "cht")
    result = json.loads(open(paths["result_json"]).read())
    verified = json.loads(open(paths["gates_verified_json"]).read())

    assert result["campaign_node"] == "IN49-W3-N04"
    assert result["gate_summary"]["n_gates"] == len(GATES)
    assert set(verified["gates"]) == set(GATES)
    for gid, spec in verified["gates"].items():
        assert {"value", "threshold", "pass", "quantity", "reference"} <= set(spec)
        assert spec["pass"] is True
    assert len(paths["figures"]) >= 6
    for fig in paths["figures"]:
        assert (tmp_path / "cht" / "figures" / fig["name"]).exists()
        assert fig["caption"]
    # key numbers are carried into the artifact
    assert result["metrics"]["scalars"]["graetz_lbm_rel_err"] < 0.03
    assert result["metrics"]["scalars"]["interface_flux_residual"] < 1e-6