"""Subcell depth, atmospheric connectivity, material scaling and fail-closed scoring."""

import json

import numpy as np
import pytest
import torch

from tensorlbm import cac_lbm

pytest.importorskip("scipy")
from benchmarks.water_entry.single_drop.run import CASE, DropTracker, open_depth, scales
from benchmarks.water_entry.single_drop.score import reference, score_run


def test_depth_excludes_sealed_bubble_and_measures_a_half_fill_crossing():
    field = np.zeros((12, 11, 11))
    field[:7] = 1
    field[2, 5, 5] = 0
    assert open_depth(field, 6.5, 4) == 0
    field[5:7, 5, 5] = [0.75, 0.25]
    assert open_depth(field, 6.5, 4) == pytest.approx(1)


def test_drop_tracker_distinguishes_contact_from_disappearance():
    field = np.zeros((12, 8, 8))
    field[:4] = 1
    field[6:8, 3:5, 3:5] = 1
    tracker = DropTracker()
    assert tracker.update(field, 0, 3.5).sum() == 8
    field[4:6, 3:5, 3:5] = 1
    assert tracker.update(field, 1, 3.5) is None
    assert tracker.contact_step == 1
    field[4:6] = 0
    tracker = DropTracker()
    tracker.update(field, 0, 3.5)
    field[6:] = 0
    with pytest.raises(ValueError, match="identity lost"):
        tracker.update(field, 1, 3.5)


def test_si_similarity_survives_spatial_and_temporal_refinement():
    case = json.loads(CASE.read_text())
    for n, u in [(12, 0.04), (18, 0.02)]:
        sc = scales(case, n, u)
        assert u * n / sc["mu_l"] == pytest.approx(
            case["water_density_kg_m3"]
            * case["velocity_m_s"]
            * case["diameter_m"]
            / case["water_dynamic_viscosity_pa_s"]
        )
        assert u * u * n / sc["sigma"] == pytest.approx(
            case["water_density_kg_m3"]
            * case["velocity_m_s"] ** 2
            * case["diameter_m"]
            / case["surface_tension_n_m"]
        )
        # Table 3 publishes rounded properties, while Table 2 rounds Re to an integer.
        assert u * n / sc["mu_l"] == pytest.approx(case["case3_measurements"]["Re"], abs=1)
        assert round(u * u * n / sc["sigma"], 1) == case["case3_measurements"]["We"]
        assert round(u * u / (sc["g"] * n), 1) == case["case3_measurements"]["Fr_velocity_squared"]
        assert u * u / (sc["g"] * n) == pytest.approx(
            case["velocity_m_s"] ** 2 / (case["gravity_m_s2"] * case["diameter_m"])
        )


def test_score_uses_only_healthy_post_contact_observations():
    case = json.loads(CASE.read_text())
    t, h = reference(case)
    assert len(t) == 18 and np.all(np.diff(t) > 0)

    def row(step, time, depth, mach=0.1):
        return dict(
            step=step,
            time_ms=time,
            depth_over_D=depth,
            mach_max=mach,
            rho_min=0.001,
            volume_drift=0.0,
        )

    run = dict(
        history=[row(0, 0, 0), row(1, 1, 0), row(2, 2, 1), row(3, 3, 2, 0.5)],
        contact_step=1,
        impact_speed_m_s=1.4,
        stop_reason="mach_limit",
    )
    result = score_run(run, np.array([0.5, 1.5]), np.array([0.6, 1.4]), case)
    assert result["matched_points"] == 1
    assert result["comparisons"][0]["simulation"] == 0.5
    assert result["peak_relative_error"] is None and not result["numeric_pass"]
    run["contact_step"] = None
    assert score_run(run, t, h, case)["matched_points"] == 0


def test_trt_retains_bgk_when_both_rates_match_and_preserves_force_impulse():
    shape = (5, 5, 5)
    nu = 1 / 6

    def state(rate):
        sim = cac_lbm.CACSim(
            shape,
            rho_l=1,
            rho_g=1,
            mu_l=nu,
            mu_g=nu,
            sigma=0.001,
            width=4,
            gravity=(0.0, 0.0, 0.0001),
            hydro_odd_rate=rate,
            dtype=torch.float64,
        )
        sim.initialize(torch.full(shape, 0.4, dtype=torch.float64))
        return sim

    a, b, c = state(None), state(1.0), state(0.8)
    for sim in (a, b, c):
        sim.collide()
    torch.testing.assert_close(a.g, b.g, rtol=0, atol=1e-15)
    for sim in (a, c):
        pz = (sim.g * sim.cf[2]).sum(0)
        torch.testing.assert_close(pz, torch.full_like(pz, 0.0001), rtol=0, atol=1e-15)


def test_stencil_neumann_preserves_wall_population_memory():
    shape = (6, 4, 4)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[[0, -1]] = True
    sim = cac_lbm.CACSim(
        shape,
        rho_l=1,
        rho_g=0.1,
        mu_l=0.1,
        mu_g=0.01,
        sigma=0.001,
        width=4,
        wall_mask=mask,
        axes="wpp",
        wall_phi_policy="stencil_neumann",
        dtype=torch.float64,
    )
    phi = torch.full(shape, 0.2, dtype=torch.float64)
    phi[1] = 0.7
    sim.initialize(phi)
    old = sim.f.clone()
    sim.refresh_solid_ghosts()
    torch.testing.assert_close(sim.phase_for_stencil(phi)[0], phi[1])
    torch.testing.assert_close(sim.f, old, atol=0, rtol=0)


def test_competing_conversion_debits_are_local_and_conservative():
    from tensorlbm.free_surface_lbm import _capacity_redistribution
    from tensorlbm.free_surface_topology_transaction import TopologyTransactionError

    mass = torch.zeros((7, 7, 7), dtype=torch.float64)
    receivers = torch.zeros_like(mass, dtype=torch.bool)
    demoted = torch.zeros_like(receivers)
    receivers[3, 3, 3] = True
    demoted[3, 4, 3] = True
    mass[3, 3, 3] = 0.00145
    mass[3, 4, 3] = 0.4
    excess = torch.zeros_like(mass)
    excess[3, 3, 2] = -0.15651
    excess[3, 3, 4] = -0.1
    delta, links = _capacity_redistribution(excess, mass, receivers, demoted, True)
    assert float((mass + delta).min()) >= -1e-15
    assert float(delta.sum()) == pytest.approx(float(excess.sum()))
    assert {v["receiver"] for v in links} <= {(3, 3, 3), (3, 4, 3)}
    for donor in ((3, 3, 2), (3, 3, 4)):
        assert sum(v["mass_delta"] for v in links if v["donor"] == donor) == pytest.approx(
            excess[donor].item()
        )
    with pytest.raises(TopologyTransactionError, match="insufficient local"):
        _capacity_redistribution(excess, mass, receivers, torch.zeros_like(demoted))


def test_identical_complete_curve_can_be_scored_and_peak_is_independent():
    case = json.loads(CASE.read_text())
    t, h = reference(case)
    times = np.r_[0, t[:11], (t[10] + t[11]) / 2, t[11:]]
    depths = np.r_[0, h[:11], h.max() * 1.08, h[11:]]
    run = {
        "history": [
            dict(step=i, time_ms=v, depth_over_D=d, mach_max=0.1, rho_min=0.001, volume_drift=0.0)
            for i, (v, d) in enumerate(zip(times, depths))
        ],
        "contact_step": 0,
        "impact_speed_m_s": 1.4,
        "stop_reason": "window_complete",
    }
    r = score_run(run, t, h, case)
    assert r["matched_points"] == 18 and r["max_full_curve_relative_error"] == 0
    assert r["peak_relative_error"] == pytest.approx(0.08) and not r["numeric_pass"]


def test_receiverless_positive_conversion_keeps_droplet_mass():
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.free_surface_lbm import INTERFACE, free_surface_step, init_flags_from_fill

    fill = torch.zeros((7, 7, 7), dtype=torch.float64)
    fill[3, 3, 3] = 0.05
    solid = torch.zeros_like(fill, dtype=torch.bool)
    flags = init_flags_from_fill(fill, solid)
    zero = torch.zeros_like(fill)
    f = equilibrium3d((flags == INTERFACE).to(fill.dtype), zero, zero, zero)
    _, new_fill, new_flags, new_mass, _ = free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), capacity_redistribution=True
    )
    assert new_flags[3, 3, 3].item() == INTERFACE
    assert new_mass.sum().item() == pytest.approx(0.05)
    assert float(new_fill[3, 3, 3]) == pytest.approx(0.05)


def test_basic_interface_flux_is_nonzero_and_antisymmetric():
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.free_surface_lbm import INTERFACE, free_surface_step

    z, y, x = torch.meshgrid(*(torch.arange(6, dtype=torch.float64),) * 3, indexing="ij")
    fill = 0.4 + 0.02 * x
    solid = torch.zeros_like(fill, dtype=torch.bool)
    flags = torch.full_like(fill, INTERFACE, dtype=torch.int8)
    zero = torch.zeros_like(fill)
    f = equilibrium3d(torch.ones_like(fill), 0.01 * x, zero, zero)
    _, _, _, mass, _ = free_surface_step(
        f, fill, flags, solid, mass=fill.clone(), freeze_topology=True, simple_mass_exchange=True
    )
    assert torch.max(torch.abs(mass - fill)) > 1e-5
    assert float(mass.sum()) == pytest.approx(float(fill.sum()), abs=1e-11)


def test_numeric_grid_agreement_keeps_unresolved_experiment_pending(tmp_path):
    from benchmarks.water_entry.single_drop.score import assess

    case = json.loads(CASE.read_text())
    case["domain_and_material_match_confirmed"] = True
    t, h = reference(case)
    paths = []
    for n, offset in ((8, 0.02), (12, 0.01)):
        path = tmp_path / f"grid{n}.json"
        rows = [
            dict(step=i, time_ms=time, depth_over_D=depth, rho_min=1, mach_max=0.1, volume_drift=0)
            for i, (time, depth) in enumerate(zip(np.r_[0, t], np.r_[0, h * (1 - offset)]))
        ]
        path.write_text(
            json.dumps(
                dict(
                    case_id="santini_case3",
                    side_view_measurement={
                        "fitted_downward_speed_m_s": 1.55,
                        "camera_interval_speed_m_s": 1.4,
                    },
                    solver="CACSim/D3Q19",
                    cells_per_D=n,
                    contact_step=0,
                    impact_speed_m_s=1.4,
                    stop_reason="window_complete",
                    history=rows,
                )
            )
        )
        paths.append(path)
    result = assess(paths, case)
    assert result["passed_3pct_and_converged"]
    assert result["status"] == "pending"
    assert result["times_below_experimental_3pct_resolution_ms"] == [t[0]]


def test_projected_half_fill_center_and_velocity():
    from benchmarks.water_entry.single_drop.silhouette import centre_z, summarize

    z, y, x = np.indices((18, 18, 18))
    sphere = 1 / (1 + np.exp(np.sqrt((z - 10.25) ** 2 + (y - 8.5) ** 2 + (x - 8.5) ** 2) - 4))
    core = sphere >= 0.5
    sphere[:2] = 1
    assert centre_z(sphere, core) == pytest.approx(10.25, abs=0.03)
    rows = [dict(time_ms=t, centre_height_mm=2 - 1.4 * t) for t in (0, 0.1, 0.3)]
    assert summarize(rows)["fitted_downward_speed_m_s"] == pytest.approx(1.4)
    assert summarize(rows)["camera_interval_speed_m_s"] is None
    rows.append(dict(time_ms=1.2, centre_height_mm=2 - 1.4 * 1.2))
    assert summarize(rows)["camera_interval_speed_m_s"] == pytest.approx(1.4)
