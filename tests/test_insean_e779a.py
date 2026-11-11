"""Tests for the INSEAN E779A validation face (P129W3-N04).

The tests are deliberately cheap: geometry, operators and analysis are checked
analytically or against the repository's existing (validated) operators, and
the two lattice runs use a 6-cell-diameter rotor for a handful of steps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import tensorlbm.insean_e779a as m
from tensorlbm.propeller_benchmark import moving_wall_bounce_back_3d_with_reaction

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_spec_matches_drawing_values():
    s = m.E779A_SPEC
    assert s.diameter_m == pytest.approx(0.227)
    assert s.n_blades == 4
    assert s.pitch_ratio_07 == pytest.approx(1.10)
    assert s.expanded_area_ratio == pytest.approx(0.689)
    assert s.hub_diameter_ratio == pytest.approx(0.18)
    assert s.n_rps == pytest.approx(25.0)
    # blade rate = 4 n = 100 Hz
    assert s.n_blades * s.n_rps == pytest.approx(100.0)


def test_advance_speed_and_reynolds():
    s = m.E779A_SPEC
    assert s.advance_speed(0.6) == pytest.approx(0.6 * 25.0 * 0.227)
    assert s.reynolds_diameter(0.6) == pytest.approx(0.6 * 25.0 * 0.227**2 / 1e-6, rel=1e-9)
    assert s.resultant_inflow(0.0, 1.0) == pytest.approx(s.tip_speed())


def test_chord_scale_reproduces_expanded_area_ratio():
    for D in (12.0, 24.0, 48.0):
        area = m._mask_integral_area(D, m.E779A_SPEC)
        ear = area / (math.pi * (0.5 * D) ** 2)
        assert ear == pytest.approx(0.689, abs=1e-6)


def test_chord_shape_peaks_at_declared_radius_and_tapers():
    s = m.E779A_SPEC
    su = np.linspace(0.0, 1.0, 201)
    c = m._chord_shape(su, s)
    i_peak = int(np.argmax(c))
    r_peak = s.hub_radius_ratio + su[i_peak] * (1.0 - s.hub_radius_ratio)
    assert r_peak == pytest.approx(s.chord_peak_rfrac, abs=0.02)
    assert c[0] == pytest.approx(s.chord_root_frac, rel=1e-6)
    assert c[-1] == pytest.approx(s.chord_tip_frac, rel=1e-6)
    assert np.all(np.diff(c[:i_peak]) > 0)


def test_thickness_ratio_thicker_at_root():
    s = m.E779A_SPEC
    r = np.array([0.2, 0.5, 0.7, 0.95])
    t = m._thickness_ratio(r, s)
    assert np.all(np.diff(t) < 0)
    assert t[2] == pytest.approx(s.thickness_ratio_07, rel=1e-6)


def test_naca4_and_camber_shapes():
    xi = np.linspace(0.0, 1.0, 101)
    yt = m._naca4_half_thickness(xi)
    assert yt[0] == pytest.approx(0.0, abs=1e-8)
    # the closed-form returns the *half*-thickness for t/c = 1 (total = 1)
    assert max(yt) == pytest.approx(0.5, abs=2e-2)
    camber = m._camber_line(xi, m.E779A_SPEC)
    assert camber[0] == pytest.approx(0.0)
    assert camber[-1] == pytest.approx(0.0)
    assert camber[50] == pytest.approx(m.E779A_SPEC.camber_ratio, rel=1e-3)


def test_mask_contains_hub_and_all_blades():
    D = 16
    nx, ny, nz = 64, 48, 48
    built = m.build_e779a_mask(nx, ny, nz, 0.35 * nx, 0.5 * (ny - 1), 0.5 * (nz - 1), diameter_lu=D)
    mask, bidx = built["mask"], built["blade_index"]
    assert int(mask.sum()) > 0
    assert set(torch.unique(bidx).tolist()) == {-1, 0, 1, 2, 3}
    counts = [int((bidx == k).sum()) for k in range(4)]
    assert min(counts) > 0
    assert (max(counts) - min(counts)) / max(counts) < 0.15
    # no blade index outside the mask
    assert int((bidx[~mask] >= 0).sum()) == 0


def test_mask_rotation_by_quarter_turn_swaps_blades():
    D = 16
    nx, ny, nz = 64, 48, 48
    a = m.build_e779a_mask(nx, ny, nz, 22, 23.5, 23.5, diameter_lu=D, angle_deg=0.0)
    b = m.build_e779a_mask(nx, ny, nz, 22, 23.5, 23.5, diameter_lu=D, angle_deg=90.0)
    assert int(a["mask"].sum()) == pytest.approx(int(b["mask"].sum()), rel=0.1)
    ca = [int((a["blade_index"] == k).sum()) for k in range(4)]
    cb = [int((b["blade_index"] == k).sum()) for k in range(4)]
    assert (max(ca) - min(ca)) / max(ca) < 0.15
    assert ca == pytest.approx(cb, rel=0.15)


@pytest.mark.parametrize("D", [40, 48])
def test_geometry_gate_measurements(D):
    res = m.geometry_gate(diameter_lu=D, angles_deg=(0.0,))
    pt = res["points"][0]
    # analytic gate: the implemented chord/pitch laws reproduce the drawing
    assert pt["expanded_area_ratio_analytic"] == pytest.approx(0.689, abs=1e-6)
    # independent mask-based pitch measurement
    tol = 0.02 if D >= 48 else 0.035   # voxel pitch regression is 1/D limited
    assert pt["pitch_ratio_07_measured"] == pytest.approx(1.10, rel=tol)
    assert pt["hub_diameter_ratio_measured"] == pytest.approx(0.18, rel=0.05)


# ---------------------------------------------------------------------------
# moving-wall operator
# ---------------------------------------------------------------------------


def _operator_fixture():
    nz, ny, nx = 20, 24, 24
    f = torch.rand(19, nz, ny, nx, dtype=torch.float32) * 0.01 + 1.0 / 19.0
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    mask[5:9, 8:14, 8:12] = True
    uxw, uyw, uzw = m._wall_velocity(None, 10.0, 11.5, 9.5, 0.01, mask)
    return f, mask, uxw, uyw, uzw


def test_masked_moving_wall_matches_repository_operator():
    f, mask, uxw, uyw, uzw = _operator_fixture()
    f_mine = f.clone()
    m._masked_moving_wall(f_mine, mask, uxw, uyw, uzw)
    f_repo, _ = moving_wall_bounce_back_3d_with_reaction(
        f, mask, uxw, uyw, uzw, origin=(10.0, 11.5, 9.5)
    )
    assert torch.allclose(f_mine, f_repo, atol=1e-6)


def test_masked_moving_wall_static_limit_is_bounce_back():
    f, mask, uxw, uyw, uzw = _operator_fixture()
    zero = torch.zeros_like(uxw)
    f_mine = f.clone()
    m._masked_moving_wall(f_mine, mask, zero, zero, zero)
    expected = f.clone()
    expected[:, mask] = f[:, mask][m._OPP]
    assert torch.allclose(f_mine, expected, atol=1e-6)


def test_per_blade_impulses_sum_to_total():
    f, mask, uxw, uyw, uzw = _operator_fixture()
    imp, idx, _ = m._masked_moving_wall(f.clone(), mask, uxw, uyw, uzw)
    total = imp.sum(dim=0)
    bidx = torch.zeros_like(mask, dtype=torch.int8)
    bidx[5:9, 11:, :] = 1                      # split the block into two "blades"
    cell_b = bidx[idx[:, 0], idx[:, 1], idx[:, 2]].to(torch.int64)
    assert set(cell_b.unique().tolist()) == {0, 1}
    parts = imp[cell_b == 0].sum(dim=0) + imp[cell_b == 1].sum(dim=0)
    assert torch.allclose(parts, total, atol=1e-6)


def test_wall_velocity_is_rigid_rotation():
    mask = torch.ones(9, 9, 9, dtype=torch.bool)
    omega = 0.02
    ux, uy, uz = m._wall_velocity(None, 4.0, 4.0, 4.0, omega, mask)
    # at (y,z) = (4+1, 4): u_z = omega*1, v = 0
    assert float(uy[4, 5, 4]) == pytest.approx(0.0, abs=1e-12)
    assert float(uz[4, 5, 4]) == pytest.approx(omega, rel=1e-6)
    assert float(uy[5, 4, 4]) == pytest.approx(-omega, rel=1e-6)


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------


def test_open_water_bc_preserves_uniform_flow():
    nz, ny, nx = 12, 16, 16
    rho = torch.ones((nz, ny, nx))
    f = m._equilibrium_plane(
        torch.ones((nz, ny)), torch.full((nz, ny), 0.05),
        torch.zeros((nz, ny)), torch.zeros((nz, ny)),
    )
    f = f.unsqueeze(-1).repeat(1, 1, 1, nx)
    wall = m._wall_mask(nz, ny, nx, torch.device("cpu"))
    m._open_water_bc(f, 0.05, wall)
    interior = f[:, 2:-2, 2:-2, 1:-1]
    reference = m._equilibrium_plane(
        torch.ones((nz - 4, ny - 4)), torch.full((nz - 4, ny - 4), 0.05),
        torch.zeros((nz - 4, ny - 4)), torch.zeros((nz - 4, ny - 4)),
    ).unsqueeze(-1).repeat(1, 1, 1, nx - 2)
    assert torch.allclose(interior, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def test_block_mean_residual_constant_is_zero():
    hist = np.full(400, 0.25)
    res = m._block_mean_residual(hist, steps_per_rev=100.0, avg_revs=2.0)
    assert res["n_blocks"] == 2
    assert res["max_block_deviation"] == pytest.approx(0.0, abs=1e-12)


def test_block_mean_residual_detects_drift():
    hist = np.linspace(0.1, 0.2, 400)
    res = m._block_mean_residual(hist, steps_per_rev=100.0, avg_revs=2.0)
    assert res["max_block_deviation"] > 0.05
    assert res["last_two_block_drift"] > 0.0


def test_spectrum_recovers_blade_rate_peak():
    n_blades, rate = 4, 1.0 / 240.0
    steps = 240 * 8
    t = np.arange(steps)
    shaft = 2 * math.pi * rate * t
    signal = (
        1.0
        + 0.30 * np.sin(n_blades * shaft)
        + 0.10 * np.sin(2 * n_blades * shaft + 0.4)
    )
    out = m.blade_thrust_spectrum(signal, rate_per_step=rate, n_blades=n_blades)
    assert out["harmonics_bpf"]["k1"]["error_pct"] < 2.0
    assert out["harmonics_bpf"]["k2"]["error_pct"] < 2.0
    assert out["amplitude_ratio_2nd_1st"] == pytest.approx(0.10 / 0.30, rel=0.30)
    assert out["bpf_hz_physical"] == pytest.approx(100.0)


def test_spectrum_of_constant_signal_is_flat():
    out = m.blade_thrust_spectrum(np.ones(512), rate_per_step=1.0 / 180.0, n_blades=4)
    assert out["harmonics_bpf"]["k1"]["amp"] < 1e-9


def test_q_criterion_on_solid_body_rotation():
    n = 24
    z, y, x = np.meshgrid(
        np.arange(n) - n // 2, np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij"
    )
    ux = -y.astype(np.float32)
    uy = x.astype(np.float32)
    uz = np.zeros_like(ux)
    out = m.vortex_criteria_fields(ux, uy, uz)
    core = (slice(4, -4), slice(4, -4), slice(4, -4))
    # solid-body rotation: S = 0, so Q = 0.5 |Omega|^2 = omega^2 with omega = 1
    assert np.allclose(out["Q"][core], 1.0, atol=1e-3)
    # lambda_2 = -omega^2 for a pure rotation: the criterion flags rigid rotation
    # as a "vortex" (rotation dominates strain) -- documented behaviour, and the
    # reason the tip-vortex search below is additionally limited to the wake annulus
    assert np.allclose(out["lambda2"][core], -1.0, atol=1e-3)


def test_q_criterion_on_shear_flow():
    n = 24
    z, y, x = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij"
    )
    ux = y.astype(np.float32)
    uy = np.zeros_like(ux)
    uz = np.zeros_like(ux)
    out = m.vortex_criteria_fields(ux, uy, uz)
    core = (slice(2, -2), slice(2, -2), slice(2, -2))
    assert np.abs(out["Q"][core]).max() < 1e-4
    assert out["lambda2"][core].min() > -1e-3


def test_q_criterion_on_burgers_vortex():
    n = 32
    z, y, x = np.meshgrid(
        np.arange(n) - n / 2, np.arange(n) - n / 2, np.arange(n) - n / 2, indexing="ij"
    )
    r2 = y**2 + z**2
    a = 0.15
    circ = 1.0
    vtheta = circ / (2 * math.pi * np.maximum(np.sqrt(r2), 1e-6)) * (1 - np.exp(-r2 / a))
    ux = np.zeros_like(y, dtype=np.float32)
    uy = (-z * vtheta / np.maximum(np.sqrt(r2), 1e-6)).astype(np.float32)
    uz = (y * vtheta / np.maximum(np.sqrt(r2), 1e-6)).astype(np.float32)
    out = m.vortex_criteria_fields(ux, uy, uz)
    i = j = n // 2
    assert out["lambda2"][i, j, i] < 0.0
    assert out["Q"][i, j, i] > 0.0


def test_gci_recovers_known_order():
    # f(h) = 1.0 + 0.5 h^2 on h = 4, 2, 1 (coarse -> medium -> fine)
    hs = np.array([4.0, 2.0, 1.0])
    f = 1.0 + 0.5 * hs**2
    out = m.gci_roache(list(f), [4.0, 2.0, 1.0])
    assert out["apparent_order_p"] == pytest.approx(2.0, abs=0.05)
    assert out["gci_fine"] < 0.45
    assert out["monotone_convergence"] is True


def test_gci_handles_identical_values():
    out = m.gci_roache([0.3, 0.3, 0.3], [24.0, 32.0, 40.0])
    assert math.isfinite(out["gci_fine"])


def test_anchor_contract_withheld_without_file(tmp_path):
    out = m.load_experimental_anchor(None)
    assert out["status"] == "WITHHELD"
    out2 = m.load_experimental_anchor(tmp_path / "missing.json")
    assert out2["status"] == "WITHHELD"
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"rows": []}))
    assert m.load_experimental_anchor(p)["status"] == "WITHHELD"


def test_anchor_contract_accepts_documented_table(tmp_path):
    p = tmp_path / "anchor.json"
    p.write_text(
        json.dumps(
            {
                "rows": [{"J": 0.4, "KT": 0.3, "KQ": 0.05}, {"J": 0.6, "KT": 0.2, "KQ": 0.04}],
                "source": "unit-test fixture",
                "retrieval": "synthetic",
            }
        )
    )
    out = m.load_experimental_anchor(p)
    assert out["status"] == "REAL"
    assert len(out["rows"]) == 2


def test_openwater_criteria_detects_monotonicity_and_zero_crossing():
    pts = [
        {"j": 0.2, "kT": 0.40, "kQ": 0.06, "eta": 0.21, "convergence": {"thrust": {"max_block_deviation": 0.001}}},
        {"j": 0.6, "kT": 0.20, "kQ": 0.04, "eta": 0.48, "convergence": {"thrust": {"max_block_deviation": 0.002}}},
        {"j": 0.9, "kT": -0.02, "kQ": 0.02, "eta": -0.1, "convergence": {"thrust": {"max_block_deviation": 0.003}}},
    ]
    out = m.openwater_curve_criteria(pts)
    assert out["kt_monotone_decreasing"] is True
    assert out["jt_zero_thrust"] == pytest.approx(0.9 - 0.02 * 0.3 / 0.22, rel=0.05)
    assert out["period_average_residual_lt_1pct"] is True
    assert out["vs_experiment"]["status"] == "WITHHELD"


def test_openwater_criteria_flags_non_monotone():
    pts = [
        {"j": 0.2, "kT": 0.20, "kQ": 0.04, "eta": 0.16, "convergence": {"thrust": {"max_block_deviation": 0.02}}},
        {"j": 0.6, "kT": 0.30, "kQ": 0.05, "eta": 0.57, "convergence": {"thrust": {"max_block_deviation": 0.05}}},
    ]
    out = m.openwater_curve_criteria(pts)
    assert out["kt_monotone_decreasing"] is False
    assert out["period_average_residual_lt_1pct"] is False


def test_bemt_is_monotone_and_decays_with_advance_ratio():
    b = m.bemt_reference(m.E779A_SPEC, js=(0.1, 0.3, 0.5, 0.7, 0.9, 1.3))
    kt = np.array([r["KT"] for r in b["rows"]])
    assert np.all(np.diff(kt) < 0)
    assert kt[0] > 0.3
    assert kt[-1] < 0.1 * kt[0]


def test_bemt_reproduces_kp505_mid_range():
    """The same BEMT, driven with the KP505 geometry, must land near the
    KP505 open-water table shipped in the repository (Fujisawa et al. 2000)."""
    kp = m.E779ASpec(
        n_blades=5,
        pitch_ratio_07=0.95,
        expanded_area_ratio=0.65,
        hub_diameter_ratio=0.18,
        diameter_m=0.25,
    )
    b = m.bemt_reference(kp, js=(0.4, 0.5, 0.6, 0.7))
    ref = {0.4: 0.40, 0.5: 0.37, 0.6: 0.33, 0.7: 0.29}     # KP505 KT table
    for row in b["rows"]:
        j = round(row["J"], 2)
        assert row["KT"] == pytest.approx(ref[j], rel=0.35)


# ---------------------------------------------------------------------------
# lattice runs (tiny rotor, few steps)
# ---------------------------------------------------------------------------


def _tiny_cfg(**kw):
    base = dict(
        diameter_lu=6,
        axial_diameters=3.5,
        lateral_diameters=2.5,
        tip_mach=0.18,
        warmup_revs=0.25,
        avg_revs=0.25,
        save_fields=True,
    )
    base.update(kw)
    return m.OpenWaterConfig(**base)


def test_zero_rotation_zero_inflow_gives_zero_thrust():
    out = m.run_openwater_point(0.0, _tiny_cfg(rotation_sign=0.0, warmup_revs=0.1, avg_revs=0.1))
    assert abs(out["thrust_blades"]) < 1e-6
    assert abs(out["thrust_hub"]) < 1e-6


def test_openwater_point_reports_bookkeeping_and_geometry():
    out = m.run_openwater_point(0.0, _tiny_cfg())
    assert math.isfinite(out["kT"])
    assert out["phase_increment_deg"] <= 2.0
    # at D = 6 the four blades are not lattice-symmetric and the window is a
    # quarter turn; production grids assert the tighter 2 % band instead
    assert 0.0 < out["blade_thrust_spread_pct"] < 15.0
    assert out["convergence"]["thrust"]["n_blocks"] >= 1
    assert out["fields"]["rho"].shape == tuple(reversed(out["grid"]))  # (nz, ny, nx)


def test_negative_control_moving_wall_off_reduces_thrust():
    """Disabling the moving-wall velocity (blades still rotate geometrically)
    must remove most of the thrust: the momentum exchange is the load model."""
    rot = m.run_openwater_point(0.3, _tiny_cfg(avg_revs=0.5, warmup_revs=0.5))
    no_rot = m.run_openwater_point(
        0.3, _tiny_cfg(avg_revs=0.5, warmup_revs=0.5, rotation_sign=0.0)
    )
    assert abs(no_rot["thrust_blades"]) < 0.5 * abs(rot["thrust_blades"])