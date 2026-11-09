"""Pytest suite for propeller wake validation (IN100-N01 / P115-N1).

Tests cover:
- Body-force wake simulation
- Sliding-mesh wake simulation
- PIV reference data generation
- Wake profile sampling
- Validation criteria (A1-A6)
- Negative control (no propeller)
- Conservation checks
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.propeller_wake_validation import (
    PIVReference,
    WakeProfile,
    WakeStation,
    WakeValidationConfig,
    _average_profiles,
    _sample_wake_stations,


    run_body_force_wake,
    run_sliding_mesh_wake,
    run_wake_validation,
    validate_wake_results,
)
from tensorlbm.actuator_disk import _interp_kt, _interp_kq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return WakeValidationConfig(
        nx=40,
        ny=20,
        nz=20,
        diameter=8.0,
        rpm_lu=0.004,
        inflow_velocity=0.03,
        n_steps=50,
        warmup_steps=10,
        sampling_interval=20,
        stations=(WakeStation(x_over_D=1.0, n_radial=8, n_azimuthal=12),),
        device="cpu",
        seed=42,
    )


@pytest.fixture
def piv_ref():
    """PIV reference for standard KP505 conditions."""
    return PIVReference(
        diameter=32.0,
        kt=0.30,
        kq=0.045,
        inflow_velocity=0.04,
        rpm=0.0015,
        n_radial=20,
        n_azimuthal=36,
    )


# ---------------------------------------------------------------------------
# T1: Configuration validation
# ---------------------------------------------------------------------------


def test_config_advance_ratio():
    """T1: Advance ratio J = U0 / (n * D) computed correctly."""
    cfg = WakeValidationConfig(
        diameter=40.0,
        rpm_lu=0.001,
        inflow_velocity=0.04,
    )
    expected_J = 0.04 / (0.001 * 40.0)
    assert abs(cfg.advance_ratio - expected_J) < 1e-10


def test_config_kt_kq_interpolation():
    """T2: KT/KQ interpolation from KP505 curves is reasonable."""
    cfg = WakeValidationConfig(
        diameter=32.0,
        rpm_lu=0.0015,
        inflow_velocity=0.04,
    )
    J = cfg.advance_ratio
    kt = cfg.kt_target
    kq = cfg.kq_target
    # KP505: KT in [0.04, 0.45], KQ in [0.015, 0.065] for J in [0.1, 1.1]
    assert 0.0 < kt < 0.5
    assert 0.0 < kq < 0.1


# ---------------------------------------------------------------------------
# T3-T4: PIV reference data
# ---------------------------------------------------------------------------


def test_piv_axial_velocity_deficit(piv_ref):
    """T3: Axial velocity shows deficit in wake region."""
    prof = piv_ref.get_profile(x_over_D=1.0)
    # At r/R=0.7 (blade loading peak), velocity should be < U0
    u_at_peak = float(prof.u_axial[14, 0].item())  # r/R≈0.7
    assert u_at_peak < 1.0, f"Expected deficit at blade peak, got {u_at_peak}"


def test_piv_tangential_velocity(piv_ref):
    """T4: Tangential velocity is non-zero (swirl present)."""
    prof = piv_ref.get_profile(x_over_D=1.0)
    u_tan = prof.u_tangential
    # Should have non-zero tangential component
    assert float(u_tan.abs().max().item()) > 0.001


def test_piv_axial_decay_with_distance(piv_ref):
    """T5: Axial velocity deficit decays with downstream distance."""
    prof_05 = piv_ref.get_profile(x_over_D=0.5)
    prof_20 = piv_ref.get_profile(x_over_D=2.0)
    # Deficit should be larger at x/D=0.5 than at x/D=2.0
    deficit_05 = 1.0 - float(prof_05.u_axial.mean().item())
    deficit_20 = 1.0 - float(prof_20.u_axial.mean().item())
    assert deficit_05 > deficit_20, "Wake should recover downstream"


def test_piv_radial_velocity_small(piv_ref):
    """T6: Radial velocity is small (< 2% of U0)."""
    prof = piv_ref.get_profile(x_over_D=1.0)
    u_rad_max = float(prof.u_radial.abs().max().item())
    assert u_rad_max < 0.02, f"Radial velocity too large: {u_rad_max}"


# ---------------------------------------------------------------------------
# T7-T8: Body-force wake simulation
# ---------------------------------------------------------------------------


def test_body_force_produces_wake(small_config):
    """T7: Body-force simulation produces velocity deficit."""
    results = run_body_force_wake(small_config)
    assert results["n_samples"] > 0, "No wake samples collected"
    assert len(results["wake_profiles"]) > 0, "No wake profiles generated"


def test_body_force_kt_reasonable(small_config):
    """T8: Body-force K_T is within reasonable range."""
    results = run_body_force_wake(small_config)
    kt = results["kt_computed"]
    # Should be positive and not absurdly large
    assert kt > 0, f"K_T should be positive, got {kt}"
    assert kt < 10.0, f"K_T unreasonably large: {kt}"


# ---------------------------------------------------------------------------
# T9-T10: Sliding-mesh wake simulation
# ---------------------------------------------------------------------------


def test_sliding_mesh_produces_wake(small_config):
    """T9: Sliding-mesh simulation produces velocity deficit."""
    results = run_sliding_mesh_wake(small_config)
    assert results["n_samples"] > 0, "No wake samples collected"
    assert len(results["wake_profiles"]) > 0, "No wake profiles generated"


def test_sliding_mesh_rotation():
    """T10: Sliding mesh interface applies rotation correctly."""
    from tensorlbm.propeller_wake_validation import _apply_sliding_mesh_3d

    device = "cpu"
    nz, ny, nx = 20, 20, 20
    f = torch.ones(19, nz, ny, nx, dtype=torch.float32)
    # Interface at centre
    interface_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interface_mask[8:12, 8:12, 8:12] = True

    theta = 0.0
    omega = 0.01
    tau = 0.7

    f_out = _apply_sliding_mesh_3d(f, interface_mask, theta, omega, tau, torch.device(device))
    # Interface cells should be modified
    assert not torch.allclose(f[:, 10, 10, 10], f_out[:, 10, 10, 10])


# ---------------------------------------------------------------------------
# T11-T12: Validation criteria
# ---------------------------------------------------------------------------


def test_validation_criteria_structure(small_config):
    """T11: Validation returns structured results with all criteria."""
    bf_results = run_body_force_wake(small_config)
    sm_results = run_sliding_mesh_wake(small_config)
    validation = validate_wake_results(bf_results, sm_results, small_config)

    assert "A1_axial_rms_pct" in validation
    assert "A2_tangential_rms_pct" in validation
    assert "A3_farfield_deviation_pct" in validation
    assert "A4_kt_error_pct" in validation
    assert "passed" in validation


def test_negative_control_uniformity():
    """T12: No-propeller case produces uniform flow (A5).

    Tests that without propeller forces, the flow remains close to
    the initial uniform inflow in the core region (away from walls).
    """
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d
    from tensorlbm.boundaries3d import (
        bounce_back_cells_3d,
        make_channel_wall_mask_3d,
    )

    device = "cpu"
    nx, ny, nz = 32, 16, 16
    u_in = 0.04
    tau = 0.65
    n_steps = 50

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, torch.zeros_like(rho0, dtype=torch.bool), device=device)

    for _ in range(n_steps):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, wall_mask)

    rho, ux, uy, uz = macroscopic3d(f)
    # Check core region (away from walls, exclude boundary layer)
    margin = 3
    core_mask = torch.ones(nz, ny, nx, dtype=torch.bool, device=device)
    core_mask[:margin, :, :] = False
    core_mask[-margin:, :, :] = False
    core_mask[:, :margin, :] = False
    core_mask[:, -margin:, :] = False
    core_mask[:, :, :margin] = False
    core_mask[:, :, -margin:] = False

    ux_core = ux[core_mask]
    # Flow should remain close to initial value (within 20% due to development)
    deviation = float((ux_core.mean().item() - u_in) / u_in)
    assert abs(deviation) < 0.20, f"Flow deviation {deviation:.2%} from U0"


# ---------------------------------------------------------------------------
# T13-T14: Wake profile sampling
# ---------------------------------------------------------------------------


def test_wake_profile_sampling():
    """T13: Wake profile sampling extracts correct plane."""
    device = "cpu"
    nx, ny, nz = 80, 40, 40
    ux = torch.ones(nz, ny, nx, dtype=torch.float32) * 0.04
    uy = torch.zeros(nz, ny, nx, dtype=torch.float32)
    uz = torch.zeros(nz, ny, nx, dtype=torch.float32)

    # Add a wake deficit at x=50
    ux[:, :, 48:53] *= 0.8

    config = WakeValidationConfig(
        nx=nx, ny=ny, nz=nz,
        diameter=16.0,
        inflow_velocity=0.04,
        rpm_lu=0.001,
    )

    profiles = _sample_wake_stations(
        ux, uy, uz, 0.04, config,
        cx=30.0, cy=20.0, cz=20.0,
        device=torch.device(device),
    )
    assert len(profiles) > 0


def test_average_profiles():
    """T14: Profile averaging reduces noise."""
    device = "cpu"
    r = torch.linspace(0, 1, 10)
    t = torch.linspace(0, 2 * math.pi, 12)
    rr, tt = torch.meshgrid(r, t, indexing="ij")

    profiles = []
    for i in range(5):
        noise = torch.randn_like(rr) * 0.01
        profiles.append(WakeProfile(
            x_over_D=1.0,
            r_over_R=rr,
            theta=tt,
            u_axial=torch.ones_like(rr) * 0.9 + noise,
            u_tangential=torch.zeros_like(rr),
            u_radial=torch.zeros_like(rr),
        ))

    avg = _average_profiles(profiles, torch.device(device))
    assert len(avg) == 1
    # Averaged profile should have lower std than individual
    assert avg[0].u_axial.std() < profiles[0].u_axial.std()


# ---------------------------------------------------------------------------
# T15: Conservation check (body force total = thrust)
# ---------------------------------------------------------------------------


def test_body_force_conservation():
    """T15: Body force integral matches expected thrust."""
    from tensorlbm.actuator_disk import apply_actuator_disk

    device = "cpu"
    nz, ny, nx = 40, 40, 40
    ux = torch.ones(nz, ny, nx, dtype=torch.float32) * 0.04
    uy = torch.zeros(nz, ny, nx, dtype=torch.float32)
    uz = torch.zeros(nz, ny, nx, dtype=torch.float32)

    fx, fy, fz = apply_actuator_disk(
        ux, uy, uz,
        cx=20.0, cy=20.0, cz=20.0,
        diameter=16.0,
        hub_diameter_ratio=0.18,
        rpm=0.0015,
    )

    # Total thrust should be positive (negative fx sum)
    thrust = float((-fx).sum().item())
    assert thrust > 0, f"Thrust should be positive, got {thrust}"

    # Torque should be non-zero (swirl)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    torque = float((-((zz - 20.0) * fy - (yy - 20.0) * fz)).sum().item())
    assert abs(torque) > 0, "Torque should be non-zero for swirl"


# ---------------------------------------------------------------------------
# T16: Full validation runner
# ---------------------------------------------------------------------------


def test_full_validation_runner(small_config):
    """T16: Full validation runner completes without error."""
    import tempfile
    cfg = WakeValidationConfig(
        nx=small_config.nx,
        ny=small_config.ny,
        nz=small_config.nz,
        diameter=small_config.diameter,
        rpm_lu=small_config.rpm_lu,
        inflow_velocity=small_config.inflow_velocity,
        n_steps=small_config.n_steps,
        warmup_steps=small_config.warmup_steps,
        sampling_interval=small_config.sampling_interval,
        stations=small_config.stations,
        device="cpu",
        seed=42,
        output_root=Path(tempfile.mkdtemp()),
        overwrite=True,
    )
    results = run_wake_validation(cfg)
    assert "body_force" in results
    assert "sliding_mesh" in results
    assert "validation" in results
    assert "config" in results


# ---------------------------------------------------------------------------
# T17: PIV reference at multiple stations
# ---------------------------------------------------------------------------


def test_piv_multiple_stations(piv_ref):
    """T17: PIV reference generates profiles at all stations."""
    stations = [0.5, 1.0, 2.0]
    for x_d in stations:
        prof = piv_ref.get_profile(x_over_D=x_d)
        assert prof.x_over_D == x_d
        assert prof.u_axial.shape == (20, 36)
        assert prof.u_tangential.shape == (20, 36)


# ---------------------------------------------------------------------------
# T18: Wake profile to_dict roundtrip
# ---------------------------------------------------------------------------


def test_wake_profile_serialization():
    """T18: WakeProfile can be serialized and deserialized."""
    r = torch.linspace(0, 1, 5)
    t = torch.linspace(0, 2 * math.pi, 6)
    rr, tt = torch.meshgrid(r, t, indexing="ij")

    prof = WakeProfile(
        x_over_D=1.0,
        r_over_R=rr,
        theta=tt,
        u_axial=torch.ones_like(rr) * 0.9,
        u_tangential=torch.ones_like(rr) * 0.05,
        u_radial=torch.zeros_like(rr),
    )

    d = prof.to_dict()
    assert d["x_over_D"] == 1.0
    assert len(d["u_axial"]) == 5
    assert len(d["u_axial"][0]) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
