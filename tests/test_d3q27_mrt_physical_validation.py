"""Physical-validation gate for the D3Q27 MRT composition (analytic anchors).

These tests assert that the D3Q27 MRT collide+stream+wall composition
reproduces closed-form Navier-Stokes solutions within the ecosystem accuracy
standard (3 %), and that the acceptance is not vacuous (BGK limit, bulk-rate
independence, ``tau`` sensitivity controls).

The module-level probe is executed once (session scope) on a small grid so
the suite stays in seconds; the full-resolution audit (grids 16/24/32) is
produced by ``tools/d3q27_mrt_physical_validation_audit.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q27 import (
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.d3q27_mrt_physical_validation import (
    ARTIFACT_ID,
    PHYSICS_TOLERANCE,
    check_controls,
    check_periodic_conservation,
    measure_shear_wave_viscosity,
    measure_sound_speed,
    measure_wall_mode_decay,
    run_d3q27_mrt_physical_validation_probe,
    spectral_step_eigenvalues,
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    return run_d3q27_mrt_physical_validation_probe(grids=(16,), wall_grids=(16,))


# ---------------------------------------------------------------------------
# individual anchors
# ---------------------------------------------------------------------------


def test_shear_wave_recovers_analytic_viscosity() -> None:
    """u_x = u0 sin(k y) decays as exp(-2 nu k^2 t) with nu = (tau - 1/2)/3."""
    row = measure_shear_wave_viscosity(16, 0.65, steps=200, sample_every=20)
    assert row["analytic_viscosity"] == pytest.approx(0.05)
    assert row["rel_error"] < PHYSICS_TOLERANCE
    # the mode must actually decay, otherwise the fit would be meaningless
    assert row["energy_last"] < row["energy_first"]


def test_shear_wave_error_is_spatial_truncation() -> None:
    """The viscosity error must shrink with resolution (k^2 truncation)."""
    coarse = measure_shear_wave_viscosity(16, 0.65, steps=200)
    fine = measure_shear_wave_viscosity(32, 0.65, steps=260)
    assert fine["rel_error"] < coarse["rel_error"]
    # second-order-in-space damping error: quartering k^2 quarters the error
    assert coarse["rel_error"] / fine["rel_error"] > 2.0


def test_shear_viscosity_trend_matches_k_squared() -> None:
    """nu_eff/nu - 1 is proportional to k^2 (measured coefficient, not assumed)."""
    rows = [measure_shear_wave_viscosity(n, 0.65, steps=200) for n in (16, 32)]
    coefficients = [
        (row["measured_viscosity"] - row["analytic_viscosity"])
        / row["analytic_viscosity"]
        / row["lattice_wavenumber"] ** 2
        for row in rows
    ]
    assert coefficients[0] == pytest.approx(coefficients[1], rel=0.25)


def test_sound_speed_matches_lattice_speed_of_sound() -> None:
    """A linear acoustic mode travels at c_s = 1/sqrt(3) in lattice units."""
    row = measure_sound_speed(16, 0.65, steps=200)
    assert row["measured_sound_speed"] == pytest.approx(1.0 / math.sqrt(3.0), rel=PHYSICS_TOLERANCE)
    assert row["measured_attenuation"] > 0.0


def test_spectral_eigenvalues_match_time_domain_measurements() -> None:
    """Autograd-Jacobian eigenvalues reproduce the time-domain decay and phase."""
    spectral = spectral_step_eigenvalues(16, 0.65)
    shear = measure_shear_wave_viscosity(16, 0.65, steps=200)
    sound = measure_sound_speed(16, 0.65, steps=200)
    assert spectral["spectral_viscosity"] == pytest.approx(shear["measured_viscosity"], rel=1e-3)
    assert spectral["spectral_sound_speed"] == pytest.approx(
        sound["measured_sound_speed"], rel=1e-3
    )
    assert spectral["n_eigenvalues"] == 27


def test_wall_mode_decay_matches_no_slip_eigenmode() -> None:
    """Bounce-back walls act at half a lattice from the solid cell centres."""
    row = measure_wall_mode_decay(16, 0.65, steps=300)
    assert row["rel_error"] < PHYSICS_TOLERANCE
    assert row["implied_wall_offset"] == pytest.approx(0.5, abs=0.05)


def test_wall_offset_is_resolved_by_the_fit() -> None:
    """The implied wall position must discriminate 0.0 from 0.5 lattice units."""
    wrong_reference = measure_wall_mode_decay(16, 0.65, steps=300, wall_offset=0.0)
    # with the wrong wall convention the analytic rate is ~15 % off
    assert wrong_reference["rel_error"] > 0.05


def test_periodic_conservation_is_round_off() -> None:
    """Mass and momentum are invariants of periodic collide + stream in float64."""
    row = check_periodic_conservation()
    assert row["dtype"] == "float64"
    assert row["finite"]
    assert row["mass_rel_drift"] < 1e-12
    assert row["momentum_rel_drift"] < 1e-10


# ---------------------------------------------------------------------------
# controls: the anchors must be able to fail
# ---------------------------------------------------------------------------


def test_bgk_limit_reproduces_bgk_collision() -> None:
    """MRT with all non-conserved rates = 1/tau is exactly BGK."""
    generator = torch.Generator().manual_seed(7)
    f = torch.rand((27, 3, 4, 5), generator=generator, dtype=torch.float64) + 0.5
    tau = 0.7
    mrt = collide_mrt27(f, tau, s_e=1.0 / tau, s_eps=1.0 / tau, s_q=1.0 / tau, s_pi=1.0 / tau)
    bgk = collide_bgk27(f, tau)
    assert float((mrt - bgk).abs().max()) < 1e-12


def test_controls_block_is_honest() -> None:
    row = check_controls(0.65)
    assert row["bgk_limit_max_abs_delta"] < 1e-12
    assert row["bulk_rate_independence_rel_delta"] < 1e-3
    assert row["tau_control_rel_error"] < 0.01
    assert row["tau_control_rel_shift_vs_base"] > 0.04


def test_tau_change_moves_measured_viscosity() -> None:
    """A wrong tau must be visible: rejection is not vacuous."""
    base = measure_shear_wave_viscosity(16, 0.65, steps=200)
    shifted = measure_shear_wave_viscosity(16, 0.75, steps=200)
    assert (
        abs(shifted["measured_viscosity"] - base["measured_viscosity"]) / base["measured_viscosity"]
        > 0.1
    )


# ---------------------------------------------------------------------------
# artifact contract
# ---------------------------------------------------------------------------


def test_artifact_identity_and_scope(artifact: dict) -> None:
    assert artifact["artifact_id"] == ARTIFACT_ID
    assert artifact["lattice"] == "D3Q27"
    assert artifact["collision"] == "MRT"
    assert artifact["dtype"] == "float64"
    assert artifact["config"]["physics_tolerance"] == PHYSICS_TOLERANCE


def test_artifact_all_checks_pass(artifact: dict) -> None:
    failures = [check for check in artifact["checks"] if check["status"] != "pass"]
    assert failures == []
    assert artifact["summary"]["n_fail"] == 0
    assert artifact["summary"]["n_pass"] == len(artifact["checks"])
    assert artifact["summary"]["max_physics_rel_error"] < PHYSICS_TOLERANCE
    assert artifact["evidence_tier"] == "physical_validation"


def test_artifact_declares_withheld_scope(artifact: dict) -> None:
    withheld = {item["aspect"] for item in artifact["withheld"]}
    assert {
        "turbulence_models",
        "multiphase_and_free_surface",
        "moving_boundaries_and_AMR",
        "accelerator_backends",
        "high_reynolds_stability",
    } <= withheld


def test_artifact_is_self_hashed_and_serializable(artifact: dict) -> None:
    import hashlib
    import json

    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == artifact["artifact_sha256"]
    json.dumps(artifact)


def test_probe_is_deterministic(artifact: dict) -> None:
    repeat = run_d3q27_mrt_physical_validation_probe(grids=(16,), wall_grids=(16,))
    assert repeat["artifact_sha256"] == artifact["artifact_sha256"]


def test_measured_values_are_finite(artifact: dict) -> None:
    for section in ("shear_wave", "sound", "spectral", "wall_mode"):
        for row in artifact["measurements"][section]:
            for key, value in row.items():
                if isinstance(value, float):
                    assert math.isfinite(value), f"{section}.{key} is not finite"


def test_streaming_is_used_by_the_anchors() -> None:
    """Guard against a refactor that silently bypasses the real stepper."""
    import inspect

    import tensorlbm.d3q27_mrt_physical_validation as module

    source = inspect.getsource(module._roll)
    assert "stream27_roll" in source
    assert "collide_mrt27" in source


def test_bounce_back_is_used_by_the_wall_anchor() -> None:
    import inspect

    import tensorlbm.d3q27_mrt_physical_validation as module

    source = inspect.getsource(module.measure_wall_mode_decay)
    assert "bounce_back_cells_27" in source


def test_macroscopic_recovery_is_finite_after_long_run() -> None:
    """End-to-end output composition stays finite well past the anchor window."""
    n = 12
    y = torch.arange(n, dtype=torch.float64)
    profile = torch.sin(2.0 * math.pi * y / n).view(1, n, 1).expand(n, n, n).contiguous()
    zero = torch.zeros((n, n, n), dtype=torch.float64)
    f = equilibrium27(torch.ones((n, n, n), dtype=torch.float64), 0.01 * profile, zero, zero)
    for _ in range(400):
        f = stream27_roll(collide_mrt27(f, 0.65))
    rho, ux, uy, uz = macroscopic27(f)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(ux).all()
    assert float((rho - 1.0).abs().max()) < 1e-3
