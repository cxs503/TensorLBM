"""Fail-fast contract for lattice / feature compatibility in ``GeneralSimEngine``.

Audit ``docs/AUDIT_RUNTEST_6_TENSORLBM.md`` closed the *scalar* input-contract
holes (D5/D6/D7/D9): illegal floats are now rejected at the entry point instead of
poisoning the field inside the time loop.  The same defect class survived one
level up — an **unsupported configuration** (``lattice=D3Q27`` plus a feature that
only has a D3Q19 kernel) passed ``GeneralSimConfig.validate()`` *and* ``setup()``
and only failed inside ``run()``, after the geometry had been voxelised and the
27-direction distribution function allocated.

These tests pin the tightened contract:

* the rejection happens in ``setup()`` with **nothing allocated**
  (``uc``/``solid``/``f`` all still ``None``) — see
  :func:`test_rejection_leaves_no_state_allocated`;
* the message names the offending feature (not just "unsupported");
* ``AUTO`` selections are resolved by Reynolds number before the check, so a
  supported auto-selection is *not* falsely rejected;
* the D3Q19 path is untouched (no blanket rejection): every feature that has a
  D3Q19 kernel still builds and still advances a finite field.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.general_sim import (
    CollisionModel,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)

torch.set_num_threads(min(8, torch.get_num_threads()))


def _engine(
    lattice: LatticeModel = LatticeModel.D3Q27,
    *,
    collision: CollisionModel = CollisionModel.BGK,
    wall: WallTreatment = WallTreatment.AUTO,
    geometry: GeometrySource = GeometrySource.PARAMETRIC_SPHERE,
    viscosity: float = 1.0e-6,
    steps: int = 4,
) -> GeneralSimEngine:
    cfg = GeneralSimConfig(
        name=f"contract_{lattice.value}",
        geometry=GeometryConfig(
            source=geometry,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=lattice,
            collision=collision,
            wall_treatment=wall,
            resolution=8,
            max_steps=steps,
            snapshot_interval=10**9,
            force_sample_interval=2,
            device="cpu",
            mass_correction=False,
        ),
        output=OutputConfig(
            directory=f"/tmp/tensorlbm_contract_{lattice.value}",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=False,
        ),
    )
    return GeneralSimEngine(cfg)


# ---------------------------------------------------------------------------
# Unsupported D3Q27 combinations are refused before any allocation
# ---------------------------------------------------------------------------

UNSUPPORTED_D3Q27 = [
    ("wall_function", dict(wall=WallTreatment.WALL_FUNCTION), "wall_function"),
    ("smagorinsky_mrt", dict(collision=CollisionModel.SMAGORINSKY_MRT), "smagorinsky_mrt"),
    ("smagorinsky_bgk", dict(collision=CollisionModel.SMAGORINSKY_BGK), "smagorinsky_bgk"),
    ("trt", dict(collision=CollisionModel.TRT), "trt"),
]


@pytest.mark.parametrize("label,kwargs,needle", UNSUPPORTED_D3Q27)
def test_unsupported_d3q27_feature_rejected_at_setup(label, kwargs, needle):
    """setup() raises ``NotImplementedError`` naming the feature, not a generic message."""
    eng = _engine(**kwargs)
    with pytest.raises(NotImplementedError) as ei:
        eng.setup()
    msg = str(ei.value)
    assert needle in msg, msg
    assert "D3Q19" in msg, "message must point at the working alternative"


@pytest.mark.parametrize("label,kwargs,needle", UNSUPPORTED_D3Q27)
def test_rejection_leaves_no_state_allocated(label, kwargs, needle):
    """The refusal must cost zero voxelisation and zero field allocation.

    Pre-fix the same configuration raised inside ``run()`` — i.e. only after
    ``setup()`` had filled ``uc``/``solid``/``near``/``f`` (measured 1.74 MB of
    ``f`` at the 8-cell probe resolution, ``artifacts/contract_guards/
    lattice_probe_before.json``).
    """
    eng = _engine(**kwargs)
    with pytest.raises(NotImplementedError):
        eng.setup()
    assert eng.f is None
    assert eng.solid is None
    assert eng.uc is None
    assert eng.near is None
    assert eng.step_count == 0


def test_rejection_is_config_only_and_repeatable():
    """The guard is a pure classification: setup() fails every time, no partial state."""
    eng = _engine(wall=WallTreatment.WALL_FUNCTION)
    for _ in range(3):
        with pytest.raises(NotImplementedError, match="wall_function"):
            eng.setup()
        assert eng.f is None


# ---------------------------------------------------------------------------
# AUTO selections: resolved by Re, so no false rejection and no false acceptance
# ---------------------------------------------------------------------------


def test_auto_low_re_resolves_to_supported_d3q27_path():
    """Re=100 → bounce-back + MRT/BGK: supported, must keep working."""
    eng = _engine()  # Re = 1e-4 * 1.0 / 1e-6 = 100
    assert eng.config.reynolds_number == pytest.approx(100.0)
    info = eng.setup()
    assert info["q"] == 27
    assert info["auto_wall_treatment"] == WallTreatment.BOUNCE_BACK.value
    out = eng.run(steps=4)
    assert out["diverged"] is False


def test_auto_high_re_wall_function_rejected_before_allocation():
    """Re ≥ 1e4 → AUTO resolves to the wall function, which has no D3Q27 kernel."""
    eng = _engine(viscosity=1.0e-8)  # Re = 1e4
    assert eng.config.reynolds_number >= GeneralSimEngine.RE_WALL_FN_THRESHOLD
    with pytest.raises(NotImplementedError) as ei:
        eng.setup()
    assert "auto-selected" in str(ei.value)
    assert eng.f is None


def test_auto_collision_high_re_rejected_but_low_re_accepted():
    """AUTO collision: MRT at low Re (ok), Smagorinsky-MRT at Re ≥ 1000 (no 27 kernel)."""
    ok = _engine(collision=CollisionModel.AUTO)
    info = ok.setup()
    assert info["q"] == 27

    bad = _engine(collision=CollisionModel.AUTO, viscosity=1.0e-7)  # Re = 1000
    assert bad.config.reynolds_number >= GeneralSimEngine.RE_MRT_SMAG_THRESHOLD
    with pytest.raises(NotImplementedError, match="smagorinsky_mrt"):
        bad.setup()


def test_auto_wall_function_exempt_when_no_solid():
    """Empty channel (no solid): the wall function has nothing to model, so keep it runnable.

    This is the one exemption in the guard — it preserves the previous behaviour
    for the ``GeometrySource.NONE`` channel at Re ≥ 1e4.
    """
    eng = _engine(geometry=GeometrySource.NONE, viscosity=1.0e-8)
    assert eng.config.reynolds_number >= GeneralSimEngine.RE_WALL_FN_THRESHOLD
    info = eng.setup()  # must not raise
    assert info["q"] == 27


# ---------------------------------------------------------------------------
# D3Q19 control: no blanket rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collision,wall",
    [
        (CollisionModel.BGK, WallTreatment.BOUNCE_BACK),
        (CollisionModel.MRT, WallTreatment.BOUNCE_BACK),
        (CollisionModel.SMAGORINSKY_MRT, WallTreatment.BOUNCE_BACK),
        (CollisionModel.BGK, WallTreatment.WALL_FUNCTION),
    ],
)
def test_d3q19_features_still_run(collision, wall):
    eng = _engine(LatticeModel.D3Q19, collision=collision, wall=wall, steps=4)
    info = eng.setup()
    assert info["q"] == 19
    assert not eng.is_d3q27
    out = eng.run(steps=4)
    assert out["diverged"] is False


# ---------------------------------------------------------------------------
# setup() info contract
# ---------------------------------------------------------------------------


def test_setup_info_reports_lattice_and_direction_count():
    """``q``/``Q``/``lattice``/``is_d3q27`` must agree (no upper/lower-case drift)."""
    for lattice, q in ((LatticeModel.D3Q27, 27), (LatticeModel.D3Q19, 19)):
        eng = _engine(lattice)
        info = eng.setup()
        assert info["lattice"] == lattice.value
        assert info["q"] == q
        assert info["Q"] == q
        assert eng.is_d3q27 is (q == 27)
        assert eng.f.shape[0] == q
