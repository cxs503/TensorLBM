"""F-5 regression tests: the four runner-side fixes of Phase 0.5.

Adjudicated scope (``controller_f1b_f4_adjudication.md``, F-5):

a. ``run_suboff_amr_wallmodel(pre_restrict_hook=...)`` forwards the
   engine's pre-restriction stage to a hybrid composition (default
   ``None`` = the legacy loop, unchanged);
b. the hybrid-ii hosting switches (``fine_wall_chains``,
   ``seal_solid_mode``/``seal_erode_passes``, ``root_bounce_seal``)
   default to the legacy monolith wiring;
c. the root envelope CV axis fix — ``torch.nonzero`` on an (nz, ny, nx)
   solid yields (z, y, x) columns, so the x bounds must come from the
   third column; regression evidence on a known field;
d. the finest CV observer is decoupled from ``has_wall_chain``.

The runner-level tests use a small CPU configuration (root 96x48x48,
bare-hull length 48): the identical operator chain at CI scale.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.control_volume_force import box_control_volume, observe_control_volume_force
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.static_block_amr import NestedStaticBlockAMR3D
from tensorlbm.suboff_amr_wallmodel_runner import (
    RunDiagnostics,
    SuboffAMRWallModelConfig,
    _erode_solid,
    _root_envelope_cv_box,
    build_amr_contexts,
    make_advance,
    run_suboff_amr_wallmodel,
)


def _small_config(**overrides) -> SuboffAMRWallModelConfig:
    base: dict = dict(
        nx=96,
        ny=48,
        nz=48,
        hull_length=48.0,
        hull_type="BARE_HULL",
        l1_wall_margin=3,
        l1_wake_cells=8,
        cv_margin=2,
        steps=2,
        warmup_steps=0,
        ramp_steps=4,
        wall_diagnostic_interval=1,
        device="cpu",
    )
    base.update(overrides)
    return SuboffAMRWallModelConfig(**base)


# --------------------------------------------------------------------------
# fix (c): root envelope CV axis regression
# --------------------------------------------------------------------------
def _synthetic_solid() -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Solid with distinct per-axis extents: x long, y short, z medium."""
    nz, ny, nx = 40, 20, 56
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[6:16, 8:13, 10:42] = True  # z 6..15, y 8..12, x 10..41
    return solid, (nz, ny, nx)


def test_root_envelope_cv_box_axis_bounds() -> None:
    solid, (nz, ny, nx) = _synthetic_solid()
    cv, bounds = _root_envelope_cv_box(
        solid,
        (nz, ny, nx),
        margin=4,
        nx=nx,
        ny=ny,
        nz=nz,
        device=torch.device("cpu"),
    )
    # x bounds from the x range (the long axis), not the z range
    assert bounds["x"] == [6, 45]  # max(1, 10-4), min(nx-2, 41+4)
    assert bounds["y"] == [4, 16]  # max(1, 8-4), min(ny-2, 12+4)
    assert bounds["z"] == [2, 19]  # max(1, 6-4), min(nz-2, 15+4)
    assert int(cv.sum().item()) == (45 - 6) * (16 - 4) * (19 - 2)


def test_root_envelope_cv_known_field_magnitude() -> None:
    """Known field pair: uniform ux step -> force = -(rho*du) * N_cv exactly.

    A uniform field has zero net face import by lattice symmetry, so the
    reading isolates the control-volume volume N_cv: any axis-mislabeled
    box misreads the magnitude.
    """
    solid, (nz, ny, nx) = _synthetic_solid()
    cv, bounds = _root_envelope_cv_box(
        solid,
        (nz, ny, nx),
        margin=4,
        nx=nx,
        ny=ny,
        nz=nz,
        device=torch.device("cpu"),
    )
    rho = torch.ones((nz, ny, nx))
    zero = torch.zeros_like(rho)
    f_old = equilibrium3d(rho, torch.full_like(rho, 0.05), zero, zero, device="cpu")
    f_new = equilibrium3d(rho, torch.full_like(rho, 0.09), zero, zero, device="cpu")
    force = observe_control_volume_force(f_old, f_new, f_old, cv).force_on_body[0].item()
    n_cv = int(cv.sum().item())
    expected = -(0.09 - 0.05) * n_cv
    assert force == pytest.approx(expected, rel=1e-5)

    # the pre-fix mislabeled box (x bounds from the z range, z from x,
    # domain-clamped exactly as the buggy inline code did) reads a
    # different magnitude on the same known field
    swapped = box_control_volume(
        (nz, ny, nx),
        x0=max(1, bounds["z"][0]),
        x1=min(nx - 2, bounds["z"][1]),
        y0=max(1, bounds["y"][0]),
        y1=min(ny - 2, bounds["y"][1]),
        z0=max(1, bounds["x"][0]),
        z1=min(nz - 2, bounds["x"][1]),
        device="cpu",
    )
    assert int(swapped.sum().item()) != n_cv
    force_swapped = (
        observe_control_volume_force(f_old, f_new, f_old, swapped).force_on_body[0].item()
    )
    assert force_swapped != pytest.approx(expected, rel=1e-3)


def test_build_amr_contexts_root_cv_matches_axis_projection() -> None:
    config = _small_config()
    built = build_amr_contexts(config, torch.device("cpu"))
    root_solid = built["contexts"][0].solid
    nz, ny, nx = root_solid.shape
    # axis-unambiguous per-axis projections of the CAD solid
    x_idx = torch.nonzero(root_solid.any(dim=(0, 1))).flatten()
    y_idx = torch.nonzero(root_solid.any(dim=(0, 2))).flatten()
    z_idx = torch.nonzero(root_solid.any(dim=(1, 2))).flatten()
    m = config.cv_margin
    exp_bounds = {
        "x": [max(1, int(x_idx.min()) - m), min(nx - 2, int(x_idx.max()) + m)],
        "y": [max(1, int(y_idx.min()) - m), min(ny - 2, int(y_idx.max()) + m)],
        "z": [max(1, int(z_idx.min()) - m), min(nz - 2, int(z_idx.max()) + m)],
    }
    assert built["root_cv_bounds"] == exp_bounds
    expected_cv = box_control_volume(
        (nz, ny, nx),
        x0=exp_bounds["x"][0],
        x1=exp_bounds["x"][1],
        y0=exp_bounds["y"][0],
        y1=exp_bounds["y"][1],
        z0=exp_bounds["z"][0],
        z1=exp_bounds["z"][1],
        device="cpu",
    )
    assert torch.equal(built["contexts"][0].control_volume, expected_cv)


# --------------------------------------------------------------------------
# fix (b): hybrid-ii hosting switches
# --------------------------------------------------------------------------
def test_hybrid_switches_default_reproduce_legacy_wiring() -> None:
    built = build_amr_contexts(_small_config(), torch.device("cpu"))
    contexts = built["contexts"]
    assert len(contexts) == 2
    # legacy: the fine level owns a wall chain and the full-solid seal
    assert contexts[1].chain is not None and contexts[1].has_wall_chain
    assert contexts[1].bounce_seal is contexts[1].solid
    assert contexts[0].bounce_seal is contexts[0].solid
    assert built["bounce_seal_cells_by_level"][1] == built["solid_cells_by_level"][1]


def test_hybrid_switches_hybrid_wiring() -> None:
    config = _small_config(
        fine_wall_chains=False,
        seal_solid_mode="eroded_core",
        seal_erode_passes=3,
        root_bounce_seal=False,
    )
    built = build_amr_contexts(config, torch.device("cpu"))
    contexts = built["contexts"]
    # leaf shell owns the wall: no runner wall chain on the fine level
    assert contexts[1].chain is None and not contexts[1].has_wall_chain
    # root bounce-back off, but the root CV observer keeps the CAD solid
    assert contexts[0].bounce_seal is None and contexts[0].solid is not None
    # the finest bounce seal is the eroded interior core of the full CAD solid
    legacy = build_amr_contexts(_small_config(), torch.device("cpu"))
    full = legacy["contexts"][1].solid
    g = built["ghost"]
    expected_seal = torch.zeros_like(full)
    expected_seal[g:-g, g:-g, g:-g] = _erode_solid(full[g:-g, g:-g, g:-g], 3)
    assert torch.equal(contexts[1].solid, expected_seal)
    assert 0 < built["bounce_seal_cells_by_level"][1] < built["solid_cells_by_level"][1]
    assert built["bounce_seal_cells_by_level"][0] is None


def test_config_validate_rejects_bad_seal_settings() -> None:
    _small_config().validate()  # defaults acceptable
    with pytest.raises(ValueError, match="seal_solid_mode"):
        _small_config(seal_solid_mode="nonsense").validate()
    with pytest.raises(ValueError, match="seal_erode_passes"):
        _small_config(seal_erode_passes=0).validate()


def test_hybrid_switches_run_finite() -> None:
    config = _small_config(
        fine_wall_chains=False,
        seal_solid_mode="eroded_core",
        seal_erode_passes=3,
        root_bounce_seal=False,
    )
    artifact = run_suboff_amr_wallmodel(config)
    assert artifact["result"]["finite"] is True
    assert artifact["amr"]["bounce_seal_cells_by_level"][0] is None
    assert (
        artifact["amr"]["bounce_seal_cells_by_level"][1]
        < (artifact["amr"]["solid_cells_by_level"][1])
    )
    for entry in artifact["ct_time_series"]:
        assert math.isfinite(entry["ct_root_envelope_cv"])


# --------------------------------------------------------------------------
# fix (a): pre_restrict_hook parameter
# --------------------------------------------------------------------------
def test_run_loop_default_passes_hook_none(monkeypatch) -> None:
    recorded: dict[str, list] = {"hooks": []}
    original = NestedStaticBlockAMR3D.step

    def spy(self, advance, *, tau_by_level=None, pre_restrict_hook=None):
        recorded["hooks"].append(pre_restrict_hook)
        return original(
            self, advance, tau_by_level=tau_by_level, pre_restrict_hook=pre_restrict_hook
        )

    monkeypatch.setattr(NestedStaticBlockAMR3D, "step", spy)
    run_suboff_amr_wallmodel(_small_config())
    assert recorded["hooks"] == [None, None]  # one per root step, legacy call


def test_pre_restrict_hook_forwarding_contract() -> None:
    calls: list[tuple[tuple, tuple, list[tuple]]] = []

    def hook(fine_old, fine_new, fine_post):
        calls.append(
            (
                tuple(fine_old.shape),
                tuple(fine_new.shape),
                [tuple(t.shape) for t in fine_post],
            )
        )

    artifact = run_suboff_amr_wallmodel(_small_config(), pre_restrict_hook=hook)
    assert len(calls) == 2  # once per root step, after the fine burst
    allocated = artifact["amr"]["level_shapes_zyx"][-1]
    interior = (19, *(dim - 2 for dim in allocated))  # ghost = 1 in this engine
    for old_shape, new_shape, post_shapes in calls:
        assert old_shape == interior and new_shape == interior
        assert len(post_shapes) == 2  # one post-collision view per fine substep
        assert all(shape == interior for shape in post_shapes)


def test_inert_hook_preserves_legacy_result() -> None:
    base = run_suboff_amr_wallmodel(_small_config())
    hooked = run_suboff_amr_wallmodel(_small_config(), pre_restrict_hook=lambda a, b, c: None)
    assert base["ct_time_series"] == hooked["ct_time_series"]
    assert base["mx_finest_substeps"] == hooked["mx_finest_substeps"]
    assert all(math.isfinite(entry["ct_cv"]) for entry in hooked["ct_time_series"])


# --------------------------------------------------------------------------
# fix (d): finest CV observer decoupled from has_wall_chain
# --------------------------------------------------------------------------
def test_finest_cv_observer_decoupled_from_wall_chain() -> None:
    config = _small_config(fine_wall_chains=False)
    device = torch.device("cpu")
    built = build_amr_contexts(config, device)
    hierarchy = built["hierarchy"]
    diagnostics = RunDiagnostics()
    advance = make_advance(config, built, diagnostics)
    for step in (1, 2):
        diagnostics.step = step
        diagnostics.finest_cv_substeps.clear()
        diagnostics.finest_friction_substeps.clear()
        diagnostics.finest_pressure_substeps.clear()
        hierarchy.step(advance)
        # the finest CV fires once per fine substep even with the chain off
        assert len(diagnostics.finest_cv_substeps) == 2
        assert all(math.isfinite(value) for value in diagnostics.finest_cv_substeps)
        # no chain -> no wall totals recorded alongside
        assert diagnostics.finest_friction_substeps == [0.0, 0.0]
        assert diagnostics.finest_pressure_substeps == [0.0, 0.0]
