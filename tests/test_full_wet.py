from __future__ import annotations

import ast
import inspect
import json
import math
import textwrap
from pathlib import Path

import pytest
import torch

from tensorlbm.backends.contracts import DeviceSpec
from tensorlbm.backends.torch_backend import TorchBackend
from tensorlbm.boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.models.contracts import ModelComposition
from tensorlbm.models.torch_execution import measure_plan_overhead
from tensorlbm.solver3d import collide_mrt3d, stream3d

from tensorlbm.full_wet import (
    FullyWettedFlowConfig,
    VoxelBodyGeometry,
    run_fully_wetted_flow,
)


def _composition(**overrides: object) -> ModelComposition:
    values: dict[str, object] = {
        "lattice": "D3Q19",
        "collision": "MRT",
        "turbulence": None,
        "forcing": (),
        "boundaries": ("zou_he_channel", "stationary_bounce_back"),
        "physics_modules": {"single_phase": "incompressible"},
    }
    values.update(overrides)
    return ModelComposition(**values)  # type: ignore[arg-type]


def _cube_mask() -> torch.Tensor:
    mask = torch.zeros((7, 9, 11), dtype=torch.bool)
    mask[2:5, 3:6, 4:7] = True
    return mask


def _config(mask: torch.Tensor, *, steps: int = 3, **overrides: object) -> FullyWettedFlowConfig:
    values: dict[str, object] = {
        "geometry": VoxelBodyGeometry(mask=mask, body_id="test-body"),
        "composition": _composition(),
        "device_spec": DeviceSpec("cpu", "float32"),
        "shape": tuple(mask.shape),
        "tau": 0.6,
        "inlet_velocity": 0.04,
        "steps": steps,
    }
    values.update(overrides)
    return FullyWettedFlowConfig(**values)  # type: ignore[arg-type]


def test_voxel_body_geometry_validates_mask_identity_and_resolves_xyz_origin() -> None:
    mask = _cube_mask()
    geometry = VoxelBodyGeometry(mask=mask, body_id="cube")

    assert geometry.resolved_origin == (5.0, 4.0, 3.0)
    assert VoxelBodyGeometry(mask=mask, body_id="cube", origin=(1.0, 2.0, 3.0)).resolved_origin == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        VoxelBodyGeometry(mask=mask.to(torch.float32), body_id="cube")
    with pytest.raises(ValueError):
        VoxelBodyGeometry(mask=torch.zeros_like(mask), body_id="cube")
    with pytest.raises(ValueError):
        VoxelBodyGeometry(mask=mask, body_id="")
    with pytest.raises(ValueError):
        VoxelBodyGeometry(mask=mask, body_id="cube", reference_area=math.inf)
    with pytest.raises(ValueError):
        VoxelBodyGeometry(mask=mask, body_id="cube", origin=(1.0, 2.0))


def test_voxel_body_geometry_snapshots_caller_owned_tensor_storage() -> None:
    mask = _cube_mask()
    geometry = VoxelBodyGeometry(mask=mask, body_id="cube")
    mask.zero_()

    assert geometry.mask.any()
    assert geometry.mask.data_ptr() != mask.data_ptr()


def test_config_rejects_non_r1_backend_device_or_composition() -> None:
    mask = _cube_mask()
    with pytest.raises(ValueError):
        _config(mask, device_spec=DeviceSpec("cuda", "float32"))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(collision="BGK"))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(physics_modules={"single_phase": "compressible"}))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(physics_modules={"single_phase": "incompressible", "free_surface": "korner"}))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(turbulence="smagorinsky"))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(forcing=("guo",)))
    with pytest.raises(ValueError):
        _config(mask, composition=_composition(boundaries=("anything",)))
    with pytest.raises(ValueError):
        _config(mask, shape=(7, 9, 12))
    with pytest.raises(ValueError):
        _config(mask, tau=0.5)
    with pytest.raises(ValueError):
        _config(mask, inlet_velocity=0.15)
    with pytest.raises(ValueError):
        _config(mask, steps=0)


@pytest.mark.parametrize("mask_factory", [_cube_mask])
def test_cube_run_has_finite_fields_reaction_and_fail_closed_evidence(mask_factory) -> None:
    result = run_fully_wetted_flow(_config(mask_factory()))

    assert result.status == "COMPLETED"
    assert torch.isfinite(result.density).all()
    assert torch.isfinite(result.velocity).all()
    assert result.velocity.shape == (3, 7, 9, 11)
    assert result.reaction == tuple(-component for component in result.force)
    assert result.evidence["model_identity"]["lattice"] == "D3Q19"
    assert result.evidence["physical_control_volume"]["status"] == "not_definable"
    assert result.evidence["force"]["phase"] == "post_stream_pre_bounce_back"
    assert result.evidence["force"]["kind"] == "same_phase_momentum_exchange_diagnostic"
    assert set(result.evidence["unsupported"]) >= {
        "free_surface",
        "moving_geometry",
        "physical_control_volume_closure",
        "arbitrary_geometry_physical_accuracy_claim",
    }


def test_sphere_like_mask_runs_on_cpu_float32() -> None:
    z, y, x = torch.meshgrid(torch.arange(7), torch.arange(9), torch.arange(11), indexing="ij")
    sphere = (x - 5).square() + (y - 4).square() + (z - 3).square() <= 4
    result = run_fully_wetted_flow(_config(sphere, steps=2))

    assert result.status == "COMPLETED"
    assert result.density.dtype is torch.float32
    assert result.density.device.type == "cpu"
    assert torch.isfinite(result.velocity).all()


def test_full_wet_result_matches_direct_explicit_existing_kernel_loop_bitwise() -> None:
    config = _config(_cube_mask(), steps=3)
    result = run_fully_wetted_flow(config)
    mask = config.geometry.mask
    nz, ny, nx = config.shape
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=torch.device("cpu"))
    rho = torch.ones(config.shape, dtype=torch.float32)
    ux = torch.full_like(rho, config.inlet_velocity)
    zero = torch.zeros_like(rho)
    ux[mask] = 0.0
    direct = equilibrium3d(rho, ux, zero, zero, device=torch.device("cpu"))
    for _ in range(config.steps):
        direct = stream3d(collide_mrt3d(direct, config.tau))
        direct = apply_zou_he_channel_boundaries_3d(direct, config.inlet_velocity, wall_mask, mask)
    direct_density, direct_ux, direct_uy, direct_uz = macroscopic3d(direct)

    assert torch.equal(result.density, direct_density)
    assert torch.equal(result.velocity, torch.stack((direct_ux, direct_uy, direct_uz)))


def test_hot_loop_ast_is_prebound_and_has_required_existing_operations() -> None:
    function = ast.parse(textwrap.dedent(inspect.getsource(run_fully_wetted_flow))).body[0]
    assert isinstance(function, ast.FunctionDef)
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
    names = {node.id.lower() for node in ast.walk(loop) if isinstance(node, ast.Name)}
    assert not {"config", "geometry", "backend", "json", "logging", "agent", "registry"} & names
    attributes = {node.attr for node in ast.walk(loop) if isinstance(node, ast.Attribute)}
    assert "step" in attributes
    calls = {
        node.func.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"apply_zou_he_channel_boundaries_3d", "macroscopic3d"} <= calls


def test_overhead_observation_writes_non_candidate_cpu_artifact() -> None:
    config = _config(_cube_mask(), steps=1)
    plan = TorchBackend().compile_d3q19_mrt(config.composition, config.tau, config.device_spec)
    rho = torch.ones(config.shape, dtype=torch.float32)
    zero = torch.zeros_like(rho)
    initial = equilibrium3d(rho, zero, zero, zero)
    sample = measure_plan_overhead(
        plan,
        initial,
        lambda state: stream3d(collide_mrt3d(state, config.tau)),
        warmup_steps=1,
        measured_steps=3,
    )
    artifact_path = Path("/tmp/tensorlbm-full-wet-overhead-r1.json")
    artifact_path.write_text(json.dumps({
        "kind": "observation_only",
        "candidate": False,
        "backend": "torch",
        "device": "cpu",
        "dtype": "float32",
        "grid": list(config.shape),
        "ratio": sample.ratio,
        "direct_median_seconds": sample.direct_median_seconds,
        "plan_median_seconds": sample.plan_median_seconds,
        "limitations": "CPU small-grid observation; not representative of GPU, SDAA, or physical accuracy.",
    }, indent=2) + "\n")

    artifact = json.loads(artifact_path.read_text())
    assert artifact["candidate"] is False
    assert artifact["ratio"] == sample.ratio
