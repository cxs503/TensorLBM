"""CPU TDD for D3Q27 restart equivalence."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
import torch
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.cumulant import collide_cumulant_d3q27


@pytest.fixture(scope="module")
def module():
    path = Path(__file__).parents[1] / "examples" / "dg_suboff_cumulant_d3q27_multicard.py"
    spec = importlib.util.spec_from_file_location("d3q27_restart", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _step(module, f, solid, target_mass):
    post = collide_cumulant_d3q27(f, tau=0.83)
    module.halo_exchange(post, rank=0, world_size=1)
    result = module.apply_halfway_bounce_back_27(module.stream27_roll(post), post, solid)
    return result * (target_mass / result[..., 1:-1].sum())


def test_restart_matches_uninterrupted_and_continues_force_accumulators(tmp_path, module):
    torch.manual_seed(20260711)
    nz, ny, nx = 3, 4, 8
    rho = 1.0 + 0.01 * torch.randn(nz, ny, nx)
    initial = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
    solid_global = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid_global[1, 1:3, 2:6] = True
    f0 = torch.cat((initial[..., -1:], initial, initial[..., :1]), dim=-1)
    solid = torch.cat((solid_global[..., -1:], solid_global, solid_global[..., :1]), dim=-1)
    target_mass = initial.sum()
    metadata = module.suboff_checkpoint_metadata(nx=nx, ny=ny, nz=nz, hull_length=24.0,
                                                  re=2000.0, u_in=0.03, y_val=0.5,
                                                  world_size=1, rank=0)
    uninterrupted = f0.clone()
    for _ in range(5):
        uninterrupted = _step(module, uninterrupted, solid, target_mass)
    split = f0.clone()
    for _ in range(2):
        split = _step(module, split, solid, target_mass)
    module.save_suboff_checkpoint(tmp_path / "suboff", f=split, step=2, metadata=metadata,
                                  target_mass=target_mass, mass_cadence=1,
                                  friction_sum=1.25, pressure_sum=2.5, drag_samples=2, rank=0)
    split[..., 0] = 12345.0
    split[..., -1] = -54321.0
    resumed, state = module.load_suboff_checkpoint(tmp_path / "suboff", metadata=metadata,
                                                   shape=split.shape, rank=0, world_size=1,
                                                   device=torch.device("cpu"))
    assert (state["step"], state["friction_sum"], state["pressure_sum"], state["drag_samples"]) == (2, 1.25, 2.5, 2)
    assert torch.equal(resumed[..., 0:1], resumed[..., -2:-1])
    assert torch.equal(resumed[..., -1:], resumed[..., 1:2])
    for _ in range(3):
        resumed = _step(module, resumed, solid, state["target_mass"])
    torch.testing.assert_close(resumed, uninterrupted, rtol=2e-6, atol=2e-6)


def test_restart_rejects_incompatible_static_metadata(tmp_path, module):
    f = torch.ones((27, 1, 1, 6))
    metadata = module.suboff_checkpoint_metadata(nx=4, ny=1, nz=1, hull_length=24.0,
                                                  re=2000.0, u_in=0.03, y_val=0.5,
                                                  world_size=1, rank=0)
    module.save_suboff_checkpoint(tmp_path / "suboff", f=f, step=0, metadata=metadata,
                                  target_mass=torch.tensor(4.0), mass_cadence=100,
                                  friction_sum=0.0, pressure_sum=0.0, drag_samples=0, rank=0)
    incompatible = dict(metadata)
    incompatible["nx"] = 8
    with pytest.raises(ValueError, match="incompatible"):
        module.load_suboff_checkpoint(tmp_path / "suboff", metadata=incompatible,
                                      shape=f.shape, rank=0, world_size=1,
                                      device=torch.device("cpu"))
