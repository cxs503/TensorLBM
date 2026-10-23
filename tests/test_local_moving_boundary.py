import copy
import json
import pytest
import torch
from tensorlbm.local_moving_boundary_2d import LocalMovingDisk

@pytest.mark.parametrize('kwargs',[{},dict(center=(39.8,16.1),velocity=(.03,0.)),dict(velocity=(.02,0.),fluid_velocity=(.02,0.))])
def test_conservation_crossing_restart(kwargs):
    sim=LocalMovingDisk(**kwargs);p0=sim.f.clone()
    for _ in range(80):
        entry=sim.step()
        assert abs(entry['mass_residual'])<1e-10
        assert entry['momentum_residual']<1e-10
        assert not entry['global_mass_rescale']
    restored=LocalMovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
    for _ in range(5):sim.step();restored.step()
    assert torch.equal(sim.f,restored.f)
    assert sim.snapshot()==restored.snapshot()
    assert abs(float(sim.f.sum()-p0.sum()))<1e-9

def test_local_transfer_changes_no_far_cells():
    sim=LocalMovingDisk();out=sim.f.clone();available=~sim.solid
    before=out.clone();sim.rest_transfer(out,4,4,.01,available)
    changed=torch.nonzero((out-before).abs().sum(0)>0)
    assert len(changed)==1
    dy,dx=(changed[0]-torch.tensor([4,4])).tolist()
    assert dx*dx+dy*dy<=16
    assert torch.equal(out[1:],before[1:])

def test_capacity_failure_does_not_modify_input():
    sim=LocalMovingDisk();out=torch.zeros_like(sim.f);before=out.clone()
    with pytest.raises(RuntimeError,match='capacity'):sim.rest_transfer(out,4,4,-1.,~sim.solid)
    assert torch.equal(out,before)

def test_failed_step_atomic(monkeypatch):
    sim=LocalMovingDisk();before=sim.snapshot()
    def fail(*args):raise RuntimeError('forced capacity failure')
    monkeypatch.setattr(sim,'rest_transfer',fail)
    with pytest.raises(RuntimeError):sim.step()
    assert sim.snapshot()==before

@pytest.mark.parametrize('radius',[0,9,True,2.5,float('nan')])
def test_bad_radius(radius):
    with pytest.raises(ValueError):LocalMovingDisk(local_radius=radius)

def test_corrupt_restart():
    sim=LocalMovingDisk();state=sim.snapshot();state['f'][0][0][0]=-1.
    with pytest.raises(ValueError):LocalMovingDisk.restore(state)


def test_variable_velocity_restart():
    sim=LocalMovingDisk();sim.step();sim.velocity=torch.tensor([.03,.01],dtype=torch.float64)
    sim.step();restored=LocalMovingDisk.restore(sim.snapshot())
    assert torch.equal(sim.velocity,restored.velocity)
    assert sim.step()==restored.step()
    assert torch.equal(sim.f,restored.f)

def test_bad_current_velocity_atomic():
    sim=LocalMovingDisk();sim.velocity=torch.tensor([float('nan'),0.],dtype=torch.float64)
    before=sim.f.clone()
    with pytest.raises(ValueError):sim.step()
    assert torch.equal(sim.f,before) and sim.time==0
