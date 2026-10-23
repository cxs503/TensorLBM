import copy
import json
import pytest
import torch
from tensorlbm.balanced_local_moving_boundary_2d import BalancedLocalMovingDisk

@pytest.mark.parametrize('kwargs',[{},dict(center=(39.8,16.1),velocity=(.03,0.)),dict(velocity=(.02,0.),fluid_velocity=(.02,0.))])
def test_conservation_crossing_restart(kwargs):
    sim=BalancedLocalMovingDisk(**kwargs);p0=sim.f.clone()
    for _ in range(80):
        entry=sim.step()
        assert abs(entry['mass_residual'])<1e-10
        assert entry['momentum_residual']<1e-10
        assert not entry['global_mass_rescale']
    restored=BalancedLocalMovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
    for _ in range(5):sim.step();restored.step()
    assert torch.equal(sim.f,restored.f)
    assert sim.snapshot()==restored.snapshot()
    assert abs(float(sim.f.sum()-p0.sum()))<1e-9

def test_local_transfer_changes_no_far_cells():
    sim=BalancedLocalMovingDisk();out=sim.f.clone();available=~sim.solid
    before=out.clone();sim.rest_transfer(out,4,4,.01,available)
    changed=torch.nonzero((out-before).abs().sum(0)>0)
    assert len(changed)==1
    dy,dx=(changed[0]-torch.tensor([4,4])).tolist()
    assert dx*dx+dy*dy<=16
    assert torch.equal(out[1:],before[1:])

def test_capacity_failure_does_not_modify_input():
    sim=BalancedLocalMovingDisk();out=torch.zeros_like(sim.f);before=out.clone()
    with pytest.raises(RuntimeError,match='capacity'):sim.rest_transfer(out,4,4,-1.,~sim.solid)
    assert torch.equal(out,before)

def test_failed_step_atomic(monkeypatch):
    sim=BalancedLocalMovingDisk();before=sim.snapshot()
    def fail(*args):raise RuntimeError('forced capacity failure')
    monkeypatch.setattr(sim,'mask',fail)
    with pytest.raises(RuntimeError):sim.step()
    assert sim.snapshot()==before

@pytest.mark.parametrize('radius',[0,9,True,2.5,float('nan')])
def test_bad_radius(radius):
    with pytest.raises(ValueError):BalancedLocalMovingDisk(local_radius=radius)

def test_corrupt_restart():
    sim=BalancedLocalMovingDisk();state=sim.snapshot();state['f'][0][0][0]=-1.
    with pytest.raises(ValueError):BalancedLocalMovingDisk.restore(state)


def test_variable_velocity_restart():
    sim=BalancedLocalMovingDisk();sim.step();sim.velocity=torch.tensor([.03,.01],dtype=torch.float64)
    sim.step();restored=BalancedLocalMovingDisk.restore(sim.snapshot())
    assert torch.equal(sim.velocity,restored.velocity)
    assert sim.step()==restored.step()
    assert torch.equal(sim.f,restored.f)

def test_bad_current_velocity_atomic():
    sim=BalancedLocalMovingDisk();sim.velocity=torch.tensor([float('nan'),0.],dtype=torch.float64)
    before=sim.f.clone()
    with pytest.raises(ValueError):sim.step()
    assert torch.equal(sim.f,before) and sim.time==0


def test_homogeneous_moving_equilibrium_without_node_conversion():
    from tensorlbm.local_moving_boundary_2d import LocalMovingDisk
    kw=dict(velocity=(.00001,0.),fluid_velocity=(.00001,0.))
    sim=BalancedLocalMovingDisk(**kw);old=LocalMovingDisk(**kw);initial=sim.f.clone()
    for _ in range(10):
        entry=sim.step();old.step()
        assert entry['covered']==entry['exposed']==0
        assert abs(entry['wall_impulse_on_solid'][0])<1e-13
    assert torch.max((sim.f-initial).abs()).item()<1e-14
    assert torch.max((old.f-initial).abs()).item()>1e-7

def test_actual_population_conversion_preserves_momentum():
    sim=BalancedLocalMovingDisk(velocity=(.02,0.))
    count=0
    for _ in range(120):
        h=sim.step();count+=h['covered']+h['exposed']
        assert max(map(abs,h['conversion_impulse_on_solid']))<1e-12
    assert count>0

def test_pairing_exercised_by_actual_small_disk():
    sim=BalancedLocalMovingDisk(radius=1.5,center=(16.,16.),velocity=(.05,0.))
    for _ in range(120):sim.step()
    assert sum(h['paired_nodes'] for h in sim.history)>0
