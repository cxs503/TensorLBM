import copy
import json
import math
import pytest
import torch
from tensorlbm.cut_cell_bgk_2d import CutCellBGK2D

def nonuniform():
    nx,ny=12,10;y,x=torch.meshgrid(torch.arange(ny,dtype=torch.float64),torch.arange(nx,dtype=torch.float64),indexing='ij')
    rho=1000*(1+.001*torch.sin(2*math.pi*x/nx)*torch.cos(2*math.pi*y/ny))
    velocity=torch.stack((.01*torch.sin(2*math.pi*y/ny),.004*torch.cos(2*math.pi*x/nx)),dim=-1)
    return CutCellBGK2D(geometry={'center_m':(.56,.47)},density_kg_m3=rho,velocity_m_s=velocity)


def test_nonuniform_real_transport_and_conservation():
    s=nonuniform();before=s.group_Q.clone();initial_p=s.group_Q.sum(0)@s.c;initial_m=s.group_Q.sum();J=torch.zeros(2,dtype=torch.float64)
    for _ in range(12):
        e=s.step(s.cfl_limit_s*.5);J+=torch.tensor(e['body_impulse_Ns'])
        assert e['minimum_population_density_kg_m3']>=0
        assert abs(e['wall_mass_flux_kg'])<1e-12
        assert abs(e['mass_balance_residual_kg'])<1e-11
        assert max(map(abs,e['momentum_balance_residual_Ns']))<1e-11
    assert not torch.equal(s.group_Q,before)
    assert (s.group_Q.sum(0)@s.c-initial_p+J).abs().max()<1e-7
    assert abs(float(s.group_Q.sum()-initial_m))<1e-10
    assert torch.std(s.group_Q.sum(1)/s.group_volume_m3)>0


def test_collision_moment_invariants_and_exact_relaxation():
    s=CutCellBGK2D();q=s.group_Q.clone();mass=q.sum(1)
    q[:,0]+=mass*.02;q[:,1:5]-=mass[:,None]*.005
    relaxed=s.collision(q,.037)
    assert (relaxed.sum(1)-q.sum(1)).abs().max()<1e-13
    assert ((relaxed-q)@s.c).abs().max()<1e-13
    assert (relaxed-(s.group_Q+(q-s.group_Q)*math.exp(-.037/s.config['relaxation_time_s']))).abs().max()<1e-13


def test_static_equilibrium_with_cut_cells_preserved():
    s=CutCellBGK2D();q=s.group_Q.clone()
    for _ in range(10):s.step(s.cfl_limit_s*.5)
    assert (s.group_Q-q).abs().max()<1e-13


def test_periodic_comoving_uniform_no_body():
    s=CutCellBGK2D(solid_rectangle=False,velocity_m_s=(.02,-.015));q=s.group_Q.clone()
    for _ in range(10):s.step(s.cfl_limit_s*.5)
    assert (s.group_Q-q).abs().max()<1e-13
    assert all(e['body_impulse_Ns']==[0.,0.] for e in s.history)


def test_fixed_merge_actual_and_initial_projection_conservative():
    s=nonuniform()
    assert s.group_count<int((s.volume_m3>0).sum())
    p=s.initial_projection
    assert abs(p['mass_after_kg']-p['mass_before_kg'])<1e-11
    assert torch.tensor(p['momentum_before_Ns']).sub(torch.tensor(p['momentum_after_Ns'])).abs().max()<1e-6
    assert p['maximum_cell_population_change_kg']>0
    assert p['energy_after']['mechanical_J']<=p['energy_before']['mechanical_J']+1e-11


def test_cfl_rejection_atomic():
    s=nonuniform();before=s.snapshot()
    with pytest.raises(ValueError,match='CFL'):s.step(s.cfl_limit_s*1.01)
    assert s.snapshot()==before and s.last_step_raw is None


def test_snapshot_complete_restart_bitwise():
    s=nonuniform()
    for _ in range(4):s.step(s.cfl_limit_s*.4)
    r=CutCellBGK2D.restore(json.loads(json.dumps(s.snapshot())))
    for _ in range(6):
        assert s.step(s.cfl_limit_s*.4)==r.step(r.cfl_limit_s*.4)
        assert torch.equal(s.group_Q,r.group_Q)
    assert s.snapshot()==r.snapshot()

@pytest.mark.parametrize('field,value',[('geometry_sha256','bad'),('time_s',1.),('group_Q_kg',[[float('nan')]]),('schema','bad')])
def test_corrupt_checkpoint_rejected(field,value):
    s=CutCellBGK2D();data=s.snapshot();data[field]=value
    with pytest.raises(ValueError):CutCellBGK2D.restore(data)


def test_checkpoint_state_ledger_mismatch_rejected():
    s=nonuniform();s.step(s.cfl_limit_s*.4);data=s.snapshot();data['group_Q_kg'][0][0]*=1.01
    with pytest.raises(ValueError,match='ledger'):CutCellBGK2D.restore(data)


def test_moving_geometry_explicitly_rejected():
    with pytest.raises(ValueError,match='moving'):CutCellBGK2D(geometry={'velocity_m_s':(.01,0.)})

@pytest.mark.parametrize('kwargs',[dict(relaxation_time_s=0.),dict(merge_fraction=1.),dict(max_cfl=1.1),dict(kinetic_speed_m_s=True),dict(velocity_m_s=(.2,0.))])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):CutCellBGK2D(**kwargs)


def test_physical_relaxation_distinct_from_roll_tau():
    a=CutCellBGK2D(relaxation_time_s=.02);b=CutCellBGK2D(relaxation_time_s=.2)
    assert a.cfl_limit_s==b.cfl_limit_s
    q=a.group_Q.clone();q[:,0]+=q.sum(1)*.02;q[:,1:5]-=q.sum(1)[:,None]*.005
    assert not torch.equal(a.collision(q,.01),b.collision(q,.01))
