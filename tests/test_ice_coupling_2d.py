import json
import pytest
import torch
pytest.importorskip('tensordem')
from tensorlbm.ice_coupling_2d import CoupledIce2D,CoupledIceConfig,bilinear_map,cross2


def test_mapping_force_moment_and_work():
    x=torch.tensor([[.132,.267],[.349,.456]],dtype=torch.float64)
    J=bilinear_map(x,(24,24),.025)
    yy,xx=torch.meshgrid(torch.arange(24)*.025,torch.arange(24)*.025,indexing='ij')
    grid=torch.stack((xx,yy),-1).double().reshape(-1,2)
    # Use exact double coordinate construction, independent of map.
    grid=torch.stack(torch.meshgrid(torch.arange(24,dtype=torch.float64)*.025,
        torch.arange(24,dtype=torch.float64)*.025,indexing='ij'),-1).flip(-1).reshape(-1,2)
    F=torch.tensor([[3.,-5.],[-2.,7.]],dtype=torch.float64)
    u=grid@torch.tensor([[.2,-.1],[.3,.4]],dtype=torch.float64)
    spread=J.T@F
    torch.testing.assert_close(spread.sum(0),F.sum(0),rtol=0,atol=1e-14)
    torch.testing.assert_close(cross2(grid,spread).sum(),cross2(x,F).sum(),rtol=0,atol=1e-14)
    torch.testing.assert_close((spread*u).sum(),(F*(J@u)).sum(),rtol=0,atol=1e-14)


def test_real_coupling_and_mid_exchange_restart():
    sim=CoupledIce2D(CoupledIceConfig(duration_s=.01,exchange_steps=3))
    for _ in range(5):sim.step()
    resumed=CoupledIce2D.from_snapshot(json.loads(json.dumps(sim.snapshot())))
    for _ in range(12):
        a=sim.step(); b=resumed.step()
    torch.testing.assert_close(sim.f,resumed.f,rtol=0,atol=0)
    torch.testing.assert_close(sim.dem.positions,resumed.dem.positions,rtol=0,atol=0)
    assert a==b
    assert a['fluid_kinetic_J']>0
    assert max(abs(v) for v in a['momentum_residual_kg_m_s'])<1e-10
    assert a['mass_relative_error']<1e-12
    assert a['force_mapping_error_N']<1e-12
    assert a['moment_mapping_error_Nm']<1e-12
    assert a['adjoint_power_error_W']<1e-12


def test_dry_no_hidden_fluid_load():
    sim=CoupledIce2D(CoupledIceConfig(wet=False,duration_s=.002))
    for _ in range(4):r=sim.step()
    assert r['fluid_kinetic_J']==0
    assert r['fluid_fy_n']==0
    assert r['mass_relative_error']==0


def test_reject_out_of_support():
    with pytest.raises(ValueError):bilinear_map(torch.tensor([[-.01,.2]]),(24,24),.025)

@pytest.mark.parametrize('bad',['population','index','substeps','unknown_scalar','vector','exchange'])
def test_restart_rejects_invalid_state(bad):
    s=CoupledIce2D(CoupledIceConfig(duration_s=.002));s.step();r=s.snapshot()
    if bad=='population':r['f'][0][0][0]=-1
    elif bad=='index':r['step_index']=-1
    elif bad=='substeps':r['substeps']=999
    elif bad=='unknown_scalar':r['scalars']['step']=42
    elif bad=='vector':r['vectors']['external_impulse']=[1.]
    else:r['exchange']['ice_load']=[[1.,2.]]
    with pytest.raises(ValueError):CoupledIce2D.from_snapshot(r)
