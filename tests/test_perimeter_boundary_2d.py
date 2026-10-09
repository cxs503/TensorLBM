import pytest
import torch
from tensorlbm.perimeter_boundary_2d import circle_markers,project_velocity
from tensorlbm.ice_coupling_2d import cross2


def test_direct_projection_and_wrench_work():
    positions,_=circle_markers([.5,.5],.125,.05)
    gen=torch.Generator().manual_seed(47)
    u=torch.randn(32*32,2,generator=gen,dtype=torch.float64)*.02
    target=torch.zeros(len(positions),2,dtype=torch.float64)
    result=project_velocity(positions,(32,32),1/32,u,.2,target)
    assert result['slip'].abs().max()<1e-12
    marker=result['marker_impulse'];gridI=result['grid_impulse']
    grid=torch.stack(torch.meshgrid(torch.arange(32)/32,torch.arange(32)/32,indexing='ij'),-1).flip(-1).reshape(-1,2).double()
    torch.testing.assert_close(marker.sum(0),gridI.sum(0),atol=1e-12,rtol=0)
    torch.testing.assert_close(cross2(positions,marker).sum(),cross2(grid,gridI).sum(),atol=1e-12,rtol=0)
    torch.testing.assert_close((result['J']@u*marker).sum(),(u*gridI).sum(),atol=1e-12,rtol=0)


def test_compliance_relation():
    positions,_=circle_markers([.5,.5],.125,.05)
    u=torch.ones(32*32,2,dtype=torch.float64)*.02
    target=torch.zeros(len(positions),2,dtype=torch.float64)
    result=project_velocity(positions,(32,32),1/32,u,.2,target,.1)
    torch.testing.assert_close(result['slip'],-.1*result['marker_impulse'],atol=1e-12,rtol=0)


@pytest.mark.parametrize('compliance',[-1,float('nan'),True])
def test_bad_compliance(compliance):
    p,_=circle_markers([.5,.5],.125,.05)
    with pytest.raises(ValueError):
        project_velocity(p,(32,32),1/32,torch.zeros(1024,2),.2,torch.zeros(len(p),2),compliance)


@pytest.mark.parametrize('mass',[0,-1,float('nan')])
def test_bad_mass(mass):
    p,_=circle_markers([.5,.5],.125,.05)
    with pytest.raises(ValueError):
        project_velocity(p,(32,32),1/32,torch.zeros(1024,2),mass,torch.zeros(len(p),2))
