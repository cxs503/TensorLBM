import pytest
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C

def state(boundary):
    gen=torch.Generator().manual_seed(4)
    f=torch.rand((9,*boundary.solid.shape),generator=gen,dtype=torch.float64)*.002
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f+=w[:,None,None];f[:,boundary.solid]=0
    return f

@pytest.mark.parametrize('interpolated',[False,True])
def test_random_mass_momentum(interpolated):
    b=FixedCircle(24,1/24,(.503,.497),.125);f=state(b)
    new,reaction,info=b.step(f,.8,interpolated)
    assert abs(float(new.sum()-f.sum()))<1e-10
    delta=torch.einsum('ihw,ij->j',new-f,C.double())
    assert torch.max(torch.abs(delta+reaction))<1e-12
    assert (new[:,b.solid]==0).all()
    assert info['minimum_fluid_population']>0
    if interpolated: assert info['rest_mass_repair_l1']>0
    else:assert info['rest_mass_repair_l1']==0

def test_exact_circle_intersections_both_branches():
    b=FixedCircle(32,1/32,(.503,.497),.125)
    qs=[]
    for i,yy,xx,by,bx,q in b.links:
        point=torch.stack((xx.double()/32-.503,yy.double()/32-.497),dim=1)+q[:,None]*C[i]/32
        assert torch.max(torch.abs(point.square().sum(1)-.125**2))<1e-15
        qs.extend(q.tolist())
    assert min(qs)<.5<max(qs)

def test_equilibrium_rest_unchanged():
    b=FixedCircle(24,1/24,(.503,.497),.125)
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f=w[:,None,None].expand(9,24,24).clone();f[:,b.solid]=0
    new,force,info=b.step(f,.8)
    assert torch.max(abs(new-f))<1e-15
    assert force.abs().max()<1e-15

@pytest.mark.parametrize('bad',[(True,1/24,(.5,.5),.1),(24,0,(.5,.5),.1),(24,1/24,(.1,.5),.125),(24,1/24,(float('nan'),.5),.1)])
def test_invalid_geometry(bad):
    with pytest.raises(ValueError):FixedCircle(*bad)

@pytest.mark.parametrize('kind',['tau','solid','nan','dtype'])
def test_invalid_step(kind):
    b=FixedCircle(24,1/24,(.503,.497),.125);f=state(b);tau=.8
    if kind=='tau':tau=.5
    elif kind=='solid':f[:,b.solid]=.1
    elif kind=='nan':f[0,0,0]=float('nan')
    else:f=f.float()
    with pytest.raises(ValueError):b.step(f,tau)
