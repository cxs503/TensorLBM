import torch
from tensorlbm.fixed_disk_boundary_2d import bounce_step
from tensorlbm.d2q9 import C


def test_solid_link_reflection_and_momentum():
    n=12;solid=torch.zeros(n,n,dtype=torch.bool);solid[5:7,5:7]=True
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f=w[:,None,None].expand(9,n,n).clone();f[:,solid]=0
    cu=C.double()@torch.tensor([.01,0.],dtype=torch.float64)
    f*= (1+3*cu+4.5*cu**2-1.5*.01**2)[:,None,None]
    new,reaction=bounce_step(f,solid,.7)
    assert new[:,solid].abs().max()==0
    torch.testing.assert_close(new.sum(),f.sum(),atol=1e-12,rtol=0)
    delta=torch.einsum('ihw,ij->j',new-f,C.double())
    torch.testing.assert_close(delta,-reaction,atol=1e-12,rtol=0)
    assert reaction[0]>0


def test_uniform_rest_stays_at_rest():
    solid=torch.zeros(16,16,dtype=torch.bool);solid[6:10,6:10]=True
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f=w[:,None,None].expand(9,16,16).clone();f[:,solid]=0
    new,reaction=bounce_step(f,solid,.7)
    torch.testing.assert_close(new,f,atol=1e-14,rtol=0)
    torch.testing.assert_close(reaction,torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
