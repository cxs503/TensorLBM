"""Conservative stationary solid-mask bounce-back; no moving-mask support."""
import torch
from .d2q9 import C,OPPOSITE
from .solver import collide_bgk_matmul

def bounce_step(f,solid,tau):
    """Solid populations remain zero; link reactions act ON SOLID."""
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=f.dtype)
    safe=f.clone();safe[:,solid]=w[:,None]
    post=collide_bgk_matmul(safe,tau);post[:,solid]=0
    new=torch.zeros_like(f);reaction=torch.zeros(2,dtype=f.dtype)
    for i in range(9):
        cx,cy=map(int,C[i]);incoming=torch.roll(post[i],(cy,cx),(0,1))
        source_solid=torch.roll(solid,(cy,cx),(0,1))&~solid
        incoming[source_solid]=post[int(OPPOSITE[i])][source_solid]
        incoming[solid]=0;new[i]=incoming
        dest_solid=torch.roll(solid,(-cy,-cx),(0,1))&~solid
        reaction+=2*post[i][dest_solid].sum()*C[i]
    return new,reaction

