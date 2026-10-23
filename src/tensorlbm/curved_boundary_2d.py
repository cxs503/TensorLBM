"""Stationary circle BFL interpolation with explicit node-local rest-mass repair.

The repair changes f0 only, preserving momentum. This is an experimental
conservative modification, not a claim of the unmodified BFL accuracy order.
"""
import math
import torch
from .d2q9 import C, OPPOSITE
from .solver import collide_bgk_matmul

class FixedCircle:
    def __init__(self, n, dx, center, radius):
        if not isinstance(n,int) or isinstance(n,bool) or n<8:
            raise ValueError('n must be integer >=8')
        if len(center)!=2 or not all(math.isfinite(v) for v in (*center,dx,radius)) or dx<=0 or radius<=0:
            raise ValueError('finite positive geometry required')
        if any(v-radius<=dx or v+radius>=(n-1)*dx for v in center):
            raise ValueError('circle must clear periodic seams')
        y,x=torch.meshgrid(torch.arange(n,dtype=torch.float64)*dx,torch.arange(n,dtype=torch.float64)*dx,indexing='ij')
        self.solid=(x-center[0])**2+(y-center[1])**2<radius**2
        self.links=[]
        for i in range(1,9):
            cx,cy=map(int,C[i]);mask=~self.solid & torch.roll(self.solid,(-cy,-cx),(0,1))
            yy,xx=mask.nonzero(as_tuple=True)
            rx=x[yy,xx]-center[0];ry=y[yy,xx]-center[1]
            a=dx**2*(cx**2+cy**2);b=2*dx*(rx*cx+ry*cy);c=rx**2+ry**2-radius**2
            q=(-b-torch.sqrt(b*b-4*a*c))/(2*a)
            if ((q<0)|(q>1)).any():raise ValueError('intersection outside link')
            by=(yy-cy)%n;bx=(xx-cx)%n
            if self.solid[by[q<.5],bx[q<.5]].any():raise ValueError('BFL requires fluid rear neighbour')
            self.links.append((i,yy,xx,by,bx,q))

    def step(self,f,tau,interpolated=True):
        if f.dtype!=torch.float64 or f.device.type!='cpu' or f.shape!=(9,*self.solid.shape) or not torch.isfinite(f).all():
            raise ValueError('finite CPU float64 populations required')
        if not math.isfinite(tau) or tau<=.5:raise ValueError('tau must exceed .5')
        if (f[:,self.solid]!=0).any():raise ValueError('solid populations must be zero')
        w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=f.dtype)
        safe=f.clone();safe[:,self.solid]=w[:,None]
        post=collide_bgk_matmul(safe,tau);post[:,self.solid]=0
        new=torch.stack([torch.roll(post[i],(int(C[i,1]),int(C[i,0])),(0,1)) for i in range(9)])
        new[:,self.solid]=0
        reaction=torch.zeros(2,dtype=f.dtype);repair=torch.zeros_like(f[0])
        for i,yy,xx,by,bx,q in self.links:
            j=int(OPPOSITE[i]);out=post[i,yy,xx]
            if interpolated:
                reflected=torch.where(q<.5,2*q*out+(1-2*q)*post[i,by,bx],out/(2*q)+(2*q-1)/(2*q)*post[j,yy,xx])
            else:reflected=out
            new[j,yy,xx]=reflected
            repair[yy,xx]+=out-reflected
            reaction+=(out+reflected).sum()*C[i]
        new[0]+=repair
        return new,reaction,{'rest_mass_repair_l1':float(repair.abs().sum()),'rest_mass_repair_net':float(repair.sum()),'minimum_fluid_population':float(new[:,~self.solid].min())}
