"""Experimental periodic D2Q9 moving disk, lattice units, CPU float64.

Halfway moving-wall reflection and explicit node conversion bookkeeping.
Global density rescaling is an exposed numerical reservoir, not a local
conservative refill model or a qualified curved moving boundary.
"""
import math
import torch
from .d2q9 import C, OPPOSITE
from .solver import collide_bgk_matmul

W = torch.tensor([4/9, *([1/9]*4), *([1/36]*4)], dtype=torch.float64)

def equilibrium(rho, velocity):
    cu = torch.einsum('ia,a->i', C.double(), velocity)
    return W[:, None, None]*rho[None]*(1+3*cu[:, None, None]+4.5*cu[:, None, None]**2-1.5*velocity.square().sum())

def momentum(f):
    return torch.einsum('iyx,ia->a', f, C.double())

class MovingDisk:
    """Prescribed translating circle; solid nodes carry zero populations."""
    def __init__(self, nx=40, ny=32, radius=4., center=(16.2,16.1),
                 velocity=(.01,0.), fluid_velocity=(0.,0.), tau=.8):
        vals=[radius,*center,*velocity,*fluid_velocity,tau]
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in vals) or tau<=.5 or radius<=0:
            raise ValueError('finite parameters, tau > .5 and positive radius required')
        if isinstance(nx,bool) or isinstance(ny,bool) or not isinstance(nx,int) or not isinstance(ny,int) or min(nx,ny)<8:
            raise ValueError('integer grid dimensions >= 8 required')
        if radius>=min(nx,ny)/3 or max(math.hypot(*velocity),math.hypot(*fluid_velocity))>.1:
            raise ValueError('disk/domain or lattice speed outside experimental limits')
        self.config=dict(nx=nx,ny=ny,radius=radius,center=list(center),velocity=list(velocity),fluid_velocity=list(fluid_velocity),tau=tau)
        self.center=torch.tensor(center,dtype=torch.float64)
        self.velocity=torch.tensor(velocity,dtype=torch.float64)
        self.time=0; self.solid=self.mask()
        self.f=equilibrium(torch.ones(ny,nx,dtype=torch.float64),torch.tensor(fluid_velocity,dtype=torch.float64))
        self.f[:,self.solid]=0
        self.target_mass=float(self.f.sum());self.history=[]

    def mask(self):
        y,x=torch.meshgrid(torch.arange(self.config['ny']),torch.arange(self.config['nx']),indexing='ij')
        dx=(x-self.center[0]+self.config['nx']/2)%self.config['nx']-self.config['nx']/2
        dy=(y-self.center[1]+self.config['ny']/2)%self.config['ny']-self.config['ny']/2
        return dx.square()+dy.square()<=self.config['radius']**2

    def step(self):
        old_p=momentum(self.f);old_mass=float(self.f.sum())
        safe=self.f.clone();safe[:,self.solid]=W[:,None]
        rho=safe.sum(0);post=collide_bgk_matmul(safe,self.config['tau']);post[:,self.solid]=0
        out=torch.zeros_like(post);wall_impulse=torch.zeros(2,dtype=torch.float64)
        for i in range(9):
            cx,cy=map(int,C[i]);opp=int(OPPOSITE[i])
            incoming=torch.roll(post[i],(cy,cx),(0,1))
            hit=torch.roll(self.solid,(cy,cx),(0,1))&~self.solid
            reflected=post[opp]+6*W[i]*rho*(C[i].double()@self.velocity)
            incoming[hit]=reflected[hit]
            wall_impulse-=(post[opp][hit]+reflected[hit]).sum()*C[i]
            incoming[self.solid]=0;out[i]=incoming
        bounce_p=momentum(out);bounce_mass=float(out.sum())
        old_solid=self.solid.clone();self.center+=self.velocity;self.time+=1;self.solid=self.mask()
        covered=self.solid&~old_solid;exposed=old_solid&~self.solid
        removed_mass=float(out[:,covered].sum());removed_p=momentum(out*covered[None])
        out[:,covered]=0
        # Local adjacent-fluid density average, wall velocity refill; no
        # guarantee that the converted volume represents the swept geometry.
        available=~old_solid&~self.solid;dens=out.sum(0);total=torch.zeros_like(dens);count=torch.zeros_like(dens)
        for i in range(1,9):
            cx,cy=map(int,C[i]);ok=torch.roll(available,(cy,cx),(0,1))
            total+=torch.roll(dens,(cy,cx),(0,1))*ok;count+=ok
        refill_rho=torch.where(count>0,total/count.clamp_min(1),torch.full_like(dens,bounce_mass/int((~old_solid).sum())))
        refill=equilibrium(refill_rho,self.velocity);out[:,exposed]=refill[:,exposed]
        added_mass=float(refill[:,exposed].sum());added_p=momentum(refill*exposed[None])
        converted_p=momentum(out);before_fix_mass=float(out.sum())
        scale=self.target_mass/before_fix_mass;out*=scale
        reservoir_p=momentum(out)-converted_p;reservoir_mass=self.target_mass-before_fix_mass
        self.f=out;new_p=momentum(out)
        entry=dict(time=self.time,covered=int(covered.sum()),exposed=int(exposed.sum()),
            removed_mass=removed_mass,added_mass=added_mass,removed_momentum=removed_p.tolist(),added_momentum=added_p.tolist(),
            boundary_mass_change=bounce_mass-old_mass,wall_impulse_on_solid=wall_impulse.tolist(),
            conversion_impulse_on_solid=(removed_p-added_p).tolist(),reservoir_mass=reservoir_mass,reservoir_momentum=reservoir_p.tolist(),
            mass=float(out.sum()),momentum=new_p.tolist(),
            bounce_momentum_residual=(bounce_p-old_p+wall_impulse).abs().max().item(),
            momentum_residual=(new_p-old_p+wall_impulse+removed_p-added_p-reservoir_p).abs().max().item(),
            mass_residual=float(out.sum())-old_mass-(bounce_mass-old_mass)-added_mass+removed_mass-reservoir_mass,
            minimum_population=float(out[:,~self.solid].min()))
        if not torch.isfinite(out).all() or entry['minimum_population']<0:
            raise RuntimeError('nonphysical population; reduce prescribed motion')
        self.history.append(entry);return entry

    def snapshot(self):
        return dict(schema='tensorlbm.moving-disk/1',config=self.config,time=self.time,center=self.center.tolist(),target_mass=self.target_mass,f=self.f.tolist(),history=self.history)

    @classmethod
    def restore(cls,data):
        if set(data)!={'schema','config','time','center','target_mass','f','history'} or data['schema']!='tensorlbm.moving-disk/1':
            raise ValueError('restart schema mismatch')
        if set(data['config'])!={'nx','ny','radius','center','velocity','fluid_velocity','tau'}:
            raise ValueError('restart configuration mismatch')
        if type(data['time']) is not int or data['time']<0 or len(data['history'])!=data['time']:
            raise ValueError('restart clock mismatch')
        sim=cls(**data['config']);sim.time=data['time'];sim.center=torch.tensor(data['center'],dtype=torch.float64)
        if sim.center.shape!=(2,) or not torch.isfinite(sim.center).all():
            raise ValueError('restart center invalid')
        sim.solid=sim.mask();sim.f=torch.tensor(data['f'],dtype=torch.float64);sim.target_mass=data['target_mass'];sim.history=data['history']
        if sim.f.shape!=(9,sim.config['ny'],sim.config['nx']) or not torch.isfinite(sim.f).all() or (sim.f<0).any() or sim.f[:,sim.solid].abs().max()!=0:
            raise ValueError('invalid restart populations')
        if not math.isfinite(sim.target_mass) or sim.target_mass<=0 or abs(float(sim.f.sum())-sim.target_mass)>1e-9:
            raise ValueError('invalid restart mass')
        return sim
