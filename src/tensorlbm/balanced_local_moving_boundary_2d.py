"""Equilibrium-reference moving-wall mass repair; experimental lattice units.

No global reservoir. Integer node conversion is not exact swept-volume geometry.
Transfers span at most ``local_radius`` periodic lattice cells. Failure is atomic.
"""
import copy
import math
import torch
from .moving_boundary_2d import MovingDisk, W, equilibrium, momentum
from .d2q9 import C, OPPOSITE
from .solver import collide_bgk_matmul

class BalancedLocalMovingDisk(MovingDisk):
    def __init__(self, local_radius=4, **kwargs):
        if type(local_radius) is not int or not 1 <= local_radius <= 8:
            raise ValueError('local_radius must be an integer in [1,8]')
        super().__init__(**kwargs)
        self.local_radius=local_radius

    def neighbors(self,y,x,available):
        ny,nx=available.shape
        result=[]
        for dy in range(-self.local_radius,self.local_radius+1):
            for dx in range(-self.local_radius,self.local_radius+1):
                if dx*dx+dy*dy <= self.local_radius**2 and (dx or dy):
                    yy,xx=(y+dy)%ny,(x+dx)%nx
                    if available[yy,xx]: result.append((yy,xx))
        return sorted(set(result),key=lambda p: (((p[0]-y+ny//2)%ny-ny//2)**2+((p[1]-x+nx//2)%nx-nx//2)**2,p))

    def rest_transfer(self,out,y,x,amount,available):
        neighbors=self.neighbors(y,x,available)
        if not neighbors: raise RuntimeError('no local mass receiver')
        if amount >= 0:
            out[0,neighbors[0][0],neighbors[0][1]]+=amount
            return
        needed=-amount
        capacity=sum(float(out[0,yy,xx]) for yy,xx in neighbors)
        if capacity < needed+1e-14: raise RuntimeError('insufficient local positive restpopulation capacity')
        for yy,xx in neighbors:
            take=min(needed,float(out[0,yy,xx])*.999999999999)
            out[0,yy,xx]-=take;needed-=take
            if needed<1e-14:break
        if needed>1e-12:raise RuntimeError('local transfer failed')

    def step(self):
        if self.velocity.shape != (2,) or not torch.isfinite(self.velocity).all() or float(self.velocity.norm()) > .1:
            raise ValueError("current lattice velocity must be finite and magnitude <= .1")
        old_p=momentum(self.f);old_mass=float(self.f.sum())
        safe=self.f.clone();safe[:,self.solid]=W[:,None]
        rho=safe.sum(0);post=collide_bgk_matmul(safe,self.config['tau']);post[:,self.solid]=0
        out=torch.zeros_like(post);wall=torch.zeros(2,dtype=torch.float64)
        delta=torch.zeros_like(rho)
        for i in range(9):
            cx,cy=map(int,C[i]);opp=int(OPPOSITE[i])
            incoming=torch.roll(post[i],(cy,cx),(0,1))
            hit=torch.roll(self.solid,(cy,cx),(0,1))&~self.solid
            reflected=post[opp]+6*W[i]*rho*(C[i].double()@self.velocity)
            # Subtract the equilibrium-reference link flux. On a closed
            # periodic mask its sum is identically zero, so this preserves
            # total mass without perturbing homogeneous moving equilibrium.
            reference=6*W[i]*(C[i].double()@self.velocity)
            delta[hit]+=(reflected-post[opp]-reference)[hit]
            incoming[hit]=reflected[hit]
            wall-=(post[opp][hit]+reflected[hit]).sum()*C[i]
            incoming[self.solid]=0;out[i]=incoming
        for y,x in torch.nonzero(delta).tolist():
            amount=float(delta[y,x])
            if float(out[0,y,x])-amount<0:raise RuntimeError("insufficient local rest capacity")
            out[0,y,x]-=amount
        bounce_p=momentum(out)
        new_center=self.center+self.velocity
        original_center=self.center
        self.center=new_center
        try:new_solid=self.mask()
        finally:self.center=original_center
        covered=new_solid&~self.solid;exposed=self.solid&~new_solid
        available=~self.solid&~new_solid
        paired=0
        unpaired_exposed=exposed.clone()
        for y,x in torch.nonzero(covered).tolist():
            partners=self.neighbors(y,x,unpaired_exposed)
            if partners:
                yy,xx=partners[0]
                out[:,yy,xx]=out[:,y,x];out[:,y,x]=0
                unpaired_exposed[yy,xx]=False;paired+=1
                continue
            neighbors=self.neighbors(y,x,available)
            if not neighbors:raise RuntimeError('covered node has no local receiver')
            # Spread all populations equally: preserves mass and momentum
            # while avoiding a nearest-node density doubling.
            parcel=out[:,y,x].clone()/len(neighbors)
            for yy,xx in neighbors:out[:,yy,xx]+=parcel
            out[:,y,x]=0
        conversion_before=momentum(out)
        transferred=0.
        for y,x in torch.nonzero(unpaired_exposed).tolist():
            neighbors=self.neighbors(y,x,available)
            if not neighbors:raise RuntimeError('exposed node has no local donor')
            desired=sum(float(out[:,yy,xx].sum()) for yy,xx in neighbors)/len(neighbors)
            capacity=sum(float(out[:,yy,xx].sum()) for yy,xx in neighbors)
            if desired<=0 or capacity<=desired:raise RuntimeError('insufficient local refill capacity')
            fraction=desired/capacity
            parcel=torch.zeros(9,dtype=torch.float64)
            for yy,xx in neighbors:
                take=out[:,yy,xx]*fraction
                parcel+=take;out[:,yy,xx]-=take
            # Actual donor populations preserve mass AND momentum; no
            # artificial re-equilibration at prescribed wall velocity.
            out[:,y,x]=parcel
            transferred+=desired
        conversion=conversion_before-momentum(out)
        new_p=momentum(out)
        if not torch.isfinite(out).all() or (out<0).any():raise RuntimeError('nonpositive local populations')
        mass_error=float(out.sum())-old_mass
        if abs(mass_error)>1e-9:raise RuntimeError('local mass conservation failed')
        entry=dict(time=self.time+1,covered=int(covered.sum()),exposed=int(exposed.sum()),
          mass=float(out.sum()),momentum=new_p.tolist(),wall_impulse_on_solid=wall.tolist(),
          conversion_impulse_on_solid=conversion.tolist(),local_boundary_mass_repair=float(-delta.sum()),
          exposed_mass_transfer=transferred,paired_nodes=paired,maximum_transfer_radius=self.local_radius,
          mass_residual=mass_error,bounce_momentum_residual=float((bounce_p-old_p+wall).abs().max()),
          momentum_residual=float((new_p-old_p+wall+conversion).abs().max()),
          minimum_population=float(out[:,~new_solid].min()),global_mass_rescale=False)
        self.center=new_center;self.solid=new_solid;self.f=out;self.time+=1;self.history.append(entry)
        return entry

    def snapshot(self):
        data=super().snapshot();data['schema']='tensorlbm.balanced-local-moving-disk/1';data['local_radius']=self.local_radius;data['velocity']=self.velocity.tolist()
        return copy.deepcopy(data)

    @classmethod
    def restore(cls,data):
        if set(data)!={'schema','config','time','center','target_mass','f','history','local_radius','velocity'} or data['schema']!='tensorlbm.balanced-local-moving-disk/1':
            raise ValueError('local restart schema mismatch')
        base=copy.deepcopy(data);radius=base.pop('local_radius');velocity=base.pop('velocity');base['schema']='tensorlbm.moving-disk/1'
        restored=MovingDisk.restore(base)
        result=cls(local_radius=radius,**restored.config)
        result.__dict__.update(restored.__dict__)
        result.local_radius=radius
        result.velocity=torch.tensor(velocity,dtype=torch.float64)
        if result.velocity.shape!=(2,) or not torch.isfinite(result.velocity).all() or float(result.velocity.norm())>.1:
            raise ValueError("invalid restart current velocity")
        return result
