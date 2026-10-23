"""Static cut-cell finite-volume discrete-velocity BGK, CPU float64 SI.

Continuous kinetic relaxation time is independent of roll-LBM tau. Periodic
outer faces; stationary diffuse inner rectangle. Small cells merge into fixed
control volumes. Moving geometry is explicitly unsupported.
"""
import copy
import hashlib
import json
import math
import numbers
import torch
from .d2q9 import C
from .swept_rectangle_geometry_2d import SweptRectangleGeometry

D=torch.float64
W=torch.tensor([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],dtype=D)

def finite_real(value,name,positive=True):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or (positive and value<=0):
        raise ValueError(f'invalid {name}')
    return float(value)

def equilibrium_density(rho,velocity,c):
    speed=float(c[1,0]);cu=velocity@c.T/speed**2;u2=velocity.square().sum(-1)/speed**2
    return rho[:,None]*W[None]*(1+3*cu+4.5*cu.square()-1.5*u2[:,None])

class CutCellBGK2D:
    def __init__(self,*,geometry=None,solid_rectangle=True,kinetic_speed_m_s=1.,relaxation_time_s=.2,
                 thickness_m=.2,merge_fraction=.2,max_cfl=.8,density_reference_kg_m3=1000.,
                 density_kg_m3=None,velocity_m_s=None,population_density_kg_m3=None):
        if type(solid_rectangle) is not bool:raise ValueError('solid_rectangle must be bool')
        geometry=dict(geometry or {});geometry.setdefault('velocity_m_s',(0.,0.))
        self.geometry=SweptRectangleGeometry(**geometry)
        if any(v!=0 for v in self.geometry.velocity_m_s):raise ValueError('moving cut-cell transport is not implemented')
        for name,value in [('kinetic_speed_m_s',kinetic_speed_m_s),('relaxation_time_s',relaxation_time_s),('thickness_m',thickness_m),('density_reference_kg_m3',density_reference_kg_m3)]:
            finite_real(value,name)
        merge_fraction=finite_real(merge_fraction,'merge_fraction');max_cfl=finite_real(max_cfl,'max_cfl')
        if not 0<merge_fraction<1 or not 0<max_cfl<=1:raise ValueError('merge fraction and CFL outside bounds')
        self.config=dict(geometry=self.geometry.config(),solid_rectangle=solid_rectangle,
            kinetic_speed_m_s=float(kinetic_speed_m_s),relaxation_time_s=float(relaxation_time_s),thickness_m=float(thickness_m),
            merge_fraction=merge_fraction,max_cfl=max_cfl,density_reference_kg_m3=float(density_reference_kg_m3))
        self.c=C.to(dtype=D)*kinetic_speed_m_s;self.time=0.;self.history=[]
        self._build_geometry()
        ny,nx=self.geometry.ny,self.geometry.nx
        if population_density_kg_m3 is None:
            density=density_reference_kg_m3 if density_kg_m3 is None else density_kg_m3
            rho=torch.as_tensor(density,dtype=D).clone()
            if rho.ndim==0:rho=rho.expand(ny,nx).clone()
            velocity=torch.zeros(ny,nx,2,dtype=D) if velocity_m_s is None else torch.as_tensor(velocity_m_s,dtype=D).clone()
            if velocity.shape==(2,):velocity=velocity.expand(ny,nx,2).clone()
            if rho.shape!=(ny,nx) or velocity.shape!=(ny,nx,2) or not torch.isfinite(rho).all() or not torch.isfinite(velocity).all() or bool((rho<=0).any()) or float(velocity.norm(dim=-1).max())>.1*kinetic_speed_m_s:
                raise ValueError('invalid initial density/velocity')
            density_pop=equilibrium_density(rho.reshape(-1),velocity.reshape(-1,2),self.c).T.reshape(9,ny,nx)
        else:
            density_pop=torch.as_tensor(population_density_kg_m3,dtype=D).clone()
            if density_pop.shape!=(9,ny,nx) or not torch.isfinite(density_pop).all() or bool((density_pop<0).any()):raise ValueError('invalid initial population density')
        if bool((density_pop.sum(0)[self.group_map>=0]<=0).any()):raise ValueError('initial fluid cell has zero density')
        raw=density_pop*self.volume_m3[None]
        self.initial_cell_Q_kg=raw.clone()
        self.group_Q=torch.zeros(self.group_count,9,dtype=D)
        mask=self.group_map>=0
        self.group_Q.index_add_(0,self.group_map[mask],raw[:,mask].T)
        self._validate_state(self.group_Q)
        self.initial_projection=dict(momentum_before_Ns=torch.einsum('iyx,ia->a',raw,self.c).tolist(),
            mass_before_kg=float(raw.sum()),maximum_cell_population_change_kg=float((self.Q-raw).abs().max()),
            momentum_after_Ns=(self.group_Q.sum(0)@self.c).tolist(),mass_after_kg=float(self.group_Q.sum()),
            energy_before=self._cell_energy(raw),energy_after=self._energy(self.group_Q))
        self.last_step_raw=None

    def _build_geometry(self):
        g=self.geometry;ny,nx=g.ny,g.nx;dx=g.dx_m
        if self.config['solid_rectangle']:
            static=g.at();volume=static['fluid_volume_m2'];vertical=static['vertical_open_m'];horizontal=static['horizontal_open_m'];facets=static['wall_facets']
        else:
            edges_x=torch.arange(nx+1,dtype=D)*dx;edges_y=torch.arange(ny+1,dtype=D)*dx
            wx=edges_x.diff();wy=edges_y.diff();volume=wy[:,None]*wx[None]
            vertical=wy[:,None].expand(ny,nx+1).clone();horizontal=wx[None].expand(ny+1,nx).clone();facets=[]
        self.volume_m3=volume*self.config['thickness_m'];active=volume>0
        parent=list(range(ny*nx))
        def find(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[max(a,b)]=min(a,b)
        candidates=sorted(torch.nonzero(active&(volume<self.config['merge_fraction']*dx*dx)).tolist(),key=lambda p:(float(volume[p[0],p[1]]),p))
        for y,x in candidates:
            choices=[]
            for yy,xx,aperture in [(y,(x-1)%nx,vertical[y,x]),(y,(x+1)%nx,vertical[y,x+1]),((y-1)%ny,x,horizontal[y,x]),((y+1)%ny,x,horizontal[y+1,x])]:
                if active[yy,xx] and aperture>0:choices.append((float(aperture),float(volume[yy,xx]),-(yy*nx+xx),yy,xx))
            if not choices:raise ValueError('small fluid cell has no open-face merge neighbor')
            _,_,_,yy,xx=max(choices);union(y*nx+x,yy*nx+xx)
        roots=sorted({find(y*nx+x) for y,x in torch.nonzero(active).tolist()});labels={r:k for k,r in enumerate(roots)}
        self.group_count=len(roots);self.group_map=torch.full((ny,nx),-1,dtype=torch.int64)
        for y,x in torch.nonzero(active).tolist():self.group_map[y,x]=labels[find(y*nx+x)]
        self.group_volume_m3=torch.zeros(self.group_count,dtype=D)
        self.group_volume_m3.index_add_(0,self.group_map[active],self.volume_m3[active])
        faces=[]
        for y in range(ny):
            for x in range(nx):
                left=int(self.group_map[y,(x-1)%nx]);right=int(self.group_map[y,x]);aperture=float(vertical[y,x])
                if aperture>0:
                    if min(left,right)<0:raise ValueError('open Cartesian face touches a zero-volume cell')
                    if left!=right:faces.append(dict(group_left=left,group_right=right,axis=0,area_m2=aperture*self.config['thickness_m'],face_yx=[y,x]))
        for y in range(ny):
            for x in range(nx):
                bottom=int(self.group_map[(y-1)%ny,x]);top=int(self.group_map[y,x]);aperture=float(horizontal[y,x])
                if aperture>0:
                    if min(bottom,top)<0:raise ValueError('open Cartesian face touches a zero-volume cell')
                    if bottom!=top:faces.append(dict(group_left=bottom,group_right=top,axis=1,area_m2=aperture*self.config['thickness_m'],face_yx=[y,x]))
        self.faces=faces;self.wall_facets=[]
        for f in facets:
            y,x=f['cell_yx'];group=int(self.group_map[y,x])
            if group<0:raise ValueError('wall facet not owned by fluid')
            self.wall_facets.append(dict(**f,group=group,area_m2=f['length_m']*self.config['thickness_m']))
        self.face_left=torch.tensor([f['group_left'] for f in faces],dtype=torch.int64)
        self.face_right=torch.tensor([f['group_right'] for f in faces],dtype=torch.int64)
        axes=torch.tensor([f['axis'] for f in faces],dtype=torch.int64)
        areas=torch.tensor([f['area_m2'] for f in faces],dtype=D)
        self.face_rates=self.c[:,axes].T*areas[:,None] if len(faces) else torch.zeros(0,9,dtype=D)
        self.wall_groups=torch.tensor([f['group'] for f in self.wall_facets],dtype=torch.int64)
        normals=torch.tensor([f['normal_xy'] for f in self.wall_facets],dtype=D).reshape(-1,2)
        self.wall_relative=normals@self.c.T
        self.wall_area=torch.tensor([f['area_m2'] for f in self.wall_facets],dtype=D)
        loss=torch.zeros(self.group_count,9,dtype=D)
        loss.index_add_(0,self.face_left,self.face_rates.clamp_min(0));loss.index_add_(0,self.face_right,(-self.face_rates).clamp_min(0))
        loss.index_add_(0,self.wall_groups,self.wall_relative.clamp_min(0)*self.wall_area[:,None])
        self.cfl_rate=loss/self.group_volume_m3[:,None]
        self.cfl_limit_s=self.config['max_cfl']/float(self.cfl_rate.max())
        self.geometry_data=dict(fluid_volume_m2=volume.tolist(),vertical_open_m=vertical.tolist(),horizontal_open_m=horizontal.tolist(),
            wall_facets=copy.deepcopy(self.wall_facets),faces=copy.deepcopy(faces),group_map=self.group_map.tolist(),group_volume_m3=self.group_volume_m3.tolist(),
            periodic_outer_boundary=True,stationary_geometry=True)
        self.geometry_sha256=hashlib.sha256(json.dumps(self.geometry_data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @property
    def Q(self):
        field=torch.zeros(9,self.geometry.ny,self.geometry.nx,dtype=D);mask=self.group_map>=0
        density=self.group_Q/self.group_volume_m3[:,None]
        field[:,mask]=(density[self.group_map[mask]]*self.volume_m3[mask,None]).T
        return field

    def _validate_state(self,Q):
        if Q.shape!=(self.group_count,9) or not torch.isfinite(Q).all() or bool((Q<0).any()):raise ValueError('invalid group populations')
        mass=Q.sum(1)
        if bool((mass<=0).any()):raise ValueError('zero-mass fluid group')
        velocity=Q@self.c/mass[:,None]
        if float(velocity.norm(dim=1).max())>.1*self.config['kinetic_speed_m_s']:raise ValueError('low Mach admissibility exceeded')

    def collision(self,Q,half_dt_s):
        half_dt_s=finite_real(half_dt_s,'collision duration',positive=False)
        if half_dt_s<0:raise ValueError('negative collision duration')
        self._validate_state(Q);rho=Q.sum(1)/self.group_volume_m3;velocity=Q@self.c/Q.sum(1)[:,None]
        eq=equilibrium_density(rho,velocity,self.c)*self.group_volume_m3[:,None]
        if bool((eq<0).any()):raise ValueError('negative polynomial equilibrium')
        weight=-math.expm1(-half_dt_s/self.config['relaxation_time_s'])
        return Q*(1-weight)+eq*weight

    def _cell_energy(self,Q):
        mask=self.group_map>=0;mass=Q.sum(0)[mask];moment=torch.einsum('iyx,ia->yxa',Q,self.c)[mask];volume=self.volume_m3[mask]
        rho=mass/volume;ref=self.config['density_reference_kg_m3'];cs2=self.config['kinetic_speed_m_s']**2/3
        kinetic=float((.5*moment.square().sum(1)/mass).sum());compression=float((volume*(rho*torch.log(rho/ref)-rho+ref)).sum())*cs2
        return dict(kinetic_J=kinetic,compression_J=compression,mechanical_J=kinetic+compression)

    def _energy(self,Q):
        mass=Q.sum(1);rho=mass/self.group_volume_m3;moment=Q@self.c;ref=self.config['density_reference_kg_m3'];cs2=self.config['kinetic_speed_m_s']**2/3
        kinetic=float((.5*moment.square().sum(1)/mass).sum())
        compression=float((self.group_volume_m3*(rho*torch.log(rho/ref)-rho+ref)).sum())*cs2
        positive=Q>0;ratio=Q/(self.group_volume_m3[:,None]*W[None]*ref)
        entropy=float((Q[positive]*torch.log(ratio[positive])).sum())*cs2
        return dict(kinetic_J=kinetic,compression_J=compression,mechanical_J=kinetic+compression,population_entropy_J=entropy)

    def step(self,dt_s):
        dt=finite_real(dt_s,'dt_s')
        if dt>self.cfl_limit_s*(1+1e-13):raise ValueError('transport CFL exceeded')
        before=self.group_Q.clone();self._validate_state(before)
        half=self.collision(before,dt/2);density=half/self.group_volume_m3[:,None]
        upwind=torch.where(self.face_rates>=0,density[self.face_left],density[self.face_right])
        face_flux=dt*self.face_rates*upwind;delta=torch.zeros_like(before)
        delta.index_add_(0,self.face_left,face_flux);delta.index_add_(0,self.face_right,-face_flux)
        if len(self.wall_facets):
            outgoing=self.wall_relative>0;incoming=self.wall_relative<0
            rate_out=(self.wall_relative.clamp_min(0)*density[self.wall_groups]).sum(1)
            capacity=((-self.wall_relative).clamp_min(0)*W[None]).sum(1)
            if bool((capacity<=0).any()):raise ValueError('zero diffuse wall capacity')
            wall_density=rate_out/capacity
            fwall=torch.where(outgoing,density[self.wall_groups],wall_density[:,None]*W[None])
            wall_flux=dt*self.wall_relative*fwall*self.wall_area[:,None]
            delta.index_add_(0,self.wall_groups,wall_flux)
        else:wall_flux=torch.zeros(0,9,dtype=D)
        transported=half-delta;self._validate_state(transported)
        after=self.collision(transported,dt/2);self._validate_state(after)
        body=wall_flux.sum(0);coll=(half-before).sum(0)+(after-transported).sum(0)
        total_residual=after.sum(0)-before.sum(0)+body-coll
        mass_residual=float(total_residual.sum());moment_residual=total_residual@self.c
        scale=max(float(before.sum()),1.)
        if abs(mass_residual)>1e-11*scale or float(moment_residual.abs().max())>1e-11*scale*self.config['kinetic_speed_m_s']:raise ValueError('conservation gate failed')
        e0=self._energy(before);eh=self._energy(half);et=self._energy(transported);e1=self._energy(after)
        collision_error=torch.cat((half.sum(1)-before.sum(1),after.sum(1)-transported.sum(1)))
        collision_moment_error=torch.cat(((half-before)@self.c,(after-transported)@self.c))
        entry=dict(time_start_s=self.time,time_end_s=self.time+dt,dt_s=dt,cfl=float(self.cfl_rate.max())*dt,cfl_limit_s=self.cfl_limit_s,
            mass_kg=float(after.sum()),momentum_Ns=(after.sum(0)@self.c).tolist(),body_impulse_Ns=(body@self.c).tolist(),wall_mass_flux_kg=float(body.sum()),
            collision_mass_change_kg=float(coll.sum()),collision_impulse_Ns=(coll@self.c).tolist(),mass_balance_residual_kg=mass_residual,
            momentum_balance_residual_Ns=moment_residual.tolist(),max_collision_group_mass_error_kg=float(collision_error.abs().max()),
            max_collision_group_momentum_error_Ns=float(collision_moment_error.abs().max()),minimum_population_density_kg_m3=float((after/self.group_volume_m3[:,None]).min()),
            maximum_speed_m_s=float((after@self.c/after.sum(1)[:,None]).norm(dim=1).max()),
            energy_before=e0,energy_half_collision=eh,energy_after_transport=et,energy_after=e1,wall_work_J=0.,
            physical_accuracy_qualified=False,arbitrary_state_transport=True,moving_geometry_implemented=False)
        # All validation occurs before advancing any state or accounting.
        self.group_Q=after;self.time=entry['time_end_s'];self.history.append(entry)
        self.last_step_raw=dict(half_collision_group_Q_kg=half.tolist(),transport_group_Q_kg=transported.tolist(),
            face_flux_kg=face_flux.tolist(),wall_flux_kg=wall_flux.tolist())
        return copy.deepcopy(entry)

    def snapshot(self):
        return copy.deepcopy(dict(schema='tensorlbm.static-cut-cell-bgk/1',config=self.config,time_s=self.time,group_Q_kg=self.group_Q.tolist(),
            geometry_sha256=self.geometry_sha256,history=self.history,initial_projection=self.initial_projection))

    @classmethod
    def restore(cls,data):
        if not isinstance(data,dict) or set(data)!={'schema','config','time_s','group_Q_kg','geometry_sha256','history','initial_projection'} or data['schema']!='tensorlbm.static-cut-cell-bgk/1':raise ValueError('invalid checkpoint schema')
        obj=cls(**data['config']);time=finite_real(data['time_s'],'checkpoint time',positive=False)
        if time<0 or data['geometry_sha256']!=obj.geometry_sha256:raise ValueError('checkpoint geometry or clock mismatch')
        Q=torch.tensor(data['group_Q_kg'],dtype=D);obj._validate_state(Q)
        if not isinstance(data['history'],list):raise ValueError('invalid checkpoint history')
        clock=0.
        for row in data['history']:
            if row['physical_accuracy_qualified'] is not False or row['arbitrary_state_transport'] is not True or row['moving_geometry_implemented'] is not False:
                raise ValueError('checkpoint qualification mismatch')
            start=finite_real(row['time_start_s'],'history start',False);end=finite_real(row['time_end_s'],'history end',False)
            if abs(start-clock)>1e-12 or end<=start:raise ValueError('checkpoint history clock mismatch')
            clock=end
        if abs(clock-time)>1e-12:raise ValueError('checkpoint clock mismatch')
        if data['history']:
            last=data['history'][-1];scale=max(float(Q.sum()),1.)
            if abs(finite_real(last['mass_kg'],'history mass')-float(Q.sum()))>1e-11*scale or float((torch.tensor(last['momentum_Ns'],dtype=D)-Q.sum(0)@obj.c).abs().max())>1e-11*scale*obj.config['kinetic_speed_m_s']:
                raise ValueError('checkpoint state/ledger mismatch')
        obj.group_Q=Q;obj.time=time;obj.history=copy.deepcopy(data['history']);obj.initial_projection=copy.deepcopy(data['initial_projection'])
        return obj
