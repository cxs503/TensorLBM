"""Uniform-state D2Q9 finite-volume flux verification on exact moving geometry.

Only constant co-moving states are supported. This exercises actual shared
Cartesian and diffuse moving-wall fluxes, not an arbitrary-state BGK solver.
No population initialization at newborn cells; extensive state comes from flux.
"""
import copy
import math
import torch
from .swept_rectangle_geometry_2d import SweptRectangleGeometry
from .moving_boundary_2d import equilibrium
from .d2q9 import C

D=torch.float64

class UniformSweptTransport:
    def __init__(self,*,geometry=None,density_kg_m3=1000.,thickness_m=.2,kinetic_speed_m_s=1.):
        self.geometry=SweptRectangleGeometry(**(geometry or {}))
        for v in (density_kg_m3,thickness_m,kinetic_speed_m_s):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('positive finite material scales required')
        self.config={'geometry':self.geometry.config(),'density_kg_m3':float(density_kg_m3),'thickness_m':float(thickness_m),'kinetic_speed_m_s':float(kinetic_speed_m_s)}
        self.c=C.double()*kinetic_speed_m_s
        self.wall_velocity=torch.tensor(self.geometry.velocity_m_s,dtype=D)
        if float(self.wall_velocity.norm())>.1*kinetic_speed_m_s:raise ValueError('uniform trace speed outside low Mach verification range')
        self.unit=equilibrium(torch.ones((1,1),dtype=D),self.wall_velocity/kinetic_speed_m_s)[:,0,0]
        volume=self.geometry.at(0.)['fluid_volume_m2']*thickness_m
        self.Q=self.unit[:,None,None]*density_kg_m3*volume[None]
        self.time=0.;self.history=[]

    def step(self,dt_s):
        interval=self.geometry.interval(self.time,dt_s)
        v0=interval['start']['fluid_volume_m2']*self.config['thickness_m']
        v1=interval['end']['fluid_volume_m2']*self.config['thickness_m']
        trace=self.Q.sum((1,2))/v0.sum()
        expected=trace[:,None,None]*v0[None]
        if float((self.Q-expected).abs().max())>1e-11*max(float(self.Q.abs().max()),1.):
            raise ValueError('only uniform co-moving states supported; nonuniform state must not be reset')
        rho=float(trace.sum())
        if float((trace-rho*self.unit).abs().max())>1e-11*rho:raise ValueError('trace is not the co-moving equilibrium')
        thickness=self.config['thickness_m']
        ix=interval['vertical_open_integral_m_s']*thickness
        iy=interval['horizontal_open_integral_m_s']*thickness
        # One shared Cartesian flux, used with opposite signs by adjacent cells.
        fx=trace[:,None,None]*self.c[:,0,None,None]*ix[None]
        fy=trace[:,None,None]*self.c[:,1,None,None]*iy[None]
        cart=fx[:,:,1:]-fx[:,:,:-1]+fy[:,1:,:]-fy[:,:-1,:]
        wall=torch.zeros_like(self.Q)
        for facet in interval['wall_facets']:
            y,x=facet['cell_yx'];n=torch.tensor(facet['normal_xy'],dtype=D)
            relative=(self.c-self.wall_velocity)@n
            outgoing=relative>0;incoming=relative<0
            rate_out=(relative[outgoing]*trace[outgoing]).sum()
            capacity=(-relative[incoming]*self.unit[incoming]).sum()
            if float(capacity)<=0:raise ValueError('moving wall has no incoming kinetic capacity')
            wall_density=rate_out/capacity
            fwall=torch.where(outgoing,trace,wall_density*self.unit)
            wall[:,y,x]+=relative*fwall*facet['integrated_length_m_s']*thickness
        candidate=self.Q-cart-wall
        # Only exact zero-volume cells may have cancellation-level residuals.
        # The removed roundoff mass/momentum are explicitly included in the ledger.
        dead=v1==0;roundoff=candidate[:,dead].clone()
        if roundoff.numel() and float(roundoff.abs().max())>1e-12*max(float(self.Q.abs().max()),1.):
            raise ValueError('finite mass stranded in zero-volume cell')
        candidate[:,dead]=0
        if not torch.isfinite(candidate).all() or bool((candidate<0).any()):raise ValueError('negative transported population')
        before=self.Q.sum((1,2));after=candidate.sum((1,2))
        boundary=cart.sum((1,2));body=wall.sum((1,2));cleanup=roundoff.sum(1) if roundoff.numel() else torch.zeros(9,dtype=D)
        residual=after-before+boundary+body+cleanup
        density=candidate.sum(0)[v1>0]/v1[v1>0]
        momentum=torch.einsum('iyx,ia->ayx',candidate,self.c)
        velocity=momentum[:,v1>0]/candidate.sum(0)[v1>0]
        compression=float((v1[v1>0]*(density*torch.log(density/rho)-density+rho)).sum())*self.config['kinetic_speed_m_s']**2/3
        entry={'time_start_s':self.time,'time_end_s':self.time+float(dt_s),
               'birth_cells':int(((v0==0)&(v1>0)).sum()),'death_cells':int(((v0>0)&(v1==0)).sum()),
               'mass_kg':float(after.sum()),'population_balance_residual_kg':float(residual.abs().max()),
               'momentum_balance_residual_Ns':float((residual@self.c).abs().max()),
               'body_impulse_Ns':(body@self.c).tolist(),'boundary_outgoing_impulse_Ns':(boundary@self.c).tolist(),
               'roundoff_cleanup_mass_kg':float(cleanup.sum()),'roundoff_cleanup_impulse_Ns':(cleanup@self.c).tolist(),
               'maximum_density_error_kg_m3':float((density-self.config['density_kg_m3']).abs().max()),
               'maximum_velocity_error_m_s':float((velocity-self.wall_velocity[:,None]).abs().max()),
               'density_free_energy_J':compression,'maximum_population_error_kg':float((candidate-rho*self.unit[:,None,None]*v1[None]).abs().max()),
               'gcl_residual_m2':float(interval['gcl_residual_m2'].abs().max()),
               'physical_accuracy_qualified':False,'arbitrary_state_transport':False}
        if entry['maximum_density_error_kg_m3']>1e-11*self.config['density_kg_m3'] or entry['maximum_velocity_error_m_s']>1e-11*self.config['kinetic_speed_m_s'] or entry['gcl_residual_m2']>1e-11*self.geometry.dx_m**2:
            raise ValueError('uniform transport or geometric conservation gate failed')
        self.Q=candidate;self.time=entry['time_end_s'];self.history.append(entry)
        return entry

    def snapshot(self):
        return copy.deepcopy({'schema':'tensorlbm.uniform-swept-transport/1','config':self.config,'time_s':self.time,'Q_kg':self.Q.tolist(),'history':self.history})

    @classmethod
    def restore(cls,data):
        if not isinstance(data,dict) or set(data)!={'schema','config','time_s','Q_kg','history'} or data['schema']!='tensorlbm.uniform-swept-transport/1':raise ValueError('invalid uniform transport checkpoint')
        obj=cls(**data['config']);obj.time=float(data['time_s']);obj.Q=torch.tensor(data['Q_kg'],dtype=D)
        if not math.isfinite(obj.time) or obj.time<0 or obj.Q.shape!=(9,obj.geometry.ny,obj.geometry.nx) or not torch.isfinite(obj.Q).all() or bool((obj.Q<0).any()):raise ValueError('invalid checkpoint populations or clock')
        volume=obj.geometry.at(obj.time)['fluid_volume_m2']*obj.config['thickness_m']
        expected=obj.unit[:,None,None]*obj.config['density_kg_m3']*volume[None]
        if float((obj.Q-expected).abs().max())>1e-11*max(float(expected.abs().max()),1.):raise ValueError('checkpoint does not preserve uniform state')
        if data['history'] and abs(data['history'][-1]['time_end_s']-obj.time)>1e-12:raise ValueError('checkpoint history clock mismatch')
        if not data['history'] and obj.time!=0:raise ValueError('checkpoint missing history')
        clock=0.
        for row in data['history']:
            if row['physical_accuracy_qualified'] is not False or row['arbitrary_state_transport'] is not False or not math.isfinite(row['time_start_s']) or not math.isfinite(row['time_end_s']) or abs(row['time_start_s']-clock)>1e-12 or row['time_end_s']<=clock:
                raise ValueError('invalid checkpoint history')
            clock=row['time_end_s']
        obj.history=copy.deepcopy(data['history']);return obj
