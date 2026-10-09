"""Executable P1: regularized, slipping IBM / real bonded-disk DEM.

Periodic single-phase liquid; no gravity, free surface, impermeability or ice
material calibration. Bilinear interpolation and its transpose spread SI loads.
A finite drag coefficient regularizes the exchange; it is NOT a no-slip solve.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from .d2q9 import C, macroscopic
from .solver import collide_bgk_matmul, stream
from .icebreaking import (IcebreakingForceLedger, WrenchSI,
    integrate_particle_tractions_2d, integrate_surface_traction)

@dataclass(frozen=True)
class CoupledIceConfig:
    nx: int = 48
    ny: int = 40
    dx_m: float = .025
    fluid_dt_s: float = .0005
    duration_s: float = .15
    density_kg_m3: float = 1000.
    viscosity_m2_s: float = .02
    exchange_steps: int = 1
    coupling_rate_s: float = 30.
    wet: bool = True
    ice_nx: int = 9
    ice_ny: int = 2
    ice_radius_m: float = .025
    tool_speed_m_s: float = .2
    tool_gap_m: float = .001
    breaking_strain: float = .015
    shear_breaking_strain: float = .03

    def __post_init__(self):
        for k in ('nx','ny','exchange_steps','ice_nx','ice_ny'):
            v=getattr(self,k)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError(k+' must be a positive integer')
        for k,v in asdict(self).items():
            if isinstance(v,float) and (not math.isfinite(v) or v<=0):
                raise ValueError(k+' must be positive and finite')
        if self.nx<24 or self.ny<24 or self.ice_nx<3:
            raise ValueError('domain/ice too small')
        if self.duration_s/self.fluid_dt_s<1:
            raise ValueError('duration must include a fluid step')
        if self.tau<=.5 or self.tau>2:
            raise ValueError('BGK tau must be in (.5,2]')

    @property
    def tau(self):
        return .5+3*self.viscosity_m2_s*self.fluid_dt_s/self.dx_m**2


def bilinear_map(positions_m, shape, dx):
    """Dense small-case J; reject wrap support to retain physical moment."""
    ny,nx=shape
    p=positions_m/dx
    base=torch.floor(p).long()
    if bool((base<0).any() or (base[:,0]>=nx-1).any() or (base[:,1]>=ny-1).any()):
        raise ValueError('marker crossed non-wrapping coupling support')
    frac=p-base
    J=torch.zeros((len(p),nx*ny),dtype=torch.float64)
    for i,j in ((0,0),(1,0),(0,1),(1,1)):
        w=(frac[:,0] if i else 1-frac[:,0])*(frac[:,1] if j else 1-frac[:,1])
        J[torch.arange(len(p)),(base[:,1]+j)*nx+base[:,0]+i]=w
    return J


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]

class CoupledIce2D:
    def __init__(self, config=CoupledIceConfig()):
        from tensordem import IceDEM, DEMConfig
        self.config=config
        c=config
        demc=DEMConfig(nx=c.ice_nx,ny=c.ice_ny,radius=c.ice_radius_m,thickness=.2,
            bond_stiffness=500.,contact_stiffness=500.,contact_damping=1.,
            drag=0.,tool_speed=c.tool_speed_m_s,tool_gap=c.tool_gap_m,
            breaking_strain=c.breaking_strain,shear_breaking_strain=c.shear_breaking_strain)
        self.substeps=math.ceil(c.fluid_dt_s/demc.recommended_dt)
        demc=DEMConfig(**{**asdict(demc),'dt':c.fluid_dt_s/self.substeps})
        self.dem=IceDEM(demc)
        shift=torch.tensor([c.nx*c.dx_m/2,c.ny*c.dx_m/2],dtype=torch.float64)
        self.origin_offset=shift
        # Exact float64 D2Q9 weights: existing equilibrium defaults to float32 W.
        self.weights=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
        self.f=self.weights[:,None,None].expand(9,c.ny,c.nx).clone()
        self.step_index=0
        self.history=[]
        self.exchange_sample_time=0.
        self.map=None
        self.marker_positions=None
        self.solid_velocity=None
        self.marker_force=None
        self.ice_load=None
        self.fluid_grid_force=None
        self.interface_impulse=torch.zeros(2,dtype=torch.float64)
        self.external_impulse=torch.zeros(2,dtype=torch.float64)
        self.total_initial_momentum=self.total_momentum().clone()
        self.initial_mass=float(self.f.sum())
        self.initial_dem_energy=self.dem.mechanical_energy()['mechanical_J']
        self.initial_fluid_energy=self.fluid_energy()
        self.fluid_collision_loss=0.
        self.fluid_force_work=0.
        self.interface_work=0.
        self.interface_marker_work=0.
        self.regularization_dissipation=0.
        self.tool_fluid_work=0.
        self.grid_xy=torch.stack(torch.meshgrid(
            torch.arange(c.ny,dtype=torch.float64)*c.dx_m,
            torch.arange(c.nx,dtype=torch.float64)*c.dx_m,indexing='ij'),-1).flip(-1).reshape(-1,2)-self.origin_offset

    def fluid_velocity(self):
        _,ux,uy=macroscopic(self.f)
        return torch.stack((ux,uy),-1).reshape(-1,2)*self.config.dx_m/self.config.fluid_dt_s

    def fluid_energy(self):
        rho=self.f.sum(0).reshape(-1)
        return float(.5*self.cell_mass*(rho[:,None]*self.fluid_velocity()**2).sum())

    @property
    def cell_mass(self):
        return self.config.density_kg_m3*self.config.dx_m**2*self.dem.config.thickness

    def total_momentum(self):
        if not self.config.wet:
            return self.dem.config.mass*self.dem.velocities.sum(0)
        rho=self.f.sum(0).reshape(-1)
        return self.cell_mass*(rho[:,None]*self.fluid_velocity()).sum(0)+self.dem.config.mass*self.dem.velocities.sum(0)

    def refresh_exchange(self):
        c=self.config
        self.exchange_sample_time=self.dem.time
        angle=torch.arange(24,dtype=torch.float64)*(2*math.pi/24)
        tool=self.dem.tool_position+ self.dem.config.tool_radius*torch.stack((angle.cos(),angle.sin()),1)
        self.marker_positions=torch.cat((self.dem.positions,tool))
        self.map=bilinear_map(self.marker_positions+self.origin_offset,(c.ny,c.nx),c.dx_m)
        self.solid_velocity=torch.cat((self.dem.velocities,self.dem.tool_velocity.expand(24,2)))
        interpolated=self.map@self.fluid_velocity()
        # Distributed point friction, not boundary traction from a resolved skin.
        zeta=torch.cat((torch.full((len(self.dem.positions),),self.dem.config.mass*c.coupling_rate_s),
                       torch.full((24,),.1*c.coupling_rate_s/24))).double()
        self.marker_force=zeta[:,None]*(self.solid_velocity-interpolated)
        if not c.wet:
            self.marker_force.zero_()
        self.fluid_grid_force=self.map.T@self.marker_force
        n=len(self.dem.positions)
        area=2*self.dem.config.radius*self.dem.config.thickness
        traction=-self.marker_force[:n]/area
        self.ice_load=torch.tensor(integrate_particle_tractions_2d(traction.tolist(),[area]*n),dtype=torch.float64)

    @torch.no_grad()
    def step(self):
        c=self.config
        if self.step_index % c.exchange_steps==0:
            self.refresh_exchange()
        n=len(self.dem.positions)
        initial_u=self.fluid_velocity()
        interpolated=self.map@initial_u
        grid_power=float((self.fluid_grid_force*initial_u).sum())
        marker_power=float((self.marker_force*interpolated).sum())
        force_error=float(torch.linalg.vector_norm(self.fluid_grid_force.sum(0)-self.marker_force.sum(0)))
        moment_error=float(abs(cross2(self.grid_xy,self.fluid_grid_force).sum()-cross2(self.marker_positions,self.marker_force).sum()))
        # Track the actual velocity-level staggered interface mismatch separately.
        self.interface_marker_work+=c.fluid_dt_s*float((self.marker_force*self.solid_velocity).sum())
        self.interface_work+=c.fluid_dt_s*grid_power
        self.regularization_dissipation+=c.fluid_dt_s*float((self.marker_force*(self.solid_velocity-interpolated)).sum())
        self.tool_fluid_work+=c.fluid_dt_s*float((self.marker_force[n:]*self.solid_velocity[n:]).sum())
        before=self.fluid_energy()
        if c.wet:
            lattice_impulse=self.fluid_grid_force.reshape(c.ny,c.nx,2)*c.fluid_dt_s**2/(self.cell_mass*c.dx_m)
            direction=C.double()
            delta=3*self.weights[:,None,None]*(direction[:,0,None,None]*lattice_impulse[:,:,0]+direction[:,1,None,None]*lattice_impulse[:,:,1])
            self.f+=delta
            after_force=self.fluid_energy()
            self.fluid_force_work+=after_force-before
            self.f=stream(collide_bgk_matmul(self.f,c.tau))
            self.fluid_collision_loss+=after_force-self.fluid_energy()
        demwork0=self.dem.accounting['external_work_J']
        reactions=[]
        supports=[]
        for _ in range(self.substeps):
            self.dem.step(external_forces=self.ice_load,load_time=self.dem.time)
            reactions.append(torch.tensor(self.dem.last_step['tool_reaction_N'],dtype=torch.float64))
            supports.append(torch.tensor(self.dem.last_step['support_force_N'],dtype=torch.float64))
        reaction=torch.stack(reactions).double().mean(0)
        support=torch.stack(supports).double().mean(0)
        toolfluid=-self.marker_force[n:].sum(0)
        self.external_impulse+=c.fluid_dt_s*(support-reaction-toolfluid)
        self.interface_impulse+=c.fluid_dt_s*self.ice_load.sum(0)
        self.step_index+=1
        tool_area=2*math.pi*self.dem.config.tool_radius*self.dem.config.thickness/24
        fluidwrench=integrate_surface_traction(
            [[float(x),float(y),0.] for x,y in self.marker_positions[n:]],
            [[float(x),float(y),0.] for x,y in -self.marker_force[n:]/tool_area],[tool_area]*24)
        ledger=IcebreakingForceLedger(time_s=self.dem.time,fluid_on_hull=fluidwrench,
            ice_contact_on_hull=WrenchSI(fx_n=float(reaction[0]),fy_n=float(reaction[1]),
                mz_nm=float(cross2(self.dem.tool_position,reaction))),forward_axis=(0.,-1.,0.))
        energy=self.dem.mechanical_energy()
        demres=energy['mechanical_J']+self.dem.accounting['damping_dissipation_J']+self.dem.accounting['fracture_release_J']-self.initial_dem_energy-self.dem.accounting['external_work_J']-self.dem.accounting['tool_work_J']
        residual=self.total_momentum()-self.total_initial_momentum-self.external_impulse
        self.history.append({**ledger.as_record(), 'exchange_sample_time_s':self.exchange_sample_time,
            'exchange_interval_end_s':self.exchange_sample_time+c.exchange_steps*c.fluid_dt_s,
            'force_hold_age_s':self.dem.time-c.fluid_dt_s-self.exchange_sample_time,
            'broken_bonds':self.dem.broken_bonds,
            'force_mapping_error_N':force_error,'moment_mapping_error_Nm':moment_error,
            'adjoint_power_error_W':abs(grid_power-marker_power),
            'staggered_interface_work_residual_J':self.fluid_force_work+self.dem.accounting['external_work_J']-self.tool_fluid_work+self.regularization_dissipation,
            'regularization_dissipation_J':self.regularization_dissipation,'prescribed_tool_fluid_work_J':self.tool_fluid_work,
            'marker_slip_rms_m_s':float(torch.sqrt(((self.solid_velocity-interpolated)**2).mean())),
            'momentum_residual_kg_m_s':residual.tolist(),
            'mass_relative_error':abs(float(self.f.sum())/self.initial_mass-1),
            'fluid_kinetic_J':self.fluid_energy(),'dem_mechanical_J':energy['mechanical_J'],
            'dem_discrete_energy_residual_J':demres,
            'fluid_collision_stream_kinetic_loss_J':self.fluid_collision_loss,
            'fluid_force_work_J':self.fluid_force_work,
            'ice_external_work_step_J':self.dem.accounting['external_work_J']-demwork0,
            'fracture_release_J':self.dem.accounting['fracture_release_J']})
        if not bool(torch.isfinite(self.f).all()) or float(self.f.min())<=0:
            raise FloatingPointError('LBM nonpositive/nonfinite populations')
        return self.history[-1]

    def snapshot(self):
        return {'schema':'tensorlbm.ice-coupling-restart/1','config':asdict(self.config),
            'dem':self.dem.snapshot(),'f':self.f.tolist(),'step_index':self.step_index,
            'history':self.history,'substeps':self.substeps,
            'scalars':{k:getattr(self,k) for k in ('initial_mass','initial_dem_energy','initial_fluid_energy','fluid_collision_loss','fluid_force_work','interface_work','interface_marker_work','regularization_dissipation','tool_fluid_work','exchange_sample_time')},
            'vectors':{k:getattr(self,k).tolist() for k in ('interface_impulse','external_impulse','total_initial_momentum')},
            'exchange':{k:None if getattr(self,k) is None else getattr(self,k).tolist() for k in ('map','marker_positions','solid_velocity','marker_force','ice_load','fluid_grid_force')}}

    @classmethod
    def from_snapshot(cls,record):
        import json
        from tensordem import IceDEM
        fields={'schema','config','dem','f','step_index','history','substeps','scalars','vectors','exchange'}
        if not isinstance(record,dict) or set(record)!=fields or record.get('schema')!='tensorlbm.ice-coupling-restart/1':
            raise ValueError('invalid coupled restart schema/fields')
        try:json.dumps(record,allow_nan=False)
        except (ValueError,TypeError) as exc:raise ValueError('nonfinite/invalid restart') from exc
        obj=cls(CoupledIceConfig(**record['config']))
        expected=obj.snapshot()
        for group in ('scalars','vectors','exchange'):
            if not isinstance(record[group],dict) or set(record[group])!=set(expected[group]):
                raise ValueError('invalid restart '+group+' fields')
        index=record['step_index']
        if isinstance(index,bool) or not isinstance(index,int) or index<0:
            raise ValueError('invalid coupled step index')
        if record['substeps']!=obj.substeps or isinstance(record['substeps'],bool):
            raise ValueError('inconsistent DEM subcycle count')
        obj.dem=IceDEM.from_snapshot(record['dem'])
        obj.f=torch.tensor(record['f'],dtype=torch.float64)
        if obj.f.shape!=(9,obj.config.ny,obj.config.nx) or not bool(torch.isfinite(obj.f).all()) or bool((obj.f<=0).any()):
            raise ValueError('invalid LBM restart populations')
        obj.step_index=index
        if not isinstance(record['history'],list) or len(record['history'])!=index:
            raise ValueError('invalid coupled history')
        obj.history=record['history']
        for k,v in record['scalars'].items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
                raise ValueError('invalid restart scalar')
            setattr(obj,k,v)
        for k,v in record['vectors'].items():
            tensor=torch.tensor(v,dtype=torch.float64)
            if tensor.shape!=(2,) or not bool(torch.isfinite(tensor).all()):
                raise ValueError('invalid restart vector')
            setattr(obj,k,tensor)
        n=len(obj.dem.positions);m=n+24;g=obj.config.nx*obj.config.ny
        shapes={'map':(m,g),'marker_positions':(m,2),'solid_velocity':(m,2),
                'marker_force':(m,2),'ice_load':(n,2),'fluid_grid_force':(g,2)}
        for k,v in record['exchange'].items():
            if v is None:
                if index!=0:raise ValueError('missing active exchange state')
                setattr(obj,k,None);continue
            tensor=torch.tensor(v,dtype=torch.float64)
            if tensor.shape!=shapes[k] or not bool(torch.isfinite(tensor).all()):
                raise ValueError('invalid restart exchange '+k)
            setattr(obj,k,tensor)
        if abs(obj.dem.time-index*obj.config.fluid_dt_s)>1e-10:
            raise ValueError('coupled clocks differ')
        return obj
