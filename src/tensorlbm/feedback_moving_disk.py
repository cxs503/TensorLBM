"""SI live motion adapter over the experimental MovingDisk BGK solver.

The global mass reservoir remains external; this is a coupling plumbing fixture,
not a locally conservative or calibrated moving boundary.
"""
import math
import torch
from .moving_boundary_2d import MovingDisk, momentum


def _scalar(x, name, positive=False):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or (positive and x <= 0):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else ''))
    return float(x)


def _vector(x, name):
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(name + ' must be a two component SI vector')
    return [_scalar(a, name) for a in x]


class MovingDiskFeedback:
    """CPU float64 translating disk with current-state, explicit SI exchanges."""
    def __init__(self, *, dx_m=.01, dt_s=.001, density_kg_m3=1000., thickness_m=.1,
                 nx=40, ny=32, radius_m=.04, center_m=(.162,.161), tau=.8):
        self.config = dict(dx_m=_scalar(dx_m,'dx',True), dt_s=_scalar(dt_s,'dt',True),
            density_kg_m3=_scalar(density_kg_m3,'density',True), thickness_m=_scalar(thickness_m,'thickness',True),
            nx=nx, ny=ny, radius_m=_scalar(radius_m,'radius',True), center_m=_vector(center_m,'center'), tau=tau)
        self.solver = MovingDisk(nx=nx, ny=ny, radius=radius_m/dx_m,
            center=[v/dx_m for v in center_m], tau=tau)
        self.history = []

    @property
    def time_s(self):
        return self.solver.time * self.config['dt_s']

    @property
    def center_m(self):
        return (self.solver.center * self.config['dx_m']).tolist()

    @property
    def impulse_scale(self):
        c=self.config
        return c['density_kg_m3']*c['thickness_m']*c['dx_m']**3/c['dt_s']

    def fluid_momentum_Ns(self):
        return (momentum(self.solver.f)*self.impulse_scale).tolist()

    def advance(self, *, time_s, center_m, velocity_m_s):
        clock=_scalar(time_s,'time'); position=_vector(center_m,'center'); velocity=_vector(velocity_m_s,'velocity')
        if abs(clock-self.time_s)>1e-12 or max(abs(a-b) for a,b in zip(position,self.center_m))>1e-12:
            raise ValueError('stale clock or inconsistent current body position')
        lattice=torch.tensor(velocity,dtype=torch.float64)*self.config['dt_s']/self.config['dx_m']
        if float(lattice.norm())>.1:
            raise ValueError('lattice speed exceeds experimental limit')
        self.solver.velocity=lattice
        entry=self.solver.step(); scale=self.impulse_scale
        wall=[v*scale for v in entry['wall_impulse_on_solid']]
        conversion=[v*scale for v in entry['conversion_impulse_on_solid']]
        record=dict(schema='tensor-solver.live-rigid-exchange/1', time_start_s=clock,time_end_s=self.time_s,
            center_start_m=position,center_end_m=self.center_m,velocity_held_m_s=velocity,
            impulse_on_body_Ns=[a+b for a,b in zip(wall,conversion)],wall_impulse_on_body_Ns=wall,
            conversion_impulse_on_body_Ns=conversion,reservoir_impulse_on_fluid_Ns=[v*scale for v in entry['reservoir_momentum']],
            fluid_momentum_Ns=self.fluid_momentum_Ns(),momentum_residual_Ns=entry['momentum_residual']*scale,
            body_interface_work_J=sum((a+b)*v for a,b,v in zip(wall,conversion,velocity)),
            covered=entry['covered'],exposed=entry['exposed'],physical_accuracy_qualified=False,
            boundary='halfway disk with explicit global mass reservoir',force_owner='fluid only; DEM contact excluded')
        self.history.append(record)
        return record

    def snapshot(self):
        return dict(schema='tensorlbm.live-disk-restart/1',config=self.config,solver=self.solver.snapshot(),
            velocity_lattice=self.solver.velocity.tolist(),history=self.history)

    @classmethod
    def restore(cls,data):
        if set(data)!={'schema','config','solver','velocity_lattice','history'} or data['schema']!='tensorlbm.live-disk-restart/1':
            raise ValueError('live restart schema mismatch')
        obj=cls(**data['config'])
        if obj.solver.config != data['solver']['config']:
            raise ValueError('restart SI/lattice configuration mismatch')
        obj.solver=MovingDisk.restore(data['solver'])
        velocity=_vector(data['velocity_lattice'],'restart velocity')
        if math.hypot(*velocity)>.1:
            raise ValueError('invalid restart speed')
        obj.solver.velocity=torch.tensor(velocity,dtype=torch.float64)
        if len(data['history'])!=obj.solver.time:
            raise ValueError('restart exchange history mismatch')
        obj.history=data['history']; return obj
