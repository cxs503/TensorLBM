"""Implicit regularized perimeter constraints; interior fluid is retained.

SI impulses use the same bilinear J and transpose. This is a marker boundary
projection, not cut-cell fluid exclusion or proof of off-marker impermeability.
"""
import math
import torch
from .ice_coupling_2d import bilinear_map


def circle_markers(center, radius, spacing):
    center=torch.as_tensor(center,dtype=torch.float64)
    if center.shape!=(2,) or not torch.isfinite(center).all():
        raise ValueError('finite xy center required')
    for value in (radius,spacing):
        if isinstance(value,bool) or not math.isfinite(value) or value<=0:
            raise ValueError('positive finite radius and spacing required')
    count=max(8,math.ceil(2*math.pi*radius/spacing))
    angle=torch.arange(count,dtype=torch.float64)*2*math.pi/count
    normals=torch.stack((angle.cos(),angle.sin()),1)
    return center+radius*normals,normals


def project_velocity(positions, shape, dx, velocity, cell_mass, target_velocity,
                     compliance=0.):
    """Solve (J M^-1 J^T + compliance I) impulse = target - J u.

    Compliance has units inverse kg; positive compliance is finite impedance.
    Zero compliance requires independent constraints (singular solves fail).
    Returned marker impulses act ON FLUID, with opposite solid reaction.
    """
    if isinstance(compliance,bool) or not math.isfinite(compliance) or compliance<0:
        raise ValueError('finite nonnegative compliance required')
    J=bilinear_map(positions,shape,dx)
    u=torch.as_tensor(velocity,dtype=torch.float64)
    mass=torch.as_tensor(cell_mass,dtype=torch.float64).expand(J.shape[1])
    target=torch.as_tensor(target_velocity,dtype=torch.float64)
    if u.shape!=(J.shape[1],2) or target.shape!=(len(J),2):
        raise ValueError('velocity dimensions inconsistent')
    if not all(bool(torch.isfinite(v).all()) for v in (u,mass,target)) or bool((mass<=0).any()):
        raise ValueError('finite fields and positive cell masses required')
    A=(J/mass)@J.T+compliance*torch.eye(len(J),dtype=torch.float64)
    impulse=torch.linalg.solve(A,target-J@u)
    grid_impulse=J.T@impulse
    corrected=u+grid_impulse/mass[:,None]
    return {'J':J,'marker_impulse':impulse,'grid_impulse':grid_impulse,
            'corrected_velocity':corrected,'slip':J@corrected-target}
