"""CPU frequency response of a stationary D2Q5 diffusion slab."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch

from .scalar import _C5, _CY5, _OPP5, _W5_EXACT, scalar_collide_diffusion, scalar_link_transfer, scalar_stream_slab


@dataclass(frozen=True)
class HarmonicScalarSlab:
    populations: torch.Tensor
    x_transfer: torch.Tensor
    y_transfer: torch.Tensor

    @property
    def scalar(self):
        return self.populations.sum(dim=0)


def solve_harmonic_scalar_slab(nx, ny, tau, frequency, west, east, *, tau_y=None, tau_even=None, tau_xy=0.):
    """Solve the discrete D2Q5 collision/streaming equations at one frequency.

    frequency is radians per lattice step, 0 <= frequency < π. Wall phasors
    use exp(+iωt) and are sampled at step midpoints. Supply a scalar or ny
    complex values per wall, or None for insulation. The y axis is periodic.
    Returns CPU complex128 populations at integer times and oriented link
    transfers at midpoints. x_transfer includes both wall faces toward +x;
    the last row of y_transfer crosses the periodic seam toward +y.
    tau sets the x relaxation time. tau_y and tau_even default to tau.
    tau_xy/3 sets the lattice cross diffusivity.
    """
    if any(not isinstance(n, (int,np.integer)) or n < 1 for n in (nx,ny)):
        raise ValueError("nx and ny must be positive integers")
    if not math.isfinite(tau) or tau <= .5:
        raise ValueError("tau must be finite and greater than 0.5")
    if not 0 <= frequency < math.pi:
        raise ValueError("frequency must lie in [0, pi)")
    if frequency == 0 and west is None and east is None:
        raise ValueError("steady insulated diffusion requires a temperature reference")
    wall_values = []
    for wall in (west,east):
        if wall is None:
            wall_values.append(None)
        else:
            field = np.broadcast_to(np.asarray(wall,dtype=complex),(ny,)).copy()
            if not np.isfinite(field).all():
                raise ValueError("wall phasors must be finite")
            wall_values.append(field)
    west, east = wall_values
    weights = np.asarray(_W5_EXACT)
    # Apply the same linear collision to the five basis populations.
    basis = torch.eye(5,dtype=torch.float64).reshape(5,1,5)
    collision = scalar_collide_diffusion(basis,tau,tau if tau_y is None else tau_y,
                                         tau if tau_even is None else tau_even, tau_xy=tau_xy)[:,0].numpy()
    collide = sparse.kron(sparse.csr_matrix(collision), sparse.eye(nx), format="csr")
    phase = np.exp(1j*frequency)
    midpoint = np.exp(.5j*frequency)
    identity = sparse.eye(5*nx,format="csc")
    # Each row gathers one upstream population after collision. At an x wall
    # the gather instead uses the opposite population in the same cell.
    index = np.arange(nx)
    columns = np.arange(5)[:,None]*nx + (index-_C5.numpy()[:,None])%nx
    columns[1,0] = int(_OPP5[1])*nx
    columns[3,-1] = int(_OPP5[3])*nx+nx-1
    forcing = np.zeros((5,ny,nx),dtype=complex)
    for wall,q,col in ((west,1,0),(east,3,-1)):
        if wall is not None:
            forcing[q,:,col] = 2*weights[q]*midpoint*np.fft.fft(wall)
    modes = np.zeros_like(forcing)
    for j in range(ny):
        rhs = forcing[:,j,:].ravel()
        if not np.any(rhs):
            continue
        factors = np.exp(-2j*np.pi*j*_CY5.numpy()[:,None]/ny)*np.ones((1,nx))
        if west is not None:
            factors[1,0] = -1.
        if east is not None:
            factors[3,-1] = -1.
        stream = sparse.csr_matrix((factors.ravel(),columns.ravel(),np.arange(5*nx+1)),shape=identity.shape)
        state = spsolve(phase*identity-(stream@collide).tocsc(),rhs)
        modes[:,j,:] = state.reshape(5,nx)
    populations = torch.from_numpy(np.fft.ifft(modes,axis=1))
    post = torch.from_numpy(collision).to(populations) @ populations.reshape(5,-1)
    post = post.reshape_as(populations)/midpoint
    _, exchange = scalar_stream_slab(post,west,east)
    x_transfer = torch.cat((exchange[0,:,None],scalar_link_transfer(post,"x"),-exchange[1,:,None]),dim=1)
    return HarmonicScalarSlab(populations,x_transfer,scalar_link_transfer(post,"y",periodic=True))
