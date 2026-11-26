"""CPU frequency response of a stationary D2Q5 diffusion slab."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch

from .scalar import (
    _C5,
    _CY5,
    _OPP5,
    _W5_EXACT,
    scalar_collide_diffusion,
    scalar_link_transfer,
    scalar_stream_slab,
)


@dataclass(frozen=True)
class HarmonicScalarSlab:
    populations: torch.Tensor
    x_transfer: torch.Tensor
    y_transfer: torch.Tensor

    @property
    def scalar(self):
        return self.populations.sum(dim=0)


def solve_harmonic_scalar_slab(
    nx, ny, tau, frequency, west, east, *, tau_y=None, tau_even=None, tau_xy=0.0, source=None,
    mean_temperature=None, wall_transfer=(None, None)
):
    """Solve the discrete D2Q5 collision/streaming equations at one frequency.

    frequency is radians per lattice step, 0 <= frequency < π. Wall phasors
    use exp(+iωt) and are sampled at step midpoints. Supply a scalar or ny
    complex values per wall, or None for insulation. The y axis is periodic.
    Returns CPU complex128 populations at integer times and oriented link
    transfers at midpoints. x_transfer includes both wall faces toward +x;
    the last row of y_transfer crosses the periodic seam toward +y.
    tau sets the x relaxation time. tau_y and tau_even default to tau.
    tau_xy/3 sets the lattice cross diffusivity. source is a complex midpoint
    temperature increment per lattice step, broadcastable to (ny,nx). Add it
    locally after streaming using the five equilibrium weights. For physical
    heat generation Q, supply Q*dt/(rho*cp). A steady slab with both walls
    flux-driven requires balanced wall and volume inputs and a prescribed mean.
    wall_transfer entries are inward midpoint increments for west and east;
    specify None for the corresponding wall temperature.
    """
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (nx, ny)):
        raise ValueError("nx and ny must be positive integers")
    if not math.isfinite(tau) or tau <= 0.5:
        raise ValueError("tau must be finite and greater than 0.5")
    if not 0 <= frequency < math.pi:
        raise ValueError("frequency must lie in [0, pi)")
    closed = west is None and east is None
    closed_dc = frequency == 0 and closed
    if closed_dc and mean_temperature is None:
        raise ValueError("steady flux-driven diffusion requires a temperature reference")
    if mean_temperature is not None:
        if not closed_dc or np.ndim(mean_temperature) != 0 or not np.isfinite(mean_temperature):
            raise ValueError("a finite mean temperature applies only to steady flux-driven slabs")
    wall_values = []
    for wall in (west, east):
        if wall is None:
            wall_values.append(None)
        else:
            field = np.broadcast_to(np.asarray(wall, dtype=complex), (ny,)).copy()
            if not np.isfinite(field).all():
                raise ValueError("wall phasors must be finite")
            wall_values.append(field)
    uniform_response = complex(mean_temperature) if closed_dc else 0j
    if frequency == 0 and not closed:
        # A constant scalar satisfies the steady collision and streaming
        # equations and carries no flux. Solve relative to a fixed wall.
        uniform_response = next(w[0] for w in wall_values if w is not None)
        wall_values = [None if w is None else w-uniform_response for w in wall_values]
    west, east = wall_values
    weights = np.asarray(_W5_EXACT)
    # Apply the same linear collision to the five basis populations.
    basis = torch.eye(5, dtype=torch.float64).reshape(5, 1, 5)
    collision = scalar_collide_diffusion(
        basis,
        tau,
        tau if tau_y is None else tau_y,
        tau if tau_even is None else tau_even,
        tau_xy=tau_xy,
    )[:, 0].numpy()
    collide = sparse.kron(sparse.csr_matrix(collision), sparse.eye(nx), format="csr")
    phase = np.exp(1j * frequency)
    midpoint = np.exp(0.5j * frequency)
    identity = sparse.eye(5 * nx, format="csc")
    # Each row gathers one upstream population after collision. At an x wall
    # the gather instead uses the opposite population in the same cell.
    index = np.arange(nx)
    columns = np.arange(5)[:, None] * nx + (index - _C5.numpy()[:, None]) % nx
    columns[1, 0] = int(_OPP5[1]) * nx
    columns[3, -1] = int(_OPP5[3]) * nx + nx - 1
    forcing = np.zeros((5, ny, nx), dtype=complex)
    for wall, q, col in ((west, 1, 0), (east, 3, -1)):
        if wall is not None:
            forcing[q, :, col] = 2 * weights[q] * midpoint * np.fft.fft(wall)
    if len(wall_transfer) != 2:
        raise ValueError("supply a transfer entry for each wall")
    wall_mean = 0j
    wall_scale = 0.
    for flux, (wall, q, col) in zip(wall_transfer, ((west, 1, 0), (east, 3, -1))):
        if flux is not None:
            if wall is not None:
                raise ValueError("a wall cannot prescribe both temperature and transfer")
            values = np.broadcast_to(np.asarray(flux, dtype=complex), (ny,))
            if not np.isfinite(values).all():
                raise ValueError("wall transfer must be finite")
            forcing[q, :, col] += midpoint * np.fft.fft(values)
            wall_mean += values.mean()/nx
            wall_scale += np.abs(values).mean()/nx
    if source is not None or (closed and any(v is not None for v in wall_transfer)):
        increment = np.broadcast_to(np.asarray(0. if source is None else source, dtype=complex), (ny, nx))
        if not np.isfinite(increment).all():
            raise ValueError("source phasors must be finite")
        if closed:
            average = increment.mean() + wall_mean
            if closed_dc:
                if abs(average) > 64*np.finfo(float).eps*(np.abs(increment).mean()+wall_scale):
                    raise ValueError("the steady slab requires zero net source including wall transfers")
            else:
                uniform_response = average / (2j*math.sin(frequency/2))
            increment = increment - average
        forcing += weights[:, None, None] * midpoint * np.fft.fft(increment, axis=0)
    modes = np.zeros_like(forcing)
    for j in range(ny):
        rhs = forcing[:, j, :].ravel()
        if not np.any(rhs):
            continue
        factors = np.exp(-2j * np.pi * j * _CY5.numpy()[:, None] / ny) * np.ones((1, nx))
        if west is not None:
            factors[1, 0] = -1.0
        if east is not None:
            factors[3, -1] = -1.0
        stream = sparse.csr_matrix(
            (factors.ravel(), columns.ravel(), np.arange(5 * nx + 1)), shape=identity.shape
        )
        system = phase * identity - (stream @ collide).tocsc()
        if closed and j == 0:
            # The remaining response has zero total population. Enforce its
            # known mean in place of the redundant global balance.
            system = system.tolil()
            system[0, :] = 1.
            system = system.tocsc()
            rhs = rhs.copy()
            rhs[0] = 0.
        state = spsolve(system, rhs)
        modes[:, j, :] = state.reshape(5, nx)
    populations = torch.from_numpy(np.fft.ifft(modes, axis=1))
    post = torch.from_numpy(collision).to(populations) @ populations.reshape(5, -1)
    post = post.reshape_as(populations) / midpoint
    _, exchange = scalar_stream_slab(post, west, east, wall_transfer=wall_transfer)
    x_transfer = torch.cat(
        (exchange[0, :, None], scalar_link_transfer(post, "x"), -exchange[1, :, None]), dim=1
    )
    if uniform_response != 0:
        populations = populations + torch.from_numpy(weights)[:, None, None]*uniform_response
    return HarmonicScalarSlab(
        populations, x_transfer, scalar_link_transfer(post, "y", periodic=True)
    )
