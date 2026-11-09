"""Scratch prototype: D2Q5 scalar Taylor dispersion in a plane channel."""
import math
import time

import torch

torch.set_num_threads(1)
from tensorlbm.passive_scalar import (  # noqa: E402
    collide_scalar_bgk,
    equilibrium_scalar,
    macroscopic_scalar,
    stream_scalar,
)


def walls_noflux(g, g_pre):
    g = g.clone()
    # top row: swap +y/-y  (mirror at wall node)
    g[2, -1] = g_pre[4, -1]
    g[4, -1] = g_pre[2, -1]
    g[2, 0] = g_pre[4, 0]
    g[4, 0] = g_pre[2, 0]
    return g


def pure_diffusion(ny=41, D=0.1, nsteps=2000):
    tau = 3.0 * D + 0.5
    y = torch.arange(ny, dtype=torch.float64)
    phi = torch.cos(math.pi * y / (ny - 1)).view(ny, 1).repeat(1, 4)
    z = torch.zeros(ny, 4, dtype=torch.float64)
    g = equilibrium_scalar(phi, z, z)
    a0 = (macroscopic_scalar(g) * torch.cos(math.pi * y / (ny - 1)).view(ny, 1)).mean()
    for _ in range(nsteps):
        c = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, c, z, z, tau)
        gp = g
        g = walls_noflux(stream_scalar(g), gp)
    c = macroscopic_scalar(g)
    a1 = (c * torch.cos(math.pi * y / (ny - 1)).view(ny, 1)).mean()
    rate = -math.log(abs(a1 / a0)) / nsteps
    theory = D * (math.pi / (ny - 1)) ** 2
    print(f"pure diffusion: rate={rate:.6e} theory={theory:.6e} err={(rate/theory-1)*100:.3f}%")
    return rate / theory


def channel_dispersion(ny=41, nx=256, D=0.1, Pe=10.0, nsteps=14000, measure_every=20):
    h = (ny - 1) / 2.0
    u_mean = Pe * D / h
    u_max = 1.5 * u_mean
    tau = 3.0 * D + 0.5
    y = torch.arange(ny, dtype=torch.float64)
    yy = (y - h) / h
    prof = u_max * (1.0 - yy**2)  # zero at wall nodes
    nz = torch.zeros(ny, nx, dtype=torch.float64)
    ux = prof.view(ny, 1).repeat(1, nx)
    k = 2 * math.pi / nx
    x = torch.arange(nx, dtype=torch.float64)
    phi0 = 1.0 + 1e-3 * torch.cos(k * x).view(1, nx) * torch.ones(ny, 1)
    g = equilibrium_scalar(phi0, ux, nz)
    ts, amps = [], []
    t0 = time.time()
    for s in range(nsteps):
        c = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, c, ux, nz, tau)
        gp = g
        g = walls_noflux(stream_scalar(g), gp) if False else _walls_after_stream(g, x_periodic=True)
        if s % measure_every == 0:
            c = macroscopic_scalar(g)
            cm = c[1:-1].mean(dim=0)  # cross-section average (fluid rows)
            re = float((cm * torch.cos(k * x)).sum())
            im = float((cm * torch.sin(k * x)).sum())
            ts.append(s)
            amps.append(math.hypot(re, im))
    el = time.time() - t0
    ts = torch.tensor(ts, dtype=torch.float64)
    amps = torch.tensor(amps, dtype=torch.float64)
    print(f"run {nsteps} steps in {el:.1f}s")


def _walls_after_stream(g, x_periodic=True):
    # generic: stream via roll then fix y-walls using pre-stream values is wrong here;
    # simplest consistent approach: reflect +y at top, -y at bottom after roll
    gs = stream_scalar(g)
    gs = gs.clone()
    top = g[2, -1].clone()
    bot = g[4, 0].clone()
    gs[4, -1] = top
    gs[2, 0] = bot
    return gs


if __name__ == "__main__":
    pure_diffusion()