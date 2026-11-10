"""Direct Simulation Monte Carlo (DSMC) in the same 1-D slab geometry.

Purpose
-------
This is the **independent** reference for the transitional regime of the ladder:
it shares no code, no discretisation and no closure with
:mod:`tensorlbm.rarefied.bgk_slab` (which is deterministic, linearised, BGK and
discrete-velocity).  Agreement between the two at ``Kn`` in ``[0.05, 10]`` is
therefore a genuine cross-check of the mass-flow-rate curve, and the
free-molecular asymptote gives a third, fully analytic anchor.

Method
------
* Hard-sphere molecules (``m = 1``, ``R T = 1``, number density ``n = 1`` so that
  ``rho_0 = 1``), diameter ``d`` fixed by the target mean free path
  ``lambda = 1 / (sqrt(2) pi d^2 n)``.
* Fully developed flow: no ``x`` coordinate is tracked (nothing depends on it),
  only ``(y, v_x, v_y)``; the body force ``g`` along ``x`` is applied as a velocity
  increment each step (the classical force-driven Poiseuille DSMC setup).
* Collisions: no-time-counter (NTC) scheme with a per-step maximum relative
  speed ``c_max``, then acceptance probability ``c_r / c_max``; post-collision
  velocities are isotropically scattered in the centre-of-mass frame (hard
  sphere, elastic).
* Walls: Maxwell reflection with tangential momentum accommodation coefficient
  ``sigma`` -- probability ``sigma`` diffuse (thermal wall: ``v_n`` from the
  flux-weighted half-Maxwellian, ``v_t`` from a Maxwellian at the wall
  temperature, plus the wall's tangential drift ``U_w``), probability
  ``1 - sigma`` specular (``v_n`` reversed, ``v_t`` unchanged, no tangential
  momentum transfer).
* Sampling: per-cell sums of ``v_x`` accumulated over the sampling window;
  block averaging over ``n_blocks`` sub-windows gives a statistical error bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DSMCResult:
    kn: float
    sigma: float
    y: np.ndarray
    u: np.ndarray
    u_err: np.ndarray
    q: float
    q_err: float
    g: float
    flow: str
    ncells: int
    ppc: int
    steps: int
    dt: float
    lam: float
    extra: dict = field(default_factory=dict)


def run_dsmc_slab(kn: float, sigma: float = 1.0, g: float = 0.05,
                  u_walls=(0.0, 0.0), H: float = 1.0, RT: float = 1.0,
                  ncells: int | None = None, ppc: int = 60, steps: int | None = None,
                  burn_frac: float = 0.35, n_blocks: int = 10, seed: int = 20261110,
                  flow: str | None = None, progress: bool = False) -> DSMCResult:
    """Run force-driven Poiseuille (default) or plane Couette DSMC.

    Parameters
    ----------
    kn : float
        Target Knudsen number ``lambda/H``.
    sigma : float
        Tangential momentum accommodation coefficient (diffuse fraction).
    g : float
        Body force along ``x`` (ignored when the flow is driven by moving walls).
    u_walls : (float, float)
        Tangential velocities of the bottom/top wall (Couette).
    ncells : int, optional
        Number of collision cells; defaults to ``clip(ceil(4 H/lambda), 40, 400)``
        so that the mean free path is always resolved by >= 4 cells.
    ppc : int
        Nominal particles per cell.
    steps : int, optional
        Total number of time steps (burn-in included); defaults to a
        momentum-diffusion estimate.
    """
    rng = np.random.default_rng(seed)
    lam = kn * H
    c_bar = math.sqrt(8.0 * RT / math.pi)
    if ncells is None:
        ncells = int(min(400, max(40, math.ceil(4.0 * H / lam))))
    dy = H / ncells
    n_real = 1.0                                   # number density (rho_0 = n m = 1)
    d = math.sqrt(1.0 / (math.sqrt(2.0) * math.pi * n_real * lam))
    tau = lam * math.sqrt(2.0 / (math.pi * RT))    # BGK-equivalent relaxation time
    nu = tau * RT
    dt = 0.2 * min(dy, lam) / c_bar
    if steps is None:
        t_diff = H * H / max(nu, 1e-12)
        steps = int(min(400_000, max(20_000, 6.0 * t_diff / dt)))

    npart = ncells * ppc
    F_N = dy / ppc                                  # real molecules per simulated particle
    vol = dy                                        # cell volume per unit width/length
    sig_T = math.pi * d * d
    c_max = 6.0 * math.sqrt(2.0 * RT)

    y = rng.random(npart) * H
    vx = rng.normal(0.0, math.sqrt(RT), npart)
    vy = rng.normal(0.0, math.sqrt(RT), npart)

    cu = np.zeros(ncells)
    cn = np.zeros(ncells)
    nblk = max(1, int(n_blocks))
    blk_u = np.zeros((nblk, ncells))
    blk_s = np.zeros(nblk)
    burn = int(burn_frac * steps)
    sample_steps = steps - burn
    blk_len = max(1, sample_steps // nblk)

    U0, UH = float(u_walls[0]), float(u_walls[1])
    if flow is None:
        flow = "couette" if (g == 0.0 and (U0 != 0.0 or UH != 0.0)) else "poiseuille"

    order = np.argsort(y, kind="stable")
    k = 0
    for step in range(steps):
        # ---- move + body force ------------------------------------------
        y += vy * dt
        if g != 0.0:
            vx += g * dt

        # ---- wall collisions (Maxwell: diffuse / specular) --------------
        for sgn, yw, Uw in ((-1.0, 0.0, U0), (1.0, H, UH)):
            if sgn < 0:
                hit = y < yw
            else:
                hit = y > yw
            if hit.any():
                nh = int(hit.sum())
                y[hit] = yw - sgn * np.abs(y[hit] - yw)
                diffuse = rng.random(nh) < sigma
                nd = int(diffuse.sum())
                if nd:
                    vx[hit] = np.where(
                        diffuse,
                        rng.normal(Uw, math.sqrt(RT), nh),
                        vx[hit],
                    )
                    # flux-weighted half-Maxwellian for the normal component
                    vn = np.sqrt(-2.0 * RT * np.log(rng.random(nh)))
                    vy[hit] = np.where(diffuse, -sgn * vn, -vy[hit])
                else:
                    vy[hit] = -vy[hit]

        # ---- collisions (NTC, hard sphere) ------------------------------
        cell = np.clip((y / dy).astype(np.int64), 0, ncells - 1)
        order = np.argsort(cell, kind="stable")
        cell_s = cell[order]
        starts = np.searchsorted(cell_s, np.arange(ncells), side="left")
        ends = np.searchsorted(cell_s, np.arange(ncells), side="right")
        cnt = (ends - starts).astype(float)
        n_cand = 0.5 * cnt * cnt * F_N * sig_T * c_max * dt / vol
        # NTC: the number of candidate pairs per cell is stochastic --
        # floor(n_cand) pairs plus one more with probability frac(n_cand).
        base = np.floor(n_cand).astype(np.int64)
        frac = n_cand - base
        nmax = int(base.max()) + 1 if base.size else 0
        if nmax > 0:
            slot = np.arange(nmax)[None, :]
            valid = slot < base[:, None]
            extra = rng.random(ncells) < frac
            if extra.any():
                valid[np.flatnonzero(extra), base[extra]] = True
            r1 = rng.random((ncells, nmax))
            r2 = rng.random((ncells, nmax))
            c1 = cnt[:, None]
            i1 = starts[:, None] + np.minimum((r1 * c1).astype(np.int64), np.maximum(c1 - 1, 0).astype(np.int64))
            i2 = starts[:, None] + np.minimum((r2 * c1).astype(np.int64), np.maximum(c1 - 1, 0).astype(np.int64))
            i1 = i1[valid]
            i2 = i2[valid]
            good = i1 != i2
            i1, i2 = i1[good], i2[good]
            if i1.size:
                p1 = order[i1]
                p2 = order[i2]
                dvx = vx[p1] - vx[p2]
                dvy = vy[p1] - vy[p2]
                cr = np.sqrt(dvx * dvx + dvy * dvy)
                acc = rng.random(cr.size) * c_max < cr
                p1, p2 = p1[acc], p2[acc]
                cr = cr[acc]
                if p1.size:
                    cmx = 0.5 * (vx[p1] + vx[p2])
                    cmy = 0.5 * (vy[p1] + vy[p2])
                    th = rng.random(p1.size) * 2.0 * math.pi
                    hx = 0.5 * cr * np.cos(th)
                    hy = 0.5 * cr * np.sin(th)
                    vx[p1] = cmx - hx
                    vy[p1] = cmy - hy
                    vx[p2] = cmx + hx
                    vy[p2] = cmy + hy

        # ---- sampling ---------------------------------------------------
        if step >= burn:
            np.add.at(cu, cell, vx)
            np.add.at(cn, cell, 1.0)
            blk = min(nblk - 1, (step - burn) // blk_len)
            np.add.at(blk_u[blk], cell, vx)
            blk_s[blk] += 1.0
        if progress and step % max(1, steps // 10) == 0:
            print(f"  dsmc kn={kn:g} step {step}/{steps}", flush=True)

    yc = (np.arange(ncells) + 0.5) * dy
    n_steps_s = int(cn.sum() / npart) if npart else sample_steps
    n_steps_total = max(1, sample_steps)
    u = np.where(cn > 0, cu / np.maximum(cn, 1.0), 0.0)
    # mass flow per unit width: Q = rho_0 * int u dy = F_N * sum_cells sum_t vx / n_steps
    q = float(F_N * np.sum(cu) / n_steps_total)
    q_blk = np.array([F_N * np.sum(blk_u[b]) / blk_s[b] if blk_s[b] > 0 else np.nan
                      for b in range(nblk)])
    q_err = (float(np.nanstd(q_blk, ddof=1) / math.sqrt(np.sum(blk_s > 0)))
             if np.sum(blk_s > 0) > 1 else float("nan"))
    # per-cell statistical standard error of the mean velocity
    u_err = np.sqrt(RT / np.maximum(cn, 1.0))
    return DSMCResult(
        kn=kn, sigma=sigma, y=yc, u=u, u_err=u_err, q=q, q_err=q_err, g=g,
        flow=flow, ncells=ncells, ppc=ppc, steps=steps, dt=dt, lam=lam,
        extra={"F_N": F_N, "dt_over_coll": dt / (lam / c_bar), "d_over_lam": d / lam,
               "q_fm": 0.5 * g * H * H * math.sqrt(math.pi / (2.0 * RT)),
               "nu": nu, "diffusion_times_sampled": (H * H / nu) / (n_steps_total * dt),
               "q_blocks": [float(v) for v in q_blk],
               "n_particles": npart, "cells_per_mfp": lam / dy},
    )
