"""Line-by-line NumPy transliteration of the upstream fused push-pull kernel.

Source (fetched via the GitHub contents API, snapshotted with sha256 in
``ref/PROVENANCE.json``)::

    lbm-principles-practice/code, chapter13/gpu/LBM.cu   (MIT, Kruger et al.)
      gpu_field0_index       LBM.cu:23-26
      gpu_fieldn_index       LBM.cu:33-36
      taylor_green_eval      LBM.cu:65-84
      gpu_init_equilibrium   LBM.cu:120-159
      gpu_stream_collide_save LBM.cu:172-252

Only the CUDA plumbing is dropped (thread/block indices become whole-array
slices, the ``save`` branch stays as a flag).  Arithmetic order, constants
(``w0 = 4/9``, ``ws = 1/9``, ``wd = 1/36``), direction numbering
(``6 2 5 / 3 0 1 / 7 4 8``) and the ``tauinv = 2/(6 nu + 1)`` folding are kept
byte-for-byte so that a discrepancy against the torch arm is a *convention*
discrepancy, not a formulation one.

Storage is upstream's: ``f0`` (``NX*NY``), ``f1``/``f2`` (``8*NX*NY`` each) with
the pointers swapped every step, direction-major layout so ``x`` is contiguous.
"""

from __future__ import annotations

import numpy as np

W0 = 4.0 / 9.0
WS = 1.0 / 9.0
WD = 1.0 / 36.0


def field0_index(x, y, NX):
    return NX * y + x


def fieldn_index(x, y, d, NX, NY):
    return NX * (NY * (d - 1) + y) + x


def taylor_green_eval(t, NX, NY, nu, u_max, rho0=1.0):
    """``taylor_green_eval`` (LBM.cu:65-84)."""
    kx = 2.0 * np.pi / NX
    ky = 2.0 * np.pi / NY
    td = 1.0 / (nu * (kx * kx + ky * ky))
    Y, X = np.meshgrid(np.arange(NY) + 0.5, np.arange(NX) + 0.5, indexing="ij")
    ux = -u_max * np.sqrt(ky / kx) * np.cos(kx * X) * np.sin(ky * Y) * np.exp(-1.0 * t / td)
    uy = u_max * np.sqrt(kx / ky) * np.sin(kx * X) * np.cos(ky * Y) * np.exp(-1.0 * t / td)
    P = (
        -0.25
        * rho0
        * u_max
        * u_max
        * ((ky / kx) * np.cos(2.0 * kx * X) + (kx / ky) * np.cos(2.0 * ky * Y))
        * np.exp(-2.0 * t / td)
    )
    rho = rho0 + 3.0 * P
    return rho, ux, uy


def init_equilibrium(f0, f1, rho, ux, uy):
    """``gpu_init_equilibrium`` (LBM.cu:120-159); ``f1`` is ``(8, NY, NX)``."""
    omusq = 1.0 - 1.5 * (ux * ux + uy * uy)
    tux = 3.0 * ux
    tuy = 3.0 * uy
    w0r = W0 * rho
    wsr = WS * rho
    wdr = WD * rho
    f0[:] = w0r * omusq
    c = (
        tux,
        tuy,
        -tux,
        -tuy,
        tux + tuy,
        tuy - tux,
        -(tux + tuy),
        tux - tuy,
    )
    for i, cd in enumerate(c):
        wr = wdr if i >= 4 else wsr
        f1[i] = wr * (omusq + cd * (1.0 + 0.5 * cd))


def stream_collide_save(f0, f1, f2, rho, ux, uy, nu, save):
    """``gpu_stream_collide_save`` (LBM.cu:172-252).

    Pulls the eight shifted neighbours of ``f1`` with periodic wrap, computes the
    moments, optionally writes them, then writes the post-collision populations
    into ``f2`` (directions 1..8) and back into ``f0`` in place.
    """
    NX = f1.shape[2]
    NY = f1.shape[1]
    tauinv = 2.0 / (6.0 * nu + 1.0)
    omtauinv = 1.0 - tauinv

    # wrap of the *source* index: dst(x,y) <- src(x-cx, y-cy)  (LBM.cu:181-201)
    def sh(a, cx, cy):
        return np.roll(a, shift=(cy, cx), axis=(-2, -1))

    ft = [
        sh(f1[0], 1, 0),    # 1: E
        sh(f1[1], 0, 1),    # 2: N
        sh(f1[2], -1, 0),   # 3: W
        sh(f1[3], 0, -1),   # 4: S
        sh(f1[4], 1, 1),    # 5: NE
        sh(f1[5], -1, 1),   # 6: NW
        sh(f1[6], -1, -1),  # 7: SW
        sh(f1[7], 1, -1),   # 8: SE
    ]
    ft0 = f0
    r = ft0 + ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7]
    rhoinv = 1.0 / r
    u_x = rhoinv * (ft[0] + ft[4] + ft[7] - (ft[2] + ft[5] + ft[6]))
    u_y = rhoinv * (ft[1] + ft[4] + ft[5] - (ft[3] + ft[6] + ft[7]))

    if save:
        rho[...] = r
        ux[...] = u_x
        uy[...] = u_y

    tw0r = tauinv * W0 * r
    twsr = tauinv * WS * r
    twdr = tauinv * WD * r
    omusq = 1.0 - 1.5 * (u_x * u_x + u_y * u_y)
    tux = 3.0 * u_x
    tuy = 3.0 * u_y
    f0[:] = omtauinv * ft0 + tw0r * omusq
    c = (
        tux,
        tuy,
        -tux,
        -tuy,
        tux + tuy,
        tuy - tux,
        -(tux + tuy),
        tux - tuy,
    )
    for i, cd in enumerate(c):
        twr = twdr if i >= 4 else twsr
        f2[i] = omtauinv * ft[i] + twr * (omusq + cd * (1.0 + 0.5 * cd))


class OSSFusedStepper:
    """Upstream chapter-13 fused push-pull stepping, NumPy, lattice units."""

    def __init__(self, NX, NY, nu, u_max, rho0=1.0):
        self.NX, self.NY = int(NX), int(NY)
        self.nu, self.u_max, self.rho0 = float(nu), float(u_max), float(rho0)
        self.f0 = np.zeros((self.NY, self.NX))
        self.f1 = np.zeros((8, self.NY, self.NX))
        self.f2 = np.zeros((8, self.NY, self.NX))
        self.rho = np.zeros((self.NY, self.NX))
        self.ux = np.zeros((self.NY, self.NX))
        self.uy = np.zeros((self.NY, self.NX))
        rho, ux, uy = taylor_green_eval(0, self.NX, self.NY, self.nu, self.u_max, self.rho0)
        self.rho, self.ux, self.uy = rho.copy(), ux.copy(), uy.copy()
        init_equilibrium(self.f0, self.f1, rho, ux, uy)
        self.time = 0

    def step(self, save=True):
        stream_collide_save(
            self.f0, self.f1, self.f2, self.rho, self.ux, self.uy, self.nu, save
        )
        self.f1, self.f2 = self.f2, self.f1  # swap pointers (main.cu)
        self.time += 1

    def fields(self):
        return np.concatenate([self.f0[None], self.f1], axis=0)