"""Independent transcription of the OSS ``single-phase-periodic-x-plus-constant-
dpdx-driving`` mechanism of ``joconnor22/LIFE`` (GPL-3.0 — algorithm transcribed,
no code copied).

Source of record (read from the checkout at ``/tmp/life``, ``v1.0.3``,
``10th June 2020`` — ``inc/defs.h:30-31``):

=========================================  =====================================
mechanism                                  source
=========================================  =====================================
lattice (D2Q9 velocities / weights)        ``src/Grid.cpp:1249`` / ``:1248``
time step                                  ``src/Grid.cpp:37-98`` (``lbmKernel``)
push streaming, periodic wrap in x and y   ``src/Grid.cpp:229`` / ``:240``
BGK collision + force                       ``src/Grid.cpp:243``
Guo source term (2nd order)                ``src/Grid.cpp:274-285``
macroscopic velocity, ``+F/2`` correction   ``src/Grid.cpp:297-298``
constant dp/dx volume force (frozen)       ``src/Grid.cpp:1042-1043``
force unit conversion                      ``src/Grid.cpp:1257-1260``
x-periodic = ``eFluid`` left/right walls    ``src/Grid.cpp:930-949`` + ``:400-404``
wall BC = regularised (Latt 2008), ``u=0``  ``src/Grid.cpp:302-330`` and
                                           ``regularisedBC`` (``:360-...``)
=========================================  =====================================

How the periodic + dp/dx channel is obtained in the OSS code (there is **no
shipped case that does it** — every ``examples/*/params.h`` sets ``dpdx = 0``
and ``gravity* = 0``, so the path is unexercised by the repo's own byte-exact
``RefData`` regression, ``src/../testing``): set ``WALL_LEFT`` = ``WALL_RIGHT``
= ``eFluid`` (see the comment at ``inc/params.h:64`` "set to eFluid for
periodic") and ``dpdx != 0``.  ``x`` then wraps inside the push-streaming
kernel and the pressure gradient enters as a constant Cartesian volume force.

Stage order (``lbmKernel``), which this module reproduces literally:

1. swap ``f_n <- f``, ``u_n <- u``, ``rho_n <- rho``   (``Grid.cpp:42-47``)
2. stream+collide on **all** nodes, ``f[recv] = f_n + omega (feq(u_n,rho_n) -
   f_n) + (1 - omega/2) * S(u_n, F)``                  (``Grid.cpp:243``)
3. ``macroscopic`` on the ``eFluid`` nodes only        (``Grid.cpp:90-95``)
4. ``applyBCs`` + ``macroscopic`` on the BC nodes      (``Grid.cpp:96-107``)

Notes on fidelity
-----------------
* the OSS force field ``force_xy`` is built **once**, outside the time loop,
  from ``rho`` at initialisation (``Grid.cpp:1038-1045``) and never refreshed —
  ``F`` is frozen at ``rho = rho0``;
* the OSS lattice velocity order differs from ``tensorlbm.d2q9``
  (``(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)``); this
  transcription keeps the OSS order on purpose so that the two implementations
  share nothing but the mathematics;
* wall nodes also collide and push into the fluid (stage 2 covers every node),
  which is why the collision at a wall node uses the ``u = F/2rho`` velocity
  that stage 4 leaves behind;
* float64 throughout (the OSS code is double precision).

Usage::

    k = LifeKernel2D(nx=4, ny=33, tau=0.5375, force=4.883e-6)
    k.run_until_steady()
    k.profile()          # (ny-2,) wall-normal velocity profile
"""

from __future__ import annotations

import numpy as np

# OSS lattice, ``src/Grid.cpp:1248-1249`` verbatim (index order preserved)
LIFE_C = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
LIFE_W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
          1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
LIFE_OPP = (0, 2, 1, 4, 3, 6, 5, 8, 7)

_CI = np.array(LIFE_C, dtype=np.int64)
_CF = _CI.astype(np.float64)
_W = np.array(LIFE_W, dtype=np.float64)
_OPP = np.array(LIFE_OPP, dtype=np.int64)
CS2 = 1.0 / 3.0


class LifeKernel2D:
    """``eFluid`` in x (periodic) + ``eWall`` in y + constant ``dpdx`` body force."""

    def __init__(self, nx: int, ny: int, tau: float, force: float = 0.0,
                 rho0: float = 1.0) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.force = float(force)          # lattice force density, FROZEN (Grid.cpp:1042)
        self.rho0 = float(rho0)
        n = (self.nx, self.ny)
        # Grid.cpp:1010-1035 (uniform initial velocity = 0) then
        # Grid.cpp:1056-1065: f = equilibrium(rho, u)
        self.u = np.zeros((2,) + n)
        self.rho = np.full(n, self.rho0)
        self.f = self._equilibrium(self.rho, self.u)
        self.u_n = self.u.copy()
        self.rho_n = self.rho.copy()
        self.steps = 0

    # -- operators ---------------------------------------------------------
    def _equilibrium(self, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
        """``Grid.cpp`` ``equilibrium`` — BGK branch (2nd order Hermite), vectorised."""
        ux, uy = u[0], u[1]
        cu = _CF[:, 0, None, None] * ux[None] + _CF[:, 1, None, None] * uy[None]
        cx2, cy2 = _CF[:, 0, None, None] ** 2, _CF[:, 1, None, None] ** 2
        return rho[None] * _W[:, None, None] * (
            1.0 + 3.0 * cu
            + 4.5 * (ux[None] ** 2 * (cx2 - CS2) + uy[None] ** 2 * (cy2 - CS2))
            + 9.0 * _CF[:, 0, None, None] * _CF[:, 1, None, None] * ux[None] * uy[None]
        )

    def _lattice_force(self, u: np.ndarray) -> np.ndarray:
        """``Grid.cpp:274-285`` — ``3 w_i [(c-u) + 3 c (c.u)] . F``, ``F`` frozen."""
        fx, ux, uy = self.force, u[0], u[1]
        cu = _CF[:, 0, None, None] * ux[None] + _CF[:, 1, None, None] * uy[None]
        return 3.0 * _W[:, None, None] * (
            fx * (_CF[:, 0, None, None] - ux[None] + 3.0 * _CF[:, 0, None, None] * cu))

    def _macroscopic(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``Grid.cpp:287-299`` — ``u = (sum(c f) + F/2)/rho``."""
        rho = f.sum(axis=0)
        mx = np.tensordot(_CF[:, 0], f, axes=(0, 0))
        my = np.tensordot(_CF[:, 1], f, axes=(0, 0))
        return rho, np.stack([(mx + 0.5 * self.force) / rho, my / rho])

    def _stream_collide(self) -> np.ndarray:
        """``Grid.cpp:243`` + ``:240`` (push with a full periodic wrap in x and y)."""
        feq = self._equilibrium(self.rho_n, self.u_n)
        src = self._lattice_force(self.u_n)
        f_post = self.f + self.omega * (feq - self.f) + (1.0 - 0.5 * self.omega) * src
        out = np.empty_like(f_post)
        for v in range(9):
            # recv = ((i+cx+Nx)%Nx, (j+cy+Ny)%Ny)
            out[v] = np.roll(f_post[v], shift=(int(_CI[v, 0]), int(_CI[v, 1])), axis=(0, 1))
        return out

    def _regularised_bc(self, f: np.ndarray, row: int, normal: int) -> None:
        """``Grid.cpp`` ``regularisedBC`` — axis-aligned, non-corner branch."""
        u_n = self.u_n
        u_n[:, :, row] = 0.0                      # eWall: u = 0
        sel_in = _CI[:, 1] == -normal             # c . n < 0
        sel_zero = _CI[:, 1] == 0
        fplus = f[sel_in, :, row].sum(axis=0)
        fzero = f[sel_zero, :, row].sum(axis=0)
        rho_w = (2.0 * fplus + fzero) / (1.0 - normal * u_n[1, :, row])
        self.rho_n[:, row] = rho_w
        feq = _W[:, None] * rho_w[None, :]
        sel_unk = _CI[:, 1] == normal             # unknown: they would come from the wall
        f[sel_unk, :, row] = feq[sel_unk] + (f[_OPP[sel_unk], :, row] - feq[_OPP[sel_unk]])
        fneq = f[:, :, row] - feq
        sxx = (_CF[:, 0] ** 2) @ fneq
        syy = (_CF[:, 1] ** 2) @ fneq
        sxy = (_CF[:, 0] * _CF[:, 1]) @ fneq
        coef = (
            ((_CF[:, 0] ** 2 - CS2)[:, None] * sxx[None, :])
            + ((_CF[:, 1] ** 2 - CS2)[:, None] * syy[None, :])
            + (2.0 * _CF[:, 0] * _CF[:, 1])[:, None] * sxy[None, :]
        )
        f[:, :, row] = feq + (_W[:, None] / (2.0 * CS2 * CS2)) * coef

    # -- one time step -----------------------------------------------------
    def step(self) -> None:
        # 1. swap to the start of the timestep (Grid.cpp:42-47)
        self.u_n, self.rho_n = self.u, self.rho
        # 2. stream + collide on every node (Grid.cpp:52-86, :243)
        f_new = self._stream_collide()
        # 3. macroscopic on the eFluid nodes (Grid.cpp:88-95)
        self.rho, self.u = self._macroscopic(f_new)
        # 4. BC nodes: applyBCs (eWall -> regularisedBC) then macroscopic
        for row, normal in ((0, +1), (self.ny - 1, -1)):
            self._regularised_bc(f_new, row, normal)
            rho_w, u_w = self._macroscopic(f_new)
            self.rho[:, row] = rho_w[:, row]
            self.u[:, :, row] = u_w[:, :, row]
        self.f = f_new
        self.steps += 1

    # -- diagnostics -------------------------------------------------------
    def profile(self) -> np.ndarray:
        return self.u[0].mean(axis=0)[1:-1]

    def rel_l2_vs_analytic(self, nu: float) -> float:
        """Error against the parabola whose walls are the OSS wall-**node** centres."""
        y = np.arange(1, self.ny - 1, dtype=np.float64)
        h = float(self.ny - 1)
        s = y / h
        u_ana = 4.0 * (self.force * h * h / (8.0 * nu)) * s * (1.0 - s)
        u = self.profile()
        return float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))

    def run_until_steady(self, *, tol: float = 1e-11, max_steps: int = 400_000,
                         check_every: int = 200) -> dict:
        prev = self.profile()
        n = 0
        delta = float("nan")
        while n < max_steps:
            for _ in range(check_every):
                self.step()
            n += check_every
            cur = self.profile()
            scale = max(float(np.max(np.abs(cur))), 1e-30)
            delta = float(np.max(np.abs(cur - prev))) / scale
            prev = cur
            if not np.isfinite(delta):
                return {"converged": False, "delta": float("nan"), "steps": self.steps}
            if delta < tol:
                return {"converged": True, "delta": delta, "steps": self.steps}
        return {"converged": False, "delta": delta, "steps": self.steps}