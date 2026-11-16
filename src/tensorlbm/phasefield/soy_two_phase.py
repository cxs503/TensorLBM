"""Swift--Osborn--Yeomans (SOY) free-energy two-phase D2Q9 kernel — wall-correct,
viscosity-contrast capable, with measured diagnostics.

Round 2 of anchor ``XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE``
(OSS counterpart: ``ebinan92/Fingering_dynamics`` phasefield-mcmp-poiseuille).

Why a second module next to ``xval/xval_phasefield_mcmp_poiseuille.py``
----------------------------------------------------------------------
The round-1 cross-validation reproduced the OSS free-energy scheme and matched a
*piecewise-parabolic* layered Poiseuille profile to 3.7--7.2 %.  Reviewing that
script against this repository's own lattice conventions turned up four material
defects (details and evidence in
``docs/XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE_R2.md``):

1. **Half-way bounce-back aliasing.**  The wall rows were reflected in place,
   ``f[i, wall] = f[OPPOSITE[i], wall]`` for ``i = 1..8``.  For every opposite
   pair whose members both stream across the wall (2/4, 5/7, 6/8) the first
   assignment overwrites the source of the second, so the partner direction
   becomes a *self-assignment*.  Directions 4, 7, 8 — exactly the three that
   feed the first fluid row under the top wall — were never reflected.
2. **Cyclic wrap into the wall rows.**  Streaming used ``torch.roll`` in *y*, so
   row ``ny-1`` received row ``0``'s populations and vice versa; with defect 1
   those unreflected values leaked from one wall into the opposite fluid row.
3. **Periodic ghost nodes for the phase field.**  ``_compute_gradients`` also
   rolled in *y*, i.e. the order parameter at the bottom wall saw the top wall
   instead of a ghost node.  That corrupts ``mu`` and the capillary force
   ``mu grad(psi)`` in the two rows adjacent to the walls.
4. **Degenerate reference solution.**  The analytic profile was evaluated at
   ``y = j`` while a half-way wall sits at ``y = 0.5`` (and ``ny - 1.5``), and
   the two "layers" were given the *same* viscosity.  The latter makes the
   reference identical to single-phase Poiseuille: the test cannot distinguish a
   working phase-field force from a passive scalar, which is why round 2 adds a
   viscosity-contrast arm and equilibrium (spurious-current / Laplace) arms.

Conventions used here
---------------------
* Fluid rows ``1 .. ny-2``; solid rows ``0`` and ``ny-1``.  Half-way bounce-back
  puts the no-slip planes at ``y = 0.5`` and ``y = ny - 1.5``, so the channel is
  ``H = ny - 2`` wide and the centre of row ``j`` sits at ``y = j - 0.5``
  measured from the bottom wall (the convention already used by
  ``tests/test_mcmp_layered_poiseuille.py`` for the Shan-Chen route).
* ``x`` is periodic (a straight channel is homogeneous in the streamwise
  direction, which isolates the wall/interface physics).
* Free energy ``f = (a/2) psi^2 - (a/4) psi^4 + (kappa/2) |grad psi|^2`` with
  ``a = -2 kappa / xi^2`` (so that ``psi = tanh((y-y_i)/xi)`` is the equilibrium
  profile) and ``kappa = 3 sigma xi / 4`` (so that the planar surface tension is
  ``sigma = 4 kappa / (3 xi)``; :func:`surface_tension_from_kappa` checks it).
* Wagner incompressible free-energy equilibrium for ``f`` and the corresponding
  ``mu``-diffusion equilibrium for ``g``; Guo forcing for ``mu grad(psi) + rho G``.

The module is deliberately small and pure-``torch``: it is the kernel the tests
in ``tests/test_soy_two_phase_poiseuille.py`` grade, and the runner
``scripts/xval_phasefield_mcmp_poiseuille_r2.py`` drives it to produce the
report numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorlbm.d2q9 import C as _C
from tensorlbm.d2q9 import OPPOSITE as _OPP
from tensorlbm.d2q9 import W_EXACT as _W

CX: tuple[int, ...] = tuple(int(v) for v in _C[:, 0])
CY: tuple[int, ...] = tuple(int(v) for v in _C[:, 1])
W9: tuple[float, ...] = tuple(float(v) for v in _W)
OPPOSITE: tuple[int, ...] = tuple(int(v) for v in _OPP)

CS2 = 1.0 / 3.0

#: Directions that cross a *horizontal* wall (all except 0, 1, 3).
CROSS_WALL: tuple[int, ...] = tuple(i for i in range(9) if CY[i] != 0)
#: Directions arriving at the bottom wall row from above have ``CY > 0``.
_INTO_BOTTOM: tuple[int, ...] = tuple(i for i in CROSS_WALL if CY[i] > 0)
_INTO_TOP: tuple[int, ...] = tuple(i for i in CROSS_WALL if CY[i] < 0)


# ---------------------------------------------------------------------------
# Free-energy parameters and exact reference solutions
# ---------------------------------------------------------------------------

def kappa_from_sigma_xi(sigma: float, xi: float) -> float:
    """``kappa`` for a planar interface of width ``xi`` and tension ``sigma``.

    ``sigma = kappa * int (d psi/dn)^2 dn`` with ``psi = tanh(n/xi)`` gives
    ``sigma = 4 kappa / (3 xi)``.
    """
    return 0.75 * sigma * xi


def surface_tension_from_kappa(kappa: float, xi: float) -> float:
    """Inverse of :func:`kappa_from_sigma_xi` (used by the self-consistency check)."""
    return 4.0 * kappa / (3.0 * xi)


@dataclass(frozen=True)
class LayeredPoiseuille:
    """Exact two-layer Poiseuille flow in a plane channel.

    Both layers are driven by the same body force per unit mass ``G``; the
    interface at ``y = h`` carries velocity and shear-stress continuity,
    ``nu_low * du_low/dy = nu_up * du_up/dy``.  Walls are at ``y = 0`` and
    ``y = H``.
    """

    H: float
    h: float
    G: float
    nu_low: float
    nu_up: float

    def __post_init__(self) -> None:
        a1 = -self.G / (2.0 * self.nu_low)
        a2 = -self.G / (2.0 * self.nu_up)
        k = self.nu_up / self.nu_low
        b2 = (a2 * (self.h ** 2 - self.H ** 2) - a1 * self.h ** 2) / (
            self.H - self.h + k * self.h
        )
        object.__setattr__(self, "_a1", a1)
        object.__setattr__(self, "_a2", a2)
        object.__setattr__(self, "_b2", b2)
        object.__setattr__(self, "_b1", k * b2)
        object.__setattr__(self, "_c2", -a2 * self.H ** 2 - b2 * self.H)

    def u(self, y: float) -> float:
        if y <= self.h:
            return self._a1 * y * y + self._b1 * y
        return self._a2 * y * y + self._b2 * y + self._c2

    def profile(self, y: "torch.Tensor") -> "torch.Tensor":
        low = self._a1 * y * y + self._b1 * y
        up = self._a2 * y * y + self._b2 * y + self._c2
        return torch.where(y <= self.h, low, up)

    def flow_rate(self) -> float:
        ql = self._a1 * self.h ** 3 / 3.0 + self._b1 * self.h ** 2 / 2.0
        qu = (
            self._a2 * (self.H ** 3 - self.h ** 3) / 3.0
            + self._b2 * (self.H ** 2 - self.h ** 2) / 2.0
            + self._c2 * (self.H - self.h)
        )
        return ql + qu


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@dataclass
class SOYConfig:
    """Solver configuration (lattice units)."""

    ny: int = 64
    nx: int = 4
    sigma: float = 0.01
    xi: float = 4.0
    tau_a: float = 1.0
    tau_g: float = 0.8
    Gx: float = 0.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    #: Optional viscosity contrast: ``tau_b`` > ``tau_a`` puts the *more viscous*
    #: phase in the ``psi = -1`` layer.  ``None`` = uniform relaxation time.
    tau_b: float | None = None
    #: Reproduce the round-1 wall treatment (cyclic ``roll`` in y plus in-place
    #: reflection, ``f[i, wall] = f[OPPOSITE[i], wall]`` for ``i = 1..8``).  Kept
    #: only so the aliasing defect can be demonstrated and regression-tested.
    legacy_walls: bool = False


class SOYTwoPhase:
    """SOY free-energy two-phase D2Q9 solver with exact half-way bounce-back."""

    def __init__(self, cfg: SOYConfig | None = None):
        self.cfg = cfg or SOYConfig()
        c = self.cfg
        if c.ny < 8 or c.nx < 2:
            raise ValueError("need ny >= 8 and nx >= 2")
        self.ny, self.nx = int(c.ny), int(c.nx)
        self.dtype, self.device = c.dtype, torch.device(c.device)

        self.kappa = kappa_from_sigma_xi(c.sigma, c.xi)
        self.a = -2.0 * self.kappa / c.xi ** 2
        self.xi = float(c.xi)
        self.gamma = (c.tau_g - 0.5) / 3.0

        self.cx = torch.tensor(CX, dtype=self.dtype, device=self.device).view(9, 1, 1)
        self.cy = torch.tensor(CY, dtype=self.dtype, device=self.device).view(9, 1, 1)
        self.w = torch.tensor(W9, dtype=self.dtype, device=self.device).view(9, 1, 1)
        self.w0 = W9[0]

        shape = (9, self.ny, self.nx)
        self.f = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.g = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.psi = torch.zeros((self.ny, self.nx), dtype=self.dtype, device=self.device)
        self.steps = 0

        self._y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(self.ny, 1)
        self._x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, self.nx)

    # -- viscosity helpers ---------------------------------------------------

    @property
    def nu_a(self) -> float:
        return (self.cfg.tau_a - 0.5) * CS2

    @property
    def nu_b(self) -> float:
        t = self.cfg.tau_a if self.cfg.tau_b is None else self.cfg.tau_b
        return (t - 0.5) * CS2

    def _tau_local(self) -> torch.Tensor | float:
        """Relaxation time per node: ``tau_a`` at ``psi=+1``, ``tau_b`` at ``psi=-1``."""
        if self.cfg.tau_b is None:
            return self.cfg.tau_a
        w = 0.5 * (1.0 - self.psi)
        return self.cfg.tau_a + w * (self.cfg.tau_b - self.cfg.tau_a)

    # -- initial conditions --------------------------------------------------

    def _ghost(self, ps: torch.Tensor) -> torch.Tensor:
        """Neutral-wetting closure: wall rows mirror the adjacent fluid row."""
        ps = ps.clone()
        ps[0, :] = ps[1, :]
        ps[-1, :] = ps[-2, :]
        return ps

    def set_planar(self, interface_y: float, Gx: float | None = None) -> None:
        """``psi = tanh((y - interface_y)/xi)``; ``interface_y`` in row-index units."""
        if Gx is not None:
            self.cfg.Gx = float(Gx)
        prof = torch.tanh((self._y - interface_y) / self.xi)      # (ny, 1)
        self.psi = self._ghost(prof.expand(self.ny, self.nx).contiguous())
        self._init_equilibrium()

    def set_droplet(self, R: float, cx: float | None = None, cy: float | None = None) -> None:
        if cx is None:
            cx = 0.5 * self.nx
        if cy is None:
            cy = 0.5 * self.ny
        dist = torch.sqrt((self._x - cx) ** 2 + (self._y - cy) ** 2)
        self._dist = dist
        self.R = float(R)
        self.psi = self._ghost(torch.tanh((dist - R) / self.xi))
        self._init_equilibrium()

    def _init_equilibrium(self) -> None:
        gx, gy, lap = self.gradients(self.psi)
        mu = self.chemical_potential(self.psi, lap)
        rho = torch.ones_like(self.psi)
        zero = torch.zeros_like(self.psi)
        p = self.pressure(rho, self.psi, mu)
        self.f = self._feq(rho, zero, zero, p)
        self.g = self._geq(self.psi, zero, zero, mu)
        self.steps = 0

    # -- free-energy operators ----------------------------------------------

    def _pad_y(self, ps: torch.Tensor) -> torch.Tensor:
        return torch.cat([ps[0:1, :], ps, ps[-1:, :]], dim=0)

    def gradients(self, ps: torch.Tensor):
        """Isotropic D2Q9 ``grad`` and ``lap``: periodic in ``x``, Neumann ghost in ``y``."""
        q = self._pad_y(ps)
        gx = torch.zeros_like(ps)
        gy = torch.zeros_like(ps)
        acc = torch.zeros_like(ps)
        for i in range(1, 9):
            dx, dy = CX[i], CY[i]
            sh = torch.roll(q, -dx, dims=1)[1 - dy: self.ny + 1 - dy, :]
            gx = gx + 3.0 * W9[i] * dx * sh
            gy = gy + 3.0 * W9[i] * dy * sh
            acc = acc + W9[i] * sh
        # lap = (2/cs^2) * sum_{i>0} w_i [psi(x+c_i) - psi(x)], 2/cs^2 = 6
        lap = 6.0 * (acc - (1.0 - self.w0) * ps)
        return gx, gy, lap

    def chemical_potential(self, ps: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        return self.a * ps * (1.0 - ps ** 2) - self.kappa * lap

    def pressure(self, rho: torch.Tensor, ps: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return rho / 3.0 + ps * mu / 3.0

    # -- equilibria ----------------------------------------------------------

    def _feq(self, rho, ux, uy, p):
        cu = self.cx * ux.unsqueeze(0) + self.cy * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)
        A = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, self.ny, self.nx)
        return self.w * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

    def _geq(self, ps, ux, uy, mu):
        cu = self.cx * ux.unsqueeze(0) + self.cy * uy.unsqueeze(0)
        usq = (ux ** 2 + uy ** 2).unsqueeze(0)
        gam = self.gamma
        B = torch.empty((9, self.ny, self.nx), dtype=self.dtype, device=self.device)
        B[0] = (ps - 3.0 * (1.0 - self.w0) * gam * mu) / self.w0
        B[1:] = 3.0 * gam * mu.unsqueeze(0).expand(8, self.ny, self.nx)
        return self.w * (B + ps.unsqueeze(0) * (3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq))

    # -- one time step -------------------------------------------------------

    def macroscopic(self):
        rho = self.f.sum(dim=0)
        gx, gy, lap = self.gradients(self.psi)
        mu = self.chemical_potential(self.psi, lap)
        Fx = mu * gx + rho * self.cfg.Gx
        Fy = mu * gy + rho * 0.0
        jx = (self.f * self.cx).sum(dim=0)
        jy = (self.f * self.cy).sum(dim=0)
        rho_s = rho.clamp(min=1e-12)
        ux = (jx + 0.5 * Fx) / rho_s
        uy = (jy + 0.5 * Fy) / rho_s
        return {"rho": rho, "ux": ux, "uy": uy, "mu": mu, "grad_x": gx, "grad_y": gy}

    def step(self) -> None:
        mac = self.macroscopic()
        rho, ux, uy, mu = mac["rho"], mac["ux"], mac["uy"], mac["mu"]
        Fx = mu * mac["grad_x"] + rho * self.cfg.Gx
        Fy = mu * mac["grad_y"]

        tau = self._tau_local()
        if isinstance(tau, torch.Tensor):
            tau_f = tau.unsqueeze(0)
        else:
            tau_f = tau

        p = self.pressure(rho, self.psi, mu)
        feq = self._feq(rho, ux, uy, p)
        geq = self._geq(self.psi, ux, uy, mu)

        cu = self.cx * ux.unsqueeze(0) + self.cy * uy.unsqueeze(0)
        ciF = self.cx * Fx.unsqueeze(0) + self.cy * Fy.unsqueeze(0)
        uF = (ux * Fx + uy * Fy).unsqueeze(0)
        src = (1.0 - 0.5 / tau_f) * self.w * (3.0 * ciF + 9.0 * cu * ciF - 3.0 * uF)

        self.f = self.f - (self.f - feq) / tau_f + src
        self.g = self.g - (self.g - geq) / self.cfg.tau_g

        # --- streaming: periodic in x, index-shifted in y (no cyclic wrap) ---
        f_post, g_post = self.f, self.g
        f_x = torch.empty_like(f_post)
        g_x = torch.empty_like(g_post)
        for i in range(9):
            dx = CX[i]
            f_x[i] = torch.roll(f_post[i], dx, dims=1) if dx else f_post[i]
            g_x[i] = torch.roll(g_post[i], dx, dims=1) if dx else g_post[i]
        if self.cfg.legacy_walls:
            f_stream = torch.empty_like(f_x)
            g_stream = torch.empty_like(g_x)
            for i in range(9):
                dy = CY[i]
                f_stream[i] = torch.roll(f_x[i], dy, dims=0) if dy else f_x[i]
                g_stream[i] = torch.roll(g_x[i], dy, dims=0) if dy else g_x[i]
            self.f, self.g = f_stream, g_stream
        else:
            f_stream = torch.zeros_like(f_x)
            g_stream = torch.zeros_like(g_x)
            for i in range(9):
                dy = CY[i]
                if dy == 0:
                    f_stream[i] = f_x[i]
                    g_stream[i] = g_x[i]
                elif dy > 0:
                    f_stream[i, dy:] = f_x[i, :-dy]
                    g_stream[i, dy:] = g_x[i, :-dy]
                else:
                    d = -dy
                    f_stream[i, :-d] = f_x[i, d:]
                    g_stream[i, :-d] = g_x[i, d:]
            self.f, self.g = f_stream, g_stream

        if self.cfg.legacy_walls:
            # round-1 semantics: reflect *in place*, in ascending direction order
            for arr_post, arr in ((f_post, self.f), (g_post, self.g)):
                del arr_post
                for i in range(1, 9):
                    if CY[i] == 0:
                        continue
                    opp = OPPOSITE[i]
                    arr[i, 0, :] = arr[opp, 0, :]
                    arr[i, self.ny - 1, :] = arr[opp, self.ny - 1, :]
        else:
            # exact half-way bounce-back on the fluid side
            for arr_post, arr in ((f_post, self.f), (g_post, self.g)):
                for i in _INTO_BOTTOM:        # arriving at row 1 from the wall
                    arr[i, 1, :] = arr_post[OPPOSITE[i], 1, :]
                for i in _INTO_TOP:           # arriving at row ny-2 from the wall
                    arr[i, self.ny - 2, :] = arr_post[OPPOSITE[i], self.ny - 2, :]

            # solid rows carry no information
            self.f[:, 0, :] = 0.0
            self.f[:, -1, :] = 0.0
            self.g[:, 0, :] = 0.0
            self.g[:, -1, :] = 0.0

        self.psi = self._ghost(self.g.sum(dim=0))
        self.steps += 1

    # -- diagnostics ---------------------------------------------------------

    def fluid_rows(self, field):
        return field[1:-1]

    def velocity_profile(self) -> torch.Tensor:
        """Streamwise velocity averaged over the periodic ``x`` direction."""
        mac = self.macroscopic()
        return mac["ux"][1:-1].mean(dim=1)

    def max_velocity(self) -> float:
        mac = self.macroscopic()
        u2 = mac["ux"][1:-1, :] ** 2 + mac["uy"][1:-1, :] ** 2
        return float(torch.sqrt(u2).max())

    def psi_mass(self) -> float:
        return float(self.psi[1:-1, :].sum())

    def interface_width(self) -> float:
        """Fit ``psi = tanh((y - y_i)/w)`` to the averaged profile; returns ``w``."""
        prof = self.psi[1:-1, :].mean(dim=1).cpu().numpy()
        y = torch.arange(1, self.ny - 1, dtype=torch.float64).numpy() - 0.5
        sel = (prof < 0.98) & (prof > -0.98)
        if sel.sum() < 2:
            return float("nan")
        # atanh fit through origin-shifted coordinate: psi = tanh((y-y_i)/w)
        ys, ps = y[sel], prof[sel]
        # linear regression of atanh(psi) on y
        import numpy as np

        val = np.arctanh(np.clip(ps, -0.999999, 0.999999))
        A = np.vstack([ys, np.ones_like(ys)]).T
        slope, icpt = np.linalg.lstsq(A, val, rcond=None)[0]
        return float(1.0 / slope) if slope != 0 else float("nan")

    def layered_reference(self, h_frac: float = 0.5, Gx: float | None = None) -> LayeredPoiseuille:
        """Exact reference for the current configuration.

        ``set_planar`` puts ``psi = +1`` *above* the interface and ``psi = -1``
        below, and :meth:`_tau_local` relaxes ``psi = +1`` with ``tau_a`` and
        ``psi = -1`` with ``tau_b``; hence ``nu_low = nu_b`` and ``nu_up = nu_a``.
        """
        H = float(self.ny - 2)
        return LayeredPoiseuille(
            H=H,
            h=h_frac * H,
            G=self.cfg.Gx if Gx is None else float(Gx),
            nu_low=self.nu_b,
            nu_up=self.nu_a,
        )

    def poiseuille_error(self, ref: LayeredPoiseuille) -> dict:
        u = self.velocity_profile()
        y = torch.arange(1, self.ny - 1, dtype=self.dtype) - 0.5
        u_ana = ref.profile(y)
        diff = u - u_ana
        norm = float(torch.sqrt((u_ana ** 2).sum()))
        l2 = float(torch.sqrt((diff ** 2).sum())) / norm
        linf = float(diff.abs().max()) / float(u_ana.abs().max())
        return {
            "l2_rel": l2,
            "linf_rel": linf,
            "u_max_sim": float(u.max()),
            "u_max_ana": float(u_ana.max()),
            "Q_sim": float(u.sum()),
            "Q_ana": float(u_ana.sum()),
            "profile_sim": u.cpu().numpy(),
            "profile_ana": u_ana.cpu().numpy(),
        }

    def bulk_density(self, R_inner: float = 0.6, R_outer: float = 1.4) -> dict:
        """Densities deep inside / outside a droplet (Laplace-law read-out)."""
        mac = self.macroscopic()
        rho = mac["rho"]
        dist = self._dist
        inner = dist < R_inner * self.R
        outer = dist > R_outer * self.R
        return {"rho_in": float(rho[inner].mean()), "rho_out": float(rho[outer].mean())}

    def laplace_dp(self, R_inner: float = 0.6, R_outer: float = 1.4) -> float:
        d = self.bulk_density(R_inner, R_outer)
        return CS2 * (d["rho_in"] - d["rho_out"])