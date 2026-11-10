"""LBM solver for viscoelastic (Oldroyd-B / FENE-P) plane Poiseuille flow.

Method
------
* Momentum: D2Q9 lattice-Boltzmann, BGK collision, Guo forcing.  The polymer
  feedback enters as the divergence of the polymer stress, so the momentum
  equation solved is

      d_t(rho u) + div(rho u u) = -grad p + div(2 mu_s S) + div(tau_p) + F_body

* Conformation: the upper-convected evolution of ``C`` is advanced on the same
  grid with an explicit sub-stepped scheme on top of an *implicitly* relaxed
  (hence unconditionally stable) relaxation step, optionally followed by the
  realizability projection ``C -> max(C, I)`` in the eigen-sense.

* Walls: half-way bounce-back, so the effective no-slip planes sit exactly at
  ``y = +-H`` with ``H = (ny-2)/2`` lattice units (second-order accurate).

The fully developed channel is homogeneous in x, therefore ``nx = 1`` is the
natural (exact) configuration; ``nx > 1`` is supported for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import conformation as conf

__all__ = [
    "ViscoelasticDivergenceError",
    "ChannelResult",
    "ViscoelasticChannel2D",
]

CS2 = 1.0 / 3.0

# D2Q9 stencil
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=float)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=float)
WQ = np.array(
    [4.0 / 9.0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=float,
)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)


class ViscoelasticDivergenceError(RuntimeError):
    """Raised when the explicit viscoelastic coupling loses stability.

    The viscoelastic solver has a finite Weissenberg-number stability envelope;
    beyond it the conformation tensor stops being realizable and the feedback
    force blows up.  This exception is the *declared* failure mode (a negative
    control in the verification suite), never a silent NaN.
    """


@dataclass
class ChannelResult:
    """Everything measured on a converged channel solution."""

    u: np.ndarray
    y: np.ndarray
    C: np.ndarray
    tau_p: np.ndarray
    N1: np.ndarray
    trC: np.ndarray
    gamma_dot: np.ndarray
    steps: int
    converged: bool
    du_inf: float
    H: float
    G: float
    nu0: float
    nu_s: float
    eta_p: float
    beta: float
    lam: float
    model: str
    wi_wall: float
    re: float
    u_max: float
    Q: float
    n_clipped: int
    projection: bool
    upper_convective: bool
    meta: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    @property
    def wall_shear_rate(self) -> float:
        """Shear rate at the wall, finite-differenced on the fluid nodes."""
        return float(self.gamma_dot[1])

    @property
    def tau_wall(self) -> float:
        """Total wall shear stress (solvent + polymer)."""
        return float(self.nu_s * self.gamma_dot[1] + self.tau_p[1, 0, 1])

    def as_dict(self) -> dict:
        skip = {"u", "y", "C", "tau_p", "N1", "trC", "gamma_dot"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}


class ViscoelasticChannel2D:
    """Coupled LBM + conformation-tensor solver for plane Poiseuille flow."""

    def __init__(
        self,
        ny: int = 34,
        nx: int = 1,
        nu0: float = 0.1,
        beta: float = 0.5,
        lam: float = 1.0,
        model: str = "oldroyd_b",
        L2: float | None = None,
        u_max_target: float = 0.04,
        body_force: float | None = None,
        projection: bool = True,
        upper_convective: bool = True,
        n_substeps: int = 4,
        tol: float = 1e-11,   # relative steady-state change of u
        check_every: int = 50,
        max_steps: int = 400_000,
        divergence_u: float = 0.25,
        wi_limit: float | None = None,
        snapshot_steps: tuple = (),
    ):
        if ny < 6:
            raise ValueError("ny must be >= 6 (at least 4 fluid rows)")
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        if 3.0 * beta * nu0 >= 0.5:
            raise ValueError(
                "solvent viscosity too large for BGK: need 3*beta*nu0 < 0.5"
            )

        self.ny, self.nx = int(ny), int(nx)
        self.nu0, self.beta, self.lam = float(nu0), float(beta), float(lam)
        self.nu_s = self.beta * self.nu0
        self.eta_p = (1.0 - self.beta) * self.nu0     # rho = 1 in lattice units
        self.model_name = model
        if model == "oldroyd_b":
            self.model = conf.OldroydB(self.lam)
        elif model == "fene_p":
            self.model = conf.FeneP(self.lam, L2)
        else:
            raise ValueError(f"unknown model {model!r}")

        self.H = 0.5 * (self.ny - 2)
        # y coordinate measured from the channel centre line
        self.y = np.arange(self.ny, dtype=float) - 0.5 * (self.ny - 1)
        self.G = (
            float(body_force)
            if body_force is not None
            else 2.0 * self.nu0 * u_max_target / self.H ** 2
        )

        self.projection = bool(projection)
        self.upper_convective = bool(upper_convective)
        self.n_substeps = int(n_substeps)
        self.tol = float(tol)
        self.check_every = int(check_every)
        self.max_steps = int(max_steps)
        self.divergence_u = float(divergence_u)
        self.wi_limit = wi_limit
        self.snapshot_steps = tuple(int(s) for s in snapshot_steps)
        self._snapshot_set = set(self.snapshot_steps)
        self._snapshots: dict = {}

        self.omega = 1.0 / (3.0 * self.nu_s + 0.5)
        zeros = np.zeros((self.ny, self.nx))
        self._f = self._equilibrium(np.ones((self.ny, self.nx)), zeros, zeros)
        self._C = np.broadcast_to(conf.I3, (self.ny, self.nx, 3, 3)).copy()
        self._C = np.ascontiguousarray(self._C)
        self.n_clipped_total = 0
        self._wi_wall = self.lam * abs(self.G) * self.H / self.nu0

    # ------------------------------------------------------------------ #
    # equilibrium / moments
    # ------------------------------------------------------------------ #
    def _equilibrium(self, rho, u, v):
        cu = EX[:, None, None] * u[None] + EY[:, None, None] * v[None]
        usq = u * u + v * v
        return WQ[:, None, None] * rho[None] * (
            1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq[None] / CS2
        )

    def _moments(self, f, Fx=None, Fy=None):
        rho = f.sum(axis=0)
        u = np.tensordot(EX, f, axes=(0, 0))
        v = np.tensordot(EY, f, axes=(0, 0))
        if Fx is not None:
            u = u + 0.5 * Fx
            v = v + 0.5 * Fy
        return rho, u, v

    # ------------------------------------------------------------------ #
    @staticmethod
    def _d_dy(a):
        """2nd-order derivative in y with one-sided stencils at fluid edges."""
        g = np.zeros_like(a)
        g[1:-1] = 0.5 * (a[2:] - a[:-2])
        if a.shape[0] >= 6:
            g[1] = -1.5 * a[1] + 2.0 * a[2] - 0.5 * a[3]
            g[-2] = 1.5 * a[-2] - 2.0 * a[-3] + 0.5 * a[-4]
        else:  # pragma: no cover - coarse fallback
            g[1] = a[2] - a[1]
            g[-2] = a[-2] - a[-3]
        return g

    def _d_dx(self, a):
        if self.nx < 3:
            return np.zeros_like(a)
        return 0.5 * (np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1))

    def _velocity_gradient(self, u, v):
        """Velocity-gradient tensor L = grad u on the wall-normal grid."""
        L = np.zeros((self.ny, self.nx, 3, 3))
        L[..., 0, 0] = self._d_dx(u)
        L[..., 0, 1] = self._d_dy(u)
        L[..., 1, 0] = self._d_dx(v)
        L[..., 1, 1] = self._d_dy(v)
        # solid rows carry no information
        L[0] = 0.0
        L[-1] = 0.0
        return L

    # ------------------------------------------------------------------ #
    # conformation tensor
    # ------------------------------------------------------------------ #
    def _upwind_y(self, a, v):
        g = np.zeros_like(a)
        vb = v[..., None, None]
        back = np.concatenate([a[:1], a[:-1]], axis=0)   # a[j-1]
        fwd = np.concatenate([a[1:], a[-1:]], axis=0)    # a[j+1]
        g = np.where(vb >= 0.0, a - back, fwd - a)
        g[0] = 0.0
        g[-1] = 0.0
        return g

    def _upwind_x(self, a, u):
        if self.nx < 3:
            return np.zeros_like(a)
        ub = u[..., None, None]
        back = np.roll(a, 1, axis=-2)
        fwd = np.roll(a, -1, axis=-2)
        return np.where(ub >= 0.0, a - back, fwd - a)

    def advance_conformation(self, C, u, v, L, dt=1.0):
        """Advance ``C`` by ``dt`` (split into ``n_substeps`` explicit sub-steps)."""
        h = dt / self.n_substeps
        lam = self.lam
        a_eq = self.model.coeff_a()
        for _ in range(self.n_substeps):
            if self.upper_convective:
                src = np.einsum("yxia,yxaj->yxij", L, C) + np.einsum(
                    "yxia,yxja->yxij", C, L
                )
            else:  # negative control: drop the upper-convected (stretching) term
                src = np.zeros_like(C)
            conv = -u[..., None, None] * self._upwind_x(C, u) - v[..., None, None] * self._upwind_y(C, v)
            C_star = C + h * (src + conv)
            tr_c = conf.trace3(C_star)
            f_p = self.model.peterlin(tr_c)
            # implicit (unconditionally stable) relaxation of the Peterlin term
            C = (C_star + (h / lam) * a_eq * conf.I3) / (1.0 + (h / lam) * f_p)[
                ..., None, None
            ]
        C[0] = conf.I3
        C[-1] = conf.I3
        if self.projection:
            C, n_clipped = conf.project_implementable(C)
            self.n_clipped_total += n_clipped
        return C

    def polymer_stress(self, C):
        return self.model.stress(C, self.eta_p)

    def polymer_force(self, tau):
        """div(tau_p) on the lattice grid (mean-free in x by construction)."""
        Fx = self._d_dx(tau[..., 0, 0]) + self._d_dy(tau[..., 0, 1])
        Fy = self._d_dx(tau[..., 0, 1]) + self._d_dy(tau[..., 1, 1])
        Fx[0] = Fx[-1] = 0.0
        Fy[0] = Fy[-1] = 0.0
        return Fx, Fy

    # ------------------------------------------------------------------ #
    # lattice Boltzmann step
    # ------------------------------------------------------------------ #
    def collide_stream(self, f, rho, u, v, Fx, Fy):
        cu = EX[:, None, None] * u[None] + EY[:, None, None] * v[None]
        usq = (u * u + v * v)[None]
        feq = WQ[:, None, None] * rho[None] * (
            1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq / CS2
        )
        f_post = f + self.omega * (feq - f)
        # Guo (1993): S_i = (1-w/2) w_i [ (c_i.F - u.F)/cs2 + (c_i.u)(c_i.F)/cs4 ]
        uF = u * Fx + v * Fy
        cF = EX[:, None, None] * Fx[None] + EY[:, None, None] * Fy[None]
        f_post += (1.0 - 0.5 * self.omega) * WQ[:, None, None] * (
            (cF - uF[None]) / CS2 + cu * cF / CS2 ** 2
        )
        return self._stream(f_post)

    def _stream(self, f_post):
        """Streaming + half-way bounce-back on the two solid rows."""
        if self.nx == 1:
            f_new = f_post.copy()
            for i in range(9):
                if EY[i] == 1:
                    f_new[i, 1:-1] = f_post[i, 0:-2]
                elif EY[i] == -1:
                    f_new[i, 1:-1] = f_post[i, 2:]
            for i in range(9):
                if EY[i] == 1:
                    f_new[i, 1] = f_post[OPP[i], 1]
                elif EY[i] == -1:
                    f_new[i, -2] = f_post[OPP[i], -2]
            f_new[:, 0] = 0.0
            f_new[:, -1] = 0.0
            return f_new
        f_new = np.empty_like(f_post)
        for i in range(9):
            sx, sy = int(EX[i]), int(EY[i])
            f_new[i] = np.roll(np.roll(f_post[i], sy, axis=0), sx, axis=-1)
        for i in range(9):
            if EY[i] == 1:
                f_new[i, 1] = f_post[OPP[i], 1]
            elif EY[i] == -1:
                f_new[i, -2] = f_post[OPP[i], -2]
        f_new[:, 0] = 0.0
        f_new[:, -1] = 0.0
        return f_new

    # ------------------------------------------------------------------ #
    def run(self, verbose: bool = False):
        ny, nx = self.ny, self.nx
        f = self._f.copy()
        C = self._C.copy()
        rho = np.ones((ny, nx))
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        Fx = np.full((ny, nx), self.G)   # constant body force (per unit mass)
        Fy = np.zeros((ny, nx))

        if self.wi_limit is not None and self._wi_wall > self.wi_limit:
            raise ViscoelasticDivergenceError(
                f"Wi_wall={self._wi_wall:.4g} exceeds the declared stability limit "
                f"wi_limit={self.wi_limit:.4g} for this configuration "
                f"(ny={self.ny}, beta={self.beta}); refusing to run rather than "
                f"returning a diverged or non-physical field"
            )
        converged = False
        du_inf = np.inf
        steps = 0
        for step in range(1, self.max_steps + 1):
            steps = step
            if self.eta_p > 0.0:
                L = self._velocity_gradient(u, v)
                C = self.advance_conformation(C, u, v, L)
                tau = self.polymer_stress(C)
                Fpx, Fpy = self.polymer_force(tau)
                Fx_total = Fx + Fpx
                Fy_total = Fy + Fpy
            else:                      # pure solvent (beta = 1) fast path
                Fx_total, Fy_total = Fx, Fy

            f = self.collide_stream(f, rho, u, v, Fx_total, Fy_total)
            rho, u_new, v_new = self._moments(f, Fx_total, Fy_total)

            if not np.all(np.isfinite(u_new)) or not np.all(np.isfinite(C)):
                raise ViscoelasticDivergenceError(
                    f"non-finite state at step {step} (Wi_wall={self._wi_wall:.3g}, "
                    f"lam={self.lam:.4g}, ny={ny}); the explicit viscoelastic "
                    f"coupling left its stability envelope"
                )
            if np.max(np.abs(u_new)) > self.divergence_u:
                raise ViscoelasticDivergenceError(
                    f"velocity blew up to max|u|={np.max(np.abs(u_new)):.3g} at step "
                    f"{step} (Wi_wall={self._wi_wall:.3g}); Weissenberg number "
                    f"exceeds the stability limit of the explicit scheme "
                    f"(declared wi_limit={self.wi_limit})"
                )

            if step in self._snapshot_set:
                self._snapshots[step] = {
                    "u": u_new.copy(),
                    "C": C.copy(),
                    "gamma_dot": self._velocity_gradient(u_new, v_new)[..., 0, 1],
                }

            if step % self.check_every == 0:
                # relative change: the same tol then means the same relative
                # accuracy for any centre-line velocity (an *absolute* criterion
                # makes the residual error scale like 1/u_max -- this was
                # measured and is the reason for the relative form)
                scale = float(np.max(np.abs(u_new)))
                du_inf = float(np.max(np.abs(u_new - u))) / max(scale, 1e-30)
                u, v = u_new, v_new
                if du_inf < self.tol:
                    converged = True
                    if verbose:
                        print(f"[viscoelastic] converged at step {step} du={du_inf:.3e}")
                    break
            else:
                u, v = u_new, v_new

        # final measurements
        L = self._velocity_gradient(u, v)
        tau = self.polymer_stress(C)
        tr_c = conf.trace3(C)
        modulus = self.model.peterlin(tr_c)
        N1 = (self.eta_p / self.lam) * modulus * (C[..., 0, 0] - C[..., 1, 1])
        gamma_dot = L[..., 0, 1]      # du/dy (negative for +x flow, y>0)
        Q = float(np.sum(u, axis=0).mean())
        return ChannelResult(
            u=u.copy(),
            y=self.y.copy(),
            C=C.copy(),
            tau_p=tau.copy(),
            N1=N1.copy(),
            trC=tr_c.copy(),
            gamma_dot=gamma_dot.copy(),
            steps=steps,
            converged=converged,
            du_inf=du_inf,
            H=self.H,
            G=self.G,
            nu0=self.nu0,
            nu_s=self.nu_s,
            eta_p=self.eta_p,
            beta=self.beta,
            lam=self.lam,
            model=self.model_name,
            wi_wall=self._wi_wall,
            re=float(np.max(np.abs(u)) * self.H / self.nu0),
            u_max=float(np.max(np.abs(u))),
            Q=Q,
            n_clipped=self.n_clipped_total,
            projection=self.projection,
            upper_convective=self.upper_convective,
            meta={"snapshots": self._snapshots},
        )

    # analytic reference (convenience)
    def analytic(self):
        kw = {}
        if self.model_name == "fene_p":
            kw["L2"] = self.model.L2
        return conf.analytic_channel(
            self.y[1:-1],
            self.H,
            self.G,
            self.nu0,
            self.lam,
            beta=self.beta,
            model=self.model_name,
            **kw,
        )