"""Hadamard-Rybczynski fluid-sphere drag: analytic law + axisymmetric Stokes
drag solver used to verify it.

Physical problem
----------------
A spherical fluid drop (viscosity ``mu_d``, density ``rho_d``) settles at steady
velocity ``U`` through a quiescent fluid (``mu_c``, ``rho_c``) at creeping
(Reynolds -> 0) flow.  Hadamard (1911) and Rybczynski (1911) solved the same
boundary-value problem that is verified here: one continuous velocity field
over both phases, viscosity ratio ``lambda = mu_d/mu_c`` across the spherical
interface, continuity of velocity and of tangential traction, kinematic
condition ``u_r = 0`` on the (undeformed) interface, Stokes momentum balance.
Their result::

    F = 2 pi mu_c R U (2 + 3 lambda) / (1 + lambda)                      (1)
    U = (2/3) (drho g) R^2 (1 + lambda) / (mu_c (2 + 3 lambda))          (2)

with the classical limits

    lambda -> 0    : F = 4 pi mu_c R U   (Rybczynski bubble)             (3)
    lambda -> 1    : F = 5 pi mu_c R U   (equal-viscosity drop)          (4)
    lambda -> inf  : F = 6 pi mu_c R U   (Stokes rigid sphere)           (5)

The interface stays spherical because the HR solution is the creeping-flow
limit at vanishing capillary number; a deforming interface/surfactant physics is
outside the scope of the law itself (see the report for that limitation).

Solver
------
``HRDragSolver`` is an axisymmetric (r, theta) Q1 finite-element Stokes solver
for a fluid sphere that is concentric with an outer spherical wall moving at
``U e_z`` (the drop rest frame -- working in a container with a momentum sink is
mandatory for a buoyancy problem, see the "equivalence principle" finding in
``terminal_verdict_m155.md``).  Stokes is imposed through the Helmholtz
minimum-dissipation principle with a divergence penalty::

    min_u  1/2 int 2 mu eps:eps dV + (Lambda/2) int (div u)^2 dV
    s.t.   u_r = 0 at r = R (interface, u_theta free -> traction continuity is
           natural in the weak form), u = U e_z on the outer sphere,
           u_theta = 0 on the axis.

Both the drop interface and the outer wall lie *on* mesh lines, so the viscosity
jump is sharp and ``lambda`` is not smeared over a diffuse interface.  The drag
follows from the dissipation identity (Helmholtz minimum dissipation /
reciprocal theorem)

    F U = int 2 mu eps:eps dV,                                            (6)

which avoids differentiating the numerical solution at the interface (the naive
traction integral is the noisiest quantity in this problem; (6) is exact for the
discrete minimiser and converges cleanly).

Finite-domain bias: a container of radius ``H = H_ratio * R`` confines the flow.
The classical asymptotic correction for a sphere between concentric boundaries
is ``F(H) = F_inf (1 + c1 R/H + O((R/H)^3))``, so a linear fit in ``R/H`` over
three domains recovers the unbounded ``F_inf`` -- that extrapolation is done
explicitly by ``wall_extrapolate`` and its residual is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CS2 = 1.0 / 3.0

try:  # scipy is a package dependency; import locally nonetheless
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    _SCIPY_OK = True
    _SCIPY_ERR = None
except Exception as exc:  # pragma: no cover
    sparse = None
    spsolve = None
    _SCIPY_OK = False
    _SCIPY_ERR = exc


# ---------------------------------------------------------------------------
# Analytic HR law
# ---------------------------------------------------------------------------
def _check_positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")


def _check_lam(lam: float) -> float:
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam}")
    return lam


def drag_coefficient_factor(lam: float) -> float:
    """Dimensionless HR group ``F / (2 pi mu_c R U) = (2 + 3 lam)/(1 + lam)``."""
    lam = _check_lam(lam)
    return (2.0 + 3.0 * lam) / (1.0 + lam)


def hr_drag(mu_c: float, R: float, U: float, lam: float) -> float:
    """HR drag force on a spherical fluid drop, eq. (1)."""
    _check_positive(mu_c=mu_c, R=R, U=U)
    _check_lam(lam)
    return 2.0 * np.pi * mu_c * R * U * drag_coefficient_factor(lam)


def hr_velocity(drho: float, g: float, R: float, mu_c: float, lam: float) -> float:
    """Steady HR settling velocity, eq. (2) (force balance on the drop)."""
    _check_positive(g=g, R=R, mu_c=mu_c)
    _check_lam(lam)
    return (2.0 / 3.0) * drho * g * R**2 * (1.0 + lam) / (mu_c * (2.0 + 3.0 * lam))


def bubble_limit_drag(mu_c: float, R: float, U: float) -> float:
    """``lambda -> 0`` limit of eq. (1): ``F = 4 pi mu_c R U`` (eq. 3)."""
    _check_positive(mu_c=mu_c, R=R, U=U)
    return 4.0 * np.pi * mu_c * R * U


def equal_viscosity_drag(mu_c: float, R: float, U: float) -> float:
    """``lambda = 1`` value of eq. (1): ``F = 5 pi mu_c R U`` (eq. 4)."""
    _check_positive(mu_c=mu_c, R=R, U=U)
    return 5.0 * np.pi * mu_c * R * U


def stokes_limit_drag(mu_c: float, R: float, U: float) -> float:
    """``lambda -> inf`` limit of eq. (1): ``F = 6 pi mu_c R U`` (eq. 5)."""
    _check_positive(mu_c=mu_c, R=R, U=U)
    return 6.0 * np.pi * mu_c * R * U


def infer_lambda_from_drag(mu_c: float, R: float, U: float, F: float) -> float:
    """Invert eq. (1): recover ``lambda`` from a measured drag force."""
    _check_positive(mu_c=mu_c, R=R, U=U, F=F)
    k = F / (2.0 * np.pi * mu_c * R * U)  # = (2 + 3 lam)/(1 + lam)
    if not (2.0 < k < 3.0):
        raise ValueError(
            f"drag factor {k:.4f} outside (2, 3): not resolvable as a HR drop"
        )
    return (2.0 - k) / (k - 3.0)


def terminal_velocity_from_drag(
    drag: float, drho: float, g: float, R: float, mu_c: float = 1.0
) -> float:
    """Settling velocity from a drag measured at unit velocity scale."""
    _check_positive(drag=drag, g=g, R=R)
    buoyancy = (4.0 / 3.0) * np.pi * R**3 * drho * g
    return buoyancy / drag


# ---------------------------------------------------------------------------
# Axisymmetric Stokes drag solver
# ---------------------------------------------------------------------------
@dataclass
class HRDragResult:
    """Outcome of one finite-domain Stokes drop-drag computation."""

    lam: float
    drag: float                 # force on the drop at U = 1
    n_unknowns: int
    n_elements: int
    n_drop_elements: int
    h_wall: float
    H_ratio: float
    penalty: float
    div_l2: float
    u_max: float
    meta: Dict[str, float] = field(default_factory=dict)

    def drag_factor(self, mu_c: float = 1.0, R: float = 1.0) -> float:
        """``F / (2 pi mu_c R U)`` at ``U = 1``."""
        return self.drag / (2.0 * np.pi * mu_c * R * 1.0)

    def rel_error_vs_hr(self, mu_c: float = 1.0, R: float = 1.0) -> float:
        return self.drag / hr_drag(mu_c, R, 1.0, self.lam) - 1.0


def _geometric_nodes(span: float, h0: float, growth: float) -> np.ndarray:
    """Nodes on ``[0, span]`` with first spacing ``h0`` growing by ``growth``."""
    nodes = [0.0]
    h = h0
    total = 0.0
    while total < span - 1e-12:
        h = min(h, span - total)
        total += h
        nodes.append(total)
        h *= growth
        if len(nodes) > 100000:  # pragma: no cover - safety
            raise RuntimeError("radial grid did not close")
    return np.asarray(nodes)


def _solve_growth(span: float, h0: float, n_elems: int) -> float:
    """Growth ratio so that ``n_elems`` geometric spacings fill ``span``."""
    if n_elems * h0 >= span * (1.0 - 1e-12):
        return 1.0

    def total(q: float, n: int = n_elems) -> float:
        if abs(q - 1.0) < 1e-12:
            return n * h0
        return h0 * (q**n - 1.0) / (q - 1.0)

    lo, hi = 1.0 + 1e-12, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < span:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class HRDragSolver:
    """Axisymmetric Q1 Stokes solver for a fluid sphere in a spherical container.

    Parameters
    ----------
    R : sphere radius (lattice-free units; ``mu_c`` and ``U`` carry the units).
    H_ratio : outer-wall radius / R (finite-domain confinement, extrapolated).
    h_wall : element size at the interface (and at the outer wall), in units of R.
    growth : radial mesh growth ratio away from the interface.
    n_theta : angular elements over ``[0, pi]``; default ties it to ``h_wall``.
    penalty : element-mean divergence penalty factor ``Lambda / mu_c``
        (constraint error ~ 1/penalty).
    n_drop_min : minimum number of radial elements inside the drop.
    """

    def __init__(
        self,
        R: float = 1.0,
        H_ratio: float = 20.0,
        h_wall: float = 0.02,
        growth: float = 1.35,
        n_theta: Optional[int] = None,
        penalty: float = 1.0e6,
        n_drop_min: int = 12,
    ) -> None:
        if not _SCIPY_OK:  # pragma: no cover
            raise RuntimeError(f"scipy required for HRDragSolver: {_SCIPY_ERR}")
        self.R = float(R)
        self.H = float(H_ratio) * self.R
        self.H_ratio = float(H_ratio)
        self.h_wall = float(h_wall) * self.R
        self.growth = float(growth)
        self.penalty = float(penalty)
        self._build_grid(n_theta, n_drop_min)

    # -- grid ---------------------------------------------------------------
    def _build_grid(self, n_theta: Optional[int], n_drop_min: int) -> None:
        R, H, h0 = self.R, self.H, self.h_wall
        # radial: cluster toward the interface from both sides
        q_in = self.growth
        n_in = max(n_drop_min, int(np.ceil(np.log(1.0 + (R / h0) * (q_in - 1.0)) / np.log(q_in))))
        q_in = _solve_growth(R, h0, n_in)
        inner = _geometric_nodes(R, h0, q_in)          # 0 .. R ascending
        inner = R - inner[::-1]                        # ascending, ends at 0
        inner = np.maximum.accumulate(inner)
        inner[0] = 0.0
        n_ann = int(np.ceil(np.log(1.0 + ((H - R) / h0) * (self.growth - 1.0)) / np.log(self.growth)))
        n_ann = max(n_ann, 4)
        q_out = _solve_growth(H - R, h0, n_ann)
        outer = R + _geometric_nodes(H - R, h0, q_out)  # starts at R
        r_nodes = np.concatenate([inner[:-1], outer])
        # drop the duplicate origin/interface nodes and enforce exact R
        self.r = r_nodes
        self.ir = int(np.argmin(np.abs(self.r - R)))
        self.r[self.ir] = R
        self.nr = len(self.r) - 1
        if n_theta is None:
            n_theta = int(np.ceil(np.pi * R / h0))
            n_theta = max(24, n_theta + (n_theta % 2))
        self.ntheta = int(n_theta)
        self.theta = np.linspace(0.0, np.pi, self.ntheta + 1)

    @property
    def n_nodes(self) -> int:
        return (self.nr + 1) * (self.ntheta + 1)

    # -- assembly -----------------------------------------------------------
    def _element_tables(self):
        """Per-element nodal ids, coordinates and mid-radius."""
        r, th = self.r, self.theta
        nr, nt = self.nr, self.ntheta
        i_idx, j_idx = np.meshgrid(np.arange(nr), np.arange(nt), indexing="ij")
        n0 = i_idx * (nt + 1) + j_idx
        n1 = (i_idx + 1) * (nt + 1) + j_idx
        n2 = (i_idx + 1) * (nt + 1) + j_idx + 1
        n3 = i_idx * (nt + 1) + j_idx + 1
        nodes = np.stack([n0, n1, n2, n3], axis=-1)              # (nr, nt, 4)
        rloc = np.stack([r[i_idx], r[i_idx + 1], r[i_idx + 1], r[i_idx]], axis=-1)
        tloc = np.stack([th[j_idx], th[j_idx], th[j_idx + 1], th[j_idx + 1]], axis=-1)
        r_mid = 0.5 * (r[i_idx] + r[i_idx + 1])
        return nodes.reshape(-1, 4), rloc.reshape(-1, 4), tloc.reshape(-1, 4), r_mid.ravel()

    def assemble(self, lam: float, mu_c: float = 1.0):
        """Assemble ``(A_phys, A_pen, ndof)``; both are SPD symmetric.

        Unknowns are the nodal ``(u_r, v)`` with ``v = u_theta / sin(theta)``.
        The substitution is what makes the axis regular: every ``cot(theta)`` /
        ``1/sin(theta)`` combination in the strain and divergence becomes

            (1/r) du_theta/dtheta + u_theta cot(theta)/r = (sin(theta) v_,theta
            + 2 cos(theta) v) / r,

        which is analytic at ``theta = 0, pi``.  Without it the bilinear
        interpolant of ``u_theta`` produces an O(h) spurious divergence at the
        Gauss points nearest the axis (amplified by ``cot(theta) ~ 1/Delta``),
        the penalty then dominates the functional and the minimiser distorts the
        whole flow field -- measured as a ~3.6x drag error before the fix.
        """
        lam = _check_lam(lam)
        nodes, rloc, tloc, r_mid = self._element_tables()
        n_elem = nodes.shape[0]
        drdxi = 0.5 * (rloc[:, 1] - rloc[:, 0])
        dtdeta = 0.5 * (tloc[:, 2] - tloc[:, 0])
        jac = drdxi * dtdeta
        mu_e = np.where(r_mid < self.R - 1e-12, lam * mu_c, mu_c)
        xi_a = np.array([-1.0, 1.0, 1.0, -1.0])
        et_a = np.array([-1.0, -1.0, 1.0, 1.0])
        g = 1.0 / np.sqrt(3.0)
        gp = [(-g, -g), (g, -g), (g, g), (-g, g)]

        dofs = np.empty((n_elem, 8), dtype=np.int64)
        dofs[:, 0::2] = 2 * nodes
        dofs[:, 1::2] = 2 * nodes + 1

        a_phys = np.zeros((n_elem, 8, 8))
        a_pen = np.zeros((n_elem, 8, 8))
        g_div = np.zeros((n_elem, 8))     # element-integrated divergence row
        vol = np.zeros(n_elem)            # element volume (3D, 2 pi r^2 sin t)
        for xi, eta in gp:
            N = 0.25 * (1.0 + xi_a * xi) * (1.0 + et_a * eta)
            dNxi = 0.25 * xi_a * (1.0 + et_a * eta)
            dNeta = 0.25 * et_a * (1.0 + xi_a * xi)
            r_g = N @ rloc.T                       # (n_elem,)
            t_g = N @ tloc.T
            dNdr = dNxi[None, :] / drdxi[:, None]
            dNdt = dNeta[None, :] / dtdeta[:, None]
            dV = 2.0 * np.pi * r_g**2 * np.sin(t_g) * jac
            ir = (1.0 / r_g)[:, None]
            st = np.sin(t_g)[:, None]
            ct = np.cos(t_g)[:, None]
            a_rr = np.zeros((n_elem, 8))
            a_tt = np.zeros((n_elem, 8))
            a_pp = np.zeros((n_elem, 8))
            a_rt = np.zeros((n_elem, 8))
            a_dv = np.zeros((n_elem, 8))
            # u_r part
            a_rr[:, 0::2] = dNdr
            a_tt[:, 0::2] = N * ir
            a_pp[:, 0::2] = N * ir
            a_rt[:, 0::2] = 0.5 * dNdt * ir
            a_dv[:, 0::2] = dNdr + 2.0 * N * ir
            # v = u_theta/sin(theta) part (axis-regular)
            a_tt[:, 1::2] = ir * (ct * N + st * dNdt)
            a_pp[:, 1::2] = ir * ct * N
            a_rt[:, 1::2] = 0.5 * st * (dNdr - N * ir)
            a_dv[:, 1::2] = ir * (st * dNdt + 2.0 * ct * N)
            w2mu = (2.0 * mu_e * dV)[:, None, None]
            a_phys += w2mu * (
                np.einsum("ei,ej->eij", a_rr, a_rr)
                + np.einsum("ei,ej->eij", a_tt, a_tt)
                + np.einsum("ei,ej->eij", a_pp, a_pp)
                + 2.0 * np.einsum("ei,ej->eij", a_rt, a_rt)
            )
            g_div += dV[:, None] * a_dv
            vol += dV
        # reduced-integration (element-mean) divergence penalty:
        #   (Lambda / vol_e) * (int_e div u dV)^2
        # Penalising the *element mean* rather than the pointwise divergence is
        # what keeps the constraint term free of interpolation noise: the mean
        # divergence of a smooth interpolant is O(h^2) (errors cancel inside the
        # element) while the pointwise one is O(h) and, amplified by cot(theta)
        # near the axis, used to dominate the functional.
        scale = (self.penalty * mu_c / vol)[:, None, None]
        a_pen += scale * np.einsum("ei,ej->eij", g_div, g_div)
        self._g_div = g_div
        self._vol = vol

        ndof = 2 * self.n_nodes
        rows = np.repeat(dofs[:, :, None], 8, axis=2).ravel()
        cols = np.repeat(dofs[:, None, :], 8, axis=1).ravel()
        A_phys = sparse.coo_matrix(
            (a_phys.ravel(), (rows, cols)), shape=(ndof, ndof)
        ).tocsr()
        A_pen = sparse.coo_matrix(
            (a_pen.ravel(), (rows, cols)), shape=(ndof, ndof)
        ).tocsr()
        return A_phys, A_pen, ndof

    # -- boundary conditions ------------------------------------------------
    def dirichlet_values(self, U: float = 1.0) -> Dict[int, float]:
        """Dirichlet data for the drop rest frame with outer-wall velocity ``U``.

        Unknowns are ``(u_r, v)`` with ``v = u_theta/sin(theta)``; the axis
        conditions ``u_theta = 0`` are satisfied identically by the
        substitution, so they need no entry here.
        """
        bc: Dict[int, float] = {}
        nt = self.ntheta
        th = self.theta
        ir = self.ir
        # interface: no normal flow; tangential velocity free -> the natural
        # (weak) condition is continuity of tangential traction, i.e. exactly
        # the Hadamard-Rybczynski interface condition.
        for j in range(nt + 1):
            node = ir * (nt + 1) + j
            bc[2 * node] = 0.0
        # outer wall: rigid translation U e_z  ->  u_r = U cos(theta), v = -U
        for j in range(nt + 1):
            node = self.nr * (nt + 1) + j
            bc[2 * node] = U * np.cos(th[j])
            bc[2 * node + 1] = -U
        # drop centre: one geometric point, pinned (its volume weight is O(r^2),
        # so this gauge does not enter the drag; sensitivity is tested)
        for j in range(nt + 1):
            node = j
            bc.setdefault(2 * node, 0.0)
            bc.setdefault(2 * node + 1, 0.0)
        return bc

    # -- solve --------------------------------------------------------------
    def solve(self, lam: float, mu_c: float = 1.0, U: float = 1.0) -> HRDragResult:
        """Solve the Stokes problem and return the drop drag at velocity ``U``."""
        A_phys, A_pen, ndof = self.assemble(lam, mu_c=mu_c)
        A = (A_phys + A_pen).tocsr()
        bc = self.dirichlet_values(U=U)
        d_dofs = np.array(sorted(bc.keys()), dtype=np.int64)
        u_d = np.array([bc[int(d)] for d in d_dofs], dtype=np.float64)
        mask = np.ones(ndof, dtype=bool)
        mask[d_dofs] = False
        f_dofs = np.flatnonzero(mask)
        A_ff = A[f_dofs][:, f_dofs].tocsc()
        A_fd = A[f_dofs][:, d_dofs]
        rhs = -(A_fd @ u_d)
        u_f = spsolve(A_ff, rhs)
        if not np.all(np.isfinite(u_f)):
            raise RuntimeError(f"linear solve failed for lambda={lam}")
        u = np.zeros(ndof)
        u[d_dofs] = u_d
        u[f_dofs] = u_f
        diss = float(u @ (A_phys @ u))
        div_l2 = float(np.sqrt(max(u @ (A_pen @ u), 0.0)))
        return HRDragResult(
            lam=float(lam),
            drag=diss / U,
            n_unknowns=int(ndof),
            n_elements=int(self.nr * self.ntheta),
            n_drop_elements=int(self.ir * self.ntheta),
            h_wall=self.h_wall / self.R,
            H_ratio=self.H_ratio,
            penalty=self.penalty,
            div_l2=div_l2,
            u_max=float(np.max(np.abs(u))),
            meta={"mu_c": mu_c, "U": U, "R": self.R},
        )


# ---------------------------------------------------------------------------
# Wall extrapolation and convergence helpers
# ---------------------------------------------------------------------------
def wall_extrapolate(
    H_ratios: Sequence[float], drags: Sequence[float], order: int = 1
) -> Dict[str, float]:
    """Extrapolate finite-container drag to ``H -> inf`` with ``F = F_inf + b x``.

    ``x = R/H``.  The classical confinement series is linear in ``R/H`` to
    leading order, so a least-squares line over three (or more) domains removes
    the container bias.  Returns the intercept plus the fit residual, so the
    extrapolation is auditable rather than trusted.
    """
    x = 1.0 / np.asarray(H_ratios, dtype=np.float64)
    f = np.asarray(drags, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two containers to extrapolate")
    if order == 1:
        design = np.stack([np.ones_like(x), x], axis=1)
    elif order == 2:
        design = np.stack([np.ones_like(x), x, x**2], axis=1)
    else:
        raise ValueError("order must be 1 or 2")
    coef, *_ = np.linalg.lstsq(design, f, rcond=None)
    fit = design @ coef
    resid = f - fit
    return {
        "drag_inf": float(coef[0]),
        "slope": float(coef[1]),
        "max_rel_residual": float(np.max(np.abs(resid / f))),
        "x": x.tolist(),
        "drag": f.tolist(),
    }


def richardson_order(coarse: float, medium: float, fine: float, ratio: float = 2.0) -> float:
    """Observed convergence order from three successive refinements."""
    d1 = medium - coarse
    d2 = fine - medium
    if d1 == 0.0 or d2 == 0.0:
        return float("nan")
    return float(np.log(abs(d1 / d2)) / np.log(ratio))


def hr_lambda_ladder(n: int = 8, lo: float = 0.01, hi: float = 100.0) -> np.ndarray:
    """Eight log-spaced viscosity ratios over ``[lo, hi]`` (endpoints included)."""
    if n < 2:
        raise ValueError("need n >= 2 points")
    return np.geomspace(lo, hi, n)