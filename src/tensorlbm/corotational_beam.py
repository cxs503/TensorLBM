"""Corotational 2-node planar beam finite element with Newmark-beta time
integration and a full Newton-Raphson nonlinear solver.

Motivation
----------
TensorLBM needs a structural sub-solver for slender-body fluid-structure
interaction (inverted flag / filament benchmarks, hydroelastic appendages).
The reference implementation for this class of solver in the LBM-FSI
literature is ``joconnor22/LIFE`` (Lattice Boltzmann - Immersed Boundary -
Finite Element, GPLv3), whose flexible bodies are integrated with a
*corotational* 2-node beam element plus an implicit Newmark-beta scheme solved
by Newton-Raphson iterations (see ``docs/XVAL-LBM-COROTATIONAL-2NODE-BEAM-
NEWMARK-NEWTON.md`` for the file:line inventory of that implementation).

This module provides two formulations behind one API so they can be compared
arm-for-arm inside the same integrator (single changed variable):

``formulation="consistent"`` (TensorLBM arm)
    The element is derived from a strain energy

        U = EA/(2 L0) (L - L0)^2
            + EI/L0 (2 th1^2 + 2 th1 th2 + 2 th2^2),
        th_i = theta_i - alpha   (nodal rotation minus chord angle),

    with the internal force vector ``g = dU/dq`` and the *exact* Hessian
    ``K_T = dg/dq`` used as the tangent.  Because ``U`` depends on the
    deformed length only through the chord angle ``alpha``, the energy
    gradient puts the transverse nodal force at ``(M1 + M2) / L`` -- the
    *current* element length.

``formulation="life"`` (OSS transplant)
    Faithful port of ``LIFE/src/FEMElement.cpp`` (forceVector / stiffMatrix /
    setLocalMatrices, lines 71-150 and 205-260) and of the Newmark/Newton
    machinery of ``LIFE/src/FEMBody.cpp`` (setNewmark / finishNewmark /
    newtonRaphsonDynamic, lines 40-160).  Its transverse nodal force is
    ``(M1 + M2) / L0`` (initial length) and its tangent is
    ``K_L + K_NL`` in the corotational frame.  This residual is *not* the
    gradient of the energy above once ``L != L0``; the size of that
    inconsistency is measured in the cross-validation report.

Both arms share the identical mass matrix, the identical Newmark-beta update
and the identical Newton outer loop, so any difference in the KPIs is
attributable to the element formulation alone.

Conventions
-----------
* Degrees of freedom per node: ``[u_x, u_y, theta]`` (theta in radians,
  measured from the +x axis, same convention as the chord angle).
* Element local order: ``[u_x1, u_y1, th1, u_x2, u_y2, th2]``.
* ``alpha`` = chord angle ``atan2(y2 - y1, x2 - x1)``; ``th_i`` is wrapped to
  ``(-pi, pi]``.
* Newmark: ``beta = 0.25``, ``gamma = 0.5`` by default (constant average
  acceleration; unconditionally stable, second order, no algorithmic
  damping).  ``LIFE/inc/params.h:76-77`` uses ``alpha = 0.25`` /
  ``delta = 0.5``, i.e. the same pair.
* Consistent mass matrix (``rho A L0 / 420`` coefficients), as
  ``LIFE/src/FEMElement.cpp::setLocalMatrices``.

Units are arbitrary but must be consistent (SI recommended: m, kg, s, N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "CorotationalBeam2D",
    "NewtonRecord",
    "euler_bernoulli_frequency",
    "wrap_angle",
]


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi] (cf. ``Utils::shiftAngle`` in LIFE)."""
    return np.arctan2(np.sin(a), np.cos(a))


def euler_bernoulli_frequency(n_modes: int) -> np.ndarray:
    """Dimensionless eigenvalues ``(kL)^2`` of a clamped-free Euler-Bernoulli
    beam from ``cos(kL) cosh(kL) = -1`` (roots found by bisection, no
    tabulated constants are hard-coded).

    ``omega_i = (kL)_i ** 2 * sqrt(E I / (rho A L**4))``.
    """
    roots = []
    # The n-th root lies in ((2n-1)pi/2, (2n+1)pi/2).
    for n in range(1, n_modes + 1):
        lo = (2 * n - 1) * np.pi / 2.0 + 1e-12
        hi = (2 * n + 1) * np.pi / 2.0 - 1e-12
        f = lambda x: np.cos(x) * np.cosh(x) + 1.0  # noqa: E731
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))
    return np.asarray(roots) ** 2


@dataclass
class NewtonRecord:
    """Diagnostics of one Newton solve (static step or time step)."""

    iterations: int
    residuals: list = field(default_factory=list)
    converged: bool = True
    floor_limited: bool = False

    @property
    def rate(self) -> float:
        """Observed asymptotic convergence rate ``log(r_{k+1}/r_k)/log(r_k/r_{k-1})``.

        ~2.0 for a quadratic (exact-tangent) iteration, ~1.0 for a linear one.
        Returns ``nan`` if fewer than 3 residuals are available.
        """
        r = np.asarray([x for x in self.residuals if x > 0.0])
        if r.size < 3:
            return float("nan")
        e = np.log(r)
        num = e[-1] - e[-2]
        den = e[-2] - e[-3]
        if den == 0.0:
            return float("nan")
        return float(num / den)


class CorotationalBeam2D:
    """2-node corotational planar beam chain (Euler-Bernoulli, small strain,
    large displacement / large rotation).

    Parameters
    ----------
    nodes : (n_node, 2) array_like
        Initial (undeformed) nodal coordinates, ordered along the chain.
    E, A, I, rho : float
        Young's modulus, cross-section area, second moment of area and
        density.  ``I`` may be an array (per-element) but the scalar case is
        what the benchmarks use.
    formulation : {"consistent", "life"}
        Element formulation, see module docstring.
    shear : bool
        Shear (Timoshenko) flexibility is not implemented; kept False.  Both
        LIFE and this module use Euler-Bernoulli kinematics, which is the
        right limit for the slender filaments considered here
        (L/h >= 100 for the LIFE inverted-flag case).
    """

    def __init__(
        self,
        nodes: Sequence[Sequence[float]],
        *,
        E: float,
        A: float,
        I: float,
        rho: float,
        formulation: str = "consistent",
        bc: Iterable[tuple[int, str]] | None = None,
        gravity: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        x0 = np.asarray(nodes, dtype=float)
        if x0.ndim != 2 or x0.shape[1] != 2:
            raise ValueError("nodes must be an (n_node, 2) array")
        if x0.shape[0] < 2:
            raise ValueError("need at least 2 nodes")
        if formulation not in ("consistent", "life"):
            raise ValueError("formulation must be 'consistent' or 'life'")

        self.x0 = x0.copy()
        self.E = float(E)
        self.A = float(A)
        self.I = float(I)
        self.rho = float(rho)
        self.formulation = formulation
        self.gravity = np.asarray(gravity, dtype=float)  # acceleration (m/s^2)

        self.n_node = x0.shape[0]
        self.n_elem = self.n_node - 1
        self.n_dof = 3 * self.n_node

        d = np.diff(x0, axis=0)
        self.L0 = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.L0 <= 0.0):
            raise ValueError("zero-length element")
        self.alpha0 = np.arctan2(d[:, 1], d[:, 0])

        # Node reference angles: chord angle of the adjacent element (mean of
        # both chords for interior nodes), so that the undeformed local nodal
        # rotations are identically zero -- same convention as
        # LIFE/src/FEMBody.cpp::updateFEMValues (node->angle0).
        self.node_angle0 = np.empty(self.n_node)
        self.node_angle0[0] = self.alpha0[0]
        self.node_angle0[-1] = self.alpha0[-1]
        for i in range(1, self.n_node - 1):
            self.node_angle0[i] = wrap_angle(
                0.5
                * (
                    wrap_angle(self.alpha0[i] - self.alpha0[i - 1])
                    + 2.0 * self.alpha0[i - 1]
                )
            )

        # Mass matrix (consistent, identical in both arms).
        self.M = self._assemble_mass()

        # Boundary conditions: list of (node, component) with component
        # in {"x", "y", "theta"}.  Stored as constrained global DOF indices.
        comp = {"x": 0, "y": 1, "theta": 2, "rot": 2, "ux": 0, "uy": 1}
        self.fixed_dofs: list[int] = []
        if bc:
            for node, c in bc:
                self.fixed_dofs.append(3 * int(node) + comp[c])
        self.fixed_dofs = sorted(set(self.fixed_dofs))
        self.free_dofs = np.setdiff1d(np.arange(self.n_dof), self.fixed_dofs)

        self.L_ref = float(self.L0.sum())
        # Relative residual floor set by double precision in the axial force
        # N = EA/L0 * (L - L0): the length difference is formed by cancellation
        # of O(L) coordinates, so |N| carries an absolute error ~ eps*EA.
        self.roundoff_floor = 1e-6

    # ------------------------------------------------------------------
    # element kinematics
    # ------------------------------------------------------------------
    def _elem_kinematics(self, q: np.ndarray):
        """Return per-element geometry and local rotations for DOF vector q."""
        x = self.x0[:, 0] + q[0::3]
        y = self.x0[:, 1] + q[1::3]
        th = self.node_angle0 + q[2::3]
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        L = np.hypot(dx, dy)
        alpha = np.arctan2(dy, dx)
        c = dx / L
        s = dy / L
        th1 = wrap_angle(th[:-1] - alpha)
        th2 = wrap_angle(th[1:] - alpha)
        return L, c, s, alpha, th1, th2

    # ------------------------------------------------------------------
    # internal force
    # ------------------------------------------------------------------
    def internal_forces(self, q: np.ndarray) -> np.ndarray:
        """Global internal force vector ``g(q)`` (n_dof,)."""
        q = np.asarray(q, dtype=float)
        L, c, s, alpha, th1, th2 = self._elem_kinematics(q)
        EA_L0 = self.E * self.A / self.L0
        if self.formulation == "consistent":
            u = L - self.L0
        else:  # LIFE/src/FEMElement.cpp:73 -- algebraically identical, kept verbatim
            u = (L * L - self.L0 * self.L0) / (L + self.L0)
        N = EA_L0 * u
        kb = 2.0 * self.E * self.I / self.L0
        M1 = kb * (2.0 * th1 + th2)
        M2 = kb * (th1 + 2.0 * th2)
        den = L if self.formulation == "consistent" else self.L0
        V = (M1 + M2) / den

        # local -> global: axial along (c, s), transverse along (-s, c)
        g = np.zeros(self.n_dof)
        idx = np.arange(self.n_elem)
        n1 = idx
        n2 = idx + 1
        # local -> global: axial along t_hat = (c, s), transverse along
        # n_hat = (-s, c); node 1 carries (-N, V) and node 2 (+N, -V)
        np.add.at(g, 3 * n1 + 0, -N * c - V * s)
        np.add.at(g, 3 * n1 + 1, -N * s + V * c)
        np.add.at(g, 3 * n2 + 0, N * c + V * s)
        np.add.at(g, 3 * n2 + 1, N * s - V * c)
        np.add.at(g, 3 * n1 + 2, M1)
        np.add.at(g, 3 * n2 + 2, M2)
        return g

    def strain_energy(self, q: np.ndarray) -> float:
        """Strain energy of the *consistent* element family (0 for "life" is
        undefined; this helper always evaluates the energy of the
        energy-consistent model, used to verify ``g = dU/dq``)."""
        q = np.asarray(q, dtype=float)
        L, c, s, alpha, th1, th2 = self._elem_kinematics(q)
        U_ax = 0.5 * self.E * self.A / self.L0 * (L - self.L0) ** 2
        U_b = self.E * self.I / self.L0 * (2.0 * th1**2 + 2.0 * th1 * th2 + 2.0 * th2**2)
        return float(np.sum(U_ax + U_b))

    # ------------------------------------------------------------------
    # tangent stiffness
    # ------------------------------------------------------------------
    def tangent_stiffness(self, q: np.ndarray) -> np.ndarray:
        """Global tangent stiffness ``dg/dq`` (n_dof, n_dof), dense."""
        q = np.asarray(q, dtype=float)
        L, c, s, alpha, th1, th2 = self._elem_kinematics(q)
        K = np.zeros((self.n_dof, self.n_dof))
        if self.formulation == "life":
            return self._tangent_life(q, L, c, s, th1, th2, K)

        EA_L0 = self.E * self.A / self.L0
        N = EA_L0 * (L - self.L0)
        kb = 2.0 * self.E * self.I / self.L0
        M1 = kb * (2.0 * th1 + th2)
        M2 = kb * (th1 + 2.0 * th2)

        for e in range(self.n_elem):
            ea = EA_L0[e]
            kbe = kb[e]
            a = np.zeros(6)
            a[0], a[1], a[3], a[4] = -c[e], -s[e], c[e], s[e]
            # d(alpha)/dq
            da = np.zeros(6)
            da[0], da[1], da[3], da[4] = s[e] / L[e], -c[e] / L[e], -s[e] / L[e], c[e] / L[e]
            b1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) - da
            b2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) - da
            dN = ea * a
            dM1 = kbe * (2.0 * b1 + b2)
            dM2 = kbe * (b1 + 2.0 * b2)

            # d2L/dq2 = (1/L)(P - a a^T) on the translational block
            P = np.zeros((6, 6))
            for (i, j, v) in ((0, 0, 1.0), (1, 1, 1.0), (3, 3, 1.0), (4, 4, 1.0),
                              (0, 3, -1.0), (3, 0, -1.0), (1, 4, -1.0), (4, 1, -1.0)):
                P[i, j] = v
            H_L = (P - np.outer(a, a)) / L[e]

            # d2(alpha)/dq2 (4x4 translational block)
            hu = np.array(
                [[2.0 * c[e] * s[e], -1.0 + 2.0 * s[e] ** 2],
                 [-1.0 + 2.0 * s[e] ** 2, -2.0 * c[e] * s[e]]]
            ) / L[e] ** 2
            H_alpha = np.zeros((6, 6))
            for (i, j, blk) in ((0, 0, hu[0, 0]), (0, 1, hu[0, 1]),
                                (1, 0, hu[1, 0]), (1, 1, hu[1, 1])):
                H_alpha[i, j] = blk
                H_alpha[3 + i, 3 + j] = blk
                H_alpha[i, 3 + j] = -blk
                H_alpha[3 + i, j] = -blk

            Ke = (
                np.outer(a, dN)
                + N[e] * H_L
                + np.outer(b1, dM1)
                + M1[e] * (-H_alpha)
                + np.outer(b2, dM2)
                + M2[e] * (-H_alpha)
            )
            dofs = self._elem_dofs(e)
            K[np.ix_(dofs, dofs)] += Ke
        return K

    def _tangent_life(self, q, L, c, s, th1, th2, K):
        """Tangent of the LIFE transplant: ``T^T (K_L + K_NL) T``
        (LIFE/src/FEMElement.cpp::stiffMatrix, lines 114-150)."""
        g_elem = self._elem_local_forces(L, th1, th2)
        for e in range(self.n_elem):
            L0 = self.L0[e]
            KL = np.zeros((6, 6))
            EA_L0 = self.E * self.A / L0
            EI = self.E * self.I
            KL[0, 0] = KL[3, 3] = EA_L0
            KL[0, 3] = KL[3, 0] = -EA_L0
            KL[1, 1] = KL[4, 4] = 12.0 * EI / L0**3
            KL[1, 4] = KL[4, 1] = -12.0 * EI / L0**3
            KL[1, 2] = KL[2, 1] = 6.0 * EI / L0**2
            KL[1, 5] = KL[5, 1] = 6.0 * EI / L0**2
            KL[4, 2] = KL[2, 4] = -6.0 * EI / L0**2
            KL[4, 5] = KL[5, 4] = -6.0 * EI / L0**2
            KL[2, 2] = KL[5, 5] = 4.0 * EI / L0
            KL[2, 5] = KL[5, 2] = 2.0 * EI / L0

            F = g_elem[e]
            F0 = -F[0]
            V0 = F[4]
            KNL = np.zeros((6, 6))
            KNL[0, 1] = -V0 / L0
            KNL[0, 4] = V0 / L0
            KNL[1, 0] = -V0 / L0
            KNL[1, 1] = F0 / L0
            KNL[1, 3] = V0 / L0
            KNL[1, 4] = -F0 / L0
            KNL[3, 1] = V0 / L0
            KNL[3, 4] = -V0 / L0
            KNL[4, 0] = V0 / L0
            KNL[4, 1] = -F0 / L0
            KNL[4, 3] = -V0 / L0
            KNL[4, 4] = F0 / L0

            T = np.zeros((6, 6))
            T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = c[e]
            T[0, 1] = T[3, 4] = s[e]
            T[1, 0] = T[4, 3] = -s[e]
            T[2, 2] = T[5, 5] = 1.0
            Ke = T.T @ (KL + KNL) @ T
            # LIFE applies the (approximate) tangent on the global dofs; the
            # transformation already carries the corotation.
            dofs = self._elem_dofs(e)
            K[np.ix_(dofs, dofs)] += Ke
        return K

    def _elem_local_forces(self, L, th1, th2) -> np.ndarray:
        """Per-element local internal force vectors (n_elem, 6)."""
        EA_L0 = self.E * self.A / self.L0
        u = L - self.L0
        N = EA_L0 * u
        kb = 2.0 * self.E * self.I / self.L0
        M1 = kb * (2.0 * th1 + th2)
        M2 = kb * (th1 + 2.0 * th2)
        den = L if self.formulation == "consistent" else self.L0
        V = (M1 + M2) / den
        F = np.zeros((self.n_elem, 6))
        F[:, 0] = -N
        F[:, 1] = V
        F[:, 2] = M1
        F[:, 3] = N
        F[:, 4] = -V
        F[:, 5] = M2
        return F

    def _elem_dofs(self, e: int) -> np.ndarray:
        return np.array([3 * e, 3 * e + 1, 3 * e + 2, 3 * e + 3, 3 * e + 4, 3 * e + 5])

    # ------------------------------------------------------------------
    # mass
    # ------------------------------------------------------------------
    def _assemble_mass(self) -> np.ndarray:
        M = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elem):
            L0 = self.L0[e]
            C1 = self.rho * self.A * L0 / 420.0
            Me = np.zeros((6, 6))
            Me[0, 0] = Me[3, 3] = C1 * 140.0
            Me[0, 3] = Me[3, 0] = C1 * 70.0
            Me[1, 1] = Me[4, 4] = C1 * 156.0
            Me[1, 4] = Me[4, 1] = C1 * 54.0
            Me[1, 2] = Me[2, 1] = C1 * 22.0 * L0
            Me[1, 5] = Me[5, 1] = C1 * (-13.0 * L0)
            Me[4, 2] = Me[2, 4] = C1 * 13.0 * L0
            Me[4, 5] = Me[5, 4] = C1 * (-22.0 * L0)
            Me[2, 2] = Me[5, 5] = C1 * 4.0 * L0**2
            Me[2, 5] = Me[5, 2] = C1 * (-3.0 * L0**2)
            dofs = self._elem_dofs(e)
            M[np.ix_(dofs, dofs)] += Me
        return M

    @property
    def total_mass(self) -> float:
        return float(self.rho * self.A * self.L0.sum())

    # ------------------------------------------------------------------
    # external loads
    # ------------------------------------------------------------------
    def gravity_load(self) -> np.ndarray:
        """Consistent distributed load from a constant body acceleration."""
        f = np.zeros(self.n_dof)
        gx, gy = self.gravity
        for e in range(self.n_elem):
            L0 = self.L0[e]
            C = self.rho * self.A * L0
            fe = np.zeros(6)
            fe[0] = fe[3] = gx * C / 2.0
            fe[1] = fe[4] = gy * C / 2.0
            fe[2] = gy * C * L0 / 12.0
            fe[5] = -gy * C * L0 / 12.0
            dofs = self._elem_dofs(e)
            f[dofs] += fe
        return f

    def nodal_load(self, loads: dict[int, Sequence[float]]) -> np.ndarray:
        """Point loads / moments: ``{node: (fx, fy, mz)}``."""
        f = np.zeros(self.n_dof)
        for node, vals in loads.items():
            vals = list(vals) + [0.0] * (3 - len(vals))
            f[3 * node : 3 * node + 3] += np.asarray(vals[:3], dtype=float)
        return f

    def external_load(self, loads: dict[int, Sequence[float]] | None = None) -> np.ndarray:
        f = self.gravity_load()
        if loads:
            f += self.nodal_load(loads)
        return f

    # ------------------------------------------------------------------
    # linear helpers
    # ------------------------------------------------------------------
    def linear_stiffness(self) -> np.ndarray:
        """Tangent at the undeformed state (small-displacement stiffness)."""
        return self.tangent_stiffness(np.zeros(self.n_dof))

    def mode_frequencies(self, n_modes: int = 4) -> np.ndarray:
        """Circular frequencies (rad/s) of the *linearised* model with BCs.

        Solves ``K phi = omega^2 M phi`` by Cholesky reduction of the
        (symmetric positive definite) consistent mass matrix to a standard
        symmetric eigenproblem, ``M = L L^T``, ``(L^-1 K L^-T) psi = mu psi``.
        ``numpy.linalg.eigh`` is used deliberately: the control node's
        flexiblas/OpenBLAS build segfaults inside ``scipy.linalg.eigh`` for
        generalised problems beyond ~40 DOFs (see the friction log of the
        cross-validation report), while numpy's LAPACK path is stable.
        """
        K = self.linear_stiffness()
        fr = self.free_dofs
        Kr = K[np.ix_(fr, fr)]
        Mr = self.M[np.ix_(fr, fr)]
        Lc = np.linalg.cholesky(Mr)
        Li = np.linalg.solve(Lc, np.eye(Lc.shape[0]))
        Ks = Li @ Kr @ Li.T
        Ks = 0.5 * (Ks + Ks.T)
        vals = np.linalg.eigvalsh(Ks)
        vals = np.sort(vals[vals > 0.0])[:n_modes]
        return np.sqrt(vals)

    # ------------------------------------------------------------------
    # Newton machinery
    # ------------------------------------------------------------------
    def _reduced_solve(self, K: np.ndarray, r: np.ndarray) -> np.ndarray:
        fr = self.free_dofs
        dq = np.zeros(self.n_dof)
        dq[fr] = np.linalg.solve(K[np.ix_(fr, fr)], -r[fr])
        dq[self.fixed_dofs] = 0.0
        return dq

    def solve_static(
        self,
        loads: dict[int, Sequence[float]] | None = None,
        *,
        n_steps: int = 1,
        maxit: int = 30,
        tol: float = 1e-10,
        record: bool = False,
        q0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, NewtonRecord]:
        """Incremental-load Newton-Raphson static solution.

        ``n_steps`` load increments are applied; if a step does not converge the
        increment is halved (reported through ``NewtonRecord.iterations`` /
        ``converged``) -- failure is *reported*, never silently retried into a
        different answer.
        """
        f_full = self.external_load(loads)
        q = np.zeros(self.n_dof) if q0 is None else np.array(q0, dtype=float)
        rec = NewtonRecord(iterations=0, residuals=[], converged=True)
        lam, dlam = 0.0, 1.0 / float(n_steps)
        while lam < 1.0 - 1e-15:
            target = min(1.0, lam + dlam)
            q_try, it, res = self._newton_static_step(q, target * f_full, maxit, tol)
            if not res.converged:
                dlam *= 0.5
                rec.converged = False
                if dlam < 1e-6:
                    raise RuntimeError("static load stepping stalled")
                continue
            q = q_try
            lam = target
            rec.iterations += it
            rec.residuals.extend(res.residuals)
            if rec.converged is False and dlam >= 1.0 / n_steps:
                rec.converged = True  # recovered
        if record:
            rec.residuals = res.residuals
        return q, rec

    def _newton_static_step(self, q, f, maxit, tol):
        rec = NewtonRecord(iterations=0, residuals=[], converged=False)
        fnorm = np.linalg.norm(f[self.free_dofs]) or 1.0
        prev = np.inf
        for _ in range(maxit):
            g = self.internal_forces(q)
            r = g - f
            rn = np.linalg.norm(r[self.free_dofs]) / fnorm
            rec.residuals.append(float(rn))
            if rn < tol:
                rec.converged = True
                break
            if rn < self.roundoff_floor and rn > 0.5 * prev:
                # stagnating at the double-precision floor of the axial force
                # (eps * EA / L0); reported, not hidden
                rec.converged = True
                rec.floor_limited = True
                break
            prev = rn
            K = self.tangent_stiffness(q)
            dq = self._reduced_solve(K, r)
            q = q + dq
            rec.iterations += 1
        else:  # pragma: no cover - exhausted iterations
            g = self.internal_forces(q)
            rn = np.linalg.norm((g - f)[self.free_dofs]) / fnorm
            rec.residuals.append(float(rn))
            rec.converged = rn < tol
        return q, rec.iterations, rec

    # ------------------------------------------------------------------
    # Newmark-beta + Newton
    # ------------------------------------------------------------------
    def solve_newmark(
        self,
        *,
        dt: float,
        n_steps: int,
        loads: Callable[[float], dict[int, Sequence[float]]] | None = None,
        beta: float = 0.25,
        gamma: float = 0.5,
        maxit: int = 20,
        tol: float = 1e-10,
        q0: np.ndarray | None = None,
        qdot0: np.ndarray | None = None,
        qddot0: np.ndarray | None = None,
        predictor: bool = True,
        sample: Callable[[int, float, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
    ) -> dict:
        """Implicit Newmark-beta integration with Newton-Raphson iterations.

        The algorithm is the one of ``LIFE/src/FEMBody.cpp`` (setNewmark /
        newtonRaphsonDynamic / finishNewmark + 2nd order predictor), re-used
        unchanged by both formulations.
        """
        a0 = 1.0 / (beta * dt * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a9 = dt * (1.0 - gamma)
        a10 = gamma * dt

        q = np.zeros(self.n_dof) if q0 is None else np.array(q0, dtype=float)
        qdot = np.zeros(self.n_dof) if qdot0 is None else np.array(qdot0, dtype=float)
        qddot = np.zeros(self.n_dof) if qddot0 is None else np.array(qddot0, dtype=float)

        records: list[NewtonRecord] = []
        hist_q, hist_t = [], []

        for step in range(1, n_steps + 1):
            t = step * dt
            f_ext = self.external_load(loads(t - dt) if loads else None)
            q_n, qdot_n, qddot_n = q.copy(), qdot.copy(), qddot.copy()

            if predictor:
                q = self._predict(q, qdot, qddot, dt)

            rec = NewtonRecord(iterations=0, residuals=[], converged=False)
            fnorm = np.linalg.norm(f_ext[self.free_dofs]) or 1.0
            prev = np.inf
            for _ in range(maxit):
                g = self.internal_forces(q)
                r = (
                    f_ext
                    - g
                    - self.M @ (a0 * (q - q_n) - a2 * qdot_n - a3 * qddot_n)
                )
                rn = np.linalg.norm(r[self.free_dofs]) / fnorm
                rec.residuals.append(float(rn))
                if rn < tol:
                    rec.converged = True
                    break
                if rn < self.roundoff_floor and rn > 0.5 * prev:
                    rec.converged = True
                    rec.floor_limited = True
                    break
                prev = rn
                K = self.tangent_stiffness(q)
                Kef = K + a0 * self.M
                dq = self._reduced_solve(Kef, -r)
                q = q + dq
                rec.iterations += 1
            records.append(rec)

            qddot = a0 * (q - q_n) - a2 * qdot_n - a3 * qddot_n
            qdot = qdot_n + a9 * qddot_n + a10 * qddot
            hist_q.append(q.copy())
            hist_t.append(t)
            if sample is not None:
                sample(step, t, q, qdot, qddot)

        return {
            "q": q,
            "qdot": qdot,
            "qddot": qddot,
            "t": np.asarray(hist_t),
            "q_hist": np.asarray(hist_q),
            "newton": records,
            "converged": all(r.converged for r in records),
            "max_iterations": max((r.iterations for r in records), default=0),
        }

    def _predict(self, q, qdot, qddot, dt) -> np.ndarray:
        """Constant-acceleration predictor (Newmark displacement predictor)."""
        return q + dt * qdot + 0.5 * dt * dt * qddot

    # ------------------------------------------------------------------
    # energy of the discrete system (used for conservation checks)
    # ------------------------------------------------------------------
    def kinetic_energy(self, qdot: np.ndarray) -> float:
        return float(0.5 * qdot @ self.M @ qdot)

    def potential_energy(self, q: np.ndarray, loads=None) -> float:
        """Strain energy minus work of the conservative external load (only
        the *consistent* family has a well-defined strain energy)."""
        f = self.external_load(loads)
        return self.strain_energy(q) - float(f @ q)

    # ------------------------------------------------------------------
    # reporting helpers
    # ------------------------------------------------------------------
    def node_position(self, q: np.ndarray, node: int = -1) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        i = node if node >= 0 else self.n_node + node
        return self.x0[i] + q[3 * i : 3 * i + 2]

    def tip_deflection(self, q: np.ndarray) -> np.ndarray:
        return self.node_position(q, -1) - self.x0[-1]

    def tip_rotation(self, q: np.ndarray) -> float:
        return float(q[3 * (self.n_node - 1) + 2])


def straight_beam(
    length: float,
    n_elem: int,
    *,
    E: float,
    A: float,
    I: float,
    rho: float,
    angle: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    **kwargs,
) -> CorotationalBeam2D:
    """Convenience constructor: straight beam of ``n_elem`` equal elements."""
    s = np.linspace(0.0, length, n_elem + 1)
    nodes = np.column_stack([x0 + s * np.cos(angle), y0 + s * np.sin(angle)])
    return CorotationalBeam2D(nodes, E=E, A=A, I=I, rho=rho, **kwargs)