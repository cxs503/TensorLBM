"""Automatic (symbolic) Chapman--Enskog derivation of LBM transport coefficients.

Why this module exists
----------------------
The repository states its viscosity law in many places (``tau = 0.5 + 3 nu``,
``nu = c_s^2 (tau - 1/2)``) but every occurrence is *hand typed* for one lattice
and one collision model (:mod:`tensorlbm.von_neumann.equivalent_equations`
hard-codes ``cs2 = 1/3`` and BGK ``tau``; :mod:`tensorlbm.solver` hard-codes the
D2Q9 MRT matrices).  Nothing in the tree can answer, from first principles and
without editing code:

    "for *this* stencil with *these* weights and *that* collision operator, which
     kinematic shear viscosity ``nu`` and bulk viscosity ``zeta`` does the scheme
     actually implement at second order in the Knudsen number?"

That question must be answered *before* a benchmark can claim that a measured
decay rate agrees with theory, and it must stay correct when a MRT/TRT rate
vector is retuned.  This module derives the answer symbolically.

Method (see :func:`derive_equivalent_equation`)
-----------------------------------------------
Two-scale Chapman--Enskog expansion of the lattice Boltzmann equation
(Krueger et al., *The Lattice Boltzmann Method: Principles and Practice*,
Springer 2017, ch. 4; Succi 2001 ch. 5; for the moment-space generalisation used
here: Lallemand & Luo, *Phys. Rev. E* 61 (2000) 6546, and the reference
implementation ``pylbm/analysis/equivalent_equation.py``):

    f_q = f_q^eq + eps f_q^(1) + eps^2 f_q^(2) + O(eps^3),
    d_t = eps d_t1 + eps^2 d_t2,      d_a = eps d_a1.

Order ``eps`` reproduces the Euler conservation laws; order ``eps^2`` gives the
Navier--Stokes viscous stress

    sigma_ab = sum_q c_qa c_qb f_q^(1) = -2 rho nu S_ab - rho zeta theta delta_ab,

where the non-equilibrium population follows from the collision operator written
in an arbitrary eigenbasis ``B`` with relaxation rates ``s``:

    f^(1)_q         = -[B diag(sig) B^-1]_qj (d_t1 + c_j . grad1) f_j^eq,
    sig_i           = 1/s_i - 1/2        (trapezoidal, second-order accurate),

``B = I`` with a single rate is BGK, the (symmetric, antisymmetric) direction
pairing with two rates is TRT, and ``B = M^-1`` with a moment matrix ``M`` is
MRT, so *one* code path serves every collision model and the transport
coefficients are read off the resulting symbolic stress tensor without any
hard-coded formula.

Transport coefficients are defined in linear response, so the extraction is done
at ``u -> 0`` with a homogeneous density field (``grad rho = 0``, the
incompressible Mach limit in which ``nu`` and ``zeta`` are defined).  The
``u``-dependent part of the order-``eps^2`` stress is reported separately as the
non-Navier--Stokes remainder (Burnett type, ``O(Ma * Kn)``) instead of being
silently folded into the coefficients.

Verification split
------------------
``tests/test_chapman_enskog.py`` pins the derivation against closed-form laws and
-- decisive for an *automatic* derivation -- against invariance probes (a rate
that cannot influence a coefficient must not change it).  The numerical gate
against the repository solvers lives in
``benchmarks/xval_chapman_enskog_auto/run_xval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

import sympy as sp

__all__ = [
    "Stencil",
    "CollisionModel",
    "Derivation",
    "D2Q9",
    "D3Q19",
    "D3Q27",
    "bgk",
    "trt",
    "mrt",
    "derive_equivalent_equation",
]


# --------------------------------------------------------------------------- #
# lattice stencils
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stencil:
    """Velocity set and weights of a lattice Boltzmann stencil."""

    name: str
    c: tuple[tuple[int, ...], ...]
    w: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.c) != len(self.w):
            raise ValueError(f"stencil {self.name}: {len(self.c)} velocities vs {len(self.w)} weights")
        dim = len(self.c[0])
        if any(len(v) != dim for v in self.c):
            raise ValueError(f"stencil {self.name}: mixed velocity dimensions")
        if any(w <= 0 for w in self.w):
            raise ValueError(f"stencil {self.name}: weights must be positive")
        if abs(sum(self.w) - 1) > 1e-15:
            raise ValueError(f"stencil {self.name}: weights sum to {sum(self.w)}, not 1")

    # -- structure ---------------------------------------------------------- #
    @property
    def q(self) -> int:
        return len(self.c)

    @property
    def dim(self) -> int:
        return len(self.c[0])

    def velocity_matrix(self) -> list[list[int]]:
        return [list(v) for v in self.c]

    # -- moments ------------------------------------------------------------ #
    def moment(self, exponents: Sequence[int]) -> sp.Expr:
        """Exact ``sum_q w_q prod_a c_qa^exponents[a]`` as a sympy Rational."""
        total = sp.Integer(0)
        for v, w in zip(self.c, self.w):
            term = sp.Rational(w.numerator, w.denominator)
            for a, e in enumerate(exponents):
                term *= v[a] ** e
            total += term
        return sp.nsimplify(total)

    def cs2_exact(self) -> sp.Rational:
        """Sound speed squared derived from the weights: ``sum_q w_q c_qx^2``."""
        return self.moment([2] + [0] * (self.dim - 1))

    def isotropy_checks(self) -> dict[str, bool]:
        """Whether the moments needed to close the second-order CE are isotropic.

        Second order needs ``M_ab = sum w c_a c_b = cs2 delta_ab`` and
        ``M_abcd = sum w c_a c_b c_c c_d = cs2^2 (delta_ab delta_cd +
        delta_ac delta_bd + delta_ad delta_bc)``.
        """
        cs2 = self.cs2_exact()
        second = True
        for a in range(self.dim):
            for b in range(self.dim):
                e = [0] * self.dim
                e[a] += 1
                e[b] += 1
                expected = cs2 if a == b else sp.Integer(0)
                second &= sp.simplify(self.moment(e) - expected) == 0
        fourth = True
        for a in range(self.dim):
            for b in range(self.dim):
                for cc in range(self.dim):
                    for d in range(self.dim):
                        e = [0] * self.dim
                        for idx in (a, b, cc, d):
                            e[idx] += 1
                        dab = 1 if a == b else 0
                        dcd = 1 if cc == d else 0
                        dac = 1 if a == cc else 0
                        dbd = 1 if b == d else 0
                        dad = 1 if a == d else 0
                        dbc = 1 if b == cc else 0
                        expected = cs2**2 * (dab * dcd + dac * dbd + dad * dbc)
                        fourth &= sp.simplify(self.moment(e) - expected) == 0
        return {"second_order_isotropic": bool(second), "fourth_order_isotropic": bool(fourth)}


D2Q9 = Stencil(
    name="D2Q9",
    c=((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)),
    w=(
        Fraction(4, 9),
        Fraction(1, 9),
        Fraction(1, 9),
        Fraction(1, 9),
        Fraction(1, 9),
        Fraction(1, 36),
        Fraction(1, 36),
        Fraction(1, 36),
        Fraction(1, 36),
    ),
)

D3Q19 = Stencil(
    name="D3Q19",
    c=(
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
    ),
    w=(
        Fraction(1, 3),
        *([Fraction(1, 18)] * 6),
        *([Fraction(1, 36)] * 12),
    ),
)

D3Q27 = Stencil(
    name="D3Q27",
    c=tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)),
    w=tuple(
        Fraction(8, 27)
        if (x, y, z) == (0, 0, 0)
        else Fraction(2, 27)
        if abs(x) + abs(y) + abs(z) == 1
        else Fraction(1, 54)
        if abs(x) + abs(y) + abs(z) == 2
        else Fraction(1, 216)
        for (x, y, z) in ((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))
    ),
)


# --------------------------------------------------------------------------- #
# collision operators
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CollisionModel:
    """Collision operator in eigenbasis form ``Delta = B diag(1-s) B^-1``.

    Parameters
    ----------
    basis:
        ``q x q`` matrix whose *columns* are the relaxation eigenmodes ``B``.
    rates:
        Relaxation rate ``s_i > 0`` per column.  A rate of ``None``/``0`` marks a
        *conserved* mode (never relaxed); its component of
        ``(d_t1 + c.grad) f^eq`` must vanish identically, which the derivation
        asserts instead of silently dividing by zero.
    """

    name: str
    basis: sp.Matrix
    rates: tuple[float | None, ...]

    def validate(self, q: int) -> None:
        if self.basis.shape != (q, q):
            raise ValueError(f"collision {self.name}: basis {self.basis.shape} does not match q={q}")
        if len(self.rates) != q:
            raise ValueError(f"collision {self.name}: {len(self.rates)} rates for q={q}")
        if self.basis.det() == 0:
            raise ValueError(f"collision {self.name}: singular eigenbasis")
        for s in self.rates:
            if s is not None and not s > 0:
                raise ValueError(f"collision {self.name}: relaxation rates must be positive (got {s})")

    @property
    def is_bgk(self) -> bool:
        conserved = [s is None for s in self.rates]
        live = {round(s, 12) for s in self.rates if s is not None}
        return len(live) == 1 and conserved[0] is False and all(not c for c in conserved)


def bgk(tau: float, q: int = 9) -> CollisionModel:
    """Single-relaxation-time BGK: ``B = I`` with ``s_i = 1/tau`` on every mode."""
    if not tau > 0.5:
        raise ValueError(f"BGK requires tau > 0.5 (got {tau})")
    return CollisionModel(name=f"BGK(tau={tau:g})", basis=sp.eye(q), rates=(1.0 / tau,) * q)


def _opposite_index(stencil: Stencil) -> list[int]:
    lookup = {v: i for i, v in enumerate(stencil.c)}
    return [lookup[tuple(-a for a in v)] for v in stencil.c]


def trt(stencil: Stencil, omega_plus: float, omega_minus: float) -> CollisionModel:
    """Two-relaxation-time operator via the (symmetric, antisymmetric) pairing."""
    if not (omega_plus > 0 and omega_minus > 0):
        raise ValueError("TRT rates must be positive")
    q = stencil.q
    opp = _opposite_index(stencil)
    basis = sp.zeros(q, q)
    rates: list[float | None] = [None] * q
    col = 0
    for i in range(q):
        if all(a == 0 for a in stencil.c[i]):
            basis[i, col] = 1
            rates[col] = omega_plus
            col += 1
    for i in range(q):
        if opp[i] <= i or all(a == 0 for a in stencil.c[i]):
            continue
        j = opp[i]
        basis[i, col] = 1
        basis[j, col] = 1
        rates[col] = omega_plus
        col += 1
        basis[i, col] = 1
        basis[j, col] = -1
        rates[col] = omega_minus
        col += 1
    if col != q:
        raise RuntimeError(f"TRT basis construction filled {col} of {q} columns")
    return CollisionModel(
        name=f"TRT(w+={omega_plus:g}, w-={omega_minus:g})",
        basis=basis,
        rates=tuple(rates),
    )


#: Standard D2Q9 moment matrix (row order: rho, e, eps, jx, qx, jy, qy, pxx, pxy),
#: as used by :mod:`tensorlbm.solver` for its MRT collision.
_M_D2Q9 = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (-4, -1, -1, -1, -1, 2, 2, 2, 2),
    (4, -2, -2, -2, -2, 1, 1, 1, 1),
    (0, 1, 0, -1, 0, 1, -1, -1, 1),
    (0, -2, 0, 2, 0, 1, -1, -1, 1),
    (0, 0, 1, 0, -1, 1, 1, -1, -1),
    (0, 0, -2, 0, 2, 1, 1, -1, -1),
    (0, 1, -1, 1, -1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, -1, 1, -1),
)

#: Row index of the D2Q9 moments produced by :data:`_M_D2Q9`.
D2Q9_MOMENT_ORDER = ("rho", "e", "eps", "jx", "qx", "jy", "qy", "pxx", "pxy")

#: Index in :data:`D2Q9_MOMENT_ORDER` of the shear (pxx, pxy) and bulk (e, eps) moments.
D2Q9_SHEAR_MOMENTS = ("pxx", "pxy")
D2Q9_BULK_MOMENTS = ("e", "eps")


def mrt(stencil: Stencil, moments: Sequence[Sequence[int]], rates: Sequence[float | None]) -> CollisionModel:
    """Moment-space collision: ``B = M^-1`` with one rate per moment."""
    q = stencil.q
    M = sp.Matrix(moments)
    if M.shape != (q, q):
        raise ValueError(f"MRT moment matrix {M.shape} does not match q={q}")
    if M.det() == 0:
        raise ValueError("MRT moment matrix is singular")
    return CollisionModel(name="MRT", basis=M.inv(), rates=tuple(rates))


def d2q9_mrt(
    tau: float,
    *,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_q: float = 1.9,
) -> CollisionModel:
    """Humières-style D2Q9 MRT (rates of Lallemand & Luo 2000, `s_pxx = s_pxy = 1/tau`)."""
    rates: list[float | None] = [None] * 9
    rates[1] = s_e
    rates[2] = s_eps
    rates[4] = s_q
    rates[6] = s_q
    rates[7] = 1.0 / tau
    rates[8] = 1.0 / tau
    model = mrt(D2Q9, _M_D2Q9, rates)
    return CollisionModel(name=f"MRT-D2Q9(tau={tau:g}, s_e={s_e:g}, s_eps={s_eps:g})", basis=model.basis, rates=model.rates)


# --------------------------------------------------------------------------- #
# derivation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Derivation:
    """Result of :func:`derive_equivalent_equation`."""

    stencil_name: str
    collision_name: str
    cs2_symbolic: sp.Expr
    cs2_value: float
    nu_symbolic: sp.Expr
    zeta_symbolic: sp.Expr
    nu_value: float
    zeta_value: float
    nu_over_cs2_half: float
    isotropy: dict[str, bool]
    conservation_ok: bool
    stress_linear: sp.Matrix
    remainder_terms: tuple[str, ...]
    remainder_examples: tuple[sp.Expr, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stencil": self.stencil_name,
            "collision": self.collision_name,
            "cs2_symbolic": str(self.cs2_symbolic),
            "cs2": self.cs2_value,
            "cs": math.sqrt(self.cs2_value),
            "nu_symbolic": str(self.nu_symbolic),
            "nu": self.nu_value,
            "zeta_symbolic": str(self.zeta_symbolic),
            "zeta": self.zeta_value,
            "nu_over_tau_tilde_cs2": self.nu_over_cs2_half,
            "isotropy": self.isotropy,
            "conservation_ok": self.conservation_ok,
            "remainder_terms": list(self.remainder_terms),
        }


def _equilibrium(stencil: Stencil, rho: sp.Symbol, u: list[sp.Symbol], cs2: sp.Expr) -> list[sp.Expr]:
    """Second-order (isothermal) lattice equilibrium."""
    u2 = sum(ui**2 for ui in u)
    feq = []
    for v, w in zip(stencil.c, stencil.w):
        wq = sp.Rational(w.numerator, w.denominator)
        cu = sum(vi * ui for vi, ui in zip(v, u))
        feq.append(wq * rho * (1 + cu / cs2 + (cu * cu - cs2 * u2) / (2 * cs2**2)))
    return feq


def derive_equivalent_equation(
    stencil: Stencil,
    collision: CollisionModel,
    *,
    name: str | None = None,
) -> Derivation:
    """Derive ``nu`` and ``zeta`` for ``(stencil, collision)`` by Chapman--Enskog.

    See the module docstring for the expansion.  The returned coefficients are
    exact symbolic expressions in the relaxation rates of ``collision``.
    """
    q = stencil.q
    dim = stencil.dim
    collision.validate(q)

    cs2 = sp.nsimplify(stencil.cs2_exact())
    rho = sp.Symbol("rho", positive=True)
    u = sp.symbols("u0:3", real=True)[:dim]
    grad = sp.Matrix(dim, dim, lambda a, b: sp.Symbol(f"d{a}_{b}", real=True))
    theta = sum(grad[a, a] for a in range(dim))

    feq = _equilibrium(stencil, rho, list(u), cs2)

    # (d_t1 + c_q . grad1) f^eq at linear order in u, homogeneous density:
    #   d_t1 rho = -rho theta   (continuity at order eps)
    #   d_t1 u_a = -(u . grad) u_a   (order-eps momentum, grad rho = 0)
    #   d_a rho  = 0            (incompressible Mach limit)
    v = []
    for idx, vq in enumerate(stencil.c):
        expr = sp.diff(feq[idx], rho) * (-rho * theta)
        for a in range(dim):
            # (d_t1 + c . grad1) u_a = sum_b (c_qb - u_b) d_b u_a,
            # using the order-eps momentum equation d_t1 u_a = -(u . grad) u_a
            # (the pressure-gradient term is O(grad rho) and dropped here).
            du_a = sum((vq[b] - u[b]) * grad[b, a] for b in range(dim))
            expr += sp.diff(feq[idx], u[a]) * du_a
        v.append(sp.expand(expr))

    # moment-space relaxation: f^(1) = -B diag(sig) B^-1 v
    Binv = collision.basis.inv()
    wvec = sp.expand(Binv * sp.Matrix(v))
    sig = []
    conserved_violations = 0
    for i, s in enumerate(collision.rates):
        comp = sp.simplify(sp.expand(wvec[i]))
        if s is None:
            if comp != 0:
                conserved_violations += 1
            sig.append(0)
        else:
            sig.append(sp.nsimplify(1 / sp.Float(s) - sp.Rational(1, 2)))
    if conserved_violations:
        raise ValueError(
            f"{name or stencil.name}/{collision.name}: {conserved_violations} conserved modes of the "
            "order-eps equation are non-zero, the Chapman-Enskog ansatz is inconsistent"
        )

    f1 = sp.expand(-collision.basis * sp.diag(*sig) * wvec)

    # conservation checks: mass and momentum moments of f^(1) must vanish
    mass = sp.simplify(sp.expand(sum(f1)))
    mom = [sp.simplify(sp.expand(sum(f1[idx] * stencil.c[idx][a] for idx in range(q)))) for a in range(dim)]
    conservation_ok = mass == 0 and all(m == 0 for m in mom)

    stress = sp.Matrix(dim, dim, lambda a, b: sp.simplify(sp.expand(sum(f1[idx] * stencil.c[idx][a] * stencil.c[idx][b] for idx in range(q)))))

    # linear-response extraction (u -> 0)
    sub_zero = {ui: sp.Integer(0) for ui in u}
    stress_lin = stress.subs(sub_zero)
    stress_lin = stress_lin.applyfunc(lambda e: sp.simplify(sp.expand(e)))

    if dim < 2:
        raise ValueError("shear viscosity extraction needs dim >= 2")
    denom = rho * (grad[0, 1] + grad[1, 0])
    nu = sp.simplify(sp.expand(-stress_lin[0, 1] / denom))
    if nu.free_symbols & set(grad):
        raise ValueError(f"could not isolate nu from sigma_01 = {stress_lin[0, 1]}")
    nu = sp.nsimplify(nu)

    trace = sp.simplify(sp.expand(sum(stress_lin[a, a] for a in range(dim))))
    diag_coeffs = [sp.simplify(sp.expand(sp.diff(trace, grad[a, a]))) for a in range(dim)]
    if any(sp.simplify(c - diag_coeffs[0]) != 0 for c in diag_coeffs):
        raise ValueError(f"isolation of zeta failed: diagonal coefficients differ {diag_coeffs}")
    # Standard (deviatoric) Newtonian split with the dimension-general 1/D factor,
    #     sigma_ab = -2 rho nu (S_ab - theta delta_ab / D) - rho zeta theta delta_ab,
    # whose trace is -D rho zeta theta: the deviatoric part is traceless, so the
    # whole trace carries the bulk viscosity.  Note zeta_here = zeta_grad + 2 nu / D,
    # where zeta_grad is the coefficient of theta delta_ab when the *full* symmetric
    # gradient is used; the difference is pure convention, and only the combination
    # ``2(D-1)/D nu + zeta`` (see :func:`longitudinal_viscosity`) is observable.
    zeta = sp.simplify(sp.expand(-diag_coeffs[0] / (rho * dim)))
    zeta = sp.nsimplify(zeta)

    # non-NS remainder: everything in stress that is not the NS viscous stress
    ns = sp.zeros(dim, dim)
    for a in range(dim):
        for b in range(dim):
            deviatoric = grad[a, b] + grad[b, a] - (2 * theta / dim if a == b else 0)
            ns[a, b] = -rho * nu * deviatoric - (rho * zeta * theta if a == b else 0)
    remainder = (stress - ns).applyfunc(lambda e: sp.simplify(sp.expand(e)))
    examples: list[sp.Expr] = []
    terms: list[str] = []
    for a in range(dim):
        for b in range(dim):
            for monom in sp.Add.make_args(sp.expand(remainder[a, b])):
                if monom != 0:
                    examples.append(monom)
                    terms.append(str(monom))
    uniq = tuple(dict.fromkeys(terms))
    uniq_ex = tuple(dict.fromkeys(examples))

    return Derivation(
        stencil_name=stencil.name,
        collision_name=name or collision.name,
        cs2_symbolic=cs2,
        cs2_value=float(cs2),
        nu_symbolic=nu,
        zeta_symbolic=zeta,
        nu_value=float(nu),
        zeta_value=float(zeta),
        nu_over_cs2_half=float(nu) / (float(cs2)),
        isotropy=stencil.isotropy_checks(),
        conservation_ok=bool(conservation_ok),
        stress_linear=stress_lin,
        remainder_terms=uniq,
        remainder_examples=uniq_ex,
    )


def longitudinal_viscosity(derivation: Derivation, dim: int | None = None) -> float:
    """``nu_L = 2(dim-1)/dim nu + zeta`` -- the coefficient damping sound waves.

    Sound attenuation for a compressible Navier--Stokes fluid is
    ``Re(omega) = -k^2 nu_L / 2`` with ``nu_L = (4/3) nu + zeta`` in 3D; the
    dimension-general form above reduces to both the 2D and 3D textbook cases.
    """
    d = dim if dim is not None else _dim_of(derivation)
    return 2.0 * (d - 1) / d * derivation.nu_value + derivation.zeta_value


def _dim_of(derivation: Derivation) -> int:
    return derivation.stress_linear.shape[0]
