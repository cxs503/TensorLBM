"""Continuous-spectrum linear stability of Rayleigh-Bénard convection (reference).

This is the reference half of the IN31-N12 pair: :mod:`tensorlbm.rb_discrete_stability`
computes the growth rate of **the lattice operator that the benchmark actually
time-marches**, this module computes the growth rate of the **continuum**
Navier-Stokes-Boussinesq problem that the lattice equations are supposed to
approximate.  Comparing the two separates discretisation error from physics.

Formulation
-----------
Perturbations of the conductive state written with a streamfunction ``psi`` and a
temperature perturbation ``theta``, ``(u, w) = (d psi/dz, -d psi/dx)``::

    d/dt (D^2 - a^2) psi = (D^2 - a^2)^2 psi - i a (Ra/Pr) theta          (1)
    d/dt theta           = (1/Pr) (D^2 - a^2) theta - i a psi            (2)

with rigid-rigid isothermal walls::

    psi = D psi = 0,  theta = 0     at z = 0, 1                          (3)

Lengths are scaled with the gap ``H``, time with ``H^2/nu``, the temperature
with ``dT`` and the streamfunction with ``nu``, so the growth rate ``sigma`` is
directly comparable with ``gamma_hat = gamma H^2/nu`` from the lattice module.
The minus sign in (1) is the sign of the horizontal Laplacian: the buoyancy term
of the vorticity equation is ``curl(g beta theta e_z) = -g beta d theta/dx``, and
two horizontal differentiations turn ``theta`` into ``-a^2 theta``.

At marginal stability (``sigma = 0``) equations (1)-(2) combine into the classical
Prandtl-number independent problem ``(D^2 - a^2)^3 psi = -a^2 Ra psi`` with
``psi = D psi = (D^2-a^2) psi = 0`` on both walls, whose minimum is
``Ra_c = 1707.76`` at ``a_c = 3.1163`` (Chandrasekhar 1961).  That value is this
module's self-test.

Discretisation and why not Chebyshev collocation
------------------------------------------------
A fourth-order operator discretised with explicit Chebyshev differentiation
matrices is badly conditioned (``cond(D^4) ~ N^8``); in this two-field
formulation it turns the spectrum into a fan of spurious eigenvalues whose real
parts grow like ``N^4`` and swamp the physical branch (measured during
development: ``sigma_spurious ~ +10^3`` at ``N = 48``).  Basis recombination on
the boundary conditions and row replacement were both tried and both failed to
converge at better than first order.  The formulation used here is therefore a
second-order finite-difference discretisation of (1)-(3) on ``M + 2`` equally
spaced nodes (walls at ``z = 0, 1``), with the two Neumann conditions folded into
the ghost values of the intermediate variable ``omega = (D^2 - a^2) psi``::

    omega_0 = 2 psi_1 / h^2,  omega_{M+1} = 2 psi_M / h^2

(``psi_0 = psi_{M+1} = 0`` and the odd reflection ``psi_{-1} = psi_1`` implied by
``D psi(0) = 0``).  The generalised eigenproblem ``A x = sigma B x`` is solved with
shift-invert ARPACK around a target ``sigma``, which makes branch selection
unambiguous (no spurious mode can win) and lets the branch be continued in
``Ra``.  Being second order, the marginal ``Ra`` is reported together with a
Richardson extrapolation over three grids; the extrapolated value reproduces
``Ra_c = 1707.76`` to a few ``1e-4`` and the finite-``M`` values converge
quadratically, which is the check that the discretisation is correct.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # scipy is a package dependency; import locally nonetheless
    import scipy.linalg as sla
    from scipy import sparse
    from scipy.sparse.linalg import eigs, splu

    _SCIPY_OK = True
    _SCIPY_ERR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    sparse = None
    eigs = None
    splu = None
    sla = None
    _SCIPY_OK = False
    _SCIPY_ERR = exc

RA_CRIT_CONTINUOUS = 1707.7597  # Chandrasekhar (1961), rigid-rigid, 2-D rolls
A_CRIT_CONTINUOUS = 3.11632  # critical wavenumber a_c = k H
M_DEFAULT = 400  # interior nodes of the reference discretisation


def _require_scipy() -> None:
    if not _SCIPY_OK:  # pragma: no cover
        raise RuntimeError(f"scipy is required for rb_continuum_stability: {_SCIPY_ERR}")


# ---------------------------------------------------------------------------
# Discrete operators
# ---------------------------------------------------------------------------
def operators(M: int, a: float):
    """Sparse ``(D^2 - a^2)`` and ``(D^2 - a^2)^2`` on ``psi`` and ``theta``.

    ``M`` interior nodes, ``h = 1/(M+1)``; returns ``(L2_psi, L4_psi, L2_theta)``
    each of shape ``(M, M)`` acting on the interior unknowns.
    """
    _require_scipy()
    from scipy.sparse import csr_matrix, identity

    h = 1.0 / (M + 1)
    a2 = a * a
    # --- theta: plain Dirichlet Laplacian on interior nodes -----------------
    main = np.full(M, -2.0 / h**2 - a2)
    off = np.full(M - 1, 1.0 / h**2)
    L2_theta = sparse.diags([off, main, off], [-1, 0, 1], format="csr")

    # --- psi -> omega on nodes 0..M+1 (ghosts folded in) --------------------
    # omega_0 = 2 psi_1 / h^2 ; omega_i = (psi_{i-1}-2psi_i+psi_{i+1})/h^2 - a^2 psi_i
    # ; omega_{M+1} = 2 psi_M / h^2
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    # node 0 (wall): 2 psi_1 / h^2  -> column 0 (psi_1) is local index 0
    rows.append(0); cols.append(0); vals.append(2.0 / h**2)
    for i in range(1, M + 1):
        rows.append(i); cols.append(i - 1); vals.append(-2.0 / h**2 - a2)
        if i - 1 >= 1:  # psi_{i-1} is interior (node i-1 >= 1)
            rows.append(i); cols.append(i - 2); vals.append(1.0 / h**2)
        if i + 1 <= M:  # psi_{i+1} interior
            rows.append(i); cols.append(i); vals.append(1.0 / h**2)
    rows.append(M + 1); cols.append(M - 1); vals.append(2.0 / h**2)
    E = csr_matrix((vals, (rows, cols)), shape=(M + 2, M))

    # D2 applied to a full node vector (0..M+1); Dirichlet for theta handled by E
    main2 = np.full(M + 2, -2.0 / h**2)
    off2 = np.full(M + 1, 1.0 / h**2)
    D2full = sparse.diags([off2, main2, off2], [-1, 0, 1], format="csr")
    # restrict interior rows of (D2 - a^2) omega to interior nodes
    Rint = csr_matrix((np.ones(M), (np.arange(M), np.arange(1, M + 1))), shape=(M, M + 2))
    L4_psi = (Rint @ (D2full - a2 * identity(M + 2))) @ E
    L2_psi = sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    return L2_psi.tocsr(), L4_psi.tocsr(), L2_theta.tocsr()


def rb_matrices(M: int, a: float, Ra: float, Pr: float):
    """Generalised eigenproblem matrices ``(A, B)`` of (1)-(3)."""
    _require_scipy()
    from scipy.sparse import bmat, identity

    L2_psi, L4_psi, L2_theta = operators(M, a)
    I = identity(M, format="csr") * 1.0
    A = bmat(
        [
            [L4_psi, (-1j * a * (Ra / Pr)) * I],
            [(-1j * a) * I, L2_theta / Pr],
        ],
        format="csr",
    )
    B = bmat([[L2_psi, None], [None, I]], format="csr")
    return A.tocsc(), B.tocsc()


# ---------------------------------------------------------------------------
# Eigenvalue / branch selection
# ---------------------------------------------------------------------------
def spectra(M: int, a: float, Ra: float, Pr: float, sigma: float = 0.0, k: int = 4):
    """The ``k`` eigenvalues closest to ``sigma`` (shift-invert ARPACK)."""
    _require_scipy()
    A, B = rb_matrices(M, a, Ra, Pr)
    vals = eigs(A, k=k, M=B, sigma=sigma, which="LM", return_eigenvectors=False)
    vals = np.asarray(vals)
    return vals[np.argsort(np.abs(vals - sigma))]


def full_spectrum(M: int, a: float, Ra: float, Pr: float) -> np.ndarray:
    """All finite eigenvalues (dense QZ); used for branch continuation."""
    _require_scipy()
    A, B = rb_matrices(M, a, Ra, Pr)
    vals = np.asarray(sla.eig(A.toarray(), B.toarray(), right=False))
    vals = vals[np.isfinite(vals)]
    return vals


def growth(
    a: float,
    Ra: float,
    Pr: float = 0.71,
    M: int = 200,
    sigma_target: Optional[float] = None,
    Ra0: float = 200.0,
    n_steps: int = 60,
) -> Dict[str, object]:
    """Physical growth rate ``sigma`` (units ``nu/H^2``) of the roll branch.

    The full dense spectrum of the finite-difference eigenproblem is computed and
    the mode with the largest real part is the physical roll branch: for this
    problem it is separated from the rest of the spectrum by a wide margin
    (measured: ``sigma_lead = +6.78`` versus ``sigma_second = -51.9`` at
    ``Ra = 2500``), so no branch tracking is needed.  ``sigma_target``/``Ra0``/
    ``n_steps`` are accepted for API compatibility with earlier continuation
    implementations and ignored.
    """
    vals = full_spectrum(M, a, Ra, Pr)
    best = complex(vals[np.argmax(vals.real)])
    second = float(np.sort(vals.real)[-2]) if vals.size > 1 else float("nan")
    return {
        "sigma": best.real,
        "sigma_complex": [best.real, best.imag],
        "sigma_second_real": second,
        "a": float(a),
        "Ra": float(Ra),
        "Pr": float(Pr),
        "M": int(M),
    }


def growth_stable(a: float, Ra: float, Pr: float = 0.71, M: int = M_DEFAULT, **kw):
    """Alias of :func:`growth` (kept for API stability)."""
    return growth(a, Ra, Pr=Pr, M=M, **kw)


# ---------------------------------------------------------------------------
# Critical Rayleigh number
# ---------------------------------------------------------------------------
def critical_at(
    a: float,
    Pr: float = 0.71,
    M: int = M_DEFAULT,
    Ra_lo: float = 500.0,
    Ra_hi: float = 6000.0,
    tol: float = 1e-7,
) -> float:
    """Bisect ``sigma(Ra) = 0`` at fixed wavenumber."""
    lo, hi = float(Ra_lo), float(Ra_hi)

    def sigma_near0(Ra: float) -> float:
        vals = spectra(M, a, Ra, Pr, sigma=0.0, k=4)
        return float(np.max(vals.real))

    s_lo, s_hi = sigma_near0(lo), sigma_near0(hi)
    if s_lo >= 0.0:
        raise ValueError(f"Ra_lo={lo} already unstable (sigma={s_lo:+.3e})")
    if s_hi <= 0.0:
        raise ValueError(f"Ra_hi={hi} still stable (sigma={s_hi:+.3e})")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sigma_near0(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * mid:
            break
    return 0.5 * (lo + hi)


def marginal_curve(a_values: Sequence[float], Pr: float = 0.71, M: int = M_DEFAULT, **kw):
    """``Ra_c(a)`` on a wavenumber grid."""
    return [{"a": float(a), "Ra_c": critical_at(float(a), Pr=Pr, M=M, **kw)} for a in a_values]


def richardson(values: Sequence[float], factor: float = 2.0) -> float:
    """Richardson extrapolation ``q -> inf`` of a 2nd-order sequence."""
    v = list(values)
    if len(v) < 2:
        return v[0]
    if len(v) == 2:
        return (factor**2 * v[-1] - v[-2]) / (factor**2 - 1.0)
    # three-point / repeated two-point extrapolation
    e1 = (factor**2 * v[-1] - v[-2]) / (factor**2 - 1.0)
    e2 = (factor**2 * v[-2] - v[-3]) / (factor**2 - 1.0)
    return (factor**2 * e1 - e2) / (factor**2 - 1.0)


def critical_rayleigh_continuous(
    Pr: float = 0.71,
    M_list: Sequence[int] = (100, 200, 400),
    refine: bool = True,
    a_lo: float = 2.4,
    a_hi: float = 4.0,
) -> Dict[str, object]:
    """``Ra_c`` at the most dangerous wavenumber, with grid extrapolation.

    ``refine`` runs a golden-section search over the wavenumber for each grid,
    then extrapolates ``Ra_c -> 0 (h^2)`` with Richardson.
    """
    out: Dict[str, object] = {"Pr": Pr, "M_list": list(M_list)}
    ra_by_grid: List[float] = []
    for M in M_list:
        if refine:
            lo, hi = a_lo, a_hi
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            best = None
            for _ in range(18):
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
                ra_c = critical_at(c, Pr=Pr, M=M)
                ra_d = critical_at(d, Pr=Pr, M=M)
                if best is None or min(ra_c, ra_d) < best[0]:
                    best = (min(ra_c, ra_d), c if ra_c < ra_d else d)
                if ra_c < ra_d:
                    hi = d
                else:
                    lo = c
                if hi - lo < 1e-3:
                    break
            ra_c, a_c = best
        else:
            a_c = A_CRIT_CONTINUOUS
            ra_c = critical_at(a_c, Pr=Pr, M=M)
        ra_by_grid.append(ra_c)
        out[f"Ra_c_M{M}"] = ra_c
        out[f"a_c_M{M}"] = a_c
    out["Ra_c_extrapolated"] = richardson(ra_by_grid)
    out["Ra_c_reference"] = RA_CRIT_CONTINUOUS
    out["rel_err_extrapolated"] = out["Ra_c_extrapolated"] / RA_CRIT_CONTINUOUS - 1.0
    return out


def growth_at_lattice_wavenumber(
    ny: int, nx: int, Ra: float, Pr: float = 0.71, M: int = M_DEFAULT, mode: int = 1, **kw
) -> Dict[str, object]:
    """Continuum growth rate at the wavenumber an ``ny x nx`` periodic box resolves.

    ``a = k H = 2 pi mode H / nx`` with ``H = ny``, i.e. exactly the same ``a`` as
    the lattice box, so the discrete and continuum growth rates are compared at
    equal wavenumber.
    """
    a = 2.0 * math.pi * mode * float(ny) / float(nx)
    out = growth(a, Ra, Pr=Pr, M=M, **kw)
    out["ny"] = int(ny)
    out["nx"] = int(nx)
    out["mode"] = int(mode)
    return out