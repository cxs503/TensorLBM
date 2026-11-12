"""Linear stability of two *different* 2-D Boussinesq configurations.

.. warning::

   Two configurations live in this module and they must not be compared to the
   same reference value:

   * :func:`boussinesq_operator` / :func:`growth_rate` / :func:`critical_rayleigh`
     — a differentially **side-heated** square cavity (base state
     ``Tb = 1 - x``, gravity along ``-y``, i.e. gravity **perpendicular** to the
     temperature gradient), no-slip walls, isothermal side walls, adiabatic
     top/bottom.
   * :func:`benard_operator` / :func:`benard_growth_rate` /
     :func:`benard_critical_rayleigh` / :func:`benard_critical_rayleigh_continuous`
     — a Rayleigh--Bénard layer (base state ``Tb = 1 - y``, gravity along
     ``-y``, i.e. gravity **parallel** to the gradient), rigid-rigid plates,
     isothermal plates, Bloch-periodic in ``x`` at wavenumber ``a``.

   ``Ra_c = 1707.7597`` (Chandrasekhar 1961, rigid-rigid) belongs to the
   **Bénard layer** only.  Measurement (this module, m16921): the side-heated
   square cavity is *stable* at ``Ra = 1707.76`` (``sigma = -9.1`` at ``n = 16``)
   and only loses stability near ``Ra ~ 1.45e4`` (``n = 12``, see
   ``SIDE_HEATED_RA_C_MEASURED_N12``), i.e. a factor ``~8.5`` away.  An earlier
   version of this module claimed that the ``1707.762`` value was "reproduced
   here by an independent discretisation"; that claim was false for the
   side-heated operator (it never even bracketed a root) and has been replaced
   by the Bénard path below, which does reproduce the classical value.

Why an eigenvalue problem and not a time integration
----------------------------------------------------
Time integration of the lattice-Boltzmann model cannot resolve a 1 % threshold
window: the growth rate at ``Ra / Ra_c - 1 = 0.01`` is a few times ``1e-5`` per
lattice step, i.e. millions of steps per e-fold.  Both configurations are
therefore solved as a dense linearised eigenvalue problem,

.. math::

    \\partial_t \\omega = \\nu \\nabla^2 \\omega + g\\beta\\,\\partial_x \\theta,
    \\qquad \\nabla^2 \\psi = -\\omega,
    \\qquad \\partial_t \\theta = \\alpha \\nabla^2 \\theta + u\\,\\partial_y \\bar T
    + v\\,\\partial_x \\bar T,

``sigma(Ra)`` = largest real part of the spectrum, neutral point by bisection.

Units (both configurations): ``H = 1``, ``alpha = 1``, ``nu = Pr``, hence
``g beta = Ra * Pr``; ``sigma`` is in ``alpha / H^2``.  Note the time-unit trap:
:mod:`tensorlbm.rb_continuum_stability` uses the **viscous** time scale
(``sigma`` in ``nu / H^2``), so ``sigma_here = sigma_rb_compatible_continuum / Pr``.
With ``Pr = 0.71`` a naive comparison of the two modules is off by 41 %.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "boussinesq_operator", "growth_rate", "critical_rayleigh",
    "benard_operator", "benard_growth_rate", "benard_critical_rayleigh",
    "benard_wavenumber_scan", "benard_critical_rayleigh_continuous",
    "richardson", "RA_C_BENARD_RIGID_RIGID", "A_C_BENARD_RIGID_RIGID",
    "RA_C_LINEAR_THEORY", "SIDE_HEATED_RA_C_MEASURED_N12",
]

#: Chandrasekhar (1961) §II, rigid-rigid Rayleigh--Bénard layer, 2-D rolls.
#: This is the anchor of the **Bénard** configuration below, *not* of the
#: side-heated cavity.
RA_C_BENARD_RIGID_RIGID = 1707.7597
#: Critical wavenumber ``a_c = k H`` of the same problem.
A_C_BENARD_RIGID_RIGID = 3.11632

#: Deprecated alias, kept for API stability.  The value is the Bénard-layer one
#: (see the module warning): do not use it as a reference for
#: :func:`critical_rayleigh`.
RA_C_LINEAR_THEORY = RA_C_BENARD_RIGID_RIGID

#: Neutral Rayleigh number of the **side-heated** square cavity measured with
#: this module (``n = 12``, ``Pr = 0.71``, bisection on ``sigma(Ra) = 0``,
#: log-spaced bracket) — m16921.  No literature anchor for this configuration
#: is registered in the repository, so it is recorded as a measurement only.
SIDE_HEATED_RA_C_MEASURED_N12 = 1.45e4

_BENARD_MIN_INTERVALS = 4


def _interior_index(n: int):
    """Map (i, j) interior node (1 <= i, j <= n-1) to a flat index."""
    m = n - 1

    def idx(i: int, j: int) -> int:
        return (i - 1) * m + (j - 1)

    return m, idx


# ---------------------------------------------------------------------------
# Configuration 1: side-heated square cavity (gravity perpendicular to dT/dx)
# ---------------------------------------------------------------------------
def boussinesq_operator(n: int = 24, pr: float = 0.71) -> dict:
    """Assemble ``d/dt (omega, theta) = (S0 + Ra * S1) (omega, theta)``.

    Side-heated square cavity: ``Tb = 1 - x`` (hot wall at ``x = 0``, gravity
    along ``-y``), no-slip walls with Thom vorticity, isothermal side walls and
    adiabatic top/bottom walls.  ``n`` is the number of grid intervals
    (``n + 1`` nodes per direction, nodes on the walls).

    ``Ra_c = 1707.76`` does **not** apply to this operator (see the module
    warning); use :func:`benard_*` for the Chandrasekhar anchor.
    """
    m, idx = _interior_index(n)
    h = 1.0 / n
    h2 = h * h
    M = m * m

    # ---- Poisson matrix for psi (Dirichlet psi = 0 on all walls) ----------
    Lap = np.zeros((M, M))
    walls_per_node = np.zeros(M)
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            Lap[r, r] = -4.0 / h2
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 1 <= ii <= n - 1 and 1 <= jj <= n - 1:
                    Lap[r, idx(ii, jj)] = 1.0 / h2
                else:
                    walls_per_node[r] += 1.0
    # psi = -Lap^{-1} omega  (psi = 0 on the walls is already built in)
    psi_mat = -np.linalg.inv(Lap)

    # ---- vorticity Laplacian with Thom wall vorticity ---------------------
    # omega_wall = -2 psi_adjacent / h^2   (first interior node is h from wall)
    Lap_om = Lap.copy()
    diag_corr = np.zeros(M)
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if walls_per_node[r] > 0:
                # each wall-adjacent normal direction contributes
                # (+omega_wall/h^2) = (-2 psi_node / h^4)
                diag_corr[r] = -2.0 * walls_per_node[r] / (h2 * h2)
    Lap_om = Lap_om + np.diag(diag_corr) @ psi_mat

    # ---- theta Laplacian: Dirichlet in x, Neumann (mirror) in y ----------
    Lap_th = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            # x-direction: Dirichlet on side walls
            if j == 1:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, 2)] += 1.0 / h2
            elif j == n - 1:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, n - 2)] += 1.0 / h2
            else:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i, j + 1)] += 1.0 / h2
                Lap_th[r, idx(i, j - 1)] += 1.0 / h2
            # y-direction: adiabatic (mirror) top/bottom
            if i == 1:
                Lap_th[r, r] += -1.0 / h2
                Lap_th[r, idx(2, j)] += 1.0 / h2
            elif i == n - 1:
                Lap_th[r, r] += -1.0 / h2
                Lap_th[r, idx(n - 2, j)] += 1.0 / h2
            else:
                Lap_th[r, r] += -2.0 / h2
                Lap_th[r, idx(i + 1, j)] += 1.0 / h2
                Lap_th[r, idx(i - 1, j)] += 1.0 / h2

    # ---- velocity from psi: u = d psi / d y (mirror at adiabatic walls) ---
    Dy = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if i == 1:
                Dy[r, idx(2, j)] = 1.0 / (2.0 * h)
                Dy[r, r] = -1.0 / (2.0 * h)
            elif i == n - 1:
                Dy[r, r] = 1.0 / (2.0 * h)
                Dy[r, idx(n - 2, j)] = -1.0 / (2.0 * h)
            else:
                Dy[r, idx(i + 1, j)] = 1.0 / (2.0 * h)
                Dy[r, idx(i - 1, j)] = -1.0 / (2.0 * h)
    Adv = Dy @ psi_mat  # theta_t += Adv @ omega

    # ---- d theta / dx (central differences, Dirichlet walls) -------------
    Dx = np.zeros((M, M))
    for i in range(1, n):
        for j in range(1, n):
            r = idx(i, j)
            if j == 1:
                Dx[r, r] = -1.0 / h
                Dx[r, idx(i, 2)] = 1.0 / h
            elif j == n - 1:
                Dx[r, r] = 1.0 / h
                Dx[r, idx(i, n - 2)] = -1.0 / h
            else:
                Dx[r, idx(i, j + 1)] = 1.0 / (2.0 * h)
                Dx[r, idx(i, j - 1)] = -1.0 / (2.0 * h)

    # ---- assemble S = S0 + Ra * S1 --------------------------------------
    S0 = np.zeros((2 * M, 2 * M))
    S1 = np.zeros((2 * M, 2 * M))
    # rows: omega block
    S0[:M, :M] = pr * Lap_om            # nu = Pr (alpha = 1)
    S1[:M, M:] = -pr * Dx               # -g beta d/dx with g beta = Ra * Pr
    # rows: theta block
    S0[M:, M:] = Lap_th
    S0[M:, :M] = Adv
    return {"S0": S0, "S1": S1, "n": n, "m": m, "size": 2 * M,
            "configuration": "side_heated_square_cavity"}


def growth_rate(ra: float, ops: dict | None = None, n: int = 24,
                pr: float = 0.71) -> float:
    """Largest real part of the side-heated-cavity spectrum at ``Ra``."""
    ops = ops if ops is not None else boussinesq_operator(n, pr)
    S = ops["S0"] + ra * ops["S1"]
    ev = np.linalg.eigvals(S)
    return float(np.max(ev.real))


def critical_rayleigh(n: int = 24, pr: float = 0.71, ra_lo: float = 500.0,
                      ra_hi: float = 4000.0, tol: float = 1e-6,
                      max_iter: int = 40) -> dict:
    """Neutral point of the **side-heated** cavity, secant on ``sigma(Ra) = 0``.

    The default bracket ``[500, 4000]`` does **not** contain a root: this
    configuration is still stable there (measured ``sigma(4000) = -9.1`` at
    ``n = 16``).  A non-bracketed call therefore returns ``ra_c = None`` with
    ``status = 'not_bracketed'`` and every comparison key present (``None``) —
    it never silently reports a value, and downstream code does not need to
    special-case the dict.
    """
    ops = boussinesq_operator(n, pr)
    hist: List[Dict[str, float]] = []
    f_lo = growth_rate(ra_lo, ops)
    f_hi = growth_rate(ra_hi, ops)
    hist.append({"ra": ra_lo, "sigma": f_lo})
    hist.append({"ra": ra_hi, "sigma": f_hi})
    base = {
        "pr": pr,
        "n": n,
        "resolution": f"{n}x{n} interiors",
        "history": hist,
        "bracket": (ra_lo, ra_hi),
        "sigma_bracket": (f_lo, f_hi),
        "configuration": "side_heated_square_cavity",
        "ra_c_reference": None,  # no registered literature anchor
        "ra_c_linear_theory": RA_C_BENARD_RIGID_RIGID,  # other configuration
    }
    if f_lo > 0 or f_hi < 0:
        return {**base, "ra_c": None, "deviation_rel": None,
                "status": "not_bracketed",
                "note": ("root not bracketed -- 'Ra < 1707.76 -> unstable' does "
                         "not hold for the side-heated cavity; the neutral "
                         "point is near Ra ~ 1.45e4 (see "
                         "SIDE_HEATED_RA_C_MEASURED_N12)")}
    a, fa = ra_lo, f_lo
    b, fb = ra_hi, f_hi
    ra_c: Optional[float] = None
    for _ in range(max_iter):
        # secant step, kept inside the bracket
        if fb - fa == 0.0:
            c = 0.5 * (a + b)
        else:
            c = b - fb * (b - a) / (fb - fa)
            if not (a < c < b):
                c = 0.5 * (a + b)
        fc = growth_rate(c, ops)
        hist.append({"ra": c, "sigma": fc})
        if abs(fc) < 1e-9 or (b - a) < tol * 0.5 * (a + b):
            ra_c = c
            break
        if fc > 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        ra_c = c
    return {**base, "ra_c": ra_c, "deviation_rel": None,
            "status": "ok" if ra_c is not None else "iteration_limit"}


# ---------------------------------------------------------------------------
# Configuration 2: Rayleigh--Bénard layer (gravity parallel to dT/dy)
# ---------------------------------------------------------------------------
def benard_operator(n: int = 32, a: float = A_C_BENARD_RIGID_RIGID,
                    pr: float = 0.71) -> dict:
    """Assemble the Bénard-layer operator ``d/dt (omega, theta) = (S0 + Ra S1)(...)``.

    Base state ``Tb = 1 - y`` (hot plate at ``y = 0``, gravity along ``-y``),
    rigid-rigid plates with Thom vorticity, isothermal plates (``theta = 0``),
    and **Bloch-periodic** behaviour in ``x`` at wavenumber ``a``
    (``d/dx -> i a``, ``d^2/dx^2 -> -a^2``).  Only the ``n - 1`` interior
    ``y``-nodes are unknowns, so the matrices are ``2(n-1)`` square and complex.

    The pair ``(omega, theta)`` is closed through ``psi = -lap^{-1} omega``
    (Dirichlet ``psi = 0`` on both plates), i.e. the advection of the base
    temperature ``v Tb' = -i a psi`` enters the temperature equation.

    This is a **vorticity / Thom** discretisation: it differs from the
    streamfunction pencil of :mod:`tensorlbm.rb_continuum_stability`
    (4th-order operator split as ``(D^2 - a^2)^2``, ghost-folded).  The two
    agree on ``sigma`` to ``< 1 %`` at ``n = 64`` vs ``M = 100`` once the
    ``1/Pr`` time-unit conversion is applied (measured, m16921).
    """
    if n < _BENARD_MIN_INTERVALS:
        raise ValueError(f"n must be >= {_BENARD_MIN_INTERVALS}, got {n}")
    M = n - 1
    h = 1.0 / n
    h2 = h * h
    a2 = a * a

    # ---- (D^2 - a^2) on the interior y-nodes, Dirichlet on both plates ----
    main = np.full(M, -2.0 / h2 - a2)
    off = np.full(M - 1, 1.0 / h2)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    lap_inv = np.linalg.inv(lap)
    psi_mat = -lap_inv                      # psi = -lap^{-1} omega

    # ---- Thom wall vorticity: omega_wall = -2 psi_node / h^2 -------------
    corr = np.zeros(M)
    corr[0] = -2.0 / (h2 * h2)
    corr[-1] = -2.0 / (h2 * h2)
    lap_om = lap + np.diag(corr) @ psi_mat

    ia = 1j * a
    S0 = np.zeros((2 * M, 2 * M), dtype=complex)
    S1 = np.zeros((2 * M, 2 * M), dtype=complex)
    # omega_t = Pr lap_om omega + Pr Ra i a theta
    S0[:M, :M] = pr * lap_om
    S1[:M, M:] = pr * ia * np.eye(M)
    # theta_t = lap theta + v,  v = -i a psi = +i a lap^{-1} omega
    S0[M:, M:] = lap
    S0[M:, :M] = ia * lap_inv
    return {"S0": S0, "S1": S1, "n": n, "m": M, "a": float(a), "pr": pr,
            "size": 2 * M, "h": h, "configuration": "rayleigh_benard_layer"}


def benard_growth_rate(ra: float, ops: dict | None = None, n: int = 32,
                       a: float = A_C_BENARD_RIGID_RIGID,
                       pr: float = 0.71) -> float:
    """Largest real part of the Bénard spectrum at ``(Ra, a)``, in ``alpha/H^2``."""
    ops = ops if ops is not None else benard_operator(n, a, pr)
    ev = np.linalg.eigvals(ops["S0"] + ra * ops["S1"])
    return float(np.max(ev.real))


def benard_critical_rayleigh(n: int = 32, pr: float = 0.71,
                             a: float = A_C_BENARD_RIGID_RIGID,
                             ra_lo: float = 500.0, ra_hi: float = 4000.0,
                             tol: float = 1e-5, max_iter: int = 60) -> dict:
    """Neutral Rayleigh number of the Bénard layer at fixed ``a`` (bisection).

    Returns a dict with ``ra_c`` (``None`` if the bracket does not straddle the
    marginal point), ``status``, ``sigma_bracket``, ``deviation_rel`` against
    Chandrasekhar's ``1707.7597`` and the bisection history.
    """
    ops = benard_operator(n, a, pr)
    s_lo = benard_growth_rate(ra_lo, ops)
    s_hi = benard_growth_rate(ra_hi, ops)
    hist = [{"ra": ra_lo, "sigma": s_lo}, {"ra": ra_hi, "sigma": s_hi}]
    base = {
        "n": n, "a": float(a), "pr": pr, "m": ops["m"],
        "resolution": f"{ops['m']} interior y-nodes, h = 1/{n}",
        "bracket": (ra_lo, ra_hi), "sigma_bracket": (s_lo, s_hi),
        "history": hist,
        "configuration": "rayleigh_benard_layer",
        "ra_c_reference": RA_C_BENARD_RIGID_RIGID,
        "a_reference": A_C_BENARD_RIGID_RIGID,
    }
    if s_lo > 0.0 or s_hi < 0.0:
        return {**base, "ra_c": None, "deviation_rel": None,
                "status": "not_bracketed"}
    lo, hi = float(ra_lo), float(ra_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = benard_growth_rate(mid, ops)
        hist.append({"ra": mid, "sigma": s_mid})
        if s_mid < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * 0.5 * (lo + hi):
            break
    ra_c = 0.5 * (lo + hi)
    return {**base, "ra_c": ra_c, "status": "ok",
            "sigma_marginal": benard_growth_rate(ra_c, ops),
            "deviation_rel": ra_c / RA_C_BENARD_RIGID_RIGID - 1.0}


def benard_wavenumber_scan(a_values: Sequence[float], n: int = 32,
                           pr: float = 0.71, **kw) -> List[Dict[str, float]]:
    """``Ra_c(a)`` of the Bénard layer on a wavenumber grid."""
    return [{"a": float(a), "Ra_c": benard_critical_rayleigh(n=n, pr=pr, a=float(a), **kw)["ra_c"]}
            for a in a_values]


def richardson(values: Sequence[float], factor: float = 2.0) -> float:
    """Richardson extrapolation ``h -> 0`` of a 2nd-order sequence."""
    v = list(values)
    if len(v) < 2:
        return v[0]
    if len(v) == 2:
        return (factor**2 * v[-1] - v[-2]) / (factor**2 - 1.0)
    e1 = (factor**2 * v[-1] - v[-2]) / (factor**2 - 1.0)
    e2 = (factor**2 * v[-2] - v[-3]) / (factor**2 - 1.0)
    return (factor**2 * e1 - e2) / (factor**2 - 1.0)


def benard_critical_rayleigh_continuous(
    pr: float = 0.71,
    n_list: Sequence[int] = (16, 32),
    a_center: float = A_C_BENARD_RIGID_RIGID,
    a_span: float = 0.25,
    n_a: int = 5,
    refine: bool = True,
    refine_iter: int = 6,
) -> Dict[str, object]:
    """``Ra_c`` at the most dangerous wavenumber, with grid extrapolation.

    For every grid the wavenumber is scanned over
    ``a_center +- a_span`` (``n_a`` points) and, if ``refine``, narrowed by a
    golden-section search around the best point; ``Ra_c`` is the minimum over
    the wavenumbers.  The per-grid values are then Richardson-extrapolated in
    ``h`` (the vorticity/Thom discretisation is second order), which is the
    quantitative check against Chandrasekhar's ``Ra_c = 1707.7597``.
    """
    out: Dict[str, object] = {"pr": pr, "n_list": list(n_list),
                              "a_center": float(a_center),
                              "a_span": float(a_span)}
    ra_by_grid: List[float] = []
    a_by_grid: List[float] = []
    for n in n_list:
        grid = benard_wavenumber_scan(
            np.linspace(a_center - a_span, a_center + a_span, n_a), n=n, pr=pr,
            ra_lo=500.0, ra_hi=4000.0)
        valid = [g for g in grid if g["Ra_c"] is not None]
        if not valid:
            raise ValueError(f"no bracketed wavenumber at n={n}")
        best = min(valid, key=lambda g: g["Ra_c"])
        ra_best, a_best = float(best["Ra_c"]), float(best["a"])
        if refine:
            lo, hi = a_best - 0.5 * a_span / max(n_a - 1, 1) * 2.0, a_best + a_span / max(n_a - 1, 1) * 2.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            for _ in range(refine_iter):
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
                rc = benard_critical_rayleigh(n=n, pr=pr, a=c)["ra_c"]
                rd = benard_critical_rayleigh(n=n, pr=pr, a=d)["ra_c"]
                rc = np.inf if rc is None else rc
                rd = np.inf if rd is None else rd
                if rc < rd:
                    hi, ra_try, a_try = d, rc, c
                else:
                    lo, ra_try, a_try = c, rd, d
                if ra_try < ra_best:
                    ra_best, a_best = float(ra_try), float(a_try)
        ra_by_grid.append(ra_best)
        a_by_grid.append(float(a_best))
        out[f"Ra_c_n{n}"] = ra_best
        out[f"a_c_n{n}"] = float(a_best)
    out["ra_by_grid"] = ra_by_grid
    out["a_by_grid"] = a_by_grid
    out["Ra_c_extrapolated"] = float(richardson(ra_by_grid))
    out["Ra_c_reference"] = RA_C_BENARD_RIGID_RIGID
    out["rel_err_extrapolated"] = out["Ra_c_extrapolated"] / RA_C_BENARD_RIGID_RIGID - 1.0
    out["rel_err_by_grid"] = [v / RA_C_BENARD_RIGID_RIGID - 1.0 for v in ra_by_grid]
    return out