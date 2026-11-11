"""Three-segment KBC collision operator (D3Q27 / D3Q19), Karlin-Boesch-Chikatamarla.

Motivation (campaign node ``p495w4-n03``)
----------------------------------------
The ELBM H-theorem limiter relaxes the *whole* non-equilibrium part towards the
entropic equilibrium on coarse grids (m9102 measured field-mean ``alpha`` =
0.19 .. 0.31 at ``Delta+`` = 5.6 .. 12.3).  Because that relaxation also eats
the shear stress, the extra dissipation thickens the wall layer, the channel
relaminarises and ``C_f`` lands 47-63 % off DNS.  The defect is structural: the
H-limiter is not a sub-grid model.

The KBC split fixes exactly that.  The non-equilibrium part is decomposed by
Hermite projection into three *orthogonal* segments::

    f^neq = s + b + g            (exact identity, machine precision)

======  ==================================  =====================================
segment modes (D3Q27)                       relaxation
======  ==================================  =====================================
``s``   shear: traceless 2nd order, 5       ``omega_nu = 1/tau`` -- the *exact*
                                            BGK rate, so ``nu = cs2 (tau - 1/2)``
                                            to machine precision and the shear
                                            stress can never be over-damped
``b``   bulk: trace of the 2nd order, 1     ``omega_b = 1/tau_b`` (bulk viscosity)
``g``   ghost: 3rd order and above, 17      ``alpha * omega_g``, ``alpha`` from a
                                            monotone entropy-condition solve
======  ==================================  =====================================

Mode counts quoted in the mission text (2 conserved + 4 shear / 1 bulk / 1
ghost) are the *D2Q9* counts; on D3Q27 the same split reads 4 conserved + 5
shear / 1 bulk / 17 ghost (``f`` carries 27 moments: 1 density, 3 momentum,
6 second order, 17 third-and-higher).  The decomposition below is written for
the general case and is exercised on both lattices.

The entropy-condition solve
---------------------------
After the shear and bulk segments are fixed, the post-collision state is
affine in ``alpha``::

    f*(alpha) = f^eq + (1 - omega_nu) s + (1 - omega_b) b + (1 - alpha omega_g) g

so the discrete entropy ``H(alpha) = sum_i f*_i ln(f*_i / w_i)`` is a *convex*
function of the single scalar ``alpha`` (``x ln x`` is convex and ``f*`` is
affine in ``alpha``).  Its unique minimiser on the positivity interval is found
by Newton on ``H'(alpha)`` with a bisection safeguard (``H''> 0`` is available
in closed form).  ``H`` decreases monotonically for every accepted
``alpha`` -- the H-theorem is *verified*, not assumed, and the solver only ever
returns an ``alpha`` whose entropy it has evaluated.

``alpha = 0`` means "no ghost relaxation" and ``alpha = 1`` means "ghost modes
relaxed at the base ghost rate"; the cap ``alpha_cap`` allows the strongly
damped branch (``alpha > 1``) that KBC papers use on stiff grids.

Everything here is pure NumPy and is the *reference implementation*: the
production OpenMP kernel ``kbc_channel_kernel.c`` is a straight port of it and
``tests/test_kbc_channel.py`` pins the two together population by population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CS2",
    "KBCDiagnostics",
    "Lattice",
    "collide_bgk",
    "collide_kbc",
    "decompose",
    "entropy",
    "equilibrium",
    "ghost_alpha",
    "lattice",
    "second_moments",
    "shear_rate_tensor",
]

CS2 = 1.0 / 3.0
#: population floor used by the positivity safeguard (kernel keeps a 1e-30 float floor)
GROUND = 1e-12
_FACTOR = 4.5  # 9/2 : the Hermite projector prefactor 1/(2 cs^4)


@dataclass(frozen=True)
class Lattice:
    """Velocity set, weights and the projector basis for one lattice."""

    q: int
    c: np.ndarray  # (Q, 3) int
    w: np.ndarray  # (Q,) float
    h_dev: np.ndarray  # (Q, 3, 3) c_a c_b - |c|^2/3 delta_ab   (traceless)
    h_full: np.ndarray  # (Q, 3, 3) c_a c_b - cs2 delta_ab
    c_sq: np.ndarray  # (Q,) |c|^2


def _weights(c: np.ndarray) -> np.ndarray:
    """Qian weights from the velocity set (D2Q9 / D3Q19 / D3Q27)."""
    csq = (c.astype(np.float64) ** 2).sum(axis=1)
    w = np.where(
        csq == 0,
        8.0 / 27.0 if c.shape[1] == 3 else 4.0 / 9.0,
        np.where(
            csq == 1,
            2.0 / 27.0 if c.shape[1] == 3 else 1.0 / 9.0,
            np.where(csq == 2, 1.0 / 54.0 if c.shape[1] == 3 else 1.0 / 36.0, 1.0 / 216.0),
        ),
    )
    assert abs(w.sum() - 1.0) < 1e-14, w.sum()
    return w


_LATTICE_CACHE: dict[int, Lattice] = {}


def lattice(q: int = 27) -> Lattice:
    """Return the :class:`Lattice` for ``q`` in {19, 27} (D3Q27 default)."""
    if q in _LATTICE_CACHE:
        return _LATTICE_CACHE[q]
    if q == 27:
        # Repo-canonical D3Q27 ordering (identical to src/tensorlbm/d3q27.py and
        # to the CX/CY/CZ tables of the kernels): rest, 6 faces, 12 edges,
        # 8 corners.  Keeping it identical is what makes the kernel and this
        # reference interchangeable population by population.
        c = np.array(
            [
                (0, 0, 0),
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
            ],
            dtype=np.int64,
        )
    elif q == 19:
        c = np.array(
            [[x, y, z] for x, y, z in _D3Q19_DIRS], dtype=np.int64
        )
    else:  # pragma: no cover - guarded
        raise ValueError(f"unsupported lattice q={q}")
    w = _weights(c)
    cf = c.astype(np.float64)
    csq = (cf**2).sum(axis=1)
    h_full = cf[:, :, None] * cf[:, None, :] - CS2 * np.eye(3)[None, :, :]
    h_dev = cf[:, :, None] * cf[:, None, :] - (csq / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    lat = Lattice(q=q, c=c, w=w, h_dev=h_dev, h_full=h_full, c_sq=csq)
    _LATTICE_CACHE[q] = lat
    return lat


_D3Q19_DIRS = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
]


def equilibrium(rho: np.ndarray, u: np.ndarray, lat: Lattice | None = None) -> np.ndarray:
    """Second-order (quadratic) equilibrium, shape ``(Q, *grid)``.

    ``rho`` has shape ``(*grid,)``, ``u`` has shape ``(3, *grid)``.
    """
    lat = lat or lattice(27)
    cu = np.einsum("qa,a...->q...", lat.c.astype(np.float64), u)
    usq = (u**2).sum(axis=0)
    feq = lat.w.reshape((-1,) + (1,) * usq.ndim) * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None]
    )
    return feq


def moments(f: np.ndarray, lat: Lattice | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Density and velocity (``rho`` shape ``(*grid,)``, ``u`` shape ``(3,*grid)``)."""
    lat = lat or lattice(27)
    c = lat.c.astype(np.float64)
    rho = f.sum(axis=0)
    u = np.einsum("qa,q...->a...", c, f) / rho[None]
    return rho, u


def second_moments(f: np.ndarray, lat: Lattice | None = None) -> np.ndarray:
    """``Pi_ab = sum_i c_ia c_ib f_i`` with shape ``(3, 3, *grid)``."""
    lat = lat or lattice(27)
    c = lat.c.astype(np.float64)
    return np.einsum("qa,qb,q...->ab...", c, c, f)


def decompose(
    f_neq: np.ndarray, lat: Lattice | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``f_neq`` exactly into shear ``s``, bulk ``b`` and ghost ``g``.

    The three projections are Hermite-orthogonal: ``s`` carries the whole
    traceless second moment, ``b`` the whole trace of the second moment, ``g``
    nothing of second order.  ``s + b + g == f_neq`` to round-off.
    """
    lat = lat or lattice(27)
    pi = second_moments(f_neq, lat)
    pi_tr = np.trace(pi, axis1=0, axis2=1)
    pi_dev = pi.copy()
    for a in range(3):
        pi_dev[a, a] -= pi_tr / 3.0

    w = lat.w.reshape((-1,) + (1,) * pi_tr.ndim)
    s = _FACTOR * w * np.einsum("qab,ab...->q...", lat.h_dev, pi_dev)
    f2 = _FACTOR * w * np.einsum("qab,ab...->q...", lat.h_full, pi)
    b = f2 - s
    g = f_neq - f2
    return s, b, g


def entropy(f: np.ndarray, lat: Lattice | None = None, floor: float = 1e-30) -> np.ndarray:
    """Discrete Boltzmann entropy ``H = sum_i f_i ln(f_i / w_i)`` (per cell)."""
    lat = lat or lattice(27)
    w = lat.w.reshape((-1,) + (1,) * (f.ndim - 1))
    fs = np.maximum(f, floor)
    return (fs * np.log(fs / w)).sum(axis=0)


def shear_rate_tensor(
    f: np.ndarray, tau: float, lat: Lattice | None = None
) -> np.ndarray:
    """Strain-rate tensor from the non-equilibrium stress, ``S_ab = -3 Pi^neq_ab /(2 rho tau)``.

    Used by the statistics layer only (it is the standard LBM estimator that
    avoids finite-differencing the noisy velocity field).
    """
    lat = lat or lattice(27)
    rho, u = moments(f, lat)
    pi = second_moments(f - equilibrium(rho, u, lat), lat)
    pi_tr = np.trace(pi, axis1=0, axis2=1)
    pi_dev = pi.copy()
    for a in range(3):
        pi_dev[a, a] -= pi_tr / 3.0
    return -1.5 * pi_dev / (rho[None, None] * tau)


# ---------------------------------------------------------------------------
# entropy-condition solve for the ghost segment
# ---------------------------------------------------------------------------


@dataclass
class KBCDiagnostics:
    """Per-cell (and reduced) diagnostics of one collision step."""

    alpha: np.ndarray
    delta_h: np.ndarray
    gamma: np.ndarray | None = None
    alpha_iters: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=np.int64))
    limiter_events: int = 0  # global entropy-limiter fallbacks (should be ~0)
    min_population: float = float("nan")
    max_shear_rate: float = float("nan")


def ghost_alpha(
    f: np.ndarray,
    base: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    g: np.ndarray,
    *,
    omega_nu: float,
    omega_b: float,
    omega_g: float = 1.0,
    alpha_cap: float = 2.0,
    alpha_min: float = 0.0,
    n_iter: int = 24,
    tol: float = 1e-10,
    lat: Lattice | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monotone entropy-condition solve for the ghost relaxation ``alpha``.

    ``H(alpha)`` is convex in ``alpha``; the returned ``alpha`` is the minimiser
    of ``H`` over the positivity interval ``[0, min(alpha_cap, alpha_pos))``,
    found by Newton on

        ``H'(alpha) = -omega_g sum_i g_i ln(f*_i(alpha)/w_i)``
        ``H''(alpha) =  omega_g^2 sum_i g_i^2 / f*_i(alpha)   > 0``

    with a bisection safeguard.  When ``H'(0) >= 0`` (the entropy already
    increases at ``alpha = 0``) the minimiser is ``alpha = 0``.

    Returns ``(alpha, delta_h, iters)`` where ``delta_h = H(f*(alpha)) - H(f)``
    per cell and ``iters`` is the Newton/bisection iteration count per cell.
    """
    lat = lat or lattice(27)
    w = lat.w.reshape((-1,) + (1,) * (f.ndim - 1))
    base = base + (1.0 - omega_nu) * s + (1.0 - omega_b) * b + g  # f*(0)

    # positivity bracket for alpha: f*(alpha) = base - alpha*d > 0.
    # The ghost rate is omega_g * omega_nu (i.e. omega_g is a *multiple of the
    # BGK rate*), because the discrete viscosity carries a ghost-mode
    # contribution: with the ghost relaxation tied to 1/tau the alpha = 1 case
    # is exactly BGK (nu = cs2 (tau - 1/2)); a ghost rate that is independent of
    # tau shifts the effective viscosity by O(1) (measured: cs2*tau instead of
    # cs2*(tau-1/2) at omega_g = 1 and tau = 0.73).
    d = -omega_g * omega_nu * g
    shape = f.shape[1:]
    alpha_pos = np.full(shape, np.inf)
    neg = d < 0.0
    if neg.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(neg, base / np.where(neg, -d, 1.0), np.inf)
        alpha_pos = np.minimum(alpha_pos, bound.min(axis=0))
    hi = np.minimum(alpha_cap, alpha_pos)
    hi = np.where(np.isfinite(hi), hi, alpha_cap)
    hi = np.maximum(hi, 0.0)

    def hent(a: np.ndarray) -> np.ndarray:
        fa = np.maximum(base + a[None] * d, 1e-30)
        return (fa * np.log(fa / w)).sum(axis=0)

    def dH(a: np.ndarray) -> np.ndarray:
        fa = np.maximum(base + a[None] * d, 1e-30)
        return (d * np.log(fa / w)).sum(axis=0)

    def d2H(a: np.ndarray) -> np.ndarray:
        fa = np.maximum(base + a[None] * d, 1e-30)
        return (d * d / fa).sum(axis=0)

    lo = np.zeros(shape)
    a = np.clip(
        np.where(d2H(lo) > 0, -dH(lo) / np.maximum(d2H(lo), 1e-300), 0.0), 0.0, hi
    )
    active = (dH(lo) < 0.0) & (hi > 0.0)
    # If H' <= 0 over the whole admissible interval the minimum sits at the
    # positivity bound itself.
    at_bound = active & (dH(np.maximum(hi, 1e-30)) <= 0.0)
    a = np.where(active, np.where(at_bound, hi, a), 0.0)
    todo = active & (~at_bound)
    iters = np.zeros(shape, dtype=np.int32)
    # Safeguarded Newton on H'(alpha) = 0: H is convex, so the sign of H'
    # brackets the minimiser and the Newton step is accepted only inside the
    # bracket (otherwise bisect).
    for _ in range(n_iter):
        if not todo.any():
            break
        iters += todo
        ga = dH(a)
        ha = d2H(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = np.where(ha > 0, a - ga / np.maximum(ha, 1e-300), 0.0)
        lo = np.where(todo & (ga < 0.0), a, lo)
        hi = np.where(todo & (ga > 0.0), a, hi)
        cand = np.where(np.isfinite(newton) & (newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
        converged = np.abs(cand - a) <= tol * (1.0 + np.abs(a))
        a = np.where(todo, cand, a)
        todo = todo & (~converged)
    a = np.clip(a, 0.0, hi)
    if alpha_min > 0.0:
        a = np.clip(a, alpha_min, np.maximum(hi, alpha_min))
    # NOTE: the returned alpha is the entropy-conditioned *ghost* rate for the
    # fan-out admissible interval; whether the resulting state actually lowers
    # H is verified one level up (collide_kbc / the kernel's enforce_entropy),
    # which is also where the positivity scale gamma is applied.  Keeping the
    # verification in one place is what makes the NumPy reference and the
    # OpenMP kernel agree population by population.
    delta_h = hent(a) - hent(np.zeros(shape))
    return a, delta_h, iters


def collide_kbc(
    f: np.ndarray,
    tau: float,
    *,
    tau_bulk: float | None = None,
    alpha_mode: str = "entropy",
    alpha_value: float = 1.0,
    alpha_cap: float = 2.0,
    alpha_min: float = 0.0,
    omega_g: float = 1.0,
    n_iter: int = 24,
    fallback: bool = True,
    lat: Lattice | None = None,
) -> tuple[np.ndarray, KBCDiagnostics]:
    """Three-segment KBC collision.

    Args:
        f: populations, shape ``(Q, *grid)``.
        tau: shear relaxation time, ``nu = cs2 (tau - 1/2)``.
        tau_bulk: bulk relaxation time (defaults to ``tau``).
        alpha_mode: ``"entropy"`` (monotone entropy solve) or ``"fixed"``.
        alpha_value: used when ``alpha_mode == "fixed"``.
        alpha_cap: upper bound of the entropy solve (``alpha > 1`` allowed).
        omega_g: ghost rate as a multiple of the BGK rate ``1/tau`` (``alpha =
            1, omega_g = 1`` reproduces BGK exactly).
        n_iter: Newton/bisection iterations of the entropy solve.
        fallback: if even the entropy-optimal ``alpha`` fails to lower ``H``,
            blend the whole collision towards ``f`` (ELBM-style safety net).
            The number of firings is reported; a healthy run has zero.

    Returns:
        ``(f_post, diagnostics)``.
    """
    lat = lat or lattice(int(f.shape[0]))
    assert f.shape[0] == lat.q
    omega_nu = 1.0 / tau
    tau_b = tau if tau_bulk is None else tau_bulk
    omega_b = 1.0 / tau_b

    rho, u = moments(f, lat)
    feq = equilibrium(rho, u, lat)
    f_neq = f - feq
    s, b, g = decompose(f_neq, lat)

    if alpha_mode == "entropy":
        alpha, delta_h, iters = ghost_alpha(
            f, feq, s, b, g,
            omega_nu=omega_nu, omega_b=omega_b, omega_g=omega_g,
            alpha_cap=alpha_cap, alpha_min=alpha_min, n_iter=n_iter, lat=lat,
        )
    elif alpha_mode == "fixed":
        alpha = np.full(f.shape[1:], float(alpha_value))
        iters = np.zeros(f.shape[1:], dtype=np.int32)
    else:  # pragma: no cover - guarded
        raise ValueError(f"unknown alpha_mode {alpha_mode!r}")

    f_post = feq + (1.0 - omega_nu) * s + (1.0 - omega_b) * b + (1.0 - alpha[None] * omega_g * omega_nu) * g

    # ------------------------------------------------------------------
    # positivity + entropy safeguard (per cell, one bisection on gamma)
    #
    # f*(gamma) = f + gamma (f_post - f).  Both constraints are interval
    # conditions on gamma: positivity is linear in gamma and H(gamma) is
    # convex in gamma with H(0) = H(f), so {gamma : H(gamma) <= H(f)} is an
    # interval containing 0.  Bisecating gamma keeps the *shear* rate intact
    # wherever the limiter is inactive (gamma = 1) and degrades gracefully
    # where the state is too far from equilibrium for the exact-rate
    # collision to stay admissible.
    # ------------------------------------------------------------------
    gamma = np.ones(f.shape[1:])
    d = f_post - f
    if fallback:
        neg = d < 0.0
        if neg.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                bound = np.where(neg, (f - GROUND) / np.where(neg, -d, 1.0), np.inf)
            gamma = np.minimum(gamma, np.maximum(bound.min(axis=0), 0.0))
        h0 = entropy(f, lat)

        def h_of(g: np.ndarray) -> np.ndarray:
            return entropy(f + g[None] * d, lat) - h0

        hi = gamma
        ok = h_of(hi) <= 0.0
        gamma = np.where(ok, hi, 0.0)
        todo = ~ok
        lo = np.zeros(f.shape[1:])
        for _ in range(24):
            if not todo.any():
                break
            mid = 0.5 * (lo + hi)
            hmid = h_of(mid)
            good = hmid <= 0.0
            lo = np.where(todo & good, mid, lo)
            hi = np.where(todo & (~good), mid, hi)
            gamma = np.where(todo & good, mid, gamma)
            todo = todo & ((hi - lo) > 1e-12 * (1.0 + np.abs(hi)))
        limiter_events = int((gamma < 1.0 - 1e-12).sum())
        f_post = f + gamma[None] * d
    else:
        limiter_events = 0

    delta_h = entropy(f_post, lat) - entropy(f, lat)
    diag = KBCDiagnostics(
        alpha=alpha,
        delta_h=delta_h,
        alpha_iters=iters,
        gamma=gamma,
        limiter_events=limiter_events,
        min_population=float(f_post.min()),
        max_shear_rate=float(
            np.sqrt((shear_rate_tensor(f, tau, lat) ** 2).sum(axis=(0, 1))).max()
        ),
    )
    return f_post, diag


def collide_bgk(f: np.ndarray, tau: float, lat: Lattice | None = None) -> np.ndarray:
    """Plain BGK collision (baseline / cross-check)."""
    lat = lat or lattice(int(f.shape[0]))
    rho, u = moments(f, lat)
    feq = equilibrium(rho, u, lat)
    return f + (feq - f) / tau