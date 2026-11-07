"""Discrete linear stability analysis of the D2Q9/D2Q5 Rayleigh-Bénard lattice
Boltzmann operator (node IN31-N12, part 1).

Why this module exists
----------------------
``benchmarks/pending/rayleigh_benard/lattice_predictions.json`` carries the
``gamma_pred`` reference values used by the W12 re-lock of the Rayleigh-Bénard
(RB) benchmark.  Those numbers were *not* reproducible from the repository: the
reference growth rates came from a continuous-spectrum Chebyshev eigenvalue
solve of the Navier-Stokes-Boussinesq equations, and when the lattice operator's
own critical Rayleigh number was used instead, the near-critical growth rates
still carried a systematic offset of a few percent
(``docs/reports/report_rayleigh_benard.md``).

This module closes that gap: it computes the linear growth rate ``gamma(Ra)``
of *the same lattice operator that the benchmark actually time-marches*, by
assembling (matrix-free) its Jacobian about the discrete conductive base state
and solving the resulting eigenvalue problem.

Method
------
1. **Base state.**  The conductive state ``T(y) = T_hot - dT (y+1/2)/H``,
   ``u = 0`` is a fixed point of the lattice operator including the hydrostatic
   response to the Boussinesq force.  It is obtained by iterating the step map
   from an *x-uniform* initial field: RB instability modes vary along ``x``, so
   the conductive state is stable inside the x-uniform subspace even when it is
   unstable to rolls.  That lets the same routine return the base state on both
   sides of criticality.
2. **Jacobian.**  The step map ``F: (f, g) -> (f', g')`` is the composition of
   library primitives that the benchmark uses (``thermal.temperature_collision``
   / ``temperature_stream`` / anti-bounce-back walls, ``thermal.buoyancy_force``,
   ``thermal.collide_bgk_force`` (Guo forcing), ``thermal.pre_streaming_bounce_back``,
   ``solver.stream``).  Its Jacobian at the base state is applied matrix-free by
   a central-free finite difference ``J v ~= (F(x0 + eps v) - F(x0))/eps`` in
   float64, so no hand-derived operator can drift away from the code under test.
3. **Eigenvalues.**  Arnoldi (``scipy.sparse.linalg.eigs``) on that Jacobian
   returns the leading multiplier ``lambda``; the growth rate per lattice step
   is ``gamma = ln|lambda|`` and the non-dimensional growth rate used for
   comparison with the continuous spectrum is ``gamma_hat = gamma H^2 / nu``
   (time unit ``H^2/nu``, ``nu = (tau-1/2)/3``).

Conventions match ``benchmarks/pending/rayleigh_benard/run.py``: gap
``H = ny`` lattice units with half-way walls at ``y = -1/2`` and ``y = ny-1/2``,
unit temperature difference ``dT = 1``, ``Ra = g_beta dT H^3/(nu alpha)``,
``Pr = nu/alpha`` and ``g_beta = Ra nu alpha / H^3``.

Verification
------------
* ``measure_growth_rate`` time-marches the *non-linear* library solver from the
  same initial perturbation and fits the growth rate; ``test_rb_discrete_stability``
  asserts that it matches the eigen-solver to a few percent.  That is the check
  that the assembled operator really is the operator the benchmark runs.
* ``critical_rayleigh`` reproduces the documented lattice shift of ``Ra_c``
  (+0.15% at ``ny=64``) and converges to the continuous value ``1707.76``
  (Chandrasekhar 1961) as ``ny`` grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is a package dependency; import locally nonetheless
    from scipy import sparse

    _SCIPY_OK = True
    _SCIPY_ERR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    sparse = None
    eigs = None
    _SCIPY_OK = False
    _SCIPY_ERR = exc

RA_CRIT_CONTINUOUS = 1707.7597  # Chandrasekhar (1961), rigid-rigid, 2-D rolls
A_CRIT_CONTINUOUS = 3.11632  # critical non-dimensional wavenumber a_c = k H


def _require_scipy() -> None:
    if not _SCIPY_OK:  # pragma: no cover
        raise RuntimeError(f"scipy is required for rb_discrete_stability: {_SCIPY_ERR}")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RBLatticeParams:
    """Lattice + physical parameters of the benchmark Rayleigh-Bénard set-up.

    Attributes
    ----------
    ny, nx:  vertical / horizontal lattice size.  The gap is ``H = ny`` lattice
             units and the horizontal extent is ``nx`` (periodic), so the only
             admissible horizontal wavenumbers are ``k = 2 pi m / nx``.
    Ra:      Rayleigh number ``g_beta dT H^3 / (nu alpha)``.
    tau:     D2Q9 relaxation time; ``nu = (tau - 1/2)/3``.
    tau_T:   D2Q5 relaxation time; ``alpha = (tau_T - 1/2)/3``.
    T_hot, T_cold:  wall temperatures (``dT = T_hot - T_cold``).
    mode:    horizontal mode number ``m`` (default 1 = one roll pair).
    """

    ny: int = 16
    nx: int = 32
    Ra: float = 1707.7597
    tau: float = 0.855
    tau_T: float = 1.0
    T_hot: float = 1.0
    T_cold: float = 0.0
    mode: int = 1

    # ---- derived quantities -------------------------------------------------
    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def alpha(self) -> float:
        return (self.tau_T - 0.5) / 3.0

    @property
    def Pr(self) -> float:
        return self.nu / self.alpha

    @property
    def H(self) -> float:
        return float(self.ny)

    @property
    def dT(self) -> float:
        return self.T_hot - self.T_cold

    @property
    def g_beta(self) -> float:
        return self.Ra * self.nu * self.alpha / (self.H**3 * self.dT)

    @property
    def a(self) -> float:
        """Non-dimensional wavenumber ``k H`` of the resolved horizontal mode."""
        return 2.0 * math.pi * self.mode * self.H / self.nx

    @property
    def dof(self) -> int:
        return (9 + 5) * self.ny * self.nx

    def with_Ra(self, Ra: float) -> "RBLatticeParams":
        return replace(self, Ra=float(Ra))

    def as_dict(self) -> Dict[str, float]:
        return {
            "ny": self.ny,
            "nx": self.nx,
            "mode": self.mode,
            "Ra": self.Ra,
            "tau": self.tau,
            "tau_T": self.tau_T,
            "nu": self.nu,
            "alpha": self.alpha,
            "Pr": self.Pr,
            "H": self.H,
            "g_beta": self.g_beta,
            "a": self.a,
        }


# ---------------------------------------------------------------------------
# The lattice operator (identical composition to the benchmark loop)
# ---------------------------------------------------------------------------
def _d2q9_d2q5():
    from .d2q9 import equilibrium, macroscopic
    from .solver import stream
    from .thermal import (
        W5,
        buoyancy_force,
        collide_bgk_force,
        pre_streaming_bounce_back,
        temperature_collision,
        temperature_equilibrium,
        temperature_stream,
    )

    return {
        "equilibrium": equilibrium,
        "macroscopic": macroscopic,
        "stream": stream,
        "W5": W5,
        "buoyancy_force": buoyancy_force,
        "collide_bgk_force": collide_bgk_force,
        "pre_streaming_bounce_back": pre_streaming_bounce_back,
        "temperature_collision": temperature_collision,
        "temperature_equilibrium": temperature_equilibrium,
        "temperature_stream": temperature_stream,
    }


def wall_mask(ny: int, nx: int, device=None):
    """Top/bottom no-slip rows (half-way walls), as in the benchmark."""
    import torch

    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def temperature_boundaries(g, t_hot: float, t_cold: float):
    """Bottom hot / top cold anti-bounce-back Dirichlet rows (half-way walls)."""
    L = _d2q9_d2q5()
    w = L["W5"].to(g.device)
    g = g.clone()
    g[3, 0, :] = -g[4, 0, :] + 2.0 * w[3] * t_hot  # bottom, wall at y = -1/2
    g[4, -1, :] = -g[3, -1, :] + 2.0 * w[4] * t_cold  # top, wall at y = ny - 1/2
    return g


def rb_step(f, g, p: RBLatticeParams, wall=None):
    """One full RB lattice-Boltzmann step ``(f, g) -> (f', g')``.

    Order of operations is byte-for-byte the benchmark order
    (``benchmarks/pending/rayleigh_benard/run.py``): buoyancy from the current
    fields, thermal collision + stream + anti-bounce-back walls, then D2Q9 Guo
    collision, pre-streaming bounce-back, periodic stream.
    """
    L = _d2q9_d2q5()
    if wall is None:
        wall = wall_mask(p.ny, p.nx, f.device)
    rho, ux, uy = L["macroscopic"](f)
    T = g.sum(dim=0)
    F = L["buoyancy_force"](rho, T, p.g_beta, t_ref=0.5 * (p.T_hot + p.T_cold))
    g = L["temperature_collision"](g, p.tau_T, ux, uy)
    g = L["temperature_stream"](g)
    g = temperature_boundaries(g, p.T_hot, p.T_cold)
    f_pre = f
    f = L["collide_bgk_force"](f, p.tau, F)
    f = L["pre_streaming_bounce_back"](f_pre, f, wall)
    f = L["stream"](f)
    return f, g


# ---------------------------------------------------------------------------
# Base state
# ---------------------------------------------------------------------------
@dataclass
class BaseState:
    f: "object"  # torch.Tensor (9, ny, nx), float64
    g: "object"  # torch.Tensor (5, ny, nx), float64
    residual: float  # max |x_{n+1} - x_n| at the last iteration
    steps: int

    def vector(self):
        import torch

        return torch.cat([self.f.reshape(-1), self.g.reshape(-1)])


def _initial_conduction(p: RBLatticeParams, device, dtype):
    """x-uniform conductive start: linear T, rest fluid, unit density."""
    import torch

    L = _d2q9_d2q5()
    y = torch.arange(p.ny, device=device, dtype=dtype)
    T1d = p.T_hot - p.dT * (y + 0.5) / p.H
    T = T1d.view(p.ny, 1).expand(p.ny, p.nx).clone()
    zero = torch.zeros((p.ny, p.nx), device=device, dtype=dtype)
    f = L["equilibrium"](torch.ones_like(T), zero, zero)
    g = L["temperature_equilibrium"](T, zero, zero)
    return f, g


def conductive_base_state(
    p: RBLatticeParams,
    steps: int = 4000,
    tol: float = 1e-12,
    device: str = "cpu",
    dtype=None,
    warm_start: Optional[BaseState] = None,
) -> BaseState:
    """Iterate the step map inside the x-uniform subspace to the conductive state.

    The RB instability modes carry horizontal structure, so inside the subspace
    of x-uniform fields the conductive state stays (linearly) stable for every
    ``Ra``; this is what lets the routine return the base state used for the
    Jacobian on both sides of criticality.  The residual reported is the largest
    per-iteration field change, i.e. a direct measure of how well the returned
    state is a fixed point.
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    device = torch.device(device)
    # An x-uniform field evolves identically for any horizontal box width, so the
    # conductive base state is computed on a single column (nx = 1) -- a >30x
    # speed-up with no approximation -- and then tiled to the target grid.
    p1 = replace(p, nx=1)
    if warm_start is not None:
        f = warm_start.f[:, :, :1].contiguous()
        g = warm_start.g[:, :, :1].contiguous()
    else:
        f, g = _initial_conduction(p1, device, dtype)
    wall = wall_mask(p1.ny, p1.nx, device)
    resid = float("inf")
    for _ in range(int(steps)):
        f_new, g_new = rb_step(f, g, p1, wall)
        resid = max(
            float((f_new - f).abs().max().item()),
            float((g_new - g).abs().max().item()),
        )
        f, g = f_new, g_new
        if resid < tol:
            break
    if p.nx > 1:
        f = f.detach().clone().expand(9, p.ny, p.nx).contiguous()
        g = g.detach().clone().expand(5, p.ny, p.nx).contiguous()
    return BaseState(
        f=f.detach().clone(), g=g.detach().clone(), residual=resid, steps=int(steps)
    )


# ---------------------------------------------------------------------------
# Module-block Jacobian and its spectrum
# ---------------------------------------------------------------------------
# The base state and the walls are uniform along x, so the Jacobian of the step
# map is block-diagonal in the horizontal Fourier basis: a perturbation with
# horizontal wavenumber k = 2 pi m / nx stays in that wavenumber.  Keeping only
# the block m removes the (x-uniform) mass/energy conservation modes whose
# multiplier is exactly 1 and which would otherwise mask the physical roll mode
# in a modulus-ordered Arnoldi solve -- precisely the difficulty that makes a
# naive "largest |lambda|" lattice stability analysis useless for subcritical
# decay rates.
def _mode_field(V: np.ndarray, p: RBLatticeParams, m: int):
    """Real field ``Re[ V(y) exp(i k x) ]`` for each channel (14, ny, nx)."""
    import torch

    k = 2.0 * math.pi * m / p.nx
    x = torch.arange(p.nx, dtype=torch.float64) * k
    cosx, sinx = torch.cos(x), torch.sin(x)
    Vt = torch.as_tensor(np.asarray(V, dtype=np.complex128)).reshape(14, p.ny)
    re_v, im_v = Vt.real, Vt.imag
    field = re_v.unsqueeze(-1) * cosx + (-im_v).unsqueeze(-1) * sinx
    return field.reshape(-1, p.ny, p.nx)


def _extract_mode(field, p: RBLatticeParams, m: int) -> np.ndarray:
    """Complex Fourier amplitude at mode ``m`` of a real ``(14, ny, nx)`` field."""
    import torch

    arr = field.reshape(14, p.ny, p.nx)
    spec = torch.fft.fft(arr, dim=-1)
    return (2.0 * spec[:, :, m] / p.nx).numpy().reshape(-1)


def _interior_index(p: RBLatticeParams):
    """Indices (within the 14 ny flat channel-row vector) of interior rows.

    Wall node rows (``y = 0`` and ``y = ny-1``) are excluded when
    ``interior_only`` is requested: the library's bounce-back / anti-bounce-back
    *substitutes* those rows (reflection of the pre-collision value) instead of
    relaxing them, so as state variables they carry a family of multipliers
    exactly on the unit circle that has nothing to do with the physics.
    """
    keep = []
    for c in range(14):
        for y in range(1, p.ny - 1):
            keep.append(c * p.ny + y)
    return np.asarray(keep, dtype=np.int64)


def mode_block_matrix(
    p: RBLatticeParams,
    base: Optional[BaseState] = None,
    mode: Optional[int] = None,
    eps: float = 1e-8,
    interior_only: bool = False,
) -> Tuple[np.ndarray, BaseState]:
    """Dense complex Jacobian block of the lattice operator at wavenumber ``k``.

    Dimension is ``14 ny`` (9 D2Q9 populations + 5 D2Q5 populations per vertical
    row).  The block is assembled column by column: column ``j`` is the Fourier
    amplitude at mode ``m`` of the response to the real perturbation
    ``Re[ e_j exp(i k x) ]``, so a single step of the *library* map per column
    is enough (the block is complex-linear).
    """
    import torch

    if base is None:
        base = conductive_base_state(p)
    if mode is None:
        mode = p.mode
    n_full = 14 * p.ny
    keep = _interior_index(p) if interior_only else np.arange(n_full)
    n = int(keep.size)
    with torch.no_grad():
        f_next, g_next = rb_step(base.f, base.g, p)
    base_next = torch.cat([f_next.reshape(-1), g_next.reshape(-1)])
    J = np.zeros((n, n), dtype=np.complex128)
    with torch.no_grad():
        for j in range(n):
            e = np.zeros(n_full, dtype=np.complex128)
            e[keep[j]] = 1.0
            field = _mode_field(e, p, mode)
            f = base.f + eps * field[:9]
            g = base.g + eps * field[9:]
            f1, g1 = rb_step(f, g, p)
            out = torch.cat([f1.reshape(-1), g1.reshape(-1)]) - base_next
            col = _extract_mode(out / eps, p, mode)
            if not np.all(np.isfinite(col[keep])):
                raise FloatingPointError(
                    "the lattice step map produced non-finite values: Ra is too high for "
                    f"this resolution (ny={p.ny}, Ra={p.Ra}); the base state itself diverges"
                )
            J[:, j] = col[keep]
    return J, base


def wall_weights(vec: np.ndarray, ny: int, interior_only: bool = False) -> float:
    """Fraction of an eigenvector's weight sitting on the wall node rows.

    With ``interior_only`` the state has no wall rows, so the weight is 0.
    """
    if interior_only:
        return 0.0
    arr = np.abs(np.asarray(vec, dtype=np.complex128)).reshape(14, ny)
    tot = arr.sum()
    if tot == 0.0:
        return 0.0
    return float((arr[:, 0].sum() + arr[:, -1].sum()) / tot)


def split_spectrum(
    matrix: np.ndarray,
    ny: int,
    unit_tol: float = 1e-7,
    wall_frac: float = 0.85,
    interior_only: bool = False,
):
    """Separate physically meaningful multipliers from unit-circle artifacts.

    The library's half-way bounce-back and anti-bounce-back boundaries *replace*
    the two wall-node rows (reflection of the pre-collision value, Dirichlet
    substitution) instead of relaxing them.  Those rows therefore act as
    algebraic boundary data rather than as independent degrees of freedom, and
    the linearised operator carries a family of multipliers sitting exactly on
    the unit circle (angle 0 or pi, ``| |lambda| - 1 | < 1e-9``) with no physical
    content: they exist for every ``Ra`` and are unaffected by resolution or by
    the physics.  Ordered by ``|lambda|`` alone they mask the roll mode, which is
    why a naive lattice stability analysis reports ``gamma = 0`` for every
    Rayleigh number.

    Criterion: an eigenvalue is an artifact iff ``| |lambda| - 1 | < unit_tol``.
    Typical physical multipliers near onset are ``|lambda| - 1 ~ gamma H^2/nu``,
    i.e. ``1e-5`` at ``frac = 0.95`` and ``1e-7`` only within ``1e-4`` of
    criticality, so the default ``unit_tol = 1e-7`` is safe (and the artifact
    family is exact to ``1e-12`` in float64, measured).

    Returns ``(physical, artifact)``, each a list of
    ``(eigenvalue, wall_weight)`` sorted by decreasing ``|lambda|``.
    """
    vals, vecs = np.linalg.eig(matrix)
    phys, ghost = [], []
    for i, lam in enumerate(vals):
        w = wall_weights(vecs[:, i], ny, interior_only=interior_only)
        rec = (complex(lam), w)
        if abs(abs(lam) - 1.0) < unit_tol:
            ghost.append(rec)
        else:
            phys.append(rec)
    phys.sort(key=lambda r: -abs(r[0]))
    ghost.sort(key=lambda r: -abs(r[0]))
    return phys, ghost


def growth_rate(
    p: RBLatticeParams,
    base: Optional[BaseState] = None,
    mode: Optional[int] = None,
    eps: float = 1e-8,
    matrix: Optional[np.ndarray] = None,
    wall_frac: float = 0.85,
    unit_tol: float = 1e-7,
    interior_only: bool = False,
    return_spectrum: bool = False,
) -> Dict[str, object]:
    """Discrete growth rate of the leading physical mode.

    ``gamma_step = ln|lambda_max|`` per lattice step and
    ``gamma_hat = gamma_step H^2/nu`` for comparison with the continuous
    spectrum (time unit ``H^2/nu``).  Wall-node ghost modes are removed first,
    see :func:`split_spectrum`.
    """
    if matrix is None:
        matrix, base = mode_block_matrix(
            p, base=base, mode=mode, eps=eps, interior_only=interior_only
        )
    if base is None:  # pragma: no cover - defensive
        base = conductive_base_state(p)
    phys, ghost = split_spectrum(
        matrix, p.ny, wall_frac=wall_frac, unit_tol=unit_tol, interior_only=interior_only
    )
    lam = phys[0][0]
    gamma_step = math.log(abs(lam))
    out: Dict[str, object] = {
        **p.as_dict(),
        "mode": int(mode if mode is not None else p.mode),
        "lambda_lead": [lam.real, lam.imag],
        "gamma_step": gamma_step,
        "gamma_hat": gamma_step * p.H**2 / p.nu,
        "lambda_abs": abs(lam),
        "mode_argument": math.atan2(lam.imag, lam.real),
        "wall_weight_lead": phys[0][1],
        "base_residual": base.residual,
        "block_dim": int(matrix.shape[0]),
        "n_ghost_modes": len(ghost),
        "top_physical": [[r[0].real, r[0].imag, r[1]] for r in phys[:5]],
    }
    if return_spectrum:
        out["spectrum"] = {
            "physical": [[r[0].real, r[0].imag, r[1]] for r in phys],
            "ghost": [[r[0].real, r[0].imag, r[1]] for r in ghost],
        }
    return out


# ---------------------------------------------------------------------------
# Critical Rayleigh number by bisection on gamma(Ra)
# ---------------------------------------------------------------------------
def critical_rayleigh(
    p: RBLatticeParams,
    Ra_lo: float = 1000.0,
    Ra_hi: float = 3000.0,
    tol: float = 1e-4,
    max_iter: int = 22,
    **growth_kwargs,
):
    """Bisect ``gamma(Ra) = 0`` on the discrete operator.

    Returns ``{"Ra_c": ..., "gamma_lo": ..., "gamma_hi": ..., "history": [...]}``
    with ``gamma`` per lattice step.  The bracket is validated first: at
    ``Ra_lo`` the conductive state must decay and at ``Ra_hi`` it must grow.
    """
    lo, hi = float(Ra_lo), float(Ra_hi)
    g_lo = growth_rate(p.with_Ra(lo), **growth_kwargs)["gamma_step"]
    g_hi = growth_rate(p.with_Ra(hi), **growth_kwargs)["gamma_step"]
    if g_lo >= 0.0:
        raise ValueError(f"Ra_lo={lo} is already unstable (gamma={g_lo:+.3e}); widen the bracket")
    if g_hi <= 0.0:
        raise ValueError(f"Ra_hi={hi} is still stable (gamma={g_hi:+.3e}); widen the bracket")
    history = [{"Ra": lo, "gamma_step": g_lo}, {"Ra": hi, "gamma_step": g_hi}]
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        g_mid = growth_rate(p.with_Ra(mid), **growth_kwargs)["gamma_step"]
        history.append({"Ra": mid, "gamma_step": g_mid})
        if g_mid < 0.0:
            lo, g_lo = mid, g_mid
        else:
            hi, g_hi = mid, g_mid
        if hi - lo <= tol * mid:
            break
    return {
        "Ra_c": 0.5 * (lo + hi),
        "Ra_bracket": [lo, hi],
        "gamma_lo": g_lo,
        "gamma_hi": g_hi,
        "history": history,
        "ny": p.ny,
        "nx": p.nx,
        "a": p.a,
        "Ra_c_continuous": RA_CRIT_CONTINUOUS,
        "rel_err_vs_continuous": (0.5 * (lo + hi) - RA_CRIT_CONTINUOUS) / RA_CRIT_CONTINUOUS,
    }


def wavenumber_scan(
    ny: int,
    Ra: float,
    nx_list: Sequence[int],
    tau: float = 0.855,
    tau_T: float = 1.0,
) -> List[Dict[str, object]]:
    """``gamma`` for several horizontal box widths (i.e. wavenumbers a = 2 pi H/nx)."""
    out: List[Dict[str, object]] = []
    for nx in nx_list:
        p = RBLatticeParams(ny=ny, nx=int(nx), Ra=Ra, tau=tau, tau_T=tau_T)
        r = growth_rate(p)
        out.append({"nx": int(nx), "a": p.a, "gamma_step": r["gamma_step"], "gamma_hat": r["gamma_hat"]})
    return out


# ---------------------------------------------------------------------------
# Cross-check: measured growth rate of the non-linear library solver
# ---------------------------------------------------------------------------
def measure_growth_rate(
    p: RBLatticeParams,
    steps: int = 6000,
    amp: float = 1e-6,
    sample_every: int = 20,
    fit_from: float = 0.35,
    device: str = "cpu",
) -> Dict[str, object]:
    """Time-march the non-linear library solver and fit the exponential growth.

    Initial condition: the conductive base state plus a single-roll temperature
    perturbation ``amp sin(2 pi m x/nx) sin(pi (y+1/2)/H)`` (the most unstable
    linear mode).  The amplitude tracked is the projection of the temperature
    perturbation on that mode; a linear fit of ``ln|A(t)|`` gives
    ``gamma_sim`` in lattice units.
    """
    import torch

    device = torch.device(device)
    dtype = torch.float64
    base = conductive_base_state(p, device=device, dtype=dtype)
    L = _d2q9_d2q5()
    y = torch.arange(p.ny, device=device, dtype=dtype)
    x = torch.arange(p.nx, device=device, dtype=dtype)
    mode_x = torch.sin(2.0 * math.pi * p.mode * x / p.nx).view(1, p.nx)
    mode_y = torch.sin(math.pi * (y + 0.5) / p.H).view(p.ny, 1)
    shape = amp * mode_x * mode_y

    T0 = base.g.sum(dim=0) + shape
    zero = torch.zeros((p.ny, p.nx), device=device, dtype=dtype)
    g = L["temperature_equilibrium"](T0, zero, zero)
    f = base.f.clone()
    wall = wall_mask(p.ny, p.nx, device)

    times: List[int] = []
    amps: List[float] = []
    with torch.no_grad():
        for step in range(1, int(steps) + 1):
            f, g = rb_step(f, g, p, wall)
            if step % int(sample_every) == 0:
                T = g.sum(dim=0)
                dT = T - _conduction_profile(p, device, dtype)
                a_amp = float((dT * shape / amp).sum().item())
                amps.append(abs(a_amp) + 1e-300)
                times.append(step)
    times_arr = np.asarray(times, dtype=float)
    amps_arr = np.asarray(amps, dtype=float)
    # The exponential window is the part of the history where the mode dominates:
    # above the float64 projection floor (+1e-4 relative to the initial amplitude)
    # and after the initial transient.  Fitting the whole history would mix the
    # transient (or the noise floor) into the slope and inflate the residual.
    floor = float(np.max(amps_arr)) * 1e-4
    good = np.where(amps_arr > floor)[0]
    if good.size < 3:
        good = np.arange(len(amps_arr))
    start = good[max(0, int(fit_from * good.size))]
    stop = good[-1] + 1
    t_fit = times_arr[start:stop]
    y_fit = np.log(amps_arr[start:stop])
    if len(t_fit) < 3:
        raise ValueError("not enough samples for the growth-rate fit")
    slope, intercept = np.polyfit(t_fit, y_fit, 1)
    resid = y_fit - (slope * t_fit + intercept)
    return {
        "gamma_sim": float(slope),
        "gamma_hat_sim": float(slope) * p.H**2 / p.nu,
        "fit_window": [float(t_fit[0]), float(t_fit[-1])],
        "fit_rms": float(np.sqrt(np.mean(resid**2))),
        "amplitude_history": [[int(t), float(a)] for t, a in zip(times, amps)],
        "steps": int(steps),
        "amp0": float(amp),
        "base_residual": base.residual,
    }


def _conduction_profile(p: RBLatticeParams, device, dtype):
    import torch

    y = torch.arange(p.ny, device=device, dtype=dtype)
    T1d = p.T_hot - p.dT * (y + 0.5) / p.H
    return T1d.view(p.ny, 1).expand(p.ny, p.nx)