"""Discrete linear stability of the D2Q9 channel operator in plane Poiseuille flow.

Why this module exists
----------------------
``tensorlbm.stability.von_neumann`` linearises a *uniform* base state on a
periodic box and answers "does the scheme remain bounded".  Hydrodynamic linear
*stability* is a different question: given a shear base flow ``U(y)`` between two
walls, which wall-normal/streamwise disturbance grows, and how fast?  For plane
Poiseuille flow the answer has been known since Orszag (1971) (see
:mod:`tensorlbm.stability.orr_sommerfeld`).  This module answers the same
question *for the lattice operator itself*: it assembles the exact linearisation
of the D2Q9 BGK / MRT / regularised-BGK step about the parabolic channel base
flow, on the actual lattice with the actual half-way bounce-back walls, and
returns its spectrum.

Method
------
1. **Single-mode reduction.**  The base flow is ``x``-independent and the walls
   are straight, so the linearised step commutes with ``x``-translation: a
   perturbation with a single lattice-compatible streamwise wavenumber
   ``alpha = 2 pi m / Nx`` stays in that mode exactly (the lattice streaming
   phase is ``exp(-i alpha c_ix)``).  The state is therefore ``9 x Ny`` complex
   instead of ``9 x Ny x Nx`` real.
2. **Local Jacobian.**  Every collision model here relaxes towards a local
   equilibrium, so its Jacobian ``P(y) = d f^post / d f`` depends on the base
   state only through the recovered velocity, i.e. through ``U(y)``:

       BGK   ``P = I - (I - J_eq)/tau``
       MRT   ``P = I - M^-1 S M (I - J_eq)``
       RLBM  ``P = J_eq + (1 - 1/tau) R (I - J_eq)``

   with ``J_eq = d f^eq / d f`` (:func:`von_neumann.equilibrium_jacobian`) and
   ``R`` the second-order Hermite reconstruction matrix of
   :func:`tensorlbm.solver.collide_rlbm`.  No body force appears in ``P``: the
   force that sustains the base flow is velocity independent, so it drops out of
   the perturbation equation - which is also the physics of the Orr-Sommerfeld
   problem (the disturbance has no body force).
3. **Streaming + half-way bounce-back.**  Directions that would leave the
   channel are reflected at the same node (link bounce-back), i.e. the no-slip
   wall sits half-way between the wall-adjacent node and a fictitious solid
   node: walls at ``y = -1/2`` and ``y = Ny - 1/2``, channel width ``H = Ny``.
4. **Eigenvalues.**  ``lambda`` of the resulting ``9 Ny x 9 Ny`` complex matrix;
   per-step growth ``sigma = ln|lambda|``; dimensionless growth rate in units of
   ``U_max/h`` is ``sigma h / U_max``.  The Orr-Sommerfeld branch is identified
   by maximum overlap of the reconstructed wall-normal velocity profile with the
   OS eigenfunction of :func:`orr_sommerfeld.os_spectrum` - necessary because the
   nearly neutral acoustic channel modes also have ``|lambda| ~ 1`` and would
   otherwise masquerade as "the least damped mode".

Validation surface (see ``tests/test_p269w2_n04_poiseuille_stability.py``)
-------------------------------------------------------------------------
* the numpy collision operators reproduce ``tensorlbm.solver.collide_bgk`` /
  ``collide_mrt`` / ``collide_rlbm`` to ``1e-14``;
* the assembled linear operator matches a finite-difference Jacobian of the
  real-grid channel step to ``1e-9``;
* the eigenvalue route reproduces a direct time march of the nonlinear channel
  step (independent code path) to better than 1 %;
* the uniform-flow limit reproduces the analytical Stokes damping rate
  ``-nu k^2`` (`A1`), the sign of the growth rate matches the Orr-Sommerfeld
  benchmark at sub/super-critical Reynolds numbers (`A2`), and the grid
  convergence behaves as expected (`A3`).

Conventions
-----------
* lattice units ``dx = dt = 1``, ``cs^2 = 1/3``, ``nu = (tau - 1/2)/3``;
* ``y`` nodes ``0 .. Ny-1``, walls at ``-1/2`` and ``Ny-1/2``;
* base flow ``U(y) = u_max (1 - ((y - (Ny-1)/2) / h)^2)``, ``h = Ny/2``;
* ``Re = u_max h / nu`` (centreline velocity, half-width - Orszag's convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .orr_sommerfeld import cheb_interp, os_spectrum
from .von_neumann import (
    CX,
    CY,
    M_INV_D2Q9,
    M_D2Q9,
    W,
    Collision,
    collision_jacobian,
    equilibrium,
    equilibrium_jacobian,
)

__all__ = [
    "OPPOSITE",
    "ChannelFlow",
    "LatticeModeSpectrum",
    "channel_mode_operator",
    "collide_np",
    "grid_channel_step",
    "lattice_mode_spectrum",
    "left_eigenvector",
    "mode_project",
    "richardson_extrapolate",
    "stokes_damping_benchmark",
    "time_march_growth_rate",
]

#: ``OPPOSITE[i]`` - index of the D2Q9 direction opposite to ``i``.
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

KINDS = ("bgk", "mrt", "rlbm")


# ---------------------------------------------------------------------------
# collision operators (numpy, float64/complex128) mirroring tensorlbm.solver
# ---------------------------------------------------------------------------


def _hermite_reconstruction() -> np.ndarray:
    """Second-order Hermite reconstruction matrix ``R`` of ``collide_rlbm``.

    ``f^neq,reg_i = sum_j R_ij (f_j - f_j^eq)`` with
    ``R_ij = (9/2) w_i [ (c_ix^2 - 1/3) c_jx^2 + (c_iy^2 - 1/3) c_jy^2
    + 2 c_ix c_iy c_jx c_jy ]``, exactly the reconstruction used by
    :func:`tensorlbm.solver.collide_rlbm`.
    """
    cx = CX.reshape(9, 1)
    cy = CY.reshape(9, 1)
    jx = CX.reshape(1, 9)
    jy = CY.reshape(1, 9)
    hxx = cx * cx - 1.0 / 3.0
    hyy = cy * cy - 1.0 / 3.0
    hxy = cx * cy
    return (9.0 / 2.0) * W.reshape(9, 1) * (
        hxx * (jx * jx) + hyy * (jy * jy) + 2.0 * hxy * (jx * jy)
    )


_R_MATRIX = _hermite_reconstruction()


def _mrt_matrices() -> tuple[np.ndarray, np.ndarray]:
    return M_D2Q9, M_INV_D2Q9


def _collision_matrix(
    f: np.ndarray,
    kind: str,
    tau: float,
    rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Post-collision state ``f^post`` (arrays of shape ``(9, ...)``)."""
    kind = kind.lower()
    if kind == "srt":
        kind = "bgk"
    rho = f.sum(axis=0)
    ux = (CX.reshape(9, *([1] * (f.ndim - 1))) * f).sum(axis=0) / rho
    uy = (CY.reshape(9, *([1] * (f.ndim - 1))) * f).sum(axis=0) / rho
    feq = equilibrium(rho, ux, uy)
    if kind == "bgk":
        return f - (f - feq) / tau
    if kind == "rlbm":
        return feq + (1.0 - 1.0 / tau) * (_R_MATRIX @ (f - feq).reshape(9, -1)).reshape(f.shape)
    if kind == "mrt":
        m, m_inv = _mrt_matrices()
        if rates is None:
            rates = Collision(kind="mrt", tau=tau).rates()
        s = np.asarray(rates, dtype=float).reshape(9, 1)
        shape = f.shape
        fm = f.reshape(9, -1)
        mom = m @ fm
        mom_eq = m @ feq.reshape(9, -1)
        mom_star = mom - s * (mom - mom_eq)
        return (m_inv @ mom_star).reshape(shape)
    raise ValueError(f"unsupported collision kind: {kind!r}")


def collide_np(
    f: np.ndarray,
    kind: str,
    tau: float,
    rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Post-collision distribution (public wrapper of :func:`_collision_matrix`)."""
    return _collision_matrix(np.asarray(f, dtype=f.dtype), kind, tau, rates)


# ---------------------------------------------------------------------------
# channel geometry / base flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelFlow:
    """Plane channel in lattice units plus its parabolic base flow."""

    ny: int
    Re: float
    u_max: float = 0.05
    rho0: float = 1.0

    def __post_init__(self) -> None:
        if self.ny < 4:
            raise ValueError("ny must be >= 4")
        if self.Re <= 0 or self.u_max <= 0:
            raise ValueError("Re and u_max must be positive")

    @property
    def h(self) -> float:
        """Half channel width in lattice units (walls at ``-1/2``/``ny-1/2``)."""
        return 0.5 * self.ny

    @property
    def nu(self) -> float:
        return self.u_max * self.h / self.Re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def y_nodes(self) -> np.ndarray:
        return np.arange(self.ny, dtype=float)

    @property
    def y_hat(self) -> np.ndarray:
        """Wall-normal coordinate scaled to the OS convention (``[-1, 1]``)."""
        return (self.y_nodes - 0.5 * (self.ny - 1)) / self.h

    @property
    def base_velocity(self) -> np.ndarray:
        """``U(y)`` sampled at the lattice nodes (``U(+-1) = 0`` at the walls)."""
        return self.u_max * (1.0 - self.y_hat**2)

    def summary(self) -> dict:
        return {
            "ny": self.ny,
            "Re": float(self.Re),
            "u_max": float(self.u_max),
            "nu": float(self.nu),
            "tau": float(self.tau),
            "h": float(self.h),
            "ma": float(self.u_max / np.sqrt(1.0 / 3.0)),
        }


# ---------------------------------------------------------------------------
# local Jacobians
# ---------------------------------------------------------------------------


def local_jacobian(
    kind: str,
    u: float,
    tau: float,
    rates: Sequence[float] | None = None,
    rho0: float = 1.0,
) -> np.ndarray:
    """``d f^post / d f`` at a base state with streamwise velocity ``u``."""
    kind = kind.lower()
    if kind == "srt":
        kind = "bgk"
    j_eq = equilibrium_jacobian(rho0, (float(u), 0.0))
    eye = np.eye(9)
    if kind == "bgk":
        coll = Collision(kind="bgk", tau=float(tau))
        return eye - coll.operator_matrix() @ (eye - j_eq)
    if kind == "mrt":
        coll = Collision(kind="mrt", tau=float(tau))
        if rates is not None:
            coll = Collision(
                kind="mrt",
                tau=float(tau),
                s_e=float(rates[1]),
                s_eps=float(rates[2]),
                s_q=float(rates[4]),
            )
        return eye - coll.operator_matrix() @ (eye - j_eq)
    if kind == "rlbm":
        return j_eq + (1.0 - 1.0 / tau) * (_R_MATRIX @ (eye - j_eq))
    raise ValueError(f"unsupported collision kind: {kind!r}")


# ---------------------------------------------------------------------------
# single-mode linear operator
# ---------------------------------------------------------------------------


def channel_mode_operator(
    flow: ChannelFlow,
    alpha: float,
    kind: str = "bgk",
    rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Linearised single-mode step matrix ``L`` of shape ``(9*ny, 9*ny)``.

    Args:
        flow: channel geometry + base flow.
        alpha: *lattice* streamwise wavenumber (``2 pi m / Nx``).
        kind: ``"bgk"``, ``"mrt"`` or ``"rlbm"``.
        rates: optional MRT rate vector (ignored by the other models).
    """
    ny = flow.ny
    tau = flow.tau
    U = flow.base_velocity
    cx = CX.reshape(9, 1)
    cy = CY.reshape(9, 1)

    jac = np.empty((ny, 9, 9), dtype=float)
    for y in range(ny):
        jac[y] = local_jacobian(kind, U[y], tau, rates, flow.rho0)

    L = np.zeros((9 * ny, 9 * ny), dtype=complex)
    phase = np.exp(-1j * alpha * np.asarray(CX, dtype=float))
    for y in range(ny):
        for i in range(9):
            row = i * ny + y
            ys = y - int(CY[i])
            if 0 <= ys <= ny - 1:
                L[row, :, ][np.arange(9) * ny + ys] = phase[i] * jac[ys][i, :]
            else:
                j = int(OPPOSITE[i])
                L[row, np.arange(9) * ny + y] = jac[y][j, :]
    return L


# ---------------------------------------------------------------------------
# spectrum + mode identification
# ---------------------------------------------------------------------------


@dataclass
class LatticeModeSpectrum:
    """Spectrum of the linearised lattice channel operator at one operating point."""

    flow: ChannelFlow
    alpha: float  #: lattice wavenumber
    kind: str
    lam: np.ndarray  #: eigenvalues
    vecs: np.ndarray  #: right eigenvectors, shape ``(n_eig, 9*ny)``
    sigma: np.ndarray  #: per-step growth rate ``ln|lambda|``
    selected: int = -1  #: index of the Orr-Sommerfeld-like mode
    overlap: float = 0.0  #: |<v_lattice, v_OS>| of the selected mode
    os_reference: Any = None

    @property
    def alpha_nd(self) -> float:
        """Streamwise wavenumber in units of ``1/h``."""
        return self.alpha * self.flow.h

    @property
    def index_least_damped(self) -> int:
        return int(np.argmax(np.abs(self.lam)))

    @property
    def sigma_leading(self) -> float:
        return float(self.sigma[self.index_least_damped])

    @property
    def eigenvector(self) -> np.ndarray:
        return self.vecs[self.selected]

    def velocity_profile(self, vec: np.ndarray | None = None) -> np.ndarray:
        """Wall-normal velocity perturbation ``v(y)`` of an eigenvector."""
        g = np.asarray(self.eigenvector if vec is None else vec).reshape(9, self.flow.ny)
        return (np.asarray(CY, dtype=float).reshape(9, 1) * g).sum(axis=0)

    def streamwise_profile(self, vec: np.ndarray | None = None) -> np.ndarray:
        g = np.asarray(self.eigenvector if vec is None else vec).reshape(9, self.flow.ny)
        return (np.asarray(CX, dtype=float).reshape(9, 1) * g).sum(axis=0)

    def sigma_nondim(self, index=None):
        """Growth rate in units of ``U_max / h`` (``omega_i`` of OS theory)."""
        idx = self.selected if index is None else index
        scale = self.flow.h / self.flow.u_max
        if np.ndim(idx) == 0:
            return float(self.sigma[idx] * scale)
        return self.sigma[idx] * scale

    def summary(self) -> dict[str, Any]:
        idx_os = self.selected
        idx_ld = self.index_least_damped
        return {
            "kind": self.kind,
            "ny": self.flow.ny,
            "Re": float(self.flow.Re),
            "alpha_lattice": float(self.alpha),
            "alpha_nd": float(self.alpha_nd),
            "sigma_selected_nondim": float(self.sigma_nondim(idx_os)),
            "overlap_os": float(self.overlap),
            "sigma_least_damped_nondim": float(self.sigma_nondim(idx_ld)),
            "which_is_least_damped": "os_mode" if idx_ld == idx_os else "acoustic",
            "os_reference": None if self.os_reference is None else self.os_reference.summary(),
        }


def lattice_mode_spectrum(
    flow: ChannelFlow,
    alpha: float,
    kind: str = "bgk",
    rates: Sequence[float] | None = None,
    *,
    n_cheb: int = 80,
    dense_max: int = 1500,
    n_eigs: int = 16,
    operator: np.ndarray | None = None,
    shift_invert: bool = True,
) -> LatticeModeSpectrum:
    """Assemble, diagonalise and identify the Orr-Sommerfeld branch.

    ``alpha`` is the *lattice* wavenumber; the OS reference is evaluated at the
    matching non-dimensional ``alpha_nd = alpha * h`` and ``Re``.
    """
    L = channel_mode_operator(flow, alpha, kind, rates) if operator is None else operator
    n = L.shape[0]
    alpha_nd = alpha * flow.h
    os_spec = os_spectrum(alpha_nd, flow.Re, n_cheb)

    if n <= dense_max:
        lam, vecs = np.linalg.eig(L)
        vecs = vecs.T
    else:
        # Solver for large Ny: the spectrum of the channel operator is crowded at
        # |lambda| ~ 1 (slow acoustic/viscous channel modes), so a shift-free
        # Arnoldi iteration does not separate the Orr-Sommerfeld branch.  The
        # shift is the per-step lattice eigenvalue predicted by the OS benchmark,
        # exp(sigma_hat u_max/h - i alpha c_r u_max/h); shift-and-invert then
        # converges in a handful of iterations.
        from scipy.sparse import csc_matrix  # noqa: PLC0415
        from scipy.sparse.linalg import LinearOperator, eigs, splu  # noqa: PLC0415

        scale = flow.u_max / flow.h
        c_os = os_spec.c_leading
        sigma_hat = alpha_nd * c_os.imag
        omega_hat = alpha_nd * c_os.real
        shift = complex(np.exp(sigma_hat * flow.h / flow.u_max),
                        0.0) * np.exp(-1j * omega_hat * flow.h / flow.u_max)
        if shift_invert:
            lu = splu(csc_matrix(L - shift * np.eye(n)))
            opinv = LinearOperator((n, n), matvec=lu.solve,
                                   dtype=np.complex128)
            lam, vecs = eigs(L, k=min(n_eigs, 8), sigma=shift, OPinv=opinv,
                             which="LM", ncv=min(n - 1, 30), tol=1e-11,
                             maxiter=5000)
        else:  # pragma: no cover
            lam, vecs = eigs(L, k=min(n_eigs, n - 2), which="LM",
                             ncv=min(n - 1, 60), tol=1e-10, maxiter=100000)
        vecs = vecs.T
    sigma = np.log(np.abs(lam))
    v_os = cheb_interp(os_spec.v_leading, os_spec.y, flow.y_hat)
    v_os = v_os / np.linalg.norm(v_os)

    best, best_ov = -1, -1.0
    for idx in range(vecs.shape[0]):
        g = vecs[idx].reshape(9, flow.ny)
        v = (np.asarray(CY, dtype=float).reshape(9, 1) * g).sum(axis=0)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            continue
        ov = abs(np.vdot(v / norm, v_os))
        if ov > best_ov:
            best, best_ov = idx, ov

    return LatticeModeSpectrum(
        flow=flow,
        alpha=float(alpha),
        kind=kind,
        lam=lam,
        vecs=vecs,
        sigma=sigma,
        selected=best,
        overlap=float(best_ov),
        os_reference=os_spec,
    )


# ---------------------------------------------------------------------------
# real-grid channel step (used for the FD cross-audit and the time march)
# ---------------------------------------------------------------------------


def grid_channel_step(
    f: np.ndarray,
    kind: str,
    tau: float,
    rates: Sequence[float] | None = None,
    force_x: float = 0.0,
) -> np.ndarray:
    """One full channel step on a real grid ``(9, ny, nx)``: collide, force, stream+BB."""
    post = _collision_matrix(f, kind, tau, rates)
    if force_x:
        post = post + (3.0 * W.reshape(9, 1, 1) * CX.reshape(9, 1, 1) * force_x)
    out = np.empty_like(post)
    for i in range(9):
        out[i] = np.roll(post[i], (int(CY[i]), int(CX[i])), axis=(0, 1))
    for i in range(9):
        j = int(OPPOSITE[i])
        if CY[i] > 0:  # arriving from the bottom wall
            out[i, 0, :] = post[j, 0, :]
        elif CY[i] < 0:  # arriving from the top wall
            out[i, -1, :] = post[j, -1, :]
    return out


def mode_project(f: np.ndarray, mode: int = 1) -> np.ndarray:
    """Extract the complex streamwise Fourier component ``mode`` of a real field."""
    nx = f.shape[-1]
    ft = np.fft.fft(f, axis=-1) / nx
    return ft[..., mode]


def time_march_growth_rate(
    flow: ChannelFlow,
    alpha: float,
    kind: str = "bgk",
    *,
    k_box: int = 3,
    mode: int = 1,
    n_steps: int = 40000,
    amplitude: float = 1e-3,
    sample_every: int = 50,
    warmup: int = 200,
    rates: Sequence[float] | None = None,
    seed_profile: np.ndarray | None = None,
    projector: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure the growth rate of one channel mode by time marching the real step.

    Independent path: integrates :func:`grid_channel_step` (a real, nonlinear
    lattice Boltzmann simulation) and fits the exponential growth of the
    streamwise Fourier component ``mode`` of the perturbation ``f - <f>_x``.

    Args:
        flow: channel operating point (``Re`` also sets the driving force).
        alpha: *lattice* wavenumber ``2 pi mode / (k_box * ny)`` (must match).
        k_box: ``Nx = k_box * ny`` lattice nodes along ``x``.
        mode: streamwise Fourier index (``>= 1``).
        n_steps, sample_every: integration length and sampling stride.
        amplitude: initial perturbation amplitude (kept small: linear regime).
        warmup: steps discarded before fitting (acoustic + start-up transients).
        seed_profile: optional complex wall-normal profile for ``u_y``.  Defaults
            to the OS eigenfunction interpolated on the lattice nodes, which
            maximises the projection on the mode of interest.
        projector: optional complex weights of shape ``(9, ny)`` defining the
            mode-selective readout ``A(t) = <w, g(t)>`` (left eigenvector of the
            linearised operator for the Orr-Sommerfeld branch).  Without it the
            readout is the plain ``L2`` norm of the wall-normal velocity mode,
            which is polluted by the slower decaying acoustic channel modes.
    """
    nx = int(k_box * flow.ny)
    ny = flow.ny
    alpha_target = 2.0 * np.pi * mode / nx
    if abs(alpha_target - alpha) > 1e-10:
        raise ValueError(
            f"alpha={alpha} is not representable on Nx={nx}, mode={mode}: "
            f"expected {alpha_target}"
        )
    tau = flow.tau
    # body force sustaining the parabolic base flow: u_max = F h^2 / (2 nu)
    force_x = 2.0 * flow.nu * flow.u_max / flow.h**2

    rho = np.full((ny, nx), flow.rho0)
    ux = np.tile(flow.base_velocity.reshape(ny, 1), (1, nx))
    uy = np.zeros((ny, nx))
    if seed_profile is None:
        spec = os_spectrum(alpha * flow.h, flow.Re, 80)
        prof = cheb_interp(spec.v_leading, spec.y, flow.y_hat)
    else:
        prof = np.asarray(seed_profile, dtype=complex)
    prof = prof / np.max(np.abs(prof))
    xs = np.arange(nx)
    phase = np.exp(1j * alpha_target * xs)
    uy = uy + amplitude * np.real(prof.reshape(ny, 1) * phase.reshape(1, nx))
    # divergence-free companion u_x = (i/alpha) d v/dy (spectral in y via diff)
    dvdy = np.gradient(prof, flow.y_hat)
    ux_prof = (1j / alpha_target) * dvdy
    ux = ux + amplitude * np.real(ux_prof.reshape(ny, 1) * phase.reshape(1, nx))
    f = equilibrium(rho, ux, uy)

    samples: list[tuple[int, float, complex, float]] = []
    diverged_at = None
    for step in range(int(n_steps)):
        f = grid_channel_step(f, kind, tau, rates, force_x=force_x)
        if not np.isfinite(f).all() or np.abs(f).max() > 10.0:
            diverged_at = step
            break
        if step % sample_every == 0:
            fx = f.mean(axis=2, keepdims=True)
            pert = f - fx
            g = mode_project(pert, mode)
            v = (np.asarray(CY, dtype=float).reshape(9, 1) * g).sum(axis=0)
            if projector is None:
                a = float(np.linalg.norm(v))
            else:
                a = float(abs(np.vdot(np.asarray(projector).ravel(), g.ravel())))
                if a == 0.0:
                    a = 1e-300
            # also the resolved base-flow centreline velocity (drift diagnostic)
            rho_m = fx.sum(axis=0)
            ux_m = (np.asarray(CX, dtype=float).reshape(9, 1, 1) * fx).sum(axis=0) / rho_m
            samples.append((step, float(a), complex(g.sum()), float(ux_m.max())))

    arr = np.array([(s[0], s[1]) for s in samples], dtype=float)
    keep = (arr[:, 0] >= warmup) & (arr[:, 1] > 0.0)
    t = arr[keep, 0]
    a = arr[keep, 1]
    # fit only while the mode is inside the linear window
    ref = a[0]
    window = (a > 1e-6 * ref) & (a < 1e3 * ref)
    t, a = t[window], a[window]
    if t.size < 5:
        raise RuntimeError("time march produced too few samples in the linear window")
    coef = np.polyfit(t, np.log(a), 1)
    sigma_lat = float(coef[0])
    resid = float(np.std(np.log(a) - np.polyval(coef, t)))
    return {
        "sigma_lattice": sigma_lat,
        "sigma_nondim": sigma_lat * flow.h / flow.u_max,
        "log_amplitude_residual": resid,
        "n_samples": int(t.size),
        "n_steps": int(n_steps),
        "amplitude_start": float(arr[0, 1]),
        "amplitude_end": float(arr[-1, 1]),
        "base_flow_drift": float(samples[0][3] - samples[-1][3]),
        "diverged_at": diverged_at,
        "readout": "left_eigen_projection" if projector is not None else "l2_norm",
        "nx": nx,
        "alpha_lattice": float(alpha_target),
    }


def left_eigenvector(L: np.ndarray, lam: complex) -> np.ndarray:
    """Left eigenvector ``w`` of ``L`` for the eigenvalue ``lam`` (``w^H L = lam w^H``).

    Obtained from the SVD of ``(L - lam I)^H``: the right singular vector of the
    smallest singular value is the left eigenvector of ``L``.  Used as the
    mode-selective readout of the time march (a linear functional of the state
    that removes the contamination of the slower decaying acoustic modes).
    """
    n = L.shape[0]
    mat = (L - complex(lam) * np.eye(n)).conj().T
    _, _, vh = np.linalg.svd(mat)
    return vh[-1].conj()


# ---------------------------------------------------------------------------
# A1: uniform-flow Stokes damping benchmark
# ---------------------------------------------------------------------------


def stokes_damping_benchmark(
    n: int = 32,
    mode: int = 1,
    tau: float = 0.6,
    n_steps: int = 2000,
    amplitude: float = 1e-6,
    u0: float = 0.0,
) -> dict[str, Any]:
    """A1 - decay of a transverse shear wave in a uniform flow.

    Three independent numbers are compared:

    * ``sigma_continuum = -nu k^2`` with ``nu = (tau - 1/2)/3`` (analytical);
    * ``sigma_von_neumann`` - the shear eigenvalue of the lattice amplification
      matrix (:func:`von_neumann.spectrum`, linear analysis);
    * ``sigma_measured`` - fitted decay of a full nonlinear periodic-box run
      (:func:`von_neumann.run_modal_box`).
    """
    import math

    from .von_neumann import Collision, run_modal_box, spectrum

    k = 2.0 * math.pi * mode / n
    coll = Collision(kind="bgk", tau=tau)
    nu = coll.viscosity
    sigma_cont = -nu * k * k
    spec = spectrum(k, 0.0, coll, (u0, 0.0))
    idx = spec.index_of("shear")
    sigma_vn = float(np.log(abs(spec.mu[idx])))
    run = run_modal_box(
        k, 0.0, coll, (u0, 0.0), n=n, n_steps=n_steps, seed="shear",
        target="shear", amplitude=amplitude, early_stop=False,
    )
    sigma_meas = float(run.measured_growth)
    return {
        "n": int(n),
        "mode": int(mode),
        "tau": float(tau),
        "nu": float(nu),
        "k": float(k),
        "sigma_continuum": float(sigma_cont),
        "sigma_von_neumann": sigma_vn,
        "sigma_measured": sigma_meas,
        "rel_error_von_neumann": float(abs(sigma_vn - sigma_cont) / abs(sigma_cont)),
        "rel_error_measured": float(abs(sigma_meas - sigma_cont) / abs(sigma_cont)),
    }


# ---------------------------------------------------------------------------
# A3: Richardson extrapolation
# ---------------------------------------------------------------------------


def richardson_extrapolate(
    values: Sequence[float], levels: Sequence[float] | None = None
) -> dict[str, Any]:
    """Richardson extrapolation of a three-level grid sequence.

    Uses the observed order ``p`` fitted from three successive levels
    (``h`` halving assumed), the extrapolated limit and the residual of the
    finest level relative to that limit.
    """
    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise ValueError("richardson_extrapolate expects exactly three levels")
    lev = [float(x) for x in (levels if levels is not None else (64, 128, 256))]
    d1 = vals[1] - vals[0]
    d2 = vals[2] - vals[1]
    if d1 == 0.0 or d2 == 0.0 or d1 * d2 <= 0.0:
        p = 2.0
        note = "order not identifiable from signs; assumed p=2"
    else:
        ratio = d1 / d2
        if ratio <= 1.0:
            p = 2.0
            note = "convergence ratio <= 1 (non-monotone); assumed p=2"
        else:
            p = float(np.log(ratio) / np.log(lev[2] / lev[1]))
            note = "order fitted from three levels"
    limit = vals[2] + d2 / (2.0**p - 1.0)
    residual = abs(vals[2] - limit) / abs(limit) if limit != 0.0 else float("inf")
    return {
        "levels": lev,
        "values": vals,
        "observed_order": p,
        "order_note": note,
        "limit": limit,
        "residual_finest_rel": float(residual),
    }