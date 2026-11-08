"""Von Neumann linear stability analysis for D2Q9 LBM (BGK and MRT).

Theory
------
Linearise the lattice Boltzmann equation around a *uniform* base state
``f_i^eq(rho0, u0)``::

    f_i(x + c_i, t + 1) = f_i(x, t) + [C (f - f^eq)]_i

with the collision operator ``C`` (``I/tau`` for BGK, ``M^-1 S M`` for MRT).
Inserting the Fourier ansatz ``delta f_i = v_i exp(i (k . x - omega t))``
gives the eigenvalue problem

    A(theta) v = exp(-i omega) v,      A(theta) = diag(exp(-i theta . c_i)) P

where ``theta = k`` (lattice units: dx = dt = 1) and

    P = d f^post / d f = I - C (I - J_eq),   J_eq = d f^eq / d f

is the linearised collision.  The eigenvalues ``mu_j`` of ``A`` are the
per-timestep amplification factors of the ``j``-th Fourier mode:

    growth rate   sigma_j = ln |mu_j|          (per timestep; > 0 == unstable)
    phase speed   c_j     = -arg(mu_j) / |k|   (convention exp(i(k.x - omega t)))

and ``lambda_max(theta) = max_j |mu_j|``.  The scheme is linearly stable in the
von Neumann sense iff ``max_theta lambda_max(theta) <= 1``.

Mode identification is *not* done by eyeballing moment content: the
non-hydrodynamic ("kinetic") modes of BGK share the transverse-momentum
structure of the shear mode, so instead the physical branches are identified by
maximum overlap with the k -> 0 eigendirections:

* shear  - the f-space direction of a purely transverse velocity perturbation
  (exactly an eigenvector of ``P`` with eigenvalue 1 at ``k = 0``),
* sound  - the f-space direction of a pure density perturbation; the two
  acoustic branches are the two largest overlaps, ordered by phase speed.

Measured side
-------------
:func:`run_modal_box` integrates the *full nonlinear* periodic-box LBM in
float64 with a single Fourier mode as initial perturbation and fits the
exponential growth of that mode (physical observable: transverse momentum for
the shear branch, density for the acoustic branches, or the left-eigenvector
projection for the branch-selective ``seed="eigen"`` protocol).  The linear
and nonlinear paths share only the lattice definition (``c_i``, ``w_i``) and
the collision-model definition - the linearisation and the time integration are
independent code paths.

Numerical conventions
---------------------
* Lattice units: ``dx = dt = 1``, ``cs^2 = 1/3``.
* BGK: ``C = I / tau``, kinematic viscosity ``nu = (tau - 1/2) / 3``.
* MRT (Lallemand & Luo 2000 moment ordering):
  ``(rho, e, eps, jx, qx, jy, qy, pxx, pxy)`` with relaxation rates
  ``(0, s_e, s_eps, 0, s_q, 0, s_q, 1/tau, 1/tau)``; identical to
  :func:`tensorlbm.solver.collide_mrt` (locked by a test).
* ``Ma`` is defined with ``cs = 1/sqrt(3)``: ``u0 = Ma / sqrt(3)`` along +x
  unless another base velocity is given.

References
----------
* Sterling & Chen, "Stability analysis of lattice Boltzmann methods",
  J. Comput. Phys. 123 (1996) 196-206.
* Lallemand & Luo, "Theory of the lattice Boltzmann method: dispersion,
  dissipation, isotropy, Galilean invariance, and stability",
  Phys. Rev. E 61 (2000) 6546.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice (same velocity ordering as tensorlbm.constants.D2Q9)
# ---------------------------------------------------------------------------

CX = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
CY = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
CVEL = np.stack([CX, CY], axis=1)  # (9, 2)

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

#: Tolerance of the stability verdict.  At ``theta = 0`` the three conserved
#: modes have ``|mu| = 1`` up to round-off, so ``lambda_max`` never drops below
#: ``1 + O(1e-15)``; any physical instability of interest is orders of
#: magnitude larger than this tolerance.
STABILITY_TOL = 1e-9

MOMENT_ORDER = ("rho", "e", "eps", "jx", "qx", "jy", "qy", "pxx", "pxy")

#: Lallemand & Luo D2Q9 transformation matrix ``M``.
M_D2Q9 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=float,
)
M_INV_D2Q9 = np.linalg.inv(M_D2Q9)

_I9 = np.eye(9)


# ---------------------------------------------------------------------------
# Collision models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Collision:
    """Collision-model descriptor.

    Parameters
    ----------
    kind:
        ``"bgk"`` (single relaxation time) or ``"mrt"`` (multi relaxation time).
    tau:
        Relaxation time of the shear-stress moments; ``nu = (tau - 1/2)/3``.
    s_e, s_eps, s_q:
        MRT relaxation rates of the ``e``, ``eps`` and ``q`` moment families
        (ignored by BGK).  Defaults equal :func:`tensorlbm.solver.collide_mrt`.
    """

    kind: str = "bgk"
    tau: float = 0.6
    s_e: float = 1.64
    s_eps: float = 1.54
    s_q: float = 1.7

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind == "srt":
            kind = "bgk"
        if kind not in ("bgk", "mrt"):
            raise ValueError(f"unsupported collision kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(f"tau must be finite and > 0, got {self.tau}")

    @property
    def viscosity(self) -> float:
        """Chapman-Enskog shear viscosity ``nu = (tau - 1/2)/3``."""
        return (self.tau - 0.5) / 3.0

    def rates(self) -> np.ndarray:
        """MRT relaxation-rate vector in :data:`MOMENT_ORDER`."""
        if self.kind == "bgk":
            return np.full(9, 1.0 / self.tau)
        return np.array(
            [
                0.0,
                self.s_e,
                self.s_eps,
                0.0,
                self.s_q,
                0.0,
                self.s_q,
                1.0 / self.tau,
                1.0 / self.tau,
            ]
        )

    def operator_matrix(self) -> np.ndarray:
        """Collision matrix ``C`` such that ``f^post = f - C (f - f^eq)``."""
        if self.kind == "bgk":
            return _I9 / self.tau
        return M_INV_D2Q9 @ np.diag(self.rates()) @ M_D2Q9

    def label(self) -> str:
        if self.kind == "bgk":
            return f"BGK(tau={self.tau:.4g})"
        return f"MRT(tau={self.tau:.4g},s_e={self.s_e:g},s_eps={self.s_eps:g},s_q={self.s_q:g})"


# ---------------------------------------------------------------------------
# Equilibrium and its Jacobian
# ---------------------------------------------------------------------------


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium on a grid, shape ``(9,) + rho.shape``.

    Equivalent to :func:`tensorlbm.lattice.equilibrium` up to round-off
    (locked by ``tests/test_lbm_von_neumann_stability.py``).
    """
    rho = np.asarray(rho, dtype=float)
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    shape = (9,) + (1,) * rho.ndim
    cu = 3.0 * (CX.reshape(shape) * ux[None] + CY.reshape(shape) * uy[None])
    u2 = (ux * ux + uy * uy)[None]
    return W.reshape(shape) * rho[None] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)


def equilibrium_jacobian(rho0: float = 1.0, u0: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """``J_eq[i, j] = d f_i^eq / d f_j`` at the uniform state ``(rho0, u0)``.

    ``f_i^eq = w_i rho [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u^2]`` and
    ``d u_a / d f_j = (c_ja - u_a) / rho``.
    """
    ux, uy = float(u0[0]), float(u0[1])
    cxu = CX * ux + CY * uy
    u2 = ux * ux + uy * uy
    a_i = W * (1.0 + 3.0 * cxu + 4.5 * cxu * cxu - 1.5 * u2)
    b = W[:, None] * (3.0 * CVEL + 9.0 * cxu[:, None] * CVEL - 3.0 * np.array([[ux, uy]]))
    return a_i[:, None] + b @ (CVEL - np.array([[ux, uy]])).T


def collision_jacobian(
    coll: Collision, rho0: float = 1.0, u0: Sequence[float] = (0.0, 0.0)
) -> np.ndarray:
    """Linearised post-collision map ``P = d f^post / d f`` at ``(rho0, u0)``."""
    j_eq = equilibrium_jacobian(rho0, u0)
    return _I9 - coll.operator_matrix() @ (_I9 - j_eq)


# ---------------------------------------------------------------------------
# von Neumann amplification matrix and spectrum
# ---------------------------------------------------------------------------


def amplification_matrix(
    kx: float,
    ky: float,
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    rho0: float = 1.0,
) -> np.ndarray:
    """Complex ``9 x 9`` amplification matrix ``A = diag(exp(-i k.c)) P``."""
    p = collision_jacobian(coll, rho0, u0)
    phase = np.exp(-1j * (kx * CX + ky * CY))
    return phase[:, None] * p


def _unit(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    kn = math.hypot(kx, ky)
    if kn == 0.0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    khat = np.array([kx, ky]) / kn
    return khat, np.array([-khat[1], khat[0]])


def seed_direction(
    kind: str,
    kx: float,
    ky: float,
    rho0: float = 1.0,
    u0: Sequence[float] = (0.0, 0.0),
    eps: float = 1e-7,
) -> np.ndarray:
    """f-space direction of a physical perturbation (used to identify branches).

    ``kind="shear"``: purely transverse velocity perturbation ``u0 + eps*p``;
    ``kind="sound"``: pure density perturbation ``rho0 + eps`` at fixed ``u0``.
    """
    _, phat = _unit(kx, ky)
    ux, uy = float(u0[0]), float(u0[1])
    base = equilibrium(np.array(rho0), np.array(ux), np.array(uy))
    if kind == "shear":
        pert = equilibrium(
            np.array(rho0), np.array(ux + eps * phat[0]), np.array(uy + eps * phat[1])
        )
    elif kind == "sound":
        pert = equilibrium(np.array(rho0 + eps), np.array(ux), np.array(uy))
    else:
        raise ValueError(f"unknown seed direction kind {kind!r}")
    return (pert - base) / eps


@dataclass
class Spectrum:
    """Eigen-decomposition of the von Neumann amplification matrix."""

    kx: float
    ky: float
    coll: Collision
    u0: tuple[float, float]
    mu: np.ndarray  # (9,) eigenvalues; |mu| = per-step amplification factor
    eigvecs: np.ndarray  # (9, 9) right eigenvectors (columns)
    left: np.ndarray  # (9, 9) left eigenvectors (rows): left @ A = diag(mu) @ left
    labels: list[str]
    shares: np.ndarray  # (9, 3) mass / parallel / perpendicular momentum shares
    shear_index: int
    sound_indices: tuple[int, int]

    # -- derived quantities ------------------------------------------------
    @property
    def theta(self) -> float:
        return math.hypot(self.kx, self.ky)

    @property
    def growth(self) -> np.ndarray:
        """Per-timestep growth rate ``ln|mu|``."""
        return np.log(np.abs(self.mu))

    @property
    def phase_speed(self) -> np.ndarray:
        """Phase speed ``-arg(mu)/|k|`` along the wave vector."""
        k = self.theta
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.angle(self.mu) / k

    @property
    def lam_max(self) -> float:
        return float(np.max(np.abs(self.mu)))

    @property
    def worst_index(self) -> int:
        return int(np.argmax(np.abs(self.mu)))

    def index_of(self, label: str) -> int | None:
        """Index for ``"shear"``, ``"sound+"``, ``"sound-"``, ``"max"``."""
        if label == "max":
            return self.worst_index
        if label == "shear":
            return self.shear_index
        if label == "sound+":
            for j in self.sound_indices:
                if self.phase_speed[j] > 0:
                    return j
            return self.sound_indices[0]
        if label == "sound-":
            for j in self.sound_indices:
                if self.phase_speed[j] < 0:
                    return j
            return self.sound_indices[-1]
        raise ValueError(f"unknown mode label {label!r}")


def _labels(
    left: np.ndarray, kx: float, ky: float, mu: np.ndarray, rho0: float, u0
) -> tuple[list[str], int, tuple[int, int]]:
    """Name branches by maximum overlap with the physical seed directions."""
    g_shear = seed_direction("shear", kx, ky, rho0, u0)
    g_sound = seed_direction("sound", kx, ky, rho0, u0)
    ov_shear = np.abs(left @ g_shear)
    ov_sound = np.abs(left @ g_sound)
    shear_index = int(np.argmax(ov_shear))
    order = np.argsort(ov_sound)[::-1]
    sound_indices = (int(order[0]), int(order[1]))
    labels = ["kinetic"] * 9
    labels[shear_index] = "shear"
    for j in sound_indices:
        labels[j] = "sound"
    return labels, shear_index, sound_indices


def _shares(eigvecs: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Diagnostic moment shares (mass / parallel / perpendicular momentum)."""
    khat, phat = _unit(kx, ky)
    out = np.zeros((9, 3))
    for j in range(eigvecs.shape[1]):
        v = eigvecs[:, j]
        nv = np.linalg.norm(v)
        if nv == 0.0:
            continue
        mass = float(np.abs(np.sum(v))) / nv
        mom = (CVEL.T @ v) / nv  # complex (2,)
        par = float(np.abs(np.vdot(khat.astype(complex), mom)))
        perp = float(np.abs(np.vdot(phat.astype(complex), mom)))
        tot = mass + par + perp
        if tot > 0.0:
            out[j] = (mass / tot, par / tot, perp / tot)
    return out


def spectrum(
    kx: float,
    ky: float,
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    rho0: float = 1.0,
) -> Spectrum:
    """Eigen-decompose the amplification matrix at wave vector ``(kx, ky)``."""
    a = amplification_matrix(kx, ky, coll, u0, rho0)
    mu, vecs = np.linalg.eig(a)
    left = np.linalg.inv(vecs)  # rows are left eigenvectors: left @ A = diag(mu) @ left
    labels, shear_index, sound_indices = _labels(left, kx, ky, mu, rho0, u0)
    return Spectrum(
        kx=float(kx),
        ky=float(ky),
        coll=coll,
        u0=(float(u0[0]), float(u0[1])),
        mu=mu,
        eigvecs=vecs,
        left=left,
        labels=labels,
        shares=_shares(vecs, kx, ky),
        shear_index=shear_index,
        sound_indices=sound_indices,
    )


def lambda_max_grid(
    coll: Collision,
    kxs: np.ndarray,
    kys: np.ndarray,
    u0: Sequence[float] = (0.0, 0.0),
    rho0: float = 1.0,
) -> np.ndarray:
    """Batched ``lambda_max(kx, ky)`` over a tensor product wave-vector grid.

    The amplification matrix is batched (``numpy.linalg.eigvals`` supports
    stacked 9x9 matrices), so a 40x40 scan costs milliseconds instead of
    thousands of Python-level eigen-decompositions.
    """
    p_mat = collision_jacobian(coll, rho0, u0)
    kx = np.asarray(kxs, dtype=float)[:, None, None]  # (nx,1,1)
    ky = np.asarray(kys, dtype=float)[None, :, None]  # (1,ny,1)
    phase = np.exp(-1j * (kx * CX[None, None, :] + ky * CY[None, None, :]))  # (nx,ny,9)
    a = phase[..., :, None] * p_mat  # (nx, ny, 9, 9)
    mu = np.linalg.eigvals(a)
    return np.abs(mu).max(axis=-1)


def spectral_radius(
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    n_theta: int = 41,
    theta_max: float = math.pi,
    rho0: float = 1.0,
) -> dict[str, Any]:
    """Scan ``lambda_max(theta)`` over ``theta_x, theta_y in [0, pi]``.

    The D2Q9 lattice and a base flow along ``x`` are invariant under
    ``theta_y -> -theta_y`` and under a simultaneous sign flip of both
    components, so the first quadrant exhausts the spectrum.
    """
    thetas = np.linspace(0.0, theta_max, n_theta)
    lam = lambda_max_grid(coll, thetas, thetas, u0, rho0)
    i, j = np.unravel_index(int(np.argmax(lam)), lam.shape)
    return {
        "lambda_max": float(lam[i, j]),
        "theta_argmax": (float(thetas[i]), float(thetas[j])),
        "n_theta": n_theta,
        "u0": tuple(u0),
    }


def stability_point(
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    n_theta: int = 41,
) -> dict[str, Any]:
    """von Neumann verdict at one ``(tau, Ma)`` point."""
    scan = spectral_radius(coll, u0, n_theta=n_theta)
    scan.update(
        {
            "stable": scan["lambda_max"] <= 1.0 + STABILITY_TOL,
            "collision": coll.label(),
            "tau": coll.tau,
            "viscosity": coll.viscosity,
            "ma": float(np.hypot(u0[0], u0[1]) / CS),
        }
    )
    return scan


# ---------------------------------------------------------------------------
# Equivalent (second-order) equation: numerical viscosity from attenuation
# ---------------------------------------------------------------------------


def numerical_viscosity(
    k: float,
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    direction: float = 0.0,
) -> dict[str, Any]:
    """Shear-branch attenuation and the implied numerical viscosity.

    For wave vector ``k (cos d, sin d)`` the second-order (Chapman-Enskog)
    truncation predicts ``sigma_shear = -nu_num k^2 + O(k^4)``, hence

        ``nu_num(k) = -sigma_shear(k) / k^2``.

    The ``k -> 0`` limit recovers ``nu = (tau - 1/2)/3``; the ``k^2``
    coefficient is the leading (second-order) dispersion-dissipation
    correction reported by this module.
    """
    kx = k * math.cos(direction)
    ky = k * math.sin(direction)
    sp = spectrum(kx, ky, coll, u0)
    j = sp.shear_index
    sigma = float(sp.growth[j])
    return {
        "k": float(k),
        "direction": float(direction),
        "sigma_shear": sigma,
        "nu_num": float(-sigma / (k * k)) if k > 0 else float("nan"),
        "phase_speed": float(sp.phase_speed[j]),
        "lambda": float(abs(sp.mu[j])),
        "nu_ce": coll.viscosity,
        "branch": sp.labels[j],
    }


def shear_branch_curve(
    k_values: Sequence[float],
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    direction: float = 0.0,
    rho0: float = 1.0,
) -> list[dict[str, Any]]:
    """Track the shear branch over ``k_values`` by eigenvector continuation.

    At ``k -> 0`` the shear eigendirection is exactly the transverse-velocity
    perturbation, but at finite ``k`` the eigenvectors of ``A`` are not
    orthogonal and the kinetic eigenvalues of MRT/BGK can be (nearly) degenerate
    with hydrodynamic ones (e.g. ``|mu_kinetic| = 0`` at ``tau = 1``).  Picking
    the branch by moment content then jumps between branches.  This function
    follows the branch by maximum right-eigenvector overlap with the previous
    ``k`` (continuation), starting from the exact ``k -> 0`` direction, and
    reports the tracking overlap so the trace can be audited.
    """
    ks = np.sort(np.asarray(k_values, dtype=float))
    if ks.size == 0:
        return []
    out: list[dict[str, Any]] = []
    k0 = max(float(ks[0]), 1e-6)
    v_ref = seed_direction("shear", k0 * math.cos(direction), k0 * math.sin(direction), rho0, u0)
    v_ref = v_ref / max(np.linalg.norm(v_ref), 1e-300)
    for k in ks:
        kx = float(k) * math.cos(direction)
        ky = float(k) * math.sin(direction)
        sp = spectrum(kx, ky, coll, u0, rho0)
        overlap = np.abs(sp.eigvecs.conj().T @ v_ref)
        idx = int(np.argmax(overlap))
        v = sp.eigvecs[:, idx]
        v = v / max(np.linalg.norm(v), 1e-300)
        out.append(
            {
                "k": float(k),
                "index": idx,
                "mu": complex(sp.mu[idx]),
                "sigma": float(sp.growth[idx]),
                "nu_num": float(-sp.growth[idx] / (k * k)) if k > 0 else float("nan"),
                "phase_speed": float(sp.phase_speed[idx]),
                "overlap": float(overlap[idx]),
                "label_instantaneous": sp.labels[idx],
            }
        )
        v_ref = v
    return out


# ---------------------------------------------------------------------------
# Measured side: periodic-box single-mode experiment (full nonlinear LBM)
# ---------------------------------------------------------------------------


def _collide_np(f: np.ndarray, coll: Collision) -> np.ndarray:
    """Full nonlinear collision in float64, shape ``(9, ny, nx)``."""
    rho = f.sum(axis=0)
    ux = (CX[:, None, None] * f).sum(axis=0) / rho
    uy = (CY[:, None, None] * f).sum(axis=0) / rho
    feq = equilibrium(rho, ux, uy)
    if coll.kind == "bgk":
        return f - (f - feq) / coll.tau
    m = np.einsum("ij,jyx->iyx", M_D2Q9, f)
    m_eq = np.einsum("ij,jyx->iyx", M_D2Q9, feq)
    s = coll.rates().reshape(9, 1, 1)
    m = m - s * (m - m_eq)
    return np.einsum("ij,jyx->iyx", M_INV_D2Q9, m)


def _stream_np(f: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], shift=(int(CY[i]), int(CX[i])), axis=(0, 1))
    return out


def _fourier_mode(f: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Complex Fourier coefficient of ``f`` at ``(kx, ky)``, shape ``(9,)``."""
    ny, nx = f.shape[1], f.shape[2]
    xs = np.arange(nx)[None, :]
    ys = np.arange(ny)[:, None]
    phase = np.exp(-1j * (kx * xs + ky * ys))
    return np.einsum("iyx,yx->i", f, phase) / (nx * ny)


@dataclass
class ModalRun:
    """Result of one periodic-box single-mode experiment."""

    kx: float
    ky: float
    coll: Collision
    u0: tuple[float, float]
    seed: str
    target: str
    observable: str
    amplitude: float
    n_steps: int
    steps_run: int
    diverged: bool
    t: np.ndarray
    modal: np.ndarray  # (T, 9) complex Fourier coefficients of the perturbation
    target_index: int
    predicted_mu: complex
    measured_mu: complex
    measured_growth: float
    measured_phase_speed: float
    fit_window: tuple[int, int]
    fit_r2: float
    fit_stderr: float
    purity: float
    max_amp: float
    notes: dict[str, Any]

    @property
    def predicted_growth(self) -> float:
        return float(np.log(abs(self.predicted_mu)))

    @property
    def predicted_phase_speed(self) -> float:
        k = math.hypot(self.kx, self.ky)
        return -float(np.angle(self.predicted_mu)) / k if k > 0 else float("nan")

    @property
    def modal_growth_factor(self) -> float:
        """Growth factor of the perturbation norm over the whole run."""
        n0 = float(np.linalg.norm(self.modal[0]))
        return float(np.linalg.norm(self.modal[-1]) / max(n0, 1e-300))

    @property
    def diverged_or_grown(self) -> bool:
        """Nonlinear blow-up **or** growth by >= 100x (declared C2 criterion)."""
        return bool(self.diverged or self.modal_growth_factor >= 100.0)

    @property
    def growth_rel_error(self) -> float:
        denom = max(abs(self.predicted_growth), 1e-12)
        return abs(self.measured_growth - self.predicted_growth) / denom

    @property
    def phase_speed_rel_error(self) -> float:
        denom = max(abs(self.predicted_phase_speed), 1e-12)
        return abs(self.measured_phase_speed - self.predicted_phase_speed) / denom

    def summary(self) -> dict[str, Any]:
        return {
            "kx": self.kx,
            "ky": self.ky,
            "collision": self.coll.label(),
            "base_u": self.u0,
            "seed": self.seed,
            "target": self.target,
            "observable": self.observable,
            "target_branch": self.notes.get("target_branch"),
            "lambda_pred": abs(self.predicted_mu),
            "growth_pred": self.predicted_growth,
            "growth_meas": self.measured_growth,
            "growth_rel_err": self.growth_rel_error,
            "phase_speed_pred": self.predicted_phase_speed,
            "phase_speed_meas": self.measured_phase_speed,
            "phase_speed_rel_err": self.phase_speed_rel_error,
            "fit_r2": self.fit_r2,
            "fit_stderr": self.fit_stderr,
            "fit_window": list(self.fit_window),
            "purity": self.purity,
            "steps_run": self.steps_run,
            "diverged": self.diverged,
            "max_amp": self.max_amp,
        }


def _fit_exponential(
    t: np.ndarray, amp: np.ndarray, i0: int, i1: int
) -> tuple[float, float, float, float]:
    """Least-squares fit of ``ln amp(t) = a + s t`` on ``[i0, i1)``.

    Returns ``(Re s, Im s, R^2, stderr)`` with ``s = ln(mu)`` the complex
    per-step growth (``Re s = ln|mu|``, ``Im s = arg(mu)``).
    """
    tt = t[i0:i1]
    aa = amp[i0:i1]
    mask = np.abs(aa) > 0.0
    tt = tt[mask]
    aa = aa[mask]
    if tt.size < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ln_mag = np.log(np.abs(aa))
    ph = np.unwrap(np.angle(aa))
    mat = np.vstack([np.ones_like(tt), tt]).T
    coef_mag, *_ = np.linalg.lstsq(mat, ln_mag, rcond=None)
    coef_ph, *_ = np.linalg.lstsq(mat, ph, rcond=None)
    pred = mat @ coef_mag
    ss_res = float(np.sum((ln_mag - pred) ** 2))
    ss_tot = float(np.sum((ln_mag - ln_mag.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(tt.size - 2, 1)
    stderr = math.sqrt(ss_res / dof) / math.sqrt(max(np.sum((tt - tt.mean()) ** 2), 1e-30))
    return float(coef_mag[1]), float(coef_ph[1]), r2, stderr


def run_modal_box(
    kx: float,
    ky: float,
    coll: Collision,
    u0: Sequence[float] = (0.0, 0.0),
    *,
    n: int = 32,
    n_steps: int = 4000,
    seed: str = "shear",
    target: str = "dominant",
    observable: str = "auto",
    amplitude: float = 1e-6,
    rho0: float = 1.0,
    blowup_tol: float = 0.05,
    max_rel_amp: float = 1e-3,
    spinup: int = 12,
    early_stop: bool = True,
) -> ModalRun:
    """Integrate a periodic box seeded with one Fourier mode and fit its growth.

    Parameters
    ----------
    kx, ky:
        Wave vector in lattice units.  For an ``n x n`` box use
        ``kx = 2 pi mx / n`` (periodic compatibility).
    seed:
        ``"shear"``  - physical transverse velocity wave, ``u = u0 + A p cos(k.x)``
                       (no eigendata used);
        ``"sound"``  - physical density wave ``rho = rho0 + A cos(k.x)`` at ``u0``;
        ``"eigen"``  - the exact right eigenvector of the target eigenvalue
                       (branch-selective travelling wave / instability probe).
    target:
        ``"dominant"`` - the reference branch is the one carrying the largest
                         modal amplitude at the start of the fit window
                         (declared rule; the data choose the branch, the
                         *rate* is still compared against the analytical
                         eigenvalue), or an explicit branch: ``"shear"``,
                         ``"sound+"``, ``"sound-"``, ``"max"``.
    observable:
        ``"auto"`` (vorticity for the shear branch, density for the acoustic
        branches, left-eigenvector projection for kinetic branches),
        ``"vorticity"``, ``"transverse_momentum"``, ``"density"`` or
        ``"projection"``.
    amplitude:
        Initial perturbation amplitude; ``1e-6`` keeps quadratic terms at
        ``~1e-12`` so the run stays in the linear regime.
    blowup_tol, max_rel_amp:
        Abort thresholds (nonlinear blow-up / leaving the linear window).
    early_stop:
        Stop once the perturbation has left the useful window (five decades of
        decay below the spin-up amplitude, or ``1e6`` growth above it).  Disable
        for divergence probes that must exhaust ``n_steps``.
    """
    ny = nx = int(n)
    knorm = math.hypot(kx, ky)
    if knorm == 0.0:
        raise ValueError("run_modal_box needs a non-zero wave vector")
    khat, phat = _unit(kx, ky)

    xs = np.arange(nx)[None, :]
    ys = np.arange(ny)[:, None]
    phase_real = np.cos(kx * xs + ky * ys)
    phase_complex = np.exp(1j * (kx * xs + ky * ys))

    sp = spectrum(kx, ky, coll, u0, rho0)

    # -- initial condition ------------------------------------------------
    ux0, uy0 = float(u0[0]), float(u0[1])
    f0 = equilibrium(np.full((ny, nx), rho0), np.full((ny, nx), ux0), np.full((ny, nx), uy0))
    if seed == "shear":
        du = amplitude * phase_real
        f0 = equilibrium(
            np.full((ny, nx), rho0),
            np.full((ny, nx), ux0) + du * phat[0],
            np.full((ny, nx), uy0) + du * phat[1],
        )
    elif seed == "sound":
        f0 = equilibrium(
            np.full((ny, nx), rho0 + amplitude * phase_real),
            np.full((ny, nx), ux0),
            np.full((ny, nx), uy0),
        )
    elif seed == "eigen":
        idx_seed = sp.index_of("max" if target == "dominant" else target)
        if idx_seed is None:  # pragma: no cover - defensive
            idx_seed = sp.worst_index
        v = sp.eigvecs[:, idx_seed]
        v = v / max(np.max(np.abs(v)), 1e-300)
        f0 = f0 + np.real(v[:, None, None] * phase_complex[None]) * amplitude
    else:
        raise ValueError(f"unknown seed {seed!r}")

    # -- integrate --------------------------------------------------------
    f = f0.copy()
    modal = np.empty((n_steps + 1, 9), dtype=complex)
    modal[0] = _fourier_mode(f, kx, ky)  # k != 0: perturbation == deviation
    max_amp = 0.0
    diverged = False
    steps_run = 0
    norm0 = float(np.linalg.norm(modal[0]))
    lo = max(1e-13, norm0 * 1e-6)
    hi = norm0 * 1e7
    for t in range(1, n_steps + 1):
        f = _stream_np(_collide_np(f, coll))
        modal[t] = _fourier_mode(f, kx, ky)
        steps_run = t
        rho = f.sum(axis=0)
        ux = (CX[:, None, None] * f).sum(axis=0) / rho
        uy = (CY[:, None, None] * f).sum(axis=0) / rho
        max_u = float(np.max(np.hypot(ux - ux0, uy - uy0)))
        max_drho = float(np.max(np.abs(rho - rho0)))
        max_amp = max(max_amp, max_u, max_drho)
        if not np.isfinite(max_u) or max_u > blowup_tol:
            diverged = True
            break
        if max_u > max_rel_amp:
            break
        if early_stop and t > spinup:
            nrm = float(np.linalg.norm(modal[t]))
            if nrm < lo or nrm > hi:
                break
    modal = modal[: steps_run + 1]
    t = np.arange(modal.shape[0], dtype=float)

    proj = modal @ sp.left.T  # (T, 9) modal coefficients (left eigenvectors)
    energy = np.abs(proj) ** 2
    norm = np.linalg.norm(modal, axis=1)  # gauge-invariant perturbation size

    # -- reference branch (declared rule: largest modal amplitude at spin-up) --
    i0 = min(int(spinup), max(norm.size - 6, 0))
    i0 = min(max(i0, 0), max(norm.size - 6, 0))
    if target == "dominant":
        j = int(np.argmax(np.abs(proj[i0])))
        selection = "data-dominant@fit-start"
    else:
        j = sp.index_of(target)
        if j is None:  # pragma: no cover - defensive
            j = sp.worst_index
        selection = "explicit"
    branch = sp.labels[j]
    idx = int(j)
    mu_pred = complex(sp.mu[idx])
    purity_series = energy[:, idx] / np.maximum(energy.sum(axis=1), 1e-300)

    # -- fit window: after spin-up, single-mode, inside the linear band, -----
    #    above the float64 round-off floor.  A measurement counts only while
    #    the reference branch carries >= 99% of the modal energy (declared
    #    quality gate) and the O(1) background noise floor (~1e-16) is far
    #    below the modal amplitude (>= 1e-12).
    a_start = max(float(norm[i0]), 1e-300)
    ceiling = min(1e-2, a_start * 1e6)  # stay in the linear regime
    floor = max(1e-12, a_start * 1e-5)  # stay above float64 modal noise
    i1 = norm.size
    for tt in range(i0 + 1, norm.size):
        if norm[tt] > ceiling or norm[tt] < floor or purity_series[tt] < 0.99:
            i1 = tt
            break
    i1 = int(min(max(i1, i0 + 6), norm.size))

    # -- observable -------------------------------------------------------
    if observable == "auto":
        if branch == "shear":
            observable = "vorticity"
        elif branch == "sound":
            observable = "density"
        else:
            observable = "projection"
    if observable == "vorticity":
        mom = modal @ CVEL  # (T, 2) complex
        obs = 1j * (kx * mom[:, 1] - ky * mom[:, 0]) / rho0
    elif observable == "transverse_momentum":
        obs = (modal @ (CVEL @ phat)) / rho0
    elif observable == "density":
        obs = modal.sum(axis=1)
    elif observable == "projection":
        obs = modal @ sp.left[idx]
    else:
        raise ValueError(f"unknown observable {observable!r}")

    amp = np.abs(obs)
    purity_series = energy[:, idx] / np.maximum(energy.sum(axis=1), 1e-300)
    # if the reference branch is only weakly present in the chosen observable,
    # fall back to the modal projection (reported in notes)
    fallback = False
    if amp[i0:i1].size and np.max(amp[i0:i1]) == 0.0:  # pragma: no cover
        obs = modal @ sp.left[idx]
        amp = np.abs(obs)
        fallback = True

    growth, phase_rate, r2, stderr = _fit_exponential(t, obs, i0, i1)
    measured_mu = complex(
        math.exp(growth) * math.cos(phase_rate), math.exp(growth) * math.sin(phase_rate)
    )
    purity = float(np.mean(purity_series[i0:i1])) if i1 > i0 else float("nan")

    return ModalRun(
        kx=float(kx),
        ky=float(ky),
        coll=coll,
        u0=(ux0, uy0),
        seed=seed,
        target=target,
        observable=observable,
        amplitude=float(amplitude),
        n_steps=int(n_steps),
        steps_run=int(steps_run),
        diverged=bool(diverged),
        t=t,
        modal=modal,
        target_index=int(idx),
        predicted_mu=mu_pred,
        measured_mu=measured_mu,
        measured_growth=float(growth),
        measured_phase_speed=float(-phase_rate / knorm),
        fit_window=(int(i0), int(i1)),
        fit_r2=float(r2),
        fit_stderr=float(stderr),
        purity=purity,
        max_amp=float(max_amp),
        notes={
            "target_branch": branch,
            "selection": selection,
            "n": int(n),
            "lambda_pred": float(abs(mu_pred)),
            "purity_final": float(purity_series[min(i1, purity_series.size - 1)]),
            "observable_fallback": fallback,
            "modal_growth_factor": float(
                np.linalg.norm(modal[-1]) / max(np.linalg.norm(modal[0]), 1e-300)
            ),
            "k_norm": float(knorm),
        },
    )
