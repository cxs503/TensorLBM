"""Full von Neumann linear stability analysis for D2Q9/D3Q19 LBM.

Covers BGK, TRT, and MRT collision operators on both D2Q9 and D3Q19 lattices.
Computes exact numerical dispersion relations ω(k), dissipation Im ω(k),
stability envelopes, and compares with analytical Navier–Stokes linear response.

Theory
------
Linearise the lattice Boltzmann equation around a uniform base state
f_i^eq(ρ₀, u₀).  Inserting the Fourier ansatz
δf_i = v_i exp(i(k·x − ωt)) gives the eigenvalue problem

    A(k) v = exp(−iω) v,    A(k) = diag(exp(−ik·c_i)) P

where P = I − C(I − J_eq) is the linearised collision operator.
The eigenvalues μ_j of A are the per-timestep amplification factors:

    growth rate   σ_j = ln|μ_j|          (> 0 → unstable)
    phase speed   c_j = −arg(μ_j)/|k|

Since μ = exp(−iω), we have ω = i ln μ, so:
    Re(ω) = −arg(μ)     → phase speed = Re(ω)/|k|
    Im(ω) = ln|μ|       → dissipation (negative = damping)

For the shear branch, the Chapman–Enskog prediction is:
    ω = −i ν k²    →    Im ω = −ν k²

For the acoustic branch, BGK gives:
    ω = ±c_s k − i ν k²    (same attenuation as shear)

The Stokes prediction (zero bulk viscosity) would give Im ω = −(2/3)νk²,
but BGK LBM has inherent bulk viscosity, so the actual attenuation is −νk².
This is a known and documented property of the BGK collision operator.

Numerical conventions
---------------------
* Lattice units: dx = dt = 1, c_s² = 1/3.
* BGK: C = I/τ, ν = (τ − 1/2)/3.
* TRT: τ₊ = τ, τ₋ = 1/2 + Λ/(τ − 1/2), ν = (τ₊ − 1/2)/3.
  Optimal Λ = 3/16 eliminates wall-placement errors in Poiseuille flow.
* MRT: moment-space relaxation with independent rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# ===========================================================================
# Lattice definitions
# ===========================================================================

# --- D2Q9 ---
D2Q9_CX = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
D2Q9_CY = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
D2Q9_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
D2Q9_C = np.stack([D2Q9_CX, D2Q9_CY], axis=1)  # (9, 2)
D2Q9_Q = 9
D2Q9_OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# --- D3Q19 ---
D3Q19_C = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
], dtype=float)
D3Q19_W = np.array(
    [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
)
D3Q19_Q = 19
D3Q19_OPPOSITE = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
)

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

STABILITY_TOL = 1e-9

# ===========================================================================
# D2Q9 MRT matrices (Lallemand & Luo 2000)
# ===========================================================================

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

# ===========================================================================
# D3Q19 MRT matrices
# ===========================================================================

def _build_d3q19_mrt_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Build D3Q19 MRT moment matrix and its inverse."""
    cx, cy, cz = D3Q19_C[:, 0], D3Q19_C[:, 1], D3Q19_C[:, 2]
    e2 = cx**2 + cy**2 + cz**2
    e4 = e2**2
    matrix = np.array(
        [
            np.ones(19),
            19.0 * e2 - 30.0,
            (21.0 * e4 - 53.0 * e2 + 24.0) / 2.0,
            cx,
            cx * e2 * (5.0 * e2 - 9.0) / 2.0,
            cy,
            cy * e2 * (5.0 * e2 - 9.0) / 2.0,
            cz,
            cz * e2 * (5.0 * e2 - 9.0) / 2.0,
            3.0 * cx**2 - e2,
            cx**2 - cy**2,
            cx * cy,
            cx * cz,
            cy * cz,
            (3.0 * e2 - 5.0) * (3.0 * cx**2 - e2) / 2.0,
            (3.0 * e2 - 5.0) * (cx**2 - cy**2) / 2.0,
            cx**2 * cy,
            cx**2 * cz,
            cy**2 * cx,
        ]
    )
    assert np.linalg.matrix_rank(matrix) == 19
    return matrix, np.linalg.inv(matrix)

M_D3Q19, M_INV_D3Q19 = _build_d3q19_mrt_matrices()

_I9 = np.eye(9)
_I19 = np.eye(19)


# ===========================================================================
# Collision model descriptor
# ===========================================================================

@dataclass(frozen=True)
class Collision:
    """Collision-model descriptor for von Neumann analysis.

    Parameters
    ----------
    kind : "bgk" | "trt" | "mrt"
    tau : relaxation time for shear viscosity (τ₊ for TRT)
    lattice : "d2q9" | "d3q19"
    lambda_trt : TRT magic parameter Λ (only for kind="trt")
    s_e, s_eps, s_q, s_pi : MRT relaxation rates (only for kind="mrt")
    """

    kind: str = "bgk"
    tau: float = 0.6
    lattice: str = "d2q9"
    lambda_trt: float = 3.0 / 16.0
    s_e: float = 1.64
    s_eps: float = 1.54
    s_q: float = 1.7
    s_pi: float | None = None

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in ("bgk", "trt", "mrt"):
            raise ValueError(f"unsupported collision kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        lat = self.lattice.lower()
        if lat not in ("d2q9", "d3q19"):
            raise ValueError(f"unsupported lattice: {self.lattice!r}")
        object.__setattr__(self, "lattice", lat)
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(f"tau must be finite and > 0, got {self.tau}")

    @property
    def q(self) -> int:
        return D2Q9_Q if self.lattice == "d2q9" else D3Q19_Q

    @property
    def viscosity(self) -> float:
        """Chapman–Enskog shear viscosity ν = (τ₊ − 1/2)/3."""
        return (self.tau - 0.5) / 3.0

    def _rates_d2q9(self) -> np.ndarray:
        if self.kind == "bgk":
            return np.full(9, 1.0 / self.tau)
        if self.kind == "trt":
            tau_minus = 0.5 + self.lambda_trt / (self.tau - 0.5)
            s_plus = 1.0 / self.tau
            s_minus = 1.0 / tau_minus
            # MOMENT_ORDER = (rho, e, eps, jx, qx, jy, qy, pxx, pxy)
            return np.array([0.0, s_plus, s_plus, 0.0, s_minus, 0.0, s_minus, s_plus, s_plus])
        # MRT
        s_pi = self.s_pi if self.s_pi is not None else self.s_e
        return np.array([
            0.0, self.s_e, self.s_eps, 0.0, self.s_q, 0.0, self.s_q,
            1.0 / self.tau, 1.0 / self.tau,
        ])

    def _rates_d3q19(self) -> np.ndarray:
        """19-entry relaxation rate vector for D3Q19 MRT ordering.

        D3Q19 MRT moment ordering (matching solver3d._mrt3d_s_vec):
        0:rho  1:e  2:eps  3:jx  4:qx  5:jy  6:qy  7:jz  8:qz
        9:pxx  10:pww  11:pxy  12:pxz  13:pyz
        14:Txx  15:Tww  16:cubic_x  17:cubic_y  18:cubic_z

        Conserved: 0, 3, 5, 7 (rho, jx, jy, jz) → rate 0
        Shear stress: 9-13 → rate 1/tau
        """
        if self.kind == "bgk":
            return np.full(19, 1.0 / self.tau)
        if self.kind == "trt":
            tau_minus = 0.5 + self.lambda_trt / (self.tau - 0.5)
            s_plus = 1.0 / self.tau
            s_minus = 1.0 / tau_minus
            rates = np.zeros(19)
            # Even (symmetric) moments: rho(0), e(1), eps(2), pxx(9), pww(10),
            #   pxy(11), pxz(12), pyz(13), Txx(14), Tww(15)
            even_idx = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15]
            # Odd (antisymmetric) moments: jx(3), qx(4), jy(5), qy(6),
            #   jz(7), qz(8), cubic_x(16), cubic_y(17), cubic_z(18)
            odd_idx = [3, 4, 5, 6, 7, 8, 16, 17, 18]
            for i in even_idx:
                rates[i] = 0.0 if i == 0 else s_plus
            for i in odd_idx:
                rates[i] = 0.0 if i in (3, 5, 7) else s_minus
            return rates
        # MRT: match solver3d._mrt3d_s_vec structure
        s_pi = self.s_pi if self.s_pi is not None else self.s_e
        s_nu = 1.0 / self.tau
        # head = [0, s_e, s_eps, 0, s_q, 0, s_q, 0, s_q]  (indices 0-8)
        # s_nu * 5  (indices 9-13, shear stress)
        # tail = [s_pi, s_pi, 1, 1, 1]  (indices 14-18)
        return np.array([
            0.0, self.s_e, self.s_eps, 0.0, self.s_q, 0.0, self.s_q,
            0.0, self.s_q,
            s_nu, s_nu, s_nu, s_nu, s_nu,
            s_pi, s_pi, 1.0, 1.0, 1.0,
        ])

    def operator_matrix(self) -> np.ndarray:
        """Collision matrix C such that f^post = f − C(f − f^eq)."""
        q = self.q
        if self.kind == "bgk":
            return np.eye(q) / self.tau
        if self.lattice == "d2q9":
            M, M_inv = M_D2Q9, M_INV_D2Q9
            rates = self._rates_d2q9()
        else:
            M, M_inv = M_D3Q19, M_INV_D3Q19
            rates = self._rates_d3q19()
        return M_inv @ np.diag(rates) @ M

    def label(self) -> str:
        if self.kind == "bgk":
            return f"BGK({self.lattice},τ={self.tau:.4g})"
        if self.kind == "trt":
            return f"TRT({self.lattice},τ₊={self.tau:.4g},Λ={self.lambda_trt:.4g})"
        return f"MRT({self.lattice},τ={self.tau:.4g})"


# ===========================================================================
# Equilibrium and Jacobian
# ===========================================================================

def _get_lattice(lattice: str):
    if lattice == "d2q9":
        return D2Q9_C, D2Q9_W, D2Q9_Q
    return D3Q19_C, D3Q19_W, D3Q19_Q


def equilibrium(lattice: str, rho: np.ndarray, *u: np.ndarray) -> np.ndarray:
    """D2Q9 or D3Q19 equilibrium, shape (Q,) + rho.shape."""
    C, W, Q = _get_lattice(lattice)
    rho = np.asarray(rho, dtype=float)
    u_arr = np.stack(u, axis=0)
    dim = len(u)
    shape = (Q,) + (1,) * rho.ndim
    cu = 3.0 * sum(C[:, d].reshape(shape) * u[d][None] for d in range(dim))
    u2 = sum(u[d] * u[d] for d in range(dim))[None]
    return W.reshape(shape) * rho[None] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)


def equilibrium_jacobian(lattice: str, rho0: float = 1.0, u0: Sequence[float] | None = None) -> np.ndarray:
    """J_eq[i,j] = d f_i^eq / d f_j at uniform state."""
    if u0 is None:
        u0 = (0.0, 0.0) if lattice == "d2q9" else (0.0, 0.0, 0.0)
    C, W, Q = _get_lattice(lattice)
    dim = len(u0)
    cxu = sum(C[:, d] * u0[d] for d in range(dim))
    u2 = sum(u0[d]**2 for d in range(dim))
    a_i = W * (1.0 + 3.0 * cxu + 4.5 * cxu * cxu - 1.5 * u2)
    b = W[:, None] * (3.0 * C + 9.0 * cxu[:, None] * C - 3.0 * np.array(u0)[None, :])
    return a_i[:, None] + b @ (C - np.array(u0)[None, :]).T


def collision_jacobian(coll: Collision, rho0: float = 1.0, u0: Sequence[float] | None = None) -> np.ndarray:
    """Linearised post-collision map P = I − C(I − J_eq)."""
    if u0 is None:
        u0 = (0.0,) * (2 if coll.lattice == "d2q9" else 3)
    j_eq = equilibrium_jacobian(coll.lattice, rho0, u0)
    q = coll.q
    return np.eye(q) - coll.operator_matrix() @ (np.eye(q) - j_eq)


# ===========================================================================
# Amplification matrix and spectrum
# ===========================================================================

def amplification_matrix(
    k: np.ndarray,
    coll: Collision,
    u0: Sequence[float] | None = None,
    rho0: float = 1.0,
) -> np.ndarray:
    """Complex Q×Q amplification matrix A = diag(exp(−ik·c)) P."""
    if u0 is None:
        u0 = (0.0,) * (2 if coll.lattice == "d2q9" else 3)
    C, _, Q = _get_lattice(coll.lattice)
    p = collision_jacobian(coll, rho0, u0)
    k_dot_c = sum(k[d] * C[:, d] for d in range(len(k)))
    phase = np.exp(-1j * k_dot_c)
    return phase[:, None] * p


def spectrum(
    k: np.ndarray,
    coll: Collision,
    u0: Sequence[float] | None = None,
    rho0: float = 1.0,
) -> dict[str, Any]:
    """Compute eigenvalues and identify hydrodynamic branches."""
    if u0 is None:
        u0 = (0.0,) * (2 if coll.lattice == "d2q9" else 3)
    C, _, Q = _get_lattice(coll.lattice)
    dim = len(k)
    A = amplification_matrix(np.array(k, dtype=float), coll, u0, rho0)
    mu, eigvecs = np.linalg.eig(A)
    left = np.linalg.inv(eigvecs)

    knorm = float(np.linalg.norm(k))

    if knorm > 1e-15:
        khat = np.array(k) / knorm
        if dim == 2:
            transverse = np.array([-khat[1], khat[0]])
        else:
            transverse = np.array([-khat[1], khat[0], 0.0])
            if abs(transverse[0]) < 1e-10 and abs(transverse[1]) < 1e-10:
                transverse = np.array([0.0, -khat[2], khat[1]])

        eps = 1e-7
        ux0 = list(u0)
        base_eq = equilibrium(coll.lattice, np.array(rho0), *[np.array(u) for u in ux0])

        ux_shear = [ux0[d] + eps * transverse[d] for d in range(dim)]
        shear_eq = equilibrium(coll.lattice, np.array(rho0), *[np.array(u) for u in ux_shear])
        shear_dir = (shear_eq - base_eq).flatten() / eps
        shear_dir /= max(np.linalg.norm(shear_dir), 1e-300)

        sound_eq = equilibrium(coll.lattice, np.array(rho0 + eps), *[np.array(u) for u in ux0])
        sound_dir = (sound_eq - base_eq).flatten() / eps
        sound_dir /= max(np.linalg.norm(sound_dir), 1e-300)

        overlaps_shear = np.abs(eigvecs.conj().T @ shear_dir)
        overlaps_sound = np.abs(eigvecs.conj().T @ sound_dir)

        shear_index = int(np.argmax(overlaps_shear))

        remaining = [i for i in range(Q) if i != shear_index]
        sound_overlaps_remaining = [(overlaps_sound[i], i) for i in remaining]
        sound_overlaps_remaining.sort(reverse=True)
        sound_indices = tuple(sound_overlaps_remaining[i][1] for i in range(min(2, len(sound_overlaps_remaining))))
    else:
        shear_index = 0
        sound_indices = (0, 0)
        for i in range(Q):
            if abs(mu[i] - 1.0) < 1e-10:
                shear_index = i
                break
        sound_indices = (shear_index, shear_index)

    growth = np.log(np.abs(mu))
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_speed = -np.angle(mu) / max(knorm, 1e-300)

    return {
        "mu": mu,
        "eigvecs": eigvecs,
        "left": left,
        "growth": growth,
        "phase_speed": phase_speed,
        "shear_index": shear_index,
        "sound_indices": sound_indices,
        "lam_max": float(np.max(np.abs(mu))),
        "worst_index": int(np.argmax(np.abs(mu))),
    }


# ===========================================================================
# Dispersion relation utilities
# ===========================================================================

def numerical_viscosity_shear(
    k: float,
    coll: Collision,
    direction: float = 0.0,
    u0: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Shear-branch attenuation and implied numerical viscosity."""
    if u0 is None:
        u0 = (0.0,) * (2 if coll.lattice == "d2q9" else 3)
    dim = len(u0)
    k_vec = np.array([k * math.cos(direction)] + [k * math.sin(direction)] + [0.0] * max(0, dim - 2))[:dim]
    sp = spectrum(k_vec, coll, u0)
    j = sp["shear_index"]
    sigma = float(sp["growth"][j])
    nu_num = float(-sigma / (k * k)) if k > 0 else float("nan")
    return {
        "k": k,
        "sigma_shear": sigma,
        "nu_num": nu_num,
        "nu_ce": coll.viscosity,
        "nu_rel_err": abs(nu_num / coll.viscosity - 1.0) if k > 0 and coll.viscosity > 0 else float("nan"),
    }


def acoustic_attenuation(
    k: float,
    coll: Collision,
    direction: float = 0.0,
    u0: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Acoustic branch: Re ω and Im ω vs analytical predictions.

    For BGK LBM: ω = ±c_s k − i ν k²  (same attenuation as shear)
    Stokes (zero bulk viscosity): ω = ±c_s k − i (2/3) ν k²
    """
    if u0 is None:
        u0 = (0.0,) * (2 if coll.lattice == "d2q9" else 3)
    dim = len(u0)
    k_vec = np.array([k * math.cos(direction)] + [k * math.sin(direction)] + [0.0] * max(0, dim - 2))[:dim]
    sp = spectrum(k_vec, coll, u0)

    si = sp["sound_indices"]
    if sp["phase_speed"][si[0]] > 0:
        sound_idx = si[0]
    elif len(si) > 1:
        sound_idx = si[1]
    else:
        sound_idx = si[0]

    mu = sp["mu"][sound_idx]
    omega = 1j * np.log(mu)
    re_omega = float(np.real(omega))
    im_omega = float(np.imag(omega))

    cs = CS
    nu = coll.viscosity
    re_omega_analytical = cs * k
    im_omega_bgk = -nu * k * k
    im_omega_stokes = -(2.0 / 3.0) * nu * k * k

    return {
        "k": k,
        "re_omega": re_omega,
        "im_omega": im_omega,
        "re_omega_analytical": re_omega_analytical,
        "im_omega_bgk": im_omega_bgk,
        "im_omega_stokes": im_omega_stokes,
        "re_omega_rel_err": abs(re_omega / re_omega_analytical - 1.0) if re_omega_analytical != 0 else float("nan"),
        "im_omega_vs_bgk_rel_err": abs(im_omega / im_omega_bgk - 1.0) if im_omega_bgk != 0 else float("nan"),
        "im_omega_vs_stokes_rel_err": abs(im_omega / im_omega_stokes - 1.0) if im_omega_stokes != 0 else float("nan"),
        "phase_speed": float(sp["phase_speed"][sound_idx]),
        "cs_num": float(re_omega / k) if k > 0 else float("nan"),
        "cs_rel_err": abs(re_omega / (cs * k) - 1.0) if k > 0 else float("nan"),
    }


def _lambda_max_grid_nonconserved(
    coll: Collision,
    thetas_x: np.ndarray,
    thetas_y: np.ndarray,
    u0: Sequence[float],
) -> np.ndarray:
    """λ_max excluding conserved modes (|μ|≈1 at k=0)."""
    dim = len(u0)
    nx, ny = len(thetas_x), len(thetas_y)
    lam = np.zeros((nx, ny))
    for i, tx in enumerate(thetas_x):
        for j, ty in enumerate(thetas_y):
            k = np.array([tx, ty] + [0.0] * max(0, dim - 2))[:dim]
            sp = spectrum(k, coll, u0)
            # Exclude eigenvalues very close to 1 (conserved modes)
            non_conserved = np.abs(sp["mu"] - 1.0) > 1e-6
            if np.any(non_conserved):
                lam[i, j] = float(np.max(np.abs(sp["mu"][non_conserved])))
            else:
                lam[i, j] = 1.0
    return lam


def _lambda_max_grid(
    coll: Collision,
    thetas_x: np.ndarray,
    thetas_y: np.ndarray,
    u0: Sequence[float],
) -> np.ndarray:
    """λ_max on a grid of (θ_x, θ_y), including all modes."""
    dim = len(u0)
    nx, ny = len(thetas_x), len(thetas_y)
    lam = np.zeros((nx, ny))
    for i, tx in enumerate(thetas_x):
        for j, ty in enumerate(thetas_y):
            k = np.array([tx, ty] + [0.0] * max(0, dim - 2))[:dim]
            sp = spectrum(k, coll, u0)
            lam[i, j] = sp["lam_max"]
    return lam


def stability_scan(
    coll: Collision,
    tau_values: Sequence[float] | None = None,
    ma_values: Sequence[float] | None = None,
    n_theta: int = 33,
    theta_max: float = math.pi,
) -> list[dict[str, Any]]:
    """Scan λ_max over τ and Ma to build stability envelope."""
    if tau_values is None:
        tau_values = np.linspace(0.5001, 2.0, 30)
    if ma_values is None:
        ma_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    thetas = np.linspace(0.0, theta_max, n_theta)
    results = []

    for tau in tau_values:
        c = Collision(kind=coll.kind, tau=tau, lattice=coll.lattice,
                      lambda_trt=coll.lambda_trt, s_e=coll.s_e, s_eps=coll.s_eps, s_q=coll.s_q)
        for ma in ma_values:
            u0 = (ma * CS, 0.0) if coll.lattice == "d2q9" else (ma * CS, 0.0, 0.0)
            lam_grid = _lambda_max_grid(c, thetas, thetas, u0)
            lam_max = float(np.max(lam_grid))
            results.append({
                "tau": tau,
                "ma": ma,
                "lambda_max": lam_max,
                "stable": lam_max <= 1.0 + STABILITY_TOL,
            })
    return results


# ===========================================================================
# TRT Λ optimality comparison
# ===========================================================================

def trt_lambda_comparison(
    tau: float = 0.6,
    lattice: str = "d2q9",
    lambda_values: Sequence[float] = (0.0, 0.05, 0.1, 3.0/16.0, 0.25, 0.3, 0.4),
    n_theta: int = 41,
) -> list[dict[str, Any]]:
    """Compare max|μ| (non-conserved) for different TRT Λ values at fixed τ."""
    thetas = np.linspace(0.0, math.pi, n_theta)
    results = []

    bgk = Collision(kind="bgk", tau=tau, lattice=lattice)
    u0 = (0.0,) * (2 if lattice == "d2q9" else 3)
    bgk_lam = _lambda_max_grid_nonconserved(bgk, thetas, thetas, u0)
    bgk_max = float(np.max(bgk_lam))
    results.append({
        "kind": "bgk",
        "tau": tau,
        "lambda_param": None,
        "lambda_max": bgk_max,
    })

    for lam_val in lambda_values:
        trt = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=lam_val)
        trt_lam = _lambda_max_grid_nonconserved(trt, thetas, thetas, u0)
        trt_max = float(np.max(trt_lam))
        results.append({
            "kind": "trt",
            "tau": tau,
            "lambda_param": lam_val,
            "lambda_max": trt_max,
        })

    return results


# ===========================================================================
# Isotropy check
# ===========================================================================

def isotropy_check(
    coll: Collision,
    k: float = 0.01,
    angles: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Check c_s^num at different propagation directions."""
    if angles is None:
        angles = [0.0, math.pi / 8, math.pi / 4]
    results = []
    for angle in angles:
        if coll.lattice == "d2q9":
            k_vec = np.array([k * math.cos(angle), k * math.sin(angle)])
        else:
            k_vec = np.array([k * math.cos(angle), k * math.sin(angle), 0.0])
        sp = spectrum(k_vec, coll)
        si = sp["sound_indices"]
        if sp["phase_speed"][si[0]] > 0:
            sound_idx = si[0]
        elif len(si) > 1:
            sound_idx = si[1]
        else:
            sound_idx = si[0]
        cs_num = float(sp["phase_speed"][sound_idx])
        results.append({
            "angle_deg": math.degrees(angle),
            "cs_num": cs_num,
            "cs_rel_err": abs(cs_num / CS - 1.0),
        })
    return results
