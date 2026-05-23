"""D3Q27 lattice: 27-direction 3D LBM.

D3Q27 includes all 27 velocity directions (rest + face + edge + corner),
offering higher isotropy than D3Q19 at the cost of additional memory.

Velocity ordering
-----------------
  0 :  rest (0,0,0)
  1–6 :  face-centre  ±x, ±y, ±z
  7–18 : edge-centre  (±1,±1,0), (±1,0,±1), (0,±1,±1)
 19–26 : corner       (±1,±1,±1)

Weights: w_rest = 8/27, w_face = 2/27, w_edge = 1/54, w_corner = 1/216.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# D3Q27 lattice velocities (cx, cy, cz)
# ---------------------------------------------------------------------------
C = torch.tensor(
    [
        # rest
        [ 0,  0,  0],
        # face-centre (±x, ±y, ±z)
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
        # edge-centre (xy plane)
        [ 1,  1,  0], [-1, -1,  0],
        [ 1, -1,  0], [-1,  1,  0],
        # edge-centre (xz plane)
        [ 1,  0,  1], [-1,  0, -1],
        [ 1,  0, -1], [-1,  0,  1],
        # edge-centre (yz plane)
        [ 0,  1,  1], [ 0, -1, -1],
        [ 0,  1, -1], [ 0, -1,  1],
        # corners (±1, ±1, ±1)
        [ 1,  1,  1], [-1, -1, -1],
        [ 1,  1, -1], [-1, -1,  1],
        [ 1, -1,  1], [-1,  1, -1],
        [ 1, -1, -1], [-1,  1,  1],
    ],
    dtype=torch.int64,
)

W = torch.tensor(
    [
        8 / 27,                          # rest
        2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27,  # face
        1 / 54, 1 / 54, 1 / 54, 1 / 54,  # edge xy
        1 / 54, 1 / 54, 1 / 54, 1 / 54,  # edge xz
        1 / 54, 1 / 54, 1 / 54, 1 / 54,  # edge yz
        1 / 216, 1 / 216, 1 / 216, 1 / 216,  # corners +
        1 / 216, 1 / 216, 1 / 216, 1 / 216,  # corners -
    ],
    dtype=torch.float32,
)

# Opposite direction mapping: OPPOSITE[i] = j where C[j] = -C[i]
OPPOSITE = torch.tensor(
    [
        0,     # 0  ↔ 0
        2, 1,  # 1  ↔ 2
        4, 3,  # 3  ↔ 4
        6, 5,  # 5  ↔ 6
        8,  7,  # 7  ↔ 8
        10,  9,  # 9  ↔ 10
        12, 11,  # 11 ↔ 12
        14, 13,  # 13 ↔ 14
        16, 15,  # 15 ↔ 16
        18, 17,  # 17 ↔ 18
        20, 19,  # 19 ↔ 20
        22, 21,  # 21 ↔ 22
        24, 23,  # 23 ↔ 24
        26, 25,  # 25 ↔ 26
    ],
    dtype=torch.int64,
)


def equilibrium27(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute D3Q27 equilibrium distribution.

    Args:
        rho: Density of shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components of shape ``(nz, ny, nx)``.

    Returns:
        Tensor of shape ``(27, nz, ny, nx)``.
    """
    if device is None:
        device = rho.device
    c = C.to(device)
    w = W.to(device).view(27, 1, 1, 1)

    u_sq = ux * ux + uy * uy + uz * uz
    cu = (
        c[:, 0].view(27, 1, 1, 1) * ux
        + c[:, 1].view(27, 1, 1, 1) * uy
        + c[:, 2].view(27, 1, 1, 1) * uz
    )
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic27(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover rho, ux, uy, uz from D3Q27 distributions.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.

    Returns:
        Tuple ``(rho, ux, uy, uz)`` each of shape ``(nz, ny, nx)``.
    """
    if device is None:
        device = f.device
    c = C.to(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    uz = (f * c[:, 2].view(27, 1, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy, uz


# ---------------------------------------------------------------------------
# Streaming cache
# ---------------------------------------------------------------------------
_stream27_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def stream27(f: torch.Tensor) -> torch.Tensor:
    """Vectorised streaming step for D3Q27 (periodic boundaries).

    Uses cached index tensors per (shape, device) to avoid repeated
    construction inside the simulation loop.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.

    Returns:
        Streamed tensor of the same shape.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    device = f.device
    cache_key = (nz, ny, nx, device.type, device.index)

    if cache_key not in _stream27_cache:
        c = C.to(device)  # (27, 3)

        z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx

        q_idx = torch.arange(27, device=device).view(27, 1, 1, 1).expand(27, nz, ny, nx)
        z_idx = z_src.view(27, nz, 1, 1).expand(27, nz, ny, nx)
        y_idx = y_src.view(27, 1, ny, 1).expand(27, nz, ny, nx)
        x_idx = x_src.view(27, 1, 1, nx).expand(27, nz, ny, nx)
        _stream27_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream27_cache[cache_key]
    return f[q_idx, z_idx, y_idx, x_idx]


# ---------------------------------------------------------------------------
# BGK collision
# ---------------------------------------------------------------------------

def collide_bgk27(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision for D3Q27.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time (τ > 0.5).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    return f - (f - feq) / tau


# ---------------------------------------------------------------------------
# Smagorinsky helpers (shared with turbulence.py)
# ---------------------------------------------------------------------------

def _neq_stress_norm_27(f_neq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the non-equilibrium stress tensor for D3Q27."""
    device = f_neq.device
    c = C.to(device).float()  # (27, 3)
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    pi_xx = (cx * cx * f_neq).sum(0)
    pi_yy = (cy * cy * f_neq).sum(0)
    pi_zz = (cz * cz * f_neq).sum(0)
    pi_xy = (cx * cy * f_neq).sum(0)
    pi_xz = (cx * cz * f_neq).sum(0)
    pi_yz = (cy * cz * f_neq).sum(0)

    return torch.sqrt(
        pi_xx ** 2 + pi_yy ** 2 + pi_zz ** 2
        + 2.0 * (pi_xy ** 2 + pi_xz ** 2 + pi_yz ** 2)
    )


def _smagorinsky_tau_27(
    tau: float,
    pi_norm: torch.Tensor,
    rho: torch.Tensor,
    C_s: float,
) -> torch.Tensor:
    rho_safe = torch.clamp(rho, min=1e-12)
    discriminant = tau ** 2 + 18.0 * C_s ** 2 * pi_norm / rho_safe
    return 0.5 * (tau + torch.sqrt(torch.clamp(discriminant, min=0.0)))


def collide_smagorinsky_bgk27(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """D3Q27 BGK collision with Smagorinsky LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time τ₀ > 0.5.
        C_s: Smagorinsky constant (default 0.1).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff = _smagorinsky_tau_27(tau, pi_norm, rho, C_s)

    return f - f_neq / tau_eff.unsqueeze(0)


# ---------------------------------------------------------------------------
# MRT collision
# ---------------------------------------------------------------------------

def _build_d3q27_mrt_matrices() -> tuple[list[list[float]], list[list[float]]]:
    """Build the 27×27 MRT transformation matrix for D3Q27.

    Uses the complete tensor-product polynomial basis x^a y^b z^c for
    a,b,c ∈ {0,1,2} evaluated at the D3Q27 velocity set {−1,0,1}^3.
    This basis is guaranteed to be linearly independent on D3Q27
    (it is the Lagrange interpolation basis over {−1,0,1}^3).

    Moment ordering (i = 9*a + 3*b + c for a,b,c ∈ {0,1,2}):
        i=0  (a,b,c)=(0,0,0): ρ  (conserved)
        i=1  (a,b,c)=(0,0,1): jz (conserved)
        i=2  (a,b,c)=(0,0,2): Σcz²
        i=3  (a,b,c)=(0,1,0): jy (conserved)
        i=4  (a,b,c)=(0,1,1): Pyz
        i=5  (a,b,c)=(0,1,2): higher
        i=6  (a,b,c)=(0,2,0): Σcy²
        i=7  (a,b,c)=(0,2,1): higher
        i=8  (a,b,c)=(0,2,2): Σcy²cz²
        i=9  (a,b,c)=(1,0,0): jx (conserved)
        i=10 (a,b,c)=(1,0,1): Pxz
        i=11 (a,b,c)=(1,0,2): higher
        i=12 (a,b,c)=(1,1,0): Pxy
        i=13 (a,b,c)=(1,1,1): Pxyz
        i=14 (a,b,c)=(1,1,2): higher
        i=15 (a,b,c)=(1,2,0): higher
        i=16 (a,b,c)=(1,2,1): higher
        i=17 (a,b,c)=(1,2,2): higher
        i=18 (a,b,c)=(2,0,0): Σcx²
        i=19 (a,b,c)=(2,0,1): higher
        i=20 (a,b,c)=(2,0,2): Σcx²cz²
        i=21 (a,b,c)=(2,1,0): higher
        i=22 (a,b,c)=(2,1,1): higher
        i=23 (a,b,c)=(2,1,2): higher
        i=24 (a,b,c)=(2,2,0): Σcx²cy²
        i=25 (a,b,c)=(2,2,1): higher
        i=26 (a,b,c)=(2,2,2): Σcx²cy²cz²
    """
    import numpy as np

    c_np = C.numpy().astype(np.float64)
    cx, cy, cz = c_np[:, 0], c_np[:, 1], c_np[:, 2]

    rows = []
    for a in range(3):
        for b in range(3):
            for c_exp in range(3):
                rows.append(cx ** a * cy ** b * cz ** c_exp)
    M = np.array(rows)
    rank = np.linalg.matrix_rank(M)
    assert rank == 27, f"D3Q27 MRT matrix is rank-deficient (rank={rank})"
    M_inv = np.linalg.inv(M)
    return M.tolist(), M_inv.tolist()


_M_D3Q27_DATA, _M_D3Q27_INV_DATA = _build_d3q27_mrt_matrices()


def _get_d3q27_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    M = torch.tensor(_M_D3Q27_DATA, dtype=torch.float32, device=device)
    M_inv = torch.tensor(_M_D3Q27_INV_DATA, dtype=torch.float32, device=device)
    return M, M_inv


def collide_mrt27(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """Multi-relaxation-time (MRT) collision for D3Q27.

    Uses the tensor-product polynomial basis x^a y^b z^c (a,b,c ∈ {0,1,2}).
    Physical stress modes: indices 4 (Pyz), 10 (Pxz), 12 (Pxy),
    plus diagonal stress entries from indices 2, 6, 18 and their combinations.
    For simplicity this implementation applies:
    - Conserved moments (ρ and j): rate 0
    - Stress-related modes (4,10,12 off-diagonal; 2,6,18 diagonal): rate 1/tau
    - All other non-hydrodynamic modes: rate s_q or s_e or s_eps

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Shear relaxation time (τ > 0.5).
        s_e, s_eps, s_q, s_pi: Non-hydrodynamic relaxation rates.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q27_mrt_matrices(device)

    s_nu = 1.0 / tau
    # Moment index ordering: i = 9*a + 3*b + c  (a,b,c in {0,1,2})
    # Conserved: i=0 (ρ), i=3 (jy), i=9 (jx), i=1 (jz)
    # Stress off-diag: i=4 (Pyz), i=10 (Pxz), i=12 (Pxy)
    # Stress diagonal combinations: i=2, i=6, i=18
    # Higher order: everything else
    conserved = {0, 1, 3, 9}
    stress = {2, 4, 6, 10, 12, 18}
    s_rates = []
    for idx in range(27):
        if idx in conserved:
            s_rates.append(0.0)
        elif idx in stress:
            s_rates.append(s_nu)
        elif idx in {8, 20, 24, 26}:   # high-order corner modes
            s_rates.append(s_eps)
        else:
            s_rates.append(s_q)

    s_vec = torch.tensor(s_rates, dtype=f.dtype, device=device)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    feq_flat = feq.reshape(27, -1)

    m = M @ f_flat
    m_eq = M @ feq_flat
    m_star = m - s_vec.unsqueeze(1) * (m - m_eq)
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


def collide_smagorinsky_mrt27(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 MRT collision with Smagorinsky LES sub-grid turbulence model.

    The stress modes receive a spatially varying relaxation rate
    ``1/τ_eff(x)`` from the Smagorinsky model; all other modes use fixed rates.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time τ₀ > 0.5.
        C_s: Smagorinsky constant (default 0.1).
        s_e, s_eps, s_q, s_pi: Non-hydrodynamic relaxation rates.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q27_mrt_matrices(device)

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff = _smagorinsky_tau_27(tau, pi_norm, rho, C_s)
    s_nu_field = 1.0 / tau_eff  # (nz, ny, nx)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    feq_flat = feq.reshape(27, -1)
    s_nu_flat = s_nu_field.reshape(-1)
    N = f_flat.shape[1]

    # Same moment classification as collide_mrt27
    conserved = {0, 1, 3, 9}
    stress = {2, 4, 6, 10, 12, 18}
    s_rates = []
    for idx in range(27):
        if idx in conserved:
            s_rates.append(0.0)
        elif idx in stress:
            s_rates.append(0.0)  # will be overwritten
        elif idx in {8, 20, 24, 26}:
            s_rates.append(s_eps)
        else:
            s_rates.append(s_q)

    s_fixed = torch.tensor(s_rates, dtype=f.dtype, device=device)
    stress_modes = torch.tensor(sorted(stress), device=device)
    s_vec = s_fixed.unsqueeze(1).expand(27, N).clone()
    s_vec[stress_modes] = s_nu_flat.unsqueeze(0).expand(len(stress), N)

    m = M @ f_flat
    m_eq = M @ feq_flat
    m_star = m - s_vec * (m - m_eq)
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


def correct_mass27(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    """Rescale all populations to restore the total mass to *target_mass*.

    Useful after applying boundary conditions that do not conserve mass
    exactly (e.g. pressure BCs) to prevent long-term mass drift.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        target_mass: Desired total mass (sum of all populations).

    Returns:
        Rescaled distribution tensor of the same shape.
    """
    current_mass = float(f.sum().item())
    if abs(current_mass) < 1e-30:
        return f
    return f * (target_mass / current_mass)


__all__ = [
    # Lattice constants
    "C",
    "W",
    "OPPOSITE",
    # Lattice functions
    "equilibrium27",
    "macroscopic27",
    # Streaming
    "stream27",
    # Collision
    "collide_bgk27",
    "collide_mrt27",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    # Utilities
    "correct_mass27",
]
