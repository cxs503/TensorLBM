from __future__ import annotations

import torch

from .d3q19 import C, equilibrium3d, macroscopic3d

# ---------------------------------------------------------------------------
# D3Q19 MRT transformation matrix (Lallemand & Luo 2000, d'Humières basis)
# Row order: rho, e, epsilon, jx, qx, jy, qy, jz, qz,
#            3pxx, pixx, pww, piww, pxy, pyz, pxz, mx, my, mz
# Velocity order matches d3q19.C (see d3q19.py for the full listing)
# ---------------------------------------------------------------------------
_M_D3Q19 = torch.tensor(
    [
        # rho
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        # e  (19*u2 - 30)
        [-30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
        # epsilon  ((21*u4 - 53*u2 + 24)/2)
        [ 12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        # jx  (cx)
        [  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
        # qx  ((5*u2-9)*cx)
        [  0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
        # jy  (cy)
        [  0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1],
        # qy  ((5*u2-9)*cy)
        [  0,  0,  0, -4,  4,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1],
        # jz  (cz)
        [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1],
        # qz  ((5*u2-9)*cz)
        [  0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1],
        # 3pxx  (3*cx2 - u2)
        [  0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
        # pixx  ((3*u2-5)*(3*cx2-u2))
        [  0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
        # pww  (cy2 - cz2)
        [  0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
        # piww  ((3*u2-5)*(cy2-cz2))
        [  0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
        # pxy  (cx*cy)
        [  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        # pyz  (cy*cz)
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1],
        # pxz  (cx*cz)
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0],
        # mx  ((cy2-cz2)*cx)
        [  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0],
        # my  ((cz2-cx2)*cy)
        [  0,  0,  0,  0,  0,  0,  0, -1,  1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
        # mz  ((cx2-cy2)*cz)
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1, -1,  1,  1, -1],
    ],
    dtype=torch.float32,
)
_M_INV_D3Q19 = torch.linalg.inv(_M_D3Q19)


def collide_bgk3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step for D3Q19."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_mrt3d(f: torch.Tensor, tau: float, s: torch.Tensor | None = None) -> torch.Tensor:
    """Multiple-relaxation-time (MRT) collision step for D3Q19.

    Uses the Lallemand & Luo (2000) d'Humières transformation matrix.  The
    19 relaxation rates correspond to the moment ordering: rho, e, epsilon,
    jx, qx, jy, qy, jz, qz, 3pxx, pixx, pww, piww, pxy, pyz, pxz, mx,
    my, mz.  Conserved modes (rho, jx, jy, jz) are never relaxed regardless
    of the values supplied for those positions in *s*.

    Args:
        f:   Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Viscous relaxation time.  Used for the stress modes (3pxx, pww,
             pxy, pyz, pxz) when *s* is ``None``.
        s:   Optional length-19 tensor of per-mode relaxation rates.  When
             ``None`` the standard defaults from Lallemand & Luo (2000) are
             used, with stress modes set to ``1/tau``.

    Returns:
        Post-collision distribution tensor with the same shape as *f*.
    """
    device = f.device
    M = _M_D3Q19.to(device)
    M_inv = _M_INV_D3Q19.to(device)

    if s is None:
        s_nu = 1.0 / tau
        s = torch.tensor(
            # rho   e     eps   jx    qx    jy    qy    jz    qz
            [0.0,  1.19, 1.4,  0.0,  1.2,  0.0,  1.2,  0.0,  1.2,
             # 3pxx  pixx  pww   piww  pxy   pyz   pxz   mx    my    mz
             s_nu, 1.4,  s_nu, 1.4,  s_nu, s_nu, s_nu, 1.98, 1.98, 1.98],
            dtype=torch.float32,
            device=device,
        )
    else:
        s = s.to(device)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.view(19, -1)
    feq_flat = feq.view(19, -1)

    m = M @ f_flat        # (19, nz*ny*nx)
    meq = M @ feq_flat    # (19, nz*ny*nx)

    m_out = m - s.unsqueeze(-1) * (m - meq)

    return (M_inv @ m_out).view(19, nz, ny, nx)


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Streaming by shifting each discrete direction for D3Q19.

    Args:
        f: distribution tensor of shape (19, nz, ny, nx).

    Returns:
        Streamed tensor of the same shape.
    """
    streamed = torch.empty_like(f)
    for i in range(19):
        cx, cy, cz = int(C[i, 0].item()), int(C[i, 1].item()), int(C[i, 2].item())
        # f[i] has shape (nz, ny, nx); dims=(0,1,2) correspond to z, y, x
        streamed[i] = torch.roll(f[i], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return streamed


__all__ = [
    "collide_bgk3d",
    "collide_mrt3d",
    "stream3d",
]
