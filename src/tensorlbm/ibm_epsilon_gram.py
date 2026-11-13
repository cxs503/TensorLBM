"""IBM epsilon-Gram implicit correction — semantics-exact transplant of
joconnor22/LIFE (commit 2af2428) into TensorLBM.

This module reproduces, line-for-line, the IBM force-transfer correction used
by LIFE.  It is a *normalisation* fix: a per-marker multiplier ``epsilon`` is
obtained by solving a small linear (Gram) system so that the regularised-Dirac
spreading of the Lagrangian force is corrected for (a) the discrete delta not
integrating to exactly one and (b) the overlap of neighbouring markers'
supports.

Source map (zero hand-copy; every rule cited to the OSS file:line):

* 3-point regularised delta, support |r| <= 1.5
    inc/Utils.h:220-231  ``diracDelta(dist)``
      |r| <= 0.5   : (1 + sqrt(1 - 3 r^2)) / 3
      0.5<|r|<=1.5 : (5 - 3 r - sqrt(-3 (1-r)^2 + 1)) / 6
      |r| > 1.5    : 0
* support build (5x5 window about round(x), keep |dx|<1.5 & |dy|<1.5, no wrap)
    src/IBMNode.cpp:140-180  (``stencilWidth = 1.5``)
* marker spacing ds = min distance to any other marker of the same body
    src/IBMNode.cpp:186-205  (``computeDs``)
* force spread  F_lattice += force * epsilon * ds * diracVal
    src/IBMNode.cpp:73-74    (``spread``)
* Gram system   A[i,j] = sum_s  w_i(s) * delta(|x_j - x_i(s)|)
                                        * delta(|y_j - y_i(s)|) * ds_j
                solve  A * epsilon = 1   (b = ones)
    src/Objects.cpp:235-320  (``computeEpsilon``)
* recompute cadence: t==0 for rigid bodies, every step for flexible bodies
    src/Objects.cpp:271-272

The transplant covers LIFE's 2D path; the 3D path is out of scope for this
XVAL and is recorded as a limitation.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "life_delta_3pt",
    "build_support_2d",
    "compute_ds_2d",
    "compute_epsilon_gram",
    "spread_force_2d",
    "interpolate_force_2d",
]


# ---------------------------------------------------------------------------
# Delta kernel  (inc/Utils.h:220-231)
# ---------------------------------------------------------------------------


def life_delta_3pt(r: torch.Tensor) -> torch.Tensor:
    """LIFE 3-point regularised Dirac delta (inc/Utils.h:220-231).

    phi(r) = (1 + sqrt(1 - 3 r^2)) / 3                 if |r| <= 0.5
             (5 - 3|r| - sqrt(-3 (1-|r|)^2 + 1)) / 6   if 0.5 < |r| <= 1.5
             0                                          if |r| > 1.5
    """
    ra = r.abs()
    inner = (1.0 + torch.sqrt((1.0 - 3.0 * ra * ra).clamp(min=0.0))) / 3.0
    outer = (5.0 - 3.0 * ra - torch.sqrt((-3.0 * (1.0 - ra) ** 2 + 1.0).clamp(min=0.0))) / 6.0
    m_in = ra <= 0.5
    m_out = (ra > 0.5) & (ra <= 1.5)
    return torch.where(m_in, inner, torch.where(m_out, outer, torch.zeros_like(r)))


# ---------------------------------------------------------------------------
# Support build  (src/IBMNode.cpp:140-180)
# ---------------------------------------------------------------------------


def build_support_2d(
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    ny: int,
    nx: int,
) -> list[dict]:
    """Build the per-marker support (lattice nodes + diracVal), LIFE semantics.

    For each marker at (x_k, y_k) in lattice units:
      inear = round(x_k), jnear = round(y_k)
      loop i in [inear-2, inear+2], j in [jnear-2, jnear+2]
      keep node (i, j) iff |x_k - i| < 1.5 and |y_k - j| < 1.5
                     and 0 <= i <= nx-1 and 0 <= j <= ny-1   (no periodic wrap)
      diracVal = delta(|x_k - i|) * delta(|y_k - j|)

    Returns a list (one dict per marker) with int tensors ``ix``, ``iy`` and a
    float tensor ``w`` (the diracVal) of the same length.
    """
    device = marker_x.device
    dtype = marker_x.dtype
    n = marker_x.shape[0]
    supports: list[dict] = []
    for k in range(n):
        xk = float(marker_x[k].item())
        yk = float(marker_y[k].item())
        inear = int(round(xk))
        jnear = int(round(yk))
        ix_list: list[int] = []
        iy_list: list[int] = []
        w_list: list[float] = []
        for i in range(inear - 2, inear + 3):
            dx = abs(xk - i)
            if dx >= 1.5 or i < 0 or i > nx - 1:
                continue
            wx = float(life_delta_3pt(torch.tensor(dx, dtype=dtype, device=device)).item())
            if wx == 0.0:
                continue
            for j in range(jnear - 2, jnear + 3):
                dy = abs(yk - j)
                if dy >= 1.5 or j < 0 or j > ny - 1:
                    continue
                wy = float(life_delta_3pt(torch.tensor(dy, dtype=dtype, device=device)).item())
                w = wx * wy
                if w != 0.0:
                    ix_list.append(i)
                    iy_list.append(j)
                    w_list.append(w)
        supports.append(
            {
                "ix": torch.tensor(ix_list, dtype=torch.long, device=device),
                "iy": torch.tensor(iy_list, dtype=torch.long, device=device),
                "w": torch.tensor(w_list, dtype=dtype, device=device),
            }
        )
    return supports


# ---------------------------------------------------------------------------
# Marker spacing  (src/IBMNode.cpp:186-205, computeDs)
# ---------------------------------------------------------------------------


def compute_ds_2d(marker_x: torch.Tensor, marker_y: torch.Tensor) -> torch.Tensor:
    """Per-marker ds = minimum distance to any other marker (LIFE computeDs)."""
    n = marker_x.shape[0]
    xs = marker_x.tolist()
    ys = marker_y.tolist()
    ds = torch.full((n,), 10.0, dtype=marker_x.dtype, device=marker_x.device)
    for a in range(n):
        best = 10.0
        for b in range(n):
            if a == b:
                continue
            d = math.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if d < best:
                best = d
        ds[a] = torch.tensor(best, dtype=marker_x.dtype, device=marker_x.device)
    return ds


# ---------------------------------------------------------------------------
# Gram system  (src/Objects.cpp:235-320, computeEpsilon)
# ---------------------------------------------------------------------------


def compute_epsilon_gram(
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    supports: list[dict],
    ds: torch.Tensor,
) -> torch.Tensor:
    """Solve the LIFE Gram system  A * epsilon = 1  (Objects.cpp:235-320).

    A[i,j] = sum_s  w_i(s) * delta(|x_j - x_i(s)|) * delta(|y_j - y_i(s)|) * ds_j
    where (x_j, y_j) is the position of marker j and w_i(s) the diracVal of
    support s of marker i.  Returns epsilon (shape (n,)).
    """
    n = len(supports)
    dtype = ds.dtype
    device = ds.device
    xs = marker_x.tolist()
    ys = marker_y.tolist()
    A = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        si = supports[i]
        for j in range(n):
            xj = xs[j]
            yj = ys[j]
            acc = 0.0
            for s in range(si["w"].shape[0]):
                xi_s = float(si["ix"][s].item())
                yj_s = float(si["iy"][s].item())
                wx = float(life_delta_3pt(torch.tensor(abs(xj - xi_s), dtype=dtype, device=device)).item())
                wy = float(life_delta_3pt(torch.tensor(abs(yj - yj_s), dtype=dtype, device=device)).item())
                acc += float(si["w"][s].item()) * wx * wy * float(ds[j].item())
            A[i, j] = torch.tensor(acc, dtype=dtype, device=device)
    b = torch.ones(n, dtype=dtype, device=device)
    return torch.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Vectorised (sparse-matrix) form — same semantics, for time-loop use
# ---------------------------------------------------------------------------


def build_spread_matrix_2d(
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, list[dict]]:
    """Return (S, supports) with S[k, node] = w_k(s) (node = j*nx + i).

    S^T maps marker forces (scaled by eps*ds) to the lattice; S maps a lattice
    field back to the markers.  Identical weights to the loop form.
    """
    supports = build_support_2d(marker_x, marker_y, ny, nx)
    n = len(supports)
    n_nodes = ny * nx
    S = torch.zeros((n, n_nodes), dtype=marker_x.dtype, device=marker_x.device)
    for k in range(n):
        sk = supports[k]
        node = sk["iy"] * nx + sk["ix"]
        S[k, node] = sk["w"]
    return S, supports


def spread_force_2d_vec(
    fx_marker: torch.Tensor,
    fy_marker: torch.Tensor,
    S: torch.Tensor,
    epsilon: torch.Tensor,
    ds: torch.Tensor,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised spread: F_grid = S^T (f * eps * ds), reshaped to (ny, nx)."""
    scale = (epsilon * ds).unsqueeze(1)
    f = torch.stack([fx_marker, fy_marker], dim=1) * scale
    fx_grid = (S.t() @ f[:, 0]).view(ny, nx)
    fy_grid = (S.t() @ f[:, 1]).view(ny, nx)
    return fx_grid, fy_grid


def interpolate_force_2d_vec(
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised interpolate: F_marker = S @ F_grid."""
    fx_m = S @ fx_grid.reshape(-1)
    fy_m = S @ fy_grid.reshape(-1)
    return fx_m, fy_m


# ---------------------------------------------------------------------------
# Spread / interpolate  (src/IBMNode.cpp:73-74 spread; updateMacroscopic interp)
# ---------------------------------------------------------------------------


def spread_force_2d(
    fx_marker: torch.Tensor,
    fy_marker: torch.Tensor,
    supports: list[dict],
    epsilon: torch.Tensor,
    ds: torch.Tensor,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread marker forces to the lattice: F += f * eps * ds * diracVal."""
    dtype = fx_marker.dtype
    device = fx_marker.device
    fx_grid = torch.zeros((ny, nx), dtype=dtype, device=device)
    fy_grid = torch.zeros((ny, nx), dtype=dtype, device=device)
    for k in range(len(supports)):
        sk = supports[k]
        scale_x = float(fx_marker[k].item()) * float(epsilon[k].item()) * float(ds[k].item())
        scale_y = float(fy_marker[k].item()) * float(epsilon[k].item()) * float(ds[k].item())
        for s in range(sk["w"].shape[0]):
            i = int(sk["ix"][s].item())
            j = int(sk["iy"][s].item())
            w = float(sk["w"][s].item())
            fx_grid[j, i] += scale_x * w
            fy_grid[j, i] += scale_y * w
    return fx_grid, fy_grid


def interpolate_force_2d(
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
    supports: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate the Eulerian force field back onto the markers.

    F_k = sum_s F(i_k(s), j_k(s)) * w_k(s)   (same kernel, symmetric weights).
    """
    n = len(supports)
    dtype = fx_grid.dtype
    device = fx_grid.device
    fx_m = torch.zeros(n, dtype=dtype, device=device)
    fy_m = torch.zeros(n, dtype=dtype, device=device)
    for k in range(n):
        sk = supports[k]
        for s in range(sk["w"].shape[0]):
            i = int(sk["ix"][s].item())
            j = int(sk["iy"][s].item())
            w = float(sk["w"][s].item())
            fx_m[k] += w * float(fx_grid[j, i].item())
            fy_m[k] += w * float(fy_grid[j, i].item())
    return fx_m, fy_m
