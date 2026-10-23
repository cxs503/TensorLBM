"""Frozen analytic references for the C1 color-gradient track.

Geometry convention (derived from the discrete lattice, NOT fitted):
  * Channel runs over ``ny`` rows; rows ``0`` and ``ny-1`` are solid
    (bounce-back) rows.  Half-way bounce-back places the no-slip planes at
    ``Y = 0.5`` and ``Y = ny - 1.5`` (one half-node inside the solid rows).
    Verified: single-phase deficits against this convention are 0.09-0.35%
    for tau in [0.7, 2.0] at ny=48/96 (probe_wallslip); tau = 0.55 shows a
    genuine extra tau-dependent slip (delta ~ 1.0-1.8 nodes) and is
    therefore excluded from walled arms (tau >= 0.7 rule).
  * Fluid rows ``j = 1 .. ny-2`` sit at ``Y_j = j - 0.5``.
  * Channel width ``W = ny - 2``.

Interface plane convention (FROZEN for G2, geometric and measurable):
  * The interface is initialized as ``tanh((j - half)/s)``, ``half = ny//2``.
  * The effective interface plane for the sharp-jump analytic is the center
    of the first blue-majority fluid row of the STEADY phi field:
    ``Y_i = j_first_blue - 0.5``; pass ``half = j_first_blue`` to
    :func:`two_layer_profile_wallplanes`.
  * Rationale: the diffuse color band with a viscosity contrast realizes an
    effective sharp jump at the 1/nu-weighted (mobility) centroid of the
    band, which for 10:1 sits ~1 node above the phi zero crossing (Yi scan,
    out/probe_10to1_v2.json + out/probe_rule128.json: this rule lands within
    0.2 node of the scan minimum; validation ny=128 relL2 = 0.56%).

Two-layer Stokes solution with per-unit-mass driving ``gx`` (equal-density
layers: dynamic-viscosity ratio = kinematic ratio ``mu_ratio = nu_lo/nu_hi``):

  u_sigma(Y) = a_sigma Y^2 + b_sigma Y (+ c_sigma)
  a_sigma    = -gx / (2 nu_sigma)
  u_lo(0) = 0,  u_hi(W) = 0
  u_lo(Y_i) = u_hi(Y_i)                    (velocity continuity)
  mu_lo u_lo'(Y_i) = mu_hi u_hi'(Y_i)      (stress continuity)

Solving the 2x2 system for (b_lo, b_hi):
  [mu_ratio,  -1     ] [b_lo]   [2 Y_i (a_hi - mu_ratio a_lo)          ]
  [Y_i,        W - Y_i] [b_hi] = [a_hi (Y_i - W)(Y_i + W) - a_lo Y_i^2 ]
  c_hi = -a_hi W^2 - b_hi W

All functions return profiles indexed by fluid row ``j = 1 .. ny-2``
(evaluation at ``Y_j``), length ``ny - 2``.
"""

from __future__ import annotations

import torch


def first_blue_row(phi_column: torch.Tensor) -> int:
    """Index of the first blue-majority fluid row (phi <= 0), scanning up
    from the first fluid row.  ``phi_column`` is the steady color field on
    one column (length ``ny`` including the two solid rows)."""
    for j in range(1, len(phi_column) - 1):
        if float(phi_column[j]) <= 0.0:
            return j
    raise RuntimeError("no blue-majority row found")


def two_layer_profile_wallplanes(
    ny: int,
    half: int,
    gx: float,
    nu_lo: float,
    nu_hi: float,
    mu_ratio: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Piecewise-parabolic two-layer profile at the fluid node planes."""
    W = float(ny - 2)
    Yi = float(half - 0.5)
    a_lo = -gx / (2.0 * nu_lo)
    a_hi = -gx / (2.0 * nu_hi)

    A00, A01 = mu_ratio, -1.0
    A10, A11 = Yi, W - Yi
    b0 = 2.0 * Yi * (a_hi - mu_ratio * a_lo)
    b1 = a_hi * (Yi - W) * (Yi + W) - a_lo * Yi * Yi
    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-20:
        raise ValueError("degenerate two-layer system")
    b_lo = (b0 * A11 - b1 * A01) / det
    b_hi = (A00 * b1 - A10 * b0) / det
    c_hi = -a_hi * W * W - b_hi * W

    j = torch.arange(1, ny - 1, dtype=dtype)
    Y = j - 0.5
    u = torch.where(
        Y <= Yi,
        a_lo * Y * Y + b_lo * Y,
        a_hi * Y * Y + b_hi * Y + c_hi,
    )
    return u


def single_phase_profile_wallplanes(
    ny: int,
    gx: float,
    nu: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Single-phase parabola ``u = gx Y (W - Y) / (2 nu)`` at fluid rows."""
    W = float(ny - 2)
    Y = torch.arange(1, ny - 1, dtype=dtype) - 0.5
    return gx * Y * (W - Y) / (2.0 * nu)


def two_layer_profile_fd(
    ny: int,
    half: float,
    gx: float,
    nu_lo: float,
    nu_hi: float,
    n_refine: int = 8,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Independent FD cross-check of :func:`two_layer_profile_wallplanes`.

    Solves ``mu u'' = -gx`` (rho = 1, equal densities so mu = nu) on a grid
    refined ``n_refine`` times, with Dirichlet walls at Y=0/W, the sharp
    viscosity jump at ``Y_i`` and face viscosities assigned by face-midpoint
    rule (``mu_face[k] = mu(y[k] + h/2)``).  Because the exact solution is
    piecewise quadratic with the jump on a node, this scheme reproduces it
    to roundoff — an independent confirmation of the closed form's
    ODE/wall/interface conditions, not of the 2x2 solver algebra alone.
    """
    W = float(ny - 2)
    Yi = float(half - 0.5)
    mu_lo, mu_hi = float(nu_lo), float(nu_hi)

    n = (ny - 2) * n_refine + 1  # refined nodes including both walls
    h = W / (n - 1)
    y_face = (torch.arange(n - 1, dtype=dtype) + 0.5) * h  # face midpoints
    mu_face = torch.where(
        y_face <= Yi,
        torch.tensor(mu_lo, dtype=dtype),
        torch.tensor(mu_hi, dtype=dtype),
    )

    lower = torch.zeros(n, dtype=dtype)  # lower[i] = A[i, i-1]
    upper = torch.zeros(n, dtype=dtype)  # upper[i] = A[i, i+1]
    diag = torch.zeros(n, dtype=dtype)
    rhs = torch.zeros(n, dtype=dtype)
    diag[0] = 1.0
    diag[-1] = 1.0
    idx = torch.arange(1, n - 1)
    lower[idx] = -mu_face[:-1]
    diag[idx] = mu_face[:-1] + mu_face[1:]
    upper[idx] = -mu_face[1:]
    rhs[idx] = gx * h * h

    # Thomas algorithm
    cp = torch.zeros(n, dtype=dtype)
    dp = torch.zeros(n, dtype=dtype)
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    u = torch.zeros(n, dtype=dtype)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    # sample at fluid planes Y_j (linear interpolation between refined nodes)
    Yj = torch.arange(1, ny - 1, dtype=dtype) - 0.5
    pos = Yj / h
    i0 = pos.floor().long().clamp(0, n - 2)
    frac = pos - i0.to(dtype)
    return u[i0] * (1 - frac) + u[i0 + 1] * frac
