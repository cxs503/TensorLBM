"""Analytical / high-accuracy reference solutions for yield-stress flow with moving walls.

Reference solutions used to close the P82W5-N04 anchors for the D3Q19
yield-stress kernel with moving walls:

1. ``bingham_channel_profile`` / ``bingham_channel_flow_rate``
   Exact plane-Poiseuille Bingham solution (plug half-width ``y_p = tau_y/G``).

2. ``bingham_couette_poiseuille_profile``
   Exact Bingham plane Couette--Poiseuille solution (one wall moving at ``U_w``
   **and** a streamwise body force ``G``).  Zero pressure gradient is *not*
   admissible for a partial plug (see 3); with a body force the plug thickness
   has the closed form ``delta_p = 2*tau_y/G`` and the plug sits between the two
   yielded layers.  ``tau_0`` (bottom-wall shear) is the only root of a
   monotone scalar equation, solved here by bisection to machine precision.

3. ``bingham_plane_couette_regime``
   Pure Bingham plane Couette (one wall moving, **no** body force) is
   degenerate: ``d(tau)/dy = 0`` forces a *uniform* shear stress, so either the
   whole gap yields (linear profile, no plug) or the material cannot deform at
   all (no admissible steady solution for ``U_w != 0``).  A partial plug is
   inadmissible, i.e. the previously published formula
   ``delta_p = H - mu_0*U_w/tau_y`` (see ``yield_stress_analytical_3d``) is not
   a solution of the Couette problem.  This function classifies the regime so
   callers can report it explicitly instead of quoting a spurious plug.

4. ``newtonian_square_duct_series``
   Accurate Newtonian square-duct (Fourier series, ~1e-12 relative).

5. ``bingham_square_duct_reference``
   Regularisation-free Bingham square-duct solution on a refined grid by the
   augmented Lagrangian (Glowinski) algorithm -- the reference for the
   regularised LBM.  Node positions coincide exactly with the LBM cell centres
   for any even ``refine`` (walls at ``y=0`` and ``y=N``, cell centre of fluid
   cell ``i`` at ``i-0.5``).

All solutions are in lattice units (``rho = 1``, ``c_s^2 = 1/3``), i.e. the body
force appears as an acceleration ``G = rho*a`` with ``rho = 1``.

References
----------
* Bird, Stewart & Lightfoot (2002) *Transport Phenomena*, 2nd ed., §5.
* Chhabra & Richardson (2008) *Non-Newtonian Flow and Applied Rheology*, 2nd ed.
* Glowinski & Wachsmuth (2011) On the finite element approximation of the
  Bingham fluid flow problem, *Comput. Methods Appl. Math.*
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "bingham_channel_profile",
    "bingham_channel_flow_rate",
    "bingham_plug_half_width",
    "bingham_couette_poiseuille_profile",
    "bingham_couette_poiseuille_plug",
    "bingham_plane_couette_regime",
    "newtonian_square_duct_series",
    "bingham_square_duct_reference",
    "bingham_rectangular_duct_reference",
    "bingham_square_duct_flow_rate",
    "rectangular_duct_shape_factor",
    "bingham_2d_3d_flow_ratio",
]


# ---------------------------------------------------------------------------
# 1. plane Poiseuille (pressure driven, both walls at rest)
# ---------------------------------------------------------------------------
def bingham_plug_half_width(G: float, tau_y: float, half_gap: float) -> float:
    """Plug half-width ``y_p = tau_y/G`` clamped to the half gap."""
    if G <= 0.0 or tau_y <= 0.0:
        return 0.0
    return min(tau_y / G, half_gap)


def bingham_channel_profile(
    H: float,
    G: float,
    tau_y: float,
    mu_0: float,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Exact Bingham plane-Poiseuille velocity profile.

    Walls at ``y = 0`` and ``y = H`` (half-way bounce-back convention).

        u(s) = (G/(2 mu_0)) (h^2 - s^2) - (tau_y/mu_0) (h - s),  s > y_p
        u(s) = u_p = (G/(2 mu_0)) (h - y_p)^2,                  s <= y_p

    with ``h = H/2``, ``s = |y - H/2|`` and ``y_p = tau_y/G``.

    Args:
        H: channel width (wall to wall).
        G: body force acceleration (rho = 1).
        tau_y: yield stress.
        mu_0: plastic viscosity.
        y: sample positions (default: LBM cell centres ``0.5 .. H-0.5``).

    Returns:
        Velocity array evaluated at ``y``.
    """
    H = float(H)
    h = 0.5 * H
    if y is None:
        y = np.arange(1, int(round(H)) + 1, dtype=float) - 0.5
    y = np.asarray(y, dtype=float)
    s = np.abs(y - 0.5 * H)

    y_p = bingham_plug_half_width(G, tau_y, h)
    u_p = G / (2.0 * mu_0) * (h - y_p) ** 2
    u_shear = (G / (2.0 * mu_0)) * (h**2 - s**2) - (tau_y / mu_0) * (h - s)
    return np.where(s <= y_p, u_p, u_shear)


def bingham_channel_flow_rate(H: float, G: float, tau_y: float, mu_0: float) -> float:
    """Exact Bingham plane-Poiseuille flow rate per unit depth.

    ``Q = 2 * [u_p*y_p + int_{y_p}^{h} u ds]``; reduces to ``G H^3/(12 mu_0)``
    for ``tau_y -> 0``.
    """
    H = float(H)
    h = 0.5 * H
    y_p = bingham_plug_half_width(G, tau_y, h)
    u_p = G / (2.0 * mu_0) * (h - y_p) ** 2
    int_shear = (G / (2.0 * mu_0)) * (
        h**2 * (h - y_p) - (h**3 - y_p**3) / 3.0
    ) - (tau_y / mu_0) * 0.5 * (h - y_p) ** 2
    return 2.0 * (u_p * y_p + int_shear)


# ---------------------------------------------------------------------------
# 2. Couette--Poiseuille (one wall moving + body force): exact
# ---------------------------------------------------------------------------
def _couette_poiseuille_u_at_wall(H: float, G: float, tau_y: float,
                                  mu_0: float, tau_0: float) -> float:
    """Top-wall velocity ``u(H)`` produced by a bottom-wall shear ``tau_0``.

    Bingham law in the yielded layers: ``tau = tau_y + mu_0*du/dy`` (lower, tau>0)
    and ``tau = -tau_y + mu_0*du/dy`` (upper, tau<0), i.e.
    ``du/dy = (tau -+ tau_y)/mu_0``.  In the plug ``|tau| <= tau_y`` and
    ``du/dy = 0``.
    """
    tau_1 = tau_0 - G * H
    if tau_y <= 0.0:                                   # Newtonian
        return (tau_0 * H - 0.5 * G * H * H) / mu_0

    if tau_0 > tau_y and tau_1 < -tau_y:               # two yielded layers + plug
        y1 = (tau_0 - tau_y) / G
        y2 = (tau_0 + tau_y) / G
        u_p = (tau_0 - tau_y) ** 2 / (2.0 * mu_0 * G)
        top = ((tau_0 + tau_y) * (H - y2) - 0.5 * G * (H * H - y2 * y2)) / mu_0
        return u_p + top

    if tau_0 > tau_y:                                  # plug attached to top wall
        return (tau_0 - tau_y) ** 2 / (2.0 * mu_0 * G)

    # plug attached to the (stationary) bottom wall; yielded layer on top
    if tau_1 < -tau_y:
        y2 = (tau_0 + tau_y) / G
        return ((tau_0 + tau_y) * (H - y2) - 0.5 * G * (H * H - y2 * y2)) / mu_0

    return 0.0                                          # fully unyielded


def bingham_couette_poiseuille_plug(H: float, G: float, tau_y: float,
                                    mu_0: float, u_wall: float) -> dict:
    """Exact Bingham Couette--Poiseuille: bottom-wall shear and plug bounds.

    Returns a dict with ``tau_0`` (bottom-wall shear), ``y1`` (lower yield
    surface), ``y2`` (upper yield surface), ``u_plug`` and ``regime`` in
    ``{"plug", "plug_top", "plug_bottom", "yielded", "unyielded"}``.
    """
    H = float(H)
    if tau_y <= 0.0:
        tau_0 = (u_wall * mu_0 + 0.5 * G * H * H) / H
        return {"tau_0": tau_0, "y1": 0.0, "y2": 0.0, "u_plug": 0.0,
                "delta_p": 0.0, "regime": "yielded"}

    lo = -tau_y
    hi = tau_y + G * H + 4.0 * tau_y
    f_lo = _couette_poiseuille_u_at_wall(H, G, tau_y, mu_0, lo) - u_wall
    f_hi = _couette_poiseuille_u_at_wall(H, G, tau_y, mu_0, hi) - u_wall
    if f_lo > 0.0:            # not reachable with the bracket: clamp
        tau_0 = lo
    elif f_hi < 0.0:
        tau_0 = hi
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = _couette_poiseuille_u_at_wall(H, G, tau_y, mu_0, mid) - u_wall
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-15 * max(1.0, abs(hi)):
                break
        tau_0 = 0.5 * (lo + hi)

    tau_1 = tau_0 - G * H
    out = {"tau_0": tau_0, "y1": 0.0, "y2": 0.0, "u_plug": 0.0,
           "delta_p": 0.0, "regime": "unyielded", "tau_1": tau_1}
    if tau_0 > tau_y and tau_1 < -tau_y:
        y1 = (tau_0 - tau_y) / G
        y2 = (tau_0 + tau_y) / G
        out.update(y1=y1, y2=y2, delta_p=y2 - y1, regime="plug",
                   u_plug=(tau_0 - tau_y) ** 2 / (2.0 * mu_0 * G))
    elif tau_0 > tau_y:
        y1 = (tau_0 - tau_y) / G
        out.update(y1=y1, y2=H, delta_p=H - y1, regime="plug_top",
                   u_plug=(tau_0 - tau_y) ** 2 / (2.0 * mu_0 * G))
    elif tau_1 < -tau_y:
        y2 = (tau_0 + tau_y) / G
        out.update(y1=0.0, y2=y2, delta_p=y2, regime="plug_bottom",
                   u_plug=0.0)
    elif tau_y <= 0.0:
        out["regime"] = "yielded"
    return out


def bingham_couette_poiseuille_profile(
    H: float,
    G: float,
    tau_y: float,
    mu_0: float,
    u_wall: float,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Exact Bingham Couette--Poiseuille profile (one wall moving + body force)."""
    H = float(H)
    if y is None:
        y = np.arange(1, int(round(H)) + 1, dtype=float) - 0.5
    y = np.asarray(y, dtype=float)
    plug = bingham_couette_poiseuille_plug(H, G, tau_y, mu_0, u_wall)
    tau_0 = plug["tau_0"]
    y1, y2 = plug["y1"], plug["y2"]

    if tau_y <= 0.0:
        return (tau_0 * y - 0.5 * G * y * y) / mu_0

    # Bingham law: u' = (tau - tau_y)/mu_0 in the lower yielded layer and
    # u' = (tau + tau_y)/mu_0 in the upper one (tau < -tau_y there)
    u_low = ((tau_0 - tau_y) * y - 0.5 * G * y * y) / mu_0
    u_p = plug["u_plug"]
    u_up = u_p + ((tau_0 + tau_y) * (y - y2)
                  - 0.5 * G * (y * y - y2 * y2)) / mu_0
    u = np.where(y <= y1, u_low, np.where(y >= y2, u_up, u_p))
    # regimes where one of the layers is absent
    if plug["regime"] == "plug_top":
        u = np.where(y <= y1, u_low, u_p)
    elif plug["regime"] == "plug_bottom":
        u = np.where(y >= y2, u_up, 0.0)
    elif plug["regime"] == "unyielded":
        u = np.zeros_like(y)
    return u


def bingham_plane_couette_regime(H: float, tau_y: float, mu_0: float,
                                 u_wall: float) -> dict:
    """Classify pure Bingham plane Couette (one wall moving, no body force).

    Zero pressure gradient forces ``tau = const``.  Hence

    * ``|tau_w| = mu_0*|U_w|/H >= tau_y`` -> the whole gap yields, the profile is
      linear, the plug thickness is exactly ``0``;
    * ``|tau_w| < tau_y`` -> the material cannot deform, but the no-slip
      conditions (u=0 at the fixed wall, u=U_w at the moving wall) cannot both
      be met by a rigid body: **no steady solution exists**.

    A partial plug (``0 < delta_p < H``) is therefore *inadmissible* for the pure
    Couette problem, contrary to ``delta_p = H - mu_0*U_w/tau_y``.
    """
    H = float(H)
    tau_w = mu_0 * abs(u_wall) / H
    yielded = tau_y <= 0.0 or tau_w >= tau_y
    return {
        "tau_wall": tau_w,
        "yield_ratio": (tau_y / tau_w) if tau_w > 0 else float("inf"),
        "regime": "fully_yielded" if yielded else "no_steady_solution",
        "delta_p": 0.0 if yielded else float("nan"),
        "profile": "linear", 
    }


# ---------------------------------------------------------------------------
# 4. Newtonian square duct (Fourier series)
# ---------------------------------------------------------------------------
def newtonian_square_duct_series(N: int, G: float, mu_0: float,
                                 n_terms: int = 400) -> np.ndarray:
    """Newtonian square-duct velocity profile (Fourier series, cell centres).

    ``u(y,z) = sum_{m,n odd} 16 G / (mu_0 pi^4 m n (m^2/N^2 + n^2/N^2))
       sin(m pi y / N) sin(n pi z / N)``  with walls at ``0`` and ``N``.

    Returns an ``(N, N)`` array indexed ``[iy, iz]`` with the cell centre
    coordinates ``y = iy+0.5``, ``z = iz+0.5``.
    """
    N = int(N)
    y = np.arange(1, N + 1) - 0.5
    z = np.arange(1, N + 1) - 0.5
    Y, Z = np.meshgrid(y / N, z / N, indexing="ij")
    u = np.zeros_like(Y)
    for m in range(1, 2 * n_terms, 2):
        sm = np.sin(m * math.pi * Y) / m
        for n in range(1, 2 * n_terms, 2):
            u += sm * (np.sin(n * math.pi * Z) / n) / (m * m + n * n)
    return u * (16.0 * G * N * N / (mu_0 * math.pi**4))


def rectangular_duct_shape_factor(aspect_ratio: float = 1.0) -> float:
    """Flow-rate shape factor ``Q_rect / (G H^4 / (12 mu))`` for a rectangle."""
    alpha = max(float(aspect_ratio), 1e-6)
    s = sum(math.tanh((2 * k + 1) * math.pi / (2.0 * alpha)) / (2 * k + 1) ** 5
            for k in range(80))
    return 1.0 - 192.0 / math.pi**5 * alpha * s


# ---------------------------------------------------------------------------
# 5. Bingham square duct: regularisation-free reference (augmented Lagrangian)
# ---------------------------------------------------------------------------
def _sine_basis(n: int) -> tuple[np.ndarray, np.ndarray]:
    """DST-I basis ``S[j-1,k-1] = sin(pi*j*k/n)`` (j,k=1..n-1) and eigenvalues."""
    j = np.arange(1, n)
    S = np.sin(np.pi * np.outer(j, j) / n)
    lam = 2.0 - 2.0 * np.cos(np.pi * j / n)          # divided by h^2 by caller
    return S, lam


def _solve_dirichlet_sine(Sy: np.ndarray, lamy: np.ndarray,
                          Sz: np.ndarray, lamz: np.ndarray,
                          R: np.ndarray) -> np.ndarray:
    """Solve ``-Laplace u = R`` on a box with ``u = 0`` on all four faces.

    ``u`` lives on the interior nodes ``1..n-1`` of each direction; the transform
    is exact (it is the eigenbasis of the discrete 5-point Dirichlet Laplacian).
    """
    ny, nz = Sy.shape[0] + 1, Sz.shape[0] + 1
    Rh = Sy.T @ R @ Sz
    U = (Sy @ (Rh / (lamy[:, None] + lamz[None, :])) @ Sz.T) * (2.0 / ny) * (2.0 / nz)
    return U


def _al_bingham_duct(
    ny_cells: int,
    nz_cells: int,
    G: float,
    tau_y: float,
    mu_0: float,
    *,
    refine: int = 8,
    n_iter: int = 4000,
    r_factor: float = 5.0,
    tol: float = 1e-12,
) -> dict:
    """Augmented-Lagrangian Bingham duct solve on a refined grid (regularisation free).

    Minimises ``J(v) = mu_0/2 |grad v|^2 + tau_y |grad v| - G v`` with ``v = 0`` on
    the four walls of the rectangle ``[0, ny_cells] x [0, nz_cells]``.  The
    ``u``-update is a constant-coefficient Poisson solve (exact sine transform);
    the plastic multiplier and the strain-rate field are updated by the standard
    Glowinski projection step.
    """
    ny_cells = int(ny_cells)
    nz_cells = int(nz_cells)
    ny = int(refine) * ny_cells
    nz = int(refine) * nz_cells
    h = 1.0 / float(refine)
    Sy, lamy = _sine_basis(ny)
    Sz, lamz = _sine_basis(nz)
    lamy = lamy / (h * h)
    lamz = lamz / (h * h)

    def grad(u):
        gx = np.empty_like(u)
        gx[:, :-1] = (u[:, 1:] - u[:, :-1]) / h
        gx[:, -1] = -u[:, -1] / h          # face against the east wall (u = 0)
        gy = np.empty_like(u)
        gy[:-1, :] = (u[1:, :] - u[:-1, :]) / h
        gy[-1, :] = -u[-1, :] / h          # face against the north wall (u = 0)
        return gx, gy

    def div(vx, vy, u):
        """div(grad u) is the exact 5-point Dirichlet Laplacian."""
        d = np.empty_like(vx)
        d[:, 0] = (vx[:, 0] - u[:, 0] / h) / h          # west wall face gradient
        d[:, 1:] = (vx[:, 1:] - vx[:, :-1]) / h
        d[0, :] += (vy[0, :] - u[0, :] / h) / h         # south wall face gradient
        d[1:, :] += (vy[1:, :] - vy[:-1, :]) / h
        return d

    r = r_factor * mu_0
    u = np.zeros((ny - 1, nz - 1))
    gx, gy = grad(u)
    lx = np.zeros_like(u)
    ly = np.zeros_like(u)
    res = np.inf
    it = 0
    for it in range(1, n_iter + 1):
        u_old = u
        rhs = G * np.ones_like(u) - r * div(gx, gy, u) + div(lx, ly, u * 0.0)
        u = _solve_dirichlet_sine(Sy, lamy, Sz, lamz, rhs) / (mu_0 + r)
        gx_new, gy_new = grad(u)
        sx = gx_new + lx / r
        sy = gy_new + ly / r
        s_abs = np.sqrt(sx * sx + sy * sy) + 1e-300
        shrink = np.maximum(0.0, 1.0 - tau_y / (r * s_abs))
        gx = shrink * sx
        gy = shrink * sy
        lx = lx + r * (gx_new - gx)
        ly = ly + r * (gy_new - gy)
        scale = max(float(np.max(np.abs(u))), 1e-300)
        du = float(np.max(np.abs(u - u_old))) / scale
        res = float(np.max(np.abs(gx_new - gx)))
        if du < tol:
            break

    u_fine = np.zeros((ny + 1, nz + 1))
    u_fine[1:-1, 1:-1] = u
    jy = (refine * (np.arange(1, ny_cells + 1) - 0.5)).astype(int)
    jz = (refine * (np.arange(1, nz_cells + 1) - 0.5)).astype(int)
    u_c = u_fine[np.ix_(jy, jz)]
    return {"u": u_c, "u_fine": u_fine, "iters": it, "al_residual": res,
            "n_fine": (ny, nz), "refine": refine, "Q": float(u_c.sum())}


def bingham_square_duct_reference(N: int, G: float, tau_y: float, mu_0: float,
                                  *, refine: int = 8, **kw) -> dict:
    """Regularisation-free Bingham square-duct solution (AL, refined grid).

    Node positions coincide exactly with the LBM cell centres for even
    ``refine`` (walls at ``0`` and ``N``, cell centre ``i`` at ``i-0.5``).
    Returns a dict with ``u`` (``N x N``, ``[iy, iz]``), ``Q`` (flow rate),
    ``iters`` and the AL residual.
    """
    return _al_bingham_duct(int(N), int(N), G, tau_y, mu_0, refine=refine, **kw)


def bingham_rectangular_duct_reference(ny: int, nz: int, G: float, tau_y: float,
                                       mu_0: float, *, refine: int = 8,
                                       **kw) -> dict:
    """Bingham flow in a rectangular duct ``ny x nz`` (AL reference)."""
    return _al_bingham_duct(ny, nz, G, tau_y, mu_0, refine=refine, **kw)


def bingham_square_duct_flow_rate(N: int, G: float, tau_y: float, mu_0: float,
                                  *, refine: int = 8) -> float:
    """Flow rate of the Bingham square duct (fine-grid AL reference)."""
    return bingham_square_duct_reference(N, G, tau_y, mu_0, refine=refine)["Q"]


def bingham_2d_3d_flow_ratio(N: float, G: float, tau_y: float, mu_0: float,
                             *, refine: int = 8) -> dict:
    """Ratio ``Q_2D / Q_3D_square`` at equal gap, body force and rheology.

    ``Q_2D`` is the exact plane-Poiseuille Bingham flow rate per unit depth,
    ``Q_3D`` the Bingham square-duct flow rate (AL reference).  The Newtonian
    limit of the ratio is ``1/f`` with ``f = rectangular_duct_shape_factor(1)``.
    """
    q2d = bingham_channel_flow_rate(N, G, tau_y, mu_0)
    q3d = bingham_square_duct_flow_rate(int(N), G, tau_y, mu_0, refine=refine)
    f = rectangular_duct_shape_factor(1.0)
    return {
        "Q_2D": q2d,
        "Q_3D": q3d,
        "ratio": q2d / q3d if q3d else float("inf"),
        "newtonian_ratio": 1.0 / f,
        "shape_factor": f,
    }