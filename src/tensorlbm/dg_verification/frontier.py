"""IN65-N04 -- DG convection/diffusion verification *frontier*.

The DG verification face built by IN70-N01 established the ladder (rotating cone,
manufactured solution, Sod tube, square wave, mass ledger) with a Rusanov
interface flux and left four explicit open issues.  This module is the next
rung: it attacks those issues with new machinery rather than re-running the
same cases.

What is new here (see ``docs/IN65_N04_dg_frontier.md``)
-------------------------------------------------------
1. :class:`EulerDG1D_HLLC` -- an Euler DG solver whose spatial operator is
   *pure* (no limiter inside the RHS), with a switchable interface flux
   (:func:`..hllc.hllc_flux` HLLC vs Rusanov) and the Zhang-Shu
   positivity-preserving limiter at stage boundaries.  Rusanov has a 146 %
   mass-flux error across the Sod contact while HLLC is exact there
   (``tests/test_dg_frontier_in65.py::test_hllc_contact_exact``) -- that
   difference is what the previous node's ideal-gas Sod run was fighting.
2. **Kink-limited order theory.**  The LeVeque cone is only ``C^0``: its first
   derivative jumps on the cone surface, so the initial data is in
   ``H^{3/2-eps}`` and *no* DG degree can beat ``O(h^{1.5})`` in L2 on that
   problem.  :func:`bench_kink_order` measures exactly that scaling for a 1-D
   tent function, and :func:`bench_rotating` splits the 2-D error into a kink
   band and the smooth rest, so the cap is diagnosed instead of asserted.
3. **Dissipation/dispersion metrics** (:func:`bench_square_wave_metrics`): total
   variation decay, overshoot, L1 error against the exactly translated profile,
   plus the amplitude (dissipation) and phase (dispersion) of a single Fourier
   mode over one period, limiter on/off.
4. A **variable-coefficient manufactured solution** (:func:`bench_mms_varcoeff`)
   that grades the *rotating* velocity path together with the LDG gradient
   ``q``, so the advection operator is verified independently of the (kinked)
   cone initial data.
5. A **DG <-> LBM <-> analytic cross-examination** of the isothermal Sod tube
   (:func:`bench_sod_lbm_crosscheck`) against the verified D2Q9 profiles in
   ``benchmarks/verified/sod_shock_tube``.

Every benchmark returns plain-``dict``/``float`` data, so the JSON artefacts and
the report figures are produced from one source of truth (zero hand-copied
numbers).
"""

from __future__ import annotations

import math
import time

import numpy as np

from .benchmarks import (
    cone_initial,
    exact_isothermal_riemann,
    exact_sod,
    fit_order,
    gaussian_initial,
    shock_position,
)
from .element import RefElement, ref_element
from .hllc import AdmissibleError, hllc_flux, rusanov_flux, zhang_shu_limit
from .limiter import tvb_limit
from .scalar import (
    advection_rhs_2d,
    apply_axis,
    gauss_coords_2d,
    mass_ratio_drift,
    node_coords_2d,
    ssp_rk3,
)

__all__ = [
    "ScalarDG1D",
    "EulerDG1D_HLLC",
    "bench_rotating",
    "bench_kink_order",
    "bench_mms_varcoeff",
    "bench_sod_hllc",
    "bench_sod_lbm_crosscheck",
    "bench_square_wave_metrics",
    "bench_euler_mass_ledger",
    "contact_position",
    "contact_width",
    "total_variation",
    "mode_amplitude_phase",
    "hump_initial",
    "tent_initial",
    "narrow_gaussian_initial",
]


# ---------------------------------------------------------------------------
# 1-D scalar DG (advection) -- small, pure operator
# ---------------------------------------------------------------------------


class ScalarDG1D:
    """Nodal DG for ``u_t + a u_x = 0`` on a periodic 1-D grid (float64).

    Deliberately minimal: one velocity, one upwind flux (exact for
    constant-coefficient advection), and an optional TVB limiter applied once
    per *complete* step -- limiting inside the RK stages destroys the formal
    order of the time integrator (the previous Euler solver did it inside
    ``rhs``).
    """

    def __init__(
        self,
        degree: int = 1,
        n_cells: int = 128,
        length: float = 1.0,
        velocity: float = 1.0,
        use_limiter: bool = False,
        tvb_M: float = 0.0,
    ) -> None:
        self.degree = int(degree)
        self.elem: RefElement = ref_element(degree)
        self.n_cells = int(n_cells)
        self.length = float(length)
        self.dx = self.length / self.n_cells
        self.a = float(velocity)
        self.use_limiter = bool(use_limiter)
        self.tvb_M = float(tvb_M)
        xc = (np.arange(self.n_cells) + 0.5) * self.dx
        self.x = xc[:, None] + 0.5 * self.dx * self.elem.nodes[None, :]

    def rhs(self, u: np.ndarray) -> np.ndarray:
        f = self.a * u
        dvol = apply_axis(f, self.elem.Gr, -1)
        u_l = u[..., -1]  # right trace of cell i
        u_r = np.roll(u[..., 0], -1, axis=-1)  # left trace of cell i+1
        up = self.a >= 0.0
        f_face = np.where(up, self.a * u_l, self.a * u_r)
        f_left = np.roll(f_face, 1, axis=-1)
        surf = f_left[..., None] * self.elem.S[:, 0] + f_face[..., None] * self.elem.S[:, 1]
        return apply_axis(dvol - surf, self.elem.Minv, -1) * (2.0 / self.dx)

    def step(self, dt: float) -> None:
        u = ssp_rk3(self._u, dt, self.rhs)
        if self.use_limiter:
            u = tvb_limit(u, self.elem, self.dx, self.tvb_M)
        self._u = u

    def cfl_dt(self, cfl: float = 0.3) -> float:
        return cfl * self.dx / ((2 * self.degree + 1) * max(abs(self.a), 1e-12))

    def set_state(self, u: np.ndarray) -> None:
        self._u = np.array(u, dtype=np.float64, copy=True)

    @property
    def u(self) -> np.ndarray:
        return self._u

    def advance(self, t_end: float, cfl: float = 0.3, max_steps: int = 5_000_000) -> dict:
        t = 0.0
        steps = 0
        while t < t_end - 1e-15 and steps < max_steps:
            dt = min(self.cfl_dt(cfl), t_end - t)
            self.step(dt)
            t += dt
            steps += 1
        return {"t": t, "steps": steps}

    def cell_means(self) -> np.ndarray:
        return self._u.mean(axis=-1)

    def cell_centers(self) -> np.ndarray:
        return (np.arange(self.n_cells) + 0.5) * self.dx


# ---------------------------------------------------------------------------
# 1-D Euler DG with HLLC / Rusanov and Zhang-Shu positivity limiter
# ---------------------------------------------------------------------------


class EulerDG1D_HLLC:
    """Nodal DG for the 1-D Euler equations (ideal gas or isothermal closure).

    Differences to :class:`..dg_verification.euler1d.EulerDG1D` (which is kept
    untouched for regression comparison):

    * the interface flux is selectable -- ``"hllc"`` (contact-resolving) or
      ``"rusanov"`` (the previous default);
    * :meth:`rhs` is a *pure* spatial operator (no limiting inside it), so the
      SSP-RK3 stages see a consistent ODE and manufactured-solution order
      verification is meaningful;
    * positivity is enforced by the Zhang-Shu scaling limiter
      (:func:`..hllc.zhang_shu_limit`), which is conservative and keeps the
      shape of the element instead of replacing it by its mean.
    """

    def __init__(
        self,
        degree: int = 1,
        n_cells: int = 200,
        length: float = 1.0,
        gamma: float = 1.4,
        isothermal: bool = False,
        cs2: float = 1.0 / 3.0,
        flux: str = "hllc",
        wave_speed: str = "toro",
        use_limiter: bool = True,
        tvb_M: float = 0.0,
        positivity: bool = True,
        cfl: float = 0.3,
        bc: str = "periodic",
    ) -> None:
        self.degree = int(degree)
        self.elem: RefElement = ref_element(degree)
        self.n_cells = int(n_cells)
        self.length = float(length)
        self.dx = self.length / self.n_cells
        self.gamma = float(gamma)
        self.isothermal = bool(isothermal)
        self.cs2 = float(cs2)
        self.flux_kind = str(flux)
        self.wave_speed_estimate = str(wave_speed)
        self.use_limiter = bool(use_limiter)
        self.tvb_M = float(tvb_M)
        self.positivity = bool(positivity)
        self.cfl_default = float(cfl)
        self.bc = str(bc)
        self.n_comp = 2 if isothermal else 3
        self._gamma_eff = None if isothermal else self.gamma
        self.stats = {"positivity_calls": 0, "theta_min": 1.0, "steps": 0}
        xc = (np.arange(self.n_cells) + 0.5) * self.dx
        self.x = xc[:, None] + 0.5 * self.dx * self.elem.nodes[None, :]

    # -- state -------------------------------------------------------------
    def from_primitive(self, rho, u, p) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        U = np.zeros((self.n_comp,) + rho.shape, dtype=np.float64)
        U[0] = rho
        U[1] = rho * u
        if not self.isothermal:
            U[2] = p / (self.gamma - 1.0) + 0.5 * rho * u * u
        return U

    def primitive(self, U: np.ndarray) -> tuple:
        rho = U[0]
        u = U[1] / np.maximum(rho, 1e-300)
        if self.isothermal:
            p = self.cs2 * rho
        else:
            p = (self.gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
        return rho, u, p

    def wave_speed(self, U: np.ndarray) -> np.ndarray:
        rho, u, p = self.primitive(U)
        if self.isothermal:
            c = np.full_like(rho, math.sqrt(self.cs2))
        else:
            c = np.sqrt(self.gamma * np.maximum(p, 1e-300) / np.maximum(rho, 1e-300))
        return np.abs(u) + c

    def physical_flux(self, U: np.ndarray) -> np.ndarray:
        rho, u, p = self.primitive(U)
        F = np.zeros_like(U)
        F[0] = U[1]
        F[1] = U[1] * u + p
        if not self.isothermal:
            F[2] = u * (U[2] + p)
        return F

    def numerical_flux(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        if self.flux_kind == "hllc":
            return hllc_flux(UL, UR, gamma=self._gamma_eff, cs2=self.cs2,
                             wave_speed=self.wave_speed_estimate)
        if self.flux_kind == "rusanov":
            return rusanov_flux(UL, UR, gamma=self._gamma_eff, cs2=self.cs2)
        raise ValueError(f"unknown flux {self.flux_kind!r}")

    # -- operator ----------------------------------------------------------
    def rhs(self, U: np.ndarray) -> np.ndarray:
        """Pure spatial operator (no limiter, no positivity repair)."""
        F = self.physical_flux(U)
        dvol = apply_axis(F, self.elem.Gr, -1)
        if self.bc not in ("periodic", "outflow"):
            raise ValueError(f"unknown bc {self.bc!r}")
        UL = U[..., -1]  # right trace of cell i
        UR = np.roll(U[..., 0], -1, axis=-1)  # left trace of cell i+1
        F_face = self.numerical_flux(UL, UR)  # (comp, N): face i = i+1/2
        F_left = np.roll(F_face, 1, axis=-1)  # face i-1/2, periodic wrap
        if self.bc == "outflow":
            # Zero-gradient (transmissive) ends.  On a periodic array the wrap
            # face N-1 plays *both* outer faces, so it cannot carry a single
            # flux: the two boundary fluxes are overwritten independently.
            # Ghost state == interior trace  =>  flux == physical flux.
            Fl = self.physical_flux(U[..., 0][..., :1])  # left face of cell 0
            Fr = self.physical_flux(U[..., -1][..., -1:])  # right face of cell N-1
            F_left = np.concatenate([Fl.reshape(-1, 1), F_left[..., 1:]], axis=-1)
            F_face = np.concatenate([F_face[..., :-1], Fr.reshape(-1, 1)], axis=-1)
        surf = F_left[..., None] * self.elem.S[:, 0] + F_face[..., None] * self.elem.S[:, 1]
        return apply_axis(dvol - surf, self.elem.Minv, -1) * (2.0 / self.dx)

    # -- limiters ----------------------------------------------------------
    def limit(self, U: np.ndarray) -> np.ndarray:
        if not self.use_limiter:
            return U
        out = np.empty_like(U)
        for c in range(self.n_comp):
            out[c] = tvb_limit(U[c], self.elem, self.dx, self.tvb_M, axis=-2)
        return out

    def enforce_positivity(self, U: np.ndarray) -> np.ndarray:
        if not self.positivity:
            return U
        self.stats["positivity_calls"] += 1
        try:
            out, theta = zhang_shu_limit(U, gamma=self._gamma_eff, cs2=self.cs2)
        except AdmissibleError:
            # The cell *average* itself is inadmissible, which can only happen if
            # the CFL bound was violated.  Fall back to the mean state (last
            # resort, recorded in the stats).
            Ubar = U.mean(axis=-1, keepdims=True)
            out = np.ones_like(U) * Ubar
            theta = np.zeros(U.shape[1])
        self.stats["theta_min"] = min(self.stats["theta_min"], float(theta.min()))
        return out

    # -- time integration --------------------------------------------------
    def step(self, dt: float) -> None:
        U0 = self._U
        k1 = self.enforce_positivity(U0 + dt * self.rhs(U0))
        k2 = self.enforce_positivity(0.75 * U0 + 0.25 * (k1 + dt * self.rhs(k1)))
        Un = self.enforce_positivity((1.0 / 3.0) * U0 + (2.0 / 3.0) * (k2 + dt * self.rhs(k2)))
        self._U = self.limit(Un)
        self.stats["steps"] += 1

    def cfl_dt(self, cfl: float | None = None) -> float:
        cfl = self.cfl_default if cfl is None else cfl
        a = float(np.abs(self.wave_speed(self._U)).max())
        return cfl * self.dx / ((2 * self.degree + 1) * max(a, 1e-12))

    def advance(self, t_end: float, cfl: float | None = None, max_steps: int = 10_000_000) -> dict:
        t = 0.0
        steps = 0
        while t < t_end - 1e-15 and steps < max_steps:
            dt = min(self.cfl_dt(cfl), t_end - t)
            self.step(dt)
            t += dt
            steps += 1
        return {"t": t, "steps": steps, **self.stats}

    def set_state(self, U: np.ndarray) -> None:
        self._U = np.array(U, dtype=np.float64, copy=True)
        self.stats = {"positivity_calls": 0, "theta_min": 1.0, "steps": 0}

    @property
    def U(self) -> np.ndarray:
        return self._U

    # -- diagnostics -------------------------------------------------------
    def cell_means(self, U: np.ndarray | None = None) -> np.ndarray:
        U = self._U if U is None else U
        return U.mean(axis=-1)

    def cell_centers(self) -> np.ndarray:
        return (np.arange(self.n_cells) + 0.5) * self.dx

    def nodal_field(self, comp: int = 0, U: np.ndarray | None = None) -> np.ndarray:
        U = self._U if U is None else U
        return U[comp].reshape(-1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def total_variation(values: np.ndarray) -> float:
    """Discrete total variation ``sum |u_{i+1} - u_i|``."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.abs(np.diff(v)).sum())


def contact_position(xc: np.ndarray, rho: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Locate a contact (density jump) inside ``[x_lo, x_hi]``.

    The Sod contact is the *largest density jump* of the problem, so inside a
    window that excludes the shock the maximal-gradient locator finds it; the
    peak is refined parabolically to second order in the sampling interval.
    """
    xc = np.asarray(xc, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    idx = np.where((xc >= x_lo) & (xc <= x_hi))[0]
    if idx.size < 3:
        return float("nan")
    d = np.abs(np.diff(rho[idx]))
    k = int(np.argmax(d))
    x0 = float(xc[idx[k]])
    if 0 < k < d.size - 1:
        d0, d1, d2 = d[k - 1], d[k], d[k + 1]
        den = d0 - 2.0 * d1 + d2
        if abs(den) > 1e-300:
            x0 += 0.5 * (d0 - d2) / den * float(xc[1] - xc[0])
    return x0


def contact_width(xc: np.ndarray, rho: np.ndarray, x_lo: float, x_hi: float) -> int:
    """Number of cells spanned by the contact transition inside the window.

    Counts cells whose density sits strictly between the two plateau levels of
    the window: a first-order scheme gives ``~1/CFL`` cells, an unlimited P1 DG
    gives a handful, and a contact-resolving flux gives the minimum.
    """
    xc = np.asarray(xc, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    idx = np.where((xc >= x_lo) & (xc <= x_hi))[0]
    if idx.size < 3:
        return 0
    r = rho[idx]
    hi = float(np.max(r))  # left/upper plateau level
    lo = float(np.min(r))  # right/lower plateau level
    # 2 % of the plateau gap on each side filters the plateau itself
    tol = 0.02 * (hi - lo)
    return int(np.count_nonzero((r < hi - tol) & (r > lo + tol)))


def mode_amplitude_phase(xc: np.ndarray, values: np.ndarray, k: int = 1, length: float = 1.0) -> dict:
    """Complex Fourier coefficient of mode ``k`` of a periodic sampled field.

    ``c_k = (1/L) int u e^{-2 pi i k x / L}``: modulus = amplitude (dissipation
    measure), argument = phase (dispersion measure).
    """
    xc = np.asarray(xc, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    c = complex(np.mean(v * np.exp(-2j * np.pi * k * xc / length)))
    return {"amplitude": abs(c), "phase": float(np.angle(c))}

# ---------------------------------------------------------------------------
# B1: rotating field (cone / hump / gaussian) with kink-band error split
# ---------------------------------------------------------------------------


def narrow_gaussian_initial(x: np.ndarray, y: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Smooth companion that stays inside the *inscribed disk* (r < 0.5).

    The rotating test velocity ``a = (0.5 - y, x - 0.5)`` is the rigid rotation
    about the box centre, and it is **not smooth on the period torus**: across
    ``y = 0/1`` the x-velocity jumps from ``0.5`` to ``-0.5`` (and likewise for
    ``a_y`` across ``x = 0/1``).  The identity "exact solution at ``t = 2 pi`` ==
    initial data" therefore only holds for data whose support is carried around
    without ever crossing the box wall -- i.e. inside the circle of radius 0.5
    about (0.5, 0.5).  A sigma = 0.1 Gaussian centred at (0.5, 0.75) reaches
    r = 0.25 + 3 sigma = 0.55 and violates that: its ``L2`` error then plateaus
    at ~2e-3 independent of degree and grid (exactly the "P2 floor" reported as
    an open issue by IN70-N01).  sigma = 0.05 keeps 3 sigma inside r = 0.40.
    """
    return np.exp(-((x - 0.5) ** 2 + (y - 0.75) ** 2) / (2.0 * sigma ** 2))


def hump_initial(x: np.ndarray, y: np.ndarray, radius: float = 0.2) -> np.ndarray:
    """LeVeque cosine hump ``(1 + cos(pi r / radius)) / 4`` (C1, no kink)."""
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2)
    return np.where(r < radius, 0.25 * (1.0 + np.cos(np.pi * r / radius)), 0.0)


def _band_mask(X: np.ndarray, Y: np.ndarray, radius: float, width: float) -> np.ndarray:
    """True inside a band of half-width ``width`` around the cone's kink circle."""
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.75) ** 2)
    return np.abs(r - radius) <= width


def _error_split(u, exact_fn, elem, dx, dy, radius, band_w) -> dict:
    """L1/L2 error, split into the kink band and the smooth remainder."""
    ny, nx = u.shape[0], u.shape[1]
    X, Y, W = gauss_coords_2d(ny, nx, elem, dx, dy)
    uq = apply_axis(apply_axis(u, elem.V, -1), elem.V, -2)
    err = uq - exact_fn(X, Y)
    band = _band_mask(X, Y, radius, band_w)
    tot = float(W.sum())
    out: dict = {"L2": float(np.sqrt((err * err * W).sum() / tot))}
    e2_total = float((err * err * W).sum())
    for tag, m in (("band", band), ("smooth", ~band)):
        w = W[m]
        e = err[m]
        wt = max(float(w.sum()), 1e-300)
        out[f"L1_{tag}"] = float((np.abs(e) * w).sum() / wt)
        out[f"L2_{tag}"] = float(np.sqrt((e * e * w).sum() / wt))
        out[f"frac_err_{tag}"] = float((e * e * w).sum() / max(e2_total, 1e-300))
    return out


def bench_rotating(
    n_list: tuple[int, ...] = (16, 32, 64),
    degrees: tuple[int, ...] = (1, 2),
    initial: str = "cone",
    t_end: float = 2.0 * math.pi,
    cfl: float = 0.3,
    band_cells: float = 2.0,
) -> dict:
    """One full revolution of a scalar field in ``a = (0.5 - y, x - 0.5)``.

    ``initial`` -- ``"cone"`` (LeVeque, C0: kink on the base circle),
    ``"hump"`` (C1 companion) or ``"gaussian"`` (C-infinity, the *order*
    reference).

    The squared error is split into a band of ``band_cells`` cells around the
    kink circle and the smooth remainder.  If the kink band carries almost all
    of the error and its L2 order saturates near 1.5 while the smooth part keeps
    converging at ``p + 1``, the rotating-cone order is *data limited*, not
    scheme limited.
    """
    ic = {
        "cone": cone_initial,
        "gaussian": gaussian_initial,          # sigma = 0.1 (IN70): leaves the inscribed disk
        "smooth": narrow_gaussian_initial,     # sigma = 0.05: valid order companion
        "hump": hump_initial,
    }[initial]
    out: dict = {
        "benchmark": "rotating_field",
        "initial_condition": initial,
        "domain": [0.0, 1.0, 0.0, 1.0],
        "velocity": "(0.5 - y, x - 0.5)",
        "t_end": t_end,
        "cfl": cfl,
        "band_cells": band_cells,
        "rows": [],
    }
    for p in degrees:
        el = ref_element(p)
        for n in n_list:
            dx = dy = 1.0 / n
            X, Y = node_coords_2d(n, n, el, dx, dy)
            u = ic(X, Y)
            axf = 0.5 - Y
            ayf = X - 0.5
            dt = cfl * dx / ((2 * p + 1) * math.sqrt(0.5))
            nsteps = int(math.ceil(t_end / dt))
            dt = t_end / nsteps
            t0 = time.time()
            for _ in range(nsteps):
                u = ssp_rk3(u, dt, advection_rhs_2d, axf, ayf, el, dx, dy, "upwind")
            wall = time.time() - t0
            err = _error_split(u, ic, el, dx, dy, radius=0.15, band_w=band_cells * dx)
            row = {
                "degree": p,
                "n_cells": n,
                "h": dx,
                "n_steps": nsteps,
                "wall_s": wall,
                **err,
                "mass_drift": mass_ratio_drift(ic(X, Y), u, el, dx, dy),
            }
            out["rows"].append(row)
            print(
                f"[rot {initial}] P{p} n={n:4d} L2={row['L2']:.3e} "
                f"band={row['L2_band']:.3e} smooth={row['L2_smooth']:.3e} "
                f"frac_band={row['frac_err_band']:.2f} steps={nsteps} wall={wall:.1f}s",
                flush=True,
            )
    out["orders"] = {}
    for p in degrees:
        rows = [r for r in out["rows"] if r["degree"] == p]
        h = [r["h"] for r in rows]
        out["orders"][f"P{p}"] = {
            "L2": fit_order(h, [r["L2"] for r in rows]),
            "L2_smooth": fit_order(h, [r["L2_smooth"] for r in rows]),
            "L2_band": fit_order(h, [r["L2_band"] for r in rows]),
            "theory": p + 1,
        }
    return out


# ---------------------------------------------------------------------------
# B2: kink-limited order (1-D tent) vs smooth order
# ---------------------------------------------------------------------------


def tent_initial(x: np.ndarray, center: float = 0.5, radius: float = 0.2) -> np.ndarray:
    """``max(0, 1 - |x - center| / radius)``: C0 with a kink at the base."""
    return np.maximum(0.0, 1.0 - np.abs(x - center) / radius)


def gauss_initial(x: np.ndarray, center: float = 0.5, sigma: float = 0.08) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _gauss_1d(n_cells: int, elem: RefElement, dx: float):
    """Gauss quadrature points/weights over a 1-D periodic grid."""
    nq = elem.wq.size
    xq = np.polynomial.legendre.leggauss(nq)[0]
    xc = (np.arange(n_cells) + 0.5) * dx
    X = xc[:, None] + 0.5 * dx * xq[None, :]
    W = np.broadcast_to((0.5 * dx * elem.wq)[None, :], X.shape)
    return X, W


def bench_kink_order(
    n_list: tuple[int, ...] = (32, 64, 128, 256, 512),
    degrees: tuple[int, ...] = (1, 2, 3),
    initial: str = "tent",
    t_end: float = 0.5,
    cfl: float = 0.3,
) -> dict:
    """1-D periodic advection of a kinked vs a smooth field (order separation).

    Theory: a C0 field with a jump in ``u_x`` belongs to ``H^{3/2-eps}``, so the
    L2 error of *any* DG degree is bounded by ``O(h^{1.5})``; the smooth
    companion must instead converge at ``p + 1``.  Running both through the same
    solver separates "scheme order" from "data regularity" -- the two things the
    rotating-cone gate conflates.
    """
    ic = {"tent": tent_initial, "gaussian": gauss_initial}[initial]
    out: dict = {
        "benchmark": "kink_order_1d",
        "initial_condition": initial,
        "velocity": 1.0,
        "t_end": t_end,
        "cfl": cfl,
        "rows": [],
    }
    for p in degrees:
        for n in n_list:
            sol = ScalarDG1D(degree=p, n_cells=n, velocity=1.0)
            sol.set_state(ic(sol.x))
            info = sol.advance(t_end, cfl)
            X, W = _gauss_1d(n, sol.elem, sol.dx)
            uq = apply_axis(sol.u, sol.elem.V, -1)
            err = uq - ic(np.mod(X - t_end, 1.0))  # u(x,t) = u0((x - a t) mod L)
            Wf = W
            L2 = float(np.sqrt((err * err * Wf).sum() / Wf.sum()))
            L1 = float((np.abs(err) * Wf).sum() / Wf.sum())
            row = {
                "degree": p,
                "n_cells": n,
                "h": sol.dx,
                "n_steps": info["steps"],
                "L2": L2,
                "L1": L1,
                "Linf": float(np.abs(err).max()),
            }
            out["rows"].append(row)
            print(f"[kink {initial}] P{p} n={n:4d} L2={L2:.3e} L1={L1:.3e}", flush=True)
    out["orders"] = {}
    for p in degrees:
        rows = [r for r in out["rows"] if r["degree"] == p]
        out["orders"][f"P{p}"] = {
            "L2": fit_order([r["h"] for r in rows], [r["L2"] for r in rows]),
            "L1": fit_order([r["h"] for r in rows], [r["L1"] for r in rows]),
            "theory_smooth": p + 1,
            "theory_kink_cap": 1.5,
        }
    return out


# ---------------------------------------------------------------------------
# B3: variable-coefficient manufactured solution (+ LDG gradient order)
# ---------------------------------------------------------------------------


def _mms2_exact(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)


def _mms2_source(x, y, t, ax, ay, nu):
    """``S = u_t + a.grad u - nu Laplacian u`` for the rotating velocity field."""
    e = np.exp(-t)
    s, c = np.sin(np.pi * x), np.cos(np.pi * x)
    sy, cy = np.sin(np.pi * y), np.cos(np.pi * y)
    u = s * sy * e
    adv = (ax * np.pi * c * sy + ay * np.pi * s * cy) * e
    diff = -nu * (-2.0 * np.pi ** 2 * u)
    return -u + adv + diff


def ldg_gradient_2d(u: np.ndarray, elem: RefElement, dx: float, dy: float) -> tuple:
    """LDG gradient ``q = grad u`` of a nodal 2-D DG field (same stencil as RHS)."""
    from .scalar import face_term

    out = []
    for dim in (1, 0):
        cell_axis, node_axis, h = (-2, -1, dx) if dim == 1 else (-3, -2, dy)
        u_hat = u.take(-1, axis=node_axis)
        u_hat_l = np.roll(u_hat, 1, axis=cell_axis)
        q = -apply_axis(u, elem.Gr, node_axis) + face_term(u_hat_l, u_hat, elem, node_axis, u.ndim)
        out.append(apply_axis(q, elem.Minv, node_axis) * (2.0 / h))
    return out[0], out[1]


def bench_mms_varcoeff(
    n_list: tuple[int, ...] = (8, 16, 32, 64),
    degrees: tuple[int, ...] = (1, 2, 3),
    nu: float = 0.05,
    t_end: float = 0.2,
    cfl_safety: float = 1.0,
) -> dict:
    """MMS on ``[0,2]^2`` with the *rotating* velocity ``a = (0.5 - y, x - 0.5)``.

    Unlike the IN70 manufactured solution (constant ``a = (1,1)``) the velocity
    here is the one the rotating-cone benchmark actually uses, so the
    variable-coefficient volume and face terms are graded on their own, with no
    kink anywhere in the data.  The LDG gradient ``q = grad u`` is graded too.
    """
    from .scalar import l2_error_vs  # local import keeps the module head lean
    from .stability import stable_dt

    L = 2.0
    out: dict = {
        "benchmark": "mms_varcoeff",
        "exact": "sin(pi x) sin(pi y) exp(-t)",
        "velocity": "(0.5 - y, x - 0.5)",
        "domain": [0.0, L, 0.0, L],
        "nu": nu,
        "t_end": t_end,
        "rows": [],
    }
    for p in degrees:
        el = ref_element(p)
        for n in n_list:
            dx = dy = L / n
            X, Y = node_coords_2d(n, n, el, dx, dy)
            axf = 0.5 - Y
            ayf = X - 0.5
            u = _mms2_exact(X, Y, 0.0)
            amax = float(np.sqrt(np.max(axf ** 2 + ayf ** 2)))
            dt = stable_dt(p, dx, advective_speed=amax, nu=nu, safety=cfl_safety, ndim=2)
            nsteps = int(math.ceil(t_end / dt))
            dt = t_end / nsteps
            t = 0.0
            t0 = time.time()
            for _ in range(nsteps):
                from .scalar import convection_diffusion_rhs_2d

                L0 = convection_diffusion_rhs_2d(u, axf, ayf, nu, el, dx, dy) + _mms2_source(
                    X, Y, t, axf, ayf, nu
                )
                u1 = u + dt * L0
                L1 = convection_diffusion_rhs_2d(u1, axf, ayf, nu, el, dx, dy) + _mms2_source(
                    X, Y, t + dt, axf, ayf, nu
                )
                u2 = 0.75 * u + 0.25 * (u1 + dt * L1)
                L2s = convection_diffusion_rhs_2d(u2, axf, ayf, nu, el, dx, dy) + _mms2_source(
                    X, Y, t + 0.5 * dt, axf, ayf, nu
                )
                u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L2s)
                t += dt
            wall = time.time() - t0
            e_u = l2_error_vs(u, lambda x, y: _mms2_exact(x, y, t_end), el, dx, dy)
            # LDG gradient vs the analytic gradient, at the same Gauss points
            qx, qy = ldg_gradient_2d(u, el, dx, dy)
            Xq, Yq, W = gauss_coords_2d(n, n, el, dx, dy)
            qxq = apply_axis(apply_axis(qx, el.V, -1), el.V, -2)
            qyq = apply_axis(apply_axis(qy, el.V, -1), el.V, -2)
            ex = np.pi * np.cos(np.pi * Xq) * np.sin(np.pi * Yq) * np.exp(-t_end)
            ey = np.pi * np.sin(np.pi * Xq) * np.cos(np.pi * Yq) * np.exp(-t_end)
            exx = (qxq - ex) ** 2
            eyy = (qyq - ey) ** 2
            qL2 = float(np.sqrt(((exx + eyy) * W).sum() / W.sum()))
            out["rows"].append(
                {
                    "degree": p,
                    "n_cells": n,
                    "h": dx,
                    "n_steps": nsteps,
                    "L2_u": e_u["L2"],
                    "L1_u": e_u["L1"],
                    "Linf_u": e_u["Linf"],
                    "L2_q": qL2,
                    "wall_s": wall,
                }
            )
            print(
                f"[mms-var] P{p} n={n:3d} L2(u)={e_u['L2']:.3e} L2(q)={qL2:.3e} steps={nsteps}",
                flush=True,
            )
    out["orders"] = {}
    for p in degrees:
        rows = [r for r in out["rows"] if r["degree"] == p]
        h = [r["h"] for r in rows]
        out["orders"][f"P{p}"] = {
            "L2_u": fit_order(h, [r["L2_u"] for r in rows]),
            "L2_q": fit_order(h, [r["L2_q"] for r in rows]),
            "theory_u": p + 1,
            "theory_q": p + 1,
        }
    return out


# ---------------------------------------------------------------------------
# B4: ideal-gas Sod tube -- HLLC vs Rusanov, pressure-based shock locator
# ---------------------------------------------------------------------------


def _sod_exact_waves(t_end: float, gamma: float = 1.4) -> dict:
    """Analytic wave positions of the Sod problem from the exact star state."""
    ref = exact_sod(np.array([0.5]), t_end)
    p_star, u_star = ref["p_star"], ref["u_star"]
    rho_sl, rho_sr = ref["rho_star_l"], ref["rho_star_r"]
    c_l = ref["c_l"]
    s_shock = (rho_sr * u_star - 0.125 * 0.0) / (rho_sr - 0.125)
    c_sl = c_l * (p_star / 1.0) ** ((gamma - 1.0) / (2.0 * gamma))
    return {
        "p_star": p_star,
        "u_star": u_star,
        "rho_star_l": rho_sl,
        "rho_star_r": rho_sr,
        "shock_speed": s_shock,
        "shock_x": 0.5 + s_shock * t_end,
        "contact_x": 0.5 + u_star * t_end,
        "rarefaction_head_x": 0.5 + (0.0 - c_l) * t_end,
        "rarefaction_tail_x": 0.5 + (u_star - c_sl) * t_end,
    }


def bench_sod_hllc(
    n_list: tuple[int, ...] = (100, 200, 400, 800),
    degrees: tuple[int, ...] = (1, 2),
    fluxes: tuple[str, ...] = ("hllc", "rusanov"),
    t_end: float = 0.2,
    cfl: float = 0.3,
    tvb_M: float = 0.0,
) -> dict:
    """Ideal-gas Sod shock tube (gamma = 1.4) with switchable interface flux.

    Three measurements per run:

    * shock position from the **pressure** gradient -- pressure is continuous
      across a contact, so this is the physically correct shock locator;
    * shock position from the **density** gradient -- the Sod contact is the
      largest density jump, so this locator latches onto the *contact* and
      reports a spurious ``~(x_contact - x_shock)`` error.  This is exactly the
      25 % "FAIL" recorded by IN70-N01, and it is a measurement artefact, not a
      solver error;
    * shock strength (density ratio) and contact smearing width, which are what
      actually separate HLLC from Rusanov.
    """
    ex = _sod_exact_waves(t_end)
    out: dict = {
        "benchmark": "sod_shock_tube_ideal_gas",
        "gamma": 1.4,
        "ic": {"rho_l": 1.0, "u_l": 0.0, "p_l": 1.0, "rho_r": 0.125, "u_r": 0.0, "p_r": 0.1},
        "t_end": t_end,
        "cfl": cfl,
        "tvb_M": tvb_M,
        "bc": "outflow (transmissive): a periodic tube of length 1 would let the "
              "wrap-around Riemann problem reach the plateau before t = 0.2",
        "exact": ex,
        "rows": [],
    }
    x_fine = np.linspace(0.0, 1.0, 40001)
    ref = exact_sod(x_fine, t_end)
    for flux in fluxes:
        for p in degrees:
            for n in n_list:
                sol = EulerDG1D_HLLC(degree=p, n_cells=n, length=1.0, gamma=1.4,
                                     flux=flux, use_limiter=True, tvb_M=tvb_M, cfl=cfl,
                                     bc="outflow")
                xc0 = (np.arange(n) + 0.5) / n
                side = (xc0 < 0.5)[:, None]
                rho = np.where(side, 1.0, 0.125) * np.ones_like(sol.x)
                pr = np.where(side, 1.0, 0.1) * np.ones_like(sol.x)
                sol.set_state(sol.from_primitive(rho, np.zeros_like(sol.x), pr))
                t0 = time.time()
                info = sol.advance(t_end, cfl=cfl)
                wall = time.time() - t0
                xc = sol.cell_centers()
                rho_c = sol.cell_means()[0]
                u_c = sol.cell_means()[1] / rho_c
                p_c = (sol.gamma - 1.0) * (sol.cell_means()[2] - 0.5 * rho_c * u_c ** 2)
                xs_p = shock_position(xc, p_c)  # pressure: correct locator
                xs_d = shock_position(xc, rho_c)  # density: catches the contact
                x_contact = contact_position(xc, rho_c, ex["contact_x"] - 0.1, ex["contact_x"] + 0.1)
                rho_ex = np.interp(xc, x_fine, ref["rho"])
                u_ex = np.interp(xc, x_fine, ref["u"])
                p_ex = np.interp(xc, x_fine, ref["p"])
                travel = ex["shock_x"] - 0.5
                # plateau level of the star-right region (between contact and
                # shock): the median of the central 40 % is immune to the
                # smearing wings of a limited contact, unlike a point sample
                # two cells upstream of the shock.
                x_lo = ex["contact_x"] + 0.3 * (ex["shock_x"] - ex["contact_x"])
                x_hi = ex["contact_x"] + 0.7 * (ex["shock_x"] - ex["contact_x"])
                win = (xc >= x_lo) & (xc <= x_hi)
                i_out = int(np.argmin(np.abs(xc - (xs_p + 2.0 * sol.dx))))
                row = {
                    "flux": flux,
                    "degree": p,
                    "n_cells": n,
                    "h": sol.dx,
                    "n_steps": info["steps"],
                    "wall_s": wall,
                    "theta_min": info["theta_min"],
                    "positivity_calls": info["positivity_calls"],
                    "shock_x_pressure": xs_p,
                    "shock_x_exact": ex["shock_x"],
                    "shock_pos_err_pct": 100.0 * abs(xs_p - ex["shock_x"]) / travel,
                    "shock_pos_err_cells": abs(xs_p - ex["shock_x"]) / sol.dx,
                    "shock_x_density": xs_d,
                    "shock_pos_err_pct_density": 100.0 * abs(xs_d - ex["shock_x"]) / travel,
                    "rho_plateau_star_r": float(np.median(rho_c[win])) if np.any(win) else float("nan"),
                    "rho_right_state": float(rho_c[i_out]),
                    "shock_strength_err_pct": 100.0
                    * abs(float(np.median(rho_c[win])) - ex["rho_star_r"])
                    / ex["rho_star_r"],
                    "contact_x": x_contact,
                    "contact_x_exact": ex["contact_x"],
                    "contact_pos_err_cells": (
                        abs(x_contact - ex["contact_x"]) / sol.dx if np.isfinite(x_contact) else float("nan")
                    ),
                    "contact_width_cells": contact_width(
                        xc, rho_c, ex["contact_x"] - 0.05, ex["shock_x"] - 0.02
                    ),
                    "L1_rho": float(np.mean(np.abs(rho_c - rho_ex))),
                    "L1_u": float(np.mean(np.abs(u_c - u_ex))),
                    "L1_p": float(np.mean(np.abs(p_c - p_ex))),
                    "min_rho": float(rho_c.min()),
                    "max_rho": float(rho_c.max()),
                }
                row["gate_shock_2pct"] = bool(row["shock_pos_err_pct"] < 2.0)
                out["rows"].append(row)
                print(
                    f"[sod {flux}] P{p} n={n:4d} xs_p={xs_p:.4f} (exact {ex['shock_x']:.4f}, "
                    f"{row['shock_pos_err_pct']:.2f}%) xs_d={xs_d:.4f} "
                    f"ctr_w={row['contact_width_cells']} steps={info['steps']}",
                    flush=True,
                )
    return out


# ---------------------------------------------------------------------------
# B5: isothermal Sod -- DG <-> verified LBM <-> analytic cross-examination
# ---------------------------------------------------------------------------


def _profile_jump_width(x: np.ndarray, rho: np.ndarray, lo: float, hi: float) -> float:
    """Effective transition width (in cells) ``jump / max|d rho/dx|`` in a window."""
    idx = np.where((x >= lo) & (x <= hi))[0]
    if idx.size < 3:
        return float("nan")
    r = rho[idx]
    g = np.abs(np.diff(r))
    if g.max() <= 1e-300:
        return float("nan")
    return float((r.max() - r.min()) / g.max())


def bench_sod_lbm_crosscheck(
    npz_path: str = "benchmarks/verified/sod_shock_tube/profiles_nx2000.npz",
    n_cells: int = 2000,
    times: tuple[int, ...] = (100, 200, 300, 400),
    degrees: tuple[int, ...] = (1, 2),
    flux: str = "hllc",
    cfl: float = 0.3,
    rho_l: float = 1.0,
    rho_r: float = 0.25,
    cs2: float = 1.0 / 3.0,
) -> dict:
    """Isothermal Sod tube: DG(HLLC) vs the verified D2Q9 run vs analytic.

    The D2Q9 equilibrium closes at ``p = cs^2 rho`` (``cs^2 = 1/3``), so the only
    physical reference for the lattice run is the *isothermal* Riemann problem --
    the same closure this DG solver is run with.  Note that the isothermal
    system has **no contact/entropy wave** (two characteristics only), so the
    comparison is on the shock (position, strength, thickness) and on the
    rarefaction fan.

    This is also the diagnostic for the IN70 open issue "P2 degrades at
    t = 400": with a pure RHS and the Zhang-Shu limiter the long-time behaviour
    of the limited P2 run is recorded over 100..400 shock crossings.
    """
    data = np.load(npz_path)
    cs = math.sqrt(cs2)
    xd = n_cells // 2
    ref0 = exact_isothermal_riemann(np.array([0.0]), 1.0, x0=0.0, rho_l=rho_l, rho_r=rho_r, cs=cs)
    rho_star, u_star, W = ref0["rho_star"], ref0["u_star"], ref0["w_shock"]
    x_lbm = np.arange(n_cells) + 0.5
    out: dict = {
        "benchmark": "sod_isothermal_dg_vs_lbm",
        "npz": npz_path,
        "closure": "p = cs2 rho, cs2 = 1/3",
        "ic": {"rho_l": rho_l, "rho_r": rho_r, "u": 0.0},
        "n_cells": n_cells,
        "flux": flux,
        "cfl": cfl,
        "analytic": {
            "rho_star": rho_star,
            "u_star": u_star,
            "shock_speed": W,
            "rarefaction_head_speed": -cs,
            "rarefaction_tail_speed": u_star - cs,
        },
        "lbm_rows": [],
        "dg_rows": [],
    }
    for tt in times:
        rho_lbm = np.asarray(data[f"t{tt}"], dtype=np.float64)
        lo = xd + 0.5 * W * tt
        hi = xd + 1.5 * W * tt
        xs_lbm = contact_position(x_lbm, rho_lbm, lo, hi)
        x_exact = xd + W * tt
        # analytic profile over the clean window
        xa = x_lbm
        ra = exact_isothermal_riemann(xa, tt, x0=xd, rho_l=rho_l, rho_r=rho_r, cs=cs)
        clean = (x_lbm >= cs * tt) & (x_lbm <= n_cells - W * tt)
        jump = rho_l - rho_r
        out["lbm_rows"].append(
            {
                "t": tt,
                "shock_x": xs_lbm,
                "shock_x_exact": x_exact,
                "shock_pos_err_pct": 100.0 * abs(xs_lbm - x_exact) / (W * tt),
                "shock_thickness_cells": _profile_jump_width(x_lbm, rho_lbm, lo, hi),
                "rho_star_measured": float(np.interp(xd + 0.5 * (u_star - cs + W) * tt, x_lbm, rho_lbm)),
                "L2_vs_analytic_normalised": float(
                    np.sqrt(np.mean((rho_lbm[clean] - ra["rho"][clean]) ** 2)) / jump
                ),
            }
        )
        print(
            f"[sod-lbm] t={tt} xs={xs_lbm:.2f} (exact {x_exact:.2f}, "
            f"{out['lbm_rows'][-1]['shock_pos_err_pct']:.2f}%) "
            f"width={out['lbm_rows'][-1]['shock_thickness_cells']:.1f} cells",
            flush=True,
        )

    for p in degrees:
        sol = EulerDG1D_HLLC(
            degree=p, n_cells=n_cells, length=float(n_cells), isothermal=True,
            cs2=cs2, flux=flux, use_limiter=True, positivity=True, cfl=cfl,
        )
        xc0 = (np.arange(n_cells) + 0.5)
        side = (xc0 < xd)[:, None]
        sol.set_state(
            sol.from_primitive(
                np.where(side, rho_l, rho_r) * np.ones_like(sol.x),
                np.zeros_like(sol.x),
                np.zeros_like(sol.x),
            )
        )
        t_done = 0.0
        for tt in times:
            t0 = time.time()
            info = sol.advance(float(tt) - t_done, cfl=cfl)
            t_done = float(tt)
            wall = time.time() - t0
            rho_dg = sol.cell_means()[0]
            u_dg = sol.cell_means()[1] / rho_dg
            loc = sol.cell_centers()
            lo = xd + 0.5 * W * tt
            hi = xd + 1.5 * W * tt
            xs_dg = contact_position(loc, rho_dg, lo, hi)
            x_exact = xd + W * tt
            ra = exact_isothermal_riemann(loc, tt, x0=xd, rho_l=rho_l, rho_r=rho_r, cs=cs)
            clean = (loc >= cs * tt) & (loc <= n_cells - W * tt)
            rho_lbm = np.asarray(data[f"t{tt}"], dtype=np.float64)
            out["dg_rows"].append(
                {
                    "degree": p,
                    "t": tt,
                    "n_steps": info["steps"],
                    "wall_s": wall,
                    "theta_min": info["theta_min"],
                    "shock_x": xs_dg,
                    "shock_x_exact": x_exact,
                    "shock_pos_err_pct": 100.0 * abs(xs_dg - x_exact) / (W * tt),
                    "shock_thickness_cells": _profile_jump_width(loc, rho_dg, lo, hi),
                    "rho_star_measured": float(np.interp(xd + 0.5 * (u_star - cs + W) * tt, loc, rho_dg)),
                    "u_star_measured": float(np.interp(xd + 0.5 * (u_star - cs + W) * tt, loc, u_dg)),
                    "L2_vs_analytic_normalised": float(
                        np.sqrt(np.mean((rho_dg[clean] - ra["rho"][clean]) ** 2)) / (rho_l - rho_r)
                    ),
                    "L2_vs_lbm_normalised": float(
                        np.sqrt(np.mean((rho_dg[clean] - rho_lbm[clean]) ** 2)) / (rho_l - rho_r)
                    ),
                }
            )
            print(
                f"[sod-lbm DG] P{p} t={tt} xs={xs_dg:.2f} "
                f"({out['dg_rows'][-1]['shock_pos_err_pct']:.2f}%) "
                f"width={out['dg_rows'][-1]['shock_thickness_cells']:.1f} cells "
                f"wall={wall:.1f}s",
                flush=True,
            )
    return out


# ---------------------------------------------------------------------------
# B6: square wave -- quantified dissipation / dispersion, TVD
# ---------------------------------------------------------------------------


def bench_square_wave_metrics(
    n_list: tuple[int, ...] = (64, 128, 256),
    degrees: tuple[int, ...] = (1, 2),
    periods: tuple[int, ...] = (1, 2, 4),
    cfl: float = 0.3,
    limiter_states: tuple[bool, ...] = (True, False),
) -> dict:
    """Advect a square wave and a single Fourier mode; quantify the errors.

    For the square wave (the TVD / limiter case):

    * ``TV_ratio = TV(u_cellmeans(T)) / TV(u0)`` -- limiter dissipation shows up as
      ``TV_ratio < 1``; a scheme that creates oscillations gives ``> 1``;
    * ``overshoot_pct`` / ``undershoot_pct`` -- the "no new extrema" TVD criterion;
    * ``L1`` against the exactly translated profile.

    For the single mode ``sin(2 pi x)`` over one period: ``amplitude_ratio``
    (dissipation) and ``phase_err_rad`` (dispersion) of the first Fourier
    coefficient.  These are the two scalar numbers that quantify numerical
    diffusion and numerical dispersion instead of eyeballing a profile.
    """
    out: dict = {
        "benchmark": "square_wave_metrics",
        "velocity": 1.0,
        "periods": list(periods),
        "cfl": cfl,
        "square_rows": [],
        "mode_rows": [],
    }
    length = 1.0
    for use_lim in limiter_states:
        for p in degrees:
            for n in n_list:
                sol = ScalarDG1D(degree=p, n_cells=n, velocity=1.0, use_limiter=use_lim, tvb_M=0.0)
                x = sol.x
                xx = (x - 0.25) % length
                sol.set_state(np.where(xx < 0.5, 1.0, 0.0))
                u0_mean = sol.cell_means()
                tv0 = total_variation(u0_mean)
                t_done = 0.0
                for per in periods:
                    t0 = time.time()
                    sol.advance(float(per) - t_done, cfl)
                    t_done = float(per)
                    wall = time.time() - t0
                    xc = sol.cell_centers()
                    uc = sol.cell_means()
                    xxc = (xc - 0.25 - per) % length
                    exact = np.where(xxc < 0.5, 1.0, 0.0)
                    row = {
                        "limiter": bool(use_lim),
                        "degree": p,
                        "n_cells": n,
                        "period": per,
                        "TV_ratio": total_variation(uc) / tv0,
                        "overshoot_pct": 100.0 * max(0.0, float(uc.max()) - 1.0),
                        "undershoot_pct": 100.0 * max(0.0, -float(uc.min())),
                        "L1": float(np.mean(np.abs(uc - exact))),
                        "Linf": float(np.abs(uc - exact).max()),
                        "wall_s": wall,
                    }
                    row["tvd_no_new_extrema"] = bool(row["overshoot_pct"] <= 1e-12 and row["undershoot_pct"] <= 1e-12)
                    out["square_rows"].append(row)
                    print(
                        f"[square lim={int(use_lim)}] P{p} n={n:4d} T={per} "
                        f"TV={row['TV_ratio']:.4f} over={row['overshoot_pct']:.2f}% "
                        f"under={row['undershoot_pct']:.2f}% L1={row['L1']:.3e}",
                        flush=True,
                    )
                # single-mode dispersion/dissipation over one period
                sol2 = ScalarDG1D(degree=p, n_cells=n, velocity=1.0, use_limiter=use_lim, tvb_M=0.0)
                sol2.set_state(np.sin(2 * np.pi * sol2.x))
                xc = sol2.cell_centers()
                a0 = mode_amplitude_phase(xc, np.sin(2 * np.pi * xc), 1, length)
                sol2.advance(1.0, cfl)
                uc2 = sol2.cell_means()
                a1 = mode_amplitude_phase(xc, uc2, 1, length)
                exact_mode = np.sin(2 * np.pi * (xc - 1.0))
                out["mode_rows"].append(
                    {
                        "limiter": bool(use_lim),
                        "degree": p,
                        "n_cells": n,
                        "amplitude_ratio": a1["amplitude"] / a0["amplitude"],
                        "phase_err_rad": float(np.angle(np.exp(1j * (a1["phase"] - a0["phase"])))),
                        "L2": float(np.sqrt(np.mean((uc2 - exact_mode) ** 2))),
                    }
                )
                print(
                    f"[mode lim={int(use_lim)}] P{p} n={n:4d} amp={out['mode_rows'][-1]['amplitude_ratio']:.5f} "
                    f"phase_err={out['mode_rows'][-1]['phase_err_rad']:.4f}",
                    flush=True,
                )
    return out


# ---------------------------------------------------------------------------
# B7: conservation ledgers
# ---------------------------------------------------------------------------


def bench_euler_mass_ledger(
    n_cells: int = 200,
    degrees: tuple[int, ...] = (1, 2),
    fluxes: tuple[str, ...] = ("hllc", "rusanov"),
    t_end: float = 0.5,
    cfl: float = 0.3,
) -> dict:
    """Periodic Euler Sod run to t = 0.5; relative drift of mass/momentum/energy.

    A DG scheme whose interface flux is single-valued conserves the three
    invariants exactly up to round-off: the cell means are updated as a
    telescoping sum, and the limiters (minmod and Zhang-Shu) only touch the
    ``k >= 1`` modes of the Legendre expansion, i.e. they leave the cell means
    untouched by construction.
    """
    out: dict = {
        "benchmark": "euler_conservation_ledger",
        "t_end": t_end,
        "cfl": cfl,
        "rows": [],
    }
    for flux in fluxes:
        for p in degrees:
            sol = EulerDG1D_HLLC(degree=p, n_cells=n_cells, gamma=1.4, flux=flux,
                                 use_limiter=True, cfl=cfl)
            xc0 = (np.arange(n_cells) + 0.5) / n_cells
            side = (xc0 < 0.5)[:, None]
            rho = np.where(side, 1.0, 0.125) * np.ones_like(sol.x)
            pr = np.where(side, 1.0, 0.1) * np.ones_like(sol.x)
            U = sol.from_primitive(rho, np.zeros_like(sol.x), pr)
            sol.set_state(U)
            mean0 = sol.cell_means().mean(axis=-1)
            info = sol.advance(t_end, cfl)
            mean1 = sol.cell_means().mean(axis=-1)
            # scale each invariant by a physical reference: mass by itself, the
            # momentum by the total mass (velocity units -- the periodic Sod
            # total momentum is exactly zero, so a *relative* number would be
            # meaningless), the energy by its own integral.
            mass0 = abs(float(mean0[0]))
            drift = [abs(float(mean1[0] - mean0[0])) / max(mass0, 1e-300),
                     abs(float(mean1[1] - mean0[1])) / max(mass0, 1e-300),
                     abs(float(mean1[2] - mean0[2])) / max(abs(float(mean0[2])), 1e-300)]
            row = {
                "flux": flux,
                "degree": p,
                "n_cells": n_cells,
                "n_steps": info["steps"],
                "mass_drift": drift[0],
                "momentum_drift": drift[1],
                "energy_drift": drift[2],
                "theta_min": info["theta_min"],
            }
            out["rows"].append(row)
            print(
                f"[ledger {flux}] P{p} mass={drift[0]:.3e} mom={drift[1]:.3e} E={drift[2]:.3e}",
                flush=True,
            )
    return out
