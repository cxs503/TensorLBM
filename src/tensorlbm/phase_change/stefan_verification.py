"""Enthalpy thermal LBM and verification drivers for the Stefan problem.

Numerical method
----------------
A **zero-velocity thermal lattice-Boltzmann equation** (D2Q5 by default, D2Q9
optional, ``cs2 = 1/3``, ``alpha = cs2*(tau - 1/2)``) acts as the discrete
diffusion operator for the *sensible* part of the enthalpy, combined with a
plain equilibrium enthalpy inversion::

    H        = cp*T + L*f_l                      (conserved state variable)
    H^{n+1}  = H^n + cp*(T^lbm - T^n)            (lattice diffusion, sensible part)
    (T, f_l) = H^{-1}(H^{n+1})                   (piecewise inversion at T_freeze)

The population set is then rescaled to the inverted temperature, keeping the
non-equilibrium (heat-flux) part.  In a closed domain the scheme conserves total
enthalpy exactly: the BGK collision term has no zeroth moment and streaming only
moves populations, so ``sum(H^{n+1}) = sum(H^n)`` up to round-off (checked in
``tests/test_stefan_phase_change_verification.py``).

Dirichlet nodes are imposed by resetting the node populations to equilibrium at
the prescribed temperature after streaming.  The *phase state* of a Dirichlet
node is prescribed explicitly (``liquid_fraction`` argument) rather than taken
from the inversion, so a boundary sitting exactly at ``T_freeze`` keeps the
physical phase instead of collapsing to the ``f_l = 0`` branch of the inversion.

Known limitation: the scheme is an *equilibrium* enthalpy method, so metastable
(undercooled) liquid cannot be represented; the cylindrical verification case is
therefore run on the melting branch (see ``stefan_exact`` module docstring).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq

from .stefan_exact import (
    CylindricalSimilarity,
    NeumannSimilarity,
    l2_profile_error,
)

__all__ = [
    "EnthalpyThermalLBM",
    "run_planar_stefan",
    "run_cylindrical_wall_solidification",
    "sharp_interface_radial_solidification",
    "quasi_steady_radial_front",
    "run_grid_convergence",
]

_CS2 = 1.0 / 3.0

_LATTICES: dict[str, tuple[tuple[tuple[int, int], ...], tuple[float, ...]]] = {
    "D2Q5": (
        ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)),
        (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    ),
    "D2Q9": (
        (
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ),
        (
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ),
    ),
}


class EnthalpyThermalLBM:
    """Zero-velocity thermal LBE + equilibrium enthalpy inversion.

    Parameters
    ----------
    nx, ny : int
        Grid size (``ny = 1`` gives the 1-D configuration with a periodic ``y``).
    tau : float
        BGK relaxation time; ``alpha = (tau - 1/2)/3``.
    cp, latent, t_freeze : float
        Material constants in lattice units (``rho = 1``).
    lattice : str
        ``"D2Q5"`` (default) or ``"D2Q9"``.
    periodic : tuple of bool
        ``(periodic_x, periodic_y)``.  Non-periodic boundaries must be covered
        by a Dirichlet mask before stepping.
    """

    def __init__(
        self,
        nx: int,
        ny: int = 1,
        tau: float = 1.0,
        cp: float = 1.0,
        latent: float = 1.0,
        t_freeze: float = 0.0,
        lattice: str = "D2Q5",
        periodic: tuple[bool, bool] = (False, True),
    ) -> None:
        if lattice not in _LATTICES:
            raise ValueError(f"unknown lattice {lattice!r}")
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive diffusivity")
        c, w = _LATTICES[lattice]
        self.c = tuple(c)
        self.w = np.asarray(w, dtype=float)
        self.lattice = lattice
        self.nx, self.ny = int(nx), int(ny)
        self.tau = float(tau)
        self.alpha = _CS2 * (self.tau - 0.5)
        self.cp = float(cp)
        self.latent = float(latent)
        self.t_freeze = float(t_freeze)
        self.periodic = periodic

        self.g = np.zeros((len(self.c), self.nx, self.ny), dtype=float)
        self.T = np.full((self.nx, self.ny), self.t_freeze, dtype=float)
        self.f_l = np.zeros((self.nx, self.ny), dtype=float)
        self.H = np.full((self.nx, self.ny), self.cp * self.t_freeze, dtype=float)

        self.dirichlet_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.dirichlet_T = np.full((self.nx, self.ny), self.t_freeze, dtype=float)
        self.dirichlet_fl = np.zeros((self.nx, self.ny), dtype=float)
        self.steps_done = 0

    # -- setup --------------------------------------------------------------
    def set_field(self, T: np.ndarray, f_l: np.ndarray | None = None) -> None:
        """Initialise temperature and (optionally) liquid fraction."""
        T = np.asarray(T, dtype=float).reshape(self.nx, self.ny)
        self.T = T.copy()
        if f_l is None:
            f_l = np.where(T > self.t_freeze, 1.0, 0.0)
        self.f_l = np.asarray(f_l, dtype=float).reshape(self.nx, self.ny).copy()
        self.H = self.cp * self.T + self.latent * self.f_l
        self.g[:] = self.w[:, None, None] * self.T[None, :, :]

    def add_dirichlet(
        self,
        mask: np.ndarray,
        temperature: float | np.ndarray,
        liquid_fraction: float | np.ndarray | None = None,
    ) -> None:
        """Fix temperature (and phase state) on ``mask``."""
        mask = np.asarray(mask, dtype=bool).reshape(self.nx, self.ny)
        temp = np.asarray(temperature, dtype=float)
        if temp.ndim == 0:
            temp = np.full((self.nx, self.ny), float(temp))
        else:
            temp = temp.reshape(self.nx, self.ny)
        if liquid_fraction is None:
            fl = np.where(temp > self.t_freeze, 1.0, 0.0)
        else:
            fl_arr = np.asarray(liquid_fraction, dtype=float)
            fl = (
                np.full((self.nx, self.ny), float(fl_arr))
                if fl_arr.ndim == 0
                else fl_arr.reshape(self.nx, self.ny)
            )
        self.dirichlet_mask |= mask
        self.dirichlet_T[mask] = temp[mask]
        self.dirichlet_fl[mask] = fl[mask]
        self._apply_dirichlet()

    def update_dirichlet_temperature(self, temperature: float | np.ndarray) -> None:
        """Update the prescribed Dirichlet values (time-dependent BC)."""
        temp = np.asarray(temperature, dtype=float)
        if temp.ndim == 0:
            self.dirichlet_T[self.dirichlet_mask] = float(temp)
        else:
            temp = temp.reshape(self.nx, self.ny)
            self.dirichlet_T[self.dirichlet_mask] = temp[self.dirichlet_mask]
        self._apply_dirichlet()

    def _apply_dirichlet(self) -> None:
        mask = self.dirichlet_mask
        if not mask.any():
            return
        Td = self.dirichlet_T[mask]
        fld = self.dirichlet_fl[mask]
        self.T[mask] = Td
        self.f_l[mask] = fld
        self.H[mask] = self.cp * Td + self.latent * fld
        for i in range(len(self.c)):
            self.g[i][mask] = self.w[i] * Td

    # -- enthalpy inversion -------------------------------------------------
    def invert(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equilibrium enthalpy inversion ``H -> (T, f_l)``."""
        h_s = self.cp * self.t_freeze
        h_l = h_s + self.latent
        T = np.where(
            H <= h_s,
            H / self.cp,
            np.where(H >= h_l, (H - self.latent) / self.cp, self.t_freeze),
        )
        f_l = np.clip((H - self.cp * T) / self.latent, 0.0, 1.0)
        return T, f_l

    # -- LBE sweep ----------------------------------------------------------
    def _stream(self, gpost: np.ndarray) -> np.ndarray:
        g = np.empty_like(gpost)
        px, py = self.periodic
        for i, (cx, cy) in enumerate(self.c):
            a = gpost[i]
            if cx == 0:
                b = a
            elif px:
                b = np.roll(a, cx, axis=0)
            else:
                b = np.zeros_like(a)
                if cx > 0:
                    b[cx:, :] = a[:-cx, :]
                else:
                    b[:cx, :] = a[-cx:, :]
            if cy == 0:
                g[i] = b
            elif py:
                g[i] = np.roll(b, cy, axis=1)
            else:
                d = np.zeros_like(b)
                if cy > 0:
                    d[:, cy:] = b[:, :-cy]
                else:
                    d[:, :cy] = b[:, -cy:]
                g[i] = d
        return g

    def step(self, n: int = 1) -> None:
        """Advance ``n`` lattice time steps (``dt = dx = 1``)."""
        for _ in range(n):
            T_old = self.T
            geq = self.w[:, None, None] * T_old[None, :, :]
            gpost = self.g - (self.g - geq) / self.tau
            self.g = self._stream(gpost)
            T_star = self.g.sum(axis=0)

            self.H = self.H + self.cp * (T_star - T_old)
            T_new, fl_new = self.invert(self.H)
            self.g += self.w[:, None, None] * (T_new - T_star)[None, :, :]
            self.T = T_new
            self.f_l = fl_new
            self._apply_dirichlet()
            self.steps_done += 1

    # -- diagnostics --------------------------------------------------------
    def total_enthalpy(self, mask: np.ndarray | None = None) -> float:
        H = self.cp * self.T + self.latent * self.f_l
        if mask is not None:
            H = np.where(np.asarray(mask, dtype=bool).reshape(self.nx, self.ny), H, 0.0)
        return float(np.sum(H))

    def sensible_enthalpy(self, mask: np.ndarray | None = None) -> float:
        S = self.cp * (self.T - self.t_freeze)
        if mask is not None:
            S = np.where(np.asarray(mask, dtype=bool).reshape(self.nx, self.ny), S, 0.0)
        return float(np.sum(S))

    def latent_content(self, mask: np.ndarray | None = None) -> float:
        """Stored latent heat relative to the fully solid state."""
        Lc = self.latent * self.f_l
        if mask is not None:
            Lc = np.where(np.asarray(mask, dtype=bool).reshape(self.nx, self.ny), Lc, 0.0)
        return float(np.sum(Lc))

    def wall_heat_rate(self, side: str = "left", order: int = 2) -> float:
        """Heat rate through a boundary face (per unit depth, lattice units).

        Positive means heat flows *into* the domain.  ``order=2`` uses the
        second-order one-sided stencil ``(3T0 - 4T1 + T2)/2`` (independent of
        the enthalpy bookkeeping, hence a non-trivial consistency check against
        the enthalpy budget); ``order=1`` uses ``(T0 - T1)`` — the stencil
        implicitly contained in the discrete Laplacian, i.e. an identity by
        construction (reported only for transparency).
        """
        T = self.T
        k = self.cp * self.alpha
        if side == "left":
            if order == 1:
                return float(k * (T[0, :].mean() - T[1, :].mean()))
            return float(
                k * (3.0 * T[0, :].mean() - 4.0 * T[1, :].mean() + T[2, :].mean()) / 2.0
            )
        if side == "right":
            n = self.nx
            if order == 1:
                return float(k * (T[n - 1, :].mean() - T[n - 2, :].mean()))
            return float(
                k
                * (3.0 * T[n - 1, :].mean() - 4.0 * T[n - 2, :].mean() + T[n - 3, :].mean())
                / 2.0
            )
        raise ValueError(f"unknown side {side!r}")

    def front_x(self, solid_behind_left: bool = True, mask: np.ndarray | None = None) -> float:
        """Planar front position from the phase integral over unmasked cells."""
        reach = np.ones((self.nx, self.ny), dtype=bool)
        if mask is not None:
            reach = np.asarray(mask, dtype=bool).reshape(self.nx, self.ny)
        phase = (1.0 - self.f_l) if solid_behind_left else self.f_l
        return float(np.sum(np.where(reach, phase, 0.0)))

    def front_half_crossing(self, solid_behind_left: bool = True) -> float:
        """Front position from the ``f_l = 0.5`` crossing (linear interpolation)."""
        prof = self.f_l[:, self.ny // 2]
        if solid_behind_left:
            idx = np.where(prof >= 0.5)[0]
            if idx.size == 0:
                return float(self.nx - 1)
            i = int(idx[0])
            if i == 0:
                return 0.0
            f0, f1 = prof[i - 1], prof[i]
            frac = (0.5 - f0) / (f1 - f0) if f1 != f0 else 0.0
            return float(i - 1 + frac)
        idx = np.where(prof <= 0.5)[0]
        if idx.size == 0:
            return float(self.nx - 1)
        i = int(idx[0])
        if i == 0:
            return 0.0
        f0, f1 = prof[i - 1], prof[i]
        frac = (f0 - 0.5) / (f0 - f1) if f0 != f1 else 0.0
        return float(i - 1 + frac)

    def interface_radius_polar(
        self,
        centre: tuple[float, float],
        n_theta: int = 72,
        r_max: float | None = None,
        dr: float = 0.25,
        liquid_inside: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Front radius ``r(theta)`` from the ``f_l = 0.5`` crossing.

        ``f_l`` is sampled bilinearly on a fine ``dr`` grid along each ray from
        ``centre``.  With ``liquid_inside=True`` the centre is liquid
        (``f_l > 0.5``) and the first downward crossing is reported (e.g. inward
        freezing from a cold wall); otherwise the solid surrounds the centre.
        """
        cx, cy = centre
        if r_max is None:
            r_max = 0.5 * min(self.nx, self.ny) - 2.0
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        radii = np.empty(n_theta)
        samples = np.arange(0.0, r_max, dr)
        f = self.f_l
        nx, ny = self.nx, self.ny

        for k, th in enumerate(theta):
            xs = cx + samples * math.cos(th)
            ys = cy + samples * math.sin(th)
            i0 = np.clip(np.floor(xs).astype(int), 0, nx - 2)
            j0 = np.clip(np.floor(ys).astype(int), 0, ny - 2)
            fx = xs - i0
            fy = ys - j0
            vals = (
                f[i0, j0] * (1 - fx) * (1 - fy)
                + f[i0 + 1, j0] * fx * (1 - fy)
                + f[i0, j0 + 1] * (1 - fx) * fy
                + f[i0 + 1, j0 + 1] * fx * fy
            )
            if liquid_inside:
                below = np.where(vals <= 0.5)[0]
                below = below[below > 0]
                if below.size == 0:
                    radii[k] = r_max
                    continue
                i = int(below[0])
                v0, v1 = vals[i - 1], vals[i]
            else:
                above = np.where(vals >= 0.5)[0]
                if above.size == 0:
                    radii[k] = r_max
                    continue
                i = int(above[0])
                if i == 0:
                    radii[k] = 0.0
                    continue
                v0, v1 = vals[i - 1], vals[i]
            frac = (0.5 - v0) / (v1 - v0) if v1 != v0 else 0.0
            radii[k] = samples[i - 1] + frac * dr
        return theta, radii

    def solid_area(self, mask: np.ndarray | None = None, dx: float = 1.0) -> float:
        """Solid content ``sum(1 - f_l)`` over unmasked cells (``dx^2`` each)."""
        phase = 1.0 - self.f_l
        if mask is not None:
            phase = np.where(np.asarray(mask, dtype=bool).reshape(self.nx, self.ny), phase, 0.0)
        return float(np.sum(phase) * dx * dx)


# ---------------------------------------------------------------------------
# 1-D planar driver
# ---------------------------------------------------------------------------
def run_planar_stefan(
    mode: str = "freezing",
    nx: int = 400,
    stefan_number: float = 0.5,
    tau: float = 1.0,
    front_fraction: float = 0.25,
    sample_fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    lattice: str = "D2Q5",
    energy_audit: bool = True,
) -> dict[str, Any]:
    """Run a 1-D one-phase Stefan case and compare with the Neumann solution.

    ``mode="freezing"``: cold wall at ``x = 0`` (``T_wall = T_freeze - dT``) with
    liquid initially at ``T_freeze``; ``mode="melting"``: hot wall with solid
    initially at ``T_freeze``.  The right boundary stays at ``T_freeze`` far
    beyond the diffusion layer so that the semi-infinite solution applies, and
    the wall heat input is accumulated for the energy audit.
    """
    if mode not in ("freezing", "melting"):
        raise ValueError("mode must be 'freezing' or 'melting'")
    side = "solid" if mode == "freezing" else "liquid"
    latent, cp, t_freeze = 1.0, 1.0, 0.0

    alpha = _CS2 * (tau - 0.5)
    exact = NeumannSimilarity(
        stefan_number=stefan_number,
        alpha=alpha,
        side=side,
        t_freeze=t_freeze,
        cp=cp,
        latent=latent,
    )
    lam = exact.lam
    s_target = front_fraction * nx
    t_end = (s_target / (2.0 * lam)) ** 2 / alpha
    n_steps = max(1, int(round(t_end)))

    sim = EnthalpyThermalLBM(
        nx=nx,
        ny=1,
        tau=tau,
        cp=cp,
        latent=latent,
        t_freeze=t_freeze,
        lattice=lattice,
        periodic=(False, True),
    )
    interior = np.zeros((nx, 1), dtype=bool)
    interior[1:-1, :] = True
    wall_mask = np.zeros((nx, 1), dtype=bool)
    wall_mask[0, 0] = True
    far_mask = np.zeros((nx, 1), dtype=bool)
    far_mask[nx - 1, 0] = True

    if mode == "freezing":
        sim.set_field(np.full((nx, 1), t_freeze), np.ones((nx, 1)))  # all liquid
        sim.add_dirichlet(wall_mask, exact.boundary_temperature, 0.0)  # cold wall: solid
        sim.add_dirichlet(far_mask, t_freeze, 1.0)  # far liquid at T_freeze
        H0 = float(np.count_nonzero(interior)) * latent
    else:
        sim.set_field(np.full((nx, 1), t_freeze), np.zeros((nx, 1)))  # all solid
        sim.add_dirichlet(wall_mask, exact.boundary_temperature, 1.0)  # hot wall: liquid
        sim.add_dirichlet(far_mask, t_freeze, 0.0)  # far solid at T_freeze
        H0 = 0.0

    x = np.arange(nx, dtype=float)
    sample_steps = sorted({max(1, int(round((f**2) * t_end))) for f in sample_fractions})
    if sample_steps[-1] < n_steps:
        sample_steps.append(n_steps)

    series: list[dict[str, Any]] = []
    q_1st = 0.0
    q_2nd = 0.0
    done = 0
    last_profile: dict[str, Any] | None = None

    for target in sample_steps:
        while done < target:
            if energy_audit:
                q_1st += sim.wall_heat_rate("left", order=1)
                q_2nd += sim.wall_heat_rate("left", order=2)
            sim.step(1)
            done += 1
        t = float(done)
        s_exact = exact.front(t)
        if mode == "freezing":
            s_num = sim.front_x(solid_behind_left=True, mask=interior)
            s_local = sim.front_half_crossing(solid_behind_left=True) - 1.0
        else:
            s_num = sim.front_x(solid_behind_left=False, mask=interior)
            s_local = sim.front_half_crossing(solid_behind_left=False) - 1.0
        entry: dict[str, Any] = {
            "step": int(done),
            "time": t,
            "front_exact": s_exact,
            "front_integral": s_num,
            "front_local": s_local,
            "front_rel_error": abs(s_num - s_exact) / s_exact,
            "front_local_rel_error": abs(s_local - s_exact) / s_exact,
        }
        if energy_audit:
            H_total = sim.total_enthalpy(mask=interior)
            entry.update(
                {
                    "wall_heat_input_first_order": q_1st,
                    "wall_heat_input_second_order": q_2nd,
                    "wall_heat_input_exact": exact.wall_heat_input(t),
                    "enthalpy_total": H_total,
                    "enthalpy_sensible": sim.sensible_enthalpy(mask=interior),
                    "enthalpy_latent": sim.latent_content(mask=interior),
                    "energy_change": H_total - H0,
                }
            )
        series.append(entry)
        if done == sample_steps[-1]:
            T_exact = exact.temperature(x, t)
            last_profile = {
                "x": x.tolist(),
                "T_num": sim.T[:, 0].tolist(),
                "T_exact": T_exact.tolist(),
                "f_l": sim.f_l[:, 0].tolist(),
                "front_exact": s_exact,
                "front_integral": s_num,
            }

    final = series[-1]
    profile_err = l2_profile_error(
        np.asarray(last_profile["T_num"]),
        np.asarray(last_profile["T_exact"]),
        x,
        final["front_exact"],
        band=4.0,
        dT=exact.dT,
    )

    result: dict[str, Any] = {
        "case": f"planar_{mode}",
        "mode": mode,
        "lattice": lattice,
        "grid": {"nx": nx, "ny": 1, "dx": 1.0, "dt": 1.0},
        "parameters": {
            "stefan_number": stefan_number,
            "alpha": alpha,
            "tau": tau,
            "cp": cp,
            "latent": latent,
            "t_freeze": t_freeze,
            "boundary_temperature": exact.boundary_temperature,
            "lambda": lam,
            "steps": n_steps,
            "time_end": t_end,
            "front_fraction": front_fraction,
        },
        "series": series,
        "final": {
            "front_exact": final["front_exact"],
            "front_integral": final["front_integral"],
            "front_rel_error": final["front_rel_error"],
            "front_local": final["front_local"],
            "front_local_rel_error": final["front_local_rel_error"],
            "profile_error": profile_err,
        },
        "profile": last_profile,
    }

    if energy_audit:
        d_h = final["energy_change"]
        e_2nd, e_1st = final["wall_heat_input_second_order"], final["wall_heat_input_first_order"]
        e_exact = final["wall_heat_input_exact"]
        latent_change = final["enthalpy_latent"] - H0
        sensible = final["enthalpy_sensible"]
        result["energy"] = {
            "wall_heat_input_1st_order": e_1st,
            "wall_heat_input_2nd_order": e_2nd,
            "wall_heat_input_exact": e_exact,
            "wall_input_1st_vs_exact_rel_error": abs(abs(e_1st) - e_exact) / e_exact,
            "wall_input_2nd_vs_exact_rel_error": abs(abs(e_2nd) - e_exact) / e_exact,
            "enthalpy_change": d_h,
            "sensible_change": sensible,
            "latent_change": latent_change,
            "closure_1st_order": abs(abs(e_1st) - abs(d_h)) / abs(d_h),
            "closure_2nd_order": abs(abs(e_2nd) - abs(d_h)) / abs(d_h),
            # independent decomposition: latent absorbed == input - stored sensible
            "latent_vs_input_minus_sensible": abs(
                latent_change - (abs(e_2nd) - sensible)
            )
            / abs(latent_change),
            "latent_share_of_input": abs(latent_change) / abs(e_2nd),
        }
    return result


# ---------------------------------------------------------------------------
# sharp-interface (front-tracking) reference for inward radial solidification
# ---------------------------------------------------------------------------
def sharp_interface_radial_solidification(
    r_wall: float,
    shell0: float,
    stefan_number: float = 0.5,
    alpha: float | None = None,
    cp: float = 1.0,
    latent: float = 1.0,
    t_freeze: float = 0.0,
    n_nodes: int = 400,
    t_end: float | None = None,
    n_output: int = 80,
    cfl: float = 0.4,
) -> dict[str, Any]:
    """Independent sharp-interface (front-tracking) reference: inward freezing.

    A cold wall at ``r = r_wall`` (kept at ``T_freeze - dT``) freezes a liquid held
    at ``T_freeze``; the solid shell grows inward.  In the shell
    ``r_f(t) <= r <= r_wall``::

        T_t = alpha*(T_rr + T_r/r),   T(r_f) = T_freeze,   T(r_wall) = T_wall
        dr_f/dt = cp*alpha*T_y(0)/((r_wall - r_f)*L)       (Stefan condition, < 0)

    discretised on the Landau-transformed coordinate
    ``y = (r - r_f)/(r_wall - r_f) in [0, 1]`` (``y = 0`` at the front, ``y = 1``
    at the wall) with **backward Euler** for the radial diffusion (tridiagonal
    solve) and an explicit, mesh-motion-limited step for the interface.  This is a
    *different numerical method* from the enthalpy LBE (no phase smearing, no
    enthalpy inversion) and serves as the reference for the 2-D circular-front
    case, for which no closed-form solution exists.

    The initial state is a linear profile across a shell of thickness ``shell0``
    — the same state the 2-D LBE is initialised with, so both codes start from
    identical data and no start-up transient is being compared.
    """
    from scipy.linalg import solve_banded

    if alpha is None:
        alpha = 1.0 / 6.0
    dT = stefan_number * latent / cp
    t_wall = t_freeze - dT
    r_f = r_wall - shell0
    y = np.linspace(0.0, 1.0, n_nodes)
    dy = y[1] - y[0]
    T = t_freeze + (t_wall - t_freeze) * y  # linear profile, front -> wall

    def freeze_time(rf: float) -> float:
        """Quasi-steady (Planck) estimate of the time to reach ``rf``."""
        yy = max(rf / r_wall, 1e-3)
        return (latent * r_wall**2 / (4.0 * cp * alpha * dT)) * (
            (1.0 - yy**2) + 2.0 * yy**2 * math.log(yy)
        )

    if t_end is None:
        t_end = freeze_time(0.5 * r_wall) - freeze_time(r_f)

    out_t, out_r = [0.0], [r_f]
    t = 0.0
    next_out = t_end / n_output
    k = cp * alpha
    guard = 0
    while t < t_end and guard < 2_000_000:
        guard += 1
        L_s = r_wall - r_f
        r = r_f + y * L_s
        Ty = np.gradient(T, dy)
        r_f_dot = k * Ty[0] / (L_s * latent)  # negative: front moves inward

        # explicit mesh-motion (advection) term
        adv = r_f_dot * (1.0 - y) / L_s * Ty
        # implicit radial diffusion: (I - dt*A) T_new = T + dt*adv
        a = alpha / (L_s**2 * dy**2) - alpha / (L_s * r * 2.0 * dy)
        c = alpha / (L_s**2 * dy**2) + alpha / (L_s * r * 2.0 * dy)
        b = -2.0 * alpha / (L_s**2 * dy**2)

        v = max(abs(r_f_dot), 1e-14)
        dt = cfl * min(dy * L_s / v, L_s / v)
        dt = min(dt, t_end - t, next_out - t)

        ab = np.zeros((3, n_nodes))
        ab[0, 1:] = -dt * c[:-1]      # super-diagonal
        ab[1, :] = 1.0 - dt * b       # diagonal
        ab[2, :-1] = -dt * a[1:]      # sub-diagonal
        ab[1, 0] = 1.0
        ab[0, 1] = 0.0
        ab[1, -1] = 1.0
        ab[2, -2] = 0.0
        rhs = T + dt * adv
        rhs[0] = t_freeze
        rhs[-1] = t_wall
        T = solve_banded((1, 1), ab, rhs)
        T[0] = t_freeze
        T[-1] = t_wall
        r_f = r_f + dt * r_f_dot
        t += dt
        if t >= next_out - 1e-12:
            out_t.append(t)
            out_r.append(r_f)
            next_out += t_end / n_output
    return {
        "t": np.asarray(out_t),
        "r_front": np.asarray(out_r),
        "parameters": {
            "r_wall": r_wall,
            "shell0": shell0,
            "stefan_number": stefan_number,
            "alpha": alpha,
            "n_nodes": n_nodes,
            "t_end": t_end,
            "t_wall": t_wall,
        },
    }

def quasi_steady_radial_front(
    r_wall: float, t: float, stefan_number: float = 0.5, alpha: float | None = None,
    cp: float = 1.0, latent: float = 1.0,
) -> float:
    """Planck quasi-steady front position for inward cylindrical freezing.

    Solves ``t = (L*r_wall**2/(4*cp*alpha*dT)) * ((1-y**2) + 2*y**2*log(y))`` for
    ``y = r_f/r_wall`` (closed form of the heat-balance approximation; exact only
    in the limit ``Ste -> 0``, quoted in the report as an anchor).
    """
    if alpha is None:
        alpha = 1.0 / 6.0
    dT = stefan_number * latent / cp
    scale = latent * r_wall**2 / (4.0 * cp * alpha * dT)

    def res(y: float) -> float:
        y = max(y, 1e-9)
        return scale * ((1.0 - y * y) + 2.0 * y * y * math.log(y)) - t

    lo, hi = 1e-6, 1.0
    if res(hi) >= 0.0:
        return r_wall
    return float(brentq(res, lo, hi, xtol=1e-12)) * r_wall


def run_cylindrical_wall_solidification(
    r_wall: int = 40,
    shell0: int = 4,
    stefan_number: float = 0.5,
    tau: float = 1.0,
    lattice: str = "D2Q5",
    n_theta: int = 72,
    sample_count: int = 8,
    travel_fraction: float = 0.6,
    reference_nodes: int = 400,
) -> dict[str, Any]:
    """2-D cylindrical solidification driven by a cold circular wall.

    Liquid at ``T_freeze`` inside a cold wall at ``r = r_wall`` (``T_wall =
    T_freeze - dT``) freezes inward; by symmetry the exact front is a circle and
    the front radius satisfies the Stefan problem for which there is **no
    closed-form solution**.  The run is therefore verified against

    * the **front shape** — angular deviation of ``r(theta)`` from the circle
      (a direct test of the isotropy of the 2-D advancing front), and
    * an **independently implemented sharp-interface front-tracking reference**
      (``sharp_interface_radial_solidification``, different numerical method,
      Richardson-checked in the shipped report), plus the classical Planck
      quasi-steady closed form as a physical anchor.

    Both codes are initialised with the same data: liquid at ``T_freeze`` inside
    ``r_f0 = r_wall - shell0`` and a linear profile across the initial shell.
    The front radius is measured from the liquid content, ``r_f = sqrt(A_l/pi)``
    (sub-cell accurate and unambiguous here because liquid has ``f_l = 1``
    exactly and solid has ``f_l = 0``).
    """
    latent, cp, t_freeze = 1.0, 1.0, 0.0
    alpha = _CS2 * (tau - 0.5)
    dT = stefan_number * latent / cp
    t_wall = t_freeze - dT
    r_f0 = float(r_wall - shell0)

    def freeze_time(rf: float) -> float:
        yy = max(rf / r_wall, 1e-3)
        return (latent * r_wall**2 / (4.0 * cp * alpha * dT)) * (
            (1.0 - yy**2) + 2.0 * yy**2 * math.log(yy)
        )

    r_target = max(r_f0 - travel_fraction * r_f0, 0.2 * r_wall)
    t_end = max(freeze_time(r_target) - freeze_time(r_f0), 1.0)
    n_steps = int(round(t_end))

    n_dom = int(2 * r_wall + 3)
    centre = ((n_dom - 1) / 2.0, (n_dom - 1) / 2.0)
    ii, jj = np.meshgrid(np.arange(n_dom), np.arange(n_dom), indexing="ij")
    rr = np.hypot(ii - centre[0], jj - centre[1])
    wall_radius = float(r_wall)

    sim = EnthalpyThermalLBM(
        nx=n_dom,
        ny=n_dom,
        tau=tau,
        cp=cp,
        latent=latent,
        t_freeze=t_freeze,
        lattice=lattice,
        periodic=(False, False),
    )
    shell = (rr >= r_f0) & (rr < wall_radius)
    T_init = np.where(rr < wall_radius, t_freeze, t_wall)
    T_init = np.where(
        shell, t_freeze + (t_wall - t_freeze) * (rr - r_f0) / max(shell0, 1e-9), T_init
    )
    fl_init = np.where(rr < r_f0, 1.0, 0.0)
    sim.set_field(T_init, fl_init)
    outer = rr >= wall_radius
    sim.add_dirichlet(outer, t_wall, 0.0)
    liquid_mask = ~outer

    def liquid_radius() -> float:
        area = float(np.sum(sim.f_l[liquid_mask]))
        return math.sqrt(area / math.pi)

    r_meas0 = liquid_radius()

    checkpoints = np.unique(
        np.linspace(0, n_steps, sample_count + 1)[1:].astype(int)
    )
    series: list[dict[str, Any]] = []
    done = 0
    theta = radii = None
    for target in checkpoints:
        while done < int(target):
            sim.step(1)
            done += 1
        t = float(done)
        r_area = liquid_radius()
        theta, radii = sim.interface_radius_polar(
            centre, n_theta=n_theta, r_max=wall_radius - 0.5, liquid_inside=True
        )
        r_mean = float(radii.mean())
        dev = radii - r_mean
        qs = quasi_steady_radial_front(
            wall_radius, t + freeze_time(r_f0), stefan_number, alpha, cp, latent
        )
        series.append(
            {
                "step": int(done),
                "time": t,
                "r_area": r_area,
                "r_mean_polar": r_mean,
                "r_quasi_steady": qs,
                "travel_area": r_meas0 - r_area,
                "travel_polar": r_meas0 - r_mean,
                "shape_max_dev_rel": float(np.max(np.abs(dev)) / r_mean),
                "shape_rms_dev_rel": float(np.sqrt(np.mean(dev**2)) / r_mean),
            }
        )

    # ---- independent sharp-interface reference on the same initial data ----
    ref = sharp_interface_radial_solidification(
        r_wall=wall_radius,
        shell0=float(shell0),
        stefan_number=stefan_number,
        alpha=alpha,
        cp=cp,
        latent=latent,
        t_freeze=t_freeze,
        n_nodes=reference_nodes,
        t_end=t_end,
        n_output=max(sample_count, 40),
    )
    ref_coarse = sharp_interface_radial_solidification(
        r_wall=wall_radius,
        shell0=float(shell0),
        stefan_number=stefan_number,
        alpha=alpha,
        cp=cp,
        latent=latent,
        t_freeze=t_freeze,
        n_nodes=max(reference_nodes // 2, 20),
        t_end=t_end,
        n_output=max(sample_count, 40),
    )
    r_ref_at = lambda tt: float(np.interp(tt, ref["t"], ref["r_front"]))  # noqa: E731
    r_refc_at = lambda tt: float(np.interp(tt, ref_coarse["t"], ref_coarse["r_front"]))  # noqa: E731

    for entry in series:
        t = entry["time"]
        rr_ref = r_ref_at(t)
        entry["r_reference"] = rr_ref
        entry["r_rel_error_vs_reference"] = abs(entry["r_area"] - rr_ref) / rr_ref
        entry["r_rel_error_polar_vs_reference"] = abs(entry["r_mean_polar"] - rr_ref) / rr_ref
        entry["r_rel_error_vs_quasi_steady"] = (
            abs(entry["r_area"] - entry["r_quasi_steady"]) / entry["r_quasi_steady"]
        )

    ref_delta = float(np.max(np.abs(ref["r_front"] - ref_coarse["r_front"]))) / float(
        np.mean(ref["r_front"])
    )
    final = series[-1]
    result: dict[str, Any] = {
        "case": "cylindrical_wall_solidification",
        "mode": "inward solidification from a cold circular wall (liquid at T_freeze)",
        "lattice": lattice,
        "grid": {
            "n": n_dom,
            "dx": 1.0,
            "dt": 1.0,
            "wall_radius": wall_radius,
            "initial_shell": float(shell0),
        },
        "parameters": {
            "stefan_number": stefan_number,
            "alpha": alpha,
            "tau": tau,
            "t_wall": t_wall,
            "steps": n_steps,
            "r_f0": r_f0,
            "r_meas0": r_meas0,
        },
        "series": series,
        "final": {
            "time": final["time"],
            "r_area": final["r_area"],
            "r_mean_polar": final["r_mean_polar"],
            "r_reference": final["r_reference"],
            "r_quasi_steady": final["r_quasi_steady"],
            "r_rel_error_vs_reference": final["r_rel_error_vs_reference"],
            "r_rel_error_polar_vs_reference": final["r_rel_error_polar_vs_reference"],
            "r_rel_error_vs_quasi_steady": final["r_rel_error_vs_quasi_steady"],
            "shape_max_dev_rel": final["shape_max_dev_rel"],
            "shape_rms_dev_rel": final["shape_rms_dev_rel"],
        },
        "reference": {
            "method": "sharp-interface front tracking (Landau transform, radial)",
            "n_nodes": int(reference_nodes),
            "n_nodes_coarse": int(max(reference_nodes // 2, 20)),
            "self_convergence_rel": ref_delta,
            "t": ref["t"].tolist(),
            "r_front": ref["r_front"].tolist(),
        },
        "polar": {
            "theta": np.asarray(theta).tolist(),
            "radius": np.asarray(radii).tolist(),
            "centre": list(centre),
        },
        "field": {"n": n_dom, "f_l_centre_row": sim.f_l[:, int(centre[1])].tolist()},
    }
    return result

# ---------------------------------------------------------------------------
# resolution study
# ---------------------------------------------------------------------------
def run_grid_convergence(
    resolutions: Sequence[int] = (25, 50, 100, 200),
    mode: str = "freezing",
    stefan_number: float = 0.5,
    tau: float = 1.0,
    front_fraction: float = 0.25,
    lattice: str = "D2Q5",
) -> dict[str, Any]:
    """Planar Stefan resolution study: front and profile error versus resolution.

    ``resolutions`` are the number of cells the front has travelled at the end of
    the run (``nx = 4 * n_front``), so the run time scales as ``n_front**2``.
    A least-squares fit of ``log(error)`` versus ``log(1/n_front)`` gives the
    observed order of accuracy.
    """
    runs: list[dict[str, Any]] = []
    for n_front in resolutions:
        nx = 4 * int(n_front)
        res = run_planar_stefan(
            mode=mode,
            nx=nx,
            stefan_number=stefan_number,
            tau=tau,
            front_fraction=front_fraction,
            sample_fractions=(1.0,),
            lattice=lattice,
            energy_audit=False,
        )
        runs.append(
            {
                "n_front": int(n_front),
                "nx": nx,
                "steps": res["parameters"]["steps"],
                "front_rel_error": res["final"]["front_rel_error"],
                "front_local_rel_error": res["final"]["front_local_rel_error"],
                "profile_l2_rel": res["final"]["profile_error"]["l2_rel"],
                "profile_linf_rel": res["final"]["profile_error"]["linf_rel"],
            }
        )

    def fit(xs: Sequence[float], ys: Sequence[float]) -> float:
        # order of accuracy p from error ~ (1/n_front)**p  (positive p = converging)
        x = np.log(1.0 / np.asarray(xs, dtype=float))
        y = np.log(np.asarray(ys, dtype=float))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    hs = [r["n_front"] for r in runs]
    orders = {}
    for key in ("front_rel_error", "front_local_rel_error", "profile_l2_rel", "profile_linf_rel"):
        vals = [r[key] for r in runs]
        orders[key] = fit(hs, vals) if all(v > 0 for v in vals) else float("nan")
    return {
        "case": f"planar_{mode}_grid_convergence",
        "mode": mode,
        "stefan_number": stefan_number,
        "runs": runs,
        "observed_order": orders,
        "note": (
            "observed order from a least-squares fit of log(error) vs log(1/n_front); "
            "the enthalpy (diffuse-interface) front measure is expected to converge "
            "first order, the interior temperature profile second order"
        ),
    }