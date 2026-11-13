"""Self-propulsion actuator-disk model for ship propellers (D3Q19 LBM).

This module implements the P162-N05 task: a *ship self-propulsion body-force
(actuator disk) model* built in four layers, each of which is independently
verifiable:

1. **Volume-force distribution** -- ``build_disk_force`` distributes thrust and
   torque over a disk of diameter ``D`` as

       f_x(r, z)  = +T  * h(z/delta) g(r/R) / Z_g
       f_th(r,z)  =  Q  * h(z/delta) w(r/R) / (r Z_w)

   with a selectable radial shape ``g`` (``uniform`` / ``elliptic`` /
   ``betz``, the classical zero-at-hub-and-tip minimum-induced-loss shape),
   a selectable axial thickness shape ``h`` (``box`` / ``cosine`` / ``gauss``)
   and *discrete* normalisation ``Z_g``, ``Z_w`` so that the applied grid force
   integrates exactly to the requested thrust ``T`` and torque ``Q``
   (checked by :func:`disk_force_integrals`).

2. **Self-propulsion iteration** -- ``solve_self_propulsion`` finds the
   revolutions ``n`` (equivalently ``J``) that balances propeller thrust against
   hull resistance ``R(U)``.  Both a damped Newton-Raphson iteration (analytic
   ``dK_T/dJ`` from the polynomial anchor) and Brent's bracketing method are
   provided; the residual ``|T - R|/R`` is reported per iteration.

3. **Thrust-torque-advance-coefficient mutual adjudication** --
   ``adjudicate_coefficients`` compares ``K_T(J)``, ``K_Q(J)`` and
   ``eta_0(J) = J K_T / (2 pi K_Q)`` from three *independent* sources:
   the analytic open-water anchor, a geometry-based blade-element/momentum
   (BEM) solution, and the LBM actuator-disk run itself (thrust/torque
   recovered from the axial momentum flux of the simulated flow, not from the
   imposed force).

4. **Open-water analytic anchor** -- ``OpenWaterAnchor`` is a smooth
   polynomial representation of the published KP505 open-water curve
   (Fujisawa et al. 2000 / SIMMAN 2010, as tabulated in
   :func:`tensorlbm.selfpropulsion.kp505_reference_table`).  The coefficients
   are produced by least squares *at call time* from the tabulated data, so no
   coefficient in this module is hand-transcribed.

Verification physics (Froude/Rankine actuator-disk momentum theory)
-------------------------------------------------------------------
For a uniformly loaded disk of area ``A`` in a stream ``U`` the thrust
coefficient ``C_T = T / (0.5 rho U^2 A)`` fixes the axial induction ``a`` by
``C_T = 4 a (1 + a)``, hence the disk velocity ``u_d = U(1 + a)`` and the
far-wake velocity ``u_w = U(1 + 2 a)``.  ``rankine_disk`` returns these, and the
LBM driver ``run_actuator_disk_lbm`` reports the measured counterparts so that
the node criteria C1/C4/C6/C7 can be machine-checked.

References
----------
- Froude, R. E. (1889) *Trans. INA* 30 -- actuator-disk momentum theory.
- Rankine, W. J. M. (1865) *Trans. INA* 6 -- actuator disk in a streamtube.
- Fujisawa, J. et al. (2000); SIMMAN (2010) -- KP505 open-water data.
- Guo, Z., Zheng, C., Shi, B. (2002) *Phys. Rev. E* 65, 046308 -- forcing
  scheme used for the body force (second order in the force).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # torch is optional for the analytic layers
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

__all__ = [
    "RADIAL_SHAPES",
    "AXIAL_SHAPES",
    "DiskForceSpec",
    "radial_profile",
    "axial_profile",
    "build_disk_force",
    "disk_force_integrals",
    "OpenWaterAnchor",
    "kp505_anchor",
    "ResistanceModel",
    "solve_self_propulsion",
    "solve_self_propulsion_brent",
    "rankine_disk",
    "adjudicate_coefficients",
    "LatticeConfig",
    "run_actuator_disk_lbm",
]

RADIAL_SHAPES = ("uniform", "elliptic", "betz")
AXIAL_SHAPES = ("box", "cosine", "gauss")


# ---------------------------------------------------------------------------
# 1. Radial / axial loading shapes
# ---------------------------------------------------------------------------


def radial_profile(eps: np.ndarray, kind: str = "uniform") -> np.ndarray:
    """Un-normalised radial loading shape ``g(eps)``, ``eps = r / R``.

    ``uniform``  : ``g = 1`` -- the classical uniformly loaded disk.
    ``elliptic`` : ``g = sqrt(1 - eps^2)`` -- elliptic loading (zero at the tip,
                   maximum at the axis), the potential-flow lifting-line shape.
    ``betz``     : ``g = eps sqrt(1 - eps^2)`` -- zero loading at hub and tip
                   with the maximum at ``eps = 1/sqrt(2)``, the signature of the
                   Betz minimum-induced-loss circulation.  This is a *parametric
                   shape* with the Betz sign structure, not the exact Goldstein
                   solution (declared as a limitation).
    """
    e = np.clip(np.asarray(eps, dtype=float), 0.0, 1.0)
    if kind == "uniform":
        return np.ones_like(e)
    if kind == "elliptic":
        return np.sqrt(np.maximum(1.0 - e * e, 0.0))
    if kind == "betz":
        return e * np.sqrt(np.maximum(1.0 - e * e, 0.0))
    raise ValueError(f"unknown radial loading kind {kind!r}; expected one of {RADIAL_SHAPES}")


def axial_profile(s: np.ndarray, kind: str = "cosine") -> np.ndarray:
    """Un-normalised axial thickness shape ``h(s)``, ``s`` in half-thickness units.

    ``box``    : top-hat (the usual one-cell-thick actuator disk)
    ``cosine`` : ``0.5 (1 + cos(pi s))`` on ``|s| <= 1`` -- compact support and
                 a continuous derivative at the disk edge
    ``gauss``  : ``exp(-(s/0.4)^2)`` -- Gaussian spreading
    """
    a = np.abs(np.asarray(s, dtype=float))
    if kind == "box":
        return (a <= 1.0).astype(float)
    if kind == "cosine":
        return np.where(a <= 1.0, 0.5 * (1.0 + np.cos(math.pi * np.clip(a, 0.0, 1.0))), 0.0)
    if kind == "gauss":
        return np.exp(-((a / 0.4) ** 2))
    raise ValueError(f"unknown axial loading kind {kind!r}; expected one of {AXIAL_SHAPES}")


@dataclass(frozen=True)
class DiskForceSpec:
    """Specification of an actuator-disk body-force distribution.

    All lengths are in lattice units.  ``thrust`` is the total thrust magnitude:
    the disk delivers ``+thrust`` of streamwise momentum per unit time to the
    fluid (i.e. the force on the fluid points downstream, ``+x``; the reaction
    on the ship is the thrust).  ``torque`` is the total torque about ``+x``.
    """

    diameter: float
    thrust: float
    torque: float
    hub_ratio: float = 0.18
    loading: str = "uniform"
    axial: str = "cosine"
    swirl_loading: str | None = None
    half_thickness: float = 0.75  # in *diameter* units (cells = half_thickness * D)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")
        if not 0.0 <= self.hub_ratio < 1.0:
            raise ValueError("hub_ratio must be in [0, 1)")
        if self.loading not in RADIAL_SHAPES:
            raise ValueError(f"loading must be one of {RADIAL_SHAPES}")
        if self.swirl_loading is not None and self.swirl_loading not in RADIAL_SHAPES:
            raise ValueError(f"swirl_loading must be one of {RADIAL_SHAPES}")
        if self.axial not in AXIAL_SHAPES:
            raise ValueError(f"axial must be one of {AXIAL_SHAPES}")
        if self.half_thickness <= 0:
            raise ValueError("half_thickness must be positive")


def build_disk_force(spec: DiskForceSpec, shape: tuple[int, int, int]) -> dict:
    """Discretise the actuator-disk volume force on a ``(nz, ny, nx)`` grid.

    Returns ``fx``, ``fy``, ``fz`` (force per cell, i.e. force density times the
    unit cell volume) plus the discrete normalisation sums, and the *realised*
    integral thrust/torque.

    The axial coordinate is ``x`` and the disk lies normal to ``x`` at
    ``spec.center``.  The disk weight is ``h(s) g(eps)`` evaluated on the
    discrete cell set (not the continuum integral ``pi (R^2 - R_h^2)``): that is
    what makes the applied thrust equal the requested ``T`` to machine
    precision, which both the ``T = 0`` negative control and the momentum-flux
    adjudication rely on.
    """
    nz, ny, nx = shape
    cx, cy, cz = spec.center
    R = 0.5 * spec.diameter
    if R <= 1.0:
        raise ValueError("diameter too small for the given grid")

    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nx, dtype=float),
        indexing="ij",
    )
    dy = yy - cy
    dz = zz - cz
    r = np.sqrt(dy * dy + dz * dz)
    eps = r / R

    half_t = spec.half_thickness * spec.diameter
    s = (xx - cx) / max(half_t, 1e-12)
    h = axial_profile(s, spec.axial)

    annulus = (eps >= spec.hub_ratio) & (eps <= 1.0)
    mask = annulus & (h > 0.0)

    zeros = np.zeros(shape, dtype=float)
    if (not mask.any()) or (spec.thrust == 0.0 and spec.torque == 0.0):
        return {
            "fx": zeros,
            "fy": zeros.copy(),
            "fz": zeros.copy(),
            "mask": mask,
            "thrust_applied": 0.0,
            "torque_applied": 0.0,
            "thrust_weight": 0.0,
            "torque_weight": 0.0,
            "n_cells": int(mask.sum()),
        }

    g = radial_profile(eps, spec.loading)
    gw = radial_profile(eps, spec.swirl_loading or spec.loading)

    wt = np.where(mask, h * g, 0.0)
    zt = float(wt.sum())
    ws = np.where(mask, h * gw, 0.0)
    zs = float(ws.sum())

    fx = np.zeros(shape, dtype=float)
    fy = np.zeros(shape, dtype=float)
    fz = np.zeros(shape, dtype=float)
    if zt > 0.0:
        # The propeller adds streamwise momentum to the fluid (it pushes the
        # water aft in the ship frame); the reaction on the ship is the thrust.
        fx = spec.thrust * wt / zt
    if zs > 0.0 and spec.torque != 0.0:
        phi = np.arctan2(dz, dy)
        f_theta = np.where(mask, spec.torque * ws / zs, 0.0) / np.maximum(r, 1e-12)
        fy = -f_theta * np.sin(phi)
        fz = f_theta * np.cos(phi)

    return {
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "mask": mask,
        "thrust_applied": float(fx.sum()),
        "torque_applied": float(((yy - cy) * fz - (zz - cz) * fy).sum()),
        "thrust_weight": zt,
        "torque_weight": zs,
        "n_cells": int(mask.sum()),
    }


def disk_force_integrals(
    fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, center: tuple[float, float, float]
) -> dict:
    """Integrate a discrete body-force field to thrust and x-torque."""
    _, cy, cz = center
    nz, ny, nx = fx.shape
    zz, yy, _ = np.meshgrid(
        np.arange(nz, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nx, dtype=float),
        indexing="ij",
    )
    return {
        "thrust": float(fx.sum()),
        "torque": float(((yy - cy) * fz - (zz - cz) * fy).sum()),
    }


# ---------------------------------------------------------------------------
# 2. Open-water analytic anchor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenWaterAnchor:
    """Polynomial open-water anchor ``K_T(J)``, ``K_Q(J)`` with analytic slope.

    Coefficients are least-squares fits to a tabulated open-water curve.  The
    tuple ``kt_coeffs``/``kq_coeffs`` are stored in *increasing* power order.
    """

    kt_coeffs: tuple[float, ...]
    kq_coeffs: tuple[float, ...]
    j_range: tuple[float, float]
    source: str = ""
    fit_rmse: tuple[float, float] = (0.0, 0.0)

    def kt(self, j: float | np.ndarray) -> np.ndarray:
        return np.polyval(self.kt_coeffs[::-1], j)

    def kq(self, j: float | np.ndarray) -> np.ndarray:
        return np.polyval(self.kq_coeffs[::-1], j)

    def dkt_dj(self, j: float) -> float:
        c = self.kt_coeffs
        return float(sum(i * c[i] * j ** (i - 1) for i in range(1, len(c))))

    def dkq_dj(self, j: float) -> float:
        c = self.kq_coeffs
        return float(sum(i * c[i] * j ** (i - 1) for i in range(1, len(c))))

    def eta0(self, j: float | np.ndarray) -> np.ndarray:
        jj = np.asarray(j, dtype=float)
        return jj * self.kt(jj) / (2.0 * math.pi * np.maximum(self.kq(jj), 1e-12))

    def thrust(self, j: float, rho: float, n: float, D: float) -> float:
        """``T = K_T(J) rho n^2 D^4``."""
        return float(self.kt(j)) * rho * n * n * D ** 4

    def torque(self, j: float, rho: float, n: float, D: float) -> float:
        """``Q = K_Q(J) rho n^2 D^5``."""
        return float(self.kq(j)) * rho * n * n * D ** 5


def kp505_anchor(degree: int = 4) -> OpenWaterAnchor:
    """Open-water anchor fitted to the published KP505 curve.

    The tabulated curve lives in
    :func:`tensorlbm.selfpropulsion.kp505_reference_table` (Fujisawa et al.
    2000 / SIMMAN 2010); the polynomial is fitted here so that the coefficients
    are derived rather than transcribed.
    """
    from .selfpropulsion import kp505_reference_table

    table = kp505_reference_table()
    j = np.asarray(table["J"], dtype=float)
    kt = np.asarray(table["KT"], dtype=float)
    kq = np.asarray(table["KQ"], dtype=float)
    ckt = np.polyfit(j, kt, degree)
    ckq = np.polyfit(j, kq, degree)
    rmse_kt = float(np.sqrt(np.mean((np.polyval(ckt, j) - kt) ** 2)))
    rmse_kq = float(np.sqrt(np.mean((np.polyval(ckq, j) - kq) ** 2)))
    return OpenWaterAnchor(
        kt_coeffs=tuple(float(c) for c in ckt[::-1]),
        kq_coeffs=tuple(float(c) for c in ckq[::-1]),
        j_range=(float(j.min()), float(j.max())),
        source=(
            "KP505 open-water data (Fujisawa et al. 2000; SIMMAN 2010), "
            f"degree-{degree} least-squares fit"
        ),
        fit_rmse=(rmse_kt, rmse_kq),
    )


# ---------------------------------------------------------------------------
# 3. Hull resistance model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResistanceModel:
    """ITTC-78 style resistance ``R(U) = 0.5 rho U^2 S (1+k) C_F(U) + wave term``.

    ``C_F = 0.075 / (log10 Re - 2)^2`` with ``Re = U L / nu`` (ITTC-57 line).
    ``wave_coeff`` adds a Froude-scaled residuary term
    ``C_R = wave_coeff Fr^4 / (1 + Fr^2)``; it only makes the resistance curve
    non-trivial -- the self-propulsion iteration is model-agnostic.
    """

    length: float = 230.0
    wetted_area: float = 9500.0
    form_factor: float = 1.15
    rho: float = 1025.0
    nu: float = 1.19e-6
    wave_coeff: float = 0.0
    gravity: float = 9.81

    def cf(self, U: float) -> float:
        re = max(U * self.length / self.nu, 1e3)
        return 0.075 / (math.log10(re) - 2.0) ** 2

    def resistance(self, U: float) -> float:
        if U <= 0.0:
            return 0.0
        q = 0.5 * self.rho * U * U * self.wetted_area
        r = q * self.form_factor * self.cf(U)
        if self.wave_coeff:
            fr = U / math.sqrt(self.gravity * self.length)
            r += q * self.wave_coeff * fr ** 4 / (1.0 + fr * fr)
        return r


# ---------------------------------------------------------------------------
# 4. Self-propulsion point iteration
# ---------------------------------------------------------------------------



def _physical_bracket(
    U: float,
    anchor: OpenWaterAnchor,
    rho: float,
    D: float,
    wake_fraction: float,
    j_lo: float = 0.15,
    j_hi: float | None = None,
) -> tuple[float, float]:
    """Bracket ``n`` so that the advance ratio stays inside the anchor's range.

    ``T(n) = rho D^4 n^2 K_T(J)`` with ``J = V_A/(nD)`` is monotonically
    increasing in ``n`` whenever ``K_T/J^2`` decreases, which holds over the
    fitted range.  Outside the tabulated range the polynomial anchor extrapolates
    wildly, so the search for the propulsion point is confined to
    ``J in [j_lo, j_hi]`` (``j_hi`` defaults to the top of the anchor's table).
    """
    va = U * (1.0 - wake_fraction)
    top = anchor.j_range[1] if j_hi is None else j_hi
    return va / (top * D), va / (j_lo * D)


def _selfprop_residual(
    n: float,
    U: float,
    anchor: OpenWaterAnchor,
    resistance: ResistanceModel,
    rho: float,
    D: float,
    wake_fraction: float,
    thrust_deduction: float,
    mode: str,
) -> tuple[float, float]:
    """Residual ``scale*T(n) - R`` and its derivative w.r.t. ``n``.

    ``mode = "gross"``     : ``T - R = 0`` (as written in the node spec)
    ``mode = "effective"`` : ``T (1 - t) - R = 0`` with thrust deduction ``t``

    The advance ratio always uses the effective inflow ``V_A = U (1 - w)``,
    i.e. ``J = V_A / (n D)``.
    """
    if n <= 0.0:
        return float("inf"), 0.0
    va = U * (1.0 - wake_fraction)
    j = va / (n * D)
    kt = float(anchor.kt(j))
    dkt = anchor.dkt_dj(j)
    thrust = kt * rho * n * n * D ** 4
    dthrust = rho * D ** 4 * (2.0 * n * kt - n * j * dkt)  # dJ/dn = -J/n
    scale = (1.0 - thrust_deduction) if mode == "effective" else 1.0
    return scale * thrust - resistance.resistance(U), scale * dthrust


def solve_self_propulsion(
    U: float,
    anchor: OpenWaterAnchor,
    resistance: ResistanceModel,
    D: float,
    *,
    rho: float = 1025.0,
    wake_fraction: float = 0.0,
    thrust_deduction: float = 0.0,
    mode: str = "gross",
    n0: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 40,
    damping: float = 1.0,
) -> dict:
    """Newton-Raphson self-propulsion point: solve ``T(n) - R(U) = 0`` for ``n``.

    Returns ``n``, ``J``, ``K_T``, ``K_Q``, thrust/torque/power and the
    *iteration history* (residual and relative residual per iteration), so that
    a ``|T - R|/R < 1e-3`` criterion within 20 iterations is machine-checkable.

    A bisection fallback keeps a monotone bracket if a Newton step leaves the
    positive-``n`` region or the bracket; the history records the step type.
    """
    r_required = resistance.resistance(U)
    if r_required <= 0.0:
        raise ValueError("resistance must be positive")
    if mode not in ("gross", "effective"):
        raise ValueError("mode must be 'gross' or 'effective'")

    if n0 is None:
        # Start from the *design* advance ratio (the middle of the anchor's
        # validity range): propellers are designed to operate near it, and a
        # KT(0)-based guess is badly wrong whenever the ship ends up running at
        # J >~ 1 (where KT is small).
        j_start = 0.5 * (anchor.j_range[0] + anchor.j_range[1])
        n0 = U * (1.0 - wake_fraction) / (j_start * D)
    n = float(n0)
    history: list[dict] = []

    lo, hi = _physical_bracket(U, anchor, rho, D, wake_fraction)

    converged = False
    step_kind = "newton"
    for it in range(1, max_iter + 1):
        res, dres = _selfprop_residual(n, U, anchor, resistance, rho, D,
                                       wake_fraction, thrust_deduction, mode)
        rel = abs(res) / r_required
        history.append({"iteration": it, "n": float(n), "residual": float(res),
                        "rel_residual": float(rel), "step": step_kind})
        if rel < tol:
            converged = True
            break
        n_new = n - damping * res / dres if (dres != 0.0 and step_kind == "newton") else float("nan")
        if not math.isfinite(n_new) or not (n_new > 0.0) or not (lo < n_new < hi):
            if res > 0.0:
                hi = n
            else:
                lo = n
            n_new = 0.5 * (lo + hi)
            step_kind = "bisection"
        else:
            step_kind = "newton"
        n = n_new

    va = U * (1.0 - wake_fraction)
    j = va / (n * D)
    kt = float(anchor.kt(j))
    kq = float(anchor.kq(j))
    thrust = kt * rho * n * n * D ** 4
    torque = kq * rho * n * n * D ** 5
    return {
        "n": float(n),
        "J": float(j),
        "KT": kt,
        "KQ": kq,
        "thrust": float(thrust),
        "torque": float(torque),
        "resistance": float(r_required),
        "rel_residual": float(abs(history[-1]["rel_residual"])),
        "iterations": len(history),
        "iterations_to_1e-3": next(
            (h["iteration"] for h in history if h["rel_residual"] < 1e-3), None
        ),
        "converged": bool(converged),
        "history": history,
        "delivered_power": float(2.0 * math.pi * n * torque),
        "eta0": float(j * kt / (2.0 * math.pi * kq)),
        "mode": mode,
        "wake_fraction": wake_fraction,
        "thrust_deduction": thrust_deduction,
    }


def solve_self_propulsion_brent(
    U: float,
    anchor: OpenWaterAnchor,
    resistance: ResistanceModel,
    D: float,
    *,
    rho: float = 1025.0,
    wake_fraction: float = 0.0,
    thrust_deduction: float = 0.0,
    mode: str = "gross",
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict:
    """Brent-bracketing self-propulsion solve (cross-check of the Newton result)."""
    from scipy.optimize import brentq  # local import: scipy is optional

    r_required = resistance.resistance(U)
    target = r_required / ((1.0 - thrust_deduction) if mode == "effective" else 1.0)
    kt0 = max(abs(float(anchor.kt(0.0))), 1e-6)
    n_guess = math.sqrt(target / (rho * D ** 4 * kt0))

    def f(n: float) -> float:
        res, _ = _selfprop_residual(n, U, anchor, resistance, rho, D,
                                    wake_fraction, thrust_deduction, mode)
        return res

    lo, hi = _physical_bracket(U, anchor, rho, D, wake_fraction)
    if f(lo) * f(hi) > 0.0:  # pragma: no cover - defensive
        raise RuntimeError("self-propulsion residual does not bracket a root")
    n = brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
    j = U * (1.0 - wake_fraction) / (n * D)
    kt = float(anchor.kt(j))
    kq = float(anchor.kq(j))
    return {
        "n": float(n),
        "J": float(j),
        "KT": kt,
        "KQ": kq,
        "thrust": kt * rho * n * n * D ** 4,
        "torque": kq * rho * n * n * D ** 5,
        "resistance": r_required,
        "rel_residual": float(abs(f(n)) / r_required),
        "method": "brentq",
    }


# ---------------------------------------------------------------------------
# 5. Rankine / Froude actuator-disk momentum theory
# ---------------------------------------------------------------------------


def rankine_disk(thrust: float, rho: float, U: float, area: float) -> dict:
    """Rankine-Froude momentum theory for a uniformly loaded actuator disk.

    ``C_T = T / (0.5 rho U^2 A)``, ``C_T = 4 a (1 + a)`` so that

        ``a = (-1 + sqrt(1 + C_T)) / 2``,
        ``u_disk = U (1 + a)``, ``u_far = U (1 + 2 a)``, ``eta = 1/(1 + a)``.

    The negative root is selected automatically, so the ``T = 0`` negative
    control gives ``a = 0`` and ``u_disk = u_far = U``.
    """
    ct = thrust / (0.5 * rho * U * U * area)
    disc = 1.0 + ct
    if disc < 0.0:
        raise ValueError("C_T < -1 is outside momentum theory")
    a = (-1.0 + math.sqrt(disc)) / 2.0
    return {
        "CT": float(ct),
        "a": float(a),
        "u_disk": float(U * (1.0 + a)),
        "u_far": float(U * (1.0 + 2.0 * a)),
        "eta_ideal": float(1.0 / (1.0 + a)) if a > -1.0 else float("nan"),
        "power_input": float(thrust * U * (1.0 + a)),
    }


# ---------------------------------------------------------------------------
# 6. Mutual adjudication of K_T / K_Q / J
# ---------------------------------------------------------------------------


def adjudicate_coefficients(j: float, sources: dict[str, tuple[float, float]]) -> dict:
    """Cross-compare ``K_T``/``K_Q`` at one advance ratio from several sources.

    ``sources`` maps ``name -> (KT, KQ)``; the first key is the reference.
    Returns per-source ``eta0 = J KT / (2 pi KQ)`` and relative discrepancies,
    with ``max_kt_error`` / ``max_kq_error`` being the numbers criterion C3
    consumes.
    """
    if not sources:
        raise ValueError("no sources given")
    names = list(sources)
    ref = names[0]
    kt_ref, kq_ref = sources[ref]
    rows = []
    for name in names:
        kt, kq = sources[name]
        rows.append(
            {
                "source": name,
                "J": float(j),
                "KT": float(kt),
                "KQ": float(kq),
                "eta0": float(j * kt / (2.0 * math.pi * kq)) if kq else float("nan"),
                "kt_error_vs_ref": float(abs(kt - kt_ref) / abs(kt_ref)) if kt_ref else float("nan"),
                "kq_error_vs_ref": float(abs(kq - kq_ref) / abs(kq_ref)) if kq_ref else float("nan"),
            }
        )
    others = [r for r in rows if r["source"] != ref]
    return {
        "J": float(j),
        "reference": ref,
        "rows": rows,
        "max_kt_error": max((r["kt_error_vs_ref"] for r in others), default=0.0),
        "max_kq_error": max((r["kq_error_vs_ref"] for r in others), default=0.0),
    }# ---------------------------------------------------------------------------
# 7. D3Q19 lattice-BGK solver with Guo forcing
# ---------------------------------------------------------------------------

_D3Q19 = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
]
_W = [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12
# Specular (free-slip) reflection reverses the *wall-normal* component only.
_Y_MIRROR = [_D3Q19.index((cx, -cy, cz)) for (cx, cy, cz) in _D3Q19]
_Z_MIRROR = [_D3Q19.index((cx, cy, -cz)) for (cx, cy, cz) in _D3Q19]


@dataclass
class LatticeConfig:
    """Configuration of a self-propulsion actuator-disk LBM case.

    Lengths are in lattice cells; velocities in lattice units (``c_s = 1/sqrt(3)``).
    ``diameter`` scales the geometry, so a grid-refinement study runs the same
    physical case with a larger ``diameter`` and proportionally larger
    ``nx, ny, nz`` (all length ratios, hence the non-dimensional physics, are
    unchanged).
    """

    nx: int = 200
    ny: int = 80
    nz: int = 80
    diameter: float = 20.0
    hub_ratio: float = 0.18
    u_inlet: float = 0.05
    tau: float = 0.55
    thrust: float = 0.0
    torque: float = 0.0
    loading: str = "uniform"
    axial: str = "cosine"
    half_thickness: float = 0.75
    disk_x_over_D: float = 2.5
    n_steps: int = 8000
    sample_steps: int = 1500
    report_every: int = 1000
    far_station_over_D: float = 4.0
    name: str = "case"


def _torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the LBM layer")
    return torch


def _strain_and_divergence(ux, uy, uz):
    """Velocity gradients, strain-rate invariant and divergence on a Cartesian grid."""
    dxx, dxy, dxz = np.gradient(ux, axis=(0, 1, 2))
    dyx, dyy, dyz = np.gradient(uy, axis=(0, 1, 2))
    dzx, dzy, dzz = np.gradient(uz, axis=(0, 1, 2))
    s_xx, s_yy, s_zz = dxx, dyy, dzz
    s_xy = 0.5 * (dxy + dyx)
    s_xz = 0.5 * (dxz + dzx)
    s_yz = 0.5 * (dyz + dzy)
    nz, ny, nx = ux.shape
    inner = (slice(2, nz - 2), slice(2, ny - 2), slice(2, nx - 2))
    s2 = (s_xx ** 2 + s_yy ** 2 + s_zz ** 2
          + 2.0 * (s_xy ** 2 + s_xz ** 2 + s_yz ** 2))
    div = dxx + dyy + dzz
    return s2[inner], div, inner


def _viscous_dissipation(ux, uy, uz, mu):
    """``2 mu int S_ij S_ij dV`` on a Cartesian grid (central differences)."""
    s2, _, _ = _strain_and_divergence(ux, uy, uz)
    return float(2.0 * mu * s2.sum())


def run_actuator_disk_lbm(cfg: LatticeConfig) -> dict:
    """Run one self-propulsion actuator-disk LBM case; return diagnostics.

    Scheme: D3Q19 BGK with Guo (2002) forcing for the disk body force,
    equilibrium inlet, zero-gradient outlet and specular (free-slip) lateral
    walls.  Returned diagnostics (lattice units, ``rho = 1``):

    ``rankine``         : momentum-theory ``a``, ``u_disk``, ``u_far``
    ``u_disk_lbm``      : disk-plane mean axial velocity
    ``u_far_lbm``       : far-station mean axial velocity over ``r < 0.5 R``
    ``c1_error``        : ``|u_far_lbm - u_far_theory| / u_far_theory``
    ``thrust_momentum`` : thrust recovered from the axial momentum-flux jump
                          across the disk control volume (independent of the
                          imposed force value)
    ``energy``          : work input, kinetic + pressure energy flux, viscous
                          dissipation and the residual (criterion C6)
    """
    torch_mod = _torch()
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    shape = (nz, ny, nx)
    D = float(cfg.diameter)
    R = 0.5 * D
    area = math.pi * R * R
    cx = float(cfg.disk_x_over_D) * D
    cy, cz = 0.5 * ny, 0.5 * nz
    if cx - 2.0 * D < 2 or cx + (cfg.far_station_over_D + 1.0) * D > nx:
        raise ValueError(
            f"domain too short: nx={nx}, D={D}, disk at x={cx:.1f}; need >=2D upstream "
            f"and >(far station + 1D) downstream"
        )

    spec = DiskForceSpec(
        diameter=D,
        thrust=float(cfg.thrust),
        torque=float(cfg.torque),
        hub_ratio=cfg.hub_ratio,
        loading=cfg.loading,
        axial=cfg.axial,
        half_thickness=cfg.half_thickness,
        center=(cx, cy, cz),
    )
    force = build_disk_force(spec, shape)
    force_present = bool(
        np.any(force["fx"]) or np.any(force["fy"]) or np.any(force["fz"])
    )

    ci = torch_mod.tensor(_D3Q19, dtype=torch_mod.float32)
    w = torch_mod.tensor(_W, dtype=torch_mod.float32).view(19, 1, 1, 1)
    cxv = ci[:, 0].view(19, 1, 1, 1)
    cyv = ci[:, 1].view(19, 1, 1, 1)
    czv = ci[:, 2].view(19, 1, 1, 1)
    u0 = float(cfg.u_inlet)
    cs2 = 1.0 / 3.0
    nu = cs2 * (cfg.tau - 0.5)
    mu = nu  # rho = 1

    fu = torch_mod.tensor(np.ascontiguousarray(force["fx"]), dtype=torch_mod.float32)
    fv = torch_mod.tensor(np.ascontiguousarray(force["fy"]), dtype=torch_mod.float32)
    fw = torch_mod.tensor(np.ascontiguousarray(force["fz"]), dtype=torch_mod.float32)

    f = torch_mod.empty((19, nz, ny, nx), dtype=torch_mod.float32)
    cu0 = cxv * u0
    f[:] = w * (1.0 + 3.0 * cu0 + 4.5 * cu0 * cu0 - 1.5 * u0 * u0)

    inv_tau = 1.0 / float(cfg.tau)
    pref = 1.0 - 1.0 / (2.0 * float(cfg.tau))

    def _sl(c: int) -> tuple[slice, slice]:
        if c == 0:
            return slice(None), slice(None)
        if c > 0:
            return slice(1, None), slice(0, -1)
        return slice(0, -1), slice(1, None)

    stream_tbl = []
    for dx, dy, dz in _D3Q19:
        dz_d, dz_s = _sl(dz)
        dy_d, dy_s = _sl(dy)
        dx_d, dx_s = _sl(dx)
        stream_tbl.append((dz_d, dy_d, dx_d, dz_s, dy_s, dx_s))

    inlet_dirs = [i for i, c in enumerate(_D3Q19) if c[0] > 0]
    outlet_dirs = [i for i, c in enumerate(_D3Q19) if c[0] < 0]
    fy_hole = [i for i, c in enumerate(_D3Q19) if c[1] > 0]
    fy_hole_hi = [i for i, c in enumerate(_D3Q19) if c[1] < 0]
    fz_hole = [i for i, c in enumerate(_D3Q19) if c[2] > 0]
    fz_hole_hi = [i for i, c in enumerate(_D3Q19) if c[2] < 0]

    feq_inlet = torch_mod.empty((len(inlet_dirs), nz, ny), dtype=torch_mod.float32)
    for k, i in enumerate(inlet_dirs):
        cux = _D3Q19[i][0] * u0
        feq_inlet[k] = _W[i] * (1.0 + 3.0 * cux + 4.5 * cux * cux - 1.5 * u0 * u0)

    # --- sampling geometry ---
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=float), np.arange(ny, dtype=float), np.arange(nx, dtype=float),
        indexing="ij",
    )
    rr = np.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    disk_layer = np.abs(xx - cx) <= cfg.half_thickness * D
    disk_plane = disk_layer & (rr >= cfg.hub_ratio * R) & (rr <= R)
    x_far = min(int(round(cx + cfg.far_station_over_D * D)), nx - 2)
    far_mask = (np.abs(xx - x_far) < 0.5) & (rr < 0.5 * R)
    x_up = max(2, int(round(cx - 2.0 * D)))
    x_dn = min(nx - 3, int(round(cx + 3.0 * D)))

    work = 0.0
    work_sample = 0.0
    acc_rho = torch_mod.zeros((nz, ny, nx), dtype=torch_mod.float32)
    acc_ux = torch_mod.zeros((nz, ny, nx), dtype=torch_mod.float32)
    acc_uy = torch_mod.zeros((nz, ny, nx), dtype=torch_mod.float32)
    acc_uz = torch_mod.zeros((nz, ny, nx), dtype=torch_mod.float32)
    n_sample = 0
    ke_start = 0.0
    dev_max = 0.0
    history: list[dict] = []
    sample_from = max(1, cfg.n_steps - cfg.sample_steps)
    step = 0

    for step in range(1, cfg.n_steps + 1):
        rho = f.sum(0)
        ux = ((cxv * f).sum(0) + 0.5 * fu) / rho
        uy = ((cyv * f).sum(0) + 0.5 * fv) / rho
        uz = ((czv * f).sum(0) + 0.5 * fw) / rho
        u2 = ux * ux + uy * uy + uz * uz
        cu = cxv * ux + cyv * uy + czv * uz
        feq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        f *= 1.0 - inv_tau
        f += feq * inv_tau
        if force_present:
            cuf = cxv * fu + cyv * fv + czv * fw
            fuf = fu * ux + fv * uy + fw * uz
            f += w * pref * (3.0 * (cuf - fuf) + 9.0 * cu * cuf)
            work += float(fuf[disk_layer].sum())
            if step >= sample_from:
                work_sample += float(fuf[disk_layer].sum())

        fs = torch_mod.empty_like(f)
        for i, (dz_d, dy_d, dx_d, dz_s, dy_s, dx_s) in enumerate(stream_tbl):
            fs[i, dz_d, dy_d, dx_d] = f[i, dz_s, dy_s, dx_s]

        for k, i in enumerate(inlet_dirs):          # inlet: equilibrium at U
            fs[i, :, :, 0] = feq_inlet[k]
        for i in outlet_dirs:                        # outlet: zero gradient
            fs[i, :, :, -1] = fs[i, :, :, -2]
        for i in fy_hole:                            # free-slip walls (specular)
            fs[i, :, 0, :] = fs[_Y_MIRROR[i], :, 0, :]
        for i in fy_hole_hi:
            fs[i, :, -1, :] = fs[_Y_MIRROR[i], :, -1, :]
        for i in fz_hole:
            fs[i, 0, :, :] = fs[_Z_MIRROR[i], 0, :, :]
        for i in fz_hole_hi:
            fs[i, -1, :, :] = fs[_Z_MIRROR[i], -1, :, :]
        f = fs

        if step == sample_from:
            rho_s = f.sum(0)
            ux_s = (cxv * f).sum(0) / rho_s
            uy_s = (cyv * f).sum(0) / rho_s
            uz_s = (czv * f).sum(0) / rho_s
            ke_start = float((0.5 * rho_s * (ux_s ** 2 + uy_s ** 2 + uz_s ** 2)).sum())
        if step >= sample_from:
            rho = f.sum(0)
            ux = (cxv * f).sum(0) / rho
            uy = (cyv * f).sum(0) / rho
            uz = (czv * f).sum(0) / rho
            acc_rho += rho
            acc_ux += ux
            acc_uy += uy
            acc_uz += uz
            n_sample += 1
            if not force_present:
                dev_max = max(dev_max, float((ux - u0).abs().max()))

        if cfg.report_every and step % cfg.report_every == 0:
            history.append(
                {
                    "step": step,
                    "u_center_downstream_over_U": float(
                        ux[nz // 2, ny // 2, min(int(cx) + int(2 * D), nx - 1)]
                    ) / u0,
                    "work_rate": float(work) / step,
                }
            )

    if n_sample == 0:
        raise RuntimeError("sample_steps must be smaller than n_steps")

    rho_m = (acc_rho / n_sample).numpy().astype(float)
    ux_m = (acc_ux / n_sample).numpy().astype(float)
    uy_m = (acc_uy / n_sample).numpy().astype(float)
    uz_m = (acc_uz / n_sample).numpy().astype(float)

    u_disk_lbm = float(ux_m[disk_plane].mean()) if disk_plane.any() else float("nan")
    u_far_lbm = float(ux_m[far_mask].mean()) if far_mask.any() else float("nan")

    ct = float(cfg.thrust) / (0.5 * 1.0 * u0 * u0 * area)
    rankine = rankine_disk(float(cfg.thrust), 1.0, u0, area)
    c1_error = abs(u_far_lbm - rankine["u_far"]) / abs(rankine["u_far"])

    # ---- thrust / torque recovered from the simulated flow ----
    press = cs2 * rho_m
    mom_x = rho_m * ux_m * ux_m + press
    flux_up = float(mom_x[:, :, x_up].sum())
    flux_dn = float(mom_x[:, :, x_dn].sum())
    thrust_momentum = flux_dn - flux_up
    # Angular-momentum flux about the x axis, through a cross-section plane:
    #   M_x(x) = sum_yz [ (y-cy) rho u_x u_z - (z-cz) rho u_x u_y ]
    # The shaft torque is the *jump* of this flux across the disk, not a volume sum.
    ang = (yy - cy) * (rho_m * ux_m * uz_m) - (zz - cz) * (rho_m * ux_m * uy_m)
    torque_momentum = float(ang[:, :, x_dn].sum() - ang[:, :, x_up].sum())

    # ---- energy balance over the sample window ----
    def _eflux(irho, iux, iuy, iuz, iplane):
        # Kinetic and gauge-fixed pressure energy flux through a plane.  The
        # pressure term is (p - p0) u_x with p0 = c_s^2 rho0: the constant part
        # cancels against mass conservation, and dropping it stops a ~0.1%
        # mass-flux imbalance from swamping the O(1e-3) power balance.
        ke = 0.5 * (iux ** 2 + iuy ** 2 + iuz ** 2)
        p_dev = cs2 * (irho - 1.0)
        kinetic = float((irho * ke * iux)[:, :, iplane].sum())
        pressure = float((p_dev * iux)[:, :, iplane].sum())
        return kinetic, pressure

    ke_end = float((0.5 * rho_m * (ux_m ** 2 + uy_m ** 2 + uz_m ** 2)).sum())
    storage_rate = (ke_end - ke_start) / n_sample if n_sample else 0.0
    # Streamtube-restricted balance: Rankine momentum theory states the energy
    # identity P = dKE + Phi over the *slipstream* control volume.  Masking the
    # flux planes with u_x > U approximates that streamtube (its lateral boundary
    # is a stream surface, so no energy crosses it).
    def _stream_flux(iplane):
        ke = 0.5 * (ux_m ** 2 + uy_m ** 2 + uz_m ** 2)
        inside = ux_m[:, :, iplane] > u0 * 1.0001
        kinetic = float((rho_m * ke * ux_m)[:, :, iplane][inside].sum())
        pressure = float((cs2 * (rho_m - 1.0) * ux_m)[:, :, iplane][inside].sum())
        return kinetic, pressure, int(inside.sum())

    # The upstream end of the streamtube carries the *undisturbed* stream, so its
    # energy flux is 0.5 U^2 * mdot with mdot the slipstream mass flux measured at
    # the downstream plane (masking the uniform inlet plane is meaningless).
    st_ke_out, st_p_out, st_n_out = _stream_flux(nx - 3)
    inside_out = ux_m[:, :, nx - 3] > u0 * 1.0001
    m_dot_wake = float((rho_m * ux_m)[:, :, nx - 3][inside_out].sum())
    st_ke_in = 0.5 * u0 * u0 * m_dot_wake
    st_p_in = 0.0
    st_n_in = int(inside_out.sum())

    ke_out, p_out = _eflux(rho_m, ux_m, uy_m, uz_m, nx - 3)
    ke_in, p_in = _eflux(rho_m, ux_m, uy_m, uz_m, 2)
    kinetic_flux = ke_out - ke_in
    pressure_flux = p_out - p_in
    energy_flux = kinetic_flux + pressure_flux
    s2_field, div_u, inner_idx = _strain_and_divergence(ux_m, uy_m, uz_m)
    dissipation = float(2.0 * mu * s2_field.sum())
    # compression work: for a compressible (isothermal LBM) flow the mechanical
    # energy balance carries an extra -int p div(u) dV term that is absent in the
    # incompressible limit; it is O(Ma^2) and must be measured, not assumed.
    compression_work = float((cs2 * rho_m * div_u)[inner_idx].sum())

    # Work rate averaged over the *sampling window* (steady state), not the
    # whole run: the startup transient is excluded from the energy balance.
    power_in = work_sample / n_sample
    # viscous dissipation inside the streamtube
    inside3d = ux_m > u0 * 1.0001
    diss_st = _viscous_dissipation(np.where(inside3d, ux_m, u0),
                                   np.where(inside3d, uy_m, 0.0),
                                   np.where(inside3d, uz_m, 0.0), mu)
    residual = power_in - energy_flux - dissipation - storage_rate
    residual_ke_only = power_in - kinetic_flux - dissipation - storage_rate
    energy = {
        "power_in": power_in,
        "power_in_whole_run": work / max(step, 1),
        "kinetic_pressure_flux": energy_flux,
        "kinetic_flux": kinetic_flux,
        "pressure_flux": pressure_flux,
        "residual_ke_only": residual_ke_only,
        "compression_work": compression_work,
        "residual_comp": power_in + compression_work - energy_flux - dissipation - storage_rate,
        "residual_comp_rel": (
            (power_in + compression_work - energy_flux - dissipation - storage_rate) / power_in
            if abs(power_in) > 1e-12 else 0.0
        ),
        "viscous_dissipation": dissipation,
        "storage_rate": storage_rate,
        "streamtube": {
            "kinetic_flux": st_ke_out - st_ke_in,
            "pressure_flux": st_p_out - st_p_in,
            "viscous_dissipation": diss_st,
            "residual": power_in - (st_ke_out - st_ke_in) - diss_st - storage_rate,
            "residual_rel": (
                (power_in - (st_ke_out - st_ke_in) - diss_st - storage_rate) / power_in
                if abs(power_in) > 1e-12 else 0.0
            ),
            "mass_flux_wake": m_dot_wake,
            "area_wake": st_n_out,
            "cells_in": [st_n_in, st_n_out],
        },
        "kinetic_energy_start": ke_start,
        "kinetic_energy_end": ke_end,
        "residual": residual,
        "residual_without_storage": power_in - energy_flux - dissipation,
        "residual_ke_only_rel": (
            residual_ke_only / power_in if abs(power_in) > 1e-12 else 0.0
        ),
        "residual_rel": residual / power_in if abs(power_in) > 1e-12 else 0.0,
        "planes": {"inlet": 2, "outlet": nx - 3},
    }

    # radial profile at the far station (figures / C1)
    sel = rr[:, :, x_far] < R
    prof_r = (rr[:, :, x_far][sel] / R)
    prof_u = (ux_m[:, :, x_far][sel] / u0)
    order = np.argsort(prof_r)

    return {
        "name": cfg.name,
        "config": {
            "nx": nx, "ny": ny, "nz": nz, "diameter": D, "hub_ratio": cfg.hub_ratio,
            "u_inlet": u0, "tau": cfg.tau, "nu": nu, "thrust": float(cfg.thrust),
            "torque": float(cfg.torque), "loading": cfg.loading, "axial": cfg.axial,
            "half_thickness": cfg.half_thickness, "n_steps": cfg.n_steps,
            "sample_steps": cfg.sample_steps, "disk_x": cx, "x_far": x_far,
            "x_up": x_up, "x_dn": x_dn, "reynolds_D": u0 * D / nu,
            "disk_cells": int(force["n_cells"]),
        },
        "force_checks": {
            "thrust_requested": float(cfg.thrust),
            "thrust_applied": force["thrust_applied"],
            "torque_requested": float(cfg.torque),
            "torque_applied": force["torque_applied"],
            "thrust_rel_error": (
                abs(force["thrust_applied"] - cfg.thrust) / abs(cfg.thrust)
                if cfg.thrust else 0.0
            ),
            "torque_rel_error": (
                abs(force["torque_applied"] - cfg.torque) / abs(cfg.torque)
                if cfg.torque else 0.0
            ),
        },
        "rankine": rankine,
        "u_disk_lbm": u_disk_lbm,
        "u_far_lbm": u_far_lbm,
        "u_disk_rel_error": abs(u_disk_lbm - rankine["u_disk"]) / rankine["u_disk"],
        "u_far_rel_error": abs(u_far_lbm - rankine["u_far"]) / rankine["u_far"],
        "c1_error": c1_error,
        "CT_target": ct,
        "thrust_momentum": thrust_momentum,
        "thrust_momentum_rel_error": (
            abs(thrust_momentum - cfg.thrust) / abs(cfg.thrust) if cfg.thrust else 0.0
        ),
        "torque_momentum": torque_momentum,
        "energy": energy,
        "uniform_dev_max": dev_max,
        "u_max_abs": float(np.max(np.abs(ux_m))),
        "rho_range": [float(rho_m.min()), float(rho_m.max())],
        "profiles": {
            "r_over_R": prof_r[order].tolist(),
            "u_over_U_far": prof_u[order].tolist(),
            "x_over_D": ((xx[0, 0, :] - cx) / D).tolist(),
            "u_centerline_over_U": (ux_m[nz // 2, ny // 2, :] / u0).tolist(),
        },
        "history": history,
    }