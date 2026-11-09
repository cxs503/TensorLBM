"""Ice/water drag closure as a *pluggable* callback for the DEM solver.

Criterion **C6** of mission P103W2C-N01 asks for an end-to-end hook: the
TensorDEM particle solver must accept the roughness-dependent water-drag
coefficient ``C_w`` as a *callback*, so the constant entry of the drift model
can be switched on and off without editing the solver, and the momentum the
callback removes from the water must be exactly the momentum it hands to the
particles (exchange residual at round-off, ``< 1e-10`` relative).

Interface
---------
A callback receives a :class:`IceWaterDragState` (relative velocity, particle
size, and whatever roughness attributes the caller attaches) and returns a
:class:`CwResult`.  ``C_w`` is defined as usual through
``tau = rho_w C_w |u_rel| u_rel``; the solver turns it into a force on a
cross-section ``A`` as ``F = 0.5 rho_w C_w A |u_rel| u_rel`` -- exactly the
form of the other drag models in :mod:`tensorlbm.dem.dem_solver`, so the
callback and ``schiller_naumann`` are interchangeable in a run.

Three implementations are provided:

``ConstantCwCallback``
    the ``m5682``-style constant ``5.5e-3`` (baseline, for A/B switching);
``RoughnessCwCallback``
    the three-zone closure of :mod:`tensorlbm.ice_roughness.regime_law` --
    ``C_w`` solved self-consistently from the relative velocity through the
    log law with the measured roughness function;
``TabulatedCwCallback``
    a look-up of the LBM ``C_w`` table (``k_s^+, lambda/k_s``) with bilinear
    interpolation in ``ln C_w``, for consumers that must not depend on the
    ice-roughness module.

Momentum bookkeeping
--------------------
:class:`MomentumExchange` accumulates the impulse the solver applies to the
particles and the reaction it must therefore apply to the water.  The reaction
is *recomputed independently* from the recorded drag states
(:meth:`MomentumExchange.water_reaction`) rather than copied from the particle
side, so a solver that used a different formula on the two sides -- or applied
the impulse twice -- cannot pass :meth:`MomentumExchange.residual`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "IceWaterDragState",
    "CwResult",
    "ConstantCwCallback",
    "RoughnessCwCallback",
    "TabulatedCwCallback",
    "MomentumExchange",
    "EIGHT_CM2_PER_M2",
]

#: Mirror of the platform rule "8 cm = 2 m" is *not* used here; the constant is
#: kept as an explicit conversion for readers of the drift literature.
EIGHT_CM2_PER_M2 = 2.0


@dataclass(frozen=True)
class IceWaterDragState:
    """Input of a ``C_w`` callback: one particle/water interface at one instant."""

    relative_speed: float          # |u_water - u_particle|, m/s
    area: float = 1.0              # reference cross-section, m^2
    particle_index: int = 0
    time: float = 0.0
    ks_m: float = 0.0              # ice-underside roughness height, m
    lambda_over_ks: float = 5.0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_speed": float(self.relative_speed), "area": float(self.area),
            "particle_index": int(self.particle_index), "time": float(self.time),
            "ks_m": float(self.ks_m), "lambda_over_ks": float(self.lambda_over_ks),
        }


@dataclass(frozen=True)
class CwResult:
    """Output of a ``C_w`` callback."""

    c_w: float
    u_tau: float = float("nan")
    ks_plus: float = float("nan")
    regime: str = "unknown"
    source: str = "callback"

    def as_dict(self) -> dict[str, Any]:
        return {"C_w": float(self.c_w), "u_tau": float(self.u_tau),
                "ks_plus": float(self.ks_plus), "regime": self.regime,
                "source": self.source}


class ConstantCwCallback:
    """``C_w`` fixed at a constant -- the ``m5682`` baseline."""

    def __init__(self, c_w: float = 5.5e-3):
        if not (c_w > 0.0):
            raise ValueError("C_w must be positive")
        self.c_w = float(c_w)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, state: IceWaterDragState) -> CwResult:
        self.calls.append(state.as_dict())
        return CwResult(self.c_w, source="constant")


class RoughnessCwCallback:
    """Three-zone closure ``C_w(k_s^+, lambda/k_s)`` as a DEM callback.

    Wraps :func:`tensorlbm.ice_roughness.regime_law.solve_friction_velocity`, so
    ``C_w`` is solved self-consistently with the relative velocity: a faster
    drift raises ``k_s^+`` and therefore the drag.
    """

    def __init__(self, law, k_s_m: float, *, z_ref_m: float = 10.0,
                 nu: float = 1.83e-6, clip_ks_plus: tuple[float, float] | None = None):
        if not (k_s_m > 0.0):
            raise ValueError("k_s must be positive")
        self.law = law
        self.k_s_m = float(k_s_m)
        self.z_ref_m = float(z_ref_m)
        self.nu = float(nu)
        self.clip_ks_plus = clip_ks_plus
        self.calls: list[dict[str, Any]] = []

    def __call__(self, state: IceWaterDragState) -> CwResult:
        from ..ice_roughness import regime_law as rl

        u = max(float(state.relative_speed), 1e-9)
        sol = rl.solve_friction_velocity(u, self.k_s_m, self.law,
                                         z_ref_m=self.z_ref_m, nu=self.nu)
        ks_plus = float(sol["ks_plus"])
        clipped = False
        if self.clip_ks_plus is not None:
            lo, hi = self.clip_ks_plus
            if not (lo <= ks_plus <= hi):
                clipped = True
        self.calls.append({**state.as_dict(), "C_w": float(sol["C_w"]),
                           "ks_plus": ks_plus, "clipped": clipped})
        return CwResult(float(sol["C_w"]), u_tau=float(sol["u_tau"]),
                        ks_plus=ks_plus, regime=str(sol["regime"]),
                        source="three_zone_closure")


class TabulatedCwCallback:
    """Bilinear (in ``ln k_s^+``, ``ln lambda/k_s``) look-up of an LBM ``C_w`` table.

    ``table`` rows must expose ``k_s_plus``, ``lambda_over_ks`` and ``C_w`` (the
    dicts produced by
    :func:`tensorlbm.ice_roughness.propagation.cw_table`).  The physical
    roughness height ``k_s_m`` of the consumer maps its ``u_tau`` onto the
    tabulated ``k_s^+``.  Outside the tabulated ``k_s^+`` range the nearest edge
    value is used and the call is recorded as ``extrapolated``, so a consumer
    cannot silently use the table where it has no data.
    """

    def __init__(self, table: Sequence[Mapping[str, Any]], *,
                 k_s_m: float, lambda_over_ks: float = 5.0,
                 nu: float = 1.83e-6):
        self.rows = [r for r in table if float(r.get("C_w", float("nan"))) > 0.0]
        if not self.rows:
            raise ValueError("empty C_w table")
        if not (k_s_m > 0.0):
            raise ValueError("k_s must be positive")
        self.k_s_m = float(k_s_m)
        self.lambda_over_ks = float(lambda_over_ks)
        self.nu = float(nu)
        self.calls: list[dict[str, Any]] = []
        self._lams, self._kss, self._c_w = self._grid()

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lams = np.array(sorted({float(r["lambda_over_ks"]) for r in self.rows}))
        kss = np.array(sorted({float(r["k_s_plus"]) for r in self.rows}))
        lncw = np.full((lams.size, kss.size), np.nan)
        for r in self.rows:
            i = int(np.searchsorted(lams, float(r["lambda_over_ks"])))
            j = int(np.searchsorted(kss, float(r["k_s_plus"])))
            lncw[i, j] = math.log(float(r["C_w"]))
        if np.any(~np.isfinite(lncw)):  # incomplete grid -> fill by row mean
            for i in range(lams.size):
                row = lncw[i]
                if np.any(~np.isfinite(row)):
                    row[~np.isfinite(row)] = float(np.nanmean(row))
        return lams, kss, lncw

    def __call__(self, state: IceWaterDragState) -> CwResult:
        u = max(float(state.relative_speed), 1e-9)
        lam = float(self.lambda_over_ks)
        u_tau = 0.01
        c_w = 5.5e-3
        for _ in range(50):
            ks_plus = self.k_s_m * u_tau / self.nu
            c_w = self._interp(ks_plus, lam)
            u_tau_new = math.sqrt(c_w) * u
            if abs(u_tau_new - u_tau) <= 1e-10 * max(u_tau_new, 1e-30):
                u_tau = u_tau_new
                break
            u_tau = 0.5 * (u_tau + u_tau_new)
        ks_plus = self.k_s_m * u_tau / self.nu
        extrapolated = bool(ks_plus < self._kss[0] or ks_plus > self._kss[-1]
                            or lam < self._lams[0] or lam > self._lams[-1])
        self.calls.append({**state.as_dict(), "C_w": c_w, "ks_plus": ks_plus,
                           "extrapolated": extrapolated})
        return CwResult(c_w, u_tau=u_tau, ks_plus=ks_plus,
                        regime="table", source="tabulated")

    def _interp(self, ks_plus: float, lam: float) -> float:
        """Bilinear interpolation of ``ln C_w`` on the ``(ln k_s^+, ln lambda/k_s)`` grid."""
        k = float(np.clip(ks_plus, self._kss[0], self._kss[-1]))
        l = float(np.clip(lam, self._lams[0], self._lams[-1]))
        x, y = math.log(k), math.log(l)
        xi = np.log(self._kss)
        yi = np.log(self._lams)
        j = int(np.clip(np.searchsorted(xi, x) - 1, 0, xi.size - 2)) if xi.size > 1 else 0
        i = int(np.clip(np.searchsorted(yi, y) - 1, 0, yi.size - 2)) if yi.size > 1 else 0
        if xi.size > 1:
            tx = (x - xi[j]) / (xi[j + 1] - xi[j])
        else:
            tx = 0.0
        if yi.size > 1:
            ty = (y - yi[i]) / (yi[i + 1] - yi[i])
        else:
            ty = 0.0
        j1, i1 = min(j + 1, xi.size - 1), min(i + 1, yi.size - 1)
        lncw = ((1 - tx) * (1 - ty) * self._c_w[i, j] + tx * (1 - ty) * self._c_w[i, j1]
                + (1 - tx) * ty * self._c_w[i1, j] + tx * ty * self._c_w[i1, j1])
        return float(math.exp(lncw))


class MomentumExchange:
    """Impulse ledger for the particle/water drag exchange (criterion C6).

    ``record`` is called by the solver with the force it is about to apply to a
    particle; ``water_reaction`` recomputes the opposite impulse independently
    from the stored drag states, so the residual measures the *implementation*,
    not Newton's third law by construction.
    """

    def __init__(self, rho_water: float = 1025.0):
        self.rho_water = float(rho_water)
        self._particle_impulse = np.zeros(3, dtype=np.float64)
        self._states: list[dict[str, Any]] = []

    def record(self, force: np.ndarray, dt: float, *, c_w: float, area: float,
               relative_velocity: Sequence[float]) -> None:
        self._particle_impulse += np.asarray(force, dtype=np.float64) * float(dt)
        self._states.append({"c_w": float(c_w), "area": float(area), "dt": float(dt),
                             "u_rel": np.asarray(relative_velocity, dtype=np.float64)})

    def water_reaction(self) -> np.ndarray:
        """Impulse the water must receive: ``-0.5 rho C_w A |u| u dt`` per event."""
        total = np.zeros(3, dtype=np.float64)
        for s in self._states:
            u = s["u_rel"]
            speed = float(np.linalg.norm(u))
            total -= 0.5 * self.rho_water * s["c_w"] * s["area"] * speed * u * s["dt"]
        return total

    @property
    def particle_impulse(self) -> np.ndarray:
        return self._particle_impulse.copy()

    def residual(self) -> dict[str, float]:
        water = self.water_reaction()
        momentum = self._particle_impulse + water           # should vanish
        scale = float(np.linalg.norm(self._particle_impulse) + np.linalg.norm(water))
        rel = float(np.linalg.norm(momentum) / scale) if scale > 0.0 else 0.0
        return {
            "particle_impulse": [float(v) for v in self._particle_impulse],
            "water_reaction": [float(v) for v in water],
            "net_impulse": [float(v) for v in momentum],
            "scale": scale,
            "relative_residual": rel,
            "n_events": len(self._states),
        }