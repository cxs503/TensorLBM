"""Added-mass instability calibration for partitioned fluid-structure coupling.

Makes the interface stability of a partitioned (Dirichlet-Neumann) FSI
sub-iteration a measurable, falsifiable quantity.

1. ``InterfaceModel1D`` -- 1-DOF linear added-mass ("piston"/"balloon")
   interface model with closed-form error factor

       rho_dd = |1 - omega * c_a * rho_f / rho_s|      (omega = 1: explicit)

   where ``c_a = A/(rho_f V)``.  ``PistonPartitioned1D`` runs the same
   fixed-point iteration numerically so analytic and numerical factors are
   cross-validated.
2. ``AitkenRelaxation`` / ``IQNILSRelaxation`` -- interface accelerators.
3. ``HeavingDiskPartitioned`` -- 2D D2Q9 LBM (``moving_boundary_2d.MovingDisk``,
   momentum-exchange impulse) coupled to a 1-DOF heaving body on a spring.

Conventions: lattice units, ``rho_f = 1``, dx = dt = 1.  A coupling window is
``window`` LBM steps; impulses are accumulated over the window.  Structural
mass per unit depth ``m_s = density_ratio * rho_f * V``.  The interface state
is the heave velocity ``v``.  The **pseudo-time (incremental) fixed point**

    v^{k+1} = v^k + omega_k (I(v^k) + I_ext)/m_s,   I(v) = -A (v - v^n) + I_0

has error amplification ``1 - omega A/m_s`` (the closed form above).  The
**time-accurate** scheme ``v^{n+1} = v^n + (I(v)+I_ext)/m_s`` instead has the
textbook amplification ``-A/m_s``; both are reported.

References: Causin, Gerbeau & Nobile, CMAME 194 (2005); Degroote et al.,
CMAME 198 (2009) (IQN-ILS); Irons & Tuck, IJNME 1 (1969) (Aitken).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .fsi_partitioned import clamp_aitken_factor

__all__ = [
    "InterfaceModel1D",
    "PistonPartitioned1D",
    "FixedRelaxation",
    "AitkenRelaxation",
    "IQNILSRelaxation",
    "rho_dd_analytic",
    "critical_density_ratio",
    "estimate_rho_dd",
    "HeavingDiskPartitioned",
    "identify_added_mass_damping",
    "potential_added_mass",
    "scan_stability",
]

Scheme = Literal["none", "aitken", "iqn-ils"]


# --------------------------------------------------------------------------- #
# closed-form interface theory
# --------------------------------------------------------------------------- #
def rho_dd_analytic(
    rho_s_over_rho_f: float,
    c_a: float = 1.0,
    omega: float = 1.0,
    form: Literal["incremental", "residual"] = "incremental",
) -> float:
    """Closed-form interface error-propagation factor ``rho_dd``.

    ``incremental``: ``|1 - omega c_a rho_f/rho_s|`` (default; = |1 - omega A/m_s|)
    ``residual``:    ``|1 - omega (1 + c_a rho_f/rho_s)|`` (structure with inertia)
    """
    if rho_s_over_rho_f <= 0.0:
        raise ValueError("density ratio must be positive")
    if omega <= 0.0:
        raise ValueError("relaxation parameter must be positive")
    ratio = c_a / rho_s_over_rho_f
    if form == "incremental":
        return abs(1.0 - omega * ratio)
    if form == "residual":
        return abs(1.0 - omega * (1.0 + ratio))
    raise ValueError(f"unknown form {form!r}")


def critical_density_ratio(
    c_a: float = 1.0,
    omega: float = 1.0,
    form: Literal["incremental", "residual"] = "incremental",
) -> float:
    """``rho_s/rho_f`` at the stability limit.

    incremental: rho_dd >= 1  <=>  rho_s/rho_f <= omega c_a / 2
    residual   : rho_dd >= 1  <=>  rho_s/rho_f <= c_a / (2/omega - 1)
    """
    if form == "incremental":
        return 0.5 * omega * c_a
    if form == "residual":
        return c_a / max(2.0 / omega - 1.0, 1e-12)
    raise ValueError(f"unknown form {form!r}")


@dataclass(frozen=True)
class InterfaceModel1D:
    """1-DOF linear added-mass interface model (per unit depth)."""

    mass_s: float
    added_mass: float
    stiffness: float = 0.0
    displacement: float = 0.0
    velocity: float = 0.0
    window: float = 1.0
    offset_impulse: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_s <= 0.0 or self.added_mass < 0.0:
            raise ValueError("mass_s must be positive and added_mass non-negative")

    @property
    def added_mass_ratio(self) -> float:
        return self.added_mass / self.mass_s

    @property
    def external_impulse(self) -> float:
        return -self.stiffness * self.displacement * self.window

    def impulse(self, velocity: float) -> float:
        return self.offset_impulse - self.added_mass * (velocity - self.velocity)

    def exact_velocity(self) -> float:
        return self.velocity + (self.offset_impulse + self.external_impulse) / self.added_mass

    def rho_dd(self, omega: float = 1.0) -> float:
        return abs(1.0 - omega * self.added_mass_ratio)


# --------------------------------------------------------------------------- #
# interface accelerators
# --------------------------------------------------------------------------- #
class FixedRelaxation:
    """Constant under-relaxation ``dx = omega * r``."""

    def __init__(self, omega: float = 1.0) -> None:
        if omega <= 0.0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)

    def reset(self) -> None:
        return None

    def step(self, state: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
        return self.omega * np.asarray(residual, dtype=np.float64), self.omega

    def describe(self) -> dict[str, Any]:
        return {"scheme": "fixed", "omega": self.omega}


class AitkenRelaxation:
    """Aitken dynamic relaxation: ``omega_k = -omega_{k-1} r^{k-1}.dr / |dr|^2``.

    ``omega_min = 0`` (default) is scale free: only positivity is enforced,
    via :func:`tensorlbm.fsi_partitioned.clamp_aitken_factor`.  A positive
    absolute floor diverges for ``beta_eff > 1/omega_min - 1`` -- the former
    default ``1e-4`` therefore failed at ``beta_eff = 2e4``
    (``xval/aitken_underrelaxation_life/out/scale_guard.json``).

    ``step`` returns the *increment* ``omega * r`` (not ``state + omega r``),
    unlike its sibling in :mod:`tensorlbm.fsi_partitioned`.
    """

    def __init__(
        self, omega0: float = 1.0, omega_min: float = 0.0, omega_max: float = 2.0
    ) -> None:
        if not (0.0 <= omega_min <= omega0 <= omega_max) or omega0 <= 0.0:
            raise ValueError("require 0 <= omega_min <= omega0 <= omega_max")
        self.omega0 = float(omega0)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.omega = float(omega0)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self.omega = self.omega0
        self._prev = None

    def step(self, state: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.asarray(residual, dtype=np.float64)
        if self._prev is not None:
            dr = r - self._prev
            denom = float(dr @ dr)
            if denom > 1e-300:
                omega = -self.omega * float(self._prev @ dr) / denom
                if math.isfinite(omega):
                    self.omega = clamp_aitken_factor(
                        omega, omega_min=self.omega_min, omega_max=self.omega_max
                    )
        self._prev = r.copy()
        return self.omega * r, self.omega

    def describe(self) -> dict[str, Any]:
        return {
            "scheme": "aitken",
            "omega0": self.omega0,
            "omega_min": self.omega_min,
            "omega_max": self.omega_max,
        }


class IQNILSRelaxation:
    """Interface quasi-Newton with least squares (IQN-ILS), bounded history.

    ``dx = r - W c`` with ``c = argmin ||r + V c||``, ``V = [dR_j]``,
    ``W = [dX_j]`` (at most ``reuse`` most recent pairs, differences stored in
    normalised form), solved by filtered QR.
    """

    def __init__(
        self,
        reuse: int = 5,
        omega: float = 1.0,
        rcond: float = 1e-10,
        max_step: float | None = None,
    ) -> None:
        if reuse < 1:
            raise ValueError("reuse must be >= 1")
        self.reuse = int(reuse)
        self.omega = float(omega)
        self.rcond = float(rcond)
        self.max_step = max_step
        self._dR: list[np.ndarray] = []
        self._dX: list[np.ndarray] = []
        self._prev_r: np.ndarray | None = None
        self._prev_x: np.ndarray | None = None
        self._last_omega = self.omega
        self.last_cond = float("nan")

    def reset(self) -> None:
        self._dR.clear()
        self._dX.clear()
        self._prev_r = None
        self._prev_x = None
        self._last_omega = self.omega

    def _record(self, x: np.ndarray, r: np.ndarray) -> None:
        if self._prev_r is not None and self._prev_x is not None:
            self._dR.append(r - self._prev_r)
            self._dX.append(x - self._prev_x)
            if len(self._dR) > self.reuse:
                self._dR.pop(0)
                self._dX.pop(0)
        self._prev_r = r.copy()
        self._prev_x = x.copy()

    def step(self, state: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(state, dtype=np.float64)
        r = np.asarray(residual, dtype=np.float64)
        self._record(x, r)
        dx = r.copy()  # no history: plain fixed-point (omega = 1) step
        if self._dR:
            scales = np.array([max(float(np.linalg.norm(dr)), 1e-300) for dr in self._dR])
            V = np.column_stack([dr / s for dr, s in zip(self._dR, scales)])
            W = np.column_stack([dxx / s for dxx, s in zip(self._dX, scales)])
            try:
                u, sv, vt = np.linalg.svd(V, full_matrices=False)
                keep = sv > self.rcond * max(float(sv.max()), 1e-300)
                if np.any(keep):
                    # c = argmin ||V c - r||  (secant: V c ~ J^-1 r), step -W c
                    c = vt[keep].T @ ((u[:, keep].T @ r) / sv[keep])
                    if np.all(np.isfinite(c)):
                        dx = -(W @ c)
                        self.last_cond = float(sv.max() / max(float(sv[keep].min()), 1e-300))
            except np.linalg.LinAlgError:  # pragma: no cover
                dx = r.copy()
        dx = self.omega * dx
        self._last_omega = self.omega
        if self.max_step is not None:
            norm = float(np.linalg.norm(dx))
            if norm > self.max_step > 0.0:
                dx *= self.max_step / norm
                self._last_omega = self.omega * self.max_step / norm
        return dx, self._last_omega

    def describe(self) -> dict[str, Any]:
        return {"scheme": "iqn-ils", "reuse": self.reuse, "omega": self.omega}


def _make_relaxation(scheme: Scheme, omega: float, reuse: int, max_step: float | None):
    if scheme == "none":
        return FixedRelaxation(omega)
    if scheme == "aitken":
        return AitkenRelaxation(omega0=omega, omega_max=max(2.0, 2.0 * omega))
    if scheme == "iqn-ils":
        return IQNILSRelaxation(reuse=reuse, omega=omega, max_step=max_step)
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# error-sequence diagnostics
# --------------------------------------------------------------------------- #
def estimate_rho_dd(
    errors: Sequence[float], skip: int = 1, tail: int | None = None
) -> dict[str, Any]:
    """Estimate the error-propagation factor from sub-iteration increments.

    Returns ``rho_ratio`` (geometric mean of ``|e_{k+1}/e_k|`` over the tail),
    ``rho_fit`` (``exp(slope)`` of a log-linear fit), ``sigma`` (log growth per
    iteration; ``sigma > 0`` means divergence), ``r2`` and the sample count.
    """
    e = np.abs(np.asarray([float(x) for x in errors], dtype=np.float64))
    if e.size == 0:
        raise ValueError("empty error sequence")
    pos = e[e > 0.0]
    floor = (float(pos.min()) * 1e-12) if pos.size else 1e-300
    e = np.maximum(e, floor)
    idx = np.arange(e.size, dtype=np.float64)
    start = min(skip, max(e.size - 1, 0))
    sel = slice(start, None if tail is None else max(start + 2, e.size - tail))
    ee, ii = e[sel], idx[sel]
    if ee.size < 2:
        ee, ii = e, idx
    ratios = np.abs(ee[1:] / ee[:-1])
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    rho_ratio = float(np.exp(np.mean(np.log(ratios)))) if ratios.size else float("nan")
    slope, intercept = np.polyfit(ii, np.log(ee), 1)
    pred = slope * ii + intercept
    ss_res = float(np.sum((np.log(ee) - pred) ** 2))
    ss_tot = float(np.sum((np.log(ee) - np.log(ee).mean()) ** 2))
    return {
        "rho_ratio": rho_ratio,
        "rho_fit": float(math.exp(slope)),
        "sigma": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
        "n_samples": int(ee.size),
        "first": float(e[0]),
        "last": float(e[-1]),
    }


# --------------------------------------------------------------------------- #
# 1D partitioned anchor
# --------------------------------------------------------------------------- #
class PistonPartitioned1D:
    """Numerical sub-iteration of :class:`InterfaceModel1D` (pseudo-time form)."""

    def __init__(self, model: InterfaceModel1D) -> None:
        self.model = model
        self.history: list[dict[str, Any]] = []

    def sub_iterate(
        self,
        scheme: Scheme = "none",
        omega: float = 1.0,
        max_iter: int = 80,
        tol: float = 1e-12,
        reuse: int = 5,
    ) -> dict[str, Any]:
        m = self.model
        acc = _make_relaxation(scheme, omega, reuse, max_step=None)
        acc.reset()
        v = m.velocity
        increments: list[float] = []
        omegas: list[float] = []
        residuals: list[float] = []
        converged = False
        for _ in range(max_iter):
            r = (m.impulse(v) + m.external_impulse) / m.mass_s
            residuals.append(float(r))
            dx, om = acc.step(np.array([v]), np.array([r]))
            v += float(dx[0])
            increments.append(float(dx[0]))
            omegas.append(float(om))
            if abs(increments[0]) > 0.0 and abs(increments[-1]) <= tol * abs(increments[0]):
                converged = True
                break
        est = estimate_rho_dd(increments)
        record = {
            "scheme": scheme,
            "omega": omega,
            "iterations": len(increments),
            "converged": converged,
            "velocity": v,
            "exact_velocity": m.exact_velocity(),
            "velocity_error": abs(v - m.exact_velocity()),
            "increments": increments,
            "omegas": omegas,
            "residuals": residuals,
            "rho_dd_numeric": est["rho_ratio"],
            "rho_dd_fit": est["rho_fit"],
            "growth_sigma": est["sigma"],
            "fit_r2": est["r2"],
            "rho_dd_analytic": m.rho_dd(omega),
            "diverged": (not converged) and bool(est["rho_ratio"] > 1.0),
        }
        self.history.append(record)
        return record


# --------------------------------------------------------------------------- #
# 2D LBM + 1-DOF heaving body
# --------------------------------------------------------------------------- #
@dataclass
class _FluidState:
    """Cheap cloneable fluid+body state (avoids the JSON snapshot round-trip)."""

    f: Any
    center: Any
    solid: Any
    time: int

    def clone(self) -> "_FluidState":
        return _FluidState(self.f.clone(), self.center.clone(), self.solid.clone(), self.time)


class HeavingDiskPartitioned:
    """D2Q9 periodic box + heaving circular body (1 DOF) with sub-iterations.

    Body cross-section area ``V = pi R^2`` (unit depth), structural mass
    ``m_s = density_ratio * rho_f * V``, linear spring ``spring_c`` about
    ``y_eq``.  A coupling window spans ``window`` LBM steps; the fluid impulse
    is the momentum-exchange wall impulse (optionally plus node-conversion
    impulse) accumulated over that window.
    """

    def __init__(
        self,
        nx: int = 48,
        ny: int = 48,
        radius: float = 6.0,
        density_ratio: float = 1.0,
        tau: float = 0.55,
        window: int = 10,
        spring_c: float = 1.0,
        rho_f: float = 1.0,
        y_eq: float | None = None,
        initial_displacement: float = 0.0,
        initial_velocity: float = 0.0,
        impulse_channels: Sequence[str] = ("wall",),
        max_velocity: float = 0.05,
    ) -> None:
        import torch

        from .moving_boundary_2d import MovingDisk

        torch.set_num_threads(1)
        self._torch = torch
        self.nx, self.ny, self.radius = int(nx), int(ny), float(radius)
        self.density_ratio = float(density_ratio)
        self.tau = float(tau)
        self.window = int(window)
        self.spring_c = float(spring_c)
        self.rho_f = float(rho_f)
        self.impulse_channels = tuple(impulse_channels)
        self.max_velocity = float(max_velocity)
        cy = self.ny / 2.0 - 0.5 if y_eq is None else float(y_eq)
        self.y_eq = cy
        self.volume = math.pi * self.radius**2
        self.mass_s = self.density_ratio * self.rho_f * self.volume
        self.center0 = (self.nx / 2.0 - 0.5, cy + float(initial_displacement))
        self.disk = MovingDisk(
            nx=self.nx,
            ny=self.ny,
            radius=self.radius,
            center=[self.center0[0], self.center0[1]],
            velocity=[0.0, 0.0],
            fluid_velocity=[0.0, 0.0],
            tau=self.tau,
        )
        self._base = _FluidState(
            self.disk.f.clone(), self.disk.center.clone(), self.disk.solid.clone(), 0
        )
        self.velocity = float(initial_velocity)

    # -- fluid solve ------------------------------------------------------ #
    def _spawn(self, state: _FluidState, velocity: float):
        from .moving_boundary_2d import MovingDisk

        sim = MovingDisk(
            nx=self.nx,
            ny=self.ny,
            radius=self.radius,
            center=[float(state.center[0]), float(state.center[1])],
            velocity=[0.0, 0.0],
            fluid_velocity=[0.0, 0.0],
            tau=self.tau,
        )
        sim.f = state.f.clone()
        sim.center = state.center.clone()
        sim.solid = state.solid.clone()
        sim.history = []
        sim.time = int(state.time)
        sim.velocity = self._torch.tensor([0.0, float(velocity)], dtype=self._torch.float64)
        return sim

    def fluid_impulse(self, velocity: float, state: _FluidState | None = None) -> dict[str, Any]:
        """Run one coupling window with wall velocity ``velocity``.

        Returns the accumulated wall / conversion impulse (heave component) and
        the fluid state reached (``None`` if the LBM went non-physical, in which
        case ``failed=True``).
        """
        base = self._base if state is None else state
        sim = self._spawn(base, velocity)
        I_wall = 0.0
        I_conv = 0.0
        try:
            for _ in range(self.window):
                entry = sim.step()
                I_wall += float(entry["wall_impulse_on_solid"][1])
                I_conv += float(entry["conversion_impulse_on_solid"][1])
        except (RuntimeError, ValueError) as exc:
            return {
                "impulse": float("nan"),
                "wall": float("nan"),
                "conversion": float("nan"),
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "state": None,
            }
        total = I_wall + (I_conv if "conversion" in self.impulse_channels else 0.0)
        return {
            "impulse": total,
            "wall": I_wall,
            "conversion": I_conv,
            "failed": False,
            "state": _FluidState(sim.f, sim.center, sim.solid, sim.time),
        }

    # -- discrete added-mass operator ------------------------------------- #
    def measure_impulse_added_mass(
        self, amplitude: float = 2e-3, window: int | None = None
    ) -> dict[str, Any]:
        """Discrete added-mass operator of the coupling window.

        Central difference of the accumulated impulse w.r.t. the wall velocity:
        ``K = -(I(+a) - I(-a)) / (2a)``; ``c_a = K / (rho_f V)``.
        """
        win_saved = self.window
        if window is not None:
            self.window = int(window)
        try:
            plus = self.fluid_impulse(self.velocity + amplitude)
            minus = self.fluid_impulse(self.velocity - amplitude)
            zero = self.fluid_impulse(self.velocity)
        finally:
            self.window = win_saved
        if window is not None and not (plus["failed"] or minus["failed"]):
            plus["window"] = minus["window"] = zero["window"] = int(window)
        if plus["failed"] or minus["failed"]:
            raise RuntimeError("LBM failed during added-mass identification")
        K = -(plus["impulse"] - minus["impulse"]) / (2.0 * amplitude)
        return {
            "K": float(K),
            "c_a": float(K / (self.rho_f * self.volume)),
            "added_mass": float(K),
            "amplitude": float(amplitude),
            "impulse_zero": float(zero["impulse"]),
            "window": int(window if window is not None else self.window),
            "volume": float(self.volume),
            "nonlinearity": float(
                abs((plus["impulse"] + minus["impulse"] - 2.0 * zero["impulse"]))
                / max(abs(plus["impulse"] - minus["impulse"]), 1e-30)
            ),
        }

    # -- sub-iteration (pseudo-time / incremental form) -------------------- #
    def sub_iterate(
        self,
        scheme: Scheme = "none",
        omega: float = 1.0,
        max_iter: int = 40,
        tol: float = 1e-9,
        reuse: int = 5,
        state: _FluidState | None = None,
        velocity: float | None = None,
    ) -> dict[str, Any]:
        """Solve one coupling window by sub-iteration (pseudo-time form).

        ``v^{k+1} = v^k + omega_k (I(v^k) + I_ext)/m_s`` with the fluid re-solved
        from the window-start state at every iterate.  Returns the increment
        sequence and the estimated amplification factor.
        """
        base = self._base if state is None else state
        v0 = self.velocity if velocity is None else float(velocity)
        y_n = float(base.center[1])
        ext = -self.spring_c * (y_n - self.y_eq) * self.window
        acc = _make_relaxation(scheme, omega, reuse, max_step=2.0 * self.max_velocity)
        acc.reset()
        v = v0
        increments: list[float] = []
        omegas: list[float] = []
        impulses: list[float] = []
        converged = False
        failed = False
        error = None
        accepted = None
        clipped = False
        clip_index: int | None = None
        for _ in range(max_iter):
            v_run = float(np.clip(v, -self.max_velocity, self.max_velocity))
            if v_run != v:
                clipped = True
            res = self.fluid_impulse(v_run, state=base)
            if res["failed"]:
                failed = True
                error = res.get("error")
                break
            impulse = res["impulse"]
            impulses.append(float(impulse))
            r = np.array([(impulse + ext) / self.mass_s])
            dx, om = acc.step(np.array([v]), r)
            dv = float(dx[0])
            v += dv
            increments.append(dv)
            omegas.append(float(om))
            accepted = res["state"]
            if clip_index is None and abs(v) > self.max_velocity:
                clip_index = len(increments)
            if increments[0] != 0.0 and abs(dv) <= tol * abs(increments[0]):
                converged = True
                break
        est = (
            estimate_rho_dd(increments)
            if increments
            else {
                "rho_ratio": float("nan"),
                "rho_fit": float("nan"),
                "sigma": float("nan"),
                "r2": float("nan"),
            }
        )
        return {
            "scheme": scheme,
            "omega": omega,
            "window": self.window,
            "density_ratio": self.density_ratio,
            "mass_s": float(self.mass_s),
            "iterations": len(increments),
            "converged": bool(converged),
            "lbm_failed": bool(failed),
            "velocity_clipped": bool(clipped),
            "error": error,
            "velocity": float(v),
            "final_impulse": float(impulses[-1]) if impulses else float("nan"),
            "increments": increments,
            "omegas": omegas,
            "rho_dd_numeric": float(est["rho_ratio"]),
            "rho_dd_fit": float(est["rho_fit"]),
            "rho_dd_preclip": float(
                estimate_rho_dd(increments[: clip_index or len(increments)])["rho_ratio"]
            )
            if increments
            else float("nan"),
            "growth_sigma_preclip": float(
                estimate_rho_dd(increments[: clip_index or len(increments)])["sigma"]
            )
            if increments
            else float("nan"),
            "clip_index": clip_index,
            "growth_sigma": float(est["sigma"]),
            "fit_r2": float(est["r2"]),
            "diverged": bool(
                (not converged) and (failed or clipped or (est["rho_ratio"] or 0.0) >= 1.0)
            ),
            "_state": accepted,
        }

    # -- forced harmonic motion: added mass and damping -------------------- #
    def measure_forced_added_mass(
        self,
        omega: float,
        displacement_amplitude: float = 0.2,
        n_periods: float = 3.0,
        state: _FluidState | None = None,
    ) -> dict[str, Any]:
        """Added mass and damping from prescribed harmonic heave.

        The body position follows ``y(t) = y0 + A sin(omega t)``; the velocity is
        piecewise constant per coupling window.  The accumulated impulse is
        fitted against the prescribed acceleration and velocity,

            ``I_n ~ -A_add * a_n + B_eff * u_n + const``

        which separates the inertial (added-mass) from the resistive (damping)
        response.  Returns ``added_mass``, ``c_a = A_add/(rho_f V)``,
        ``damping`` and the fit quality.
        """
        base = self._base if state is None else state
        y0 = float(base.center[1])
        n_win = max(int(round(n_periods * 2.0 * math.pi / (omega * self.window))), 8)
        t_mid: list[float] = []
        acc: list[float] = []
        vel: list[float] = []
        imp: list[float] = []
        cur = base
        for n in range(n_win):
            t0 = n * self.window
            t1 = (n + 1) * self.window
            y_t0 = y0 + displacement_amplitude * math.sin(omega * t0)
            y_t1 = y0 + displacement_amplitude * math.sin(omega * t1)
            u = (y_t1 - y_t0) / self.window
            if abs(u) > self.max_velocity:
                return {
                    "failed": True,
                    "error": "prescribed velocity above limit",
                    "c_a": float("nan"),
                }
            res = self.fluid_impulse(u, state=cur)
            if res["failed"]:
                return {"failed": True, "error": res.get("error"), "c_a": float("nan")}
            cur = res["state"]
            tm = 0.5 * (t0 + t1)
            t_mid.append(tm)
            acc.append(-displacement_amplitude * omega**2 * math.sin(omega * tm))
            vel.append(u)
            imp.append(res["impulse"])
        A = np.vstack([np.asarray(acc), np.asarray(vel), np.ones(len(t_mid))]).T
        coef, res_, rank, _ = np.linalg.lstsq(A, np.asarray(imp), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((np.asarray(imp) - pred) ** 2))
        ss_tot = float(np.sum((np.asarray(imp) - np.mean(imp)) ** 2))
        added_mass = float(-coef[0])
        damping = float(coef[1])
        return {
            "failed": bool(rank < 3),
            "added_mass": added_mass,
            "c_a": float(added_mass / (self.rho_f * self.volume)),
            "damping_impulse_per_velocity": damping,
            "omega": float(omega),
            "displacement_amplitude": float(displacement_amplitude),
            "n_windows": int(n_win),
            "fit_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
            "density_ratio": float(self.density_ratio),
            "volume": float(self.volume),
            "mass_s": float(self.mass_s),
        }

    # -- time-accurate scheme (free decay) -------------------------------- #
    def run_time_accurate(
        self,
        n_windows: int,
        scheme: Scheme = "none",
        omega: float = 1.0,
        max_iter: int = 40,
        tol: float = 1e-10,
        reuse: int = 5,
    ) -> dict[str, Any]:
        """Time-accurate staggered coupling over ``n_windows`` coupling steps.

        Structure solve ``v^{n+1} = v^n + (I(v) + I_ext)/m_s`` driven to
        convergence by the residual-form sub-iteration
        ``v^{k+1} = v^k + omega r^k`` with
        ``r^k = (I(v^k) + I_ext)/m_s - (v^k - v^n)``.
        """
        t: list[float] = []
        y_hist: list[float] = []
        v_hist: list[float] = []
        sub_iters: list[int] = []
        failed = False
        error = None
        for n in range(n_windows):
            base = self._base
            y_n = float(base.center[1])
            v_n = self.velocity
            ext = -self.spring_c * (y_n - self.y_eq) * self.window
            acc = _make_relaxation(scheme, omega, reuse, max_step=2.0 * self.max_velocity)
            acc.reset()
            v = v_n
            k_used = max_iter
            for k in range(max_iter):
                res = self.fluid_impulse(v, state=base)
                if res["failed"]:
                    failed, error = True, res.get("error")
                    break
                r = np.array([(res["impulse"] + ext) / self.mass_s - (v - v_n)])
                dx, _ = acc.step(np.array([v]), r)
                dv = float(dx[0])
                v += dv
                if abs(dv) <= tol * max(abs(v) + 1e-3 * self.max_velocity, 1e-30):
                    k_used = k + 1
                    break
            if failed:
                break
            commit = self.fluid_impulse(v, state=base)
            if commit["failed"]:
                failed, error = True, commit.get("error")
                break
            self._base = commit["state"]
            self.velocity = float(v)
            t.append(float(n + 1) * self.window)
            y_hist.append(float(self._base.center[1]) - self.y_eq)
            v_hist.append(float(v))
            sub_iters.append(k_used)
            if abs(v) > 40.0 * self.max_velocity:
                error = "velocity blow-up"
                failed = True
                break
        return {
            "t": t,
            "y": y_hist,
            "v": v_hist,
            "lbm_failed": failed,
            "error": error,
            "n_windows": len(t),
            "n_sub_iterations": int(sum(sub_iters)),
            "max_sub_iterations": int(max(sub_iters)) if sub_iters else 0,
            "converged_all": bool(all(si < max_iter for si in sub_iters)),
        }


def potential_added_mass(
    nx: int,
    ny: int,
    radius: float,
    center: tuple[float, float] | None = None,
    rho: float = 1.0,
    d_scale: float = 1.0,
) -> dict[str, Any]:
    """Potential-flow added mass of a circular body in a periodic box.

    Solves the unit-velocity perturbation potential

        lap(psi) = 0 in the fluid,  dpsi/dn = n.y on the body,  psi periodic,

    by a finite-volume Neumann problem on the same cell mask used by the LBM
    moving boundary, and evaluates the added mass from the kinetic energy

        A = rho * int_fluid |grad psi|^2 dV        (=> rho * pi R^2 unbounded)

    ``d_scale`` multiplies all lengths (for unit-system changes only; the
    returned ``c_a = A/(rho*pi*R^2 d_scale^2)`` is scale invariant).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nx, ny = int(nx), int(ny)
    if center is None:
        center = (nx / 2.0 - 0.5, ny / 2.0 - 0.5)
    cx, cy = float(center[0]), float(center[1])
    Y, X = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    dx = (X - cx + nx / 2.0) % nx - nx / 2.0
    dy = (Y - cy + ny / 2.0) % ny - ny / 2.0
    solid = dx**2 + dy**2 <= radius**2
    fluid = ~solid
    index = -np.ones((ny, nx), dtype=np.int64)
    index[fluid] = np.arange(int(fluid.sum()), dtype=np.int64)
    n = int(fluid.sum())
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n, dtype=np.float64)
    # outward (fluid -> body) normal flux source on body faces
    for j in range(ny):
        for i in range(nx):
            if not fluid[j, i]:
                continue
            k = int(index[j, i])
            n_nb = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii = (i + di) % nx
                jj = (j + dj) % ny
                if fluid[jj, ii]:
                    rows.append(k)
                    cols.append(int(index[jj, ii]))
                    vals.append(1.0)
                    n_nb += 1
                else:
                    # prescribed flux through the body face (sign irrelevant for
                    # |grad psi|^2); enters the balance as -q_body
                    rhs[k] += dj
            rows.append(k)
            cols.append(k)
            vals.append(-float(n_nb))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    psi_flat = spla.spsolve(A.tocsc(), rhs)
    psi = np.zeros((ny, nx), dtype=np.float64)
    psi[fluid] = psi_flat

    def face_flux(j, i, dj, di):
        jj, ii = (j + dj) % ny, (i + di) % nx
        if fluid[jj, ii] and fluid[j, i]:
            return psi[jj, ii] - psi[j, i]
        if fluid[j, i]:
            return -(dj if dj != 0 else 0.0)
        if fluid[jj, ii]:
            return dj if dj != 0 else 0.0
        return 0.0

    energy = 0.0
    for j in range(ny):
        for i in range(nx):
            if not fluid[j, i]:
                continue
            qxp = face_flux(j, i, 0, 1)
            qxm = -face_flux(j, i, 0, -1)
            qyp = face_flux(j, i, 1, 0)
            qym = -face_flux(j, i, -1, 0)
            gx = 0.5 * (qxp + qxm)
            gy = 0.5 * (qyp + qym)
            energy += gx * gx + gy * gy
    volume = float(fluid.sum())
    added = rho * energy
    ref = rho * math.pi * radius**2
    return {
        "added_mass": float(added),
        "c_a": float(added / ref),
        "body_area": float(solid.sum()),
        "fluid_area": volume,
        "radius": float(radius),
        "nx": nx,
        "ny": ny,
        "blockage": float(2.0 * radius / nx),
    }


def identify_added_mass_damping(
    t: Sequence[float],
    y: Sequence[float],
    mass_s: float,
    stiffness: float,
) -> dict[str, Any]:
    """Identify ``(omega, zeta, A, B)`` from a heave free-decay record.

    ``omega`` from a linear fit of zero-crossing times (each crossing is half a
    period), ``zeta`` from the log-decrement of successive local extrema,
    ``A = C/omega^2 - m_s`` and ``B = 2 zeta (m_s + A) omega``.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if tt.size < 8:
        raise ValueError("need at least 8 samples")
    sign = np.sign(yy)
    cross = np.where(np.diff(sign) != 0)[0]
    crossings = []
    for i in cross:
        y0, y1 = yy[i], yy[i + 1]
        if y1 == y0:
            continue
        crossings.append(tt[i] + (-y0 / (y1 - y0)) * (tt[i + 1] - tt[i]))
    crossings = np.asarray(crossings, dtype=np.float64)
    if crossings.size < 4:
        raise ValueError("not enough zero crossings for frequency identification")
    k = np.arange(crossings.size, dtype=np.float64)
    design = np.vstack([np.ones_like(k), k]).T
    coef = np.linalg.lstsq(design, crossings, rcond=None)[0]
    half_period = float(coef[1])
    omega = math.pi / half_period
    resid = crossings - design @ coef
    crossing_rms = float(np.sqrt(np.mean(resid**2)))
    # log decrement between successive local maxima of |y| (half period apart)
    ay = np.abs(yy)
    peaks = [
        float(ay[i])
        for i in range(1, ay.size - 1)
        if ay[i] >= ay[i - 1] and ay[i] > ay[i + 1] and ay[i] > 0.0
    ]
    amps = peaks
    zeta = float("nan")
    if len(amps) >= 3:
        logs = np.log(np.asarray(amps[:-1]) / np.asarray(amps[1:]))
        keep = logs[np.isfinite(logs) & (np.abs(logs) < 1.0)]
        if keep.size:
            zeta = float(np.mean(keep) / math.pi)  # delta_half / pi = zeta
    zeta = max(zeta, 0.0) if math.isfinite(zeta) else float("nan")
    added_mass = stiffness / omega**2 - mass_s
    damping = 2.0 * zeta * (mass_s + added_mass) * omega if math.isfinite(zeta) else float("nan")
    return {
        "omega": float(omega),
        "period": float(2.0 * math.pi / omega),
        "half_period": half_period,
        "n_crossings": int(crossings.size),
        "crossing_fit_rms": crossing_rms,
        "zeta": float(zeta),
        "added_mass": float(added_mass),
        "added_mass_ratio": float(added_mass / mass_s) if mass_s else float("nan"),
        "damping": float(damping),
        "amplitude_first": float(amps[0]) if amps else float("nan"),
        "amplitude_last": float(amps[-1]) if amps else float("nan"),
        "n_extrema": len(amps),
    }


def scan_stability(
    make_case,
    density_ratios: Sequence[float],
    omegas: Sequence[float],
    scheme: Scheme = "none",
    max_iter: int = 40,
    tol: float = 1e-9,
    reuse: int = 5,
) -> dict[str, Any]:
    """Scan the ``(rho_s/rho_f, omega)`` plane for the convergence domain.

    ``make_case(density_ratio)`` must return a fresh
    :class:`HeavingDiskPartitioned`.  For every ratio the largest ``omega`` on
    the (ascending) grid that still converges is recorded as the numerical
    stability boundary.
    """
    records: list[dict[str, Any]] = []
    for ratio in density_ratios:
        case = make_case(float(ratio))
        row = []
        for om in omegas:
            rec = case.sub_iterate(
                scheme=scheme, omega=float(om), max_iter=max_iter, tol=tol, reuse=reuse
            )
            row.append(rec)
            records.append(
                {
                    "density_ratio": float(ratio),
                    "omega": float(om),
                    "scheme": scheme,
                    "converged": bool(rec["converged"]),
                    "iterations": int(rec["iterations"]),
                    "rho_dd_numeric": float(rec["rho_dd_numeric"]),
                    "growth_sigma": float(rec["growth_sigma"]),
                    "lbm_failed": bool(rec["lbm_failed"]),
                }
            )
        boundary = 0.0
        for rec in row:
            if rec["converged"]:
                boundary = float(rec["omega"])
            else:
                break
        for entry in records:
            if entry["density_ratio"] == float(ratio):
                entry["stable_boundary_omega"] = boundary
    return {
        "records": records,
        "densities": list(map(float, density_ratios)),
        "omegas": list(map(float, omegas)),
    }
