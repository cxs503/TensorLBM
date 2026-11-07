"""Linear ship-maneuvering (horizontal-plane) dynamics: steady turn and transient.

This module implements the classic *linearized sway–yaw* maneuvering model used
in ship hydrodynamics / marine control (Abkowitz 1964; Nomoto et al. 1957;
Fossen 2011, ch. 7).  It is the maneuvering-plane analogue of the seakeeping
6-DOF :mod:`tensorlbm.rigid_body_6dof` stack and is likewise a **caller-driven
linear tool**: all hydrodynamic derivatives are inputs.

Body axes
---------
``x`` forward, ``y`` to starboard, ``z`` down; yaw rate ``r`` positive to
starboard, rudder ``delta`` positive to starboard.  The equilibria used here
are the *steady turn* of the well-known linear model

.. math::

    m\\,\\dot v - Y_v\\,v - (Y_r - m\\,U)\\,r = Y_\\delta\\,\\delta
    \\\\
    I_z\\,\\dot r - N_v\\,v - N_r\\,r = N_\\delta\\,\\delta

with optional added-mass (acceleration) derivatives
``Y_vdot, Y_rdot, N_vdot, N_rdot`` folded into the effective mass matrix.

For the **steady turn** (``vdot = rdot = 0``) the fixed point is an exact 2x2
linear solve; the turning radius is ``R = U / r`` and the drift angle is
``beta = atan2(-v, U)`` (positive when the velocity vector lies to port of the
heading, i.e. outward in a starboard turn).  The Nomoto gain is
``K = r / delta``.

References
----------
* Abkowitz, M.A. (1964). *Lectures on Ship Hydrodynamics and Stability*.
* Nomoto, K., Taguchi, T., Honda, K., Hirano, S. (1957). On the steering
  qualities of ships. *J. Zosen Kiokai*, 101, 41-52.
* Fossen, T.I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion
  Control*, Wiley, ch. 7 (SNAME prime system, linearized sway-yaw).
* Lewis, E.V. (ed.) (1989). *Principles of Naval Architecture*, Vol. III
  (Motions in Waves and Controllability), ch. 9 -- steady-turn formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "ManeuveringModel",
    "SteadyTurn",
    "simulate_turn",
    "analytic_transient",
    "NOMINAL_TANKER_PRIME",
    "prime_to_dimensional",
    "ZigZagResult",
    "simulate_zigzag",
    "analytic_zigzag",
    "imo_msc137_zigzag_limits",
    "check_imo_msc137_zigzag",
    "IMO_MSC137_ZIGZAG_LIMITS",
    "IMO_MSC137_SOURCE",
]


# --------------------------------------------------------------------------- #
# helper: non-dimensional (SNAME prime) -> dimensional conversion
# --------------------------------------------------------------------------- #
def prime_to_dimensional(
    prime: dict,
    L: float,
    d: float,
    U: float,
    rho: float = 1000.0,
) -> dict:
    """Convert SNAME-prime lateral derivatives to dimensional form.

    Prime conventions (Fossen 2011 eq. 7.10-7.16):
    ``v' = v/U``, ``r' = r L / U``, ``Y' = Y / (1/2 rho U^2 L d)``,
    ``N' = N / (1/2 rho U^2 L^2 d)``.  Hence the dimensional factors are::

        Y_v = Y'_v * (1/2) rho U L   d      Y_r = Y'_r * (1/2) rho U L^2 d
        N_v = N'_v * (1/2) rho U L^2 d      N_r = N'_r * (1/2) rho U L^3 d
        Y_d = Y'_d * (1/2) rho U^2 L d      N_d = N'_d * (1/2) rho U^2 L^2 d
    """
    k1 = 0.5 * rho * U * L * d
    k2 = 0.5 * rho * U * L**2 * d
    k3 = 0.5 * rho * U * L**3 * d
    k4 = 0.5 * rho * U**2 * L * d
    k5 = 0.5 * rho * U**2 * L**2 * d
    return {
        "Y_v": prime["Y_v"] * k1,
        "Y_r": prime["Y_r"] * k2,
        "N_v": prime["N_v"] * k2,
        "N_r": prime["N_r"] * k3,
        "Y_delta": prime["Y_delta"] * k4,
        "N_delta": prime["N_delta"] * k5,
    }


# A representative large-tanker (KVLCC2-class) *linear* derivative set chosen to
# give a directionally-stable ship with a realistic steady turn.  Magnitudes are
# order-of-magnitude consistent with published PMM/MMG data (Yasukawa &
# Yoshimura 2015; SIMMAN 2008); they are representative, NOT a digit-exact
# reprint.  See the benchmark README for the calibrated R/L and drift angle.
NOMINAL_TANKER_PRIME = {
    "Y_v": -0.65,
    "Y_r": 0.05,
    "N_v": -0.095,
    "N_r": -0.08,
    "Y_delta": -0.16,  # rudder side force to port for delta > 0 (starboard)
    "N_delta": 0.04,  # bow-to-starboard yaw couple for delta > 0
}


@dataclass
class SteadyTurn:
    """Result of the exact steady-turn (fixed-point) solve."""

    delta: float
    v: float
    r: float
    R: float
    R_over_L: float
    beta_deg: float
    K: float
    det: float
    stable: bool
    speed: float

    def as_dict(self) -> dict:
        return {
            "delta_rad": self.delta,
            "delta_deg": math.degrees(self.delta),
            "v": self.v,
            "r": self.r,
            "R": self.R,
            "R_over_L": self.R_over_L,
            "beta_deg": self.beta_deg,
            "K": self.K,
            "det": self.det,
            "directionally_stable": self.stable,
            "speed": self.speed,
        }


@dataclass
class ManeuveringModel:
    """Linear sway-yaw maneuvering model of one ship at forward speed ``U``.

    All derivatives are dimensional (N, N.m per m/s and per rad/s).  Added-mass
    (acceleration) derivatives default to zero; they do not enter the steady-turn
    fixed point but do shape the transient.
    """

    L: float
    d: float
    U: float
    mass: float
    inertia_z: float
    Y_v: float
    Y_r: float
    N_v: float
    N_r: float
    Y_delta: float
    N_delta: float
    x_G: float = 0.0
    rho: float = 1000.0
    Y_vdot: float = 0.0
    Y_rdot: float = 0.0
    N_vdot: float = 0.0
    N_rdot: float = 0.0
    Y_vv: float = 0.0
    N_vv: float = 0.0
    name: str = "tanker"

    # ---------------- matrices ------------------------------------------- #
    def mass_matrix(self) -> torch.Tensor:
        """Effective (2x2) mass matrix including added mass."""
        return torch.tensor(
            [
                [self.mass - self.Y_vdot, self.mass * self.x_G - self.Y_rdot],
                [self.mass * self.x_G - self.N_vdot, self.inertia_z - self.N_rdot],
            ],
            dtype=torch.float64,
        )

    def damping_matrix(self) -> torch.Tensor:
        """Damping/coriolis matrix ``A`` in ``M qdot = A q + B delta``."""
        return torch.tensor(
            [
                [self.Y_v, self.Y_r - self.mass * self.U],
                [self.N_v, self.N_r - self.mass * self.x_G * self.U],
            ],
            dtype=torch.float64,
        )

    def input_vector(self, delta: float) -> torch.Tensor:
        return torch.tensor([self.Y_delta * delta, self.N_delta * delta], dtype=torch.float64)

    def nonlinear_force(self, v: float) -> torch.Tensor:
        """Cross-flow (``|v| v``) lateral force / yaw moment."""
        return torch.tensor([self.Y_vv * abs(v) * v, self.N_vv * abs(v) * v], dtype=torch.float64)

    def has_nonlinear(self) -> bool:
        return self.Y_vv != 0.0 or self.N_vv != 0.0

    # ---------------- exact steady turn ---------------------------------- #
    def steady_turn(self, delta: float) -> SteadyTurn:
        """Exact fixed-point solution of the (possibly nonlinear) sway-yaw model."""
        A = self.damping_matrix()
        b = self.input_vector(delta)
        if self.has_nonlinear():
            v, r = self._steady_turn_nonlinear(delta)
        else:
            # M qdot = A q + b ; steady state solves A q = -b
            q = torch.linalg.solve(A, -b)
            v, r = float(q[0]), float(q[1])
        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        R = self.U / r if r != 0.0 else float("inf")
        beta = math.degrees(math.atan2(-v, self.U))
        K = r / delta if delta != 0.0 else float("nan")
        ev = torch.linalg.eigvals(torch.linalg.solve(self.mass_matrix(), A))
        stable = bool(torch.all(ev.real < 0).item())
        return SteadyTurn(
            delta=delta,
            v=v,
            r=r,
            R=R,
            R_over_L=R / self.L,
            beta_deg=beta,
            K=K,
            det=det,
            stable=stable,
            speed=math.hypot(self.U, v),
        )

    def _steady_turn_nonlinear(self, delta: float) -> tuple[float, float]:
        """Solve the nonlinear steady-turn equilibrium by bisection on v.

        Steady state satisfies ``Y_v v + (Y_r-mU) r + Y_vv|v|v + Y_d delta = 0``
        and ``N_v v + N_r r + N_vv|v|v + N_d delta = 0``.  ``r`` is eliminated
        from the second equation, leaving a scalar equation ``f(v)=0``.
        """
        Y_v, Y_r = self.Y_v, self.Y_r
        N_v, N_r, N_d = self.N_v, self.N_r, self.N_delta
        mU = self.mass * self.U
        Y_d = self.Y_delta

        def r_of(v: float) -> float:
            return -(N_v * v + self.N_vv * abs(v) * v + N_d * delta) / N_r

        def f(v: float) -> float:
            return Y_v * v + (Y_r - mU) * r_of(v) + self.Y_vv * abs(v) * v + Y_d * delta

        lo, hi = -3.0 * self.U, 3.0 * self.U
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0.0:  # expand the bracket if needed
            lo, hi = -10.0 * self.U, 10.0 * self.U
            flo, fhi = f(lo), f(hi)
            if flo * fhi > 0.0:
                raise RuntimeError("could not bracket the nonlinear steady-turn root")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
            if hi - lo < 1e-12 * self.U:
                break
        v = 0.5 * (lo + hi)
        return v, r_of(v)

    def nomoto_K(self) -> float:
        """Nomoto gain ``K = r/delta`` from the raw derivatives (closed form)."""
        den = self.Y_v * self.N_r - self.N_v * (self.Y_r - self.mass * self.U)
        num = self.N_v * self.Y_delta - self.Y_v * self.N_delta
        return num / den

    def modes(self) -> dict:
        """Eigenvalues / eigenvectors / time constants of ``M^-1 A``."""
        M = self.mass_matrix()
        A = self.damping_matrix()
        evals, evecs = torch.linalg.eig(torch.linalg.solve(M, A))
        ev = evals.real
        taus = [-1.0 / float(e) for e in ev if e < 0]
        return {
            "eigenvalues": [float(e) for e in ev],
            "time_constants": taus,
            "eigenvectors": evecs,
        }


# --------------------------------------------------------------------------- #
# time integration
# --------------------------------------------------------------------------- #
def _rhs(model: ManeuveringModel, q: torch.Tensor, delta: float) -> torch.Tensor:
    """Return ``[vdot, rdot]`` for state ``q = [v, r]``."""
    M = model.mass_matrix()
    A = model.damping_matrix()
    v = float(q[0])
    return torch.linalg.solve(M, A @ q + model.input_vector(delta) + model.nonlinear_force(v))


def simulate_turn(
    model: ManeuveringModel,
    delta: float,
    dt: float,
    n_steps: int,
    integrator: str = "semi_implicit_euler",
    ramp_time: float = 0.0,
) -> dict:
    """Integrate the maneuvering equations from rest for ``n_steps`` of ``dt``.

    State is ``[v, r, x, y, psi]`` (sway velocity, yaw rate, surge/side
    displacement, heading).  Kinematics use the caller-supplied forward speed
    ``model.U``:
    ``xdot = U cos psi - v sin psi``, ``ydot = U sin psi + v cos psi``,
    ``psidot = r``.

    ``ramp_time > 0`` linearly ramps the rudder from 0 to ``delta`` (a fairer
    approximation of a real rudder order); ``ramp_time = 0`` is a step input.

    Returns a dict of histories (python floats for v/r; tensors for x/y).
    """
    # Model accessors are invoked for their validation side effect; the
    # coefficients themselves are unused on this nonlinear time-marching path.
    model.mass_matrix()
    model.damping_matrix()

    q = torch.zeros(2, dtype=torch.float64)  # v, r
    pos = torch.zeros(2, dtype=torch.float64)  # x, y
    psi = 0.0

    v_hist: list[float] = []
    r_hist: list[float] = []
    psi_hist: list[float] = []
    x_hist: list[float] = []
    y_hist: list[float] = []

    def _delta_at(t: float) -> float:
        if ramp_time > 0.0 and t < ramp_time:
            return delta * (t / ramp_time)
        return delta

    for step in range(n_steps):
        t = step * dt
        d_now = _delta_at(t)

        # record the state at time ``t`` (BEFORE the step update) so that
        # history[k] corresponds exactly to t = k*dt (and history[0] == IC).
        v_now, r_now = float(q[0]), float(q[1])
        v_hist.append(v_now)
        r_hist.append(r_now)
        psi_hist.append(psi)
        x_hist.append(float(pos[0]))
        y_hist.append(float(pos[1]))

        if integrator == "semi_implicit_euler":
            # explicit velocity update, then position uses the NEW velocity
            # (mirrors tensorlbm.rigid_body_6dof.cummins_step).
            qd = _rhs(model, q, d_now)
            q = q + qd * dt
            v, r = float(q[0]), float(q[1])
            psi_new = psi + r * dt
            x_new = float(pos[0]) + (model.U * math.cos(psi) - v * math.sin(psi)) * dt
            y_new = float(pos[1]) + (model.U * math.sin(psi) + v * math.cos(psi)) * dt
            pos = torch.tensor([x_new, y_new], dtype=torch.float64)
            psi = psi_new
        elif integrator == "rk4":

            def f(state: torch.Tensor) -> torch.Tensor:
                return _rhs(model, state, d_now)

            k1 = f(q)
            k2 = f(q + 0.5 * dt * k1)
            k3 = f(q + 0.5 * dt * k2)
            k4 = f(q + dt * k3)
            q = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            v, r = float(q[0]), float(q[1])
            # trapezoidal kinematics with midpoint heading
            psi_mid = psi + 0.5 * r * dt
            x_new = float(pos[0]) + (model.U * math.cos(psi_mid) - v * math.sin(psi_mid)) * dt
            y_new = float(pos[1]) + (model.U * math.sin(psi_mid) + v * math.cos(psi_mid)) * dt
            pos = torch.tensor([x_new, y_new], dtype=torch.float64)
            psi = psi + r * dt
        else:
            raise ValueError(f"unknown integrator {integrator!r}")

    # append the final state at t = n_steps*dt so the steady window can use it
    v_hist.append(float(q[0]))
    r_hist.append(float(q[1]))
    psi_hist.append(psi)
    x_hist.append(float(pos[0]))
    y_hist.append(float(pos[1]))

    return {
        "t": [step * dt for step in range(n_steps + 1)],
        "v": v_hist,
        "r": r_hist,
        "psi": psi_hist,
        "x": x_hist,
        "y": y_hist,
        "v_final": v_hist[-1],
        "r_final": r_hist[-1],
        "integrator": integrator,
        "dt": dt,
        "n_steps": n_steps,
    }


def analytic_transient(model: ManeuveringModel, delta: float, t) -> tuple:
    """Exact modal solution of the linear sway-yaw ODE from rest.

    ``q(t) = q_ss + c1 psi1 exp(l1 t) + c2 psi2 exp(l2 t)`` with constants fixed
    by ``q(0) = 0``.  Returns ``(v, r)`` arrays.
    """
    import numpy as np

    M = model.mass_matrix().numpy()
    A = model.damping_matrix().numpy()
    b = model.input_vector(delta).numpy()
    L = np.linalg.solve(M, A)
    _c = np.linalg.solve(M, b)  # particular solution; mode analysis below uses q_ss
    q_ss = np.linalg.solve(A, -b)
    evals, evecs = np.linalg.eig(L)
    coeff = np.linalg.solve(evecs, -q_ss)
    tt = np.asarray(t, dtype=float)
    q = np.array([q_ss for _ in tt])
    for lam, phi, cc in zip(evals, evecs.T, coeff):
        q = q + np.outer(np.exp(lam * tt) * cc, phi)
    return q[:, 0], q[:, 1]


# --------------------------------------------------------------------------- #
# zig-zag (overshoot) manoeuvre: closed-loop helm + IMO MSC.137(76) gate
# --------------------------------------------------------------------------- #
# The steady turn above is an *open-loop* manoeuvre (rudder held fixed).  The
# zig-zag manoeuvre is the standard closed-loop course-keeping test: the helm is
# ordered to ``+delta`` until the heading deviation reaches ``+psi_zz``, then
# reversed to ``-delta``, etc., and the quality index is the **overshoot angle**
# -- how far the heading travels past the reversal heading before the yaw rate
# changes sign.  Two independent routes are provided:
#
#   * :func:`simulate_zigzag` -- event-driven time integration of the same ODE
#     used by :func:`simulate_turn`, with the helm reversals located *inside* the
#     step by bisection (no switch-time quantisation to ``dt``);
#   * :func:`analytic_zigzag` -- the exact piecewise-modal solution: within each
#     rudder segment the ODE is linear time-invariant, so ``q(t)`` and the
#     heading ``psi(t)`` are closed-form sums of two exponentials, and the
#     switch/peak times are roots of those closed forms.
#
# Route (1) is numerical, route (2) is analytic; agreement between them is the
# verification criterion used by ``benchmarks/verified/maneuvering_zigzag``.

#: IMO Resolution MSC.137(76) annex 2.3 zig-zag overshoot-angle limits [deg].
#: ``L/V`` is the ship-length / approach-speed ratio in seconds.
IMO_MSC137_ZIGZAG_LIMITS = {
    "10/10": {"first_lt10s": 10.0, "first_ge10s": 20.0},
    "20/20": {"first": 25.0, "second": 40.0},
}
IMO_MSC137_SOURCE = "IMO Res. MSC.137(76) annex 2.3 (zig-zag overshoot angles)"


@dataclass
class ZigZagResult:
    """Outcome of a zig-zag manoeuvre (closed-loop helm)."""

    delta_deg: float
    heading_deg: float
    route: str
    dt: float = float("nan")
    events: list = field(default_factory=list)
    overshoots_deg: list = field(default_factory=list)
    reversal_times: list = field(default_factory=list)
    peak_times: list = field(default_factory=list)
    max_heading_deg: float = 0.0
    terminated: str = "ok"
    history: dict = field(default_factory=dict)

    @property
    def first_overshoot_deg(self) -> float:
        return self.overshoots_deg[0] if self.overshoots_deg else float("nan")

    @property
    def second_overshoot_deg(self) -> float:
        return self.overshoots_deg[1] if len(self.overshoots_deg) > 1 else float("nan")

    def as_dict(self) -> dict:
        return {
            "delta_deg": self.delta_deg,
            "heading_deg": self.heading_deg,
            "route": self.route,
            "dt": self.dt,
            "first_overshoot_deg": self.first_overshoot_deg,
            "second_overshoot_deg": self.second_overshoot_deg,
            "reversal_times": self.reversal_times,
            "peak_times": self.peak_times,
            "max_heading_deg": self.max_heading_deg,
            "event_count": len(self.events),
            "events": self.events,
            "terminated": self.terminated,
        }


def _step_sway_yaw(
    model: ManeuveringModel,
    q: torch.Tensor,
    psi: float,
    dt: float,
    delta: float,
    integrator: str,
) -> tuple[torch.Tensor, float]:
    """Advance ``(q=[v, r], psi)`` by ``dt`` with a constant rudder ``delta`` [rad].

    Heading is integrated with the matching quadrature on the yaw rate
    (consistent Euler for the Euler path, full RK4 stage quadrature with ``psi``
    carried as a third state for the RK4 path), so ``psi`` -- the quantity the helm logic acts on -- carries the same
    order of accuracy as ``q`` and does not silently dominate the error.
    """
    if integrator == "semi_implicit_euler":
        qd = _rhs(model, q, delta)
        q_new = q + qd * dt
        psi_new = psi + dt * float(q_new[1])  # mirrors simulate_turn
    elif integrator == "rk4":
        k1 = _rhs(model, q, delta)
        k2 = _rhs(model, q + 0.5 * dt * k1, delta)
        k3 = _rhs(model, q + 0.5 * dt * k2, delta)
        k4 = _rhs(model, q + dt * k3, delta)
        q_new = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # psi integrated as a third RK4 state (dpsi/dt = r): stage yaw rates
        r1 = float(q[1])
        r2 = float((q + 0.5 * dt * k1)[1])
        r3 = float((q + 0.5 * dt * k2)[1])
        r4 = float((q + dt * k3)[1])  # stage value at t+dt, not the averaged new state
        psi_new = psi + (dt / 6.0) * (r1 + 2.0 * r2 + 2.0 * r3 + r4)
    else:
        raise ValueError(f"unknown integrator {integrator!r}")
    return q_new, psi_new


def _bisect_event(
    model: ManeuveringModel,
    q0: torch.Tensor,
    psi0: float,
    dt: float,
    delta: float,
    integrator: str,
    kind: str,
    target: float = 0.0,
    value0: float = 0.0,
    tol_frac: float = 1e-13,
) -> tuple[torch.Tensor, float, float]:
    """Locate the first event inside one step and return ``(q, psi, fraction)``.

    ``kind="heading"`` finds ``psi(f*dt) == target`` (helm reversal);
    ``kind="peak"`` finds the yaw-rate sign change, i.e. ``r(f*dt) == 0``
    (heading extremum).  ``value0`` is the pre-step value of the monitored
    quantity, so the bisection starts from a bracketing interval.
    """
    lo, hi = 0.0, 1.0
    q_lo, psi_lo = q0, psi0
    q_hi, psi_hi = _step_sway_yaw(model, q0, psi0, dt, delta, integrator)
    for _ in range(200):
        if hi - lo < tol_frac:
            break
        mid = 0.5 * (lo + hi)
        q_m, psi_m = _step_sway_yaw(model, q0, psi0, mid * dt, delta, integrator)
        val_m = (psi_m - target) if kind == "heading" else float(q_m[1])
        val_lo = (psi_lo - target) if kind == "heading" else float(q_lo[1])
        if (val_m == 0.0) or (val_lo * val_m <= 0.0):
            hi, q_hi, psi_hi = mid, q_m, psi_m
        else:
            lo, q_lo, psi_lo = mid, q_m, psi_m
    del value0  # only used to document the bracket sign at the call site
    mid = 0.5 * (lo + hi)
    q_e, psi_e = _step_sway_yaw(model, q0, psi0, mid * dt, delta, integrator)
    return q_e, psi_e, mid


def simulate_zigzag(
    model: ManeuveringModel,
    delta_deg: float,
    heading_deg: float | None = None,
    dt: float = 0.05,
    integrator: str = "rk4",
    n_overshoots: int = 2,
    max_time: float = 900.0,
    record_every: int = 1,
) -> ZigZagResult:
    """Closed-loop zig-zag manoeuvre (``delta``/``heading``) from rest.

    The helm is ordered to ``+delta``; every time the heading deviation reaches
    ``+heading`` it is reversed to ``-delta``, then to ``+delta`` at
    ``-heading``, and so on.  After each reversal the heading keeps travelling
    until the yaw rate changes sign; the excess over the reversal heading is the
    **overshoot angle**.  Helm reversals and heading extrema are located by
    bisection inside a step, so their times are ``dt``-accurate only to the
    integrator's truncation error (not to the step granularity).

    ``heading_deg`` defaults to ``delta_deg`` (the classical ``10/10`` and
    ``20/20`` tests).
    """
    hdg = float(abs(delta_deg) if heading_deg is None else heading_deg)
    if hdg <= 0.0:
        raise ValueError("heading_deg must be positive")
    h_target = math.radians(hdg)
    d = math.radians(float(delta_deg))

    q = torch.zeros(2, dtype=torch.float64)
    psi = 0.0
    t = 0.0
    rudder = d
    target = h_target if d > 0 else -h_target
    peak_pending = False
    res = ZigZagResult(
        delta_deg=float(delta_deg),
        heading_deg=hdg,
        route=integrator,
        dt=dt,
    )
    hist = {"t": [], "psi_deg": [], "r": [], "rudder_deg": []}
    step = 0
    while t < max_time and len(res.overshoots_deg) < n_overshoots:
        q0, psi0 = q, psi
        q_new, psi_new = _step_sway_yaw(model, q0, psi0, dt, rudder, integrator)

        # ---- 1. heading extremum (yaw-rate sign change) ------------------ #
        if peak_pending and float(q0[1]) * float(q_new[1]) < 0.0:
            q_e, psi_e, frac = _bisect_event(
                model, q0, psi0, dt, rudder, integrator, "peak", value0=float(q0[1])
            )
            res.overshoots_deg.append(math.degrees(abs(psi_e)) - hdg)
            res.peak_times.append(t + frac * dt)
            res.events.append(
                {
                    "t": t + frac * dt,
                    "kind": "heading_peak",
                    "heading_deg": math.degrees(psi_e),
                    "rudder_deg": math.degrees(rudder),
                    "yaw_rate": 0.0,
                }
            )
            q, psi, t = q_e, psi_e, t + frac * dt
            peak_pending = False
            continue

        # ---- 2. helm reversal (heading reaches the ordered deviation) ----- #
        if (psi0 - target) * (psi_new - target) <= 0.0:
            q_e, psi_e, frac = _bisect_event(
                model, q0, psi0, dt, rudder, integrator, "heading", target=target
            )
            res.reversal_times.append(t + frac * dt)
            res.events.append(
                {
                    "t": t + frac * dt,
                    "kind": "helm_reversal",
                    "heading_deg": math.degrees(psi_e),
                    "rudder_deg": math.degrees(-rudder),
                    "yaw_rate": float(q_e[1]),
                }
            )
            q, psi, t = q_e, psi_e, t + frac * dt
            rudder = -rudder
            target = -target
            peak_pending = True
            continue

        q, psi, t = q_new, psi_new, t + dt
        if step % record_every == 0:
            hist["t"].append(t)
            hist["psi_deg"].append(math.degrees(psi))
            hist["r"].append(float(q[1]))
            hist["rudder_deg"].append(math.degrees(rudder))
        step += 1
        res.max_heading_deg = max(res.max_heading_deg, abs(math.degrees(psi)))

    res.history = hist
    if len(res.overshoots_deg) < n_overshoots:
        res.terminated = "max_time_reached"
    return res


def _sway_yaw_modes(model: ManeuveringModel):
    """Return ``(evals, evecs, evecs_inv, L, Minv)`` for the linear sway-yaw ODE."""
    import numpy as np

    M = model.mass_matrix().numpy()
    A = model.damping_matrix().numpy()
    L = np.linalg.solve(M, A)
    evals, evecs = np.linalg.eig(L)
    evecs_inv = np.linalg.inv(evecs)
    return evals, evecs, evecs_inv, L, np.linalg.inv(M)


def _segment_solution(model, modes, delta: float, q0, psi0: float, tau):
    """Closed-form ``(q(tau), psi(tau))`` inside one constant-rudder segment."""
    import numpy as np

    evals, evecs, evecs_inv = modes[0], modes[1], modes[2]
    b = model.input_vector(delta).numpy()
    q_ss = np.linalg.solve(model.damping_matrix().numpy(), -b)
    c = evecs_inv @ (np.asarray(q0, dtype=float) - q_ss)
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    E = np.exp(np.outer(tau, evals))
    q = q_ss[None, :] + (E * c[None, :]) @ evecs.T
    r_ss = q_ss[1]
    grow = np.where(np.abs(evals) > 1e-14, (E - 1.0) / np.where(evals == 0, 1, evals), tau[:, None])
    psi = psi0 + r_ss * tau + ((grow * c[None, :]) @ evecs[1, :])
    return q.real, psi.real


def _first_root(f, t_hi: float, n_scan: int = 4000) -> float | None:
    """First root of a scalar function on ``[0, t_hi]`` by scan + bisection."""
    import numpy as np

    ts = np.linspace(0.0, t_hi, n_scan)
    vals = f(ts)
    for i in range(len(ts) - 1):
        if vals[i] == 0.0:
            return float(ts[i])
        if vals[i] * vals[i + 1] < 0.0:
            lo, hi = float(ts[i]), float(ts[i + 1])
            flo = float(vals[i])
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = float(f(np.array([mid]))[0])
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
                if hi - lo < 1e-14:
                    break
            return 0.5 * (lo + hi)
    return None


def analytic_zigzag(
    model: ManeuveringModel,
    delta_deg: float,
    heading_deg: float | None = None,
    n_overshoots: int = 2,
    max_segment_time: float = 600.0,
) -> ZigZagResult:
    """Exact piecewise-modal zig-zag solution (the verification reference).

    Each rudder segment is an LTI system, so ``q(tau)`` and ``psi(tau)`` are
    closed-form; the helm reversals are roots of ``psi(tau) = +-heading`` and the
    heading extrema are roots of ``r(tau) = 0``.  The root times are therefore
    exact to the bisection tolerance -- there is no time-discretisation error at
    all, which makes this route the reference for :func:`simulate_zigzag`.
    """
    import numpy as np

    if model.has_nonlinear():
        raise ValueError(
            "analytic_zigzag is the exact reference for the *linear* sway-yaw model; "
            "this model carries cross-flow terms (Y_vv/N_vv), use simulate_zigzag"
        )
    hdg = float(abs(delta_deg) if heading_deg is None else heading_deg)
    if hdg <= 0.0:
        raise ValueError("heading_deg must be positive")
    h_target = math.radians(hdg)
    d = math.radians(float(delta_deg))
    modes = _sway_yaw_modes(model)

    q = np.zeros(2)
    psi = 0.0
    t = 0.0
    rudder = d
    target = h_target if d > 0 else -h_target
    res = ZigZagResult(delta_deg=float(delta_deg), heading_deg=hdg, route="analytic")
    hist = {"t": [], "psi_deg": [], "r": [], "rudder_deg": []}
    peak_pending = False

    for _ in range(4 * n_overshoots + 2):
        if len(res.overshoots_deg) >= n_overshoots:
            break
        if peak_pending:
            # yaw-rate zero inside this segment -> heading extremum
            tau = _first_root(
                lambda tau_: _segment_solution(model, modes, rudder, q, psi, tau_)[0][:, 1],
                max_segment_time,
            )
            if tau is None:
                res.terminated = "no_peak_found"
                break
            q_p, psi_p = _segment_solution(model, modes, rudder, q, psi, tau)
            psi_end = float(psi_p[0])
            res.overshoots_deg.append(math.degrees(abs(psi_end) - h_target))
            res.peak_times.append(t + tau)
            res.events.append(
                {
                    "t": t + tau,
                    "kind": "heading_peak",
                    "heading_deg": math.degrees(psi_end),
                    "rudder_deg": math.degrees(rudder),
                    "yaw_rate": 0.0,
                }
            )
            # continue the same segment past the peak to record history
            taus = np.linspace(0.0, min(tau, max_segment_time), 200)
            qs, psis = _segment_solution(model, modes, rudder, q, psi, taus)
            hist["t"].extend((t + taus).tolist())
            hist["psi_deg"].extend(np.degrees(psis).tolist())
            hist["r"].extend(qs[:, 1].tolist())
            hist["rudder_deg"].extend([math.degrees(rudder)] * len(taus))
            res.max_heading_deg = max(res.max_heading_deg, abs(math.degrees(psi_end)))
            q, psi = q_p[0], psi_end
            t = t + tau
            peak_pending = False
            continue

        tau = _first_root(
            lambda tau_: _segment_solution(model, modes, rudder, q, psi, tau_)[1] - target,
            max_segment_time,
        )
        if tau is None:
            res.terminated = "no_reversal_found"
            break
        taus = np.linspace(0.0, tau, 200)
        qs, psis = _segment_solution(model, modes, rudder, q, psi, taus)
        hist["t"].extend((t + taus).tolist())
        hist["psi_deg"].extend(np.degrees(psis).tolist())
        hist["r"].extend(qs[:, 1].tolist())
        hist["rudder_deg"].extend([math.degrees(rudder)] * len(taus))
        q, psi = qs[-1], float(psis[-1])
        t = t + tau
        res.reversal_times.append(t)
        res.max_heading_deg = max(res.max_heading_deg, abs(math.degrees(psi)))
        res.events.append(
            {
                "t": t,
                "kind": "helm_reversal",
                "heading_deg": math.degrees(psi),
                "rudder_deg": math.degrees(-rudder),
                "yaw_rate": float(q[1]),
            }
        )
        rudder = -rudder
        target = -target
        peak_pending = True

    res.history = hist
    return res


def imo_msc137_zigzag_limits(
    delta_deg: float, heading_deg: float | None = None, L_over_V: float | None = None
) -> dict:
    """IMO MSC.137(76) overshoot-angle limits for a ``delta``/``heading`` test [deg].

    ``10/10``: first overshoot ``<= 10 deg`` when ``L/V < 10 s``, ``<= 20 deg``
    otherwise.  ``20/20``: first ``<= 25 deg``, second ``<= 40 deg``.
    Returns ``{"applicable": False, "reason": ...}`` for combinations the
    resolution does not cover (e.g. ``20/10``) -- no limit is invented.
    """
    hdg = float(abs(delta_deg) if heading_deg is None else heading_deg)
    key = f"{delta_deg:g}/{hdg:g}"
    if key == "10/10":
        if L_over_V is None:
            return {
                "applicable": False,
                "reason": "10/10 limit depends on L/V; pass L_over_V",
                "source": IMO_MSC137_SOURCE,
            }
        lim = (
            IMO_MSC137_ZIGZAG_LIMITS["10/10"]["first_lt10s"]
            if L_over_V < 10.0
            else IMO_MSC137_ZIGZAG_LIMITS["10/10"]["first_ge10s"]
        )
        return {
            "applicable": True,
            "key": key,
            "limits_deg": {"first": lim},
            "L_over_V_s": L_over_V,
            "source": IMO_MSC137_SOURCE,
        }
    if key == "20/20":
        return {
            "applicable": True,
            "key": key,
            "limits_deg": dict(IMO_MSC137_ZIGZAG_LIMITS["20/20"]),
            "L_over_V_s": L_over_V,
            "source": IMO_MSC137_SOURCE,
        }
    return {
        "applicable": False,
        "reason": f"IMO MSC.137(76) does not define criteria for the {key} zig-zag",
        "source": IMO_MSC137_SOURCE,
    }


def check_imo_msc137_zigzag(result: ZigZagResult, L: float, U: float) -> dict:
    """Apply the IMO MSC.137(76) overshoot limits to one zig-zag result."""
    L_over_V = L / U
    spec = imo_msc137_zigzag_limits(result.delta_deg, result.heading_deg, L_over_V)
    if not spec.get("applicable"):
        return {**spec, "checks": [], "pass": None}
    limits = spec["limits_deg"]
    checks = []
    meas = {
        "first": result.first_overshoot_deg,
        "second": result.second_overshoot_deg,
    }
    for name, lim in limits.items():
        got = meas.get(name, float("nan"))
        checks.append(
            {
                "quantity": f"{name}_overshoot_deg",
                "measured": got,
                "limit": lim,
                "margin_deg": lim - got if got == got else float("nan"),
                "pass": bool(got == got and got <= lim),
            }
        )
    return {**spec, "checks": checks, "pass": all(c["pass"] for c in checks)}