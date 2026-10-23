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
from typing import Optional

import torch

__all__ = [
    "ManeuveringModel",
    "SteadyTurn",
    "simulate_turn",
    "analytic_transient",
    "NOMINAL_TANKER_PRIME",
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
    "Y_delta": -0.16,   # rudder side force to port for delta > 0 (starboard)
    "N_delta": 0.04,    # bow-to-starboard yaw couple for delta > 0
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
        return torch.tensor([self.Y_vv * abs(v) * v, self.N_vv * abs(v) * v],
                            dtype=torch.float64)

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
            delta=delta, v=v, r=r, R=R, R_over_L=R / self.L, beta_deg=beta,
            K=K, det=det, stable=stable, speed=math.hypot(self.U, v),
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
            return (Y_v * v + (Y_r - mU) * r_of(v)
                    + self.Y_vv * abs(v) * v + Y_d * delta)

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
    return torch.linalg.solve(M, A @ q + model.input_vector(delta)
                              + model.nonlinear_force(v))


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
    M = model.mass_matrix()
    A = model.damping_matrix()

    q = torch.zeros(2, dtype=torch.float64)      # v, r
    pos = torch.zeros(2, dtype=torch.float64)    # x, y
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
    c = np.linalg.solve(M, b)
    q_ss = np.linalg.solve(A, -b)
    evals, evecs = np.linalg.eig(L)
    coeff = np.linalg.solve(evecs, -q_ss)
    tt = np.asarray(t, dtype=float)
    q = np.array([q_ss for _ in tt])
    for lam, phi, cc in zip(evals, evecs.T, coeff):
        q = q + np.outer(np.exp(lam * tt) * cc, phi)
    return q[:, 0], q[:, 1]