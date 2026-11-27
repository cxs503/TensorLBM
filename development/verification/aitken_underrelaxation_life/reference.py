"""Reference implementations for the displacement-Aitken fixed-point XVAL.

Three independent reference paths are provided for the same physical target
(a 1-DOF piston wetted by an incompressible fluid column):

``monolithic_newmark``
    The discrete ground truth.  The coupled system
    ``(m_s + m_a) a + c v + k d = F_ext`` is advanced with *one* Newmark
    solve per step (no sub-iteration).  Every partitioned arm must reproduce
    this sequence if its fixed point converges.

``analytic_sdof``
    The continuous ground truth: closed-form solution of
    ``(m_s + m_a) d'' + c d' + k d = F0 sin(Omega t)`` with rest initial
    conditions (under-damped case).  Used for the physical accuracy KPI.

``life_fixed_point_step``
    A transcription of the strong-coupling sub-iteration of
    joconnor22/LIFE (``oss_solutions`` row 258, feature
    ``NEW:LBM-FSI-BLOCK-GS-AITKEN``, tag
    ``displacement-aitken-underrelaxation-fixed-point``):

        do { U0 = 2.5 U_n - 2 U_{n-1} + 0.5 U_{n-2}   (order-2 predictor)
             interpolate -> FEM full re-solve (Newton-Raphson here: exact
             linear Newmark solve) -> residual }
        while (res > 1e-8 and it < 20)
        Aitken for k >= 2:
            relax = -relax * (r_{k-1} . (r_k - r_{k-1})) / |r_k - r_{k-1}|^2
            U = U_{k-1} + relax * (U - U_{k-1})
        residual norm normalised by ref_L * sqrt(DOF)

    The transcription works on the **displacement** interface variable and
    uses the FEM structure solve with the *structural* mass only (the added
    mass lives exclusively in the fluid operator), i.e. the ``explicit``
    staggered variant of the repo model.  No clipping of ``relax``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GAMMA = 0.5
BETA_N = 0.25


@dataclass(frozen=True)
class Piston:
    """1-DOF piston + fluid column.

    ``m_added`` is set from the nominal added-mass ratio so that
    ``beta_eff = m_added / m_eff_solid`` takes the prescribed value, with
    ``m_eff_solid = m_struct + c gamma dt + k beta_N dt**2`` (Newmark mass).
    """

    m_struct: float
    beta_eff: float
    stiffness: float
    damping: float
    dt: float

    @property
    def m_eff_solid(self) -> float:
        return (
            self.m_struct
            + self.damping * GAMMA * self.dt
            + self.stiffness * BETA_N * self.dt * self.dt
        )

    @property
    def m_added(self) -> float:
        return self.beta_eff * self.m_eff_solid

    @property
    def period(self) -> float:
        return 2.0 * math.pi * math.sqrt((self.m_struct + self.m_added) / self.stiffness)

    @property
    def omega_star(self) -> float:
        """Analytic Aitken fixed point ``1/(1+beta_eff)``."""
        return 1.0 / (1.0 + self.beta_eff)

    def to_dict(self) -> dict:
        return {
            "m_struct": self.m_struct,
            "m_added": self.m_added,
            "m_eff_solid": self.m_eff_solid,
            "beta_eff": self.beta_eff,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "dt": self.dt,
            "period": self.period,
            "dt_over_period": self.dt / self.period,
            "omega_star": self.omega_star,
        }


def _known_terms(p: Piston, d: float, v: float, a_prev: float) -> float:
    """Explicit part of the Newmark momentum equation (sign convention of
    ``tensorlbm.fsi_partitioned``)."""
    return -(
        p.damping * (v + p.dt * (1.0 - GAMMA) * a_prev)
        + p.stiffness * (d + p.dt * v + p.dt * p.dt * (0.5 - BETA_N) * a_prev)
    )


def monolithic_newmark(
    p: Piston, *, n_steps: int, f_ext: callable
) -> dict[str, np.ndarray]:
    """One Newmark solve per step with the *total* mass ``m_s + m_a``."""
    m_tot = p.m_eff_solid + p.m_added
    d = v = a_prev = 0.0
    out = {k: [] for k in ("t", "d", "v", "a")}
    for n in range(n_steps):
        t = (n + 1) * p.dt
        f = float(f_ext(t))
        a = (f + _known_terms(p, d, v, a_prev)) / m_tot
        d_new = d + p.dt * v + p.dt * p.dt * ((0.5 - BETA_N) * a_prev + BETA_N * a)
        v_new = v + p.dt * ((1.0 - GAMMA) * a_prev + GAMMA * a)
        out["t"].append(t)
        out["d"].append(d_new)
        out["v"].append(v_new)
        out["a"].append(a)
        d, v, a_prev = d_new, v_new, a
    return {k: np.asarray(val, dtype=float) for k, val in out.items()}


def analytic_sdof(p: Piston, *, n_steps: int, amp: float, freq: float) -> dict[str, np.ndarray]:
    """Closed form of ``M d'' + C d' + K d = amp sin(freq t)``, ``d(0)=v(0)=0``."""
    m_tot = p.m_struct + p.m_added
    k, c = p.stiffness, p.damping
    w_n = math.sqrt(k / m_tot)
    zeta = c / (2.0 * math.sqrt(k * m_tot))
    if not 0.0 < zeta < 1.0:
        raise ValueError("analytic_sdof expects an under-damped system")
    w_d = w_n * math.sqrt(1.0 - zeta * zeta)
    s = k - m_tot * freq * freq
    den = s * s + (c * freq) ** 2
    a_h = amp * s / den
    b_h = -amp * c * freq / den
    c1 = -b_h
    c2 = (zeta * w_n * c1 - a_h * freq) / w_d

    def d_of(t):
        return (
            a_h * math.sin(freq * t)
            + b_h * math.cos(freq * t)
            + math.exp(-zeta * w_n * t) * (c1 * math.cos(w_d * t) + c2 * math.sin(w_d * t))
        )

    ts = np.array([(n + 1) * p.dt for n in range(n_steps)], dtype=float)
    return {"t": ts, "d": np.array([d_of(t) for t in ts], dtype=float)}


@dataclass
class LifeStepLog:
    iterations: int
    converged: bool
    omegas: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)


def life_fixed_point_step(
    p: Piston,
    *,
    d_n: float,
    d_nm1: float,
    d_nm2: float,
    v_n: float,
    a_n: float,
    f_ext: float,
    tol: float = 1e-8,
    max_iter: int = 20,
    ref_len: float = 1.0,
    predictor: bool = True,
) -> tuple[float, float, float, LifeStepLog]:
    """One time step of joconnor22/LIFE's displacement-Aitken fixed point.

    Returns ``(d_next, v_next, a_next, log)``; the residual norm is
    ``|d_change| / (ref_len * sqrt(DOF))`` with ``DOF = 1``.
    """
    m_eff = p.m_eff_solid
    known = _known_terms(p, d_n, v_n, a_n)

    # order-2 extrapolated predictor (LIFE: 2.5 U_n - 2 U_{n-1} + 0.5 U_{n-2})
    u_prev = 2.5 * d_n - 2.0 * d_nm1 + 0.5 * d_nm2 if predictor else d_n

    relax = 1.0
    log = LifeStepLog(iterations=0, converged=False)
    r_prev: float | None = None
    u = u_prev
    u_new, a_new, v_new = u_prev, a_n, v_n
    for it in range(1, max_iter + 1):
        # structural kinematics from the candidate displacement (Newmark inversion)
        a_c = (u - d_n - p.dt * v_n - p.dt * p.dt * (0.5 - BETA_N) * a_n) / (BETA_N * p.dt * p.dt)
        v_c = v_n + p.dt * ((1.0 - GAMMA) * a_n + GAMMA * a_c)
        # fluid operator: added mass reacts to the interface acceleration
        f_fluid = f_ext - p.m_added * a_c
        # FEM full re-solve (linear Newmark is exact; no sub-iteration needed)
        a_s = (f_fluid + known) / m_eff
        u_s = d_n + p.dt * v_n + p.dt * p.dt * ((0.5 - BETA_N) * a_n + BETA_N * a_s)
        r = u_s - u
        res = abs(r) / (ref_len * math.sqrt(1.0))
        log.residuals.append(res)
        log.iterations = it
        if res < tol:
            u_new, a_new, v_new = u_s, a_s, v_n + p.dt * ((1.0 - GAMMA) * a_n + GAMMA * a_s)
            log.converged = True
            break
        if r_prev is not None:
            dr = r - r_prev
            denom = dr * dr
            if denom > 1e-300:
                cand = -relax * (r_prev * dr) / denom
                if math.isfinite(cand):
                    relax = cand           # LIFE: no clipping
        r_prev = r
        u = u + relax * r                  # U = U_{k-1} + relax (U_s - U_{k-1})
        log.omegas.append(relax)
    if not log.converged:
        # documented non-convergence path: return the last iterate consistently
        a_c = (u - d_n - p.dt * v_n - p.dt * p.dt * (0.5 - BETA_N) * a_n) / (BETA_N * p.dt * p.dt)
        v_c = v_n + p.dt * ((1.0 - GAMMA) * a_n + GAMMA * a_c)
        return u, v_c, a_c, log
    return u_new, v_new, a_new, log


def life_run(p: Piston, *, n_steps: int, f_ext: callable, **kwargs) -> dict:
    """Full LIFE-style time loop (predictor history + Aitken fixed point)."""
    d = v = a_prev = 0.0
    hist_d = [0.0, 0.0]
    out = {k: [] for k in ("t", "d", "v", "a", "iters", "omega")}
    converged = True
    per_step = []
    for n in range(n_steps):
        t = (n + 1) * p.dt
        d_next, v_next, a_next, log = life_fixed_point_step(
            p, d_n=d, d_nm1=hist_d[-1], d_nm2=hist_d[-2], v_n=v, a_n=a_prev,
            f_ext=float(f_ext(t)), **kwargs
        )
        out["t"].append(t)
        out["d"].append(d_next)
        out["v"].append(v_next)
        out["a"].append(a_next)
        out["iters"].append(log.iterations)
        out["omega"].append(log.omegas[-1] if log.omegas else 1.0)
        per_step.append(log)
        converged = converged and log.converged
        hist_d.append(d_next)
        d, v, a_prev = d_next, v_next, a_next
    r = {k: np.asarray(val, dtype=float) for k, val in out.items()}
    r["converged"] = converged
    r["per_step"] = per_step
    return r