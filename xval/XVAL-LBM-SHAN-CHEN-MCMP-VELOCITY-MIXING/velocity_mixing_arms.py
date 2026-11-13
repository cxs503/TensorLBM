"""Arms for XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING.

One question only: **who mixes what velocity, and with which normalisation**,
in the two-component Shan-Chen (MCMP) LBM.  Everything else in the solver is
held fixed, so that the difference between arms is attributable to the mixing
rule (``predeclared.json`` -> ``variables``).

The OSS reference
-----------------
``sorush-khajepor/listLBM``, technique ``shan-chen-mcmp-velocity-mixing``
(inventory id 44, dumped at
``xval/horizontal_survey_empty/data/oss_solutions_dump.csv`` line 40).  The
inventoried method text and its source citations are::

    mcmp: per-step interaction force over the (ncomps-1) other components via
    precomputed othercomp/othercomppotid index tables, accumulated as a body
    force acc*rho; collision is "velocity mixing, no source term":
        u = (sum_i sum_q c f + F)/(rho_1 + rho_2) + F/rho_i
    then each component uses f_eq(rho_i, u); if rho_1+rho_2 is too small, u=0.
    -- collision/pseudopotential/mcmp_mod.f90:114-155, 161-180, 182-236,
       239-246, 248-278; examples/2d/drop_rise_mcmp_8boxes/main.f90:96-125
    -- per-pair G table: mcmp_mod.f90:263; extforce is an ACCELERATION
       (multiplied by rho); the implementation normalises the force term by
       the total density but the force by the component density
       (mcmp_mod.f90:142) - the inventory flags this as not self-consistent
       and notes the canonical Guo path (:161-180) is not bound.

The formula is read two ways, and the ambiguity is itself a finding, so both
are implemented as separate arms rather than one being silently chosen:

``LBM4`` (reading A: the accumulated body force is the *total* force)
    u_sigma = [sum_s rho_s u_s + (F_1 + F_2)] / rho_tot + F_sigma / rho_sigma

``LBM3`` (reading B: the force is the *colliding component's* force)
    u_sigma = [sum_s rho_s u_s + F_sigma] / rho_tot + F_sigma / rho_sigma

Three further differences from this library are reproduced faithfully (they
are part of the port, not accidental):

* the velocity shift coefficient is **1**, not ``tau`` ("no source term").
  For the velocity-shift forcing family the momentum delivered per step is
  ``s*F/tau`` (see :func:`tensorlbm.multiphase.collide_sc_two_component`), so
  coefficient 1 delivers ``F/tau``: identical to this library at ``tau=1`` and
  ``1/tau`` times too much momentum otherwise.  ``LBM4-TAU`` isolates it.
* the depleted-cell guard: ``u = 0`` where ``rho_1+rho_2 < RHO_FLOOR``,
  whereas the library clamps ``rho_tot`` to ``1e-12`` and lets the ratio stand
  (W4 measures the difference).
* the force itself is **not** re-implemented: every arm, OSS included, calls
  ``tensorlbm.multiphase.sc_two_component_force``.  That is deliberate - the
  subject of the cross-validation is the mixing rule, and holding the force
  fixed is what makes the arm differences interpretable.  Where the OSS force
  is *not* the same object (weights 3x lattice weights -> G rescaled; wet-node
  wall) is listed in the report's difference table instead of being smuggled
  into the arm.

Nothing here mutates the package: the arms are pure functions of the incoming
populations plus a configuration, and the shipped entry point is exercised
unchanged by the ``LIB-ENTRY`` control.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from tensorlbm.cases.two_layer_poiseuille import (
    interface_crossing,
    run_two_layer_poiseuille,
)
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_two_component, sc_two_component_force
from tensorlbm.solver import stream
from tensorlbm.boundary.boundaries import bounce_back_cells

__all__ = [
    "ARMS",
    "RHO_FLOOR",
    "make_collide",
    "init_static_droplet",
    "run_static_droplet",
    "run_stratified_poiseuille",
    "run_library_entry",
    "momentum_ledger",
    "near_vacuum_guard",
    "stabilised_radius",
]

# Guard from the OSS path ("rho_1+rho_2 too small -> u=0").  The source does not
# state the threshold; 1e-6 is far below any physical density in these runs
# (>=1e-2) and far above the library's 1e-12 clamp, so it only ever fires in a
# genuinely depleted cell.
RHO_FLOOR = 1e-6

MixFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# The shared mixing step
# ---------------------------------------------------------------------------
def _mix_velocity(
    mixing: str,
    shift: str,
    rho1: torch.Tensor,
    u1x: torch.Tensor,
    u1y: torch.Tensor,
    rho2: torch.Tensor,
    u2x: torch.Tensor,
    u2y: torch.Tensor,
    F1x: torch.Tensor,
    F1y: torch.Tensor,
    F2x: torch.Tensor,
    F2y: torch.Tensor,
    tau1: float,
    tau2: float,
) -> tuple[torch.Tensor, ...]:
    """Return ``(u1eq_x, u1eq_y, u2eq_x, u2eq_y)`` for one mixing convention.

    ``mixing`` is the *changed* variable of the experiment; ``shift`` selects
    the momentum-transfer coefficient (``"tau"`` = this library's velocity
    shift, ``"one"`` = the OSS "no source term" form).
    """
    r1 = torch.clamp(rho1, min=1e-12)
    r2 = torch.clamp(rho2, min=1e-12)
    rt = r1 + r2
    jx = rho1 * u1x + rho2 * u2x
    jy = rho1 * u1y + rho2 * u2y
    s1 = tau1 if shift == "tau" else 1.0
    s2 = tau2 if shift == "tau" else 1.0

    if mixing == "self":
        m1x, m1y, m2x, m2y = u1x, u1y, u2x, u2y
        g1x, g1y, g2x, g2y = (s1 * F1x / r1, s1 * F1y / r1, s2 * F2x / r2, s2 * F2y / r2)
    elif mixing == "barycentric":
        mx, my = jx / rt, jy / rt
        m1x = m1y = m2x = m2y = None  # type: ignore[assignment]
        g1x, g1y, g2x, g2y = (s1 * F1x / r1, s1 * F1y / r1, s2 * F2x / r2, s2 * F2y / r2)
        return (mx + g1x, my + g1y, mx + g2x, my + g2y)
    elif mixing == "lbm_total":
        mx, my = (jx + F1x + F2x) / rt, (jy + F1y + F2y) / rt
        g1x, g1y, g2x, g2y = (s1 * F1x / r1, s1 * F1y / r1, s2 * F2x / r2, s2 * F2y / r2)
        return (mx + g1x, my + g1y, mx + g2x, my + g2y)
    elif mixing == "lbm_percomp":
        m1x, m1y = (jx + F1x) / rt, (jy + F1y) / rt
        m2x, m2y = (jx + F2x) / rt, (jy + F2y) / rt
        g1x, g1y, g2x, g2y = (s1 * F1x / r1, s1 * F1y / r1, s2 * F2x / r2, s2 * F2y / r2)
        return (m1x + g1x, m1y + g1y, m2x + g2x, m2y + g2y)
    elif mixing == "ctrl_norm":
        # Deliberate defect: the barycentric mixing of this library, but the
        # force shift normalised by the TOTAL density.  Must fail the metric.
        mx, my = jx / rt, jy / rt
        g1x, g1y = s1 * F1x / rt, s1 * F1y / rt
        g2x, g2y = s2 * F2x / rt, s2 * F2y / rt
        return (mx + g1x, my + g1y, mx + g2x, my + g2y)
    else:
        raise ValueError(f"unknown mixing rule {mixing!r}")

    return (m1x + g1x, m1y + g1y, m2x + g2x, m2y + g2y)


def make_collide(mixing: str, shift: str = "tau", guard: bool = False) -> MixFn:
    """Build a collision operator for one arm.

    Args:
        mixing: mixing rule (see :func:`_mix_velocity`); ``"got_mix"`` and
            ``"got_self"`` delegate to the shipped library function instead.
        shift:  ``"tau"`` (library velocity shift) or ``"one"`` (OSS form).
        guard:  apply the OSS depleted-cell guard (``u=0`` where the total
            density is below :data:`RHO_FLOOR`).
    """

    if mixing == "got_mix":
        u_eq: str | None = "mixture"
        from_mixing = "barycentric"
    elif mixing == "got_self":
        u_eq = "self"
        from_mixing = "self"
    else:
        u_eq = None
        from_mixing = mixing

    def collide(
        f1: torch.Tensor,
        f2: torch.Tensor,
        *,
        G_12: float,
        tau1: float = 1.0,
        tau2: float = 1.0,
        gx: float = 0.0,
        gy: float = 0.0,
        solid_mask: torch.Tensor | None = None,
        wall_mode: str = "mirror",
        sc_form: str = "standard",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if u_eq is not None:
            # The CONTROL/LIB arms are the shipped function, unmodified.
            return collide_sc_two_component(
                f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, gx=gx, gy=gy,
                solid_mask=solid_mask, u_eq=u_eq, force_scheme="shift",
                wall_mode=wall_mode, sc_form=sc_form,
            )

        rho1, u1x, u1y = macroscopic(f1)
        rho2, u2x, u2y = macroscopic(f2)
        F1x, F1y, F2x, F2y = sc_two_component_force(
            rho1, rho2, G_12, gx, gy, solid_mask, wall_mode=wall_mode, sc_form=sc_form
        )
        e1x, e1y, e2x, e2y = _mix_velocity(
            from_mixing, shift, rho1, u1x, u1y, rho2, u2x, u2y,
            F1x, F1y, F2x, F2y, tau1, tau2,
        )
        if guard:
            depleted = (rho1 + rho2) < RHO_FLOOR
            e1x = torch.where(depleted, torch.zeros_like(e1x), e1x)
            e1y = torch.where(depleted, torch.zeros_like(e1y), e1y)
            e2x = torch.where(depleted, torch.zeros_like(e2x), e2x)
            e2y = torch.where(depleted, torch.zeros_like(e2y), e2y)

        feq1 = equilibrium(rho1, e1x, e1y)
        feq2 = equilibrium(rho2, e2x, e2y)
        out1 = f1 - (f1 - feq1) / tau1
        out2 = f2 - (f2 - feq2) / tau2
        if solid_mask is not None:
            m = solid_mask.unsqueeze(0)
            out1 = torch.where(m, f1, out1)
            out2 = torch.where(m, f2, out2)
        return out1, out2

    return collide


#: arm name -> (mixing rule, shift coefficient, OSS depleted-cell guard)
ARMS: dict[str, tuple[str, str, bool]] = {
    "GOT-MIX": ("got_mix", "tau", False),
    "GOT-SELF": ("got_self", "tau", False),
    "LBM4": ("lbm_total", "one", True),
    "LBM3": ("lbm_percomp", "one", True),
    "LBM4-TAU": ("lbm_total", "tau", True),
    "CTRL-NORM": ("ctrl_norm", "tau", False),
}

PRIMARY_ARMS = ["GOT-MIX", "GOT-SELF", "LBM4", "LBM3", "CTRL-NORM"]


def build_collide(arm: str) -> MixFn:
    mixing, shift, guard = ARMS[arm]
    return make_collide(mixing, shift, guard)


# ---------------------------------------------------------------------------
# Workload W1: static droplet (spurious currents + Young-Laplace)
# ---------------------------------------------------------------------------
def init_static_droplet(
    n: int,
    radius: float,
    rho1_in: float = 0.92,
    rho2_in: float = 0.08,
    rho1_out: float = 0.14,
    rho2_out: float = 0.86,
    interface_width: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same tanh droplet initialisation as ``tensorlbm.mcmp_verification``."""
    idx = torch.arange(n, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    centre = (n - 1) / 2.0
    r = torch.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / interface_width))
    rho1 = rho1_out + s * (rho1_in - rho1_out)
    rho2 = rho2_out + s * (rho2_in - rho2_out)
    z = torch.zeros_like(rho1)
    return equilibrium(rho1, z, z), equilibrium(rho2, z, z)


def _barycentric(f1: torch.Tensor, f2: torch.Tensor):
    r1, u1x, u1y = macroscopic(f1)
    r2, u2x, u2y = macroscopic(f2)
    rt = torch.clamp(r1 + r2, min=1e-12)
    return r1, r2, (r1 * u1x + r2 * u2x) / rt, (r1 * u1y + r2 * u2y) / rt, rt


def run_static_droplet(
    arm: str,
    *,
    n: int = 64,
    radius: float = 12.0,
    G_12: float = -2.5,
    steps: int = 6000,
    tau1: float = 1.0,
    tau2: float = 1.0,
    wall_mode: str = "mirror",
    sc_form: str = "standard",
    check_every: int = 200,
    stop_tol: float = 1e-4,
    sc_form_force: str | None = None,
) -> dict[str, Any]:
    """Relax a static droplet under one arm; return the measured fields.

    Convergence criterion is the library's: two consecutive windows with a
    relative change of ``|u_mix|max`` below ``stop_tol`` (a steady parasitic
    vortex is a plateau, not a decay to zero).
    """
    collide = build_collide(arm)
    f1, f2 = init_static_droplet(n, radius)
    history: list[list[float]] = []
    u_prev = float("inf")
    stable = 0
    converged = False
    step_run = 0
    with torch.no_grad():
        for step in range(1, steps + 1):
            f1, f2 = collide(
                f1, f2, G_12=G_12, tau1=tau1, tau2=tau2,
                wall_mode=wall_mode, sc_form=sc_form_force or sc_form,
            )
            f1, f2 = stream(f1), stream(f2)
            step_run = step
            if step % check_every == 0:
                _, _, ux, uy, _ = _barycentric(f1, f2)
                u_max = float(torch.sqrt(ux * ux + uy * uy).max())
                history.append([float(step), u_max])
                if not (u_max == u_max) or u_max == float("inf"):
                    break
                if abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                    stable += 1
                    if stable >= 2:
                        converged = True
                        break
                else:
                    stable = 0
                u_prev = u_max
        rho1, rho2, ux, uy, rt = _barycentric(f1, f2)
        finite = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())

    return {
        "arm": arm,
        "n": n,
        "radius_init": radius,
        "G_12": G_12,
        "tau1": tau1,
        "tau2": tau2,
        "steps": step_run,
        "converged": converged,
        "finite": finite,
        "u_max": float(torch.sqrt(ux * ux + uy * uy).max()),
        "u_max_rank": float(torch.sqrt(ux * ux + uy * uy)[1:-1, 1:-1].max()) if n > 4 else float("nan"),
        "history": history,
        "rho1_centre": float(rho1[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2].mean()),
        "rho2_centre": float(rho2[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2].mean()),
        "rho1_corner": float(rho1[:4, :4].mean()),
        "rho2_corner": float(rho2[:4, :4].mean()),
        "u_profile_cx": [float(v) for v in ux[n // 2, n // 2 :]],
        "rho1_profile_cx": [float(v) for v in rho1[n // 2, n // 2 :]],
        "ux_field_sample": [[float(v) for v in ux[i, :]] for i in range(0, n, max(1, n // 16))],
        "uy_field_sample": [[float(v) for v in uy[i, :]] for i in range(0, n, max(1, n // 16))],
    }


def stabilised_radius(rho1: torch.Tensor | list, rho_av: float) -> float:
    """Radius of the ``rho1 = rho_av`` crossing on the +x ray from the centre."""
    row = list(rho1) if not torch.is_tensor(rho1) else [float(v) for v in rho1]
    for j in range(len(row) - 1):
        if (row[j] - rho_av) * (row[j + 1] - rho_av) <= 0 and row[j] != row[j + 1]:
            return j + (rho_av - row[j]) / (row[j + 1] - row[j])
    return float("nan")


def laplace_metrics(run: dict[str, Any]) -> dict[str, float]:
    """``dp`` from the equilibrium pressure (centre box minus corner box), R."""
    # p = cs^2 (rho1+rho2) + G12 cs^2 rho1 rho2  (linear psi, this library)
    cs2 = 1.0 / 3.0
    G = run["G_12"]
    rt_c = run["rho1_centre"] + run["rho2_centre"]
    rt_o = run["rho1_corner"] + run["rho2_corner"]
    p_in = cs2 * rt_c + G * cs2 * run["rho1_centre"] * run["rho2_centre"]
    p_out = cs2 * rt_o + G * cs2 * run["rho1_corner"] * run["rho2_corner"]
    rho_av = 0.5 * (run["rho1_centre"] + run["rho1_corner"])
    R = stabilised_radius(run["rho1_profile_cx"], rho_av)
    return {"dp": p_in - p_out, "R": R, "sigma": (p_in - p_out) * R}


# ---------------------------------------------------------------------------
# Workload W2: stratified two-layer Poiseuille
# ---------------------------------------------------------------------------
def run_stratified_poiseuille(
    arm: str | None,
    *,
    ny: int = 25,
    nx: int = 4,
    steps: int = 20000,
    G_x: float = 5e-5,
    G_12: float = -2.5,
    tau_lower: float = 1.0,
    tau_upper: float = 1.0,
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    interface_width: float = 3.0,
    wall_mode: str = "mirror",
    sc_form: str = "standard",
    half: int | None = None,
    sample_every: int | None = None,
) -> dict[str, Any]:
    """Two-layer channel, one arm.  ``arm=None`` -> the shipped library entry point."""
    if arm is None:
        out = run_two_layer_poiseuille(
            ny=ny, nx=nx, steps=steps, G_x=G_x, G_12=G_12, tau_lower=tau_lower,
            tau_upper=tau_upper, rho_heavy=rho_heavy, rho_light=rho_light,
            interface_width=interface_width, wall_mode=wall_mode, sc_form=sc_form,
            u_eq="mixture", force_scheme="shift", half=half,
        )
        out["arm"] = "LIB-ENTRY"
        return out

    collide = build_collide(arm)
    half = ny // 2 if half is None else half
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    alpha0 = 0.5 * (1.0 - torch.tanh((ys - float(half)) / interface_width))
    rho1_0 = rho_heavy * alpha0 + rho_light * (1.0 - alpha0)
    rho2_0 = rho_light * alpha0 + rho_heavy * (1.0 - alpha0)
    z = torch.zeros((ny, nx))
    f1, f2 = equilibrium(rho1_0, z, z), equilibrium(rho2_0, z, z)
    mass0 = (float(rho1_0.sum()), float(rho2_0.sum()))
    sample_every = sample_every or max(1, min(steps // 10, 2000))
    hist: list[float] = []
    finite = True
    with torch.no_grad():
        for step in range(1, steps + 1):
            f1, f2 = collide(
                f1, f2, G_12=G_12, tau1=tau_lower, tau2=tau_upper, gx=G_x,
                solid_mask=wall, wall_mode=wall_mode, sc_form=sc_form,
            )
            f1, f2 = stream(f1), stream(f2)
            f1, f2 = bounce_back_cells(f1, wall), bounce_back_cells(f2, wall)
            if step % sample_every == 0 or step == steps:
                r1, ux1, _ = macroscopic(f1)
                r2, ux2, _ = macroscopic(f2)
                hist.append(float((ux1 + ux2).mean(dim=1).sum()))
                if not bool(torch.isfinite(f1).all() and torch.isfinite(f2).all()):
                    finite = False
                    break
        r1, ux1, uy1 = macroscopic(f1)
        r2, ux2, uy2 = macroscopic(f2)
        finite = finite and bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())
        rt = torch.clamp(r1 + r2, min=1e-12)
        u_mix = (r1 * ux1 + r2 * ux2) / rt
        uy_mix = (r1 * uy1 + r2 * uy2) / rt
        u_prof = u_mix.mean(dim=1)
        alpha_prof = ((r1 - rho_light) / (rho_heavy - rho_light)).mean(dim=1).clamp(0.0, 1.0)

    steady = False
    if len(hist) >= 4:
        steady = abs(hist[-1] - hist[-3]) / max(abs(hist[-1]), 1e-30) < 1e-3
    return {
        "arm": arm,
        "config": dict(ny=ny, nx=nx, half=half, steps=steps, G_x=G_x, G_12=G_12,
                       tau_lower=tau_lower, tau_upper=tau_upper, rho_heavy=rho_heavy,
                       rho_light=rho_light, interface_width=interface_width,
                       wall_mode=wall_mode, sc_form=sc_form,
                       wall_convention="halfway (H=ny-2)"),
        "finite": finite,
        "steady": steady,
        "Q_mix": float(u_prof.sum()),
        "u_max_mix": float(u_mix[1:-1].abs().max()),
        "spurious_mix": float(uy_mix[1:-1].abs().max()),
        "interface_crossing": interface_crossing(alpha_prof.tolist()),
        "alpha_profile": alpha_prof.tolist(),
        "u_profile_mix": u_prof.tolist(),
        "mass_drift": [float(r1.sum()) - mass0[0], float(r2.sum()) - mass0[1]],
        "flux_history": hist,
    }


def run_library_entry(**kw) -> dict[str, Any]:
    """The shipped entry point, unchanged (harness control)."""
    return run_stratified_poiseuille(None, **kw)


# ---------------------------------------------------------------------------
# Workload W3: momentum ledger
# ---------------------------------------------------------------------------
def momentum_ledger(
    arm: str,
    *,
    n: int = 48,
    G_12: float = -2.5,
    steps: int = 400,
    tau1: float = 1.0,
    tau2: float = 1.0,
    sc_form: str = "standard",
) -> dict[str, Any]:
    """Global momentum of a closed periodic two-component box with an interface.

    No body force, no walls: the only momentum source is the discrete
    inter-component force and the mixing convention.  A momentum-conserving
    arm keeps ``sum_s sum_i sum_x c_i f_s`` at its initial value.
    """
    collide = build_collide(arm)
    idx = torch.arange(n, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    # Symmetric-in-x, stratified-in-y initial field: a non-symmetric field would
    # hide a momentum leak in a round-off-level initial imbalance.
    s = 0.5 * (1.0 - torch.tanh((yy - n / 2) / 2.0))
    rho1_0 = 0.14 + s * (0.92 - 0.14)
    rho2_0 = 0.86 - s * (0.86 - 0.08)
    z = torch.zeros_like(rho1_0)
    f1, f2 = equilibrium(rho1_0, z, z), equilibrium(rho2_0, z, z)

    from tensorlbm.d2q9 import C as _C  # lattice velocities

    c = _C.float()

    def momentum(fa, fb):
        pa = (fa.unsqueeze(-1) * c.view(9, 1, 1, 2)).sum(dim=(0, 1, 2))
        pb = (fb.unsqueeze(-1) * c.view(9, 1, 1, 2)).sum(dim=(0, 1, 2))
        return (pa + pb)

    p0 = momentum(f1, f2)
    hist = []
    with torch.no_grad():
        for step in range(1, steps + 1):
            # bookkeeping of the force's own box sum, for the interpretation
            r1, u1x, u1y = macroscopic(f1)
            r2, u2x, u2y = macroscopic(f2)
            F1x, F1y, F2x, F2y = sc_two_component_force(r1, r2, G_12, 0.0, 0.0, None, sc_form=sc_form)
            f1, f2 = collide(f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, sc_form=sc_form)
            f1, f2 = stream(f1), stream(f2)
            if step % 50 == 0 or step == 1:
                p = momentum(f1, f2)
                hist.append([float(step), float(p[0] - p0[0]), float(p[1] - p0[1])])
    pT = momentum(f1, f2)
    # Box sums of the force terms that the mixing conventions redistribute.
    r1, u1x, u1y = macroscopic(f1)
    r2, u2x, u2y = macroscopic(f2)
    F1x, F1y, F2x, F2y = sc_two_component_force(r1, r2, G_12, 0.0, 0.0, None, sc_form=sc_form)
    rt = torch.clamp(r1 + r2, min=1e-12)
    src_percomp = float(((r1 * F1x + r2 * F2x) / rt).sum())
    src_total = float((F1x + F2x).sum())
    return {
        "arm": arm,
        "G_12": G_12,
        "steps": steps,
        "n": n,
        "P0": [float(p0[0]), float(p0[1])],
        "PT": [float(pT[0]), float(pT[1])],
        "drift": [float(pT[0] - p0[0]), float(pT[1] - p0[1])],
        "drift_rate_per_step": [float((pT[0] - p0[0]) / steps), float((pT[1] - p0[1]) / steps)],
        "history": hist,
        "box_sum_F1_plus_F2_x": src_total,
        "box_sum_component_weighted_F_x": src_percomp,
        "finite": bool(torch.isfinite(f1).all() and torch.isfinite(f2).all()),
    }


# ---------------------------------------------------------------------------
# Workload W4: near-vacuum cell
# ---------------------------------------------------------------------------
def near_vacuum_guard(*, rho_tiny: float = 1e-8) -> dict[str, Any]:
    """One collision on a field with a depleted cell: guard vs clamp."""
    out: dict[str, Any] = {"rho_tiny": rho_tiny, "arms": {}}
    for arm in ("GOT-MIX", "LBM4", "LBM3"):
        collide = build_collide(arm)
        f1, f2 = init_static_droplet(16, 6.0)
        # Deplete one 2x2 patch near the centre (both components -> rho_tiny).
        rho1 = torch.zeros((16, 16))
        rho2 = torch.zeros((16, 16))
        r1, u1x, u1y = macroscopic(f1)
        r2, u2x, u2y = macroscopic(f2)
        rho1[:] = r1
        rho2[:] = r2
        rho1[7:9, 7:9] = rho_tiny
        rho2[7:9, 7:9] = 0.0
        z = torch.zeros((16, 16))
        # Give the depleted cell a small but non-zero momentum (a real depleted
        # cell is never exactly at rest).
        ux = torch.zeros((16, 16))
        ux[7:9, 7:9] = 1e-3
        g1, g2 = equilibrium(rho1, ux, z), equilibrium(rho2, z, z)
        with torch.no_grad():
            o1, o2 = collide(g1, g2, G_12=-2.5, tau1=1.0, tau2=1.0)
            a1, ax1, ay1 = macroscopic(o1)
            a2, ax2, ay2 = macroscopic(o2)
            rt = a1 + a2
            umix = (a1 * ax1 + a2 * ax2) / torch.clamp(rt, min=1e-30)
        patch = umix[7:9, 7:9]
        out["arms"][arm] = {
            "finite": bool(torch.isfinite(o1).all() and torch.isfinite(o2).all()),
            "u_mix_patch_max": float(patch.abs().max()),
            "u_mix_global_max": float(umix.abs().max()),
            "rho_tot_patch": float(rt[7:9, 7:9].mean()),
        }
    return out