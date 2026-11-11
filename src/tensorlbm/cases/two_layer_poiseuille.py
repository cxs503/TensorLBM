"""Two-layer (stratified) co-moving Poiseuille flow — library entry point.

A plane channel of `ny` rows (x-periodic, uniform in x) is filled with two
immiscible Shan–Chen components: component 1 (``rho_heavy``) in the lower half,
component 2 (``rho_light``) in the upper half, separated by a diffuse interface
of width `interface_width`.  A uniform body force ``G_x`` per unit mixture
volume drives both layers, so the exact solution is the classical two-layer
co-moving Poiseuille profile (velocity and shear stress continuous at the
interface, no slip at both walls).

This function is the *solver-side* API used by the benchmark of the ledger item
``GAP-LBM-MCMP-POISEUILLE`` (``cases/mcmp_stratified_poiseuille/``) and by
``scripts/run_two_layer_poiseuille_check.py``.  It returns only simulated
quantities — the analytic reference lives with the benchmark, so that library
code carries no case-specific constants.

Wall conventions
----------------
Post-streaming bounce-back on rows ``0`` and ``ny-1`` puts the effective no-slip
planes on the half-cell, i.e. at ``y = 0.5`` and ``y = ny - 1.5`` (``H = ny - 2``,
mid-plane ``h = H/2``).  Use that convention when comparing with the closed
form (it was calibrated in the benchmark: the ``H = ny-1`` cell-centre
convention over-predicts ``u_max``/``Q`` by 8 %/12 % at ``ny=25``).

Physics notes
-------------
* ``wall_mode="mirror"`` gives a neutral-wetting wall (the solid cells enter the
  interaction sum with the mirrored nearest-fluid density).  The historical
  ``"dry"`` wall injects an artificial normal force ``O(|G12| rho^2)`` on the
  first fluid row and is kept only for bit-identical backward compatibility.
* ``sc_form="edm"`` uses the exact-difference (well-balanced) split of the pair
  force; it removes most of the parasitic current at the diffuse interface that
  the standard backward-gather form leaves behind (see the R3 benchmark).
"""
from __future__ import annotations

import math

import torch

from ..boundary.boundaries import bounce_back_cells
from ..d2q9 import equilibrium, macroscopic
from ..multiphase import collide_sc_two_component
from ..solver import stream

__all__ = ["run_two_layer_poiseuille", "interface_crossing"]


def interface_crossing(alpha_profile) -> float:
    """Interpolated row where the heavy-component fraction crosses 0.5."""
    a = list(alpha_profile)
    for j in range(len(a) - 1):
        if (a[j] - 0.5) * (a[j + 1] - 0.5) <= 0 and a[j] != a[j + 1]:
            return j + (0.5 - a[j]) / (a[j + 1] - a[j])
    return float("nan")


def run_two_layer_poiseuille(
    ny: int = 25,
    nx: int = 4,
    steps: int = 20000,
    G_x: float = 5e-5,
    G_12: float = -2.5,
    tau_lower: float = 1.0,
    tau_upper: float = 0.75,
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    interface_width: float = 3.0,
    wall_mode: str = "mirror",
    sc_form: str = "edm",
    u_eq: str = "viscous",
    force_scheme: str = "guo",
    half: int | None = None,
) -> dict:
    """Run the stratified two-component channel and return the simulated metrics.

    Returns a dict with ``Q_mix`` (volume flux of the mixture velocity),
    ``Q_component`` (flux of each component velocity), ``interface_crossing``,
    ``alpha_profile``, ``u_profile_mix``, ``u_max_mix``, ``spurious_mix``
    (``max |u_y,mix|`` over fluid rows), ``phi_contrast``, ``steady`` and
    ``finite``.
    """
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

    sample_every = max(1, min(steps // 10, 2000))
    hist = []
    with torch.no_grad():
        for step in range(1, steps + 1):
            f1, f2 = collide_sc_two_component(
                f1, f2, G_12=G_12, tau1=tau_lower, tau2=tau_upper, gx=G_x,
                solid_mask=wall, u_eq=u_eq, force_scheme=force_scheme,
                wall_mode=wall_mode, sc_form=sc_form,
            )
            f1, f2 = stream(f1), stream(f2)
            f1, f2 = bounce_back_cells(f1, wall), bounce_back_cells(f2, wall)
            if step % sample_every == 0 or step == steps:
                r1, ux1, _ = macroscopic(f1)
                r2, ux2, _ = macroscopic(f2)
                hist.append(float((ux1 + ux2).mean(dim=1).sum()))
                if not bool(torch.isfinite(f1).all() and torch.isfinite(f2).all()):
                    break

        r1, ux1, uy1 = macroscopic(f1)
        r2, ux2, uy2 = macroscopic(f2)
        finite = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())

        rho_tot = torch.clamp(r1 + r2, min=1e-12)
        u_mix = (r1 * ux1 + r2 * ux2) / rho_tot
        uy_mix = (r1 * uy1 + r2 * uy2) / rho_tot
        u_mix_prof = u_mix.mean(dim=1)
        alpha_prof = ((r1 - rho_light) / (rho_heavy - rho_light)).mean(dim=1).clamp(0.0, 1.0)

    steady = False
    if len(hist) >= 4:
        tail = hist[-3:]
        steady = abs(tail[-1] - tail[0]) / max(abs(tail[-1]), 1e-30) < 1e-3

    bulk = alpha_prof[1:-1]
    return dict(
        config=dict(ny=ny, nx=nx, half=half, steps=steps, G_x=G_x, G_12=G_12,
                    tau_lower=tau_lower, tau_upper=tau_upper, rho_heavy=rho_heavy,
                    rho_light=rho_light, interface_width=interface_width,
                    wall_mode=wall_mode, sc_form=sc_form, u_eq=u_eq,
                    force_scheme=force_scheme, wall_convention="halfway (H=ny-2)"),
        finite=finite,
        steady=steady,
        Q_mix=float(u_mix_prof.sum()),
        Q_component=(float(ux1.mean(dim=1).sum()), float(ux2.mean(dim=1).sum())),
        interface_crossing=interface_crossing(alpha_prof.tolist()),
        alpha_profile=alpha_prof.tolist(),
        u_profile_mix=u_mix_prof.tolist(),
        u_max_mix=float(u_mix[1:-1].abs().max()),
        spurious_mix=float(uy_mix[1:-1].abs().max()),
        phi_contrast=float(bulk.max() - bulk.min()),
        mass_drift=(float(r1.sum()) - mass0[0], float(r2.sum()) - mass0[1]),
        nu_lower=(1.0 / 3.0) * (tau_lower - 0.5),
        nu_upper=(1.0 / 3.0) * (tau_upper - 0.5),
    )
