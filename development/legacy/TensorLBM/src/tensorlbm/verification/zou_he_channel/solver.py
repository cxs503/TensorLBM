"""D2Q9 channel solver driven purely by Zou-He inlet/outlet reconstruction.

Compute kernel of the cross-validation case
``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``.  The point is to exercise
``tensorlbm.boundary.zou_he_channel`` as the *only* streamwise closure of an
open channel and to compare the converged field with the exact plane-Poiseuille
solution -- no body force, no periodic wrap, no analytic inlet profile in the
primary arm.

Setup (all lattice units)
-------------------------
* two solid rows (``y = 0`` and ``y = ny - 1``) with half-way bounce-back, so the
  effective channel width is ``H = ny - 2`` measured between the wall mid-planes
  at ``y = 0.5`` and ``y = ny - 1.5``;
* BGK/SRT collision, constant ``tau`` (kinematic viscosity ``nu = (tau-1/2)/3``);
* streamwise closure: Zou-He reconstruction at ``x = 0`` (inlet) and ``x = nx-1``
  (outlet), selectable per side as ``"velocity"`` or ``"pressure"``.

Arms
----
``"pressure-pressure"`` (primary): ``rho_in = 1 + drho/2``, ``rho_out = 1 - drho/2``.
    The mass flux is *emergent*.  Analytic reference for the same driving:

        u(y) = G / (2 nu) * yt * (H - yt),   yt = y - 0.5,   G = -dp/dx = -(1/3) drho/dx

    Because the boundary-to-boundary length carries an O(1) convention
    ambiguity, ``G`` is taken from a least-squares fit of the *computed* mean
    density profile, and the nominal ``drho / (nx - 1)`` is reported as a
    separate (stricter) metric.

``"velocity-pressure"`` (secondary, the OSS closure): prescribed parabolic inlet
    velocity from the analytic solution ``u_in(y)``, pressure outlet.  Here the
    flux is imposed, so this arm probes the reconstruction's consistency when the
    inlet profile is itself the analytic answer.

``"equilibrium-pressure"`` (negative control): unknown populations at the inlet
    are replaced by the local equilibrium of the *known* moments.  Same geometry,
    same outlet: a closure-quality contrast arm, expected to miss the analytic
    profile by far more than the 3 % target -- i.e. the metric is able to fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ...boundary.d2q9 import C2Q9, W2Q9, OPPOSITE_2Q9, CS2_2Q9
from ...boundary.zou_he_channel import zou_he_wall

CX = C2Q9[:, 0].tolist()
CY = C2Q9[:, 1].tolist()
_W = W2Q9
_OPP = OPPOSITE_2Q9.tolist()
_CS2 = 1.0 / 3.0


def equilibrium_d2q9(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Second-order D2Q9 equilibrium, shape (9, ny, nx)."""
    w = _W.to(rho.device).view(9, 1, 1)
    cx = C2Q9[:, 0].to(rho.device).view(9, 1, 1).to(rho.dtype)
    cy = C2Q9[:, 1].to(rho.device).view(9, 1, 1).to(rho.dtype)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    usq = ux * ux + uy * uy
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0))


@dataclass
class ChannelCase:
    """Geometry / physics of one channel arm (lattice units)."""

    height: int = 32               # fluid rows H; ny = H + 2
    length: int = 160              # nx
    tau: float = 0.65
    drho: float = 4.0e-3           # inlet-outlet density difference
    arm: str = "pressure-pressure"
    init: str = "analytic"         # "analytic" | "uniform"
    tol: float = 1.0e-6           # relative change between block averages
    max_steps: int = 40000
    check_every: int = 200
    burn_in: int = 1500           # steps before time averaging starts
    ramp_steps: int = 1000        # quasi-static ramp of the end densities
    avg_window: int = 200         # block length of the time average
    dtype: torch.dtype = torch.float64
    refit_length: Optional[int] = None   # unused, kept for report clarity
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def ny(self) -> int:
        return self.height + 2

    @property
    def nx(self) -> int:
        return self.length

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def g_nominal(self) -> float:
        """Nominal driving gradient if the boundaries were ``nx-1`` apart."""
        return self.drho / (3.0 * (self.nx - 1))

    @property
    def u_max_nominal(self) -> float:
        return self.g_nominal * self.height ** 2 / (8.0 * self.nu)


def analytic_profile(case: ChannelCase, g: float) -> torch.Tensor:
    """Exact plane-Poiseuille profile for gradient ``g`` over the fluid rows."""
    y = torch.arange(1, case.ny - 1, dtype=case.dtype)
    yt = y - 0.5
    h = float(case.height)
    return g / (2.0 * case.nu) * yt * (h - yt)


def _stream(f_post: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    """Pull streaming with half-way bounce-back on solid neighbours."""
    _, ny, nx = f_post.shape
    f = torch.zeros_like(f_post)
    f[0] = f_post[0]
    for i in range(1, 9):
        cx, cy = CX[i], CY[i]
        ty0, ty1 = max(0, cy), ny + min(0, cy)
        tx0, tx1 = max(0, cx), nx + min(0, cx)
        sy0, sy1 = ty0 - cy, ty1 - cy
        sx0, sx1 = tx0 - cx, tx1 - cx
        block = f[i, ty0:ty1, tx0:tx1]
        block.copy_(f_post[i, sy0:sy1, sx0:sx1])
        solid_src = ~fluid[sy0:sy1, sx0:sx1]
        if bool(solid_src.any()):
            bounce = f_post[_OPP[i], ty0:ty1, tx0:tx1]
            block[solid_src] = bounce[solid_src]
    return f


def full_profile(case: ChannelCase, scale: float = 1.0) -> torch.Tensor:
    """Analytic profile embedded in a full-height array (zeros on the solid rows).

    The Zou-He closures operate on a *whole* boundary layer (all ``ny`` rows);
    the solid rows are included for index alignment only -- they are never
    sources for a fluid node, so their prescribed velocity is inert.
    """
    prof = torch.zeros(case.ny, dtype=case.dtype)
    prof[1:case.ny - 1] = analytic_profile(case, case.g_nominal).to(case.dtype) * scale
    return prof


def _equilibrium_inlet(f: torch.Tensor, ux_profile: torch.Tensor) -> None:
    """Negative-control closure: inlet unknowns set to local equilibrium."""
    line = f[:, :, 0]
    rho = line.sum(0)
    ux = (line * C2Q9[:, 0:1].to(f.device).to(f.dtype)).sum(0) / rho
    uy = (line * C2Q9[:, 1:2].to(f.device).to(f.dtype)).sum(0) / rho
    ux = ux_profile
    feq = equilibrium_d2q9(rho, ux, uy)
    for i in (1, 5, 7):
        line[i] = feq[i]


def run_channel(case: ChannelCase, verbose: bool = False) -> Dict[str, object]:
    """March ``case`` to steady state and return the measured field + metrics."""
    torch.set_default_dtype(case.dtype)
    ny, nx = case.ny, case.nx
    def end_densities(step: int):
        """Ramp the imposed densities: an abrupt start excites acoustic modes
        that a soft (Zou-He) pressure boundary barely damps."""
        f_ramp = 1.0 if case.ramp_steps <= 0 else min(1.0, step / float(case.ramp_steps))
        return 1.0 + f_ramp * case.drho / 2.0, 1.0 - f_ramp * case.drho / 2.0

    fluid = torch.zeros((ny, nx), dtype=torch.bool)
    fluid[1:ny - 1, :] = True

    rho = torch.ones((ny, nx), dtype=case.dtype)
    ux = torch.zeros((ny, nx), dtype=case.dtype)
    uy = torch.zeros((ny, nx), dtype=case.dtype)
    if case.init == "analytic":
        # Parabolic velocity *and* the linear density ramp set by the two end
        # densities: starting from a uniform density leaves the pressure field to
        # relax through a slow acoustic mode and dominates the run time.
        prof = analytic_profile(case, case.g_nominal).to(case.dtype)
        ux[1:ny - 1, :] = prof.view(-1, 1)
        xline = torch.arange(nx, dtype=case.dtype) / max(nx - 1, 1)
        rho[1:ny - 1, :] = 1.0 + case.drho * (0.5 - xline).view(1, -1)
    f = equilibrium_d2q9(rho, ux, uy)

    mass_defect = float("nan")
    step = 0
    ux_sum = torch.zeros_like(ux)
    rho_sum = torch.zeros_like(rho)
    navg = 0
    block_mean: Optional[torch.Tensor] = None
    block_start = 0
    converged = False
    # telemetry: drift of the running time average between consecutive blocks,
    # one entry per completed block.  A grid-refinement study needs to show that
    # the residual is *at* the floor, not merely that max_steps was reached.
    drift_hist: List[float] = []
    drift_steps: List[int] = []
    # Second residual: the drift of the *streamwise-averaged* profile.  The
    # pointwise criterion above measures the acoustic ripple a soft Zou-He
    # boundary does not damp, so it can stay at 1e-2 forever even when the mean
    # field is at its floor.  Averaging along x removes the standing wave and
    # leaves the quantity the Poiseuille comparison actually uses.
    drift_profile_hist: List[float] = []
    prof_prev: Optional[torch.Tensor] = None
    for step in range(1, case.max_steps + 1):
        # --- macroscopic + BGK collision -------------------------------------
        rho = f.sum(0)
        ux = (f * C2Q9[:, 0].view(9, 1, 1).to(case.dtype)).sum(0) / rho
        uy = (f * C2Q9[:, 1].view(9, 1, 1).to(case.dtype)).sum(0) / rho
        feq = equilibrium_d2q9(rho, ux, uy)
        f_post = f - (f - feq) / case.tau

        # --- streaming -------------------------------------------------------
        f = _stream(f_post, fluid)

        # --- streamwise Zou-He closure --------------------------------------
        rho_in, rho_out = end_densities(step)
        if case.arm == "pressure-pressure":
            zou_he_wall(f, "west", mode="pressure", rho=rho_in)
            zou_he_wall(f, "east", mode="pressure", rho=rho_out)
        elif case.arm == "velocity-pressure":
            scale = 1.0 if case.ramp_steps <= 0 else min(1.0, step / float(case.ramp_steps))
            zou_he_wall(f, "west", mode="velocity", ux=full_profile(case, scale), uy=0.0)
            zou_he_wall(f, "east", mode="pressure", rho=rho_out)
        elif case.arm == "equilibrium-pressure":
            scale = 1.0 if case.ramp_steps <= 0 else min(1.0, step / float(case.ramp_steps))
            _equilibrium_inlet(f, full_profile(case, scale))
            zou_he_wall(f, "east", mode="pressure", rho=rho_out)
        else:
            raise ValueError(f"unknown arm {case.arm!r}")

        # --- block time averaging of the (oscillating) soft-BC field ---------
        if step > case.burn_in:
            ux_sum += ux
            rho_sum += rho
            navg += 1
            if navg % case.avg_window == 0:
                mean = ux_sum / navg
                if block_mean is not None:
                    u_scale = max(mean.abs().max().item(), 1e-30)
                    drift = (mean - block_mean).abs().max().item() / u_scale
                    drift_hist.append(drift)
                    drift_steps.append(step)
                    prof_now = mean[1:ny - 1, :].mean(dim=1)
                    if prof_prev is not None:
                        drift_profile_hist.append(
                            (prof_now - prof_prev).abs().max().item()
                            / max(prof_now.abs().max().item(), 1e-30)
                        )
                    prof_prev = prof_now.clone()
                    if verbose:
                        print(f"  step {step:6d}  block drift = {drift:.3e}")
                    if drift < case.tol:
                        converged = True
                block_mean = mean.clone()
                block_start = step
                if converged:
                    break

        if step % case.check_every == 0:
            # streamwise mass flux, per column, over the *fluid* rows only: a
            # steady incompressible channel must have constant q(x).  The solid
            # rows carry no physical populations (they are never sources for a
            # fluid target -- links that end in a wall are closed by bounce-back),
            # so including them in a flux sum is meaningless.
            cx = C2Q9[:, 0].view(9, 1, 1).to(case.dtype)
            q_col = (f[:, 1:ny - 1, :] * cx).sum(dim=(0, 1))
            mass_defect = (q_col[0] - q_col[-1]).abs().item() / max(
                q_col[1:-1].abs().mean().item(), 1e-30
            )
            if verbose and (step % (case.check_every * 20) == 0):
                print(f"  step {step:6d}  q(x) defect = {mass_defect:.3e}")

    rho = f.sum(0)
    ux = (f * C2Q9[:, 0].view(9, 1, 1).to(case.dtype)).sum(0) / rho
    uy = (f * C2Q9[:, 1].view(9, 1, 1).to(case.dtype)).sum(0) / rho
    if navg > 0:                       # report the time-averaged field
        ux = ux_sum / navg
        rho = rho_sum / navg

    # --- gradient from the computed density field (least squares in x) -------
    interior = slice(1, ny - 1)
    rho_bar = rho[interior, :].mean(dim=0)
    x = torch.arange(nx, dtype=case.dtype)
    xm, rm = x.mean(), rho_bar.mean()
    slope = ((x - xm) * (rho_bar - rm)).sum() / ((x - xm) ** 2).sum()
    g_measured = -slope.item() / 3.0

    u_mid = ux[interior, :].mean(dim=1)
    prof_meas = ux[interior, nx // 2].clone()

    q_col = (f[:, 1:ny - 1, :] * C2Q9[:, 0].view(9, 1, 1).to(case.dtype)).sum(dim=(0, 1))

    return {
        "case": case,
        "steps": step,
        "navg": navg,
        "converged": converged,
        "drift_hist": drift_hist,
        "drift_steps": drift_steps,
        "drift_last": (drift_hist[-1] if drift_hist else float("nan")),
        "drift_profile_hist": drift_profile_hist,
        "drift_profile_last": (
            drift_profile_hist[-1] if drift_profile_hist else float("nan")
        ),
        "q_col": q_col,
        "f": f,
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "x": x,
        "rho_bar": rho_bar,
        "rho_slope": slope.item(),
        "g_measured": g_measured,
        "u_profile_mid": u_mid,
        "u_profile": prof_meas,
        "mass_defect": mass_defect,
    }


def profile_metrics(case: ChannelCase, u_sim: torch.Tensor, g: float) -> Dict[str, float]:
    """Compare a simulated profile with the exact parabola for gradient ``g``."""
    u_ref = analytic_profile(case, g)
    scale = u_ref.abs().max().item()
    diff = u_sim - u_ref
    l2 = (diff.pow(2).mean().sqrt() / scale).item()
    linf = (diff.abs().max() / scale).item()
    umax_err = abs(u_sim.abs().max().item() - scale) / scale
    # normalised (shape-only) error removes any bulk flow-rate offset
    shape = u_sim / u_sim.abs().max().item() - u_ref / scale
    shape_l2 = shape.pow(2).mean().sqrt().item()
    return {
        "u_max_sim": u_sim.abs().max().item(),
        "u_max_ref": scale,
        "rel_l2": l2,
        "rel_linf": linf,
        "rel_umax": umax_err,
        "shape_rel_l2": shape_l2,
    }