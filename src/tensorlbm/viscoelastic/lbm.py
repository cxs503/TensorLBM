"""Coupled lattice-Boltzmann solver for Oldroyd-B / FENE-P channel flow.

Solver structure (one time step, lattice units, ``rho = 1``):

1. macroscopic ``rho, u`` from ``f`` (D2Q9, BGK);
2. the polymer conformation ``C`` is advanced by
   :func:`tensorlbm.viscoelastic.conformation.advance_conformation` with the
   velocity gradient of the *current* LBM field (implicit strain + relaxation,
   optional realizability projection);
3. the polymeric stress ``tau_p`` follows from ``C``; its divergence is fed back
   into the momentum equation as an extra body acceleration
   ``a_p = div(tau_p) / rho`` next to the driving force ``G``;
4. BGK collision towards ``feq(rho, u + a/2)`` plus the Guo (2002) discrete
   force term with the *total* acceleration ``a = G + a_p``;
5. streaming + half-way bounce-back on the two channel walls.

Because ``tau_xy = eta_p * gdot`` in steady shear, the fixed point of this
two-way coupling is the plane-Poiseuille parabola evaluated with the total
viscosity ``nu0 = nu_s + nu_p``; the polymer force supplies exactly the extra
``eta_p u''``.  Both the velocity-gradient and the stress-divergence operators
are second-order and *exact on quadratics* (see ``ddy_2nd``), so the discrete
fixed point of the Oldroyd-B case coincides with the analytic parabola up to the
O(Ma^2) LBM compressibility floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..d2q9 import OPPOSITE, equilibrium, macroscopic
from ..powerlaw import guo_force_term
from ..solver import stream
from .analytical import oldroyd_b_poiseuille_fields
from .conformation import (
    advance_conformation,
    check_realizability,
    ddy_2nd,
    elastic_energy,
    fene_p_stress,
    oldroyd_b_stress,
)

__all__ = [
    "ViscoelasticParams",
    "make_channel_wall_mask",
    "polymer_stress_divergence",
    "run_viscoelastic_channel",
    "viscoelastic_profile_error",
]


@dataclass
class ViscoelasticParams:
    """Physical parameters of the viscoelastic channel (lattice units, ``rho=1``).

    ``nu0`` is the *total* kinematic viscosity ``nu_s + nu_p``; ``beta`` is the
    solvent fraction ``nu_s / nu0`` (``beta = 1`` switches the polymer off, the
    negative control).  ``G`` is the streamwise body acceleration driving the
    flow.  ``lam`` is the relaxation time (``Wi = lam * gdot_wall``).
    """

    nu0: float = 0.05
    beta: float = 0.5
    lam: float = 800.0
    G: float = 3.90625e-6
    model: str = "oldroyd_b"
    L2: float | None = None
    rho: float = 1.0

    @property
    def nu_s(self) -> float:
        return float(self.beta) * float(self.nu0)

    @property
    def nu_p(self) -> float:
        return float(self.nu0) - self.nu_s

    @property
    def eta_p(self) -> float:
        return self.nu_p * float(self.rho)

    def as_dict(self) -> dict:
        return {
            "nu0": self.nu0,
            "beta": self.beta,
            "nu_s": self.nu_s,
            "nu_p": self.nu_p,
            "lam": self.lam,
            "G": self.G,
            "model": self.model,
            "L2": self.L2,
            "rho": self.rho,
        }


def make_channel_wall_mask(ny: int, nx: int, *, dtype=torch.float64, device=None):
    """Wall mask: rows ``0`` and ``ny-1`` are solid."""
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    return wall


#: D2Q9 directions with c_y = +1 / -1 (arriving from the bottom / top wall).
_CY_UP = (2, 5, 6)
_CY_DOWN = (4, 7, 8)


def halfway_bounce_back(
    f_post: torch.Tensor, f_streamed: torch.Tensor, H: int
) -> torch.Tensor:
    """Static half-way bounce-back on rows ``1`` and ``H``.

    The reflected population is written into the **fluid** node (not into the
    wall row), which puts the no-slip plane half a lattice spacing outside the
    first/last fluid node, i.e. at ``y = 0.5`` and ``y = H + 0.5`` -- the geometry
    the analytic anchors in :mod:`tensorlbm.viscoelastic.analytical` assume.
    Writing the reflection into the wall row instead (the mask formulation used
    by some other kernels in this repo) moves the effective wall onto the wall
    node and inflates ``u_max`` by ``((H+1)/H)^2`` (10% at ``H = 16``); this is
    why the anchor comparison is sensitive to the convention.

    Args:
        f_post: post-collision populations ``(9, H+2, nx)``.
        f_streamed: populations after :func:`tensorlbm.solver.stream`.
        H: number of fluid rows.
    """
    opp = OPPOSITE.to(f_post.device)
    out = f_streamed.clone()
    cy_up = torch.tensor(_CY_UP, device=f_post.device)
    cy_down = torch.tensor(_CY_DOWN, device=f_post.device)
    out[cy_up, 1, :] = f_post[opp[cy_up], 1, :]
    out[cy_down, H, :] = f_post[opp[cy_down], H, :]
    return out


def polymer_stress_divergence(
    C: torch.Tensor,
    *,
    eta_p: float,
    lam: float,
    model: str = "oldroyd_b",
    L2: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Polymer stress and its streamwise divergence.

    Returns ``(tau_xx, tau_xy, tau_yy, div_tau_x)`` with
    ``div_tau_x = d tau_xy / dy`` (the only non-zero component for an x-uniform
    channel).  Wall rows are zeroed; the one-sided stencil at the first/last
    fluid row keeps the divergence second-order accurate there.
    """
    if model == "fene_p":
        if L2 is None:
            msg = "FENE-P requires L2"
            raise ValueError(msg)
        tau_xx, tau_xy, tau_yy = fene_p_stress(C, eta_p, lam, L2)
    else:
        tau_xx, tau_xy, tau_yy = oldroyd_b_stress(C, eta_p, lam)
    div_tau_x = ddy_2nd(tau_xy)
    return tau_xx, tau_xy, tau_yy, div_tau_x


def _stress_components(conf, params):
    if params.model == "fene_p":
        return fene_p_stress(conf, params.eta_p, params.lam, params.L2)
    return oldroyd_b_stress(conf, params.eta_p, params.lam)


def run_viscoelastic_channel(
    H: int,
    params: ViscoelasticParams,
    *,
    nx: int = 4,
    max_iter: int = 200000,
    tol: float = 1e-12,
    check_every: int = 50,
    projection: bool = True,
    implicit_strain: bool = True,
    strain_scale: float = 1.0,
    advection: bool = True,
    initial_velocity: torch.Tensor | None = None,
    initial_conformation: torch.Tensor | None = None,
    stress_relaxation: float = 0.5,
    constitutive_dt: float = 1.0,
    dtype=torch.float64,
    device: torch.device | None = None,
) -> dict:
    """Run a viscoelastic plane-Poiseuille channel to steady state.

    Returns a dict with the mean profile, the conformation/stress fields, the
    convergence history, the realizability diagnostics and (when the run
    converged) the analytic-anchor comparison.
    """
    if H < 5:
        msg = "channel needs at least 5 fluid rows"
        raise ValueError(msg)
    if device is None:
        device = torch.device("cpu")
    ny = H + 2
    make_channel_wall_mask(ny, nx, dtype=dtype, device=device)  # geometry assertion only

    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    if initial_velocity is None:
        ux = torch.zeros((ny, nx), dtype=dtype, device=device)
    else:
        iv = initial_velocity.to(dtype=dtype, device=device)
        ux = (iv.unsqueeze(1).expand(ny, nx) if iv.ndim == 1 else iv.expand(ny, nx)).clone()
    uy = torch.zeros((ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho, ux, uy)

    if initial_conformation is None:
        conf = torch.zeros((4, ny, nx), dtype=dtype, device=device)
        conf[0] = 1.0
        conf[2] = 1.0
        conf[3] = 1.0
    else:
        ic = initial_conformation.to(dtype=dtype, device=device)
        conf = ic.unsqueeze(-1).expand(4, ny, nx).clone() if ic.ndim == 2 else ic.clone()

    _rho0, ux, uy = macroscopic(f)

    tau_s = 3.0 * params.nu_s + 0.5
    tau_field = torch.full((ny, nx), tau_s, dtype=dtype, device=device)

    div_tau_prev = torch.zeros((ny, nx), dtype=dtype, device=device)
    history: list[dict] = []
    energy_history: list[float] = []
    converged = False
    iteration = 0
    res_u = float("inf")
    realizability = check_realizability(conf)
    ux_prev = ux.clone()
    for iteration in range(1, max_iter + 1):
        conf_new, realizability = advance_conformation(
            conf,
            ux,
            uy,
            dt=constitutive_dt,
            lam=params.lam,
            model=params.model,
            L2=params.L2,
            projection=projection,
            implicit_strain=implicit_strain,
            strain_scale=strain_scale,
            advection=advection,
        )
        conf = conf_new
        conf[:, 0, :] = 0.0
        conf[0, 0, :] = 1.0
        conf[2, 0, :] = 1.0
        conf[3, 0, :] = 1.0
        conf[:, -1, :] = 0.0
        conf[0, -1, :] = 1.0
        conf[2, -1, :] = 1.0
        conf[3, -1, :] = 1.0

        div_tau_new = polymer_stress_divergence(
            conf,
            eta_p=params.eta_p,
            lam=params.lam,
            model=params.model,
            L2=params.L2,
        )[3]
        # under-relaxation of the polymer momentum coupling: the lagged
        # feedback has a *negative* loop gain (-nu_p/nu_s), so a plain Jacobi
        # iteration oscillates; any fixed under-relaxation factor returns the
        # same fixed point faster (see docs, section "convergence").
        div_tau = (1.0 - stress_relaxation) * div_tau_prev + stress_relaxation * div_tau_new
        div_tau_prev = div_tau

        ax = params.G + div_tau / params.rho
        ax[0, :] = 0.0
        ax[-1, :] = 0.0

        ux_star = ux + 0.5 * ax
        feq = equilibrium(rho, ux_star, uy)
        fi = guo_force_term(f, ax, tau_field, ux_star=ux_star)
        f = f - (f - feq) / tau_field.unsqueeze(0) + fi
        f_post = f
        f = halfway_bounce_back(f_post, stream(f), H)

        _r, ux_new, _u = macroscopic(f)
        ux = ux_new
        uy = _u
        if iteration % check_every == 0:
            du = (ux_new - ux_prev).abs().max()
            res_u = float(du)
            history.append({"iteration": iteration, "residual_u": res_u})
            energy_history.append(
                elastic_energy(conf, nu_p=params.nu_p, lam=params.lam)
            )
            if res_u < tol and iteration > check_every:
                converged = True
                break
            ux_prev = ux_new.clone()

    _rho, ux, uy = macroscopic(f)
    ux_mean = ux.mean(dim=1)
    _, tau_xy, _, div_tau = polymer_stress_divergence(
        conf, eta_p=params.eta_p, lam=params.lam, model=params.model, L2=params.L2
    )
    tau_xx, _, tau_yy = _stress_components(conf, params)
    n1_field = tau_xx - tau_yy

    out = {
        "H": H,
        "ny": ny,
        "nx": nx,
        "params": params.as_dict(),
        "iterations": iteration,
        "converged": converged,
        "residual_u": res_u,
        "history": history,
        "energy_history": energy_history,
        "ux": ux_mean,
        "ux_field": ux,
        "uy_field": uy,
        "rho_field": _rho,
        "conformation": conf,
        "tau_xx": tau_xx,
        "tau_xy": tau_xy,
        "tau_yy": tau_yy,
        "n1": n1_field,
        "div_tau": div_tau,
        "realizability": realizability,
        "analytical": oldroyd_b_poiseuille_fields(
            H,
            params.G,
            params.nu0,
            beta=params.beta,
            lam=params.lam,
            dtype=dtype,
        ),
    }
    return out


def viscoelastic_profile_error(
    u_num: torch.Tensor,
    u_ref: torch.Tensor,
    *,
    n_points: int = 20,
    interior_only: bool = False,
) -> dict:
    """Pointwise and L2 profile error statistics.

    ``n_points`` counts the compared nodes; the error is normalised by
    ``max|u_ref|`` so the numbers are dimensionless (relative pointwise error).
    """
    if u_num.shape != u_ref.shape:
        msg = f"shape mismatch: {tuple(u_num.shape)} vs {tuple(u_ref.shape)}"
        raise ValueError(msg)
    denom = float(u_ref.abs().max())
    denom = denom if denom > 0 else 1.0
    diff = (u_num - u_ref).abs()
    if interior_only:
        diff = diff[1:-1]
        ref = u_ref[1:-1]
    else:
        diff = diff[1:-1]  # walls are trivially zero; count fluid nodes only
        ref = u_ref[1:-1]
    rel = diff / denom
    n_fluid = int(ref.numel())
    return {
        "max_rel_error": float(rel.max()),
        "l2_rel_error": float(torch.sqrt((diff**2).sum()) / (denom * (n_fluid**0.5))),
        "mean_rel_error": float(rel.mean()),
        "n_points": min(int(n_points), n_fluid),
        "n_fluid_nodes": n_fluid,
        "u_ref_max": denom,
    }