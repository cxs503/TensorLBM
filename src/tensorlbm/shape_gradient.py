"""Flow-response shape derivatives for 2-D LBM (closes the ``adjoint.py`` gap).

Why this module exists
----------------------
``tensorlbm.adjoint`` (the "frozen-field surrogate") differentiates the
*objective only*, on a flow field produced by an ordinary forward solve.  The
time-stepping operator is not in the graph, therefore the surrogate cannot
answer "how would the flow have evolved if the geometry moved".  For a lift
objective this omission is not a small correction: the wake and the
wall-induced pressure asymmetry *are* the lift, so the frozen-field number is
not a degraded version of the shape derivative -- it is a different quantity
(measured in ``benchmarks/adjoint_verdict_diff_lbm.py``: cos(shipped
surrogate, unrolled AD) = -1.000, i.e. exactly anti-parallel).

This module supplies the quantity that *is* the shape derivative.  The disk is
represented by an analytic signed distance function

.. math:: \\phi(x; c, r) = |x - c| - r ,\\qquad
          w(x) = \\sigma\\big(\\phi(x)/\\epsilon\\big) \\in [0, 1]

(``w`` = *fluid* weight, saturated to exactly 0/1 beyond ``30 ε``), so the
centre ``c`` -- in particular the transverse position ``y_c`` -- is a 0-dim
tensor that stays in the autograd graph through the whole rollout.  The three
operators are the convex homotopies of the hard-mask chain, so at ``ε → 0``
the solver degenerates to a staircase-mask D2Q9 BGK channel solver:

* collision with Guo forcing, skipped softly:
  :math:`f \\leftarrow w\\,\\mathrm{collide}(f) + (1-w)\\,f`;
* periodic streaming (a permutation, hence mass conserving to round-off);
* soft full-way bounce-back on the node:
  :math:`f \\leftarrow w\\,f + (1-w)\\,f[\\bar q]`;
* soft momentum-exchange force (Ladd wet-node form):
  :math:`F_\\alpha = 2\\sum_x (1-w(x))\\sum_q c_{q\\alpha} f_{\\text{probe}}[q,x]`.

Method references
-----------------
* LBM equation set and the Guo forcing term: Krüger, Kusumaatmaja, Kuzmin,
  Shardt, Silva & Viggen, *The Lattice Boltzmann Method* (Springer 2017),
  ch. 5 (forcing, Eq. 5.58) and ch. 6 (bounce-back / momentum exchange).
* Momentum-exchange force (Ladd): Ladd, *J. Fluid Mech.* 271 (1994) 285;
  Ladd, *J. Fluid Mech.* 271 (1994) 311.
* Shape derivative / Hadamard structure and the need for the flow response:
  Pironneau, *J. Fluid Mech.* 59 (1973) 97; Mohammadi & Pironneau,
  *Applied Shape Optimization for Fluids* (2nd ed., 2010), ch. 2-3.
* Discrete-adjoint-through-the-solver (what this module does for a geometric
  parameter): Ataei & Salehipour, *Comput. Phys. Commun.* 300 (2024) 109187
  (XLB, differentiable LBM); Wassing et al., *Differentiable fluid simulations
  through the Lattice Boltzmann Method* (2024); McNamara et al. (2021) on
  backpropagation through LBM for inverse problems.
* Soft/diffuse solid as a differentiable homotopy of the staircase mask: the
  same construction as :mod:`tensorlbm.soft_geometry` (3-D) in two dimensions.

Contract
--------
:func:`shape_gradient_through_flow` returns
``shape_derivative_only=False`` together with ``includes_flow_response=True``:
the gradient it reports is the derivative of the *simulated* observable, not of
a frozen field.  Its frozen-field counterpart
(:func:`tensorlbm.adjoint.adjoint_sensitivity`) reports
``shape_derivative_only=True`` and is gated out of the optimisation path
(:func:`tensorlbm.adjoint.optimization_path_eligible`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic

__all__ = [
    "ChannelDisk",
    "SteadySolution",
    "disk_fluid_weight",
    "stream_periodic",
    "step",
    "rollout",
    "run_to_steady",
    "lift_from_probe",
    "drag_from_probe",
    "poiseuille_umax",
    "make_channel",
    "shape_gradient_through_flow",
    "frozen_field_gradient",
]

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _wall_mask(ny: int, nx: int, wall_thickness: int) -> torch.Tensor:
    m = torch.zeros(ny, nx, dtype=torch.bool)
    m[:wall_thickness, :] = True
    m[ny - wall_thickness :, :] = True
    return m


@dataclass
class ChannelDisk:
    """2-D plane channel (periodic in x, solid walls in y) with an SDF disk.

    Args:
        nx, ny: lattice size.
        diameter: disk diameter (lattice units).
        center_x, center_y: disk centre (floats or graph-connected 0-dim
            tensors; a tensor keeps the whole rollout differentiable w.r.t. it).
        tau: BGK relaxation time (> 0.5).
        body_force: x-directed body force per unit mass driving the channel.
        epsilon: SDF temperature of the soft disk (lattice units).
        wall_thickness: number of solid rows at each y wall (>= 2; the fluid
            rows are ``wall_thickness .. ny-1-wall_thickness``).
    """

    nx: int = 256
    ny: int = 96
    diameter: float | torch.Tensor = 16.0
    center_x: float | torch.Tensor = 64.0
    center_y: float | torch.Tensor = 48.0
    tau: float | torch.Tensor = 0.56
    body_force: float = 2.0e-7
    epsilon: float = 0.8
    wall_thickness: int = 3
    dtype: torch.dtype = field(default=_DTYPE)

    # -- grids -------------------------------------------------------------

    def coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=self.dtype),
            torch.arange(self.nx, dtype=self.dtype),
            indexing="ij",
        )
        return yy, xx

    def wall_mask(self) -> torch.Tensor:
        """Boolean solid mask of the two channel walls (static geometry)."""
        return _wall_mask(self.ny, self.nx, self.wall_thickness)

    def hard_mask(self) -> torch.Tensor:
        """Staircase mask of the disk (``phi < 0``) -- the frozen-field input."""
        yy, xx = self.coords()
        d = (xx - float(self.center_x)) ** 2 + (yy - float(self.center_y)) ** 2
        return d < (float(self.diameter) / 2.0) ** 2

    # -- physical scalars --------------------------------------------------

    def channel_height(self) -> float:
        """Effective wall-to-wall distance (walls sit half a cell inside)."""
        return float(self.ny - 2 * self.wall_thickness)

    def kinematic_viscosity(self) -> float:
        return (float(self.tau) - 0.5) / 3.0

    def bulk_velocity(self) -> float:
        """Analytic bulk velocity of the force-driven channel: ``g H^2 / (12 nu)``."""
        h = self.channel_height()
        return self.body_force * h * h / (12.0 * self.kinematic_viscosity())

    def reynolds(self) -> float:
        return self.bulk_velocity() * float(self.diameter) / self.kinematic_viscosity()


def disk_fluid_weight(
    geom: ChannelDisk,
    *,
    center_y: float | torch.Tensor | None = None,
    center_x: float | torch.Tensor | None = None,
    diameter: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Fluid weight ``w = sigmoid(phi/eps)`` of the soft disk, saturated to 0/1.

    ``w = 1`` in the fluid, ``w = 0`` inside the disk.  Any of the geometric
    parameters may be overridden by a graph-connected tensor.
    """
    cy = geom.center_y if center_y is None else center_y
    cx = geom.center_x if center_x is None else center_x
    d = geom.diameter if diameter is None else diameter
    yy, xx = geom.coords()
    r = 0.5 * (d if isinstance(d, torch.Tensor) else torch.as_tensor(d, dtype=geom.dtype))
    phi = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-30) - r  # < 0 inside solid
    ratio = phi / geom.epsilon
    w = torch.sigmoid(ratio)
    # saturate: |phi/eps| > 30 -> exactly 1 / 0 (logistic tail ~1e-13)
    w = torch.where(ratio > 30.0, torch.ones_like(w), w)
    w = torch.where(ratio < -30.0, torch.zeros_like(w), w)
    return w


# ---------------------------------------------------------------------------
# lattice operators
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _consts(dtype: torch.dtype) -> dict:
    """Static lattice tables (cached: building them per step dominates small grids)."""
    from tensorlbm.d2q9 import C

    w = torch.tensor([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=dtype)
    return {
        "w": w.view(9, 1, 1),
        "cx": C[:, 0].to(dtype).view(9, 1, 1),
        "cy": C[:, 1].to(dtype).view(9, 1, 1),
        "opp": OPPOSITE.to(torch.int64),
        "c": C.to(dtype),
    }


def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """D2Q9 streaming, periodic in both directions (a pure permutation)."""
    from tensorlbm.d2q9 import C

    out = []
    for q in range(9):
        cy, cx = int(C[q, 1]), int(C[q, 0])
        out.append(torch.roll(f[q], shifts=(cy, cx), dims=(-2, -1)))
    return torch.stack(out, dim=0)


def step(
    f: torch.Tensor,
    geom: ChannelDisk,
    *,
    center_y: float | torch.Tensor | None = None,
    center_x: float | torch.Tensor | None = None,
    diameter: float | torch.Tensor | None = None,
    fluid_weight: torch.Tensor | None = None,
    force: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One D2Q9 BGK step: collision (Guo forcing) -> streaming -> soft bounce-back.

    Returns ``(f_new, probe)`` where ``probe`` is the post-stream / pre-bounce
    state used by the momentum-exchange force; both stay in the autograd graph.
    """
    par = _consts(f.dtype)
    dtype = f.dtype
    tau = geom.tau if isinstance(geom.tau, torch.Tensor) else torch.as_tensor(geom.tau, dtype=dtype)
    w_disk = (
        disk_fluid_weight(geom, center_y=center_y, center_x=center_x, diameter=diameter)
        if fluid_weight is None
        else fluid_weight
    )
    wall = geom.wall_mask().to(f.device)
    w = torch.where(wall, torch.zeros_like(w_disk), w_disk)

    g = geom.body_force if force is None else force
    g = g if isinstance(g, torch.Tensor) else torch.as_tensor(g, dtype=dtype)

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    # Guo forcing (Krüger et al. 2017, Eq. 5.58), applied to fluid nodes only.
    cx, cy = par["cx"], par["cy"]
    cu = cx * ux + cy * uy
    src = 3.0 * par["w"] * ((cx - ux) + 3.0 * cu * cx) * g * w
    f_coll = f - (f - feq) / tau + (1.0 - 1.0 / (2.0 * tau)) * src
    f_post = w * f_coll + (1.0 - w) * f  # soft NoDynamics (solid is frozen)

    probe = stream_periodic(f_post)  # momentum-exchange probe
    flip = torch.index_select(probe, 0, par["opp"])
    f_new = w * probe + (1.0 - w) * flip  # soft full-way bounce-back
    return f_new, probe


def rollout(
    f: torch.Tensor,
    geom: ChannelDisk,
    n_steps: int,
    *,
    center_y: float | torch.Tensor | None = None,
    checkpoint: bool = False,
    collect: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Advance ``n_steps`` steps keeping the graph (optional checkpointing)."""
    probes: list[torch.Tensor] = []
    w = disk_fluid_weight(geom, center_y=center_y)  # geometry evaluated once per rollout
    for _ in range(n_steps):
        if checkpoint:
            f, probe = torch.utils.checkpoint.checkpoint(
                lambda ff: step(ff, geom, fluid_weight=w),
                f,
                use_reentrant=False,
            )
        else:
            f, probe = step(f, geom, fluid_weight=w)
        if collect:
            probes.append(probe)
    return f, probes


@dataclass
class SteadySolution:
    """Converged state of a forward (gradient-free) channel solve."""

    f: torch.Tensor
    geom: ChannelDisk
    lift: float
    drag: float
    steps: int
    residual: float
    history: list[float]

    def fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return macroscopic(self.f)

    def solid_weight(self) -> torch.Tensor:
        w = disk_fluid_weight(self.geom)
        return torch.where(self.geom.wall_mask(), torch.zeros_like(w), 1.0 - w)


def lift_from_probe(probe: torch.Tensor, solid_weight: torch.Tensor) -> torch.Tensor:
    """Momentum-exchange force on the disk: ``2 sum_x s(x) sum_q c_q f_probe``."""
    c = _consts(probe.dtype)["c"]
    mom = (probe * solid_weight.to(probe.dtype)).sum(dim=(1, 2))
    return 2.0 * torch.matmul(mom, c)


def drag_from_probe(probe: torch.Tensor, solid_weight: torch.Tensor) -> torch.Tensor:
    return lift_from_probe(probe, solid_weight)[0]


def lift_from_probe_y(probe: torch.Tensor, solid_weight: torch.Tensor) -> torch.Tensor:
    return lift_from_probe(probe, solid_weight)[1]


def poiseuille_umax(geom: ChannelDisk) -> float:
    """Analytic centre-line velocity of the force-driven channel, ``g H^2/(8 nu)``."""
    h = geom.channel_height()
    return geom.body_force * h * h / (8.0 * geom.kinematic_viscosity())


def make_channel(
    *,
    nx: int,
    ny: int,
    diameter: float,
    center_y: float | torch.Tensor,
    tau: float = 0.56,
    u_bulk: float = 0.05,
    epsilon: float = 0.8,
    wall_thickness: int = 3,
    center_x: float | None = None,
) -> ChannelDisk:
    """Channel whose body force is chosen to give ``u_bulk`` in the empty channel.

    ``g = 12 nu u_bulk / H^2`` (Poiseuille, bulk velocity).  ``H = ny - 2*wall``
    is the effective wall-to-wall distance; the centre line sits at
    ``y = (ny-1)/2``.
    """
    h = float(ny - 2 * wall_thickness)
    nu = (tau - 0.5) / 3.0
    g = 12.0 * nu * u_bulk / (h * h)
    cx = 0.25 * nx if center_x is None else center_x
    return ChannelDisk(
        nx=nx,
        ny=ny,
        diameter=diameter,
        center_x=cx,
        center_y=center_y,
        tau=tau,
        body_force=g,
        epsilon=epsilon,
        wall_thickness=wall_thickness,
    )


def poiseuille_profile(geom: ChannelDisk) -> torch.Tensor:
    """Analytic centre-line parabola of the force-driven channel (lattice units).

    ``u(y) = g/(2 nu) (y - y0)(y1 - y)`` with the walls halfway between the last
    fluid row and the first solid row, ``y0 = wall - 1/2``, ``y1 = ny - wall - 1/2``.
    """
    yy, _ = geom.coords()
    w = geom.wall_thickness
    y0 = w - 0.5
    y1 = geom.ny - w - 0.5
    u = geom.body_force / (2.0 * geom.kinematic_viscosity()) * (yy - y0) * (y1 - yy)
    return torch.clamp(u, min=0.0)


def initial_condition(geom: ChannelDisk, *, init: str = "poiseuille") -> torch.Tensor:
    """Initial distribution: rest, or the analytic parabola (removes the spin-up)."""
    rho = torch.ones(geom.ny, geom.nx, dtype=geom.dtype)
    if init == "rest":
        ux = torch.zeros_like(rho)
    elif init == "poiseuille":
        ux = poiseuille_profile(geom).expand(geom.ny, geom.nx).contiguous()
    else:
        raise ValueError(f"init must be 'rest' or 'poiseuille', got {init!r}")
    return equilibrium(rho, ux, torch.zeros_like(rho))


# ---------------------------------------------------------------------------
# forward solve
# ---------------------------------------------------------------------------


def run_to_steady(
    geom: ChannelDisk,
    *,
    max_steps: int = 20000,
    tol: float = 1e-10,
    check_every: int = 100,
    window: int = 20,
    min_steps: int = 500,
    init: str = "poiseuille",
    keep_history: bool = False,
) -> SteadySolution:
    """Run the forward solver until the lift/drag stop drifting.

    Convergence criterion: over the last ``check_every`` steps the change of
    (``-C_D``, ``C_L``) normalised by their magnitudes is below ``tol``.
    """
    f = initial_condition(geom, init=init)
    s = SteadySolution(f=f, geom=geom, lift=0.0, drag=0.0, steps=0, residual=1.0, history=[])
    with torch.no_grad():
        w = disk_fluid_weight(geom)
        wall = geom.wall_mask()
        prev: tuple[float, float] | None = None
        force_hist: list[float] = []
        for it in range(1, max_steps + 1):
            f, probe = step(f, geom, fluid_weight=w)
            sw = torch.where(wall, torch.zeros_like(w), 1.0 - w)
            F = lift_from_probe(probe, sw)
            if keep_history:
                force_hist.append(float(F.norm()))
            if it % check_every == 0:
                drag_f = float(F[0])
                lift_f = float(F[1])
                cur = (drag_f, lift_f)
                if prev is not None:
                    scale = max(abs(drag_f), abs(lift_f), 1e-30)
                    res = (abs(cur[0] - prev[0]) + abs(cur[1] - prev[1])) / scale
                    s.history.append(res)
                    if res < tol and it >= min_steps:
                        s.f, s.drag, s.lift, s.steps, s.residual = f, drag_f, lift_f, it, res
                        return s
                prev = cur
                s.residual = 1.0
        s.f, s.drag, s.lift, s.steps = f, float(F[0]), float(F[1]), max_steps
    return s


# ---------------------------------------------------------------------------
# flow-response shape derivative
# ---------------------------------------------------------------------------


def shape_gradient_through_flow(
    steady: SteadySolution,
    *,
    n_unroll: int = 400,
    window: int = 50,
    checkpoint: bool = False,
    u_ref: float | None = None,
    differentiate: tuple[str, ...] = ("y_c",),
) -> dict:
    """``dC_L/dy_c`` with the flow response included (unrolled discrete adjoint).

    The converged field is the fixed point of the solver; the requested
    geometric parameters are re-attached to the graph and the last ``n_unroll``
    steps are replayed with autograd, the lift being averaged over the last
    ``window`` steps.  As ``n_unroll`` grows this converges to the steady shape
    derivative (the fixed-point / "quasi-steady" sensitivity) -- the measured
    convergence in ``n_unroll`` is part of
    ``benchmarks/adjoint_verdict_diff_lbm.py``.

    Args:
        steady: converged forward solution (:func:`run_to_steady`).
        n_unroll: number of steps replayed with the graph.
        window: number of trailing steps averaged into the objective.
        checkpoint: wrap every replayed step in gradient checkpointing (flat
            activation memory, identical gradients).
        u_ref: reference velocity of the ``C_L`` normalisation (default: the
            analytic bulk velocity of the empty channel).
        differentiate: any subset of ``("y_c", "x_c", "diameter")``.

    Returns:
        dict with ``dJ_dy_c`` (kept for convenience), ``grad`` (per parameter),
        ``grad_vector`` (ordered as ``differentiate``), the objective value,
        and the explicit contract flags ``shape_derivative_only=False`` /
        ``includes_flow_response=True``.
    """
    geom = steady.geom
    u = u_ref if u_ref is not None else geom.bulk_velocity()
    scale = 0.5 * u * u * float(geom.diameter)

    params: dict[str, torch.Tensor] = {}
    base = {
        "y_c": float(geom.center_y),
        "x_c": float(geom.center_x),
        "diameter": float(geom.diameter),
    }
    for key in differentiate:
        if key not in base:
            raise ValueError(f"differentiate must be subset of {sorted(base)}, got {key!r}")
        params[key] = torch.tensor(base[key], dtype=geom.dtype, requires_grad=True)
    # non-differentiated parameters are held at their converged values
    live = {"center_y": params.get("y_c", base["y_c"]), "center_x": params.get("x_c", base["x_c"]),
            "diameter": params.get("diameter", base["diameter"])}

    f = steady.f.detach().clone()
    J = torch.zeros((), dtype=geom.dtype)
    n_meas = 0
    for k in range(n_unroll):
        args = (f,)
        if checkpoint:
            f, probe = torch.utils.checkpoint.checkpoint(
                lambda ff: step(
                    ff,
                    geom,
                    center_y=live["center_y"],
                    center_x=live["center_x"],
                    diameter=live["diameter"],
                ),
                *args,
                use_reentrant=False,
            )
        else:
            f, probe = step(
                f,
                geom,
                center_y=live["center_y"],
                center_x=live["center_x"],
                diameter=live["diameter"],
            )
        if k >= n_unroll - window:
            w = disk_fluid_weight(
                geom,
                center_y=live["center_y"],
                center_x=live["center_x"],
                diameter=live["diameter"],
            )
            sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
            J = J + lift_from_probe(probe, sw)[1] / scale
            n_meas += 1
    J = J / max(n_meas, 1)
    J.backward()
    grad = {key: float(params[key].grad) for key in differentiate}
    return {
        "dJ_dy_c": grad.get("y_c"),
        "grad": grad,
        "grad_vector": [grad[key] for key in differentiate],
        "differentiate": list(differentiate),
        "objective": float(J),
        "n_unroll": n_unroll,
        "window": window,
        "y_c": float(geom.center_y),
        "shape_derivative_only": False,
        "includes_flow_response": True,
        "optimization_path_eligible": True,
        "method": "unrolled discrete adjoint (AD through the LBM step chain)",
    }


def frozen_field_gradient(
    steady: SteadySolution,
    *,
    u_ref: float | None = None,
    differentiate: tuple[str, ...] = ("y_c",),
) -> dict:
    """Exact frozen-field derivative of the same objective (theoretical ceiling).

    ``C_L(y_c) = 2 sum_x s(x; y_c) sum_q c_{q,y} f_probe[q, x] / scale`` with
    ``f_probe`` frozen at the converged state: only the geometry weight
    ``s(x; y_c)`` moves, so the gradient contains the *explicit* (mask/metric)
    shape term and nothing of the flow response ``dJ/df . df/dy_c``.  Any
    frozen-field surrogate -- this one included, by construction -- is bounded
    by this quantity; if it fails the cosine gate, no frozen-field method can
    pass.
    """
    geom = steady.geom
    u = u_ref if u_ref is not None else geom.bulk_velocity()
    scale = 0.5 * u * u * float(geom.diameter)

    base = {"y_c": float(geom.center_y), "x_c": float(geom.center_x),
            "diameter": float(geom.diameter)}
    params = {
        key: torch.tensor(base[key], dtype=geom.dtype, requires_grad=True)
        for key in differentiate
    }
    live = {"center_y": params.get("y_c", base["y_c"]), "center_x": params.get("x_c", base["x_c"]),
            "diameter": params.get("diameter", base["diameter"])}

    with torch.no_grad():
        _, probe = step(steady.f.clone(), geom)
    probe = probe.detach()
    w = disk_fluid_weight(
        geom, center_y=live["center_y"], center_x=live["center_x"], diameter=live["diameter"]
    )
    sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
    J = lift_from_probe(probe, sw)[1] / scale
    J.backward()
    grad = {key: float(params[key].grad) for key in differentiate}
    return {
        "dJ_dy_c": grad.get("y_c"),
        "grad": grad,
        "grad_vector": [grad[key] for key in differentiate],
        "differentiate": list(differentiate),
        "objective": float(J),
        "method": "frozen-field: derivative through the geometry weight only",
        "shape_derivative_only": True,
        "includes_flow_response": False,
        "optimization_path_eligible": False,
    }
