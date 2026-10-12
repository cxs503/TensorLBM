"""Wall-model chain on octree-shell leaves — SoA port of ``wall_chain``.

Route C (owner directive 2026-10-11, common-module build): the wall-model
chain composition must not be copy-pasted into runners.  The grid form of
the chain lives in :mod:`tensorlbm.wall_chain` around
:func:`tensorlbm.wall_model.bfl_wall_function_3d`; this module is the
leaf (sparse SoA, ``(Q, n_leaf)``) counterpart composable with
:func:`tensorlbm.octree_boundary.stepping.step_octree_shell`:

* :func:`leaf_bfl_link_normals` — the BFL link-normal reconstruction of
  ``wall_model.compute_bfl_link_normal`` on the SoA ``bfl_mask`` (bitwise
  parity with the grid loop when fed the same mask flattened);
* :func:`leaf_wall_model_slip_velocity` — the ``wall_model_slip`` exchange
  closure of ``bfl_wall_function_3d`` Step 1 (no-penetration BFL wall
  velocity from the pre-stream leaf state, shear left to the stress law);
* :func:`leaf_solve_wall_law` — the Reichardt/log/... inversion via
  ``wall_model._solve_wall_law``, called directly so the arithmetic is
  the grid operator's, grouped per leaf level when the dual-viscosity
  ``nu`` is a per-leaf tensor;
* :func:`leaf_guo_body_force` — the Guo forcing of
  ``wall_model.guo_body_force_d3q19``/``d3q27`` on leaves (bitwise parity);
* :func:`leaf_wall_shear_traction` — Steps 3-5: tangential decomposition,
  wall-law solve, ``-tau_w * area`` traction applied through Guo forcing;
* :func:`leaf_dual_viscosity_wall_nu` — per-level wall-law viscosity at
  the *physical* Reynolds number
  (``wall_model.physical_wall_lattice_viscosity``), the leaf collision
  chain being stability-driven;
* :func:`build_leaf_wall_chain` — one call producing the geometric inputs
  (near-wall leaf set, normals, wall nu) with *finest-only* placement:
  the wall model lives on ``leaf_level == d_max`` leaves, where the
  boundary links actually are; coarser-band links are counted in the
  ``coarser_bfl_links`` diagnostic (their cells are overwritten by the
  finer restriction, the leaf analogue of
  ``wall_chain.count_boundary_links_outside_box`` vacuousness);
* :func:`make_leaf_wall_chain_bfl_fn` — drop-in ``bfl_fn`` factory for
  ``step_octree_shell`` wiring the above into the per-substep boundary
  stage, plus a :class:`LeafWallForceProbe` for the force split
  (single body/+drag convention by default since the libfix_20261011
  unification; the historical mixed reading survives as
  ``LeafWallForceProbe.legacy_mixed_force`` / ``convention="legacy_mixed"``).

One documented deviation from the grid chain: the grid tier decomposes
the stress with geometry normals (``compute_wall_normal`` solid-mask
gradient) and the slip closure with link normals.  Leaves do not carry a
solid mask at leaf resolution, so both use the BFL link normal here
(sign immaterial for the tangential projection).  A production runner
with analytical hull normals can override via ``stress_normals``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .bfl import _ramp_activation, bfl_apply_gather

__all__ = [
    "LeafWallChain",
    "LeafWallForceProbe",
    "build_leaf_wall_chain",
    "leaf_bfl_link_normals",
    "leaf_dual_viscosity_wall_nu",
    "leaf_guo_body_force",
    "leaf_solve_wall_law",
    "leaf_wall_model_slip_velocity",
    "leaf_wall_shear_traction",
    "make_leaf_wall_chain_bfl_fn",
]


def _lattice(Q: int, device: torch.device):
    if Q == 27:
        from tensorlbm.d3q27 import C, W
    elif Q == 19:
        from tensorlbm.d3q19 import C, W
    else:
        raise NotImplementedError(f"leaf wall chain supports D3Q19/D3Q27, got Q={Q}")
    return C.to(device=device, dtype=torch.float32), W.to(device=device, dtype=torch.float32)


def _leaf_macros(f: torch.Tensor, Q: int):
    """(rho, ux, uy, uz) per leaf from ``(Q, n_leaf)`` populations."""
    if Q == 27:
        from tensorlbm.d3q27 import macroscopic27

        rho, ux, uy, uz = macroscopic27(f.view(Q, 1, 1, -1))
    elif Q == 19:
        from tensorlbm.d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f.view(Q, 1, 1, -1))
    else:
        raise NotImplementedError(f"leaf wall chain supports D3Q19/D3Q27, got Q={Q}")
    return (rho.reshape(-1), ux.reshape(-1), uy.reshape(-1), uz.reshape(-1))


def leaf_bfl_link_normals(
    bfl_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth boundary normal from the crossing links, per leaf.

    The weighted sum ``sum_d w_d * mask_d * c_d`` points from a boundary
    leaf into the solid.  The accumulation loop replicates
    ``wall_model.compute_bfl_link_normal`` operation for operation
    (sequential over directions, weight-first product), so feeding the
    same mask flattened reproduces the grid normals bit for bit.
    """
    Q = bfl_mask.shape[0]
    c, w = _lattice(Q, bfl_mask.device)
    shape = bfl_mask.shape[1:]
    nx_n = torch.zeros(shape, device=bfl_mask.device, dtype=torch.float32)
    ny_n = torch.zeros_like(nx_n)
    nz_n = torch.zeros_like(nx_n)
    for direction in range(1, Q):
        link = bfl_mask[direction].to(torch.float32) * w[direction]
        nx_n = nx_n + link * c[direction, 0]
        ny_n = ny_n + link * c[direction, 1]
        nz_n = nz_n + link * c[direction, 2]
    mag = torch.sqrt(nx_n * nx_n + ny_n * ny_n + nz_n * nz_n)
    inv_mag = torch.where(mag > 1e-12, 1.0 / mag, torch.zeros_like(mag))
    return nx_n * inv_mag, ny_n * inv_mag, nz_n * inv_mag


def leaf_wall_model_slip_velocity(
    f_pre: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    normal_activation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``wall_model_slip`` wall velocity + density from the pre-stream state.

    ``u_wall = u_pre - activation * (u_pre . n) n`` per leaf: at full
    activation BFL enforces no penetration while the wall tangentially
    co-moves with the fluid, leaving shear to the stress law; at zero
    activation the wall moves with the fluid and creates no impulse.
    Mirrors the ``bfl_wall_mode="wall_model_slip"`` branch of
    ``bfl_wall_function_3d`` Step 1.
    """
    if not 0.0 <= normal_activation <= 1.0:
        raise ValueError("normal_activation must be in [0,1]")
    Q = f_pre.shape[0]
    rho, ux, uy, uz = _leaf_macros(f_pre, Q)
    nx_n, ny_n, nz_n = normals
    u_dot_n_pre = ux * nx_n + uy * ny_n + uz * nz_n
    return (
        rho,
        ux - normal_activation * u_dot_n_pre * nx_n,
        uy - normal_activation * u_dot_n_pre * ny_n,
        uz - normal_activation * u_dot_n_pre * nz_n,
    )


def leaf_solve_wall_law(
    u_tan_mag: torch.Tensor,
    nu: float | torch.Tensor,
    y_val: float | torch.Tensor,
    wall_law: str,
    near: torch.Tensor,
    *,
    leaf_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wall-law inversion on leaves via ``wall_model._solve_wall_law``.

    ``_solve_wall_law`` is shape-agnostic, so a scalar ``nu`` is handed
    through unchanged (bitwise identity with the grid solve on the same
    values).  A per-leaf ``nu`` (dual-viscosity chain) is grouped by leaf
    level — within one level the viscosity is constant, so each group is
    again an exact scalar solve.
    """
    from tensorlbm.wall_model import _solve_wall_law

    if not isinstance(nu, torch.Tensor):
        return _solve_wall_law(u_tan_mag, float(nu), y_val, wall_law, near)
    if leaf_level is None:
        raise ValueError("a per-leaf nu requires leaf_level for grouping")
    out = torch.zeros_like(u_tan_mag)
    ones = torch.ones_like(near)
    for level in torch.unique(leaf_level):
        group = leaf_level == level
        nu_level = nu[group]
        assert bool((nu_level == nu_level[0]).all()), "nu must be constant per level"
        sel = group & near
        if bool(sel.any()):
            out[sel] = _solve_wall_law(
                u_tan_mag[sel], float(nu_level[0]), y_val, wall_law, ones[sel]
            )
    return out


def leaf_guo_body_force(
    f: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Guo forcing (Guo et al. 2002) on ``(Q, n_leaf)`` leaves.

    ``F_i = w_i [(c_i - u).F/cs^2 + (c_i.u)(c_i.F)/cs^4]`` — the same
    expression, weight construction and evaluation order as
    ``wall_model.guo_body_force_d3q19``/``_d3q27`` so a grid field and
    its leaf flattening agree bit for bit.  As in the grid kernel this is
    an operator-split post-collision application: it injects the
    requested traction impulse exactly.
    """
    Q = f.shape[0]
    if Q == 27:
        from tensorlbm.d3q27 import C as C_LAT

        squared_speeds = (8.0 / 27.0, 2.0 / 27.0, 1.0 / 54.0, 1.0 / 216.0)
    elif Q == 19:
        from tensorlbm.d3q19 import C as C_LAT

        squared_speeds = (1.0 / 3.0, 1.0 / 18.0, 1.0 / 36.0)
    else:
        raise NotImplementedError(f"leaf Guo forcing supports D3Q19/D3Q27, got Q={Q}")
    device = f.device
    c = C_LAT.to(device=device, dtype=f.dtype)
    weights_by_squared_speed = torch.tensor(squared_speeds, device=device, dtype=f.dtype)
    w = weights_by_squared_speed[c.square().sum(dim=1).to(torch.long)]
    cs2 = 1.0 / 3.0
    u_dot_f = ux * fx + uy * fy + uz * fz
    cx = c[:, 0].unsqueeze(1)
    cy = c[:, 1].unsqueeze(1)
    cz = c[:, 2].unsqueeze(1)
    w_view = w.unsqueeze(1)
    cu_u = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    cu_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    forcing = w_view * ((cu_f - u_dot_f.unsqueeze(0)) / cs2 + cu_u * cu_f / cs2**2)
    return f + forcing


def leaf_wall_shear_traction(
    f: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    near: torch.Tensor,
    nu: float | torch.Tensor,
    *,
    wall_law: str = "reichardt",
    y_val: float = 0.5,
    shear_activation: float = 1.0,
    area_weight: torch.Tensor | None = None,
    leaf_level: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Steps 3-5 of ``bfl_wall_function_3d`` on the leaf control volumes.

    Tangential decomposition against ``normals``, wall-law inversion for
    ``u_tau``, then the traction ``-tau_w * area * t_hat`` applied to the
    boundary leaves through Guo forcing.  Each boundary leaf cell is its
    own unit control volume, so the source is ``tau_w * A`` with the wall
    distance entering only the wall-law solve — the grid tier's
    ``tau_w*A/V`` convention carried over unchanged.
    """
    if not 0.0 <= shear_activation <= 1.0:
        raise ValueError("shear_activation must be in [0,1]")
    Q = f.shape[0]
    _, ux, uy, uz = _leaf_macros(f, Q)
    nx_n, ny_n, nz_n = normals
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n
    u_tan_mag = torch.sqrt(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z).clamp(min=1e-12)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    has_tan = u_tan_mag > 1e-10
    u_tan_mag = torch.where(has_tan, u_tan_mag, u_mag)
    ut_x = torch.where(has_tan, ut_x, ux)
    ut_y = torch.where(has_tan, ut_y, uy)
    ut_z = torch.where(has_tan, ut_z, uz)
    inv_utan = 1.0 / u_tan_mag

    u_tau = leaf_solve_wall_law(u_tan_mag, nu, y_val, wall_law, near, leaf_level=leaf_level)
    tau_w = u_tau * u_tau

    near_f = near.to(f.dtype)
    traction_area = (
        near_f if area_weight is None else near_f * area_weight.to(device=f.device, dtype=f.dtype)
    )
    coef = -tau_w * traction_area * shear_activation
    fx = coef * (ut_x * inv_utan)
    fy = coef * (ut_y * inv_utan)
    fz = coef * (ut_z * inv_utan)

    f_out = leaf_guo_body_force(f, fx, fy, fz, ux, uy, uz)
    diagnostics = {
        "u_tau": u_tau,
        "tau_w": tau_w,
        "active_nodes": int(near.sum().item()),
        "shear_force": torch.stack((fx.sum(), fy.sum(), fz.sum())).to(torch.float64),
        "friction_drag_x": float(
            (tau_w * (ut_x * inv_utan) * traction_area * shear_activation).sum().item()
        ),
    }
    return f_out, diagnostics


def leaf_dual_viscosity_wall_nu(
    u_in: float,
    length_host_cells: float,
    reynolds_physical: float,
    leaf_level: torch.Tensor,
) -> torch.Tensor:
    """Per-leaf wall-law viscosity at the physical Reynolds number.

    Leaf level ``d`` is ``2^d`` times finer than the host lattice, so the
    characteristic length is ``length_host_cells * 2**d`` in that level's
    cells.  The collision chain's relaxation may be stability-driven;
    this viscosity keeps the wall law tied to the physical Reynolds
    number (``wall_model.physical_wall_lattice_viscosity``, called per
    level with scalar lengths — its argument validation is scalar-only).
    """
    from tensorlbm.wall_model import physical_wall_lattice_viscosity

    out = torch.empty(leaf_level.shape, dtype=torch.float64, device=leaf_level.device)
    for level in torch.unique(leaf_level):
        nu = physical_wall_lattice_viscosity(
            u_in, length_host_cells * float(1 << int(level)), reynolds_physical
        )
        out[leaf_level == level] = nu
    return out


@dataclass(frozen=True)
class LeafWallChain:
    """Wall-model inputs on the finest leaf level of an octree shell."""

    near: torch.Tensor
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    wall_nu: float | torch.Tensor
    leaf_level: torch.Tensor
    finest_level: int
    wall_law: str
    y_val: float
    coarser_bfl_links: int
    stress_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def build_leaf_wall_chain(
    octree,
    *,
    u_in: float,
    length_host_cells: float,
    reynolds_physical: float,
    wall_law: str = "reichardt",
    y_val: float = 0.5,
    finest_only: bool = True,
    stress_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> LeafWallChain:
    """Build the leaf wall chain (finest level only by default).

    ``near`` is the set of boundary leaves — leaves carrying at least one
    BFL link — restricted to ``leaf_level == d_max`` when ``finest_only``
    (production placement: coarser-band wall cells are overwritten by the
    finer restriction, the leaf analogue of the geometric vacuousness
    counted by ``wall_chain.count_boundary_links_outside_box``; the count
    is reported as ``coarser_bfl_links``).
    """
    levels = octree.leaf_level
    d_max = int(levels.max().item())
    boundary = octree.bfl_mask.any(dim=0)
    finest_boundary = boundary & (levels == d_max)
    if finest_only and not bool(finest_boundary.any()):
        raise ValueError("no boundary links on the finest leaf level — wall chain empty")
    near = finest_boundary if finest_only else boundary
    coarser_links = int(octree.bfl_mask[:, boundary & ~finest_boundary].sum().item())
    normals = leaf_bfl_link_normals(octree.bfl_mask)
    wall_nu = leaf_dual_viscosity_wall_nu(u_in, length_host_cells, reynolds_physical, levels)
    if finest_only:
        # single level -> single constant viscosity (scalar fast path)
        wall_nu = float(wall_nu[finest_boundary][0].item())
    return LeafWallChain(
        near=near,
        normals=normals,
        wall_nu=wall_nu,
        leaf_level=levels,
        finest_level=d_max,
        wall_law=wall_law,
        y_val=y_val,
        coarser_bfl_links=coarser_links,
        stress_normals=stress_normals,
    )


@dataclass
class LeafWallForceProbe:
    """Per-step force split of the leaf wall chain (leaf lattice units).

    Buffers are (3,) float64 on the populations' device; they are
    (re)created on the first substep of the owning factory, and ``reset``
    zeroes them in place.

    Convention (libfix_20261011 unification): ``convention="body"`` — the
    default — keeps BOTH entries in the body/+drag frame, so
    ``total_force = link + shear`` is a single-convention drag-positive
    ledger (``link_force`` is the BFL momentum exchange ON the body;
    ``shear_force`` is the Newton's-third-law reaction of the wall-law Guo
    traction that ``leaf_wall_shear_traction`` applies ON THE FLUID, i.e.
    the negated ``diag["shear_force"]``).  ``convention="legacy_mixed"``
    reproduces the historical mixed-convention accumulation (link body/+drag
    + shear fluid/-drag), whose ``total_force`` near-cancels and is NOT a
    drag reporter; it is kept for A/B comparison with pre-unification runs
    (:attr:`legacy_mixed_force` exposes that reading from either mode).
    """

    link_force: torch.Tensor | None = field(default=None)
    shear_force: torch.Tensor | None = field(default=None)
    convention: str = "body"

    def __post_init__(self) -> None:
        if self.convention not in ("body", "legacy_mixed"):
            raise ValueError(
                f"convention must be 'body' or 'legacy_mixed', got {self.convention!r}",
            )

    def _ensure(self, device: torch.device) -> None:
        ref = torch.zeros(3, dtype=torch.float64, device=device)
        if self.link_force is None or self.link_force.device != device:
            self.link_force = ref.clone()
        if self.shear_force is None or self.shear_force.device != device:
            self.shear_force = ref.clone()

    def reset(self) -> None:
        if self.link_force is not None:
            self.link_force.zero_()
        if self.shear_force is not None:
            self.shear_force.zero_()

    @property
    def total_force(self) -> torch.Tensor | None:
        """Single-convention ledger sum (body/+drag by default)."""
        if self.link_force is None or self.shear_force is None:
            return None
        return self.link_force + self.shear_force

    @property
    def legacy_mixed_force(self) -> torch.Tensor | None:
        """Historical mixed-convention sum (link body/+drag + shear fluid/-drag).

        Only meaningful for comparing against pre-unification ledgers; the
        value near-cancels by construction and must not be read as drag.
        """
        if self.link_force is None or self.shear_force is None:
            return None
        shear_fluid = self.shear_force if self.convention == "legacy_mixed" else -self.shear_force
        return self.link_force + shear_fluid


def make_leaf_wall_chain_bfl_fn(
    octree,
    chain: LeafWallChain,
    *,
    get_step: Callable[[], int],
    ramp_steps: int,
    force_weights: torch.Tensor | None = None,
    q_min: float | None = None,
    convention: str = "body",
):
    """Drop-in ``bfl_fn`` for :func:`step_octree_shell` (wall-model chain).

    Per substep: (1) slip wall velocity from the pre-stream leaf state
    (``_ramp_activation`` startup, the ``bfl_ramp_wall_velocity``
    convention), (2) gather BFL with that wall state, (3) wall-law shear
    traction through Guo forcing on the chain's near-wall leaves.  The
    link impulse and the shear impulse are returned summed as the
    substep force and split in the attached probe.

    ``convention`` (default ``"body"``) fixes the probe/return frame:
    body/+drag with the shear entry as the reaction of the fluid-side Guo
    traction.  ``"legacy_mixed"`` keeps the historical accumulation (link
    body/+drag + shear fluid/-drag; near-cancelling — diagnostic only).
    """
    probe = LeafWallForceProbe(convention=convention)

    def bfl_fn(oct, out, post, ghost_plan, ghost_vals, *, substep):
        activation = _ramp_activation(get_step(), ramp_steps)
        probe._ensure(out.device)  # noqa: SLF001 — factory-owned probe
        rho_w, uwx, uwy, uwz = leaf_wall_model_slip_velocity(
            post, chain.normals, normal_activation=activation
        )
        f_bfl, link_force = bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=ghost_plan,
            ghost_vals=ghost_vals,
            wall_velocity=(uwx, uwy, uwz),
            wall_density=rho_w,
            force_weights=force_weights,
            return_force=True,
            q_min=q_min,
        )
        stress_normals = chain.stress_normals
        if stress_normals is None:
            stress_normals = chain.normals
        f_out, diag = leaf_wall_shear_traction(
            f_bfl,
            stress_normals,
            chain.near,
            chain.wall_nu,
            wall_law=chain.wall_law,
            y_val=chain.y_val,
            shear_activation=activation,
            leaf_level=chain.leaf_level,
        )
        probe.link_force += link_force.to(torch.float64)
        # diag["shear_force"] is the Guo body force ON THE FLUID (the
        # traction ``-tau_w*A*ut_hat``, fluid/-drag).  In the unified
        # body/+drag ledger the wall feels the reaction, hence the negation;
        # "legacy_mixed" keeps the historical sign (near-cancellation, not a
        # drag reporter — see LeafWallForceProbe).
        probe.shear_force += (
            -diag["shear_force"] if probe.convention == "body" else diag["shear_force"]
        )
        return f_out, probe.total_force

    return bfl_fn, probe
