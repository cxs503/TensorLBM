"""Differentiable step chain for solver-in-the-loop autograd workflows.

`docs/differentiable_path.md` established that the eager operators
(``tensorlbm.solver3d``) are differentiable one by one.  This module turns
that property into a *usable composition contract*: a single autograd-clean
D3Q19 step — collision (fluid cells only) -> periodic gather streaming ->
optional boundary conditions on the x planes -> full-way bounce-back on a
solid mask — plus a rollout loop with optional per-step gradient
checkpointing and a differentiable momentum-exchange force probe on the
obstacle.

Everything is built from out-of-place torch ops that keep the autograd graph:

* collision is plain arithmetic (``f - (f - feq) / tau``), with ``tau``
  accepted as a graph-connected 0-dim tensor;
* streaming reuses the audited gather implementation ``solver3d.stream3d``
  (index tensors are cached constants);
* the solid is applied with ``torch.where`` selections only — no in-place
  boundary overwrites (the pattern documented as gradient-hostile in
  ``docs/differentiable_path.md``);
* the boundary conditions follow the same discipline: the boundary planes
  (inlet/outlet on x, and the four lateral walls) are *reconstructed* from
  candidate tensors selected with ``torch.where`` and re-assembled with
  ``torch.cat`` — the free-slip wall is a pure index-level population swap —
  so gradients cross the boundary overwrites exactly (see :class:`InletSpec`
  / :class:`OutletSpec` / :class:`WallSpec`);
* the collision operator is a slot: the default is single-component BGK
  (:func:`tensorlbm.solver3d.collide_bgk3d`); any callable
  ``f, tau -> f`` built from differentiable ops drops in, e.g.
  :func:`tensorlbm.turbulence.collide_smagorinsky_bgk3d` to learn the
  Smagorinsky constant ``C_s`` through the solver.

This is the TensorLBM counterpart of the solver-in-the-loop paradigm of
Autodesk XLB (JAX, Apache-2.0, Ataei & Salehipour, CPC 300:109187, 2024) and
Um et al., NeurIPS 2020: the discrete time-stepping operator itself is in the
backward graph, so ``d(loss)/d(tau)``, ``d(loss)/d(C_s)``, ``d(loss)/df0`` and
``d(loss)/du_in`` are exact derivatives of the *simulated* observable — no
frozen-field surrogate (``tensorlbm/adjoint.py``) and no hand-derived adjoint.

Scope (deliberate): single-component D3Q19 BGK-family collision, stationary
solid mask.  The default lattice stays fully periodic; passing
``inlet``/``outlet`` replaces the periodic wrap on the x planes with a
differentiable velocity inlet and a zero-gradient or convective outlet, and
``walls`` replaces the periodic wrap on the four lateral planes with
free-slip (specular reflection) or free-stream faces (the SUBOFF production
phase order) — a fully bounded, gradient-connected box (A6++), whose four
lateral faces can since A6+++ each carry their own closure through
``WallSpec.overrides``.  With all boundary arguments ``None`` the operator
stays bit-for-bit the original periodic chain.

Known limits of the bounded box (updated by A6+++): the inlet pins the
*normal* velocity only (Zou/He tangential reconstruction and turbulent /
synthetic inflow are not wired); the convective outlet uses a single uniform
convective speed on the outlet plane alone (no sponge/NSCBC pressure
relaxation); per-face wall control lets each lateral face pick one of the
three existing closures independently — no new physics (no moving-wall/lid
method, no per-face parameters beyond the free-stream values) — and on the
edge/corner lines the later-applied closure
wins on doubly-unknown directions (last write wins, not a corner-consistent
reflection).  Multi-component/free-surface/distributed paths and
memory-format optimisations are out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .d3q19 import OPPOSITE, equilibrium3d
from .d3q19 import C as C3D
from .solver3d import collide_bgk3d, stream3d

CollideFn = Callable[..., torch.Tensor]

# Direction sets on the streamwise (x) boundary planes.  Layout is
# ``(19, nz, ny, nx)`` with ``c[q, 0]`` the shift along the *last* axis;
# ``stream3d`` gathers ``f_new[q, x] = f[q, x - c_qx]``, so after streaming
# the inlet plane x = 0 holds, for the c_x = +1 directions, values wrapped
# around from the outlet — those are the unknowns the inlet closure must
# supply.  Symmetrically the c_x = -1 directions on the outlet plane x = nx-1
# wrapped around from the inlet.  ``OPPOSITE`` maps one set onto the other
# element-wise.
_INCOMING_X = (1, 7, 9, 11, 13)  # c_x = +1: unknown at the inlet plane
_OUTGOING_X = (2, 8, 10, 12, 14)  # c_x = -1: unknown at the outlet plane
_NO_SHIFT_X = (0, 3, 4, 5, 6, 15, 16, 17, 18)  # c_x = 0

# Lateral (y/z) boundary machinery.  After streaming, the plane y = 0 holds,
# for the c_y = +1 directions, values wrapped around from y = ny - 1 — those
# are the unknowns the lateral closure must supply (symmetrically c_y = -1 on
# y = ny - 1, c_z = +1 on z = 0, c_z = -1 on z = nz - 1).  The mirror tables
# negate one transverse velocity component: FLIP[axis][q] is the index of
# (c_q with c_q[axis] negated), i.e. the specular-reflection partner of q
# about the plane normal to *axis*.  They are derived from the lattice table
# (not transcribed), so they cannot drift from ``d3q19.C``.
_C_LIST = C3D.tolist()


def _mirror_table(axis: int) -> tuple[int, ...]:
    """Specular-reflection index table negating velocity component *axis*."""
    table = []
    for q in range(19):
        mirrored = list(_C_LIST[q])
        mirrored[axis] = -mirrored[axis]
        table.append(_C_LIST.index(mirrored))
    return tuple(table)


def _directions_with(axis: int, sign: int) -> tuple[int, ...]:
    """Direction indices with c_q[axis] == *sign*."""
    return tuple(q for q in range(19) if _C_LIST[q][axis] == sign)


_FLIP_Y = _mirror_table(1)
_FLIP_Z = _mirror_table(2)
_IN_Y_POS = _directions_with(1, +1)  # c_y = +1: unknown at y = 0
_IN_Y_NEG = _directions_with(1, -1)  # c_y = -1: unknown at y = ny - 1
_IN_Z_POS = _directions_with(2, +1)  # c_z = +1: unknown at z = 0
_IN_Z_NEG = _directions_with(2, -1)  # c_z = -1: unknown at z = nz - 1

# Per-face (A6+++) lateral-wall machinery.  The four lateral faces are named
# by their outward normal; ``_FACE_KEYS`` doubles as the *closure order* —
# faces are closed y = 0, y = ny - 1, z = 0, z = nz - 1, so on the
# edge/corner lines the later face wins (last write wins, unchanged from
# the shared-spec A6++ semantics).
#
# ========= ============ ============== ======================= ====================
# face key  plane        tensor axis    unknown directions       mirror table
# ========= ============ ============== ======================= ====================
# ``"-y"``  y = 0        dim 2, first   c_y = +1 (wrapped)       ``_FLIP_Y``
# ``"+y"``  y = ny - 1   dim 2, last    c_y = -1 (wrapped)       ``_FLIP_Y``
# ``"-z"``  z = 0        dim 1, first   c_z = +1 (wrapped)       ``_FLIP_Z``
# ``"+z"``  z = nz - 1   dim 1, last    c_z = -1 (wrapped)       ``_FLIP_Z``
# ========= ============ ============== ======================= ====================
_FACE_KEYS = ("-y", "+y", "-z", "+z")
_FACE_AXIS_DIM = {"-y": 2, "+y": 2, "-z": 1, "+z": 1}  # dim of the face normal
_FACE_AT_START = {"-y": True, "+y": False, "-z": True, "+z": False}
_FACE_UNKNOWN = {
    "-y": _IN_Y_POS,
    "+y": _IN_Y_NEG,
    "-z": _IN_Z_POS,
    "+z": _IN_Z_NEG,
}
_FACE_FLIP = {"-y": _FLIP_Y, "+y": _FLIP_Y, "-z": _FLIP_Z, "+z": _FLIP_Z}

__all__ = [
    "InletSpec",
    "OutletSpec",
    "WallSpec",
    "differentiable_step",
    "obstacle_force",
    "rollout",
]


@dataclass(frozen=True)
class InletSpec:
    """Differentiable velocity inlet on the plane x = 0.

    Two closures are available (``method``):

    ``"equilibrium"`` (default) — the whole inlet plane is (re)set to the
    Dirichlet equilibrium ``f_eq(rho0, u)``.  Cheap, exactly reproduces
    ``(rho0, u)`` on the plane and robust for driving a bounded campaign;
    the outgoing populations at the plane are absorbed rather than leaving
    the domain, which adds a small acoustic reflection.

    ``"zouhe"`` — only the five unknown (incoming, c_x = +1) populations are
    reconstructed with the Zou/He non-equilibrium correction

    .. math:: f_i = f_i^{eq}(\\rho_{in}, u) + f_{\\bar i} -
        f_{\\bar i}^{eq}(\\rho_{in}, u), \\qquad i \\in \\text{incoming},

    with the plane density obtained self-consistently from the known
    (streamed, i.e. post-collision) populations,

    .. math:: \\rho_{in} = \\big(\\textstyle\\sum_{c_x = 0} f +
        2 \\sum_{c_x = -1} f\\big) / (1 - u_x).

    The outgoing populations keep their streamed values, and the closure
    reproduces the prescribed *normal* velocity ``u_x`` and the Zou/He plane
    density exactly.  Two approximations are inherited from this standard
    non-equilibrium bounce-back form: the plane density differs from the
    exact Zou/He moment solution by the outgoing non-equilibrium, and the
    *tangential* momentum on the plane is whatever the ``c_x = 0``
    populations streamed in from the interior (it is not pinned to
    ``u_y``/``u_z``) — appropriate for streamwise inflow ``u = (u_in, 0, 0)``
    where the tangential contamination at the inlet is negligible.
    ``u_x`` must stay below 1 in lattice units (division by ``1 - u_x``).

    All components accept floats or 0-dim tensors; tensors with
    ``requires_grad=True`` keep the boundary condition in the autograd graph
    (e.g. calibrating the free-stream velocity against measured drag).
    """

    ux: float | torch.Tensor = 0.0
    uy: float | torch.Tensor = 0.0
    uz: float | torch.Tensor = 0.0
    rho0: float | torch.Tensor = 1.0
    method: str = "equilibrium"

    def __post_init__(self) -> None:
        if self.method not in ("equilibrium", "zouhe"):
            raise ValueError(f"inlet method must be 'equilibrium' or 'zouhe', got {self.method!r}")


@dataclass(frozen=True)
class OutletSpec:
    """Differentiable outlet on the plane x = nx - 1 (two closures).

    ``method="copy"`` (default) — zero-gradient: after streaming, the five
    unknown (outgoing, c_x = -1) populations on the outlet plane would wrap
    around from the inlet; they are replaced by the values of the fully
    interior neighbour plane x = nx - 2.  The known (c_x >= 0) populations
    keep their streamed interior values.

    ``method="convective"`` — first-order upwind convective condition on the
    same five unknown populations, discretising
    ``df/dt + U_c df/dx = 0`` with the upwind neighbour x = nx - 2 and the
    *previous step's* outlet face as the time level:

    .. math:: f_{out}^{n+1} = f_{out}^{n} + U_c\\,
        \\big(f_{out-1}^{n} - f_{out}^{n}\\big)
        = (1 - U_c)\\,f_{out}^{n} + U_c\\,f_{out-1}^{n},

    with the Courant number ``U_c = u_conv`` (lattice units, dt = dx = 1, so
    U_c = u_conv·dt/dx).  The scheme is a convex combination for
    ``0 < U_c < 1`` — the upwind CFL bound; ``U_c = 1`` degenerates to the
    plain copy, ``U_c -> 0`` freezes the plane (nothing convects out).
    Applying the condition only to the unknown directions leaves the known
    (c_x >= 0) streamed populations untouched.

    The time recursion makes the outlet depend on its own previous state, so
    the caller supplies the history: :func:`differentiable_step` takes the
    previous post-boundary outlet face as ``outlet_prev`` (``None`` seeds it
    from the step input's own outlet plane ``f[..., -1:]``, i.e. the initial
    condition on the first step), and :func:`rollout` chains the faces
    automatically (the faces stay in the autograd graph, so gradients flow
    through the recursion — including w.r.t. ``u_conv`` itself).

    ``u_conv``: the convective speed.  ``None`` (default) resolves it from
    ``inlet.ux`` at call time (the standard choice ``U_c = u_in`` — one
    physical outflow speed; an explicit float/0-dim tensor overrides it, and
    a tensor with ``requires_grad=True`` makes ``U_c`` itself a learnable
    parameter).  ``u_conv`` is ignored by ``method="copy"``.  A convective
    outlet without either source raises ``ValueError``.
    """

    method: str = "copy"
    u_conv: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.method not in ("copy", "convective"):
            raise ValueError(f"outlet method must be 'copy' or 'convective', got {self.method!r}")


@dataclass(frozen=True)
class WallSpec:
    """Differentiable lateral boundaries on y = 0, y = ny-1, z = 0, z = nz-1.

    The spec's own *method* drives every lateral face that is not overridden
    (all four by default):

    ``"periodic"`` (default) — no-op: the planes keep the periodic wrap of
    :func:`tensorlbm.solver3d.stream3d` (identical to passing ``None``).

    ``"free-slip"`` — on-node specular reflection: after streaming, the five
    unknown populations on each face (the ones that wrapped around the
    domain, e.g. c_y = +1 on y = 0) are replaced by the face's *mirror
    partner* populations, ``f[q] = f[FLIP[q]]`` with FLIP the index table
    negating the face-normal velocity component.  This is a pure
    index-level population swap: no arithmetic reconstruction, no division,
    unconditionally stable, and autograd-transparent.  Chosen over a Zou/He
    rebuild because specular reflection is *exact* for its target — the
    reflected pairs cancel, so the wall-normal velocity on the face is
    zero to machine precision and the tangential momentum is untouched,
    with none of the non-equilibrium extrapolation error a closure rebuild
    would inject.  Cost of the on-node placement (shared with on-node
    bounce-back): the effective wall sits on the plane nodes themselves
    (first-order wall placement), not half-way between nodes.

    ``"freestream"`` — the whole face is (re)set to the Dirichlet
    equilibrium ``f_eq(rho0, u_inf)`` (the same construction as the
    ``"equilibrium"`` inlet).  Stronger than free-slip (pins rho and the
    full velocity on the face, absorbing outgoing populations); use for
    far-field/wind-tunnel sides, at the price of some acoustic reflection.

    Per-face overrides (A6+++): ``overrides`` maps face keys to their own
    :class:`WallSpec`; faces not listed fall back to this spec (the default
    closure).  Keys are the four outward normals of the lateral box,

    ========= =========== =========
    key       plane       comment
    ========= =========== =========
    ``"-y"``  y = 0       lower y
    ``"+y"``  y = ny - 1  upper y
    ``"-z"``  z = 0       lower z
    ``"+z"``  z = nz - 1  upper z
    ========= =========== =========

    so e.g. ``WallSpec(method="free-slip", overrides={"+z":
    WallSpec(method="freestream", ux=0.05)})`` closes the y faces (and z = 0)
    with free-slip while the top plane z = nz - 1 is a far-field free-stream
    face — the wind-tunnel-floor layout.  Override specs may not carry
    ``overrides`` of their own (fail loudly at construction).  ``None`` (or
    absent) keeps the A6++ behaviour — one closure shared by all four faces
    — bit-for-bit.

    Edge/corner policy: the faces are closed in the order y = 0, y = ny - 1,
    z = 0, z = nz - 1 *before* the inlet/outlet closures; on the edge and
    corner lines a direction can be unknown for two closures at once, and
    the later application wins (last write wins) — e.g. at a corner the
    z closure wins over the y closure, and the inlet/outlet closures win
    over the walls.  Away from those measure-zero lines the closures touch
    disjoint direction sets, so the order is irrelevant there.  A face
    overridden to ``"periodic"`` is a no-op like the shared default.

    ``rho0``/``ux``/``uy``/``uz`` are read by ``"freestream"`` only; they
    accept floats or 0-dim tensors (graph-connected, e.g. a learnable
    free-stream speed shared with the inlet).

    Serialisation (A6+++): :meth:`to_dict` emits a plain-Python payload
    (tensor fields flattened to their numeric value — the autograd graph is
    not serialisable), :meth:`from_dict` rebuilds the spec.  Payloads
    written before per-face control (no ``"overrides"`` key) load unchanged.
    """

    method: str = "periodic"
    rho0: float | torch.Tensor = 1.0
    ux: float | torch.Tensor = 0.0
    uy: float | torch.Tensor = 0.0
    uz: float | torch.Tensor = 0.0
    overrides: Mapping[str, WallSpec] | None = None

    def __post_init__(self) -> None:
        if self.method not in ("periodic", "free-slip", "freestream"):
            raise ValueError(
                f"wall method must be 'periodic', 'free-slip' or 'freestream', got {self.method!r}"
            )
        if self.overrides is None:
            return
        bad_keys = sorted(set(self.overrides) - set(_FACE_KEYS))
        if bad_keys:
            raise ValueError(
                f"wall overrides keys must be among {_FACE_KEYS}, got {bad_keys!r}"
            )
        for key, spec in self.overrides.items():
            if not isinstance(spec, WallSpec):
                raise ValueError(
                    f"wall override for face {key!r} must be a WallSpec, "
                    f"got {type(spec).__name__}"
                )
            if spec.overrides is not None:
                raise ValueError(
                    f"wall override for face {key!r} cannot carry nested overrides"
                )

    def to_dict(self) -> dict[str, object]:
        """Plain-Python payload of this spec (JSON-compatible types).

        Tensor fields (learnable free-stream values) are flattened with
        ``.item()``: the payload stores the *numeric value*, not the graph.
        The ``"overrides"`` key is emitted only when per-face overrides are
        present, so specs without overrides serialise to exactly the
        pre-A6+++ payload shape.
        """
        payload: dict[str, object] = {
            "method": self.method,
            "rho0": _number(self.rho0),
            "ux": _number(self.ux),
            "uy": _number(self.uy),
            "uz": _number(self.uz),
        }
        if self.overrides is not None:
            payload["overrides"] = {
                key: spec.to_dict() for key, spec in self.overrides.items()
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WallSpec:
        """Rebuild a :class:`WallSpec` from :meth:`to_dict` output.

        Tolerates older/checkpoint payloads: the ``"overrides"`` key is
        optional (absent means ``None`` — the shared-closure A6++ spec),
        missing numeric fields fall back to the dataclass defaults, and
        unknown extra keys are ignored.
        """
        kwargs = {
            name: payload[name]
            for name in ("method", "rho0", "ux", "uy", "uz")
            if name in payload
        }
        overrides = payload.get("overrides")
        if overrides is not None:
            kwargs["overrides"] = {
                key: cls.from_dict(spec) for key, spec in overrides.items()
            }
        return cls(**kwargs)


def _collide_skip_solid(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    mask: torch.Tensor | None,
    collide: CollideFn | None,
) -> torch.Tensor:
    """Collision step, skipped inside the solid (NoDynamics obstacle cells).

    Solid cells carry the populations reflected by the previous bounce-back;
    relaxing them toward equilibrium would stream contaminated values into
    the fluid on the next :func:`differentiable_step`.  ``torch.where`` keeps
    the branch differentiable (the solid branch passes ``f`` through, and
    the reflected populations themselves carry gradients from earlier steps).
    """
    f_col = collide_bgk3d(f, tau) if collide is None else collide(f, tau)
    if mask is None:
        return f_col
    return torch.where(mask.unsqueeze(0), f, f_col)


def _scalar(value: float | torch.Tensor, dtype: torch.dtype, device: torch.device):
    """Coerce a Python float or 0-dim tensor to *dtype/device* (graph kept)."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _number(value: float | torch.Tensor) -> float:
    """Numeric value of a spec field (tensors flattened, graph dropped)."""
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def _dir_selector(indices: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Boolean (19, 1, 1, 1) selector, True on *indices*."""
    sel = torch.zeros((19, 1, 1, 1), dtype=torch.bool, device=device)
    sel[list(indices), 0, 0, 0] = True
    return sel


def _apply_inlet(f_str: torch.Tensor, inlet: InletSpec) -> torch.Tensor:
    """Replace the inlet plane x = 0 of the post-stream state (out-of-place).

    Returns ``cat([plane_new, f_str[..., 1:]], dim=-1)`` — a new tensor, so
    the discarded wrapped values simply drop out of the autograd graph and
    every kept entry keeps its gradient.
    """
    device, dtype = f_str.device, f_str.dtype
    nz, ny = f_str.shape[1], f_str.shape[2]
    ux = _scalar(inlet.ux, dtype, device)
    uy = _scalar(inlet.uy, dtype, device)
    uz = _scalar(inlet.uz, dtype, device)
    plane = f_str[..., :1]  # (19, nz, ny, 1) post-stream inlet slice

    if inlet.method == "equilibrium":
        rho = _scalar(inlet.rho0, dtype, device)
        # 0-dim inputs broadcast to the whole (19, nz, ny, 1) plane
        plane_new = equilibrium3d(rho, ux, uy, uz, device).expand(19, nz, ny, 1)
    else:  # "zouhe"
        rest = plane[list(_NO_SHIFT_X)].sum(dim=0)
        outgoing = plane[list(_OUTGOING_X)].sum(dim=0)
        rho_in = (rest + 2.0 * outgoing) / (1.0 - ux)  # (nz, ny, 1)
        ones = torch.ones_like(rho_in)  # broadcast the 0-dim velocity to the plane
        feq = equilibrium3d(rho_in, ux * ones, uy * ones, uz * ones, device)  # (19, nz, ny, 1)
        opp = OPPOSITE.to(device)
        # candidate for every q: feq + (f_opp - feq_opp); masked to the
        # five unknown incoming directions by the selector below
        cand = feq + (plane[opp] - feq[opp])
        plane_new = torch.where(_dir_selector(_INCOMING_X, device), cand, plane)

    return torch.cat([plane_new, f_str[..., 1:]], dim=-1)


def _apply_walls(f_str: torch.Tensor, walls: WallSpec) -> torch.Tensor:
    """Close the four lateral planes of the post-stream state (out-of-place).

    ``"periodic"`` with no overrides is a bit-exact no-op (returns *f_str*
    itself).  Without overrides the shared A6++ closure runs unchanged
    (:func:`_apply_walls_uniform`, bit-for-bit the pre-A6+++ operator); with
    ``WallSpec.overrides`` each face is closed by its own resolved spec
    (:func:`_apply_walls_per_face`).  In both branches the faces close in
    the order y = 0, y = ny - 1, z = 0, z = nz - 1 (see :class:`WallSpec`
    for the edge/corner last-write-wins policy).
    """
    if walls.method == "periodic" and walls.overrides is None:
        return f_str
    if walls.overrides is None:
        return _apply_walls_uniform(f_str, walls)
    return _apply_walls_per_face(f_str, walls)


def _apply_walls_uniform(f_str: torch.Tensor, walls: WallSpec) -> torch.Tensor:
    """Shared-spec closure of all four lateral planes (the A6++ operator).

    Free-slip rebuilds each face with ``torch.where(unknown_selector,
    plane[FLIP], plane)``; freestream rebuilds each face from a broadcast
    equilibrium.  Faces are closed y = 0, y = ny - 1, z = 0, z = nz - 1.
    """
    device, dtype = f_str.device, f_str.dtype
    nz, ny, nx = f_str.shape[1], f_str.shape[2], f_str.shape[3]

    if walls.method == "free-slip":
        flip_y = torch.tensor(_FLIP_Y, device=device)
        flip_z = torch.tensor(_FLIP_Z, device=device)
        # y = 0: unknown c_y = +1 directions take their mirror partner
        plane = f_str[:, :, :1]
        plane_new = torch.where(_dir_selector(_IN_Y_POS, device), plane[flip_y], plane)
        f = torch.cat([plane_new, f_str[:, :, 1:]], dim=2)
        # y = ny - 1: unknown c_y = -1 directions
        plane = f[:, :, -1:]
        plane_new = torch.where(_dir_selector(_IN_Y_NEG, device), plane[flip_y], plane)
        f = torch.cat([f[:, :, :-1], plane_new], dim=2)
        # z = 0 / z = nz - 1: same construction on axis 1
        plane = f[:, :1]
        plane_new = torch.where(_dir_selector(_IN_Z_POS, device), plane[flip_z], plane)
        f = torch.cat([plane_new, f[:, 1:]], dim=1)
        plane = f[:, -1:]
        plane_new = torch.where(_dir_selector(_IN_Z_NEG, device), plane[flip_z], plane)
        f = torch.cat([f[:, :-1], plane_new], dim=1)
        return f

    # "freestream": whole faces reset to f_eq(rho0, u_inf) (0-dim broadcast)
    rho = _scalar(walls.rho0, dtype, device)
    ux = _scalar(walls.ux, dtype, device)
    uy = _scalar(walls.uy, dtype, device)
    uz = _scalar(walls.uz, dtype, device)
    feq = equilibrium3d(rho, ux, uy, uz, device)  # (19, 1, 1, 1)
    y_face = feq.expand(19, nz, 1, nx)
    z_face = feq.expand(19, 1, ny, nx)
    f = torch.cat([y_face, f_str[:, :, 1:]], dim=2)
    f = torch.cat([f[:, :, :-1], y_face], dim=2)
    f = torch.cat([z_face, f[:, 1:, :]], dim=1)
    f = torch.cat([f[:, :-1, :], z_face], dim=1)
    return f


def _apply_walls_per_face(f_str: torch.Tensor, walls: WallSpec) -> torch.Tensor:
    """Per-face closure: each lateral plane follows its own resolved spec.

    Faces not listed in ``walls.overrides`` use *walls* itself (the default
    closure), in the fixed order ``_FACE_KEYS`` — the same y = 0, y = ny-1,
    z = 0, z = nz-1 sequence and edge/corner last-write-wins semantics as
    the shared-spec path.  Faces resolved to ``"periodic"`` are no-ops.
    """
    resolved = [(key, walls.overrides.get(key, walls)) for key in _FACE_KEYS]
    if all(spec.method == "periodic" for _key, spec in resolved):
        return f_str  # every face keeps the periodic wrap: bit-exact no-op
    flips = {
        table: torch.tensor(table, device=f_str.device)
        for table in set(_FACE_FLIP.values())
    }
    f = f_str
    for key, spec in resolved:
        f = _close_face(f, key, spec, flips[_FACE_FLIP[key]])
    return f


def _close_face(
    f: torch.Tensor, key: str, spec: WallSpec, flip: torch.Tensor
) -> torch.Tensor:
    """Close one lateral face *key* of the current chain state per *spec*.

    ``"periodic"`` returns *f* unchanged (no-op).  ``"free-slip"`` mirrors
    the face's unknown directions within the face plane; ``"freestream"``
    resets the whole face to ``f_eq(rho0, u_inf)`` of this spec (0-dim
    broadcast, graph-connected).  Reading the face from the current chain
    state keeps the sequential face order meaningful on the edge lines.
    """
    if spec.method == "periodic":
        return f
    dim, at_start = _FACE_AXIS_DIM[key], _FACE_AT_START[key]
    if at_start:
        plane = f[:, :, :1] if dim == 2 else f[:, :1]
    else:
        plane = f[:, :, -1:] if dim == 2 else f[:, -1:]

    if spec.method == "free-slip":
        plane_new = torch.where(
            _dir_selector(_FACE_UNKNOWN[key], f.device), plane[flip], plane
        )
    else:  # "freestream": whole face reset to f_eq(rho0, u_inf) of this spec
        device, dtype = f.device, f.dtype
        feq = equilibrium3d(
            _scalar(spec.rho0, dtype, device),
            _scalar(spec.ux, dtype, device),
            _scalar(spec.uy, dtype, device),
            _scalar(spec.uz, dtype, device),
            device,
        )  # (19, 1, 1, 1)
        face_shape = list(f.shape)
        face_shape[dim] = 1
        plane_new = feq.expand(*face_shape)

    if at_start:
        interior = f[:, :, 1:] if dim == 2 else f[:, 1:]
        return torch.cat([plane_new, interior], dim=dim)
    interior = f[:, :, :-1] if dim == 2 else f[:, :-1]
    return torch.cat([interior, plane_new], dim=dim)


def _resolve_u_conv(outlet: OutletSpec, inlet: InletSpec | None) -> float | torch.Tensor:
    """Convective speed: explicit ``u_conv`` > ``inlet.ux`` > error."""
    if outlet.u_conv is not None:
        u_c = outlet.u_conv
    elif inlet is not None:
        u_c = inlet.ux
    else:
        raise ValueError(
            "convective outlet needs a convective speed: pass OutletSpec("
            "method='convective', u_conv=...) or an inlet to derive U_c from"
        )
    value = u_c.detach().item() if isinstance(u_c, torch.Tensor) else float(u_c)
    if not 0.0 < value < 1.0:
        raise ValueError(
            "convective outlet Courant number must satisfy 0 < U_c < 1 (upwind "
            f"CFL bound), got {value!r}"
        )
    return u_c


def _apply_outlet(
    f_str: torch.Tensor,
    outlet: OutletSpec,
    inlet: InletSpec | None,
    f_prev: torch.Tensor,
) -> torch.Tensor:
    """Outlet on x = nx - 1 of the post-stream state (out-of-place).

    Only the unknown outgoing (c_x = -1) populations are touched:
    ``method="copy"`` takes the interior neighbour plane x = nx - 2
    (zero gradient); ``method="convective"`` takes the upwind recursion
    ``f_prev + U_c * (neighbour - f_prev)`` with *f_prev* the previous
    post-boundary outlet face (see :class:`OutletSpec`).  The known ones
    keep their streamed interior values.
    """
    sel = _dir_selector(_OUTGOING_X, f_str.device)
    neighbour = f_str[..., -2:-1]
    if outlet.method == "copy":
        candidate = neighbour
    else:
        u_c = _scalar(_resolve_u_conv(outlet, inlet), f_str.dtype, f_str.device)
        candidate = f_prev + u_c * (neighbour - f_prev)
    plane_new = torch.where(sel, candidate, f_str[..., -1:])
    return torch.cat([f_str[..., :-1], plane_new], dim=-1)


def differentiable_step(
    f: torch.Tensor,
    tau: float | torch.Tensor = 0.9,
    mask: torch.Tensor | None = None,
    *,
    collide: CollideFn | None = None,
    return_probe: bool = False,
    inlet: InletSpec | None = None,
    outlet: OutletSpec | None = None,
    walls: WallSpec | None = None,
    outlet_prev: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One autograd-clean D3Q19 step: collide (fluid) -> stream -> BC -> bounce-back.

    Composition (the production SUBOFF phase order, memory note "force sample
    post-stream / pre-bounce-back"):

    1. BGK collision with relaxation time *tau*, skipped inside the solid
       mask (NoDynamics), see :func:`_collide_skip_solid`;
    2. periodic gather streaming (:func:`tensorlbm.solver3d.stream3d`);
    3. if given, the boundary conditions, applied in the order lateral walls
       (:func:`_apply_walls`, all four y/z planes) -> inlet ->
       outlet (:func:`_apply_inlet`, :func:`_apply_outlet`) — the lateral
       closures clean the edge sources the x closures read, and the
       streamwise (driving) conditions win on doubly-unknown edge
       directions;
    4. full-way bounce-back on the solid: ``f_new = where(mask, f_bc[opp],
       f_bc)`` — the reflected populations leave the solid on the next
       streaming step.

    Boundary phase — why post-stream / pre-bounce-back: streaming is the
    operator whose periodic wrap injects the unphysical cross-domain data, so
    the correction belongs immediately after it; at that moment the known
    populations on the boundary planes still carry the freshest post-collision
    interior information, which is exactly what the Zou/He non-equilibrium
    correction requires (non-equilibrium of *post-collision* populations —
    the phase is self-consistent with the collision that just ran).  Applying
    the boundaries before bounce-back also keeps the force probe (below) on
    the boundary-conditioned state, matching the production sampling phase.

    With ``inlet=None, outlet=None, walls=None`` (default) the operator is
    bit-for-bit the original periodic chain.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` (the state
            returned by a previous :func:`differentiable_step`, i.e.
            post-bounce-back).
        tau: BGK relaxation time; a 0-dim tensor with ``requires_grad=True``
            stays connected to the autograd graph.
        mask: Boolean solid mask of shape ``(nz, ny, nx)``; ``None`` gives
            the plain periodic collide->stream chain (identical operator
            order to the rollouts in ``tests/test_autograd.py``).  The
            boundary planes are assumed fluid; if the mask is True there,
            bounce-back (applied last) wins over the boundary value.
        collide: Optional replacement collision operator ``f, tau -> f``.
            Must be built from differentiable ops (e.g.
            ``functools.partial(collide_smagorinsky_bgk3d, C_s=cs)`` with a
            tensor ``C_s``).
        return_probe: Additionally return the post-stream /
            post-boundary-condition / pre-bounce-back state, the sampling
            point for :func:`obstacle_force`.
        inlet: :class:`InletSpec` velocity inlet on x = 0 (``None``: the
            plane stays periodic).
        outlet: :class:`OutletSpec` zero-gradient or convective outlet on
            x = nx - 1 (``None``: the plane stays periodic).
        walls: :class:`WallSpec` lateral closure for the four y/z planes:
            without ``overrides`` the spec is shared by all faces (``None``
            or ``method="periodic"``: they stay periodic, bit-for-bit);
            ``WallSpec.overrides`` gives individual faces their own closure
            (unlisted faces keep the shared spec).
        outlet_prev: Previous step's post-boundary outlet face, shape
            ``(19, nz, ny, 1)`` — the time history the convective outlet
            recurses on (the faces stay in the autograd graph when chained
            through :func:`rollout`).  ``None`` seeds the recursion from
            this step's input state ``f[..., -1:]`` (the initial condition
            on the first step of a rollout).  Unused by the zero-gradient
            copy outlet.

    Returns:
        The stepped distribution ``f_new``; with ``return_probe=True`` a
        tuple ``(f_new, f_probe)``.  Macroscopic observables computed on
        ``f_new`` should exclude the solid cells (they hold reflected
        populations): mask with ``~mask``.  For a manually chained
        convective outlet, ``f_probe[..., -1:]`` is the outlet face to feed
        into the next step's ``outlet_prev``.
    """
    f_str = stream3d(_collide_skip_solid(f, tau, mask, collide))
    if walls is not None:
        f_str = _apply_walls(f_str, walls)
    if inlet is not None:
        f_str = _apply_inlet(f_str, inlet)
    if outlet is not None:
        f_prev = f[..., -1:] if outlet_prev is None else outlet_prev
        if f_prev.shape != f_str[..., -1:].shape:
            raise ValueError(
                "outlet_prev must be the previous outlet face of shape "
                f"{tuple(f_str[..., -1:].shape)}, got {tuple(f_prev.shape)}"
            )
        f_str = _apply_outlet(f_str, outlet, inlet, f_prev)
    probe = f_str
    if mask is None:
        return (probe, probe) if return_probe else probe
    f_new = torch.where(mask.unsqueeze(0), probe[OPPOSITE.to(probe.device)], probe)
    return (f_new, probe) if return_probe else f_new


def obstacle_force(f_probe: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Momentum-exchange force on the bounce-back obstacle (Ladd, wet-node).

    Evaluate on the post-stream / pre-bounce-back state returned by
    ``differentiable_step(..., return_probe=True)``:

    .. math:: F_\\alpha = 2 \\sum_{x \\in \\mathrm{solid}} \\sum_q
        c_{q,\\alpha}\\, f[q, x]

    (house convention: factor 2 from the bounce-back momentum exchange,
    lattice units, force on the full obstacle).  The result carries
    autograd gradients w.r.t. everything ``f_probe`` depends on.

    Args:
        f_probe: Post-stream / pre-bounce-back distribution, shape
            ``(19, nz, ny, nx)``.
        mask: Boolean solid mask of shape ``(nz, ny, nx)``.

    Returns:
        Force vector ``(fx, fy, fz)`` as a tensor of shape ``(3,)``.
    """
    c = C3D.to(device=f_probe.device, dtype=f_probe.dtype)
    momentum = (f_probe * mask.to(dtype=f_probe.dtype)).sum(dim=(1, 2, 3))
    return 2.0 * torch.matmul(momentum, c)


def rollout(
    f: torch.Tensor,
    n_steps: int,
    tau: float | torch.Tensor = 0.9,
    mask: torch.Tensor | None = None,
    *,
    collide: CollideFn | None = None,
    checkpoint: bool = False,
    return_probes: bool = False,
    inlet: InletSpec | None = None,
    outlet: OutletSpec | None = None,
    walls: WallSpec | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Roll out :func:`differentiable_step` for *n_steps*, keeping the graph.

    The plain mode stores every step's activations (memory grows linearly,
    ~9 ``(19, nz, ny, nx)`` tensors per step for BGK); ``checkpoint=True``
    wraps each step in ``torch.utils.checkpoint`` (``use_reentrant=False``)
    so activation memory stays near-flat while gradients remain identical
    (the strategy quantified in ``examples/differentiable_lbm.py`` and the
    transparent analogue of XLB's segmented checkpoint-replay adjoint).

    A convective outlet recurses on its own previous face, so the loop
    chains the history internally: step *k* receives the post-boundary
    outlet face of step *k-1* (the first step seeds from the initial
    condition's outlet plane, i.e. from ``f[..., -1:]``).  The chained faces
    stay in the autograd graph — also under ``checkpoint=True`` — so
    gradients flow through the outlet recursion, including w.r.t. the
    convective speed ``u_conv``.

    Args:
        f: Initial distribution, shape ``(19, nz, ny, nx)``.
        n_steps: Number of solver steps to apply.
        tau: Relaxation time (0-dim tensor allowed, keeps gradients).
        mask: Boolean solid mask ``(nz, ny, nx)`` or ``None`` for periodic.
        collide: Optional replacement collision operator (see
            :func:`differentiable_step`).
        checkpoint: Wrap every step in gradient checkpointing.
        return_probes: Collect the per-step post-stream / pre-bounce-back
            states for :func:`obstacle_force` (usable together with
            ``checkpoint``; the probes stay in the autograd graph).
        inlet: :class:`InletSpec` velocity inlet on x = 0 (default periodic).
        outlet: :class:`OutletSpec` zero-gradient or convective outlet on
            x = nx - 1 (default periodic).
        walls: :class:`WallSpec` lateral closure on the four y/z planes
            (default ``None``: periodic, bit-for-bit); ``WallSpec.overrides``
            closes individual faces with their own spec (unlisted faces keep
            the shared one).

    Returns:
        The final distribution; with ``return_probes=True`` a tuple
        ``(f_final, probes)`` with one probe per step.
    """
    probes: list[torch.Tensor] | None = [] if return_probes else None
    convective = outlet is not None and outlet.method == "convective"
    outlet_face: torch.Tensor | None = None  # chained history, graph-connected
    for _ in range(n_steps):
        out = _step_for_rollout(
            f,
            tau,
            mask,
            collide=collide,
            checkpoint=checkpoint,
            need_probe=return_probes or convective,
            inlet=inlet,
            outlet=outlet,
            walls=walls,
            outlet_prev=outlet_face,
        )
        if return_probes or convective:
            f, probe = out
            if return_probes:
                probes.append(probe)
            if convective:
                outlet_face = probe[..., -1:]
        else:
            f = out
    if return_probes:
        return f, probes
    return f


def _step_for_rollout(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    mask: torch.Tensor | None,
    *,
    collide: CollideFn | None,
    checkpoint: bool,
    need_probe: bool,
    inlet: InletSpec | None,
    outlet: OutletSpec | None,
    walls: WallSpec | None,
    outlet_prev: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One rollout step, optionally wrapped in gradient checkpointing."""
    if not checkpoint:
        return differentiable_step(
            f,
            tau,
            mask,
            collide=collide,
            return_probe=need_probe,
            inlet=inlet,
            outlet=outlet,
            walls=walls,
            outlet_prev=outlet_prev,
        )
    return _checkpoint(
        differentiable_step,
        f,
        tau,
        mask,
        use_reentrant=False,
        collide=collide,
        return_probe=need_probe,
        inlet=inlet,
        outlet=outlet,
        walls=walls,
        outlet_prev=outlet_prev,
    )
