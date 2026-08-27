"""Integer-id boundary-condition registry for TensorLBM (D3Q19/D3Q27).

Status quo before this module: boundary conditions were a set of explicit
functions (``boundaries3d.zou_he_inlet_velocity_3d`` & co.) with no
registration mechanism — the platform layer could not enumerate the BCs a
run uses, and there was no machine-readable description of which lattice
directions each boundary reconstructs.

Design (adapted, see licence headers below):

* **Integer-id registry** (from ``Autodesk/XLB``
  ``xlb/operator/boundary_condition/boundary_condition_registry.py``,
  Apache-2.0): every :class:`BoundaryCondition` *instance* receives a unique
  positive integer id from a process-wide registry.  Id ``0`` is reserved
  for "solid / no boundary condition" and can never be assigned.  Applying
  a BC selects its cells with one ``bc_mask == id`` comparison — a
  data-independent branch-free expression that is friendly to
  ``torch.compile`` and Triton kernels.

* **Missing-direction masks by "streaming a boolean field once"** (from
  ``Autodesk/XLB`` ``indices_boundary_masker.py``, Apache-2.0): the set of
  lattice directions whose populations are missing after streaming (pull
  scheme: ``f_new[q, x] = f_old[q, x - c_q]``) is derived by padding a
  boolean "blocked" field (solid cells ∪ non-periodic domain faces) with
  one cell and pulling it through the streaming shift once.  The mask is
  therefore *always consistent with the lattice constants* — no handcopied
  per-direction index tables (the class of error recorded at the top of
  ``triton_fused.py`` after a hand-transcribed sign mistake).

* **Overlap detection** (from ``Autodesk/XLB``
  ``xlb/helper/check_boundary_overlaps.py``, Apache-2.0): two BCs claiming
  the same cell make the outcome order-dependent and are rejected.

The physics itself is *not* re-implemented: application dispatches to the
existing, verified ``boundaries3d`` functions, so a run composed through
the registry is bit-identical (1e-6 eager tolerance) to the equivalent
direct call chain.
"""

# ---------------------------------------------------------------------------
# Licence attributions (required by the upstream licences)
#
# The registry pattern, the boolean-field streaming trick for missing-
# direction masks, and the overlap check are adapted from Autodesk XLB
# (https://github.com/Autodesk/XLB), Copyright 2023 Autodesk Inc.,
# licensed under the Apache License, Version 2.0.  Changes made for
# TensorLBM:
#   * registry is idempotent per instance and rejects re-registration,
#     unregister/reset support for test isolation (XLB only appends);
#   * BCs are declarative (kind/phase/face-or-mask/params) instead of
#     subclassing an Operator with per-backend implementations;
#   * missing masks are derived for PyTorch tensors from the TensorLBM
#     lattice constants (d3q19.C / d3q27.C) via padded slice shifts
#     instead of jax/warp kernels, with an independent brute-force
#     reference implementation for mutual verification;
#   * application dispatches to the existing boundaries3d functions;
#   * per-face periodicity support (XLB pads every face).
# ---------------------------------------------------------------------------

from __future__ import annotations

import warnings
from enum import Enum
from typing import Iterable, Sequence

import torch

__all__ = [
    "BC_ID_NONE",
    "BCKind",
    "BCPhase",
    "BoundaryCondition",
    "BoundaryConditionRegistry",
    "boundary_condition_registry",
    "build_bc_mask",
    "check_bc_overlaps",
    "check_bc_consistency",
    "derive_missing_mask",
    "derive_missing_mask_reference",
    "apply_boundary_conditions",
    "face_cells",
]

#: Reserved id: "solid / no boundary condition".  Never assigned to a BC.
BC_ID_NONE = 0


# ---------------------------------------------------------------------------
# Faces (convention identical to boundaries3d.far_field_bc_3d docstring)
# ---------------------------------------------------------------------------

_FACE_AXIS: dict[str, int] = {
    # layout is (Q, nz, ny, nx): dim 1 = z, dim 2 = y, dim 3 = x
    "x-": 3,
    "x+": 3,
    "y-": 2,
    "y+": 2,
    "z-": 1,
    "z+": 1,
}

#: Periodicity axis key for each face label.
_FACE_PERIOD_KEY: dict[str, str] = {
    "x-": "x",
    "x+": "x",
    "y-": "y",
    "y+": "y",
    "z-": "z",
    "z+": "z",
}


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        raise ValueError(f"shape must be (nz, ny, nx) with positive dims, got {tuple(shape)}")
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def face_cells(face: str, shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Boolean ``(nz, ny, nx)`` mask selecting one domain face plane.

    Face labels follow the convention documented on
    :func:`tensorlbm.boundaries3d.far_field_bc_3d`: ``"x-"``/``"x+"`` are
    the ``f[..., 0]``/``f[..., -1]`` planes (flow direction), ``"y-"``/
    ``"y+"`` the ``f[:, :, 0, :]``/``f[:, :, -1, :]`` planes and ``"z-"``/
    ``"z+"`` the ``f[:, 0, :, :]``/``f[:, -1, :, :]`` planes.
    """
    if face not in _FACE_AXIS:
        raise ValueError(f"face must be one of {sorted(_FACE_AXIS)}, got {face!r}")
    nz, ny, nx = _validate_shape(shape)
    cells = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    axis = _FACE_AXIS[face]
    index = 0 if face.endswith("-") else -1
    if axis == 1:  # z faces: f[:, 0, :, :] / f[:, -1, :, :]
        cells[index, :, :] = True
    elif axis == 2:  # y faces: f[:, :, 0, :] / f[:, :, -1, :]
        cells[:, index, :] = True
    else:  # x faces (axis == 3): f[:, :, :, 0] / f[:, :, :, -1]
        cells[:, :, index] = True
    return cells


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BCPhase(str, Enum):
    """When a BC is applied inside one LBM step.

    Mirrors lettuce's ``pre_boundaries``/``post_boundaries`` hooks and XLB's
    ``ImplementationStep`` (both cited in the module docstring).
    """

    PRE_STREAMING = "pre_streaming"
    POST_STREAMING = "post_streaming"


class BCKind(str, Enum):
    """Declarative boundary-condition kinds backed by ``boundaries3d``."""

    BOUNCE_BACK = "bounce_back"
    ZOU_HE_INLET_VELOCITY = "zou_he_inlet_velocity"
    ZOU_HE_OUTLET_PRESSURE = "zou_he_outlet_pressure"
    MOVING_LID = "moving_lid"
    FAR_FIELD = "far_field"
    PERIODIC = "periodic"


#: Kinds applied to an explicit cell mask (as opposed to a domain face).
_MASK_KINDS = frozenset({BCKind.BOUNCE_BACK})


class BoundaryCondition:
    """One declarative boundary condition.

    Exactly one of *mask* (arbitrary cell set) or *face* (a domain plane
    label, see :func:`face_cells`) must be given; ``FAR_FIELD`` spans a
    face list via ``params["faces"]`` and takes no *mask*/*face*.

    The integer ``id`` is assigned when the instance is registered with a
    :class:`BoundaryConditionRegistry`; applying the BC inside a step
    selects its cells via ``bc_mask == id`` (or the cached mask when no
    ``bc_mask`` field is built).
    """

    def __init__(
        self,
        kind: BCKind | str,
        *,
        phase: BCPhase | str = BCPhase.POST_STREAMING,
        mask: torch.Tensor | None = None,
        face: str | None = None,
        params: dict | None = None,
        name: str | None = None,
    ) -> None:
        self.kind = BCKind(kind)
        self.phase = BCPhase(phase)
        self.params: dict = dict(params or {})
        self.name = name or f"{self.kind.value}"
        self.id: int | None = None

        if self.kind in _MASK_KINDS:
            if mask is None or face is not None:
                raise ValueError(f"{self.kind} requires a cell mask (not a face)")
            if mask.dtype != torch.bool or mask.ndim != 3:
                raise ValueError("mask must be a boolean (nz, ny, nx) tensor")
            self.mask = mask
            self.face = None
        elif self.kind is BCKind.FAR_FIELD:
            if mask is not None or face is not None:
                raise ValueError("FAR_FIELD takes no mask/face; use params['faces']")
            faces = self.params.get("faces", ["x-", "x+", "y-", "y+", "z-", "z+"])
            for f in faces:
                if f not in _FACE_AXIS:
                    raise ValueError(f"invalid far-field face {f!r}")
            self.mask = None
            self.face = None
            self.params["faces"] = list(faces)
        elif self.kind is BCKind.PERIODIC:
            self.mask = None
            face = face or self.params.pop("face", None)
            if face is not None and face not in _FACE_AXIS:
                raise ValueError(f"invalid face {face!r}")
            self.face = face
        else:
            if face is None or mask is not None:
                raise ValueError(f"{self.kind} is a plane BC: give face= (not mask=)")
            if face not in _FACE_AXIS:
                raise ValueError(f"face must be one of {sorted(_FACE_AXIS)}, got {face!r}")
            self.mask = None
            self.face = face

    def cells(self, shape: Sequence[int], device: torch.device) -> torch.Tensor:
        """Boolean ``(nz, ny, nx)`` cell set of this BC on the given grid."""
        if self.kind is BCKind.FAR_FIELD:
            cells = torch.zeros(_validate_shape(shape), dtype=torch.bool, device=device)
            for face in self.params["faces"]:
                cells |= face_cells(face, shape, device)
            return cells
        if self.mask is not None:
            nz, ny, nx = _validate_shape(shape)
            if tuple(self.mask.shape) != (nz, ny, nx):
                raise ValueError(
                    f"mask shape {tuple(self.mask.shape)} does not match grid {(nz, ny, nx)}"
                )
            return self.mask.to(device)
        if self.face is not None:
            return face_cells(self.face, shape, device)
        return torch.zeros(_validate_shape(shape), dtype=torch.bool, device=device)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        target = "mask" if self.mask is not None else (self.face or "-")
        return (
            f"BoundaryCondition({self.name}, kind={self.kind.value}, id={self.id}, target={target})"
        )


class BoundaryConditionRegistry:
    """Maps BC instances to unique positive integer ids (0 = none/solid).

    Adapted from XLB's ``BoundaryConditionRegistry`` (Apache-2.0); changes
    are listed in the module-level attribution block.
    """

    def __init__(self) -> None:
        self.id_to_bc: dict[int, BoundaryCondition] = {}
        self.bc_to_id: dict[int, int] = {}
        self.next_id: int = 1

    def register(self, bc: BoundaryCondition) -> int:
        """Assign *bc* the next free id and return it."""
        if not isinstance(bc, BoundaryCondition):
            raise TypeError(f"expected BoundaryCondition, got {type(bc).__name__}")
        if bc.id is not None:
            raise ValueError(
                f"boundary condition {bc.name!r} already has id {bc.id}; "
                "create a new instance or unregister it first"
            )
        _id = self.next_id
        self.next_id += 1
        self.id_to_bc[_id] = bc
        self.bc_to_id[id(bc)] = _id
        bc.id = _id
        return _id

    def unregister(self, bc: BoundaryCondition) -> None:
        """Remove *bc* from the registry (its id is retired, never reused)."""
        _id = self.bc_to_id.pop(id(bc), None)
        if _id is None:
            raise KeyError(f"boundary condition {bc.name!r} is not registered")
        del self.id_to_bc[_id]
        bc.id = None

    def id_of(self, bc: BoundaryCondition) -> int:
        """Return the registered id of *bc* (raises when unregistered)."""
        _id = self.bc_to_id.get(id(bc))
        if _id is None:
            raise KeyError(f"boundary condition {bc.name!r} is not registered")
        return _id

    def bc_of(self, _id: int) -> BoundaryCondition:
        """Return the BC registered under *_id*."""
        if _id == BC_ID_NONE:
            raise ValueError("id 0 is reserved for solid / no boundary condition")
        try:
            return self.id_to_bc[_id]
        except KeyError:
            raise KeyError(f"no boundary condition registered with id {_id}") from None

    def __contains__(self, bc: object) -> bool:
        return isinstance(bc, BoundaryCondition) and id(bc) in self.bc_to_id

    def __len__(self) -> int:
        return len(self.id_to_bc)

    def reset(self) -> None:
        """Drop all registrations (test isolation; ids restart at 1)."""
        for bc in list(self.id_to_bc.values()):
            bc.id = None
        self.id_to_bc.clear()
        self.bc_to_id.clear()
        self.next_id = 1


#: Process-wide registry singleton (XLB-style).
boundary_condition_registry = BoundaryConditionRegistry()


def build_bc_mask(
    shape: Sequence[int],
    bcs: Iterable[BoundaryCondition],
    *,
    phase: BCPhase | str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Integer ``(nz, ny, nx)`` field: 0 = no BC, else the BC's id.

    This is the field a compiled kernel consumes: each BC's cells are
    selected with a single ``bc_mask == id`` comparison (branch-free,
    no data-dependent control flow).  Overlapping BCs would silently
    overwrite each other here (last BC in *bcs* wins) — run
    :func:`check_bc_overlaps` first.

    With *phase* set, only BCs of that application phase participate:
    pass one mask per phase when PRE and POST BCs share cells (e.g. the
    cavity lid plane meets the stationary walls on its edge lines); a
    single combined mask would hide those cells from the earlier phase's
    selection.
    """
    nz, ny, nx = _validate_shape(shape)
    if device is None:
        device = torch.device("cpu")
    if phase is not None:
        phase = BCPhase(phase)
    bc_mask = torch.zeros((nz, ny, nx), dtype=torch.int64, device=device)
    for bc in bcs:
        if phase is not None and bc.phase is not phase:
            continue
        if bc.id is None:
            raise ValueError(
                f"boundary condition {bc.name!r} is not registered; "
                "call boundary_condition_registry.register(bc) first"
            )
        cells = bc.cells((nz, ny, nx), bc_mask.device)
        if not bool(cells.any()):
            raise ValueError(f"boundary condition {bc.name!r} selects no cells")
        bc_mask[cells] = bc.id
    return bc_mask


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def check_bc_overlaps(
    bcs: Sequence[BoundaryCondition],
    shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> None:
    """Reject BC lists whose cell sets intersect within one phase.

    Adapted from XLB's ``check_boundary_overlaps`` (Apache-2.0).  Changes:
    operates on boolean cell masks (TensorLBM BCs are mask/face based, so
    within-BC duplicate indices cannot occur by construction); the failure
    mode is configurable — ``strict=True`` (default) raises ``ValueError``,
    ``strict=False`` only warns, matching XLB's non-Warp path where "the
    order in the bc list matters" is an accepted, documented state (the
    verified Poiseuille benchmark is one such case: its bounce-back wall
    legitimately reprocesses the solid cells of the inlet/outlet planes
    after the plane BCs); and **cross-phase** overlaps (one PRE_STREAMING,
    one POST_STREAMING BC) are allowed outright, because the two-phase
    pipeline applies them in a fixed order (pre before post) — the cavity
    lid plane meets the stationary walls exactly on its edge lines.
    """
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    for i, bc_a in enumerate(bcs):
        cells_a = bc_a.cells(shape, device)
        if not bool(cells_a.any()):
            raise ValueError(f"boundary condition {bc_a.name!r} selects no cells")
        for bc_b in bcs[i + 1 :]:
            if bc_a.phase is not bc_b.phase:
                continue  # fixed pipeline order: pre always precedes post
            overlap = cells_a & bc_b.cells(shape, device)
            if bool(overlap.any()):
                message = (
                    f"boundary conditions {bc_a.name!r} and {bc_b.name!r} overlap on "
                    f"{int(overlap.sum().item())} cells; the applied result depends on "
                    "BC ordering — give each cell exactly one BC unless the ordering "
                    "is intentional"
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, UserWarning, stacklevel=2)


_PLANE_KINDS = frozenset(
    {BCKind.ZOU_HE_INLET_VELOCITY, BCKind.ZOU_HE_OUTLET_PRESSURE, BCKind.MOVING_LID}
)


def check_bc_consistency(
    bcs: Sequence[BoundaryCondition],
    shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    strict_overlap: bool = True,
) -> None:
    """Validate a BC list against the grid before running.

    Checks (in order): all BCs registered, non-empty cell sets, no
    same-phase overlaps (see :func:`check_bc_overlaps` for the
    ``strict_overlap`` switch and the cross-phase exemption), and
    plane-kind BCs exactly covering their declared face.
    """
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    for bc in bcs:
        if bc.id is None or bc.id == BC_ID_NONE:
            raise ValueError(f"boundary condition {bc.name!r} has no valid registry id")
    check_bc_overlaps(bcs, shape, device=device, strict=strict_overlap)
    for bc in bcs:
        if bc.kind in _PLANE_KINDS:
            expected = face_cells(bc.face or "", shape, device)
            if not bool(torch.equal(bc.cells(shape, device), expected)):
                raise ValueError(
                    f"plane BC {bc.name!r} ({bc.kind.value}) must cover exactly face {bc.face!r}"
                )


# ---------------------------------------------------------------------------
# Missing-direction masks ("stream a boolean field once", XLB method)
# ---------------------------------------------------------------------------


def _lattice_c(lattice: str, device: torch.device) -> torch.Tensor:
    if lattice.upper() == "D3Q19":
        from .d3q19 import C
    elif lattice.upper() == "D3Q27":
        from .d3q27 import C
    else:
        raise ValueError(f"unsupported lattice {lattice!r}; expected D3Q19 or D3Q27")
    return C.to(device)


def _normalise_periodic(
    periodic: bool | Sequence[bool] | dict[str, bool],
) -> dict[str, bool]:
    """Normalise a periodicity spec to ``{"x": bool, "y": bool, "z": bool}``."""
    if isinstance(periodic, bool):
        value = {"x": periodic, "y": periodic, "z": periodic}
    elif isinstance(periodic, dict):
        value = {axis: bool(periodic.get(axis, False)) for axis in ("x", "y", "z")}
    else:
        seq = tuple(bool(p) for p in periodic)
        if len(seq) != 3:
            raise ValueError("periodic sequence must have 3 entries (z, y, x)")
        value = {"z": seq[0], "y": seq[1], "x": seq[2]}
    return value


def derive_missing_mask(
    shape: Sequence[int],
    *,
    solid_mask: torch.Tensor | None = None,
    periodic: bool | Sequence[bool] | dict[str, bool] = False,
    lattice: str = "D3Q19",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Derive the missing-direction mask by streaming a boolean field once.

    Pull-scheme streaming reads ``f_new[q, x] = f_old[q, x - c_q]``, so a
    population is *missing* at ``x`` exactly when the pull source
    ``x - c_q`` is **blocked** — outside the domain on a non-periodic face
    or inside the solid.  Instead of a hand-written per-direction table
    (the historical sign-error trap recorded in ``triton_fused.py``), the
    blocked field is padded by one cell (pad value = "blocked" on
    non-periodic faces, free on periodic faces) and pulled through the
    streaming shift once::

        missing[q, z, y, x] = blocked_pad[z - c_z + 1, y - c_y + 1, x - c_x + 1]

    implemented as one slice per direction, driven purely by the lattice
    velocity constants.

    Args:
        shape: grid ``(nz, ny, nx)``.
        solid_mask: optional boolean ``(nz, ny, nx)`` solid field.
        periodic: per-axis periodicity.  ``False`` (default) blocks all
            six outside faces; a 3-tuple is ``(z, y, x)`` order matching
            the tensor layout; a dict uses ``"x"``/``"y"``/``"z"`` keys.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        device: target device (defaults to ``solid_mask``'s or CPU).

    Returns:
        Boolean tensor of shape ``(Q, nz, ny, nx)`` — ``True`` where the
        direction's population must be supplied by a boundary condition.
    """
    nz, ny, nx = _validate_shape(shape)
    if device is None:
        device = solid_mask.device if solid_mask is not None else torch.device("cpu")
    device = torch.device(device)
    periodic_by_axis = _normalise_periodic(periodic)

    blocked = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    if solid_mask is not None:
        if solid_mask.shape != blocked.shape or solid_mask.dtype != torch.bool:
            raise ValueError(
                f"solid_mask must be boolean with shape {(nz, ny, nx)}, "
                f"got {tuple(solid_mask.shape)}/{solid_mask.dtype}"
            )
        blocked |= solid_mask.to(device)

    # Pad by one cell per axis; the padding is "blocked" wherever the
    # domain face is not periodic (a pull from there is out of bounds).
    # Periodic faces never read their pad — their pull source wraps with
    # modulo arithmetic inside the domain (see _pull_source_index).
    blocked_pad = torch.zeros((nz + 2, ny + 2, nx + 2), dtype=torch.bool, device=device)
    blocked_pad[1:-1, 1:-1, 1:-1] = blocked
    for axis, key in ((1, "z"), (2, "y"), (3, "x")):
        if not periodic_by_axis[key]:
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis - 1] = 0
            high[axis - 1] = -1
            blocked_pad[tuple(low)] = True
            blocked_pad[tuple(high)] = True

    c = _lattice_c(lattice, device)
    q_n = c.shape[0]
    missing = torch.empty((q_n, nz, ny, nx), dtype=torch.bool, device=device)
    base_index = {
        "z": torch.arange(nz, device=device).view(-1, 1, 1),
        "y": torch.arange(ny, device=device).view(1, -1, 1),
        "x": torch.arange(nx, device=device).view(1, 1, -1),
    }
    for q in range(q_n):
        # Pull sources in padded coordinates (offset +1); periodic axes
        # wrap inside the domain instead of reading the pad.  C columns
        # are (cx, cy, cz) while tensor axes are (z, y, x).
        src = tuple(
            _pull_source_index(base_index[key], int(c[q, col]), n, periodic_by_axis[key])
            for key, n, col in (("z", nz, 2), ("y", ny, 1), ("x", nx, 0))
        )
        missing[q] = blocked_pad[src]
    return missing


def _pull_source_index(coord: torch.Tensor, shift: int, n: int, is_periodic: bool) -> torch.Tensor:
    """Padded-coordinate pull-source index for one axis.

    Non-periodic: ``coord - shift + 1`` — out-of-bounds sources land on
    the blocked pad cells (0 or n+1).  Periodic: ``(coord - shift) % n + 1``
    wraps inside the domain, exactly like the modulo gather of
    :func:`tensorlbm.solver3d.stream3d`.
    """
    base = coord - shift
    if is_periodic:
        return (base % n) + 1
    return base + 1


def derive_missing_mask_reference(
    shape: Sequence[int],
    *,
    solid_mask: torch.Tensor | None = None,
    periodic: bool | Sequence[bool] | dict[str, bool] = False,
    lattice: str = "D3Q19",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Independent brute-force reference for :func:`derive_missing_mask`.

    Second algorithm for mutual verification (used by the test suite):
    instead of shifting the padded boolean field, every direction's pull
    source index is computed explicitly per axis and tested against the
    domain bounds (non-periodic faces) or wrapped (periodic faces) and
    the solid mask.  Agreement of the two algorithms plus the hand-written
    library constants in ``boundaries3d`` gives high confidence that no
    direction sign was transcribed incorrectly.
    """
    nz, ny, nx = _validate_shape(shape)
    if device is None:
        device = solid_mask.device if solid_mask is not None else torch.device("cpu")
    device = torch.device(device)
    periodic_by_axis = _normalise_periodic(periodic)
    solid = (
        solid_mask.to(device)
        if solid_mask is not None
        else torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    )

    zz = torch.arange(nz, device=device).view(-1, 1, 1)
    yy = torch.arange(ny, device=device).view(1, -1, 1)
    xx = torch.arange(nx, device=device).view(1, 1, -1)

    c = _lattice_c(lattice, device)
    q_n = c.shape[0]
    missing = torch.empty((q_n, nz, ny, nx), dtype=torch.bool, device=device)
    ((nz, periodic_by_axis["z"]), (ny, periodic_by_axis["y"]), (nx, periodic_by_axis["x"]))

    def _source(coord: torch.Tensor, shift: int, n: int, is_periodic: bool) -> torch.Tensor:
        src = coord - shift
        if is_periodic:
            return src % n
        return torch.where((src >= 0) & (src < n), src, -1)  # -1 marks out-of-bounds

    for q in range(q_n):
        cx, cy, cz = int(c[q, 0]), int(c[q, 1]), int(c[q, 2])
        sx = _source(xx, cx, nx, periodic_by_axis["x"])
        sy = _source(yy, cy, ny, periodic_by_axis["y"])
        sz = _source(zz, cz, nz, periodic_by_axis["z"])
        in_bounds = (sx >= 0) & (sy >= 0) & (sz >= 0)
        # Safe gather: clamp out-of-bounds to 0, mask afterwards.
        src_solid = solid[sz.clamp(0), sy.clamp(0), sx.clamp(0)]
        missing[q] = (~in_bounds) | src_solid
    return missing


# ---------------------------------------------------------------------------
# Application (dispatch to the verified boundaries3d functions)
# ---------------------------------------------------------------------------


def _selection(
    bc: BoundaryCondition,
    bc_mask: torch.Tensor | None,
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Cell selection for *bc*: the ``bc_mask == id`` product when available."""
    if bc_mask is not None:
        return bc_mask == bc.id
    return bc.cells(shape, device)


def apply_boundary_conditions(
    f: torch.Tensor,
    bcs: Iterable[BoundaryCondition],
    *,
    phase: BCPhase | str,
    bc_mask: torch.Tensor | None = None,
    f_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply all BCs of *phase* to *f*, dispatching by kind.

    Args:
        f: populations ``(Q, nz, ny, nx)``.
        bcs: boundary conditions (must be registered when *bc_mask* is used).
        phase: ``BCPhase.PRE_STREAMING`` (between collision and streaming;
            pass ``f_pre`` = pre-collision state) or
            ``BCPhase.POST_STREAMING`` (after streaming).
        bc_mask: optional integer id field from :func:`build_bc_mask`;
            cell-set BCs then select with ``bc_mask == id`` (branch-free).
        f_pre: pre-collision populations, required by PRE_STREAMING
            bounce-back (half-way bounce-back reads the pre-collision
            state, matching ``benchmarks/verified/cavity/3d/run.py``).

    Returns:
        Updated populations (input not modified in place where the
        underlying library function clones; BOUNCE_BACK PRE returns a new
        tensor via ``torch.where``).
    """
    phase = BCPhase(phase)
    shape = (f.shape[1], f.shape[2], f.shape[3])
    device = f.device

    from .boundaries3d import (
        bounce_back_cells_3d,
        far_field_bc_3d,
        zou_he_inlet_velocity_3d,
        zou_he_moving_lid_3d,
        zou_he_outlet_pressure_3d,
    )
    from .d3q19 import OPPOSITE

    for bc in bcs:
        if bc.phase is not phase:
            continue
        if bc_mask is not None and bc.id is None:
            raise ValueError(f"boundary condition {bc.name!r} is not registered")
        if bc.kind is BCKind.PERIODIC:
            continue  # handled by the (periodic) streaming operator
        if bc.kind is BCKind.BOUNCE_BACK:
            sel = _selection(bc, bc_mask, shape, device).to(device)
            if phase is BCPhase.PRE_STREAMING:
                if f_pre is None:
                    raise ValueError(
                        "PRE_STREAMING bounce-back requires f_pre (pre-collision state)"
                    )
                opp = OPPOSITE.to(device)
                f = torch.where(sel.unsqueeze(0), f_pre[opp], f)
            else:
                f = bounce_back_cells_3d(f, sel)
        elif bc.kind is BCKind.ZOU_HE_INLET_VELOCITY:
            f = zou_he_inlet_velocity_3d(
                f,
                float(bc.params["u_in"]),
                uy_in=float(bc.params.get("uy_in", 0.0)),
                uz_in=float(bc.params.get("uz_in", 0.0)),
            )
        elif bc.kind is BCKind.ZOU_HE_OUTLET_PRESSURE:
            f = zou_he_outlet_pressure_3d(f, rho_out=float(bc.params.get("rho_out", 1.0)))
        elif bc.kind is BCKind.MOVING_LID:
            f = zou_he_moving_lid_3d(f, float(bc.params["u_lid"]))
        elif bc.kind is BCKind.FAR_FIELD:
            faces = set(bc.params["faces"])
            bc_config = {
                "far_field_faces": sorted(faces & {"y-", "y+", "z-", "z+"}),
                "periodic_faces": [],
            }
            f = far_field_bc_3d(
                f,
                float(bc.params["u_in"]),
                obstacle_mask=bc.params.get("obstacle_mask"),
                uy=float(bc.params.get("uy", 0.0)),
                uz=float(bc.params.get("uz", 0.0)),
                bc_config=bc_config,
            )
        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"unsupported boundary kind {bc.kind}")
    return f
