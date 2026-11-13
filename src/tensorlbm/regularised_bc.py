"""Regularised boundary condition rebuilt from the non-equilibrium stress.

Why this module exists
----------------------
The shipped wall operators assume that a wall can be closed with *bounce back*:
``f_q = f_opp(q)`` (plus a Ladd momentum term for a moving wall).  That closes
the missing populations with a *reflection*, not with the *stress* the flow
actually carries at the wall, so every wall node silently injects the
non-equilibrium content of the mirrored direction -- including its ghost
(higher-order Hermite) component, which a BGK collision then damps at a rate
that depends on ``tau`` rather than on the physics.

The regularised wall closure used by ``joconnor22/LIFE`` closes the same
populations from the *second moment* of the distribution instead:

1. the incoming (fluid-side) and in-plane populations give the wall density
   through the Zou/He closure ``rho = (2 f+ + f0) / (1 - n.u)``;
2. the missing directions are first filled anti-symmetrically,
   ``f_q = f_q^eq + (f_opp(q) - f_opp(q)^eq)``;
3. the non-equilibrium stress ``S_ab = sum_q c_qa c_qb (f_q - f_q^eq)`` is
   accumulated over **all** directions of the wall node;
4. every population of the node is rewritten as
   ``f_q = f_q^eq + (w_q / (2 cs^4)) Q_q : S`` with
   ``Q_q = c_q c_q - cs^2 I`` (the second-order Hermite projector).

Step 4 is a projection: it keeps the density, the momentum and ``S`` exactly
and removes all higher-order (ghost) content -- the property that makes this
closure second order and time-step independent at a wall.  The algorithm here
is a semantics transplant of ``src/Grid.cpp:386-466`` (``regularisedBC``) and
its call sites (``src/Grid.cpp:321/333/351/368`` for ``eWall`` / ``eVelocity``
/ ``eFreeSlip`` / ``ePressure``); the OSS C++ code was read, not copied, and
the differences are listed in
``xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/OSS_DIFF.md``.

Public surface
--------------
``LatticeSpec`` / ``D2Q9`` / ``D3Q19``
    Velocity sets (ordering identical to :mod:`tensorlbm.d2q9` /
    :mod:`tensorlbm.d3q19`, asserted in the tests).
``equilibrium``, ``macroscopic``, ``nonequilibrium_stress``, ``third_moment``
    Moment helpers, all NumPy, shape ``(Q, ...)`` like the rest of the repo.
``regularised_face``
    The OSS closure applied to one boundary plane held in a ``(Q, ...)`` slice.
``apply_regularised_wall``
    Same, applied to one plane of a full ``(Q, ...)`` field (the form the
    solvers call).
``zero_gradient_tangential``
    LIFE's 1st/2nd order ``zeroGradient`` extrapolation used by the pressure
    and free-slip variants.
``run_channel`` / ``main``
    A dependency-light D2Q9 plane-channel driver (Couette, body-force
    Poiseuille, oscillating wall) with ``wall_bc in {regularised,
    bounce_back}``; ``python -m tensorlbm.regularised_bc`` is the CLI entry.

NumPy only, so the same code runs on the torch stack-free cluster nodes
(``channel_kernel_numpy`` is the in-repo precedent).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "D2Q9",
    "D3Q19",
    "ChannelConfig",
    "LatticeSpec",
    "apply_regularised_wall",
    "complete_unknowns",
    "equilibrium",
    "macroscopic",
    "main",
    "nonequilibrium_stress",
    "project_second_order",
    "regularised_face",
    "run_channel",
    "third_moment",
    "zero_gradient_tangential",
]


# ---------------------------------------------------------------------------
# Lattice velocity sets (ordering identical to tensorlbm.d2q9 / d3q19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeSpec:
    """A velocity set with its weights and the opposite-direction map."""

    name: str
    c: np.ndarray  # (Q, dims) integer velocities
    w: np.ndarray  # (Q,) weights
    opposite: np.ndarray  # (Q,) index of the opposite direction
    cs2: float = 1.0 / 3.0

    @property
    def q(self) -> int:
        return int(self.c.shape[0])

    @property
    def dims(self) -> int:
        return int(self.c.shape[1])

    def __post_init__(self) -> None:
        if self.c.shape[0] != self.w.shape[0] or self.c.shape[0] != self.opposite.shape[0]:
            raise ValueError("LatticeSpec: c/w/opposite must have the same length")
        if not np.isclose(self.w.sum(), 1.0):
            raise ValueError(f"LatticeSpec {self.name}: weights do not sum to 1 ({self.w.sum()})")
        if not np.array_equal(self.opposite[self.opposite], np.arange(self.q)):
            raise ValueError(f"LatticeSpec {self.name}: opposite map is not an involution")


D2Q9 = LatticeSpec(
    name="D2Q9",
    c=np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
        dtype=np.int64,
    ),
    w=np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]),
    opposite=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64),
)

D3Q19 = LatticeSpec(
    name="D3Q19",
    c=np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
        ],
        dtype=np.int64,
    ),
    w=np.array(
        [1 / 3]
        + [1 / 18] * 6
        + [1 / 36] * 12,
    ),
    opposite=np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64),
)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def _check_field(spec: LatticeSpec, f: np.ndarray, *, what: str = "f") -> None:
    if f.ndim < 2:
        raise ValueError(f"{what} must have shape (Q, ...), got {f.shape}")
    if f.shape[0] != spec.q:
        raise ValueError(f"{what} has {f.shape[0]} directions, {spec.name} expects {spec.q}")


def equilibrium(spec: LatticeSpec, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-order D2Q9/D3Q19 equilibrium, ``(Q, ...)`` from ``rho`` and ``(dims, ...) u``."""
    if u.shape[0] != spec.dims:
        raise ValueError(f"u must be ({spec.dims}, ...), got {u.shape}")
    if u.shape[1:] != rho.shape:
        raise ValueError(f"u {u.shape} and rho {rho.shape} shapes disagree")
    cu = np.tensordot(spec.c.astype(rho.dtype), u, axes=(1, 0))
    usq = (u * u).sum(axis=0)
    shape = (-1,) + (1,) * rho.ndim
    return spec.w.reshape(shape) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def macroscopic(spec: LatticeSpec, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``rho`` (``...``) and momentum density ``u`` (``dims, ...``), ``u = sum f c / rho``."""
    _check_field(spec, f)
    rho = f.sum(axis=0)
    mom = np.tensordot(spec.c.astype(f.dtype).T, f, axes=(1, 0))
    return rho, mom / rho


def nonequilibrium_stress(
    spec: LatticeSpec,
    f: np.ndarray,
    *,
    rho: np.ndarray | None = None,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """Non-equilibrium momentum flux ``S_ab = sum_q c_qa c_qb (f_q - f_q^eq)``."""
    _check_field(spec, f)
    if rho is None or u is None:
        rho, u = macroscopic(spec, f)
    feq = equilibrium(spec, rho, u)
    fneq = f - feq
    c = spec.c.astype(f.dtype)
    out = np.empty((spec.dims, spec.dims) + f.shape[1:], dtype=f.dtype)
    for a in range(spec.dims):
        for b in range(a, spec.dims):
            moment = np.tensordot(c[:, a] * c[:, b], fneq, axes=(0, 0))
            out[a, b] = moment
            out[b, a] = moment
    return out


def third_moment(spec: LatticeSpec, f: np.ndarray) -> np.ndarray:
    """Third moment ``M3_abc = sum_q c_qa c_qb c_qc f_q`` (the ghost/kinetic content)."""
    _check_field(spec, f)
    c = spec.c.astype(f.dtype)
    out = np.empty((spec.dims,) * 3 + f.shape[1:], dtype=f.dtype)
    for a in range(spec.dims):
        for b in range(spec.dims):
            for d in range(spec.dims):
                out[a, b, d] = np.tensordot(c[:, a] * c[:, b] * c[:, d], f, axes=(0, 0))
    return out


# ---------------------------------------------------------------------------
# The wall closure
# ---------------------------------------------------------------------------


def _broadcast_wall_velocity(
    u_wall: np.ndarray | tuple | list | float,
    *,
    shape: tuple[int, ...],
    dims: int,
    dtype: np.dtype,
) -> np.ndarray:
    if isinstance(u_wall, np.ndarray):
        if u_wall.shape == (dims,) + shape:
            return u_wall.astype(dtype, copy=False)
        if u_wall.shape == (dims,):
            return np.broadcast_to(u_wall.reshape((dims,) + (1,) * len(shape)), (dims,) + shape)
        raise ValueError(f"u_wall tensor must be ({dims},)+{shape} or ({dims},), got {u_wall.shape}")
    values = [float(u_wall)] if np.isscalar(u_wall) else [float(v) for v in u_wall]
    if len(values) == 1:
        values = values + [0.0] * (dims - 1)
    if len(values) != dims:
        raise ValueError(f"u_wall must have 1 or {dims} components, got {len(values)}")
    out = np.empty((dims,) + shape, dtype=dtype)
    for d, value in enumerate(values):
        out[d] = value
    return out


def regularised_face(
    spec: LatticeSpec,
    f_face: np.ndarray,
    normal: np.ndarray | tuple[int, ...],
    *,
    kind: str = "velocity",
    u_wall: np.ndarray | tuple | list | float = 0.0,
    rho_bc: float | np.ndarray | None = None,
    u_tangential: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild one boundary plane from its non-equilibrium stress.

    Args:
        spec: velocity set.
        f_face: post-streaming populations on the boundary plane,
            ``(Q, ...)``; the plane index is already dropped.
        normal: **inward** unit normal of the wall (integer vector, one
            non-zero component).  Directions with ``c.n = +1`` are the
            unknown (wall -> fluid) populations and are rebuilt.
        kind: ``"velocity"`` (wall/no-slip/moving wall), ``"pressure"``
            (fixed density, normal velocity from the Zou/He closure,
            tangential velocity extrapolated) or ``"free_slip"``
            (zero normal velocity, tangential velocity extrapolated).
        u_wall: wall velocity for ``kind="velocity"``; scalar/sequence or
            ``(dims,)+f_face.shape[1:]``.
        rho_bc: prescribed density for ``kind="pressure"``.
        u_tangential: prescribed tangential velocity, ``(dims,)+shape`` or
            ``(dims,)``.  Required for ``"pressure"``/``"free_slip"`` unless
            the caller accepts a zero tangential velocity.

    Returns:
        New populations ``(Q, ...)`` for the boundary plane.
    """
    _check_field(spec, f_face, what="f_face")
    normal = np.asarray(normal, dtype=np.int64)
    if normal.shape != (spec.dims,):
        raise ValueError(f"normal must be ({spec.dims},), got {normal.shape}")
    if np.count_nonzero(normal) != 1 or np.any(np.abs(normal) > 1):
        raise ValueError(f"normal must be a unit vector on one axis, got {normal.tolist()}")
    kind = str(kind).lower()
    if kind not in ("velocity", "pressure", "free_slip"):
        raise ValueError(f"kind must be velocity|pressure|free_slip, got {kind!r}")

    shape = f_face.shape[1:]
    dtype = f_face.dtype
    cn = spec.c @ normal  # (Q,) in {-1, 0, +1}
    neg = cn == -1
    zero = cn == 0

    # Zou/He density closure: f+ are the fluid-side (incoming) populations,
    # f0 the in-plane ones (src/Grid.cpp:414-425).
    fplus = f_face[neg].sum(axis=0)
    fzero = f_face[zero].sum(axis=0)

    if kind == "velocity":
        u = _broadcast_wall_velocity(u_wall, shape=shape, dims=spec.dims, dtype=dtype)
        un = np.tensordot(normal.astype(dtype), u, axes=(0, 0))
        denom = 1.0 - un
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError("regularised_face: 1 - n.u vanishes; wall velocity is supersonic")
        rho = (2.0 * fplus + fzero) / denom
    elif kind == "pressure":
        if rho_bc is None:
            raise ValueError("kind='pressure' requires rho_bc")
        rho = np.broadcast_to(np.asarray(rho_bc, dtype=dtype), shape).copy() if np.ndim(rho_bc) else np.full(
            shape, float(rho_bc), dtype=dtype
        )
        un = 1.0 - (2.0 * fplus + fzero) / rho
        ut = _resolve_tangential(spec, normal, u_tangential, shape=shape, dtype=dtype)
        u = ut + normal.astype(dtype).reshape((-1,) + (1,) * len(shape)) * un
    else:  # free_slip
        ut = _resolve_tangential(spec, normal, u_tangential, shape=shape, dtype=dtype)
        u = ut.copy()
        rho = (2.0 * fplus + fzero)  # n.u == 0
    f_completed = complete_unknowns(spec, f_face, normal, rho=rho, u=u)
    return project_second_order(spec, f_completed, rho=rho, u=u)


def complete_unknowns(
    spec: LatticeSpec,
    f_face: np.ndarray,
    normal: np.ndarray | tuple[int, ...],
    *,
    rho: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Step 2 of the closure: fill ``c.n = +1`` populations anti-symmetrically.

    ``f_q <- f_q^eq + (f_opp(q) - f_opp(q)^eq)`` for the unknown directions
    (``src/Grid.cpp:437-451``); known directions are left untouched.
    """
    _check_field(spec, f_face, what="f_face")
    normal = np.asarray(normal, dtype=np.int64)
    cn = spec.c @ normal
    feq = equilibrium(spec, rho, u)
    out = f_face.copy()
    for q in np.nonzero(cn == 1)[0]:
        o = spec.opposite[q]
        out[q] = feq[q] + (f_face[o] - feq[o])
    return out


def project_second_order(
    spec: LatticeSpec,
    f: np.ndarray,
    *,
    rho: np.ndarray | None = None,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """Steps 3-4: rebuild every population from the non-equilibrium stress.

    ``f_q = f_q^eq + (w_q / (2 cs^4)) Q_q : S`` with ``S`` the non-equilibrium
    stress of ``f`` (``src/Grid.cpp:453-466``).  The map preserves ``rho``,
    momentum and ``S`` exactly and removes the third and higher moments of the
    non-equilibrium part (the regularisation property).
    """
    _check_field(spec, f, what="f")
    if rho is None or u is None:
        rho, u = macroscopic(spec, f)
    feq = equilibrium(spec, rho, u)
    stress = nonequilibrium_stress(spec, f, rho=rho, u=u)
    dtype = f.dtype
    c = spec.c.astype(dtype)
    q2 = np.einsum("qa,qb->qab", c, c) - spec.cs2 * np.eye(spec.dims, dtype=dtype)
    proj = np.einsum("qab,ab...->q...", q2, stress)
    shape = (-1,) + (1,) * (f.ndim - 1)
    weight = (spec.w.astype(dtype) / (2.0 * spec.cs2**2)).reshape(shape)
    return feq + weight * proj


def _resolve_tangential(
    spec: LatticeSpec,
    normal: np.ndarray,
    u_tangential: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if u_tangential is None:
        return np.zeros((spec.dims,) + shape, dtype=dtype)
    return _broadcast_wall_velocity(u_tangential, shape=shape, dims=spec.dims, dtype=dtype)


def zero_gradient_tangential(
    spec: LatticeSpec,
    u: np.ndarray,
    axis: int,
    side: str,
    *,
    order: int = 2,
) -> np.ndarray:
    """LIFE ``Utils::zeroGradient`` (inc/Utils.h:186-213) on a ``(dims, ...)`` field.

    Extrapolates the tangential components onto the boundary plane ``axis`` /
    ``side`` from the first (and, for ``order=2``, second) interior plane:
    ``(4 u_1 - u_2) / 3``.  The normal component is the caller's job.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if axis < 0 or axis >= spec.dims:
        raise ValueError(f"axis must be in [0, {spec.dims})")
    if side not in ("low", "high"):
        raise ValueError("side must be 'low' or 'high'")
    step = 1 if side == "low" else -1
    n = u.shape[axis + 1]

    def plane(offset: int) -> np.ndarray:
        index = [slice(None)] * u.ndim
        index[axis + 1] = offset if step > 0 else n - 1 + offset
        return u[tuple(index)]

    if order == 1:
        return plane(step).copy()
    return (4.0 / 3.0) * plane(step) - (1.0 / 3.0) * plane(2 * step)


def apply_regularised_wall(
    spec: LatticeSpec,
    f: np.ndarray,
    axis: int,
    side: str,
    *,
    u_wall: np.ndarray | tuple | list | float = 0.0,
    kind: str = "velocity",
    rho_bc: float | None = None,
    u_tangential: np.ndarray | None = None,
    component_of_axis: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply :func:`regularised_face` to one plane of a full ``(Q, ...)`` field.

    Args:
        f: populations ``(Q, ...)``; axis 0 is the direction axis.
        axis: spatial axis of the boundary plane (0 = first spatial axis).
        side: ``"low"`` (plane index 0, inward normal ``+e_axis``) or
            ``"high"`` (plane index -1, inward normal ``-e_axis``).

    Returns a new array; the input is not modified.
    """
    _check_field(spec, f)
    if axis < 0 or axis >= spec.dims:
        raise ValueError(f"axis must be in [0, {spec.dims})")
    if side not in ("low", "high"):
        raise ValueError("side must be 'low' or 'high'")
    index = [slice(None)] * f.ndim
    index[axis + 1] = 0 if side == "low" else -1
    index = tuple(index)
    # Array axis -> lattice component.  Default is the repo 2-D layout
    # (Q, ny, nx) whose lattice components are (x, y), i.e. reversed; a
    # (Q, nx, ny, nz) layout passes component_of_axis=(0, 1, 2).
    if component_of_axis is None:
        component_of_axis = tuple(range(spec.dims))[::-1]
    normal = np.zeros(spec.dims, dtype=np.int64)
    normal[component_of_axis[axis]] = 1 if side == "low" else -1
    face = regularised_face(
        spec,
        f[index],
        normal,
        kind=kind,
        u_wall=u_wall,
        rho_bc=rho_bc,
        u_tangential=u_tangential,
    )
    out = f.copy()
    out[index] = face
    return out


# ---------------------------------------------------------------------------
# D2Q9 plane-channel driver (Couette, Poiseuille, oscillating wall)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelConfig:
    """Grid and physics of the :func:`run_channel` driver.

    The grid is ``(ny, nx)`` with x periodic.  Rows ``0`` and ``ny - 1`` are the
    boundary planes; rows ``1 .. ny - 2`` are fluid.  The wall location
    convention follows the closure: a regularised plane *is* the wall
    (``y_wall = 0`` / ``ny - 1``), while bounce back places the no-slip plane
    half a link away (``y_wall = 0.5`` / ``ny - 1.5``).
    """

    case: str = "couette"
    wall_bc: str = "regularised"
    ny: int = 32
    nx: int = 2
    tau: float = 0.8
    u_wall: float = 0.05
    body_force: float = 0.0
    omega: float = 2.0 * np.pi / 200.0
    steps: int = 200000
    check_every: int = 200
    tol: float = 1.0e-10
    init: str = "rest"

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def wall_origin(self) -> float:
        """Height of the lower no-slip plane in lattice units."""
        return 0.0 if self.wall_bc == "regularised" else 0.5

    @property
    def height(self) -> float:
        """Distance between the two no-slip planes."""
        return (self.ny - 1) - (1 if self.wall_bc == "bounce_back" else 0)


def _stream(spec: LatticeSpec, f: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for q in range(spec.q):
        cx, cy = int(spec.c[q, 0]), int(spec.c[q, 1])
        out[q] = f[q] if (cy == 0 and cx == 0) else np.roll(f[q], (cy, cx), axis=(0, 1))
    return out


def _collide(spec: LatticeSpec, f: np.ndarray, tau: float) -> np.ndarray:
    rho, u = macroscopic(spec, f)
    feq = equilibrium(spec, rho, u)
    return f - (f - feq) / tau


def _guo_body_force(spec: LatticeSpec, f: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Guo (2002) split forcing with force shape ``(dims, ny, nx)`` (post-collision)."""
    rho, u = macroscopic(spec, f)
    c = spec.c.astype(f.dtype)
    cu = np.tensordot(c, u, axes=(1, 0))
    uf = (u * force).sum(axis=0)
    cf = np.tensordot(c, force, axes=(1, 0))
    shape = (-1,) + (1,) * (f.ndim - 1)
    return f + spec.w.reshape(shape) * (3.0 * (cf - uf) + 9.0 * cu * cf)


def _halfway_bb_row(
    spec: LatticeSpec,
    f_pre: np.ndarray,
    out: np.ndarray,
    row: int,
    u_wall: np.ndarray | float,
) -> None:
    """Pre-streaming half-way bounce back with moving-wall momentum injection.

    Reproduces the repo-validated variant of
    ``benchmarks/verified/couette_2d/run.py``::

        f[row, q] = f_pre[row, opp(q)] + 2 w_q rho (c_q . u_wall) / cs^2

    with ``rho`` the pre-collision density of the wall row (in place on *out*).
    """
    rho = f_pre[:, row, :].sum(axis=0)
    u = _broadcast_wall_velocity(u_wall, shape=(f_pre.shape[2],), dims=spec.dims, dtype=f_pre.dtype)
    c = spec.c.astype(f_pre.dtype)
    cu = np.tensordot(c, u, axes=(1, 0))
    shape = (-1, 1)
    out[:, row, :] = f_pre[spec.opposite, row, :] + (2.0 * spec.w.reshape(shape) * rho[None, :] * cu) / spec.cs2


def analytic_profile(cfg: ChannelConfig, y: np.ndarray) -> np.ndarray:
    """Exact steady profile of the configured case at absolute height ``y``."""
    lower = cfg.wall_origin
    upper = lower + cfg.height
    if cfg.case == "couette":
        return cfg.u_wall * (y - lower) / (upper - lower)
    if cfg.case == "poiseuille":
        s = y - lower
        return cfg.body_force / (2.0 * cfg.nu) * s * ((upper - lower) - s)
    raise ValueError(f"no steady analytic profile for case {cfg.case!r}")


def stokes_layer(
    cfg: ChannelConfig, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Complex Stokes-layer amplitude at height ``y`` for the oscillating wall.

    ``u(d, t) = U0 exp(-d/delta) cos(omega t - d/delta)`` with
    ``d = upper - y`` and ``delta = sqrt(2 nu / omega)``; returns
    ``(delta, A)`` with ``u = Re[A exp(i omega t)]``.
    """
    upper = cfg.wall_origin + cfg.height
    delta = float(np.sqrt(2.0 * cfg.nu / cfg.omega))
    d = upper - y
    return delta, cfg.u_wall * np.exp(-(1.0 + 1.0j) * d / delta)


def _initial_state(spec: LatticeSpec, cfg: ChannelConfig) -> np.ndarray:
    ny, nx = cfg.ny, cfg.nx
    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    if cfg.init == "analytic" and cfg.case in ("couette", "poiseuille"):
        ux[1:-1] = analytic_profile(cfg, np.arange(1, ny - 1, dtype=float))[:, None]
    return equilibrium(spec, rho, np.stack([ux, np.zeros((ny, nx))]))


def _step(
    spec: LatticeSpec,
    f: np.ndarray,
    cfg: ChannelConfig,
    force: np.ndarray,
    wall_u: float,
) -> np.ndarray:
    """One lattice step; each wall closure is applied where its formulation needs it.

    The regularised closure is a *post-streaming* node closure (LIFE
    ``propagate`` then ``applyBCs``, src/Grid.cpp:321-368), the half-way
    bounce-back control is the repo-validated *pre-streaming* link closure.
    """
    f_arrival = f  # post-streaming, pre-collision state of this step
    f = _collide(spec, f, cfg.tau)
    if cfg.case == "poiseuille":
        f = _guo_body_force(spec, f, force)
    moving = cfg.case in ("couette", "oscillating_wall")
    if cfg.wall_bc == "regularised":
        f = _stream(spec, f)
        f = apply_regularised_wall(
            spec, f, axis=0, side="high", u_wall=(wall_u, 0.0) if moving else 0.0
        )
        return apply_regularised_wall(spec, f, axis=0, side="low", u_wall=0.0)
    # Half-way bounce back reflects the *arriving* (pre-collision) populations:
    # using the post-collision state here feeds the wall node's own collision
    # back into itself and diverges (verified numerically, m18423).
    out = f.copy()
    _halfway_bb_row(spec, f_arrival, out, cfg.ny - 1, (wall_u, 0.0) if moving else 0.0)
    _halfway_bb_row(spec, f_arrival, out, 0, 0.0)
    return _stream(spec, out)


def _fit_wall_origin(y: np.ndarray, ux: np.ndarray, cfg: ChannelConfig) -> dict:
    """Least-squares parabola/cubic roots -> measured distance of the wall planes."""
    out: dict = {}
    if cfg.case == "poiseuille":
        coef = np.polyfit(y, ux, 2)
        roots = np.sort(np.roots(coef)) if abs(coef[0]) > 0 else np.array([])
        residual = float(np.max(np.abs(np.polyval(coef, y) - ux)) / max(np.max(np.abs(ux)), 1e-30))
        out["fit_residual"] = residual
        if roots.size == 2:
            out["fitted_wall_low"] = float(roots[0])
            out["fitted_wall_high"] = float(roots[1])
    elif cfg.case == "couette":
        coef = np.polyfit(y, ux, 1)
        residual = float(np.max(np.abs(np.polyval(coef, y) - ux)) / max(np.max(np.abs(ux)), 1e-30))
        out["fit_residual"] = residual
        if abs(coef[0]) > 0:
            out["fitted_wall_low"] = float(-coef[1] / coef[0])
            out["fitted_wall_high"] = float((cfg.u_wall - coef[1]) / coef[0])
    return out


def run_channel(cfg: ChannelConfig) -> dict:
    """Run the configured plane channel and return a JSON-ready record.

    Steady cases (Couette, Poiseuille) stop once
    ``max|u(n) - u(n - check_every)| / max|u| < tol`` (declared before the run).
    The oscillating-wall case runs a fixed step count and is analysed over its
    last full period by a single-frequency DFT.
    """
    spec = D2Q9
    if cfg.case not in ("couette", "poiseuille", "oscillating_wall"):
        raise ValueError(f"unknown case {cfg.case!r}")
    if cfg.wall_bc not in ("regularised", "bounce_back"):
        raise ValueError(f"unknown wall_bc {cfg.wall_bc!r}")
    if cfg.ny < 8:
        raise ValueError("ny must be >= 8")
    if cfg.case == "oscillating_wall":
        return _run_oscillating(cfg, spec)

    ny, nx = cfg.ny, cfg.nx
    y = np.arange(ny, dtype=float)
    f = _initial_state(spec, cfg)
    force = np.zeros((2, ny, nx))
    if cfg.case == "poiseuille":
        force[0, 1:-1, :] = cfg.body_force

    ux_prev = None
    converged = False
    step = 0
    delta = float("inf")
    for step in range(1, cfg.steps + 1):
        f = _step(spec, f, cfg, force, cfg.u_wall)
        if step % cfg.check_every == 0:
            _, u = macroscopic(spec, f)
            ux = u[0, 1:-1, :].mean(axis=1)
            if ux_prev is not None:
                delta = float(np.max(np.abs(ux - ux_prev)) / max(np.max(np.abs(ux)), 1e-30))
                if delta < cfg.tol:
                    converged = True
                    break
            ux_prev = ux

    rho, u = macroscopic(spec, f)
    ux = u[0, 1:-1, :].mean(axis=1)
    y_fluid = y[1:-1]
    ux_exact = analytic_profile(cfg, y_fluid)
    err = ux - ux_exact
    record = {
        "case": cfg.case,
        "wall_bc": cfg.wall_bc,
        "config": asdict(cfg),
        "nu": cfg.nu,
        "wall_origin": cfg.wall_origin,
        "height": cfg.height,
        "steps_run": step,
        "converged": converged,
        "final_rel_delta": delta,
        "mass_drift": float(np.mean(rho[1:-1]) - 1.0),
        "y": y_fluid.tolist(),
        "ux": ux.tolist(),
        "ux_exact": ux_exact.tolist(),
        "l2_rel": float(np.linalg.norm(err) / np.linalg.norm(ux_exact)),
        "linf_abs": float(np.max(np.abs(err))),
        "u_max": float(np.max(np.abs(ux_exact))),
    }
    record.update(_fit_wall_origin(y_fluid, ux, cfg))
    return record


def _run_oscillating(cfg: ChannelConfig, spec: LatticeSpec) -> dict:
    """Oscillating-wall (Stokes second problem) run with last-period DFT analysis."""
    ny, nx = cfg.ny, cfg.nx
    y = np.arange(ny, dtype=float)
    delta_layer, _ = stokes_layer(cfg, y)
    if cfg.height / delta_layer < 6.0:
        raise ValueError(
            f"channel too shallow for the Stokes layer: H/delta = {cfg.height / delta_layer:.2f} < 6"
        )
    period = int(round(2.0 * np.pi / cfg.omega))
    if period < 16:
        raise ValueError("oscillation period must be >= 16 steps")
    steps = int(cfg.steps)
    if steps < 3 * period:
        raise ValueError(f"steps={steps} must cover at least 3 periods ({3 * period})")
    f = _initial_state(spec, cfg)
    force = np.zeros((2, ny, nx))
    collect_from = steps - 2 * period + 1
    steps_seen: list[int] = []
    profiles: list[np.ndarray] = []
    for step in range(1, steps + 1):
        wall_u = cfg.u_wall * np.cos(cfg.omega * step)
        f = _step(spec, f, cfg, force, wall_u)
        if step >= collect_from:
            _, u = macroscopic(spec, f)
            steps_seen.append(step)
            profiles.append(u[0, 1:-1, :].mean(axis=1))
    t_all = np.asarray(steps_seen, dtype=float)
    stack = np.asarray(profiles)  # (2 * period, ny - 2)
    last = t_all >= steps - period + 1
    t = t_all[last]
    phase = np.exp(-1.0j * cfg.omega * t)
    amp = (2.0 / t.size) * (stack[last] * phase[:, None]).sum(axis=0)
    t_prev = t_all[~last]
    amp_prev = (2.0 / t_prev.size) * (
        stack[~last] * np.exp(-1.0j * cfg.omega * t_prev)[:, None]
    ).sum(axis=0)
    y_fluid = y[1:-1]
    _, amp_ref = stokes_layer(cfg, y_fluid)
    err = amp - amp_ref
    rel = float(np.linalg.norm(err) / np.linalg.norm(amp_ref))
    phase_err = float(np.max(np.abs(np.angle(amp / amp_ref))))
    amp_err = float(np.max(np.abs(np.abs(amp) - np.abs(amp_ref))) / np.max(np.abs(amp_ref)))
    transient = float(np.max(np.abs(amp - amp_prev)) / np.max(np.abs(amp)))
    rho, _ = macroscopic(spec, f)
    return {
        "case": cfg.case,
        "wall_bc": cfg.wall_bc,
        "config": asdict(cfg),
        "nu": cfg.nu,
        "delta_stokes": delta_layer,
        "height": cfg.height,
        "H_over_delta": cfg.height / delta_layer,
        "period_steps": period,
        "steps_run": steps,
        "mass_drift": float(np.mean(rho[1:-1]) - 1.0),
        "y": y_fluid.tolist(),
        "amp_abs": np.abs(amp).tolist(),
        "amp_phase": np.angle(amp).tolist(),
        "amp_ref_abs": np.abs(amp_ref).tolist(),
        "amp_ref_phase": np.angle(amp_ref).tolist(),
        "l2_rel_complex": rel,
        "amp_max_rel": amp_err,
        "phase_max_abs": phase_err,
        "period_to_period_change": transient,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="couette", choices=("couette", "poiseuille", "oscillating_wall"))
    parser.add_argument("--wall-bc", default="regularised", choices=("regularised", "bounce_back"))
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--u-wall", type=float, default=0.05)
    parser.add_argument("--body-force", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=2.0 * np.pi / 200.0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = ChannelConfig(
        case=args.case,
        wall_bc=args.wall_bc,
        ny=args.ny,
        nx=args.nx,
        tau=args.tau,
        u_wall=args.u_wall,
        body_force=args.body_force,
        omega=args.omega,
        steps=args.steps,
    )
    record = run_channel(cfg)
    payload = json.dumps(record, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    summary = {k: v for k, v in record.items() if k in ("case", "wall_bc", "l2_rel", "l2_rel_complex", "amp_max_rel", "phase_max_abs", "steps_run", "converged", "fit_residual", "fitted_wall_low", "fitted_wall_high", "period_to_period_change")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
