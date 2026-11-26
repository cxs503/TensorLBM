"""Arms for XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO.

GT   : single-domain periodic collide+stream (no decomposition).
REF-*: the vendored vivsim helper (verbatim / two derived variants) driven
       through :mod:`jax_shim` SPMD emulation, 1-D device ring, D2Q9.
GOT  : TensorLBM ``DistributedTritonFusedSolver3D`` — real constructor, real
       ``_start_halo_exchange`` / ``_finalize_halo``, CPU + gloo, D3Q19 slabs.
CONTROL-LANE : GOT with the crossing lane tables swapped (defect injection).

The streaming kernel of the GOT arm is a CPU pull-stream (Triton kernels
cannot run on this CPU-only host).  It is *validated* by the world=1 case,
where the real single-rank halo branch of ``_start_halo_exchange`` is the only
thing that can go wrong: at world=1 GOT must equal GT bit for bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

import jax_shim  # noqa: E402

UPSTREAM = {
    "REF-A": "vivsim.multidevice",              # verbatim (MIT, sha256 pinned)
    "REF-B": "vivsim.multidevice_plane_fixed",  # derived: read plane fixed
    "REF-C": "vivsim.multidevice_fixed",        # derived: plane + pair tables
    "REF-CS": "vivsim.multidevice_fixed",       # REF-C + in-plane (diagonal) compensation
}


# ---------------------------------------------------------------------------
# Upstream module loading (verbatim source, shimmed)
# ---------------------------------------------------------------------------


def load_upstream(variant: str):
    jax_shim.install()
    import importlib

    return importlib.import_module(UPSTREAM[variant])


def d2q9_tables():
    """Direction / velocity tables exactly as vendored from vivsim."""
    jax_shim.install()
    import vivsim.lbm.basic as basic

    return (
        np.asarray(basic.VELOCITIES, dtype=np.float64),
        np.asarray(basic.WEIGHTS, dtype=np.float64),
        {k: np.asarray(getattr(basic, k)) for k in
         ("RIGHT_DIRS", "LEFT_DIRS", "UP_DIRS", "DOWN_DIRS")},
    )


# ---------------------------------------------------------------------------
# Shared physics (one function object used by every arm -> cannot bias)
# ---------------------------------------------------------------------------


def collide(f, tau, C, W):
    """BGK collision on raw populations ``f`` of shape (Q, ...)."""
    rho = f.sum(axis=0)
    u = np.tensordot(C.T, f, axes=(1, 0)) / rho          # (D, *grid)
    cu = np.tensordot(C, u, axes=(1, 0))                 # (Q, *grid)
    u2 = (u * u).sum(axis=0)
    feq = rho * W.reshape((-1,) + (1,) * (f.ndim - 1)) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
    )
    return f - (f - feq) / tau


def feq_field(rho, u, C, W):
    cu = np.tensordot(C, u, axes=(1, 0))
    u2 = (u * u).sum(axis=0)
    return rho * W.reshape((-1,) + (1,) * (rho.ndim)) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2
    )


# ---------------------------------------------------------------------------
# 2-D field (vivsim arm)
# ---------------------------------------------------------------------------


def make_field_2d(nx: int, ny: int, kind: str, seed: int):
    C, W, _ = d2q9_tables()
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    rho = 1.0 + 0.05 * np.sin(2 * np.pi * xx / nx + 0.3) * np.cos(2 * np.pi * yy / ny)
    rho = rho.T
    if kind == "asym":
        ux = 0.06 * np.sin(2 * np.pi * yy / ny) + 0.02 * np.cos(4 * np.pi * xx / nx)
        uy = 0.05 * np.cos(2 * np.pi * xx / nx + 1.1) + 0.03 * np.sin(2 * np.pi * yy / ny)
        u = np.stack([ux.T, uy.T])
    elif kind == "sym":
        u = np.zeros((2, nx, ny))
    else:
        raise ValueError(kind)
    f = feq_field(rho, u, C, W)
    rng = np.random.default_rng(seed)
    f = f + 1e-3 * rng.standard_normal(f.shape)
    return f


def stream_global_2d(f, up_basic):
    """Ground truth: upstream vivsim ``streaming()`` on the WHOLE domain."""
    jax_shim.install()
    return np.asarray(up_basic.streaming(f), dtype=np.float64)


def _split_2d(f, axis: str, n: int):
    if axis == "x":
        return [f[:, i * f.shape[1] // n:(i + 1) * f.shape[1] // n, :].copy() for i in range(n)]
    if axis == "y":
        return [f[:, :, i * f.shape[2] // n:(i + 1) * f.shape[2] // n].copy() for i in range(n)]
    raise ValueError(axis)


def _join_2d(shards, axis: str):
    if axis == "x":
        return np.concatenate(shards, axis=1)
    return np.concatenate(shards, axis=2)


# Face (lane set, boundary index) pairs touched by the upstream helper, per
# decomposition axis.  Identical for every convention: only the halo *content*
# written into them varies, so the arms differ in exactly one variable.
_FACES = {
    "y": ((np.asarray([2, 5, 6]), 0), (np.asarray([4, 7, 8]), -1)),   # UP / DOWN
    "x": ((np.asarray([1, 5, 8]), 0), (np.asarray([3, 7, 6]), -1)),   # RIGHT / LEFT
}


def _face_slices(axis: str, dirs, pos):
    """Exactly the index expressions the upstream helper writes into."""
    if axis == "y":
        return (dirs, slice(None), pos), (dirs, slice(None), pos)
    return (dirs, pos, slice(None)), (dirs, pos, slice(None))


def run_ref_arm(f0, axis: str, n_ranks: int, n_steps: int, tau: float, variant: str,
                inplane_comp: bool = False):
    """REF-* arm: split -> (collide, exchange, stream locally, halo fix-up)^N.

    ``inplane_comp`` adds the one-cell in-plane shift of the received crossing
    lanes (the diagonal correction).  Upstream has no such term; it is the
    arm that isolates whether the OSS convention is missing a ghost layer.
    """
    jax_shim.install()
    import vivsim.lbm.basic as basic

    C, W, _ = d2q9_tables()
    mod = load_upstream(variant)

    shards = _split_2d(f0, axis, n_ranks)
    mesh = jax_shim.Mesh(shards)
    for _ in range(n_steps):
        # collision is elementwise -> local shards are exactly the global result
        pre = [collide(s, tau, C, W) for s in mesh.shards]
        mesh.shards = [p.copy() for p in pre]
        exchanged = mesh.exchange(
            mod.stream_cross_devices, dir=axis, device_axis="dev", n_devices=n_ranks
        )
        # stream the *pre-stream* field: the exchange only says which values
        # belong in the boundary cells, it must not feed the shift itself
        streamed = [np.asarray(basic.streaming(p), dtype=np.float64) for p in pre]
        for r in range(n_ranks):
            for dirs, pos in _FACES[axis]:
                dst, sub = _face_slices(axis, dirs, pos)
                src = np.array(exchanged[r][sub], dtype=np.float64, copy=True)
                if inplane_comp:
                    # the orthogonal lattice component of each lane: the
                    # received boundary line has to be shifted by that amount
                    ip = C[dirs, 0] if axis == "y" else C[dirs, 1]
                    src = np.stack(
                        [np.roll(v, int(s)) if s else v for v, s in zip(src, ip)]
                    )
                streamed[r][dst] = src
        mesh.shards = streamed
    return _join_2d(mesh.shards, axis)


# ---------------------------------------------------------------------------
# 3-D D3Q19 field (GOT arm)
# ---------------------------------------------------------------------------


def d3q19_tables():
    import torch

    c = np.array(
        [
            [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
            [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
            [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        ],
        dtype=np.int64,
    )
    w = np.array(
        [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float64
    )
    return c, w


def make_field_3d(nz: int, ny: int, nx: int, seed: int, kind: str = "tgv"):
    C, W = d3q19_tables()
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    u = np.zeros((3, nz, ny, nx))
    if kind == "tgv":
        # Taylor-Green vortex, unit box, amplitude 0.05 (Ma ~ 0.087)
        u[0] = 0.05 * np.sin(2 * np.pi * x / nx) * np.cos(2 * np.pi * y / ny) * np.cos(2 * np.pi * z / nz)
        u[1] = -0.05 * np.cos(2 * np.pi * x / nx) * np.sin(2 * np.pi * y / ny) * np.cos(2 * np.pi * z / nz)
    elif kind == "rand":
        rng = np.random.default_rng(seed)
        u = 0.05 * rng.standard_normal((3, nz, ny, nx))
    else:
        raise ValueError(kind)
    rho = np.ones((nz, ny, nx))
    f = feq_field(rho, u, C, W)
    rng = np.random.default_rng(1000 + seed)
    f = f + 1e-3 * rng.standard_normal(f.shape)
    return f


def _collide_t(f, tau, C_t, W_t):
    """BGK collision (torch, float64) — same formula as numpy ``collide``."""
    import torch

    rho = f.sum(dim=0)
    m = (C_t.T.reshape(3, 19) @ f.reshape(19, -1)).reshape(3, *f.shape[1:])
    u = m / rho
    cu = torch.einsum("qa,a...->q...", C_t, u)
    u2 = (u * u).sum(dim=0)
    feq = rho * W_t.reshape(-1, *([1] * (f.dim() - 1))) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    return f - (f - feq) / tau


def _tables_t():
    import torch

    c, w = d3q19_tables()
    return torch.as_tensor(c, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)


def stream_global3d(f, C_t):
    """GT streaming on the whole periodic domain: out[q,z] = f[q, z - c_q]."""
    import torch

    out = torch.empty_like(f)
    for q in range(19):
        # C columns are (x, y, z) but the spatial dims of ``f`` are (z, y, x)
        cx, cy, cz = int(C_t[q, 0]), int(C_t[q, 1]), int(C_t[q, 2])
        g = f[q]
        if cz:
            g = torch.roll(g, cz, dims=0)
        if cy:
            g = torch.roll(g, cy, dims=1)
        if cx:
            g = torch.roll(g, cx, dims=2)
        out[q] = g
    return out


def stream_local3d(f, C_t, nz_local):
    """Pull stream inside a ghost-plane buffer (no wrap: ghosts carry the halo)."""
    import torch

    out = torch.empty_like(f)
    ny, nx = f.shape[2], f.shape[3]
    for q in range(19):
        cx, cy, cz = int(C_t[q, 0]), int(C_t[q, 1]), int(C_t[q, 2])
        g = f[q, 1 - cz: 1 - cz + nz_local]          # z: ghosts carry the halo
        if cy:
            g = torch.roll(g, cy, dims=1)            # y: full plane, wraps in-plane
        if cx:
            g = torch.roll(g, cx, dims=2)            # x: ditto
        out[q, 1:nz_local + 1] = g
    return out


def make_cpu_solver(nz_global, ny, nx, tau=0.6, halo_dtype=None, lane_swap=False):
    """Real TensorLBM solver on CPU; only the CUDA gate + GPU table factory stub."""
    import torch
    import tensorlbm.triton_fused_distributed as tfd

    tfd._single_is_available = lambda: True          # host has no CUDA
    tfd.make_lattice_tensors = lambda dev: None      # triton-only GPU cache
    exact = halo_dtype is not None and halo_dtype.itemsize == 8
    wire = torch.float32 if (halo_dtype is None or exact) else halo_dtype
    # The production constructor whitelists 2/4-byte wires only
    # (``halo_dtype must be fp32 or fp16``); a float64 buffer therefore cannot
    # ask for an exact wire.  We pass the gate with fp32 and, for the exact
    # float64 arm, widen the *wire* after construction — the gather/scatter
    # path already supports it (``plane.dtype == self._halo_dtype`` branch).
    s = tfd.DistributedTritonFusedSolver3D(
        nz_global=nz_global, ny=ny, nx=nx, tau=tau, device="cpu", halo_dtype=wire,
    )
    if exact:
        s._halo_dtype = halo_dtype
        if s._world > 1:
            s._alloc_halo_staging(halo_dtype)
    if lane_swap:  # CONTROL-LANE: send the -z lanes to the +z neighbour
        s._cross_up, s._cross_dn = s._cross_dn, s._cross_up
        s._idx_up = torch.tensor(s._cross_up, dtype=torch.int64)
        s._idx_dn = torch.tensor(s._cross_dn, dtype=torch.int64)
        if s._send_left is not None:
            s._alloc_halo_staging(s._halo_dtype)
    return s


def periodic_ghost_fill(buf, nz_local):
    """Ghost planes for a *single* rank: periodic wrap of the owned planes.

    The class's world==1 branch of ``_start_halo_exchange`` copies the nearest
    owned plane instead; with a buffer-local wrap kernel that is not the
    periodic halo (see REPORT.md §5/§6).
    """
    buf[:, 0].copy_(buf[:, nz_local])
    buf[:, -1].copy_(buf[:, 1])
    return buf


def run_got_steps(s, f_global_np, n_steps, stream_fn, w1_periodic_fill=False):
    """Drive the REAL halo methods through ``n_steps`` of collide+stream.

    ``stream_fn(buf)`` replaces only the Triton kernel; the exchange is the
    production code path (start async -> wait/land next step).
    """
    import torch

    C_t, W_t = _tables_t()
    f0 = torch.as_tensor(f_global_np, dtype=torch.float64)
    nzl = s.nz_local
    f = s.from_global(f0)                       # real method
    if w1_periodic_fill:
        periodic_ghost_fill(f, nzl)
    else:
        s._halo_handles = s._start_halo_exchange(f)  # real method (first halo)
    for _ in range(n_steps):
        s._finalize_halo()                       # real method: wait + land
        if w1_periodic_fill:
            periodic_ghost_fill(f, nzl)
        new = stream_fn(_collide_t(f, s.tau, C_t, W_t))
        s._halo_handles = s._start_halo_exchange(new)
        f = new
    s._finalize_halo()
    return f

def kinetic_energy(f, C_t):
    """Mean kinetic energy of a D3Q19 population field (torch, float64)."""
    rho = f.sum(dim=0)
    m = (C_t.T.reshape(3, 19) @ f.reshape(19, -1)).reshape(3, *f.shape[1:])
    u = m / rho
    return 0.5 * float((rho * (u * u).sum(dim=0)).mean())
