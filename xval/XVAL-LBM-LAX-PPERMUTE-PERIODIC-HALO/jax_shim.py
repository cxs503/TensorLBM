"""Minimal JAX stand-in so the *verbatim* vivsim helper can run on CPU/numpy.

The XVAL rule is "no hand-copying": the reference arm must execute the
upstream source, not a retyped paraphrase.  Upstream ``vivsim/multidevice.py``
(39 lines, MIT, sha256 recorded in ``ref/PROVENANCE.md``) only touches three
JAX surfaces, which are provided here:

* ``jax.Array``            -> ``numpy.ndarray``
* ``jax.numpy``            -> ``numpy`` (``.asarray``/``.concatenate`` used by
                             ``vivsim/lbm/basic.py``)
* ``jax.lax.ppermute(x, axis_name, perm)`` with JAX's documented semantics
  ``result[dst] = x[src]`` for every ``(src, dst)`` in ``perm``.

What JAX would do across devices is supplied by :class:`Mesh`: the SPMD
program is executed once per rank and every ``ppermute`` request is replayed
against the *shard state after the previous synchronous op*, which is what
SPMD execution means (all ranks perform op ``k`` before any rank performs
op ``k+1``).

JAX is not installed on this host (no loongarch64 ``jaxlib`` wheel — see
REPORT.md §5 friction log), so the emulation is validated two ways:

1. the ``ppermute`` semantics are the documented ones and the ring pair
   tables in the upstream file are exact inverses of each other;
2. the GOT arm performs the *real* cross-process data movement under gloo
   with the same ring and reproduces the ground truth, i.e. the ring
   semantics assumed here is the one a real distributed runtime implements.
"""

from __future__ import annotations

import sys
import types

import numpy as np

__all__ = ["Mesh", "install", "ppermute", "Pending"]


class Pending:
    """A ``ppermute`` result that has not been materialised yet."""

    __slots__ = ("mesh", "src_rank", "src_idx")

    def __init__(self, mesh: "Mesh", src_rank: int, src_idx):
        self.mesh = mesh
        self.src_rank = src_rank
        self.src_idx = src_idx


class _Setter:
    __slots__ = ("mesh", "rank", "dest_idx")

    def __init__(self, mesh: "Mesh", rank: int, dest_idx):
        self.mesh = mesh
        self.rank = rank
        self.dest_idx = dest_idx

    def set(self, value):
        if not isinstance(value, Pending):
            raise TypeError(
                "jax_shim only supports .at[idx].set(ppermute_result) — got "
                f"{type(value).__name__}"
            )
        if value.mesh is not self.mesh:
            raise RuntimeError("ppermute against a foreign Mesh")
        self.mesh.ops[self.rank].append(
            {
                "dest_rank": self.rank,
                "dest_idx": self.dest_idx,
                "src_rank": value.src_rank,
                "src_idx": value.src_idx,
            }
        )
        # Functional JAX: .set() returns a *new* array.  The replay below
        # rebuilds the real contents, so the recorder proxy only needs to
        # stay usable for the next statement in the same function body.
        return _Proxy(self.mesh, self.rank)


class _At:
    __slots__ = ("proxy",)

    def __init__(self, proxy):
        self.proxy = proxy

    def __getitem__(self, idx):
        return _Setter(self.proxy.mesh, self.proxy.rank, idx)


class _Proxy:
    """Array stand-in that records index expressions and ``.set`` writes."""

    __slots__ = ("mesh", "rank", "idx")

    def __init__(self, mesh: "Mesh", rank: int, idx=None):
        self.mesh = mesh
        self.rank = rank
        self.idx = idx

    def __getitem__(self, idx):
        return _Proxy(self.mesh, self.rank, idx)

    @property
    def at(self):
        return _At(self)


class Mesh:
    """A one-dimensional ring of device shards emulating JAX SPMD."""

    def __init__(self, shards):
        self.shards = [np.array(s, dtype=np.float64, copy=True) for s in shards]
        self.n = len(self.shards)
        self.ops: list[list[dict]] = []

    def exchange(self, fn, /, **kwargs):
        """Run ``fn`` (an upstream ``jax``-style function) SPMD across shards.

        Returns the post-exchange shards (copies).
        """
        self.ops = [[] for _ in range(self.n)]
        for r in range(self.n):
            fn(_Proxy(self, r), **kwargs)
        n_ops = max((len(o) for o in self.ops), default=0)
        for k in range(n_ops):
            # Every rank performs op k against the state left by op k-1:
            # true SPMD ordering.
            staged = [
                (r, self.ops[r][k]) for r in range(self.n) if len(self.ops[r]) > k
            ]
            values = {
                r: np.array(
                    self.shards[op["src_rank"]][op["src_idx"]], dtype=self.shards[r].dtype, copy=True
                )
                for r, op in staged
            }
            for r, op in staged:
                self.shards[r][op["dest_idx"]] = values[r]
        return [s.copy() for s in self.shards]


_CURRENT: Mesh | None = None


def ppermute(x, axis_name, perm):
    """``jax.lax.ppermute``: ``result[dst] = x[src]`` for ``(src, dst) in perm``."""
    if not isinstance(x, _Proxy):
        raise TypeError(f"ppermute expects a shim array, got {type(x).__name__}")
    mesh = x.mesh
    perm = [(int(s), int(d)) for s, d in perm]
    if sorted(s for s, _ in perm) != list(range(mesh.n)) or sorted(
        d for _, d in perm
    ) != list(range(mesh.n)):
        raise ValueError(f"perm {perm} is not a permutation of range({mesh.n})")
    src_of = {dst: src for src, dst in perm}
    return Pending(mesh, src_of[x.rank], x.idx)


def install() -> types.ModuleType:
    """Insert the ``jax`` / ``jax.lax`` / ``jax.numpy`` shims into sys.modules."""
    if "jax" in sys.modules and getattr(sys.modules["jax"], "_xval_shim", False):
        return sys.modules["jax"]

    jax = types.ModuleType("jax")
    jax._xval_shim = True
    jax.Array = np.ndarray

    lax = types.ModuleType("jax.lax")
    lax.ppermute = ppermute
    lax._xval_shim = True

    jnp = types.ModuleType("jax.numpy")
    for name in dir(np):
        if not name.startswith("_"):
            setattr(jnp, name, getattr(np, name))
    jnp._xval_shim = True

    jax.lax = lax
    jax.numpy = jnp
    sys.modules["jax"] = jax
    sys.modules["jax.lax"] = lax
    sys.modules["jax.numpy"] = jnp
    return jax