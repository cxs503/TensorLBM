"""OSS reference: lbmpy's fused stream+collide kernels, driven from Python.

This is the *external* side of the XVAL anchor
``XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR``: instead of re-deriving what
``lssfau/lbmpy`` does, the real library is imported (``lbmpy`` 2.0 +
``pystencils`` 2.0 are installed in this environment) and its
``kernel_type`` selector is exercised directly:

* ``creationfunctions.LBMConfig(kernel_type=...)`` -- the selector under test
  (``src/lbmpy/creationfunctions.py`` lines 298-440 and 619-633: the value is
  mapped to a streaming pattern/timestep pair, or dispatched to
  ``PdfFieldAccessor`` variants, else ``ValueError``).
* ``creationfunctions.create_lb_ast`` -> ``pystencils.codegen.Kernel.compile``
  -- the actual generated C kernel, JIT-compiled and executed on numpy fields.

Field layout: pystencils numpy fields are ``(nx, ny, 9)`` C-ordered, whereas
:mod:`tensorlbm` uses ``(9, ny, nx)``; :func:`to_lbmpy` / :func:`from_lbmpy`
convert between the two.

Everything here is read from the installed library -- no constants are copied
by hand.  If ``lbmpy``/``pystencils`` are not importable the caller must treat
the cross-validation as ``NOT_RUN`` (see ``bench_fused_kernel_selector.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import functools

import numpy as np

__all__ = [
    "LBMPY_AVAILABLE",
    "LbmpyKernel",
    "build_lbmpy_kernel",
    "from_lbmpy",
    "kernel_type_options",
    "to_lbmpy",
]

try:  # pragma: no cover - exercised by the runtime environment
    import lbmpy  # noqa: F401
    import pystencils as ps
    from lbmpy import LBMConfig, LBMOptimisation, Method, Stencil
    from lbmpy.creationfunctions import create_lb_ast

    LBMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only when the OSS stack is absent
    LBMPY_AVAILABLE = False


def kernel_type_options() -> dict[str, str]:
    """Return ``kernel_type -> lbmpy streaming_pattern`` for the supported values.

    Read from ``LBMConfig`` itself rather than restating the mapping by hand.
    """
    if not LBMPY_AVAILABLE:  # pragma: no cover
        raise RuntimeError("lbmpy is not installed")
    cfg = LBMConfig(stencil=Stencil.D2Q9, method=Method.SRT, relaxation_rate=1.0)
    out = {}
    for name in ("default_stream_collide", "collide_only", "stream_pull_only"):
        patch = {"kernel_type": name}
        resolved = cfg.copy(**patch) if hasattr(cfg, "copy") else cfg
        out[name] = str(getattr(resolved, "streaming_pattern", ""))
    return out


@dataclass
class LbmpyKernel:
    """A compiled lbmpy kernel plus the field names it takes."""

    kernel_type: str
    compiled: Any
    field_names: tuple[str, ...]

    def __call__(self, **fields: np.ndarray) -> None:
        self.compiled(**fields)


def direction_permutation() -> tuple[list[int], list[int]]:
    """Return the D2Q9 direction re-ordering between lbmpy and tensorlbm.

    The two libraries order the nine velocities differently (lbmpy D2Q9:
    ``(0,0),(0,1),(0,-1),(-1,0),(1,0),...``; tensorlbm ``d2q9.C``:
    ``(0,0),(1,0),(0,1),(-1,0),(0,-1),...``).  The mapping is derived from the
    two direction tables at runtime -- no hand-written permutation that could
    silently rot.  Returns ``(perm, inv)`` with ``lbmpy_slot[k] == tl[perm[k]]``.
    """
    from lbmpy.stencils import LBStencil
    from tensorlbm.d2q9 import C as _C_TL

    lb = [tuple(int(v) for v in d) for d in LBStencil("D2Q9")]
    tl = [tuple(int(v) for v in row) for row in _C_TL.tolist()]
    perm = [tl.index(d) for d in lb]
    inv = [0] * len(perm)
    for lb_idx, tl_idx in enumerate(perm):
        inv[tl_idx] = lb_idx
    return perm, inv


def direction_weights() -> np.ndarray:
    """D2Q9 weights in lbmpy direction order, read from lbmpy's own weights."""
    from lbmpy.stencils import LBStencil
    from lbmpy.maxwellian_equilibrium import get_weights

    return np.asarray(get_weights(LBStencil("D2Q9")), dtype=np.float64)


def to_lbmpy(f_np: np.ndarray) -> np.ndarray:
    """``(9, ny, nx)`` (tensorlbm) -> ``(nx, ny, 9)`` (pystencils C order)."""
    perm, _ = direction_permutation()
    arr = np.transpose(np.asarray(f_np, dtype=np.float64), (2, 1, 0))
    return np.ascontiguousarray(arr[..., perm])


def from_lbmpy(arr: np.ndarray) -> np.ndarray:
    """``(nx, ny, 9)`` (pystencils C order) -> ``(9, ny, nx)`` (tensorlbm)."""
    _, inv = direction_permutation()
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return np.ascontiguousarray(np.transpose(arr[..., inv], (2, 1, 0)))


def _feq_np(rho: float, u: np.ndarray) -> np.ndarray:
    """Second-order Maxwellian on the D2Q9 stencil in lbmpy direction order.

    Used only by the storage-convention probe (:func:`is_zero_centered`); the
    reference physics itself is always produced by lbmpy's compiled kernel.
    """
    from lbmpy.stencils import LBStencil

    dirs = np.asarray(list(LBStencil("D2Q9")), dtype=np.float64)
    w = direction_weights()
    cu = dirs @ np.asarray(u, dtype=np.float64)
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * float(np.dot(u, u)))


@functools.lru_cache(maxsize=1)
def is_zero_centered() -> bool:
    """Probe whether the generated kernels read/write ``f - w`` (delta) storage.

    The lbmpy 2.0 / pystencils 2.0 stack emits the collision in *deviation*
    form: the generated C code computes ``delta_rho`` as the plain sum of the
    nine entries and rebuilds ``rho = 1 + delta_rho`` (verified by reading
    ``Kernel.get_c_code()``), i.e. the array holds ``f_i - w_i``.  That is not
    part of lbmpy's public API contract, so the flag is *measured*: a
    collision-only kernel with ``omega = 1`` maps its input onto the
    equilibrium, and the two readings differ once the density departs from 1,

    * plain reading: ``sum(x) = rho``            -> ``out = feq(rho, j/rho)``
    * delta reading: ``sum(x) = rho - 1``        -> ``out = feq(rho, j/rho) - w``

    Feeding ``x = w + 0.5 (e_{(1,0)} - e_{(-1,0)})`` (``sum = 1``, ``j = (1,0)``)
    makes the two predictions differ by O(10%), so the probe is unambiguous.
    """
    w = direction_weights()
    x = np.zeros((4, 4, 9), dtype=np.float64)
    x[:] = w
    from lbmpy.stencils import LBStencil

    idx = {tuple(d): i for i, d in enumerate(LBStencil("D2Q9"))}
    x[..., idx[(1, 0)]] += 0.5
    x[..., idx[(-1, 0)]] -= 0.5
    out = to_lbmpy(
        run_lbmpy_steps(from_lbmpy(x), tau=1.0, n_steps=1,
                        kernel_type="collide_only", apply_centering=False)
    )[1, 1]

    plain = _feq_np(1.0, np.array([1.0, 0.0]))
    delta = _feq_np(2.0, np.array([0.5, 0.0])) - w
    return float(np.abs(out - delta).max()) < float(np.abs(out - plain).max())


def build_lbmpy_kernel(tau: float, kernel_type: str = "default_stream_collide",
                       compressible: bool = True) -> LbmpyKernel:
    """Build and JIT-compile an lbmpy D2Q9 SRT kernel of the requested granularity.

    Args:
        tau: BGK relaxation time; lbmpy's ``relaxation_rate`` is ``1/tau``.
        kernel_type: forwarded to ``LBMConfig`` unchanged, so an invalid value
            raises lbmpy's own ``ValueError``.
        compressible: lbmpy's ``compressible`` flag; ``True`` gives the
            second-order equilibrium ``w rho (1 + 3cu + 4.5cu² - 1.5u²)``,
            i.e. the same equilibrium as :func:`tensorlbm.d2q9.equilibrium`.
    """
    if not LBMPY_AVAILABLE:  # pragma: no cover
        raise RuntimeError("lbmpy/pystencils are not installed")

    f = ps.fields("f(9): double[2D]")
    fields: dict[str, Any] = {"f": f}
    if kernel_type == "default_stream_collide":
        fields["tmp"] = ps.fields("tmp(9): double[2D]")
    cfg = LBMConfig(
        stencil=Stencil.D2Q9,
        method=Method.SRT,
        relaxation_rate=1.0 / tau,
        compressible=compressible,
        kernel_type=kernel_type,
    )
    opt = LBMOptimisation(
        symbolic_field=fields["f"],
        symbolic_temporary_field=fields.get("tmp"),
    )
    kernel = create_lb_ast(lbm_config=cfg, lbm_optimisation=opt)
    compiled = kernel.compile()
    names = tuple(fld.name for fld in fields.values())
    return LbmpyKernel(kernel_type=kernel_type, compiled=compiled, field_names=names)


HALO = 1  # D2Q9 pull accessor needs one ghost layer on each side


def _pad_periodic(arr: np.ndarray) -> np.ndarray:
    """Embed a ``(nx, ny, 9)`` field into a ``(nx+2h, ny+2h, 9)`` periodic box.

    lbmpy's generated kernel iterates over the *interior* only (pystencils
    leaves boundary handling to the caller/``PystencilsSimulation``); the ghost
    layer is therefore filled with the periodic images of the opposite side,
    which is exactly the BC that :func:`tensorlbm.solver.stream` hard-codes
    with its ``% n`` index wrap.
    """
    nx, ny, q = arr.shape
    big = np.zeros((nx + 2 * HALO, ny + 2 * HALO, q), dtype=np.float64)
    big[HALO:-HALO, HALO:-HALO] = arr
    return big


def _refresh_periodic_halo(big: np.ndarray) -> None:
    """Re-impose the periodic ghost layer on a padded buffer (in place)."""
    big[0, :, :] = big[-2, :, :]
    big[-1, :, :] = big[1, :, :]
    big[:, 0, :] = big[:, -2, :]
    big[:, -1, :] = big[:, 1, :]


def run_lbmpy_steps(f_np: np.ndarray, tau: float, n_steps: int,
                    kernel_type: str = "default_stream_collide",
                    apply_centering: bool = True) -> np.ndarray:
    """Run ``n_steps`` lbmpy steps of the given kernel type from ``f_np``.

    ``f_np`` is ``(9, ny, nx)`` (tensorlbm layout); the return value uses the
    same layout.  The domain is periodic: the ghost layer of the source buffer
    is refreshed from the opposite side before each kernel call.

    For ``default_stream_collide`` the two fields (``f`` and ``tmp``) are
    swapped every step, exactly like lbmpy's own lattice-model loop; for
    ``collide_only`` the single field is updated in place and no streaming is
    performed.
    """
    ker = build_lbmpy_kernel(tau, kernel_type=kernel_type)
    w = direction_weights()
    centred = apply_centering and is_zero_centered()
    big = _pad_periodic(to_lbmpy(f_np))
    if centred:
        big -= w  # lbmpy stores f - w; the halo is zero-centred too
    buf = {"f": big}
    if "tmp" in ker.field_names:
        buf["tmp"] = np.zeros_like(big)
    for _ in range(n_steps):
        _refresh_periodic_halo(buf["f"])
        ker(**buf)
        if "tmp" in buf:
            buf["f"], buf["tmp"] = buf["tmp"], buf["f"]
    interior = buf["f"][HALO:-HALO, HALO:-HALO]
    if centred:
        interior = interior + w
    return from_lbmpy(interior)


def lbmpy_collide_only(f_np: np.ndarray, tau: float) -> np.ndarray:
    """One lbmpy ``collide_only`` step (no streaming), tensorlbm layout."""
    return run_lbmpy_steps(f_np, tau, 1, kernel_type="collide_only")