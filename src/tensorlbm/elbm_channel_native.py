"""ctypes bindings and build harness for the native ELBM D3Q27 channel kernel.

``elbm_channel_kernel.c`` is the fused OpenMP port of :mod:`tensorlbm.elbm_channel`.
It exists because the per-cell entropy solve is logarithm-heavy: as torch
elementwise passes it runs at ~0.1 Mcell/s (measured on a 128-core node) while
the fused kernel reaches O(10-100 Mcell/s), which is the difference between a
wall-resolved ELBM channel being affordable and not.

The library is built once per source hash into ``ELBM_BUILD_DIR`` (default
``~/.cache/tensorlbm_elbm``), so repeated imports are free and a source edit
forces a rebuild.  A compiler failure raises :class:`ElbmKernelBuildError`
rather than silently falling back -- a silent fallback would make a campaign
report numbers from a different code path than the one it claims to use.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "ELBM_PROFILES",
    "ElbmKernelBuildError",
    "NativeChannelResult",
    "build_kernel",
    "native_channel_run",
    "native_channel_step",
]

_HERE = Path(__file__).resolve().parent
SOURCE = _HERE / "elbm_channel_kernel.c"

#: profile block names written by the kernel, in order.
ELBM_PROFILES = ("u", "uu", "vv", "ww", "uv", "rho")


class ElbmKernelBuildError(RuntimeError):
    """Raised when the native kernel cannot be compiled."""


def _build_dir() -> Path:
    env = os.environ.get("ELBM_BUILD_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "tensorlbm_elbm"


def build_kernel(*, force: bool = False, verbose: bool = False) -> Path:
    """Compile the kernel (hash-cached) and return the shared-library path."""
    if not SOURCE.exists():  # pragma: no cover - packaging error
        raise ElbmKernelBuildError(f"kernel source missing: {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    out_dir = _build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"libelbm_channel_{digest}.so"
    if target.exists() and not force:
        return target
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:  # pragma: no cover - environment dependent
        raise ElbmKernelBuildError("no C compiler found (set CC or install gcc)")
    cmd = [cc, "-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC",
           str(SOURCE), "-o", str(target), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not target.exists():  # pragma: no cover
        raise ElbmKernelBuildError(
            "kernel build failed:\n" + " ".join(cmd) + "\n" + proc.stderr[-4000:]
        )
    if verbose:  # pragma: no cover
        print("built", target)
    return target


@lru_cache(maxsize=4)
def _library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    fp = ctypes.POINTER(ctypes.c_float)
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.elbm_channel_step.argtypes = [
        fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_double, dp,
    ]
    lib.elbm_channel_step.restype = ctypes.c_int
    lib.elbm_channel_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, dp, dp, dp, ip,
    ]
    lib.elbm_channel_run.restype = ctypes.c_int
    return lib


def lib() -> ctypes.CDLL:
    """The compiled kernel, built on first use."""
    return _library(str(build_kernel()))


def _ptr(arr: np.ndarray) -> ctypes.Array:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def native_channel_step(
    f: np.ndarray,
    alpha: np.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    force: float,
    n_iter: int = 5,
    alpha_cap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Advance one ELBM step in the native kernel.

    Args:
        f: ``(27, nz, ny, nx)`` float32 populations (torch/D3Q27 layout).
        alpha: ``(nz, ny, nx)`` float32 warm start, updated in place.
        tau: relaxation time, ``nu = (tau - 1/2)/3``.
        force: body-force acceleration (lattice units).

    Returns:
        ``(f_new, alpha, stats)`` -- new populations in the same layout.
    """
    # The kernel streams in place: buffer 1 holds the pre-collision state on
    # entry and the post-streaming state on exit, buffer 2 is scratch.
    work = np.array(f, dtype=np.float32, order="C", copy=True)
    scratch = np.empty_like(work)
    a32 = np.array(alpha, dtype=np.float32, order="C", copy=True)
    stats = np.zeros(4, dtype=np.float64)
    rc = lib().elbm_channel_step(
        _ptr(work), _ptr(scratch), _ptr(a32), nx, ny, nz, tau, force, n_iter, alpha_cap,
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:  # pragma: no cover
        raise RuntimeError(f"native elbm_channel_step failed rc={rc}")
    return work, a32, {
        "alpha_mean": float(stats[0]),
        "alpha_min": float(stats[1]),
        "alpha_max": float(stats[2]),
        "entropy_drop_max": float(stats[3]),
    }


@dataclass
class NativeChannelResult:
    """Raw output of :func:`native_channel_run`."""

    profiles: dict[str, np.ndarray]
    scalars: dict[str, float]
    history: np.ndarray

    def to_json_dict(self) -> dict[str, object]:
        return {
            "profiles": {k: v.tolist() for k, v in self.profiles.items()},
            "scalars": dict(self.scalars),
            "history": self.history.tolist(),
        }


def native_channel_run(
    *,
    nx: int,
    ny: int,
    nz: int,
    steps: int,
    warmup: int,
    sample_every: int,
    re_tau: float,
    u_tau: float,
    n_iter: int = 5,
    alpha_cap: float = 1.0,
    noise_amplitude: float = 0.06,
    seed: int = 20261110,
    log_every: int = 1000,
) -> NativeChannelResult:
    """Run one full native ELBM channel case."""
    profiles = np.zeros(6 * ny, dtype=np.float64)
    scalars = np.zeros(10, dtype=np.float64)
    history = np.zeros((100000, 8), dtype=np.float64)
    hist_count = ctypes.c_int(0)
    rc = lib().elbm_channel_run(
        nx, ny, nz, steps, warmup, sample_every, n_iter, re_tau, u_tau, alpha_cap,
        noise_amplitude, seed, log_every,
        profiles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scalars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        history.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(hist_count),
    )
    if rc != 0:  # pragma: no cover
        raise RuntimeError(f"native elbm_channel_run failed rc={rc}")
    prof = {name: profiles[k * ny:(k + 1) * ny].copy() for k, name in enumerate(ELBM_PROFILES)}
    names = (
        "samples", "min_population", "max_abs_u", "alpha_mean", "alpha_min", "alpha_max",
        "entropy_drop_max", "finite", "nu_nominal_internal", "mean_rho",
    )
    return NativeChannelResult(
        profiles=prof,
        scalars={n: float(scalars[i]) for i, n in enumerate(names)},
        history=history[: hist_count.value].copy(),
    )