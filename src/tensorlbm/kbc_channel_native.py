"""ctypes bindings and build harness for the three-segment KBC channel kernel.

``kbc_channel_kernel.c`` is the fused OpenMP port of
:mod:`tensorlbm.kbc_collision` (collision) on top of the channel scaffolding of
``elbm_channel_kernel.c`` (streaming, bounce-back, Guo forcing, statistics).
The per-cell entropy-condition solve needs ~3-6 entropy evaluations, each one
27 logarithms; as NumPy elementwise passes that runs at ~0.1 Mcell/s while the
fused kernel reaches O(10-100 Mcell/s), which is what makes a 1.4 Mcell
``Re_tau = 590`` channel affordable.

The shared object is built once per source hash into ``KBC_BUILD_DIR``
(default ``~/.cache/tensorlbm_kbc``), so repeated imports are free and a source
edit forces a rebuild.  A compiler failure raises
:class:`KbcKernelBuildError` instead of silently falling back to a slower path:
a silent fallback would make a report quote numbers from a different code path
than the one it claims to use.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "KBC_PROFILES",
    "KBC_SCALARS",
    "KbcKernelBuildError",
    "NativeKbcResult",
    "build_kernel",
    "native_channel_run",
    "native_channel_step_debug",
]

_HERE = Path(__file__).resolve().parent
SOURCE = _HERE / "kbc_channel_kernel.c"

#: statistics blocks written by the kernel, in order (profiles index * ny).
KBC_PROFILES = ("u", "uu", "vv", "ww", "uv", "rho", "pi_xy", "stress")

#: scalar slots of the driver, in order.
KBC_SCALARS = (
    "samples",
    "min_population",
    "max_abs_u",
    "alpha_mean",
    "alpha_min",
    "alpha_max",
    "dh_global_worst",
    "finite",
    "nu_nominal",
    "mean_rho",
    "gamma_mean",
    "gamma_min",
    "gamma_limited_fraction",
    "entropy_skipped_fraction",
    "ghost_fraction",
    "dh_cell_worst",
    "newton_iterations",
    "wall_stress_mean",
)


class KbcKernelBuildError(RuntimeError):
    """Raised when the native kernel cannot be compiled."""


def _build_dir() -> Path:
    env = os.environ.get("KBC_BUILD_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "tensorlbm_kbc"


def build_kernel(*, force: bool = False, verbose: bool = False) -> Path:
    """Compile the kernel (hash-cached) and return the shared-library path."""
    if not SOURCE.exists():  # pragma: no cover - packaging error
        raise KbcKernelBuildError(f"kernel source missing: {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    out_dir = _build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"libkbc_channel_{digest}.so"
    if target.exists() and not force:
        return target
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:  # pragma: no cover - environment dependent
        raise KbcKernelBuildError("no C compiler found (set CC or install gcc)")
    cmd = [cc, "-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC",
           str(SOURCE), "-o", str(target), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not target.exists():  # pragma: no cover
        raise KbcKernelBuildError(
            "kernel build failed:\n" + " ".join(cmd) + "\n" + proc.stderr[-4000:]
        )
    if verbose:  # pragma: no cover
        print("built", target)
    return target


@lru_cache(maxsize=4)
def _library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    dp = ctypes.POINTER(ctypes.c_double)
    fp = ctypes.POINTER(ctypes.c_float)
    dp = dp  # alias, keeps the intent readable below
    lib.kbc_debug_cell.argtypes = [
        fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, dp,
    ]
    lib.kbc_debug_cell.restype = ctypes.c_int
    lib.kbc_channel_step.argtypes = [
        fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int, dp,
    ]
    lib.kbc_channel_step.restype = ctypes.c_int
    lib.kbc_channel_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, dp, dp, dp, ctypes.POINTER(ctypes.c_int),
    ]
    lib.kbc_channel_run.restype = ctypes.c_int
    return lib


def _lib() -> ctypes.CDLL:
    return _library(str(build_kernel()))


def native_channel_step_debug(
    f: np.ndarray,
    *,
    tau: float,
    tau_bulk: float | None = None,
    omega_g: float = 1.0,
    alpha_cap: float = 2.0,
    alpha_min: float = 1.0,
    n_iter: int = 24,
    trigger: float = 1e-9,
    enforce_entropy: bool = True,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cell: tuple[int, int, int] = (0, 1, 0),
) -> dict:
    """Run the kernel's collision on one cell of a ``(Q, nz, ny, nx)`` field.

    Returns the kernel diagnostics and the post-collision populations, so the
    NumPy reference (:func:`tensorlbm.kbc_collision.collide_kbc`) can be pinned
    to the kernel population by population.
    """
    assert f.ndim == 4 and f.shape[0] == 27, f.shape
    f = np.ascontiguousarray(f, dtype=np.float32)
    _, nz, ny, nx = f.shape
    zi, yi, xi = cell
    out = np.zeros(13 + 2 * 27, dtype=np.float64)
    lib = _lib()
    lib.kbc_debug_cell(
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nx, ny, nz, zi, yi, xi,
        float(tau), float(tau if tau_bulk is None else tau_bulk), float(omega_g),
        float(alpha_cap), float(alpha_min), int(n_iter), float(trigger),
        int(bool(enforce_entropy)),
        float(force[0]), float(force[1]), float(force[2]),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return {
        "rho": out[0],
        "u": out[1:4].copy(),
        "alpha": out[4],
        "gamma": out[5],
        "delta_h": out[6],
        "iters": out[7],
        "skipped": out[8],
        "ghost_fraction": out[9],
        "newton_fail": out[10],
        "dh0": out[11],
        "d2h0": out[12],
        "feq": out[13:13 + 27].copy(),
        "f_post": out[13 + 27:13 + 54].copy(),
    }


@dataclass
class NativeKbcResult:
    """Wall-normal statistics and scalars of one production channel run."""

    profiles: dict[str, np.ndarray]
    scalars: dict[str, float]
    history: np.ndarray
    meta: dict[str, float] = field(default_factory=dict)

    def profile(self, name: str) -> np.ndarray:
        return self.profiles[name]


def native_channel_run(
    *,
    nx: int,
    ny: int,
    nz: int,
    steps: int,
    warmup: int,
    sample_every: int = 20,
    n_iter: int = 6,
    re_tau: float = 180.0,
    u_tau: float = 0.01,
    alpha_cap: float = 2.0,
    alpha_min: float = 1.0,
    omega_g: float = 1.0,
    tau_bulk: float = -1.0,
    trigger: float = 1e-9,
    enforce_entropy: bool = False,
    noise_amp: float = 0.05,
    seed: int = 7,
    log_every: int = 2000,
) -> NativeKbcResult:
    """Run the KBC channel kernel and return its statistics."""
    lib = _lib()
    ny = int(ny)
    profiles = np.zeros(8 * ny, dtype=np.float64)
    scalars = np.zeros(len(KBC_SCALARS), dtype=np.float64)
    history = np.zeros((20000, 12), dtype=np.float64)
    hist_count = ctypes.c_int(0)
    rc = lib.kbc_channel_run(
        int(nx), ny, int(nz), int(steps), int(warmup), int(sample_every), int(n_iter),
        float(re_tau), float(u_tau), float(alpha_cap), float(alpha_min), float(omega_g),
        float(tau_bulk),
        float(trigger), int(bool(enforce_entropy)), float(noise_amp), int(seed), int(log_every),
        profiles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scalars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        history.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(hist_count),
    )
    if rc != 0:  # pragma: no cover - allocation failure
        raise MemoryError(f"kbc_channel_run returned {rc}")
    prof = {name: profiles[k * ny:(k + 1) * ny].copy() for k, name in enumerate(KBC_PROFILES)}
    scal = {name: float(scalars[k]) for k, name in enumerate(KBC_SCALARS)}
    return NativeKbcResult(
        profiles=prof,
        scalars=scal,
        history=history[: hist_count.value].copy(),
        meta=dict(
            nx=nx, ny=ny, nz=nz, steps=steps, warmup=warmup, re_tau=re_tau, u_tau=u_tau,
            tau=0.5 + 3.0 * (u_tau * (ny - 2) / 2.0) / re_tau,
            half_height=(ny - 2) / 2.0,
        ),
    )