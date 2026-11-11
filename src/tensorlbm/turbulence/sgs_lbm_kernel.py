"""ctypes driver for the fused four-SGS D3Q19 kernel (``sgs_lbm.c``).

The C kernel is a *bit-level replica* of the production torch operators
:func:`tensorlbm.turbulence.turbulence.collide_smagorinsky_bgk3d`,
``collide_dynamic_smagorinsky_bgk3d``, ``collide_wale_bgk3d`` and
``collide_vreman_bgk3d`` (same closure algebra, same ``tau_eff`` clamps), so
the campaigns run here verify the physics that ships in the library.  The
equivalence itself is asserted by
``tests/test_sgs_four_model_verification.py`` (collide-only comparison to
``1e-13`` relative on random fields, and a multi-step trajectory comparison
against the torch ``stream3d`` + ``collide_*`` path).

Why a C kernel: the torch path costs ~0.5 s per 64^3 step and ~3.5 s per
128^3 step on this host (eager element-wise chains over 19-component
tensors), which makes a 32/64/128^3 four-model campaign a multi-day job.
The fused kernel runs 64^3 in ~4 ms and 128^3 in ~35 ms (16 threads).

Provenance: the binary is cached next to the source keyed by the source
SHA-256, so a stale binary can never be reused after an edit.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sgs_lbm.c"

#: Model name -> C model id (see ``sgs_lbm.c`` header).
MODEL_IDS: dict[str, int] = {
    "smagorinsky": 0,
    "dynamic_smagorinsky": 1,
    "wale": 2,
    "vreman": 3,
    "none": 4,  # SGS off: plain BGK (DNS arm / negative control)
}

#: Default model constants (lattice units, Delta = 1) used by the campaign.
DEFAULT_CONSTANTS: dict[str, float] = {
    "smagorinsky": 0.16,
    "dynamic_smagorinsky": 0.0,  # Cs is computed by the Germano identity
    "wale": 0.325,              # Cw; nu_t = Cw^2 (...) -> Cw = 0.325 ~ Cs 0.1
    "vreman": 0.025,            # Cv = 2.5 Cs^2 with Cs = 0.1
}


def _default_threads() -> None:
    """Give OpenMP a sane default before the library is first loaded.

    Without this, two nodes' worth of cores (128 here) are handed to every
    tiny gradient/filter loop and the fork/join overhead dominates.
    """
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(min(32, max(1, os.cpu_count() or 1)))


def build_kernel(out_dir: str | os.PathLike[str] | None = None) -> Path:
    """Compile ``sgs_lbm.c`` (gcc + OpenMP) once per source hash."""
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:  # pragma: no cover - environment guard
        raise RuntimeError("no C compiler (gcc/cc) available for sgs_lbm.c")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    root = Path(out_dir) if out_dir is not None else Path(
        os.environ.get("TENSORLBM_SGS_BUILD", "/tmp/tensorlbm_sgs_kernel")
    )
    root.mkdir(parents=True, exist_ok=True)
    binary = root / f"sgs_lbm_{digest}.so"
    if binary.exists():
        return binary
    # compile to a private name and rename atomically: several worker
    # processes may race to build the same hash on a fresh node
    tmp = binary.with_suffix(f".{os.getpid()}.tmp")
    cmd = [
        compiler, "-O3", "-fopenmp", "-march=native", "-shared", "-fPIC",
        "-o", str(tmp), str(SOURCE), "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.replace(tmp, binary)
    return binary


class SGSKernel:
    """Fused D3Q19-BGK + four-SGS solver on a periodic ``n^3`` box."""

    def __init__(self, n: int, build_dir: str | None = None) -> None:
        self.n = int(n)
        self.N = self.n ** 3
        _default_threads()
        lib = ctypes.CDLL(str(build_kernel(build_dir)))
        d = ctypes.c_double
        p = ctypes.POINTER(d)
        lib.sgs_lbm_init.restype = ctypes.c_int
        lib.sgs_lbm_init.argtypes = [ctypes.c_int]
        lib.sgs_lbm_free.restype = None
        lib.sgs_lbm_free.argtypes = []
        lib.sgs_lbm_set_bc_mode.restype = None
        lib.sgs_lbm_set_bc_mode.argtypes = [ctypes.c_int]
        lib.sgs_lbm_set_velocity.restype = None
        lib.sgs_lbm_set_velocity.argtypes = [p]
        lib.sgs_lbm_get_velocity.restype = None
        lib.sgs_lbm_get_velocity.argtypes = [p]
        lib.sgs_lbm_get_populations.restype = None
        lib.sgs_lbm_get_populations.argtypes = [p]
        lib.sgs_lbm_set_populations.restype = None
        lib.sgs_lbm_set_populations.argtypes = [p]
        lib.sgs_lbm_get_nut.restype = None
        lib.sgs_lbm_get_nut.argtypes = [p]
        lib.sgs_lbm_get_stats.restype = None
        lib.sgs_lbm_get_stats.argtypes = [p]
        lib.sgs_lbm_collide_only.restype = ctypes.c_int
        lib.sgs_lbm_collide_only.argtypes = [p, p, ctypes.c_int, d, d, d]
        lib.sgs_lbm_nu_t_field.restype = ctypes.c_int
        lib.sgs_lbm_nu_t_field.argtypes = [p, ctypes.c_int, d, d, d, p, p]
        lib.sgs_lbm_step.restype = ctypes.c_int
        lib.sgs_lbm_step.argtypes = [ctypes.c_int, d, d, d, ctypes.c_int]
        lib.sgs_lbm_diagnostics.restype = None
        lib.sgs_lbm_diagnostics.argtypes = []
        lib.sgs_lbm_set_tau_for_diag.restype = None
        lib.sgs_lbm_set_tau_for_diag.argtypes = [d]
        lib.sgs_lbm_set_threads.restype = None
        lib.sgs_lbm_set_threads.argtypes = [ctypes.c_int]
        self._lib = lib
        rc = lib.sgs_lbm_init(self.n)
        if rc != 0:
            raise RuntimeError(f"sgs_lbm_init failed (rc={rc})")
        # one team member per ~64k cells: below that the fork/join overhead of
        # the OpenMP regions dominates the gradient/filter passes.
        self.set_threads(max(1, min(int(os.environ.get("TENSORLBM_SGS_MAX_THREADS", "32")),
                                    self.N // 65536 + 1)))

    def set_threads(self, nthreads: int) -> None:
        """Size the OpenMP team (1 per ~64k cells is enough in practice)."""
        self.threads = int(nthreads)
        self._lib.sgs_lbm_set_threads(self.threads)

    # -- state ---------------------------------------------------------
    # NOTE: no __del__/sgs_lbm_free here on purpose.  The C state is global, so
    # freeing it from a finaliser would pull the buffers out from under a newer
    # SGSKernel instance (sgs_lbm_init already frees the previous state when a
    # new kernel is created), and it silently produced nonsense timings.
    def close(self) -> None:
        """Release the C buffers (the next kernel would free them anyway)."""
        self._lib.sgs_lbm_free()

    def set_bc_mode(self, mode: int) -> None:
        """0 = exact production stencil replica, 1 = periodic (default)."""
        self._lib.sgs_lbm_set_bc_mode(int(mode))

    def set_velocity(self, u: np.ndarray) -> None:
        u = np.ascontiguousarray(u, dtype=np.float64)
        if u.shape != (3, self.n, self.n, self.n):
            raise ValueError(f"u must have shape (3,{self.n},{self.n},{self.n})")
        self._lib.sgs_lbm_set_velocity(u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def get_velocity(self) -> np.ndarray:
        out = np.zeros((3, self.n, self.n, self.n), dtype=np.float64)
        self._lib.sgs_lbm_get_velocity(out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out

    def get_populations(self) -> np.ndarray:
        out = np.zeros((19, self.n, self.n, self.n), dtype=np.float64)
        self._lib.sgs_lbm_get_populations(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return out

    def set_populations(self, f: np.ndarray) -> None:
        f = np.ascontiguousarray(f, dtype=np.float64)
        self._lib.sgs_lbm_set_populations(
            f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

    def get_nut(self) -> np.ndarray:
        out = np.zeros((self.n, self.n, self.n), dtype=np.float64)
        self._lib.sgs_lbm_get_nut(out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out

    def diagnostics(self, tau: float) -> dict[str, float]:
        """Recompute the diagnostic moments of the current state (no time step)."""
        self._lib.sgs_lbm_set_tau_for_diag(ctypes.c_double(tau))
        self._lib.sgs_lbm_diagnostics()
        return self.stats()

    def stats(self) -> dict[str, float]:
        buf = np.zeros(16, dtype=np.float64)
        self._lib.sgs_lbm_get_stats(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        keys = ("rho_mean", "tke", "eps_molecular", "eps_sgs", "enstrophy",
                "nut_mean", "nut_max", "u_max", "tau_eff_max", "cs2", "nu",
                "cells", "cs2_last_collide", "eps_total", "pad0", "pad1")
        return {k: float(v) for k, v in zip(keys, buf)}

    # -- operators -----------------------------------------------------
    def step(self, model: str, tau: float, C: float | None = None,
             lambda_clip: float = 0.0, stats: bool = True) -> dict[str, float]:
        """Advance one LBM step.  ``stats=False`` skips the diagnostic sweep."""
        c = 0.0 if model == "none" else (DEFAULT_CONSTANTS[model] if C is None else float(C))
        rc = self._lib.sgs_lbm_step(MODEL_IDS[model], float(tau), float(c),
                                    float(lambda_clip), 1 if stats else 0)
        if rc != 0:  # pragma: no cover
            raise RuntimeError(f"sgs_lbm_step failed (rc={rc})")
        return self.stats()

    def collide_only(self, f: np.ndarray, model: str, tau: float,
                     C: float | None = None, lambda_clip: float = 0.0) -> np.ndarray:
        """One production-equivalent collision (no streaming)."""
        c = 0.0 if model == "none" else (DEFAULT_CONSTANTS[model] if C is None else float(C))
        fin = np.ascontiguousarray(f, dtype=np.float64)
        fout = np.zeros_like(fin)
        pt = ctypes.POINTER(ctypes.c_double)
        self._lib.sgs_lbm_collide_only(
            fin.ctypes.data_as(pt), fout.ctypes.data_as(pt),
            MODEL_IDS[model], float(tau), float(c), float(lambda_clip),
        )
        return fout

    def nu_t_field(self, u: np.ndarray, model: str, tau: float,
                   C: float | None = None, lambda_clip: float = 0.0) -> tuple[np.ndarray, float]:
        """Eddy viscosity (and Cs^2 for the dynamic model) for a velocity field."""
        c = 0.0 if model == "none" else (DEFAULT_CONSTANTS[model] if C is None else float(C))
        uu = np.ascontiguousarray(u, dtype=np.float64)
        nut = np.zeros((self.n, self.n, self.n), dtype=np.float64)
        cs2 = ctypes.c_double(0.0)
        pt = ctypes.POINTER(ctypes.c_double)
        self._lib.sgs_lbm_nu_t_field(
            uu.ctypes.data_as(pt), MODEL_IDS[model], float(tau), float(c),
            float(lambda_clip), nut.ctypes.data_as(pt), ctypes.byref(cs2),
        )
        return nut, float(cs2.value)


__all__ = ["SGSKernel", "MODEL_IDS", "DEFAULT_CONSTANTS", "build_kernel"]