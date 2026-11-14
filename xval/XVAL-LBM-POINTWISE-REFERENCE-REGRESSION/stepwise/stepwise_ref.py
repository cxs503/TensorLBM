"""lbmpy-backed full-step reference (round 2 of the pointwise regression anchor).

The pure-numpy half (definition streaming, shear-wave field, metrics) lives in
``refcore`` so the frozen-data gate can run without lbmpy.  Everything here
needs a live lbmpy install: it is used to *produce* the frozen reference and to
refresh it.

    reference step   S_ref = definition_stream ∘ Collide_lbmpy
    in-repo step     S_tl  = tensorlbm.solver.stream ∘ tensorlbm.solver.collide_*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for _p in (str(HERE), str(REPO / "src"), str(XV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refcore as core  # noqa: E402
from refcore import (D2Q9, compare, definition_stream, equilibrium_field,  # noqa: E402,F401
                     moments, sha256_file, sha256_file_bytes, shear_amplitude, shear_wave_field,
                     tensorlbm_velocities, weights)

import lbmpy_pointwise_ref as ref  # noqa: E402  (round-1 oracle helpers)

NATIVE_MRT_RATES = (1.0, 1.64, 1.54, 1.0, 1.7, 1.0, 1.7, None, None)  # None -> 1/tau


def mrt_relaxation_rates(tau: float) -> list[float]:
    """Native MRT rates used by ``tensorlbm.solver.collide_mrt``."""
    return [float(1.0 / tau) if r is None else float(r) for r in NATIVE_MRT_RATES]


class StepwiseReference:
    """Full LBM step: lbmpy collision symbols + definition streaming.

    No TensorLBM kernel code participates (only the moment matrix *data* of
    ``solver._M_D2Q9_DATA``, so that the MRT arm relaxes in the same basis).
    """

    def __init__(self, *, lattice: str = D2Q9, model: str = "srt", **params: Any) -> None:
        self.lattice = lattice
        self.model = model
        self._mrt_matrix = None
        if model == "mrt_native":
            from tensorlbm import solver  # noqa: PLC0415  (matrix data only)

            m = np.array(solver._M_D2Q9_DATA, dtype=float)
            self._mrt_matrix = m
            self.oss_model = "mrt_from_matrix"
            self.params = {"moment_matrix": m, "tau": params["tau"],
                           "relaxation_rates": mrt_relaxation_rates(params["tau"])}
        elif model == "srt":
            self.oss_model = "srt"
            self.params = {"tau": params["tau"]}
        elif model == "trt":
            self.oss_model = "trt"
            self.params = {"tau_plus": params["tau"],
                           "lambda_trt": params.get("lambda_trt", 3.0 / 16.0)}
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown model {model!r}")
        self._op = ref.PointwiseReference(lattice, self.oss_model, **self.params)

    @property
    def spec(self) -> dict[str, Any]:
        return {"lattice": self.lattice, "model": self.model,
                "params": {k: v for k, v in self.params.items() if k != "moment_matrix"}}

    def collide(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(self._op.apply(np.asarray(f, dtype=np.float64)), dtype=np.float64)

    def step(self, f: np.ndarray) -> np.ndarray:
        return definition_stream(self.collide(f), self.lattice)

    def run(self, f0: np.ndarray, n_steps: int,
            checkpoints: tuple[int, ...] = ()) -> dict[str, Any]:
        f = np.array(f0, dtype=np.float64, copy=True)
        states: dict[int, np.ndarray] = {}
        for t in range(1, n_steps + 1):
            f = self.step(f)
            if t in checkpoints:
                states[t] = f.copy()
        return {"final": f, "checkpoints": states}


# ---------------------------------------------------------------------------
# in-repo full step
# ---------------------------------------------------------------------------
def _collide(ft, model: str, tau: float, lambda_trt: float):
    from tensorlbm import solver  # noqa: PLC0415

    if model == "srt":
        return solver.collide_bgk(ft, tau=tau)
    if model == "trt":
        return solver.collide_trt(ft, tau_plus=tau, lambda_trt=lambda_trt)
    if model == "mrt_native":
        return solver.collide_mrt(ft, tau=tau, s_e=1.64, s_eps=1.54, s_q=1.7)
    raise ValueError(model)


def in_repo_step(f: np.ndarray, *, model: str = "srt", tau: float = 0.7,
                 lambda_trt: float = 3.0 / 16.0) -> np.ndarray:
    """One full step through the in-repo kernels (collide then stream)."""
    import torch  # noqa: PLC0415

    from tensorlbm import solver  # noqa: PLC0415

    post = _collide(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float64)),
                    model, tau, lambda_trt)
    return solver.stream(post).numpy().astype(np.float64)


def in_repo_run(f0: np.ndarray, *, n_steps: int, model: str = "srt", tau: float = 0.7,
                lambda_trt: float = 3.0 / 16.0) -> np.ndarray:
    import torch  # noqa: PLC0415

    from tensorlbm import solver  # noqa: PLC0415

    f = np.array(f0, dtype=np.float64, copy=True)
    for _ in range(n_steps):
        ft = torch.from_numpy(np.ascontiguousarray(f))
        f = solver.stream(_collide(ft, model, tau, lambda_trt)).numpy().astype(np.float64)
    return f


def provenance() -> dict[str, Any]:
    return ref.lbmpy_provenance()


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=float)