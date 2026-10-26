"""Reproduce the original mass gate and attribute changes by operator.

Run from the repository root:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    .venv/bin/python evidence/navier_slip_mass/reproduce.py

This is a CPU diagnostic, not a physical slip-length validation. Float64
reductions measure changes in the stored populations; they do not remove
roundoff introduced by float32 evolution.
"""

import json
import platform

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.navier_slip_bc import navier_slip_cells_3d
from tensorlbm.solver3d import stream3d


def audit(dtype, with_boundary):
    rho = torch.ones((4, 21, 16), dtype=dtype)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)
    initial = f.clone()
    bottom = torch.zeros_like(rho, dtype=torch.bool)
    top = torch.zeros_like(bottom)
    bottom[:, 0, :] = True
    top[:, -1, :] = True
    changes = dict(collision=0.0, streaming=0.0, boundary=0.0)
    max_boundary_step_change = 0.0
    max_true_drift = 0.0
    for _ in range(100):
        previous = f
        feq = equilibrium3d(*macroscopic3d(f))
        f = f - (f - feq) / 1.0
        changes["collision"] += (f.double() - previous.double()).sum().item()
        previous = f
        f = stream3d(f)
        changes["streaming"] += (f.double() - previous.double()).sum().item()
        previous = f
        if with_boundary:
            f = navier_slip_cells_3d(f, bottom, axis=1, Ls=2.0)
            f = navier_slip_cells_3d(f, top, axis=1, Ls=2.0)
        boundary_change = (f.double() - previous.double()).sum().item()
        changes["boundary"] += boundary_change
        max_boundary_step_change = max(max_boundary_step_change, abs(boundary_change))
        max_true_drift = max(max_true_drift, abs((f.double() - initial.double()).sum().item()))
    native_drift = abs(f.sum().item() - initial.sum().item())
    return {
        "dtype": str(dtype),
        "with_boundary": with_boundary,
        "initial_native_mass": initial.sum().item(),
        "final_native_mass": f.sum().item(),
        "native_absolute_drift": native_drift,
        "original_1e_6_gate_passed": native_drift < 1e-6,
        "population_signed_drift_float64": (f.double() - initial.double()).sum().item(),
        "max_population_absolute_drift_float64": max_true_drift,
        "operator_signed_changes_float64": changes,
        "max_boundary_step_absolute_change_float64": max_boundary_step_change,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": torch.get_num_threads(),
                "cases": [
                    audit(dtype, bc)
                    for dtype in (torch.float32, torch.float64)
                    for bc in (True, False)
                ],
            },
            indent=2,
        )
    )
