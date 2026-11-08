#!/usr/bin/env python3
"""Analytic verification of the Allen-Cahn phase-field kernel (m6394, 2026-11-08).

Measures, and writes ``artifacts/allen_cahn_r1_20261108/results.json``:

* **A1** equilibrium profile is stationary: ``L_inf`` drift of ``tanh(s/W)``.
* **A2** equilibrium thickness recovery: start at ``2W``, relax to ``W``.
* **A3** driven travelling wave speed vs the analytic ``v = (3/2) M h xi``.
* **H1** legacy stepper (git blob ``3c0ab6c1:src/tensorlbm/allen_cahn_lbm.py``)
  at its own default arguments - stability only.
* **H2** rewritten stepper at the same default arguments - stability only.
* **H3** static droplet: residual ``max|u|`` (spurious current) and the
  Young-Laplace pressure jump ``dp`` vs ``sigma/R``, for two density ratios.

Every number in the JSON is produced here; nothing is transcribed by hand.
Run:  PYTHONPATH=src python3 scripts/allen_cahn_verify_20261108.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.allen_cahn_lbm import (  # noqa: E402
    allen_cahn_step,
    free_energy_parameters,
    initialize_allen_cahn,
    interface_profile_1d,
    laplace_pressure_jump,
    measure_interface_width,
    step_allen_cahn_field,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402

OUT = REPO / "artifacts" / "allen_cahn_r1_20261108"
TOL = 0.01  # 1% acceptance for the shape/speed anchors


def crossing(phi: torch.Tensor) -> float:
    """Sub-cell position of ``phi = 0`` along x (profile decreases with x)."""
    values = phi[0, 0].to(torch.float64)
    idx = int(torch.nonzero(values >= 0.0).max())
    return idx + (0.0 - float(values[idx])) / (float(values[idx + 1]) - float(values[idx]))


def anchor_stationary() -> dict:
    params = free_energy_parameters(0.01, 4.0)
    ref = interface_profile_1d(96, 4.0, center=48.0)
    phi = ref.clone()
    history = []
    for step in range(1, 4001):
        phi = step_allen_cahn_field(
            phi, beta=params.beta, kappa=params.kappa, mobility=1.0, boundary="no_flux"
        )
        if step in (100, 1000, 4000):
            history.append(
                {
                    "step": step,
                    "linf_drift": float((phi - ref).abs().max()),
                    "width": measure_interface_width(phi),
                    "xc": crossing(phi),
                }
            )
    final = history[-1]
    return {
        "W_nominal": 4.0,
        "W_measured_initial": measure_interface_width(ref),
        "W_measured_final": final["width"],
        "width_rel_error": abs(final["width"] - 4.0) / 4.0,
        "linf_drift_final": final["linf_drift"],
        "interface_displacement": abs(final["xc"] - crossing(ref)),
        "history": history,
        "pass": abs(final["width"] - 4.0) / 4.0 < TOL
        and final["linf_drift"] < 0.01
        and abs(final["xc"] - crossing(ref)) < 0.05,
    }


def anchor_relaxation() -> dict:
    params = free_energy_parameters(0.01, 4.0)
    phi = interface_profile_1d(96, 8.0, center=48.0)
    history = [{"step": 0, "width": measure_interface_width(phi)}]
    for step in range(1, 8001):
        phi = step_allen_cahn_field(
            phi, beta=params.beta, kappa=params.kappa, mobility=1.0, boundary="no_flux"
        )
        if step in (100, 1000, 3000, 8000):
            history.append({"step": step, "width": measure_interface_width(phi)})
    final = history[-1]["width"]
    return {
        "W_target": 4.0,
        "W_initial": history[0]["width"],
        "W_final": final,
        "rel_error": abs(final - 4.0) / 4.0,
        "history": history,
        "pass": abs(final - 4.0) / 4.0 < TOL,
    }


def anchor_travelling_wave(mobility: float = 1.0, driving: float = 5.0e-4, steps: int = 600) -> dict:
    """Interface speed under a constant chemical-potential offset ``h``.

    Analytic asymptotic result for ``dphi/dt = -M (mu - h)`` with the double-well
    free energy: ``v = 2 M h / int(phi'^2) = (3/2) M h xi`` for ``phi = tanh(s/xi)``.
    """
    params = free_energy_parameters(0.01, 4.0)
    xi = math.sqrt(2.0 * params.kappa / params.beta)  # = 4.0
    phi = interface_profile_1d(160, 4.0, center=60.0)
    x0 = crossing(phi)
    w0 = measure_interface_width(phi)
    for _ in range(steps):
        phi = step_allen_cahn_field(
            phi,
            beta=params.beta,
            kappa=params.kappa,
            mobility=mobility,
            boundary="no_flux",
            driving=driving,
        )
    v_num = (crossing(phi) - x0) / steps
    v_analytic = 1.5 * mobility * driving * xi
    return {
        "mobility": mobility,
        "driving": driving,
        "steps": steps,
        "xi": xi,
        "v_numeric": v_num,
        "v_analytic": v_analytic,
        "rel_error": abs(v_num - v_analytic) / v_analytic,
        "width_before": w0,
        "width_after": measure_interface_width(phi),
        "pass": abs(v_num - v_analytic) / v_analytic < TOL,
    }


def _legacy_module() -> types.ModuleType:
    """Import the pre-rewrite module text straight from git (no hand copying)."""
    source = subprocess.check_output(
        ["git", "show", "3c0ab6c1:src/tensorlbm/allen_cahn_lbm.py"], cwd=REPO, text=True
    )
    module = types.ModuleType("legacy_allen_cahn_lbm")
    module.__dict__["__name__"] = "tensorlbm._legacy_allen_cahn_lbm"
    src_relative = source.replace("from .d3q19 import", "from tensorlbm.d3q19 import")
    exec(compile(src_relative, "<legacy allen_cahn_lbm>", "exec"), module.__dict__)
    return module


def default_argument_stability() -> dict:
    """Run 200 steps of both the legacy and the current stepper at defaults."""
    n = 24
    out: dict[str, dict] = {}
    legacy = _legacy_module()
    for label, (init, step_fn) in {
        "legacy": (legacy.initialize_allen_cahn, legacy.allen_cahn_step),
        "current": (initialize_allen_cahn, allen_cahn_step),
    }.items():
        f, g, phi = init(n, n, n)
        blown = None
        umax_history = []
        for step in range(1, 201):
            f, g, phi = step_fn(f, g, phi, allow_unverified_hydro=True)
            rho, ux, uy, uz = macroscopic3d(f)
            umax = float(ux.abs().max() + uy.abs().max() + uz.abs().max())
            umax_history.append(umax)
            if not torch.isfinite(phi).all() or not torch.isfinite(f).all():
                blown = step
                break
        out[label] = {
            "steps_requested": 200,
            "diverged_at_step": blown,
            "umax_step1": umax_history[0],
            "umax_last_finite": umax_history[-1],
            "phi_finite": bool(torch.isfinite(phi).all()),
            "rho_min_last": float(macroscopic3d(f)[0].min()) if torch.isfinite(f).all() else None,
        }
    return out


def droplet_case(*, rho_ratio: float, radius: float = 8.0, n: int = 32,
                 sigma: float = 0.01, W: float = 4.0, steps: int = 1500) -> dict:
    """Static droplet: spurious current and Young-Laplace pressure jump."""
    phi = interface_profile_1d(n, W, center=0.0)  # placeholder, overwritten below
    zz, yy, xx = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    c = (n - 1) / 2.0
    r = torch.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    phi = torch.tanh((radius - r) / W).contiguous()

    rho_h, rho_l = 1.0, 1.0 / rho_ratio
    rho = rho_l + (rho_h - rho_l) * (1.0 + phi) / 2.0
    zero = torch.zeros_like(phi)
    f = equilibrium3d(rho, zero, zero, zero)
    g = torch.zeros(19, *phi.shape)
    umax = []
    blown = None
    for step in range(1, steps + 1):
        f, g, phi = allen_cahn_step(
            f, g, phi, rho_h=rho_h, rho_l=rho_l, nu_h=0.1, nu_l=0.01, sigma=sigma, W=W,
            allow_unverified_hydro=True,
        )
        if step % 50 == 0:
            _, ux, uy, uz = macroscopic3d(f)
            umax.append(float(ux.abs().max() + uy.abs().max() + uz.abs().max()))
        if not torch.isfinite(f).all():
            blown = step
            break
    result = {
        "rho_ratio": rho_ratio,
        "radius": radius,
        "sigma": sigma,
        "steps_requested": steps,
        "diverged_at_step": blown,
        "umax_history_every_50": umax,
        "tail_umax": umax[-1] if umax else None,
    }
    if blown is None:
        result["laplace"] = laplace_pressure_jump(f, phi, radius=radius, sigma=sigma)
    return result


def main() -> dict:
    results = {
        "case_A1_stationary_profile": anchor_stationary(),
        "case_A2_thickness_relaxation": anchor_relaxation(),
        "case_A3_travelling_wave": anchor_travelling_wave(),
        "case_H1_H2_default_stability": default_argument_stability(),
        "case_H3_droplet_laplace": [
            droplet_case(rho_ratio=1000.0),
            droplet_case(rho_ratio=10.0),
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    for key, value in results.items():
        print(f"--- {key}")
        print(json.dumps(value, indent=2)[:1200])
    return results


if __name__ == "__main__":
    main()