#!/usr/bin/env python3
"""Assess pressure-gradient ODE wall models against MKM channel DNS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.pressure_gradient_wall_model import (  # noqa: E402
    solve_pressure_gradient_equilibrium_wall_shear,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_mkm_means(
    path: Path,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    re_tau = None
    wall_distance = []
    y_plus = []
    mean_speed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Re_tau =" in line and re_tau is None:
            re_tau = float(line.split("Re_tau =", 1)[1].split()[0])
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 3:
            wall_distance.append(float(fields[0]))
            y_plus.append(float(fields[1]))
            mean_speed.append(float(fields[2]))
    if re_tau is None or len(wall_distance) < 2:
        raise ValueError("MKM means file lacks Re_tau metadata or profile data")
    return (
        re_tau,
        torch.tensor(wall_distance, dtype=torch.float64),
        torch.tensor(y_plus, dtype=torch.float64),
        torch.tensor(mean_speed, dtype=torch.float64),
    )


def assess(
    dns_path: Path,
    *,
    minimum_y_plus: float = 30.0,
    maximum_y_plus: float = 150.0,
    maximum_mean_error_pct: float = 5.0,
    maximum_rms_error_pct: float = 7.5,
) -> dict:
    if not 0.0 < minimum_y_plus < maximum_y_plus:
        raise ValueError("invalid y+ assessment interval")
    if min(maximum_mean_error_pct, maximum_rms_error_pct) <= 0.0:
        raise ValueError("error thresholds must be positive")
    re_tau, distance, y_plus, speed = _read_mkm_means(dns_path)
    selected = (y_plus >= minimum_y_plus) & (y_plus <= maximum_y_plus)
    if int(selected.sum().item()) < 2:
        raise ValueError("insufficient MKM samples in requested y+ interval")
    # MKM normalization uses u_tau=1 and channel half-height h=1.  The exact
    # fully developed momentum balance is (1/rho) dp/dx=-u_tau^2/h=-1.
    expected_u_tau = 1.0
    nu = 1.0 / re_tau
    pressure_gradient_acceleration = -expected_u_tau**2
    models = {}
    for model in ("van_driest", "duprat"):
        result = solve_pressure_gradient_equilibrium_wall_shear(
            speed[selected],
            distance[selected],
            pressure_gradient_acceleration,
            nu,
            pressure_gradient_magnitude_acceleration=abs(
                pressure_gradient_acceleration
            ),
            eddy_viscosity_model=model,
            quadrature_points=256,
        )
        relative_error = result.friction_velocity / expected_u_tau - 1.0
        mean_error_pct = float(relative_error.mean().item() * 100.0)
        rms_error_pct = float(
            torch.sqrt(relative_error.square().mean()).item() * 100.0,
        )
        all_attached = bool(result.attached.all().item())
        accepted = (
            all_attached
            and abs(mean_error_pct) <= maximum_mean_error_pct
            and rms_error_pct <= maximum_rms_error_pct
        )
        models[model] = {
            "attached_samples": int(result.attached.sum().item()),
            "sample_count": int(result.attached.numel()),
            "minimum_predicted_u_tau": float(
                result.friction_velocity.min().item()
            ),
            "mean_predicted_u_tau": float(
                result.friction_velocity.mean().item()
            ),
            "maximum_predicted_u_tau": float(
                result.friction_velocity.max().item()
            ),
            "mean_error_pct": mean_error_pct,
            "rms_error_pct": rms_error_pct,
            "maximum_absolute_error_pct": float(
                relative_error.abs().max().item() * 100.0
            ),
            "accepted": accepted,
        }
    return {
        "schema": "tensorlbm-pressure-gradient-wall-mkm-dns-assessment-v1",
        "status": "diagnostic_only",
        "physical_validation": any(
            result["accepted"] for result in models.values()
        ),
        "source": {
            "path": str(dns_path),
            "sha256": _sha256(dns_path),
            "citation": (
                "Moser, Kim & Mansour, Physics of Fluids 11, 943-945 "
                "(1999)"
            ),
        },
        "normalization": {
            "re_tau": re_tau,
            "u_tau": expected_u_tau,
            "channel_half_height": 1.0,
            "nu": nu,
            "pressure_gradient_acceleration": pressure_gradient_acceleration,
            "pressure_gradient_definition": (
                "exact fully developed balance: dp_dx/rho=-u_tau^2/h"
            ),
        },
        "assessment": {
            "minimum_y_plus": minimum_y_plus,
            "maximum_y_plus": maximum_y_plus,
            "sample_y_plus": y_plus[selected].tolist(),
            "maximum_mean_error_pct": maximum_mean_error_pct,
            "maximum_rms_error_pct": maximum_rms_error_pct,
        },
        "models": models,
        "production_force_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dns", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-y-plus", type=float, default=30.0)
    parser.add_argument("--maximum-y-plus", type=float, default=150.0)
    parser.add_argument("--maximum-mean-error-pct", type=float, default=5.0)
    parser.add_argument("--maximum-rms-error-pct", type=float, default=7.5)
    args = parser.parse_args()
    result = assess(
        args.dns,
        minimum_y_plus=args.minimum_y_plus,
        maximum_y_plus=args.maximum_y_plus,
        maximum_mean_error_pct=args.maximum_mean_error_pct,
        maximum_rms_error_pct=args.maximum_rms_error_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
