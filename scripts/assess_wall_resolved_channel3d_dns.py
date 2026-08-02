#!/usr/bin/env python3
"""Compare a 3-D channel result with Moser-Kim-Mansour DNS mean velocity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_dns(path: Path) -> tuple[torch.Tensor, torch.Tensor, float]:
    y_plus = []
    u_plus = []
    reference_re_tau = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Re_tau =" in line and reference_re_tau is None:
            reference_re_tau = float(line.split("Re_tau =", 1)[1].split()[0])
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        if len(values) >= 3:
            y_plus.append(float(values[1]))
            u_plus.append(float(values[2]))
    if len(y_plus) < 2 or reference_re_tau is None:
        raise ValueError("DNS reference lacks profile or Re_tau metadata")
    return (
        torch.tensor(y_plus, dtype=torch.float64),
        torch.tensor(u_plus, dtype=torch.float64),
        reference_re_tau,
    )


def _read_reynolds_stress(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_plus = []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 7:
            y_plus.append(float(fields[1]))
            values.append([float(value) for value in fields[2:6]])
    if len(y_plus) < 2:
        raise ValueError("DNS Reynolds-stress reference lacks profile data")
    return (
        torch.tensor(y_plus, dtype=torch.float64),
        torch.tensor(values, dtype=torch.float64),
    )


def _linear_interpolate(
    x: torch.Tensor,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
) -> torch.Tensor:
    upper = torch.searchsorted(source_x, x).clamp(1, source_x.numel() - 1)
    lower = upper - 1
    fraction = (x - source_x[lower]) / (source_x[upper] - source_x[lower])
    return source_y[lower] + fraction * (source_y[upper] - source_y[lower])


def assess(
    result_path: Path,
    dns_path: Path,
    *,
    dns_reynolds_stress_path: Path | None = None,
    minimum_y_plus: float = 1.0,
    maximum_outer_fraction: float = 0.8,
    maximum_u_plus_rms_error: float = 1.0,
    maximum_reynolds_stress_rms_error: float = 1.0,
) -> dict:
    if minimum_y_plus < 0.0 or not 0.0 < maximum_outer_fraction <= 1.0:
        raise ValueError("invalid channel profile interval")
    if maximum_u_plus_rms_error <= 0.0:
        raise ValueError("maximum_u_plus_rms_error must be positive")
    if maximum_reynolds_stress_rms_error <= 0.0:
        raise ValueError("maximum Reynolds-stress RMS error must be positive")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "tensorlbm-wall-resolved-channel3d-result-v1":
        raise ValueError("not a wall-resolved channel3d result")
    config = result["configuration"]
    derived = result["derived"]
    profile = torch.tensor(
        result["statistics"]["mean_velocity_profile"],
        dtype=torch.float64,
    )
    ny = int(config["ny"])
    if profile.numel() != ny:
        raise ValueError("mean profile length disagrees with channel ny")
    u_tau = float(config["u_tau"])
    nu = float(derived["nu"])
    re_tau = float(config["re_tau"])
    lower = profile[1 : ny // 2]
    upper = profile[ny // 2 : ny - 1].flip(0)
    count = min(lower.numel(), upper.numel())
    symmetric_velocity = 0.5 * (lower[:count] + upper[:count])
    distance = torch.arange(1, count + 1, dtype=torch.float64) - 0.5
    sample_y_plus = distance * u_tau / nu
    sample_u_plus = symmetric_velocity / u_tau
    dns_y_plus, dns_u_plus, reference_re_tau = _read_dns(dns_path)
    selected = (
        (sample_y_plus >= minimum_y_plus)
        & (sample_y_plus <= maximum_outer_fraction * re_tau)
        & (sample_y_plus >= dns_y_plus.min())
        & (sample_y_plus <= dns_y_plus.max())
    )
    if int(selected.sum().item()) < 2:
        raise ValueError("insufficient channel samples inside DNS interval")
    evaluated_y_plus = sample_y_plus[selected]
    evaluated_u_plus = sample_u_plus[selected]
    reference_u_plus = _linear_interpolate(
        evaluated_y_plus,
        dns_y_plus,
        dns_u_plus,
    )
    error = evaluated_u_plus - reference_u_plus
    rms_error = float(torch.sqrt(error.square().mean()).item())
    maximum_error = float(error.abs().max().item())
    reference_reynolds_difference_pct = (
        (re_tau - reference_re_tau) / reference_re_tau * 100.0
    )
    profile_target_met = rms_error <= maximum_u_plus_rms_error
    source_physics_admitted = bool(result.get("physical_validation", False))
    stress_assessment = None
    stress_target_met = dns_reynolds_stress_path is None
    if dns_reynolds_stress_path is not None:
        stress_profiles = result["statistics"].get(
            "reynolds_stress_profiles_uu_vv_ww_uv",
        )
        if stress_profiles is None:
            raise ValueError("channel result lacks Reynolds-stress profiles")
        stress = torch.tensor(stress_profiles, dtype=torch.float64)
        if stress.shape != (4, ny):
            raise ValueError("channel Reynolds-stress profile has invalid shape")
        lower_stress = stress[:, 1 : ny // 2]
        upper_stress = stress[:, ny // 2 : ny - 1].flip(1)
        stress_count = min(lower_stress.shape[1], upper_stress.shape[1])
        symmetric_stress = 0.5 * (
            lower_stress[:, :stress_count] + upper_stress[:, :stress_count]
        )
        # Global y points in the opposite wall-normal direction at the upper
        # wall, so uv changes sign before wall-oriented symmetrization.
        symmetric_stress[3] = 0.5 * (
            lower_stress[3, :stress_count] - upper_stress[3, :stress_count]
        )
        symmetric_stress /= u_tau**2
        dns_stress_y_plus, dns_stress = _read_reynolds_stress(
            dns_reynolds_stress_path,
        )
        stress_selected = (
            (sample_y_plus[:stress_count] >= minimum_y_plus)
            & (sample_y_plus[:stress_count] <= maximum_outer_fraction * re_tau)
            & (sample_y_plus[:stress_count] >= dns_stress_y_plus.min())
            & (sample_y_plus[:stress_count] <= dns_stress_y_plus.max())
        )
        stress_y_plus = sample_y_plus[:stress_count][stress_selected]
        evaluated_stress = symmetric_stress[:, stress_selected]
        reference_stress = torch.stack([
            _linear_interpolate(
                stress_y_plus,
                dns_stress_y_plus,
                dns_stress[:, component],
            )
            for component in range(4)
        ])
        component_error = evaluated_stress - reference_stress
        component_rms = torch.sqrt(component_error.square().mean(dim=1))
        stress_target_met = bool(
            (component_rms <= maximum_reynolds_stress_rms_error).all().item(),
        )
        stress_assessment = {
            "component_order": ["uu", "vv", "ww", "uv"],
            "component_rms_error": component_rms.tolist(),
            "maximum_component_rms_error": float(component_rms.max().item()),
            "maximum_allowed_component_rms_error": (
                maximum_reynolds_stress_rms_error
            ),
            "target_met": stress_target_met,
            "sample_count": int(stress_selected.sum().item()),
        }
    overall = source_physics_admitted and profile_target_met and stress_target_met
    return {
        "schema": "tensorlbm-wall-resolved-channel3d-dns-assessment-v1",
        "status": (
            "admitted" if overall
            else "rejected"
        ),
        "physical_validation": overall,
        "sources": {
            "channel_result": str(result_path),
            "channel_result_sha256": _sha256(result_path),
            "dns_reference": str(dns_path),
            "dns_reference_sha256": _sha256(dns_path),
            "dns_reference_citation": (
                "Moser, Kim & Mansour, Physics of Fluids 11, 943-945 (1999)"
            ),
            "dns_reynolds_stress_reference": (
                str(dns_reynolds_stress_path)
                if dns_reynolds_stress_path is not None else None
            ),
            "dns_reynolds_stress_reference_sha256": (
                _sha256(dns_reynolds_stress_path)
                if dns_reynolds_stress_path is not None else None
            ),
        },
        "configuration": {
            "channel_re_tau": re_tau,
            "dns_re_tau": reference_re_tau,
            "re_tau_difference_pct": reference_reynolds_difference_pct,
            "minimum_y_plus": minimum_y_plus,
            "maximum_outer_fraction": maximum_outer_fraction,
            "maximum_u_plus_rms_error": maximum_u_plus_rms_error,
            "maximum_reynolds_stress_rms_error": (
                maximum_reynolds_stress_rms_error
            ),
            "sample_count": int(selected.sum().item()),
        },
        "profile_error": {
            "u_plus_rms": rms_error,
            "u_plus_maximum_absolute": maximum_error,
            "target_met": profile_target_met,
            "sample_y_plus": evaluated_y_plus.tolist(),
            "sample_u_plus": evaluated_u_plus.tolist(),
            "reference_u_plus": reference_u_plus.tolist(),
        },
        "reynolds_stress_error": stress_assessment,
        "source_channel_acceptance": result["acceptance"],
        "source_channel_physical_validation": source_physics_admitted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("dns", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dns-reynolds-stress", type=Path)
    parser.add_argument("--minimum-y-plus", type=float, default=1.0)
    parser.add_argument("--maximum-outer-fraction", type=float, default=0.8)
    parser.add_argument("--maximum-u-plus-rms-error", type=float, default=1.0)
    parser.add_argument(
        "--maximum-reynolds-stress-rms-error", type=float, default=1.0,
    )
    args = parser.parse_args()
    assessment = assess(
        args.result,
        args.dns,
        dns_reynolds_stress_path=args.dns_reynolds_stress,
        minimum_y_plus=args.minimum_y_plus,
        maximum_outer_fraction=args.maximum_outer_fraction,
        maximum_u_plus_rms_error=args.maximum_u_plus_rms_error,
        maximum_reynolds_stress_rms_error=(
            args.maximum_reynolds_stress_rms_error
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assessment, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(assessment, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
