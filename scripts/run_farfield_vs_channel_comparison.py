"""Channel vs far-field boundary comparison for sphere and cylinder Cd.

Runs targeted Re=100 cases with both boundary modes and produces a
machine-readable JSON comparison artifact.

* Sphere: D3Q19 BGK / MRT / CUMULANT, Re=100
* Cylinder: D2Q9 BGK / MRT, Re=100

Usage:
    python scripts/run_farfield_vs_channel_comparison.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tensorlbm.cylinder_cross_validation import run_single_combination as run_cylinder
from tensorlbm.sphere_cross_validation import (
    SphereCrossValidationConfig,
    _run_single_combination as run_sphere_single,
    _schiller_naumann,
)

# Schiller-Naumann reference Cd for Re=100 sphere
SPHERE_REF_CD = _schiller_naumann(100.0)
# Literature reference Cd for Re=100 cylinder (2-D)
CYLINDER_REF_CD = 1.33  # Henderson (1995), Re=100 2-D cylinder


def run_sphere_comparison() -> list[dict]:
    """Run sphere D3Q19 BGK/MRT/CUMULANT with channel vs farfield."""
    families = ["BGK", "MRT", "CUMULANT"]
    results = []
    nx = ny = nz = 20
    steps = 50

    for mode in ("channel", "farfield"):
        for family in families:
            t0 = time.time()
            config = SphereCrossValidationConfig(
                nx=nx, ny=ny, nz=nz, steps=steps, boundary_mode=mode
            )
            result = run_sphere_single(config, "D3Q19", family, "none")
            elapsed = time.time() - t0
            entry = {
                "geometry": "sphere",
                "lattice": "D3Q19",
                "collision_family": family,
                "turbulence_model": "none",
                "Re": 100.0,
                "grid": {"nx": nx, "ny": ny, "nz": nz},
                "steps": steps,
                "boundary_mode": mode,
                "Cd": result.Cd,
                "finite": result.finite,
                "steps_completed": result.steps_completed,
                "reference_Cd": SPHERE_REF_CD,
                "Cd_error_pct": (
                    abs(result.Cd - SPHERE_REF_CD) / SPHERE_REF_CD * 100.0
                    if result.Cd is not None and result.finite
                    else None
                ),
                "elapsed_s": round(elapsed, 2),
            }
            if result.Cd is not None:
                print(
                    f"  sphere D3Q19 {family:10s} {mode:9s}: "
                    f"Cd={result.Cd:.4f}  ref={SPHERE_REF_CD:.4f}  "
                    f"err={entry['Cd_error_pct']:.1f}%  ({elapsed:.1f}s)"
                )
            else:
                print(f"  sphere D3Q19 {family:10s} {mode:9s}: Cd=None  ({elapsed:.1f}s)")
            results.append(entry)
    return results


def run_cylinder_comparison() -> list[dict]:
    """Run cylinder D2Q9 BGK/MRT with channel vs farfield."""
    families = ["BGK", "MRT"]
    results = []
    nx, ny, steps = 100, 50, 200

    for mode in ("channel", "farfield"):
        for family in families:
            t0 = time.time()
            result = run_cylinder(
                family, "none", re=100, nx=nx, ny=ny, steps=steps,
                boundary_mode=mode,
            )
            elapsed = time.time() - t0
            cd = result["Cd"]
            entry = {
                "geometry": "cylinder",
                "lattice": "D2Q9",
                "collision_family": family,
                "turbulence_model": "none",
                "Re": 100.0,
                "grid": {"nx": nx, "ny": ny},
                "steps": steps,
                "boundary_mode": mode,
                "Cd": cd,
                "finite": result["finite"],
                "steps_completed": result["steps_completed"],
                "reference_Cd": CYLINDER_REF_CD,
                "Cd_error_pct": (
                    abs(cd - CYLINDER_REF_CD) / CYLINDER_REF_CD * 100.0
                    if cd is not None and result["finite"] and not (
                        isinstance(cd, float) and (cd != cd)  # NaN check
                    )
                    else None
                ),
                "elapsed_s": round(elapsed, 2),
            }
            results.append(entry)
            cd_str = f"{cd:.4f}" if cd is not None and not (isinstance(cd, float) and cd != cd) else "NaN"
            print(
                f"  cylinder D2Q9 {family:10s} {mode:9s}: "
                f"Cd={cd_str}  ref={CYLINDER_REF_CD:.4f}  "
                f"({elapsed:.1f}s)"
            )
    return results


def build_comparison_artifact(
    sphere_results: list[dict],
    cylinder_results: list[dict],
) -> dict:
    """Build the comparison artifact with channel vs farfield Cd summary."""
    # Build per-case comparison pairs
    comparisons = []
    for geom, results in [("sphere", sphere_results), ("cylinder", cylinder_results)]:
        by_family = {}
        for r in results:
            key = (r["collision_family"], r["turbulence_model"])
            by_family.setdefault(key, {})[r["boundary_mode"]] = r

        for (family, turb), modes in by_family.items():
            ch = modes.get("channel")
            ff = modes.get("farfield")
            if ch and ff and ch["Cd"] is not None and ff["Cd"] is not None:
                cd_ch = ch["Cd"]
                cd_ff = ff["Cd"]
                delta = cd_ff - cd_ch
                delta_pct = (delta / cd_ch * 100.0) if cd_ch != 0 else None
            else:
                delta = None
                delta_pct = None
            comparisons.append({
                "geometry": geom,
                "collision_family": family,
                "turbulence_model": turb,
                "Cd_channel": ch["Cd"] if ch else None,
                "Cd_farfield": ff["Cd"] if ff else None,
                "Cd_delta": delta,
                "Cd_delta_pct": delta_pct,
                "Cd_error_pct_channel": ch["Cd_error_pct"] if ch else None,
                "Cd_error_pct_farfield": ff["Cd_error_pct"] if ff else None,
                "reference_Cd": (ch or ff)["reference_Cd"] if (ch or ff) else None,
            })

    return {
        "description": "Channel wall vs far-field boundary Cd comparison for sphere and cylinder at Re=100",
        "schema_version": "tensorlbm.farfield-vs-channel-comparison/v1",
        "sphere_reference_Cd": SPHERE_REF_CD,
        "sphere_reference_source": "Schiller-Naumann correlation, Re=100",
        "cylinder_reference_Cd": CYLINDER_REF_CD,
        "cylinder_reference_source": "Henderson (1995), Re=100 2-D cylinder",
        "comparisons": comparisons,
        "sphere_results": sphere_results,
        "cylinder_results": cylinder_results,
    }


def main() -> None:
    print("=" * 70)
    print("Channel vs Far-field Boundary Cd Comparison")
    print("=" * 70)

    print("\n[1/2] Sphere D3Q19 Re=100 (BGK / MRT / CUMULANT)")
    sphere_results = run_sphere_comparison()

    print("\n[2/2] Cylinder D2Q9 Re=100 (BGK / MRT)")
    cylinder_results = run_cylinder_comparison()

    artifact = build_comparison_artifact(sphere_results, cylinder_results)

    out_path = Path("artifacts/farfield_vs_channel_cd_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n{'=' * 70}")
    print(f"Artifact written to {out_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\nSummary: Cd channel vs farfield")
    header = "{:<10} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Geometry", "Family", "Cd_ch", "Cd_ff", "Delta%", "Err_ch%", "Err_ff%"
    )
    print(header)
    print("-" * 72)
    for c in artifact["comparisons"]:
        cd_ch = "{:.4f}".format(c["Cd_channel"]) if c["Cd_channel"] is not None else "N/A"
        cd_ff = "{:.4f}".format(c["Cd_farfield"]) if c["Cd_farfield"] is not None else "N/A"
        d_pct = "{:.1f}".format(c["Cd_delta_pct"]) if c["Cd_delta_pct"] is not None else "N/A"
        e_ch = "{:.1f}".format(c["Cd_error_pct_channel"]) if c["Cd_error_pct_channel"] is not None else "N/A"
        e_ff = "{:.1f}".format(c["Cd_error_pct_farfield"]) if c["Cd_error_pct_farfield"] is not None else "N/A"
        row = "{:<10} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            c["geometry"], c["collision_family"], cd_ch, cd_ff, d_pct, e_ch, e_ff
        )
        print(row)


if __name__ == "__main__":
    main()
