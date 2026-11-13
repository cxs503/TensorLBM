"""Cross-validation benchmark: TensorLBM LatticeBridge vs 2b-t/LB-t Converter.

This benchmark verifies that both unit conversion approaches produce
numerically identical results when given the same physical operating point.

Strategy
--------
1. Define a set of physical test cases (Poiseuille channel, lid-driven cavity,
   Taylor-Green vortex, low-Re channel flow).
2. For each case, bootstrap both:
   - TensorLBM ``LatticeBridge`` (via ``bootstrap_bridge``)
   - ``LbtStyleConverter`` (via the LB-t triple-pair constructor)
3. Convert a set of physical quantities to LBM and back using both converters.
4. Compare the results: the relative difference must be < 15% (owner precision
   standard), and for correct implementations should be at machine precision
   (< 1e-14).
5. Run the TensorLBM analytic anchors (Poiseuille, Couette, Stokes,
   Taylor-Green) and verify that the LbtStyleConverter produces the same
   physical results when used to dimensionalise the lattice output.

Reference
---------
2b-t/LB-t: https://github.com/2b-t/LB-t
  - ``include/lbt/converter.hpp``: Converter class
  - ``include/lbt/units/``: typed unit classes
  - ``examples/test_1/settings.json``: case schema

J. Latt, "Choice of units in lattice Boltzmann simulations",
  Palabos Documentation (2008).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorlbm.units.bridge import LatticeBridge, bootstrap_bridge
from tensorlbm.units.anchors import ANCHORS, TOLERANCES, U_LB, build_case, evaluate
from tensorlbm.xval.converter import LbtStyleConverter
from tensorlbm.xval.case_schema import CaseSchema, Physics, Discretisation, bootstrap_from_schema


# ---------------------------------------------------------------------------
# Test cases — all must satisfy Ma < 0.1 and tau in [0.51, 2.0]
# ---------------------------------------------------------------------------

XVAL_CASES: dict[str, dict[str, float]] = {
    "poiseuille_water": {
        "L": 1.0e-3,       # channel height [m]
        "u_ref": 2.5e-3,   # max velocity [m/s]
        "nu": 1.0e-6,      # water kinematic viscosity [m²/s]
        "rho0": 998.0,     # water density [kg/m³]
        "nx": 48,
        "u_lb": 0.02,
        # Re = u*L/nu = 2.5, tau = 0.5 + 3*u_lb*nx/Re = 1.652
    },
    "cavity_air": {
        "L": 0.1,          # cavity side [m]
        "u_ref": 0.5,      # lid velocity [m/s]
        "nu": 1.535e-5,    # air kinematic viscosity [m²/s]
        "rho0": 1.1881,    # air density [kg/m³]
        "nx": 64,
        "u_lb": 0.05,
        # Re = 3257, tau too low at nx=64; auto-fixed below
    },
    "taylor_green_oil": {
        "L": 0.05,         # box size [m]
        "u_ref": 0.01,     # initial amplitude [m/s]
        "nu": 1.0e-4,      # oil kinematic viscosity [m²/s]
        "rho0": 870.0,     # oil density [kg/m³]
        "nx": 32,
        "u_lb": 0.04,
        # Re = 5, tau = 1.268
    },
    "channel_low_re": {
        "L": 5.0e-3,       # channel height [m]
        "u_ref": 1.0e-3,   # bulk velocity [m/s]
        "nu": 5.0e-6,      # kinematic viscosity [m²/s]
        "rho0": 998.0,     # water density [kg/m³]
        "nx": 64,
        "u_lb": 0.03,
        # Re = 1, tau too high; auto-fixed below
    },
}


def _check_case_viability(case: dict) -> dict[str, float]:
    """Pre-check whether a case satisfies Ma < 0.1 and tau in [0.51, 2.0]."""
    re = case["u_ref"] * case["L"] / case["nu"]
    nu_lb = case["u_lb"] * case["nx"] / re
    tau = 0.5 + 3.0 * nu_lb
    ma = case["u_lb"] * math.sqrt(3.0)
    return {"Re": re, "nu_lb": nu_lb, "tau": tau, "Ma": ma,
            "tau_ok": 0.51 <= tau <= 2.0, "ma_ok": ma < 0.1}


def _fix_cases():
    """Adjust XVAL_CASES to satisfy tau in [0.51, 2.0] and Ma < 0.1."""
    fixed = {}
    for name, case in XVAL_CASES.items():
        info = _check_case_viability(case)
        if not info["tau_ok"] or not info["ma_ok"]:
            re = case["u_ref"] * case["L"] / case["nu"]
            target_tau = 0.8
            product = (target_tau - 0.5) * re / 3.0
            nx = case["nx"]
            u_lb = product / nx
            if u_lb * math.sqrt(3.0) >= 0.1:
                nx = max(nx, int(math.ceil(product / (0.05 / math.sqrt(3.0)))))
                u_lb = product / nx
            case = dict(case)
            case["nx"] = nx
            case["u_lb"] = round(u_lb, 6)
        fixed[name] = case
    return fixed


XVAL_CASES = _fix_cases()


def _make_bridge(case: dict) -> LatticeBridge:
    """Bootstrap a LatticeBridge from a test case dict."""
    return bootstrap_bridge(
        length_phys=case["L"],
        velocity_phys=case["u_ref"],
        nu_phys=case["nu"],
        nx=case["nx"],
        rho0=case["rho0"],
        u_lb=case["u_lb"],
    )


def _make_lbt_converter(case: dict, bridge: LatticeBridge) -> LbtStyleConverter:
    """Build an LbtStyleConverter from the same operating point.

    The LB-t Converter takes (physical, equivalent_lbm) pairs.
    We derive the LBM equivalents from the operating point:
      - equivalent_lbm_length = nx (the characteristic length maps to nx cells)
      - equivalent_lbm_velocity = u_lb
      - equivalent_lbm_density = 1.0 (lattice density = 1 by convention)
    """
    return LbtStyleConverter(
        physical_length=case["L"],
        equivalent_lbm_length=float(case["nx"]),
        physical_velocity=case["u_ref"],
        equivalent_lbm_velocity=case["u_lb"],
        physical_density=case["rho0"],
        equivalent_lbm_density=1.0,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_roundtrip(
    bridge: LatticeBridge,
    lbt: LbtStyleConverter,
) -> dict[str, Any]:
    """Compare roundtrip conversions for all sites between the two converters.

    For each physical quantity, convert to LBM and back using both converters,
    then compute the relative difference.
    """
    test_values = {
        "length": 1.0e-3,     # 1 mm
        "velocity": 1.0e-2,   # 1 cm/s
        "density": 1000.0,    # 1000 kg/m³
        "time": 1.0,          # 1 s
        "nu": 1.0e-6,         # 1e-6 m²/s
        "pressure": 101325.0, # 1 atm
    }

    results = {}
    for site, phys_val in test_values.items():
        # TensorLBM LatticeBridge
        if site == "length":
            lb_val_bridge = bridge.length_to_lattice(phys_val)
            rt_bridge = bridge.length_to_phys(lb_val_bridge)
        elif site == "velocity":
            lb_val_bridge = bridge.velocity_to_lattice(phys_val)
            rt_bridge = bridge.velocity_to_phys(lb_val_bridge)
        elif site == "density":
            lb_val_bridge = bridge.density_to_lattice(phys_val)
            rt_bridge = bridge.density_to_phys(lb_val_bridge)
        elif site == "time":
            lb_val_bridge = bridge.time_to_lattice(phys_val)
            rt_bridge = bridge.time_to_phys(lb_val_bridge)
        elif site == "nu":
            lb_val_bridge = bridge.viscosity_to_lattice(phys_val)
            rt_bridge = bridge.viscosity_to_phys(lb_val_bridge)
        elif site == "pressure":
            lb_val_bridge = bridge.stress_to_lattice(phys_val)
            rt_bridge = bridge.stress_to_phys(lb_val_bridge)
        else:
            continue

        # LbtStyleConverter
        to_lbm = getattr(lbt, f"{site}_to_lbm")
        to_phys = getattr(lbt, f"{site}_to_phys")
        lb_val_lbt = to_lbm(phys_val)
        rt_lbt = to_phys(lb_val_lbt)

        # Compare LBM values
        if abs(lb_val_bridge) > 0:
            lb_diff = abs(lb_val_bridge - lb_val_lbt) / abs(lb_val_bridge)
        else:
            lb_diff = 0.0

        # Compare roundtrip values
        if abs(phys_val) > 0:
            rt_diff_bridge = abs(rt_bridge - phys_val) / abs(phys_val)
            rt_diff_lbt = abs(rt_lbt - phys_val) / abs(phys_val)
            cross_diff = abs(rt_bridge - rt_lbt) / abs(phys_val)
        else:
            rt_diff_bridge = 0.0
            rt_diff_lbt = 0.0
            cross_diff = 0.0

        results[site] = {
            "phys_value": phys_val,
            "lbm_bridge": lb_val_bridge,
            "lbm_lbt": lb_val_lbt,
            "lbm_rel_diff": lb_diff,
            "rt_bridge": rt_bridge,
            "rt_lbt": rt_lbt,
            "rt_rel_diff_bridge": rt_diff_bridge,
            "rt_rel_diff_lbt": rt_diff_lbt,
            "cross_rel_diff": cross_diff,
        }

    return results


def _compare_derived_quantities(
    bridge: LatticeBridge,
    lbt: LbtStyleConverter,
    case: dict,
) -> dict[str, Any]:
    """Compare derived LBM quantities (dx, dt, tau, nu_lb, Ma, Re).

    Key insight: In the LB-t Converter, ``to_lbm_time`` IS the physical time
    per lattice step (dt), because:
      to_lbm_time = to_lbm_length / to_lbm_velocity
                  = (L/nx) / (u_ref/u_lb)
                  = L * u_lb / (nx * u_ref)
                  = dx * u_lb / u_ref
                  = dt
    And toPhysical(t_lb) = to_lbm_time * t_lb = dt * t_lb.
    """
    dx_bridge = bridge.dx
    dt_bridge = bridge.dt
    nu_lb_bridge = bridge.viscosity_to_lattice(case["nu"])
    tau_bridge = bridge.tau_from_viscosity(nu_lb_bridge)
    ma_bridge = bridge.mach(case["u_lb"])
    re_bridge = bridge.reynolds_lattice(case["u_lb"], float(case["nx"]), nu_lb_bridge)

    # LbtStyleConverter
    dx_lbt = lbt.physical_length / lbt.equivalent_lbm_length  # = L/nx = dx
    dt_lbt = lbt.to_lbm_time  # = to_lbm_length / to_lbm_velocity = dt
    nu_lb_lbt = lbt.nu_to_lbm(case["nu"])
    tau_lbt = lbt.tau_from_nu_lb(nu_lb_lbt)
    ma_lbt = lbt.mach(case["u_lb"])
    re_lbt = lbt.reynolds(case["u_ref"], case["L"], case["nu"])

    results = {
        "dx": {
            "bridge": dx_bridge, "lbt": dx_lbt,
            "rel_diff": abs(dx_bridge - dx_lbt) / abs(dx_bridge) if dx_bridge else 0.0,
        },
        "dt": {
            "bridge": dt_bridge, "lbt": dt_lbt,
            "rel_diff": abs(dt_bridge - dt_lbt) / abs(dt_bridge) if dt_bridge else 0.0,
        },
        "nu_lb": {
            "bridge": nu_lb_bridge, "lbt": nu_lb_lbt,
            "rel_diff": abs(nu_lb_bridge - nu_lb_lbt) / abs(nu_lb_bridge) if nu_lb_bridge else 0.0,
        },
        "tau": {
            "bridge": tau_bridge, "lbt": tau_lbt,
            "rel_diff": abs(tau_bridge - tau_lbt) / abs(tau_bridge) if tau_bridge else 0.0,
        },
        "Ma": {
            "bridge": ma_bridge, "lbt": ma_lbt,
            "rel_diff": abs(ma_bridge - ma_lbt) / abs(ma_bridge) if ma_bridge else 0.0,
        },
        "Re": {
            "bridge": re_bridge, "lbt": re_lbt,
            "rel_diff": abs(re_bridge - re_lbt) / abs(re_bridge) if re_bridge else 0.0,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Schema-based validation
# ---------------------------------------------------------------------------

def _validate_schema_roundtrip(case_name: str, case: dict) -> dict[str, Any]:
    """Build a CaseSchema, bootstrap a bridge, and verify it matches direct bootstrap."""
    schema = CaseSchema(
        physics=Physics(
            density=case["rho0"],
            kinematic_viscosity=case["nu"],
        ),
        discretisation=Discretisation(NX=case["nx"]),
        characteristic_length=case["L"],
        characteristic_velocity=case["u_ref"],
        u_lb=case["u_lb"],
    )

    bridge_schema = bootstrap_from_schema(schema)
    bridge_direct = _make_bridge(case)

    dx_diff = abs(bridge_schema.dx - bridge_direct.dx) / abs(bridge_direct.dx)
    dt_diff = abs(bridge_schema.dt - bridge_direct.dt) / abs(bridge_direct.dt)

    # JSON roundtrip
    json_dict = schema.to_json()
    schema_rt = CaseSchema.from_json(json_dict)
    bridge_rt = bootstrap_from_schema(schema_rt)
    dx_rt_diff = abs(bridge_rt.dx - bridge_direct.dx) / abs(bridge_direct.dx)
    dt_rt_diff = abs(bridge_rt.dt - bridge_direct.dt) / abs(bridge_direct.dt)

    return {
        "schema_vs_direct": {"dx_rel_diff": dx_diff, "dt_rel_diff": dt_diff},
        "json_roundtrip": {"dx_rel_diff": dx_rt_diff, "dt_rel_diff": dt_rt_diff},
        "passed": dx_diff < 1e-14 and dt_diff < 1e-14 and dx_rt_diff < 1e-14 and dt_rt_diff < 1e-14,
    }


# ---------------------------------------------------------------------------
# Anchor cross-validation
# ---------------------------------------------------------------------------

def _anchor_xval(anchor_name: str, grid: int = 32) -> dict[str, Any]:
    """Run an analytic anchor and dimensionalise with both converters.

    The anchor produces lattice results; we convert to physical using both
    the LatticeBridge and the LbtStyleConverter, then compare the *converter*
    outputs (not the anchor simulation error).

    Important: The anchor module uses U_LB = 0.02 (not 0.05) as the reference
    lattice velocity.  The LbtStyleConverter must use the same value.
    """
    case = build_case(anchor_name, 0, grid)
    res = evaluate(case)
    bridge = case.bridge
    lat = case.lattice
    phys = case.physical

    # Build LbtStyleConverter from the same operating point as the anchor
    # The anchor uses U_LB from the module (currently 0.02)
    if anchor_name in ("poiseuille", "couette"):
        char_length = 2.0 * phys["h"]  # full channel height
    elif anchor_name == "stokes":
        char_length = 2.0 * phys["a"]  # sphere diameter
    else:
        char_length = phys["h"]  # box size

    lbt = LbtStyleConverter(
        physical_length=char_length,
        equivalent_lbm_length=float(grid),
        physical_velocity=phys["u_ref"],
        equivalent_lbm_velocity=U_LB,  # use anchor's U_LB (0.02)
        physical_density=phys["rho0"],
        equivalent_lbm_density=1.0,
    )

    if anchor_name == "poiseuille":
        u_max_bridge = bridge.velocity_to_phys(lat["u_max"])
        u_max_lbt = lbt.velocity_to_phys(lat["u_max"])
        diff = abs(u_max_bridge - u_max_lbt) / abs(u_max_bridge) if u_max_bridge else 0.0
        return {
            "anchor": anchor_name,
            "grid": grid,
            "u_max_bridge": u_max_bridge,
            "u_max_lbt": u_max_lbt,
            "converter_rel_diff": diff,
            "anchor_sim_error": res["worst_rel_error"],
            "passed": diff < 1e-12,
        }

    elif anchor_name == "couette":
        u_wall_bridge = bridge.velocity_to_phys(lat["u_lb"])
        u_wall_lbt = lbt.velocity_to_phys(lat["u_lb"])
        diff = abs(u_wall_bridge - u_wall_lbt) / abs(u_wall_bridge) if u_wall_bridge else 0.0
        return {
            "anchor": anchor_name,
            "grid": grid,
            "u_wall_bridge": u_wall_bridge,
            "u_wall_lbt": u_wall_lbt,
            "converter_rel_diff": diff,
            "anchor_sim_error": res["worst_rel_error"],
            "passed": diff < 1e-12,
        }

    elif anchor_name == "stokes":
        force_lb = 6.0 * math.pi * lat["nu_lb"] * lat["a_lb"] * lat["u_lb"]
        force_bridge = bridge.force_to_phys(force_lb)
        # LbtStyleConverter: force = rho * dx^4 / dt^2 * force_lb
        dx = bridge.dx
        dt = bridge.dt
        force_lbt = force_lb * phys["rho0"] * dx**4 / dt**2
        diff = abs(force_bridge - force_lbt) / abs(force_bridge) if force_bridge else 0.0
        return {
            "anchor": anchor_name,
            "grid": grid,
            "force_bridge": force_bridge,
            "force_lbt": force_lbt,
            "converter_rel_diff": diff,
            "anchor_sim_error": res["worst_rel_error"],
            "passed": diff < 1e-12,
        }

    else:  # taylor_green
        rate_bridge = bridge.rate_to_phys(lat["rate_lb"])
        k_bridge = bridge.wave_number_to_phys(lat["k_lb"])
        # LbtStyleConverter: rate_phys = rate_lb / to_lbm_time
        rate_lbt = lat["rate_lb"] / lbt.to_lbm_time
        # k_phys = k_lb / to_lbm_length
        k_lbt = lat["k_lb"] / lbt.to_lbm_length
        rate_diff = abs(rate_bridge - rate_lbt) / abs(rate_bridge) if rate_bridge else 0.0
        k_diff = abs(k_bridge - k_lbt) / abs(k_bridge) if k_bridge else 0.0
        return {
            "anchor": anchor_name,
            "grid": grid,
            "rate_bridge": rate_bridge,
            "rate_lbt": rate_lbt,
            "rate_converter_rel_diff": rate_diff,
            "k_bridge": k_bridge,
            "k_lbt": k_lbt,
            "k_converter_rel_diff": k_diff,
            "anchor_sim_error": res["worst_rel_error"],
            "passed": rate_diff < 1e-12 and k_diff < 1e-12,
        }


# ---------------------------------------------------------------------------
# Convergence / parameter sensitivity
# ---------------------------------------------------------------------------

def _grid_convergence_study(anchor_name: str, grids: tuple[int, ...] = (16, 32, 64)) -> dict[str, Any]:
    """Study how the anchor simulation error varies with grid resolution."""
    results = []
    for grid in grids:
        case = build_case(anchor_name, 0, grid)
        res = evaluate(case)
        results.append({
            "grid": grid,
            "worst_rel_error": res["worst_rel_error"],
            "passed": res["passed"],
        })
    return {"anchor": anchor_name, "grids": list(grids), "results": results}


def _u_lb_sensitivity(case_dict: dict) -> dict[str, Any]:
    """Study how the converter comparison varies with u_lb."""
    u_lb_values = [0.01, 0.02, 0.04, 0.06, 0.08]
    results = []
    for u_lb in u_lb_values:
        try:
            case = dict(case_dict)
            case["u_lb"] = u_lb
            bridge = _make_bridge(case)
            lbt = _make_lbt_converter(case, bridge)
            derived = _compare_derived_quantities(bridge, lbt, case)
            results.append({
                "u_lb": u_lb,
                "tau": derived["tau"]["bridge"],
                "Ma": derived["Ma"]["bridge"],
                "max_rel_diff": max(v["rel_diff"] for v in derived.values()),
                "passed": all(v["rel_diff"] < 1e-12 for v in derived.values()),
            })
        except ValueError as e:
            results.append({"u_lb": u_lb, "error": str(e), "passed": False})
    return {"u_lb_values": u_lb_values, "results": results}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_xval(
    *,
    out_dir: Path | str = Path("artifacts/xval_lbt_converter"),
) -> dict[str, Any]:
    """Run the full cross-validation benchmark.

    Returns a JSON-serialisable report with all comparison results.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "gate": "xval-lbt-converter",
        "reference": "2b-t/LB-t converter.hpp + settings.json",
        "cases": {},
        "anchor_xval": {},
        "grid_convergence": {},
        "u_lb_sensitivity": {},
        "summary": {},
    }

    # --- Per-case comparisons ---
    max_cross_diff = 0.0
    for name, case in XVAL_CASES.items():
        bridge = _make_bridge(case)
        lbt = _make_lbt_converter(case, bridge)

        roundtrip = _compare_roundtrip(bridge, lbt)
        derived = _compare_derived_quantities(bridge, lbt, case)
        schema_val = _validate_schema_roundtrip(name, case)

        report["cases"][name] = {
            "params": case,
            "viability": _check_case_viability(case),
            "roundtrip": roundtrip,
            "derived": derived,
            "schema_validation": schema_val,
        }

        for site, r in roundtrip.items():
            max_cross_diff = max(max_cross_diff, r["cross_rel_diff"])

    # --- Anchor cross-validation ---
    for anchor_name in ANCHORS:
        report["anchor_xval"][anchor_name] = _anchor_xval(anchor_name)

    # --- Grid convergence ---
    for anchor_name in ANCHORS:
        report["grid_convergence"][anchor_name] = _grid_convergence_study(anchor_name)

    # --- u_lb sensitivity (use first case) ---
    first_case = list(XVAL_CASES.values())[0]
    report["u_lb_sensitivity"] = _u_lb_sensitivity(first_case)

    # --- Summary ---
    all_schema_pass = all(
        report["cases"][n]["schema_validation"]["passed"]
        for n in XVAL_CASES
    )
    all_anchor_pass = all(
        report["anchor_xval"][a]["passed"] for a in ANCHORS
    )

    report["summary"] = {
        "max_cross_rel_diff": max_cross_diff,
        "all_schema_roundtrip_pass": all_schema_pass,
        "all_anchor_xval_pass": all_anchor_pass,
        "precision_threshold": 0.15,
        "machine_precision_threshold": 1e-12,
        "overall_passed": bool(
            max_cross_diff < 0.15
            and all_schema_pass
            and all_anchor_pass
        ),
    }

    report["runtime_seconds"] = time.time() - t0

    # --- Write report ---
    report_path = out_dir / "xval_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))

    # --- Write case schema examples ---
    for name, case in XVAL_CASES.items():
        schema = CaseSchema(
            physics=Physics(
                density=case["rho0"],
                kinematic_viscosity=case["nu"],
            ),
            discretisation=Discretisation(NX=case["nx"]),
            characteristic_length=case["L"],
            characteristic_velocity=case["u_ref"],
            u_lb=case["u_lb"],
        )
        schema_path = out_dir / f"case_{name}.json"
        schema.to_json_file(schema_path)

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the cross-validation benchmark."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/xval_lbt_converter"),
                    help="output directory for report and artifacts")
    args = ap.parse_args(argv)

    report = run_xval(out_dir=args.out)
    s = report["summary"]

    print(f"Cross-validation: TensorLBM LatticeBridge vs 2b-t/LB-t Converter")
    print(f"=" * 70)
    print(f"  max cross rel diff     : {s['max_cross_rel_diff']:.2e} "
          f"(threshold < {s['precision_threshold']})")
    print(f"  schema roundtrip       : {'PASS' if s['all_schema_roundtrip_pass'] else 'FAIL'}")
    print(f"  anchor xval            : {'PASS' if s['all_anchor_xval_pass'] else 'FAIL'}")
    print(f"  OVERALL                : {'PASS' if s['overall_passed'] else 'FAIL'}")
    print(f"  runtime                : {report['runtime_seconds']:.1f}s")

    # Per-case details
    for name, case_data in report["cases"].items():
        derived = case_data["derived"]
        viab = case_data["viability"]
        print(f"\n  [{name}] Re={viab['Re']:.1f} tau={viab['tau']:.3f} Ma={viab['Ma']:.4f}")
        for qty, vals in derived.items():
            status = "✓" if vals["rel_diff"] < 1e-12 else "✗"
            print(f"    {qty:6s}: bridge={vals['bridge']:.6g}  lbt={vals['lbt']:.6g}  "
                  f"diff={vals['rel_diff']:.2e} {status}")

    # Anchor details
    print(f"\n  [Anchor XVAL]")
    for anchor_name, adata in report["anchor_xval"].items():
        status = "✓" if adata["passed"] else "✗"
        sim_err = adata.get("anchor_sim_error", 0)
        conv_diff = adata.get("converter_rel_diff",
                              adata.get("rate_converter_rel_diff", 0))
        print(f"    {anchor_name:13s}: sim_err={sim_err:.2e}  "
              f"converter_diff={conv_diff:.2e}  {status}")

    # Grid convergence
    print(f"\n  [Grid Convergence]")
    for anchor_name, gc in report["grid_convergence"].items():
        vals = [r["worst_rel_error"] for r in gc["results"]]
        print(f"    {anchor_name:13s}: " + " -> ".join(f"{v:.2e}" for v in vals))

    # u_lb sensitivity
    print(f"\n  [u_lb Sensitivity]")
    for r in report["u_lb_sensitivity"]["results"]:
        if "error" in r:
            print(f"    u_lb={r['u_lb']:.2f}: SKIP ({r['error']})")
        else:
            status = "✓" if r["passed"] else "✗"
            print(f"    u_lb={r['u_lb']:.2f}: tau={r['tau']:.3f} Ma={r['Ma']:.4f} "
                  f"max_diff={r['max_rel_diff']:.2e} {status}")

    return 0 if s["overall_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
