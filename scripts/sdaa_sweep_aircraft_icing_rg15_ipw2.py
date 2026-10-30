#!/usr/bin/env python3
"""SDAA sweep for aircraft_icing_rg15_ipw2 benchmark.

Verifies that the TensorLBM aircraft icing module runs correctly on
Hygon SDAA accelerators and produces results consistent with the
CUDA reference (result.json).

Usage:
    python scripts/sdaa_sweep_aircraft_icing_rg15_ipw2.py [--quick]

The --quick flag runs a reduced-step validation (~30 s) instead of the
full production run (~642 s for all 3 cases).

Outputs:
    artifacts/sdaa_aircraft_icing_rg15_ipw2_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "pending" / "aircraft_icing_rg15_ipw2"
ARTIFACTS = REPO / "artifacts"
RESULT_JSON = CASE_DIR / "result.json"

# ---------------------------------------------------------------------------
# SDAA device probe
# ---------------------------------------------------------------------------

def probe_sdaa() -> dict:
    """Probe SDAA device and return evidence dict."""
    evidence = {
        "torch_version": torch.__version__,
        "sdaa_available": torch.sdaa.is_available(),
        "sdaa_device_count": torch.sdaa.device_count() if torch.sdaa.is_available() else 0,
    }
    if torch.sdaa.is_available():
        evidence["sdaa_device_name"] = torch.sdaa.get_device_name(0)
        props = torch.sdaa.get_device_properties(0)
        evidence["sdaa_device_properties"] = {
            "name": props.name,
            "total_memory_miB": props.total_memory,
        }
        # Test tensor creation
        dev = torch.device("sdaa:0")
        x = torch.zeros(100, 100, device=dev)
        x = x + 1.0
        evidence["sdaa_tensor_test"] = bool((x == 1).all().item())
    return evidence


# ---------------------------------------------------------------------------
# Geometry helpers (copied from _b_common.py to be self-contained)
# ---------------------------------------------------------------------------

def parse_dat(path: Path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            rows.append((x, y))
    pts = np.asarray(rows, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 10:
        raise ValueError(f"bad dat parse: {pts.shape}")
    return pts


def make_rg15_mask_fn(pts: np.ndarray):
    from matplotlib.path import Path as MplPath

    poly = np.vstack([pts, pts[:1]])
    path = MplPath(poly)

    def rg15_mask_2d(nx, ny, chord, aoa_deg=4.0, cx=None, cy=None, device="cpu"):
        if cx is None:
            cx = nx / 3.0
        if cy is None:
            cy = ny / 2.0
        dev = torch.device(device)
        xx = torch.arange(nx, device=dev, dtype=torch.float32)
        yy = torch.arange(ny, device=dev, dtype=torch.float32)
        gx, gy = torch.meshgrid(xx, yy, indexing="xy")
        dx = gx - cx
        dy = gy - cy
        aoa = math.radians(aoa_deg)
        cos_a, sin_a = math.cos(aoa), math.sin(aoa)
        xr = ((dx * cos_a - dy * sin_a) / chord).cpu().numpy().reshape(-1)
        yr = ((dx * sin_a + dy * cos_a) / chord).cpu().numpy().reshape(-1)
        inside = path.contains_points(np.column_stack([xr, yr]))
        mask = torch.from_numpy(inside.reshape(ny, nx))
        return mask.to(dev)

    return rg15_mask_2d


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def air_props(t_c: float, p_pa: float = 101300.0) -> dict:
    t_k = t_c + 273.15
    rho = p_pa / (287.05 * t_k)
    mu = 1.716e-5 * (t_k / 273.15) ** 1.5 * (273.15 + 110.4) / (t_k + 110.4)
    return {"rho_air": rho, "mu_air": mu, "t_k": t_k}


def le_diameter_phys(pts: np.ndarray, chord_phys: float) -> float:
    r = np.hypot(pts[:, 0], pts[:, 1])
    sel = pts[r < 0.012]
    if len(sel) < 5:
        sel = pts[np.argsort(r)[:20]]
    a_mat = np.column_stack([2 * sel[:, 0], 2 * sel[:, 1], np.ones(len(sel))])
    rhs = sel[:, 0] ** 2 + sel[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    xc, yc = sol[0], sol[1]
    return 2.0 * float(math.hypot(xc, yc)) * chord_phys


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

CONDITIONS = {
    "3.1_glaze": dict(t_static_c=-2.0, re_phys=5.7e5),
    "3.2_mixed": dict(t_static_c=-4.0, re_phys=5.8e5),
    "3.3_rime": dict(t_static_c=-10.0, re_phys=6.0e5),
}
COMMON_PHYS = dict(
    chord_phys=0.30, v_inf=25.0, aoa_deg=4.0, lwc=0.44e-3, mvd=24.0e-6,
    t_exposure=1200.0, p_static=101300.0, span=0.58, rh=0.975,
)
LATTICE = dict(
    nx=320, ny=160, chord_frac=0.4, u_in=0.05, collision="cumulant",
    c_s=0.1, droplet_phase="eulerian", eulerian_scheme="donor2",
    rime_density_mode="macklin", beta_window_mode="clean",
    warmup_steps=6000, steps=3000,
)


def build_cfg(case: str, t_exposure: float, steps: int, warmup: int,
              device: str = "sdaa:0", **overrides):
    sys.path.insert(0, str(REPO / "src"))
    import tensorlbm.aircraft_icing as ai

    c = CONDITIONS[case]
    ap = air_props(c["t_static_c"], COMMON_PHYS["p_static"])
    pts = parse_dat(CASE_DIR / "reference" / "RG-15_c0.3_tgap2mm.dat")
    le_d = le_diameter_phys(pts, COMMON_PHYS["chord_phys"])

    kw = dict(
        nx=LATTICE["nx"], ny=LATTICE["ny"], chord_frac=LATTICE["chord_frac"],
        u_in=LATTICE["u_in"], aoa_deg=COMMON_PHYS["aoa_deg"],
        chord_phys=COMMON_PHYS["chord_phys"], v_inf=COMMON_PHYS["v_inf"],
        lwc=COMMON_PHYS["lwc"], mvd=COMMON_PHYS["mvd"],
        t_static_c=c["t_static_c"], t_exposure=t_exposure,
        p_static=COMMON_PHYS["p_static"], rh=COMMON_PHYS["rh"],
        rho_air=ap["rho_air"], mu_air=ap["mu_air"],
        collision=LATTICE["collision"], c_s=LATTICE["c_s"],
        re_lu_target=c["re_phys"], droplet_phase=LATTICE["droplet_phase"],
        eulerian_scheme=LATTICE["eulerian_scheme"],
        rime_density_mode=LATTICE["rime_density_mode"],
        beta_window_mode=LATTICE["beta_window_mode"],
        steps=steps, warmup_steps=warmup, seed=0, device=device,
        log_every=max(1, steps // 6), thermo_model="messinger",
        le_diameter=le_d, naca_t=0.091,
    )
    kw.update(overrides)
    return ai.IcingConfig(**kw)


# ---------------------------------------------------------------------------
# Run one case
# ---------------------------------------------------------------------------

def run_one(case: str, device: str, shots: int = 1, steps: int = 3000,
            warmup: int = 6000, log_fn=None):
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(CASE_DIR))
    from _b_common import install_rg15
    from tensorlbm import aircraft_icing as ai

    install_rg15()
    cfg = build_cfg(case, 1200.0, steps, warmup, device=device)

    t0 = time.perf_counter()
    result = ai.run_glaze_icing(cfg, shots=shots, log=log_fn or (lambda m: None))
    wall_s = time.perf_counter() - t0

    bc = result["beta_curve"]
    beta_pk = float(bc["beta"].max()) if len(bc["beta"]) else None

    return {
        "case": case,
        "device": device,
        "shots": shots,
        "steps": steps,
        "warmup": warmup,
        "wall_s": wall_s,
        "beta_pk": beta_pk,
        "frozen_kg": result["audit"]["frozen"],
        "closure_error": result["audit"]["closure_error"],
        "n_f_stag": result["metrics"].get("n_f_stag"),
        "n_ice_cells": result["metrics"].get("n_ice_cells"),
        "stag_ice_mm": result["metrics"].get("stag_ice_thickness_m", 0) * 1000,
        "rendered_over_frozen": result["metrics"].get("rendered_over_frozen"),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

REFERENCE = {
    "3.1_glaze": {
        "beta_pk": 0.7815262675285339,
        "frozen_kg": 0.0004180652990734247,
        "n_f_stag": 0.6485638610369927,
        "stag_ice_mm": 0.9007498048130704,
        "final_ice_cells": 36,
    },
    "3.2_mixed": {
        "beta_pk": 0.7865551114082336,
        "frozen_kg": 0.0004138666824163865,
        "n_f_stag": 1.0,
        "stag_ice_mm": 1.2720169903794738,
        "final_ice_cells": 35,
    },
    "3.3_rime": {
        "beta_pk": 0.7880398631095886,
        "frozen_kg": 0.00042884286382963675,
        "n_f_stag": 1.0,
        "stag_ice_mm": 1.0144744679569833,
        "final_ice_cells": 36,
    },
}


def compare(sim: dict, ref: dict, threshold_pct: float = 3.0) -> dict:
    """Compare simulation results against reference."""
    comparisons = {}
    for key in ["beta_pk", "frozen_kg", "n_f_stag", "stag_ice_mm"]:
        sim_val = sim.get(key)
        ref_val = ref.get(key)
        if sim_val is None or ref_val is None:
            comparisons[key] = {"sim": sim_val, "ref": ref_val, "error_pct": None, "pass": None}
            continue
        if ref_val == 0:
            err = 0.0 if sim_val == 0 else float("inf")
        else:
            err = abs(sim_val - ref_val) / abs(ref_val) * 100
        comparisons[key] = {
            "sim": sim_val,
            "ref": ref_val,
            "error_pct": round(err, 6),
            "pass": err < threshold_pct,
        }
    # Integer comparison for ice cells
    sim_cells = sim.get("n_ice_cells")
    ref_cells = ref.get("final_ice_cells")
    if sim_cells is not None and ref_cells is not None:
        comparisons["n_ice_cells"] = {
            "sim": sim_cells,
            "ref": ref_cells,
            "match": sim_cells == ref_cells,
        }
    return comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SDAA sweep: aircraft_icing_rg15_ipw2")
    ap.add_argument("--quick", action="store_true",
                    help="Run reduced-step validation (~30 s) instead of full production")
    ap.add_argument("--case", default="3.3_rime",
                    choices=["3.1_glaze", "3.2_mixed", "3.3_rime", "all"],
                    help="Case to run (default: 3.3_rime)")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="Error threshold in percent (default: 3.0)")
    args = ap.parse_args()

    # 1. SDAA device probe
    print("=" * 60)
    print("SDAA Device Probe")
    print("=" * 60)
    sdaa_evidence = probe_sdaa()
    for k, v in sdaa_evidence.items():
        print(f"  {k}: {v}")
    assert sdaa_evidence["sdaa_available"], "SDAA not available!"
    assert sdaa_evidence.get("sdaa_tensor_test"), "SDAA tensor test failed!"

    # 2. Load reference
    with open(RESULT_JSON) as f:
        ref_data = json.load(f)

    # 3. Run validation
    print("\n" + "=" * 60)
    print("SDAA Validation Run")
    print("=" * 60)

    device = "sdaa:0"
    cases_to_run = ["3.1_glaze", "3.2_mixed", "3.3_rime"] if args.case == "all" else [args.case]

    if args.quick:
        # Quick validation: reduced steps for speed
        steps = 500
        warmup = 1000
        shots = 1
        print(f"  Mode: QUICK (steps={steps}, warmup={warmup}, shots={shots})")
    else:
        steps = 3000
        warmup = 6000
        shots = 5
        print(f"  Mode: FULL (steps={steps}, warmup={warmup}, shots={shots})")

    results = {}
    for case in cases_to_run:
        print(f"\n--- Running {case} on {device} ---")
        try:
            r = run_one(case, device, shots=shots, steps=steps, warmup=warmup)
            results[case] = r
            print(f"  wall_s={r['wall_s']:.1f}, beta_pk={r['beta_pk']:.6f}, "
                  f"frozen={r['frozen_kg']:.4e}, n_f_stag={r['n_f_stag']}, "
                  f"ice_cells={r['n_ice_cells']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[case] = {"error": str(e)}

    # 4. Compare with reference
    print("\n" + "=" * 60)
    print("Comparison with CUDA Reference")
    print("=" * 60)

    all_pass = True
    comparison_results = {}
    for case in cases_to_run:
        if "error" in results.get(case, {}):
            comparison_results[case] = {"status": "ERROR", "detail": results[case]["error"]}
            all_pass = False
            continue

        ref = REFERENCE.get(case, {})
        cmp = compare(results[case], ref, args.threshold)
        comparison_results[case] = cmp

        print(f"\n  {case}:")
        for key, val in cmp.items():
            if "error_pct" in val:
                status = "PASS" if val["pass"] else "FAIL"
                print(f"    {key}: sim={val['sim']:.8f} ref={val['ref']:.8f} "
                      f"err={val['error_pct']:.4f}% [{status}]")
                if not val["pass"]:
                    all_pass = False
            elif "match" in val:
                status = "PASS" if val["match"] else "FAIL"
                print(f"    {key}: sim={val['sim']} ref={val['ref']} [{status}]")
                if not val["match"]:
                    all_pass = False

    # 5. Generate evidence file
    evidence = {
        "case": "aircraft_icing_rg15_ipw2",
        "sdaa_evidence": sdaa_evidence,
        "run_config": {
            "device": device,
            "mode": "quick" if args.quick else "full",
            "steps": steps,
            "warmup": warmup,
            "shots": shots,
            "threshold_pct": args.threshold,
        },
        "results": results,
        "comparison": comparison_results,
        "verdict": "PASS" if all_pass else "HONEST-FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": _get_git_hash(),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    evidence_path = ARTIFACTS / "sdaa_aircraft_icing_rg15_ipw2_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=float)
    print(f"\nEvidence written to {evidence_path}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SDAA device: {sdaa_evidence.get('sdaa_device_name', 'N/A')}")
    print(f"  Cases run: {len(results)}")
    print(f"  Verdict: {evidence['verdict']}")
    if not all_pass:
        print("  NOTE: Quick mode uses reduced steps; beta_pk and frozen mass")
        print("        will differ from full production reference. Full run")
        print("        required for production-grade verification.")


def _get_git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
