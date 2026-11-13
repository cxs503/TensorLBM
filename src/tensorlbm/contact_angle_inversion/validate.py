"""Validation test for contact-angle inversion.

Runs the calibration sweep, then tests the inversion at hold-out angles
to verify that the prescribed-vs-measured error is < 15%.

This is the acceptance test for XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

# Add src to path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.contact_angle_inversion import calibrate, invert, forward
from tensorlbm.contact_angle_inversion.calibration import CalibrationResult, young_prediction
from tensorlbm.contact_line.engine import (
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)
from tensorlbm.sc_surface_tension import contact_angle_from_mask, psi_exp


def measure_theta(rho, solid, rho_l=RHO_L_DEFAULT, rho_v=RHO_V_DEFAULT):
    """Measure contact angle from density field."""
    mid = 0.5 * (rho_l + rho_v)
    result = contact_angle_from_mask(rho, solid, mid=mid)
    return result


def run_validation(
    cal: CalibrationResult,
    test_angles: list[float],
    nx: int = 128,
    ny: int = 64,
    radius: float = 15.0,
    steps: int = 8000,
    sample_interval: int = 500,
    device: str = "cpu",
) -> dict:
    """Test inversion accuracy at hold-out angles.

    For each test angle:
    1. Use calibration to find f_w
    2. Run sessile drop simulation
    3. Measure equilibrium angle
    4. Compute error

    Returns dict with per-angle results and summary.
    """
    torch.set_num_threads(4)

    channel = Channel(nx=nx, ny=ny, wall_rows=2, device=device)
    solid = channel.solid_mask()
    y_wall = channel.y_wall()

    results = []
    for theta_target in test_angles:
        t0 = time.time()

        # Invert to get f_w
        fw = invert(cal, theta_target)
        if not math.isfinite(fw):
            results.append({
                "theta_target": theta_target,
                "fw": float("nan"),
                "theta_measured": float("nan"),
                "error_deg": float("nan"),
                "error_pct": float("nan"),
                "status": "inversion_failed",
                "runtime_s": round(time.time() - t0, 2),
            })
            continue

        # Run simulation
        f, _ = sessile_drop_state(channel, radius=radius, height=0.0)
        stepper = WallStepper(
            channel,
            G=5.0,
            tau=1.0,
            rho_l=RHO_L_DEFAULT,
            rho_v=RHO_V_DEFAULT,
            psi_fn=psi_exp,
            wall_fraction=fw,
        )

        theta_history = []
        for step in range(1, steps + 1):
            f = stepper.step(f)
            if step % sample_interval == 0:
                rho = stepper.density(f)
                meas = measure_theta(rho, solid)
                if meas.get("status") == "ok":
                    theta_history.append((step, meas["theta_deg"]))

        # Final measurement
        rho = stepper.density(f)
        meas = measure_theta(rho, solid)

        theta_meas = meas.get("theta_deg", float("nan"))
        error_deg = abs(theta_meas - theta_target) if math.isfinite(theta_meas) else float("nan")
        error_pct = error_deg / theta_target * 100.0 if math.isfinite(error_deg) and theta_target > 0 else float("nan")

        # Check convergence
        drift = float("nan")
        converged = False
        if len(theta_history) >= 4:
            tail = theta_history[-4:]
            drift = abs(tail[-1][1] - tail[0][1])
            converged = drift < 1.0

        status = "ok"
        if not math.isfinite(theta_meas):
            status = "no_interface"
        elif not converged:
            status = "unconverged"
        elif error_pct > 15.0:
            status = "fail_error"

        results.append({
            "theta_target": theta_target,
            "fw": fw,
            "theta_measured": theta_meas,
            "error_deg": round(error_deg, 4) if math.isfinite(error_deg) else float("nan"),
            "error_pct": round(error_pct, 4) if math.isfinite(error_pct) else float("nan"),
            "status": status,
            "converged": converged,
            "drift_deg": round(drift, 4) if math.isfinite(drift) else float("nan"),
            "n_samples": len(theta_history),
            "runtime_s": round(time.time() - t0, 2),
        })

    # Summary
    passed = sum(1 for r in results if r["status"] == "ok")
    total = len(results)
    max_error = max((r["error_pct"] for r in results if math.isfinite(r["error_pct"])), default=0.0)

    return {
        "calibration": cal.to_dict(),
        "test_results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "max_error_pct": round(max_error, 4),
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0.0,
        },
    }


def main():
    """Run full validation pipeline."""
    print("=" * 70)
    print("XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION: Validation")
    print("=" * 70)

    # Step 1: Calibration sweep
    print("\n[1/3] Running calibration sweep...")
    cal = calibrate(
        fw_values=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        nx=128,
        ny=64,
        radius=15.0,
        steps=8000,
        sample_interval=500,
        device="cpu",
        threads=4,
    )

    print(f"  Calibration complete: {len(cal.points)} points")
    print(f"  Valid points: {sum(1 for p in cal.points if p.status == 'ok')}")
    print(f"  Young slope: {cal.young_slope:.4f}")
    print(f"  Measured slope: {cal.measured_slope:.4f}")
    print(f"  Slope ratio (Huang correction): {cal.slope_ratio:.4f}")
    print(f"  Neutral wall (Young): f_w = {cal.neutral_fw_young:.4f}")
    print(f"  Neutral wall (measured): f_w = {cal.neutral_fw_measured:.4f}")

    # Step 2: Validation at hold-out angles
    print("\n[2/3] Running validation at hold-out angles...")
    # Use angles NOT in the calibration set for hold-out testing
    # The calibration covers roughly 75-180 deg, so test angles in that range
    test_angles = [80.0, 100.0, 120.0, 140.0]

    validation = run_validation(
        cal,
        test_angles,
        nx=128,
        ny=64,
        radius=15.0,
        steps=8000,
        sample_interval=500,
        device="cpu",
    )

    print("\n  Validation results:")
    print(f"  {'Target':>8} {'f_w':>8} {'Measured':>10} {'Error deg':>10} {'Error %':>8} {'Status':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")
    for r in validation["test_results"]:
        print(f"  {r['theta_target']:8.1f} {r['fw']:8.4f} {r['theta_measured']:10.2f} "
              f"{r['error_deg']:10.2f} {r['error_pct']:8.2f} {r['status']:>12}")

    print(f"\n  Summary: {validation['summary']['passed']}/{validation['summary']['total']} passed, "
          f"max error = {validation['summary']['max_error_pct']:.2f}%")

    # Step 3: Comparison with Young relation
    print("\n[3/3] Comparison with Young relation...")
    print(f"  {'Target':>8} {'Young f_w':>10} {'Young theta':>12} {'Cal f_w':>10} {'Cal theta':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    for theta_target in test_angles:
        fw_young = (cal.psi_v + math.cos(math.radians(theta_target)) * (cal.psi_l - cal.psi_v)) / cal.psi_l
        theta_young = young_prediction(fw_young, cal.psi_l, cal.psi_v)
        fw_cal = invert(cal, theta_target)
        theta_cal = forward(cal, fw_cal) if math.isfinite(fw_cal) else float("nan")
        print(f"  {theta_target:8.1f} {fw_young:10.4f} {theta_young:12.2f} "
              f"{fw_cal:10.4f} {theta_cal:12.2f}")

    # Save results
    out_dir = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_result.json"
    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Verdict
    passed = validation["summary"]["passed"]
    total = validation["summary"]["total"]
    max_err = validation["summary"]["max_error_pct"]

    print("\n" + "=" * 70)
    if passed == total and max_err < 15.0:
        print(f"VERDICT: PASS ({passed}/{total} within 15% error)")
    else:
        print(f"VERDICT: FAIL ({passed}/{total} passed, max error = {max_err:.2f}%)")
    print("=" * 70)

    return 0 if (passed == total and max_err < 15.0) else 1


if __name__ == "__main__":
    sys.exit(main())
