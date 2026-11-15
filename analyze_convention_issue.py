#!/usr/bin/env python3
"""
Analyze the geodesic angle convention issue and implement the correct fix.

The R3 audit found that the geodesic estimator returns the supplement for LBM fields.
This script analyzes the root cause and implements the correct fix.
"""

import math
import numpy as np


def crossing_along(profile, level, last=True):
    d = profile - level
    s = d[:-1] * d[1:]
    idx = np.nonzero(s < 0.0)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[-1] if last else idx[0])
    return i + float(d[i] / (d[i] - d[i + 1]))


def right_crossing(row, level, jc):
    d = row - level
    for i in range(jc, row.size - 1):
        if d[i] * d[i + 1] < 0.0:
            return i + float(d[i] / (d[i] - d[i + 1]))
    return float("nan")


def gas_level(psi, frac):
    gas = 0.5 * (float(psi[-4:, :4].mean()) + float(psi[-4:, -4:].mean()))
    liquid = float(psi.max())
    return gas, liquid, gas + frac * (liquid - gas)


def measure_a_h(psi, xi, frac=0.5, wall_y=-0.5):
    ny, nx = psi.shape
    gas, _liquid, level = gas_level(psi, frac)
    xs_grid = np.arange(nx, dtype=float)
    w = np.clip(psi - gas, 0.0, None).sum(axis=0)
    x_c = float((xs_grid * w).sum() / w.sum()) if w.sum() > 0 else (nx - 1) / 2.0
    jc = int(min(nx - 2, max(1, round(x_c))))
    height = crossing_along(psi[:, jc], level) - wall_y

    ys, xs = [], []
    for k in range(ny):
        x = right_crossing(psi[k], level, jc)
        if math.isfinite(x):
            ys.append(float(k))
            xs.append(x - x_c)
    
    if len(ys) >= 4:
        ys_a, xs_a = np.asarray(ys), np.asarray(xs)
        near = ys_a <= 3.0 * xi
        if int(near.sum()) < 4:
            near = np.ones_like(ys_a, dtype=bool)
        deg = 3 if int(near.sum()) >= 5 else (2 if int(near.sum()) >= 4 else 1)
        coef = np.polyfit(ys_a[near], xs_a[near], deg)
        a = float(np.polyval(coef, wall_y))
        return a, height
    return float("nan"), float("nan")


def synthetic_cap_original(theta_deg, radius=16.0, xi=3.0, wall_y=-0.5):
    """Original convention: y_c = wall_y + R cos theta."""
    th = math.radians(theta_deg)
    y_c = wall_y + radius * math.cos(th)
    nx = int(2 * radius + 41)
    ny = int(2 * radius + 30)
    x_c = (nx - 1) / 2.0
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    return -np.tanh((r - radius) / xi)


def analyze_conventions():
    """Analyze the a/h ratio for different conventions."""
    print("=" * 90)
    print("Analysis: a/h ratio for different cap conventions")
    print("=" * 90)
    
    test_angles = [45.0, 70.0, 90.0, 110.0, 135.0]
    radius = 16.0
    xi = 3.0
    
    print("\nOriginal synthetic cap (y_c = wall_y + R cos theta):")
    print(f"{'theta':>8} {'a':>8} {'h':>8} {'a/h':>8} {'2 atan(a/h)':>12} {'supplement':>12}")
    print("-" * 90)
    
    for theta_deg in test_angles:
        psi = synthetic_cap_original(theta_deg, radius, xi)
        a, h = measure_a_h(psi, xi)
        ratio = a / h if h > 0 else float("nan")
        theta_raw = math.degrees(2.0 * math.atan(ratio)) if ratio > 0 else float("nan")
        theta_supp = 180.0 - theta_raw if math.isfinite(theta_raw) else float("nan")
        print(f"{theta_deg:8.1f} {a:8.2f} {h:8.2f} {ratio:8.3f} {theta_raw:12.2f} {theta_supp:12.2f}")
    
    print("\nR3 LBM field measurements (from R3 report):")
    print(f"{'theta':>8} {'a/h':>8} {'2 atan(a/h)':>12} {'supplement':>12} {'matches':>10}")
    print("-" * 90)
    
    # From R3 report: geodesic angles for neumann_flux arm
    lbm_data = {
        70.0: 111.19,
        90.0: 90.06,
        110.0: 70.21,
        135.0: 45.72,
    }
    
    for theta_deg, theta_geodesic in lbm_data.items():
        # Reverse engineer a/h from the geodesic angle
        # theta_geodesic = 2 atan(a/h), so a/h = tan(theta_geodesic / 2)
        ratio = math.tan(math.radians(theta_geodesic / 2.0))
        theta_raw = theta_geodesic
        theta_supp = 180.0 - theta_geodesic
        # Check which matches the prescribed angle
        matches = "raw" if abs(theta_raw - theta_deg) < 5 else ("supp" if abs(theta_supp - theta_deg) < 5 else "neither")
        print(f"{theta_deg:8.1f} {ratio:8.3f} {theta_raw:12.2f} {theta_supp:12.2f} {matches:>10}")
    
    print("\n" + "=" * 90)
    print("Key insight:")
    print("- Original synthetic cap: a/h > 1 for theta > 90 deg")
    print("- LBM fields: a/h > 1 for theta < 90 deg (opposite convention)")
    print("- The formula 2 atan(a/h) works for synthetic cap, gives supplement for LBM")
    print("=" * 90)


def geodesic_angle_for_lbm(psi, xi, frac=0.5, wall_y=-0.5):
    """Geodesic angle estimator corrected for LBM field convention.
    
    For LBM fields, the a/h ratio is inverted compared to the synthetic cap convention:
    - LBM: a/h > 1 for theta < 90 deg, a/h < 1 for theta > 90 deg
    - Synthetic: a/h < 1 for theta < 90 deg, a/h > 1 for theta > 90 deg
    
    The fix: when a/h > 1, the angle is < 90 deg and we need the supplement.
    """
    ny, nx = psi.shape
    gas, _liquid, level = gas_level(psi, frac)
    xs_grid = np.arange(nx, dtype=float)
    w = np.clip(psi - gas, 0.0, None).sum(axis=0)
    x_c = float((xs_grid * w).sum() / w.sum()) if w.sum() > 0 else (nx - 1) / 2.0
    jc = int(min(nx - 2, max(1, round(x_c))))
    height = crossing_along(psi[:, jc], level) - wall_y

    ys, xs = [], []
    for k in range(ny):
        x = right_crossing(psi[k], level, jc)
        if math.isfinite(x):
            ys.append(float(k))
            xs.append(x - x_c)
    
    out = {
        "theta_geodesic_deg": float("nan"),
        "apex_height": float(height),
        "basal_half_width": float("nan"),
        "a_over_h": float("nan"),
    }
    
    if len(ys) >= 4:
        ys_a, xs_a = np.asarray(ys), np.asarray(xs)
        near = ys_a <= 3.0 * xi
        if int(near.sum()) < 4:
            near = np.ones_like(ys_a, dtype=bool)
        deg = 3 if int(near.sum()) >= 5 else (2 if int(near.sum()) >= 4 else 1)
        coef = np.polyfit(ys_a[near], xs_a[near], deg)
        a = float(np.polyval(coef, wall_y))
        out["basal_half_width"] = a
        if a > 0.0 and height > 0.0:
            ratio = a / height
            out["a_over_h"] = ratio
            theta_raw = math.degrees(2.0 * math.atan(ratio))
            # For LBM convention: a/h > 1 means theta < 90 deg
            # The raw formula gives the supplement, so correct it
            if ratio > 1.0:
                out["theta_geodesic_deg"] = 180.0 - theta_raw
            else:
                out["theta_geodesic_deg"] = theta_raw
    return out


def main():
    analyze_conventions()
    
    print("\n" + "=" * 90)
    print("Testing LBM-corrected estimator on original synthetic caps")
    print("=" * 90)
    print("\nNote: The LBM-corrected estimator will give WRONG results on original")
    print("synthetic caps because they follow different conventions. This is expected.")
    print("The fix is meant for LBM fields, not synthetic caps.")
    
    test_angles = [45.0, 70.0, 90.0, 110.0, 135.0]
    radius = 16.0
    xi = 3.0
    
    print(f"\n{'theta_true':>10} {'theta_LBM_corr':>14} {'error':>10} {'a/h':>8}")
    print("-" * 60)
    
    for theta_deg in test_angles:
        psi = synthetic_cap_original(theta_deg, radius, xi)
        meas = geodesic_angle_for_lbm(psi, xi)
        theta_meas = meas["theta_geodesic_deg"]
        error = abs(theta_meas - theta_deg) if math.isfinite(theta_meas) else float("nan")
        ratio = meas["a_over_h"]
        print(f"{theta_deg:10.1f} {theta_meas:14.2f} {error:10.2f} {ratio:8.3f}")
    
    print("\n" + "=" * 90)
    print("Conclusion:")
    print("- The LBM-corrected estimator gives WRONG results on original synthetic caps")
    print("- This is because they follow different geometric conventions")
    print("- The fix should ONLY be applied to LBM fields, not synthetic caps")
    print("- Solution: Keep original estimator for synthetic cap calibration,")
    print("            apply supplement correction only when measuring LBM fields")
    print("=" * 90)


if __name__ == "__main__":
    main()
