#!/usr/bin/env python3
"""
Analyze the geodesic angle convention issue and implement the correct fix.

The R3 audit found that the geodesic estimator returns the supplement for LBM fields.
This script analyzes the root cause and implements the correct fix.

Round 5 (m23072) established the exact closed form for a sessile cap:
    z_c = -R cos(theta),  h = R(1 - cos theta),  a = R sin(theta)
for EVERY theta in (0, 180), so
    a/h = cot(theta/2)  =>  theta = 2 atan(h/a) = 2 atan(1/(a/h))
with no branch test and no caller-supplied convention.

The round-4 ``a/h > 1`` conditional and ``lbm_convention`` boolean were the
supplement-mirror (``2 atan(a/h) = 180 - 2 atan(h/a)``) and are retired.
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
    """Original (round-3) convention: y_c = wall_y + R cos theta.

    The true contact angle of this template is ``180 - theta_deg`` (the
    round-3 mislabelled cap).
    """
    th = math.radians(theta_deg)
    y_c = wall_y + radius * math.cos(th)
    nx = int(2 * radius + 41)
    ny = int(2 * radius + 30)
    x_c = (nx - 1) / 2.0
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    return -np.tanh((r - radius) / xi)


def synthetic_cap_physical(theta_deg, radius=16.0, xi=3.0, wall_y=-0.5):
    """Physical sessile convention: y_c = wall_y - R cos theta.

    The true contact angle is ``theta_deg`` for every ``theta`` in ``(0, 180)``.
    """
    th = math.radians(theta_deg)
    y_c = wall_y - radius * math.cos(th)
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

    print("\nOriginal (round-3) synthetic cap (y_c = wall_y + R cos theta):")
    print(f"{'theta':>8} {'a':>8} {'h':>8} {'a/h':>8} {'2 atan(a/h)':>12} {'2 atan(h/a)':>12} {'true':>8}")
    print("-" * 90)

    for theta_deg in test_angles:
        psi = synthetic_cap_original(theta_deg, radius, xi)
        a, h = measure_a_h(psi, xi)
        ratio = a / h if h > 0 else float("nan")
        theta_raw = math.degrees(2.0 * math.atan(ratio)) if ratio > 0 else float("nan")
        theta_corr = math.degrees(2.0 * math.atan(h / a)) if a > 0 and h > 0 else float("nan")
        # The round-3 template's true angle is 180 - theta_deg
        true = 180.0 - theta_deg
        print(f"{theta_deg:8.1f} {a:8.2f} {h:8.2f} {ratio:8.3f} {theta_raw:12.2f} {theta_corr:12.2f} {true:8.1f}")

    print("\nPhysical sessile cap (y_c = wall_y - R cos theta):")
    print(f"{'theta':>8} {'a':>8} {'h':>8} {'a/h':>8} {'2 atan(a/h)':>12} {'2 atan(h/a)':>12} {'true':>8}")
    print("-" * 90)

    for theta_deg in test_angles:
        psi = synthetic_cap_physical(theta_deg, radius, xi)
        a, h = measure_a_h(psi, xi)
        ratio = a / h if h > 0 else float("nan")
        theta_raw = math.degrees(2.0 * math.atan(ratio)) if ratio > 0 else float("nan")
        theta_corr = math.degrees(2.0 * math.atan(h / a)) if a > 0 and h > 0 else float("nan")
        print(f"{theta_deg:8.1f} {a:8.2f} {h:8.2f} {ratio:8.3f} {theta_raw:12.2f} {theta_corr:12.2f} {theta_deg:8.1f}")

    print("\n" + "=" * 90)
    print("Key insight:")
    print("- Round-3 template (y_c = wall_y + R cos theta): 2 atan(a/h) = 180 - theta")
    print("  (returns the supplement; the true angle is 180 - theta)")
    print("- Physical template (y_c = wall_y - R cos theta): 2 atan(h/a) = theta")
    print("  (returns the true angle for every theta in (0, 180))")
    print("- The round-5 closed form theta = 2 atan(h/a) needs NO branch test")
    print("=" * 90)


def geodesic_angle_physical(psi, xi, frac=0.5, wall_y=-0.5):
    """Geodesic angle estimator using the round-5 closed form.

    ``theta = 2 atan(h/a)`` holds for every sessile cap; no branch test.
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
        "h_over_a": float("nan"),
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
            out["h_over_a"] = height / a
            # round-5 closed form: theta = 2 atan(h/a) for ALL angles
            out["theta_geodesic_deg"] = math.degrees(2.0 * math.atan(height / a))
    return out


def main():
    analyze_conventions()

    print("\n" + "=" * 90)
    print("Testing round-5 estimator on physical synthetic caps")
    print("=" * 90)

    test_angles = [45.0, 70.0, 90.0, 110.0, 135.0]
    radius = 16.0
    xi = 3.0

    print(f"\n{'theta_true':>10} {'theta_meas':>14} {'error':>10} {'a/h':>8}")
    print("-" * 60)

    for theta_deg in test_angles:
        psi = synthetic_cap_physical(theta_deg, radius, xi)
        meas = geodesic_angle_physical(psi, xi)
        theta_meas = meas["theta_geodesic_deg"]
        error = abs(theta_meas - theta_deg) if math.isfinite(theta_meas) else float("nan")
        ratio = meas["a_over_h"]
        print(f"{theta_deg:10.1f} {theta_meas:14.2f} {error:10.2f} {ratio:8.3f}")

    print("\n" + "=" * 90)
    print("Conclusion:")
    print("- The round-5 closed form theta = 2 atan(h/a) recovers the true angle")
    print("  on every physical sessile cap, with no branch test.")
    print("- The round-4 'a/h > 1' conditional and 'lbm_convention' boolean were")
    print("  the supplement-mirror and are retired.")
    print("=" * 90)


if __name__ == "__main__":
    main()
