#!/usr/bin/env python3
"""Final demonstration: fix the geodesic angle convention issue.

The R3 audit found that the geodesic estimator returns the supplement angle
for LBM simulation fields. The root cause is that the synthetic caps used for
calibration follow a different geometric convention than the actual LBM fields.

Round 5 (m23072) established the exact closed form for a sessile cap:
    z_c = -R cos(theta),  h = R(1 - cos theta),  a = R sin(theta)
for EVERY theta in (0, 180), so
    a/h = cot(theta/2)  =>  theta = 2 atan(h/a) = 2 atan(1/(a/h))
with no branch test and no caller-supplied convention.

This script shows:
1. Original (round-3) synthetic cap convention: y_c = wall_y + R cos theta
   (true angle = 180 - theta)
2. Physical sessile cap convention: y_c = wall_y - R cos theta
   (true angle = theta for every theta in (0, 180))
3. The round-5 fix: theta = 2 atan(h/a) recovers the true angle on every
   physical sessile cap, with no branch test.
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
    """Measure apex height h and basal half-width a."""
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

    The true contact angle of this template is ``180 - theta_deg``.
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
    """Physical sessile cap: y_c = wall_y - R cos theta.

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


def geodesic_angle_physical(psi, xi, frac=0.5, wall_y=-0.5):
    """Geodesic angle using the round-5 closed form.

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

    if len(ys) >= 4:
        ys_a, xs_a = np.asarray(ys), np.asarray(xs)
        near = ys_a <= 3.0 * xi
        if int(near.sum()) < 4:
            near = np.ones_like(ys_a, dtype=bool)
        deg = 3 if int(near.sum()) >= 5 else (2 if int(near.sum()) >= 4 else 1)
        coef = np.polyfit(ys_a[near], xs_a[near], deg)
        a = float(np.polyval(coef, wall_y))
        if a > 0.0 and height > 0.0:
            # round-5 closed form: theta = 2 atan(h/a) for ALL angles
            return math.degrees(2.0 * math.atan(height / a))
    return float("nan")


def main():
    """Demonstrate the fix."""
    print("=" * 90)
    print("Geodesic angle convention fix for XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS")
    print("=" * 90)

    test_angles = [45.0, 70.0, 90.0, 110.0, 135.0]
    radius = 16.0
    xi = 3.0

    print("\n1. Original (round-3) synthetic cap with original formula (2 atan(a/h)):")
    print("-" * 90)
    print(f"{'theta':>8} {'theta_meas':>12} {'true':>8} {'error':>10}")
    print("-" * 90)
    for theta_deg in test_angles:
        psi = synthetic_cap_original(theta_deg, radius, xi)
        a, h = measure_a_h(psi, xi)
        ratio = a / h if h > 0 else float("nan")
        theta_raw = math.degrees(2.0 * math.atan(ratio)) if ratio > 0 else float("nan")
        true = 180.0 - theta_deg
        error = abs(theta_raw - true) if math.isfinite(theta_raw) else float("nan")
        print(f"{theta_deg:8.1f} {theta_raw:12.2f} {true:8.1f} {error:10.2f}")

    print("\n2. Physical sessile cap with round-5 closed form (2 atan(h/a)):")
    print("-" * 90)
    print(f"{'theta':>8} {'a/h':>8} {'theta_meas':>12} {'error':>10}")
    print("-" * 90)
    errors = []
    for theta_deg in test_angles:
        psi = synthetic_cap_physical(theta_deg, radius, xi)
        theta_meas = geodesic_angle_physical(psi, xi)
        error = abs(theta_meas - theta_deg) if math.isfinite(theta_meas) else float("nan")
        a, h = measure_a_h(psi, xi)
        ratio = a / h if h > 0 else float("nan")
        errors.append(error)
        print(f"{theta_deg:8.1f} {ratio:8.3f} {theta_meas:12.2f} {error:10.2f}")

    print("-" * 90)
    print(f"Max error: {max(errors):.2f} deg")
    print(f"Mean error: {sum(errors)/len(errors):.2f} deg")

    print("\n" + "=" * 90)
    print("Summary:")
    print("- The round-3 template (y_c = wall_y + R cos theta) has true angle")
    print("  180 - theta; the original formula 2 atan(a/h) returns the supplement.")
    print("- The physical sessile template (y_c = wall_y - R cos theta) has true")
    print("  angle theta for every theta in (0, 180).")
    print("- The round-5 closed form theta = 2 atan(h/a) recovers the true angle")
    print("  on every physical sessile cap, with no branch test and no")
    print("  caller-supplied convention.")
    print("=" * 90)


if __name__ == "__main__":
    main()
