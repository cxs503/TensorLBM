"""Prototype: robust sessile-drop contact-angle estimators, validated on
analytic tanh caps of known angle (independent of any boundary condition)."""
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from tensorlbm.wetting.metrics import interface_contour, fit_circle, spherical_cap_inversion


def analytic_cap(ny, nx, theta_deg, R=20.0, xi=2.0, y_wall=0.5):
    th = math.radians(theta_deg)
    x0 = (nx - 1) / 2.0
    yc = y_wall - R * math.cos(th)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.hypot(x - x0, y - yc)
    phi = np.tanh((R - r) / xi)
    phi[y < y_wall] = -1.0
    return phi


def theta_sym_circle(phi, y_wall, level=0.0, trim=0.25):
    """Circle fit with the symmetry axis pinned at the contour centroid."""
    pts = interface_contour(phi, y_wall, level)
    if len(pts) < 8:
        return float("nan"), {}
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    span = x_max - x_min
    keep = (pts[:, 0] > x_min + trim * span) & (pts[:, 0] < x_max - trim * span)
    use = pts[keep] if keep.sum() >= 6 else pts
    xc = 0.5 * (x_min + x_max)
    # y = yc + sqrt(R^2-(x-xc)^2)  ->  (y-yc)^2 + (x-xc)^2 = R^2, linear in (yc, R^2-yc^2)
    x = use[:, 0] - xc
    y = use[:, 1]
    A = np.column_stack([-2.0 * y, np.ones_like(y)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    yc = sol[0]
    R = math.sqrt(max(sol[1] + yc * yc, 1e-30))
    arg = max(-1.0, min(1.0, (y_wall - yc) / R))
    rms = float(np.sqrt(np.mean((np.hypot(use[:, 0] - xc, y - yc) - R) ** 2)))
    return math.degrees(math.acos(arg)), {"yc": yc, "R": R, "rms": rms, "n": int(len(use))}


def theta_tangent(phi, y_wall, level=0.0, frac=0.35):
    """Local slope of the interface near one contact line, averaged over both."""
    pts = interface_contour(phi, y_wall, level)
    if len(pts) < 6:
        return float("nan"), {}
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    span = x_max - x_min
    out = []
    for lo, hi, sgn in ((x_min, x_min + frac * span, +1.0), (x_max - frac * span, x_max, -1.0)):
        m = (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
        if m.sum() < 4:
            continue
        xs, ys = pts[m, 0], pts[m, 1]
        # y(x) ~ a0 + a1 (x - x_line) + a2 (x - x_line)^2 ; slope at the wall end
        x_line = x_min if sgn > 0 else x_max
        a = np.polyfit(xs - x_line, ys, 2)
        slope = a[1]
        out.append(math.degrees(math.atan(sgn * slope)))
    if not out:
        return float("nan"), {}
    return float(np.mean(out)), {"per_side": out}


def report(tag, phi, y_wall, theta_true=None):
    sym, info = theta_sym_circle(phi, y_wall)
    tan, tinfo = theta_tangent(phi, y_wall)
    try:
        cap = spherical_cap_inversion(phi, y_wall, level=0.0)
        capv = cap["theta_deg"]
    except Exception as exc:
        capv = float("nan")
    line = f"{tag:26s} sym={sym:7.2f} tan={tan:7.2f} cap={capv:7.2f}"
    if theta_true is not None:
        line += f"  (true {theta_true:5.1f}; err sym {100*(sym-theta_true)/theta_true:+6.2f}%)"
    print(line, " rms=%.3f n=%d" % (info.get("rms", float("nan")), info.get("n", 0)))
    return sym, tan, capv


if __name__ == "__main__":
    print("--- analytic tanh caps (no BC involved) ---")
    for th in (30, 45, 60, 90, 105, 120, 135, 150):
        for R, xi in ((20.0, 2.0), (14.0, 2.0), (20.0, 3.0)):
            phi = analytic_cap(96, 128, th, R=R, xi=xi)
            report(f"cap th={th} R={R} xi={xi}", phi, 0.5, theta_true=th)
    print("--- saved colour-gradient fields (R4 runs, unconverged) ---")
    for arm, target in (("ctl60", 90.0), ("gk30", 30.0), ("gk60", 60.0), ("gk90", 90.0)):
        phi = np.load(f"benchmarks/xval_akai2018_wetting/_last_phi_{arm}.npy")
        report(arm, phi, 0.5, theta_true=target)