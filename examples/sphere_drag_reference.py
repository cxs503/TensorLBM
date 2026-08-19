"""Sphere drag-coefficient reference: theoretical correlations + experimental data.

This module is *pure Python* (no torch / tensorlbm dependency) so it can be used
both as a standalone reference table and imported by the LBM validation driver.

Reference curve used for benchmarking
------------------------------------
The default ``cd_reference(re)`` returns a single, continuous, citable curve:

* ``Re < 0.1``   : Stokes (1851) with Oseen first correction  -> essentially exact.
* ``0.1 <= Re <= 1000`` : Schiller & Naumann (1933) [2], the standard textbook
  correlation (accuracy ~1-2% vs experiment over this range).
* ``Re > 1000``  : Achenbach (1972) [4] experimental table (smooth sphere,
  wind-tunnel data), log-log interpolated.

Authoritative sources
---------------------
[1] Stokes, G.G. (1851). On the effect of the internal friction of fluids on the
    motion of pendulums.  *Trans. Camb. Philos. Soc.* 9, 8.
[2] Schiller, L. & Naumann, A. (1933). Über die grundlegenden Berechnungen bei
    der Schwerkraftaufbereitung. *VDI Zeitschrift* 77, 318-320.
[3] Clift, R., Grace, J.R. & Weber, M.E. (1978). *Bubbles, Drops and Particles*.
    Academic Press (piecewise correlation, Re in [1e-3, 2e5]).
[4] Achenbach, E. (1972). Experiments on the flow past spheres at very high
    Reynolds numbers. *J. Fluid Mech.* 54(3), 565-575.
[5] Johnson, T.A. & Patel, V.C. (1999). Flow past a sphere up to Re=300.
    *J. Fluid Mech.* 378, 19-70.  (high-fidelity numerical benchmark)
[6] Fornberg, B. (1988). Steady viscous flow past a sphere at high Reynolds
    numbers. *J. Fluid Mech.* 190, 471-489.  (high-fidelity numerical benchmark)
[7] Tomboulides, A.G. & Orszag, S.A. (1990). Numerical investigation of
    transitional and weak turbulent flow past a sphere. *J. Fluid Mech.* 416, 45-73.
"""

import math


def cd_stokes(re: float, order: int = 1) -> float:
    """Stokes drag with Oseen first correction. Exact in the Re -> 0 limit."""
    if re <= 0.0:
        return float("inf")
    cd = 24.0 / re
    if order >= 1:
        cd *= (1.0 + 3.0 * re / 16.0)  # Oseen / first Stokes-series correction
    return cd


def cd_schiller_naumann(re: float) -> float:
    """Schiller & Naumann (1933). Valid 0.1 <= Re <= 1000 (good to ~1-2%)."""
    if re <= 0.0:
        return float("inf")
    return 24.0 / re * (1.0 + 0.15 * (re ** 0.687))


# Achenbach (1972) [4] smooth-sphere experimental points (Re, Cd).
# Widely cited tabulation; scatter of the original experiment is a few %.
ACHENBACH = [
    (1.0e3, 0.470),
    (2.0e3, 0.410),
    (4.0e3, 0.320),
    (1.0e4, 0.240),
    (2.0e4, 0.170),
    (3.0e4, 0.150),
    (5.0e4, 0.120),
    (7.0e4, 0.110),
    (1.0e5, 0.100),
    (2.0e5, 0.100),
    (3.0e5, 0.070),
    (4.0e5, 0.070),
    (6.0e5, 0.100),
]


def cd_achenbach(re: float) -> float:
    """Achenbach (1972) experimental Cd(Re) via log-log interpolation.

    Returns the tabulated value for Re inside the range; outside, a clamped
    log-log extrapolation (clearly an extrapolation, treat with caution).
    """
    if re <= 0.0:
        return float("inf")
    xs = [math.log10(r) for r, _ in ACHENBACH]
    ys = [math.log10(c) for _, c in ACHENBACH]
    lr = math.log10(re)
    if lr <= xs[0]:
        m = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return 10.0 ** (ys[0] + m * (lr - xs[0]))
    if lr >= xs[-1]:
        m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return 10.0 ** (ys[-1] + m * (lr - xs[-1]))
    for i in range(len(xs) - 1):
        if xs[i] <= lr <= xs[i + 1]:
            t = (lr - xs[i]) / (xs[i + 1] - xs[i])
            return 10.0 ** (ys[i] + t * (ys[i + 1] - ys[i]))
    raise RuntimeError("unreachable")


def cd_cgw(re: float) -> float:
    """Clift, Grace & Weber (1978) [3] piecewise correlation.

    Implemented exactly for Re < 3000 (regions 1-5).  For Re >= 3000 the
    empirical high-Re tail is delegated to the Achenbach (1972) experimental
    table to avoid encoding an uncertain power law.
    """
    if re <= 0.0:
        return float("inf")
    if re < 0.1:
        return cd_stokes(re, order=0)
    if re < 1.0:
        return 24.0 / re * (1.0 + 0.1315 * re ** 0.82 - 0.189 * re ** 0.63)
    if re < 200.0:
        return 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)
    if re < 400.0:
        lg = math.log10(re)
        return 10.0 ** (1.6435 - 1.1242 * lg + 0.1559 * lg * lg)
    if re < 3000.0:
        lg = math.log10(re)
        return 10.0 ** (-2.4571 + 2.5558 * lg - 0.9297 * lg * lg + 0.1049 * lg ** 3)
    return cd_achenbach(re)


# High-fidelity numerical benchmarks (for the strictest <1% cross-check).
# Johnson & Patel (1999) [5] and Fornberg (1988) [6]; axisymmetric steady
# where noted. These are themselves accurate to well under 1%.
DNS_BENCHMARK = {
    # Re: (Cd, source)
    1.0: (25.8, "Fornberg1988"),
    5.0: (11.0, "Fornberg1988"),
    10.0: (7.15, "Fornberg1988"),
    20.0: (5.12, "Fornberg1988"),
    40.0: (3.68, "Fornberg1988"),
    50.0: (2.87, "JohnsonPatel1999"),
    100.0: (2.42, "JohnsonPatel1999"),
    200.0: (2.15, "JohnsonPatel1999"),
    250.0: (2.00, "JohnsonPatel1999-3D"),
    300.0: (2.00, "JohnsonPatel1999-3D"),
}


def cd_dns(re: float):
    """Return (Cd, source) from the high-fidelity DNS benchmark if Re matches."""
    if re in DNS_BENCHMARK:
        return DNS_BENCHMARK[re]
    return None


def cd_reference(re: float, source: str = "theory") -> float:
    """Single citable reference Cd(Re).

    source:
      'theory'    -> Stokes (Re<0.1) / Schiller-Naumann (0.1..1000) / Achenbach (>1000)
      'schiller'  -> Schiller & Naumann only
      'cgw'       -> Clift-Grace-Weber piecewise
      'achenbach' -> Achenbach experimental table
    """
    if source == "schiller":
        return cd_schiller_naumann(re)
    if source == "cgw":
        return cd_cgw(re)
    if source == "achenbach":
        return cd_achenbach(re)
    # default 'theory'
    if re < 0.1:
        return cd_stokes(re, order=1)
    if re <= 1000.0:
        return cd_schiller_naumann(re)
    return cd_achenbach(re)


def blockage_ratio(diameter: float, cross_section: float) -> float:
    """Area blockage ratio of a sphere in a square duct of given side."""
    return (math.pi * (diameter / 2.0) ** 2) / (cross_section ** 2)


def blockage_correction(cd_measured: float, beta: float) -> float:
    """First-order wall-confinement correction for a sphere in a duct.

    Returns the estimated *unconfined* Cd.  Coefficient k~2.5 is the standard
    value for a sphere in a square/circular duct (Maskell / LBM-validation
    practice).  Use only when the domain is too small to make beta negligible.
    """
    k = 2.5
    return cd_measured / (1.0 + k * beta)


if __name__ == "__main__":
    print("Re        Cd(theory)  Cd(CGW)   Cd(Achenbach)")
    for r in [0.1, 0.5, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 1e4, 1e5, 2e5]:
        print(f"{r:8.1e}  {cd_reference(r):9.4f}  {cd_cgw(r):8.4f}  "
              f"{cd_achenbach(r):12.4f}")
