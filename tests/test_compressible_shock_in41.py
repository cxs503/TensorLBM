"""Tests for the IN41-N05 compressible-shock verification surface.

Every assertion checks a physical or numerical property with an independent
reference: the exact Rankine-Hugoniot relations, a synthetic shock geometry for
the detachment-angle detector, and the non-circular (shock-Mach-as-output)
shock-tube case.  The 2-D cases use small grids / short times so the suite stays
fast; the production numbers come from the cluster runs of
``benchmarks/compressible_shock_family_in41/run.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressible_shock import analysis, cases
from tensorlbm.compressible_shock.flux import GAMMA
from tensorlbm.compressible_shock.rankine_hugoniot import (
    normal_shock,
    sound_speed,
)


# ------------------------------------------------- detachment-angle detector
def _synthetic_shock_field(angle_deg: float, n: int = 201):
    """A straight shock of the given angle (deg from the x-axis).

    Density is 1 ahead of the shock and 4 behind it.  The shock line passes
    through the detector box centre (0.375, 0.425) so the front lies inside the
    measurement box for every angle; ``angle_deg`` is the angle of the shock
    *tangent* with the x-axis (90 deg = vertical / normal shock).
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    slope = 1.0 / math.tan(math.radians(angle_deg))  # dX/dY
    behind = X > 0.375 + slope * (Y - 0.425)
    rho = np.where(behind, 4.0, 1.0)
    return rho, x, y


def test_detachment_angle_normal_shock_is_90():
    """A vertical (normal) shock reads exactly 90 deg (the bias-free case)."""
    rho, x, y = _synthetic_shock_field(90.0)
    d = analysis.shock_detachment_angle(rho, x, y, corner_x=0.6, corner_y=0.5, threshold=2.5)
    assert d["n_cells"] > 0
    assert d["angle_deg"] == pytest.approx(90.0, abs=1e-6)


def test_detachment_angle_near_normal_within_bias():
    """A near-vertical shock reads within the documented discretization bias.

    The gradient of a staircase (discrete) shock is not exactly perpendicular
    to the shock line, so a shock at 85 deg reads a few degrees shallow; the
    bias vanishes as the grid refines and cancels in the grid-convergence
    difference used as the acceptance criterion.
    """
    rho, x, y = _synthetic_shock_field(85.0)
    d = analysis.shock_detachment_angle(rho, x, y, corner_x=0.6, corner_y=0.5, threshold=2.5)
    assert d["n_cells"] > 0
    assert 70.0 < d["angle_deg"] < 90.0


# ------------------------------------------------------- 1-D normal shock scan
@pytest.mark.parametrize("mach,n", [(3.0, 400), (10.0, 400)])
def test_normal_shock_jump_within_1pct(mach, n):
    """Core criterion: the prescribed R-H jump is preserved to < 1 % (rho/p/T)."""
    solver, meta = cases.normal_shock_case(mach, n=n, variant="jump")
    ref = meta["reference"]
    rho1, p1 = meta["pre_state"]["rho"], meta["pre_state"]["p"]
    solver.advance(meta["t_end"])
    rho = solver.primitives()[0]
    thr = 0.5 * (rho1 + ref["rho2"])
    xf = analysis.front_x(rho, solver.xc, thr, x_min=meta["x_diaphragm"] * 0.5)
    st = analysis.sample_state(solver.U, solver.xc, xf - 20 * solver.dx, xf - 3 * solver.dx)
    jr = analysis.jump_ratios(
        {"rho": rho1, "p": p1, "T": p1 / rho1, "u": 0.0, "a": sound_speed(rho1, p1)}, st
    )
    for key in ("rho_ratio", "p_ratio", "T_ratio"):
        err = analysis.relative_error(jr[key], ref[key])
        assert err < 0.01, f"{key} error {err * 100:.3f}% exceeds 1%"


def test_normal_shock_riemann_is_noncircular():
    """The riemann variant yields the target shock Mach number as an *output*."""
    mach = 3.0
    solver, meta = cases.normal_shock_case(mach, n=400, variant="riemann")
    ref = meta["reference"]
    rho1, p1 = meta["pre_state"]["rho"], meta["pre_state"]["p"]
    times, fronts = [], []
    for frac in (0.4, 0.7, 1.0):
        solver.advance(frac * meta["t_end"])
        rho = solver.primitives()[0]
        thr = 0.5 * (rho1 + ref["rho2"])
        times.append(solver.t)
        fronts.append(analysis.front_x(rho, solver.xc, thr, x_min=meta["x_diaphragm"] * 0.5))
    fit = analysis.fit_speed(times, fronts)
    a1 = sound_speed(rho1, p1)
    ms_measured = fit["speed"] / a1
    assert ms_measured == pytest.approx(mach, rel=0.02)


# ------------------------------------------------------------- 2-D smoke tests
def test_dmr_smoke_forms_mach_reflection():
    """A small DMR run develops a stem and the incident angle is ~60 deg."""
    solver, meta = cases.double_mach_reflection_case(nx=120, ny=30, t_end=0.15)
    solver.advance(0.15)
    rho = solver.primitives()[0]
    thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
    prof = analysis.front_profile(rho, solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"])
    assert np.isfinite(prof).sum() >= 10
    upper = np.isfinite(prof) & (solver.yc > 0.6 * solver.ly)
    b, _ = np.polyfit(solver.yc[upper], prof[upper], 1)
    angle = float(np.degrees(np.arctan2(1.0, b)))
    assert angle == pytest.approx(60.0, abs=3.0)


def test_ffs_smoke_detachment_angle_finite():
    """A small FFS run gives a finite detachment angle near the normal limit."""
    solver, meta = cases.forward_facing_step_case(nx=100, ny=33, t_end=1.0)
    solver.advance(1.0)
    rho = solver.primitives()[0]
    jump = normal_shock(meta["mach"])
    thr = 0.5 * (meta["inflow"]["rho"] * (1.0 + jump["rho_ratio"]))
    d = analysis.shock_detachment_angle(
        rho, solver.xc, solver.yc, meta["step_x"], meta["step_h"], thr
    )
    assert math.isfinite(d["angle_deg"])
    assert 55.0 < d["angle_deg"] < 95.0
