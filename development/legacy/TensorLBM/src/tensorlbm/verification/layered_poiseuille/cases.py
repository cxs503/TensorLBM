"""Arm matrix for ``XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET``.

The arm list is defined *before* any run and is read by
``xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_arm.py``.  Each arm changes
one thing with respect to its reference arm (see ``xval/.../predeclared.json``):

====  ==========  ======  =====  ======  ============  ==================
arm   family      H       nx     M       x-boundary    init / inlet
====  ==========  ======  =====  ======  ============  ==================
S32   single      32      8      1       periodic      analytic / --
S64   single      64      8      1       periodic      analytic / --
S96   single      96      8      1       periodic      analytic / --
H32   single      32      8      1       periodic      uniform  / --
I32   single      32      3H     1       inlet_outlet  analytic / analytic
I64   single      64      3H     1       inlet_outlet  analytic / analytic
I32z  single      32      3H     1       inlet_outlet  analytic / zero-gradient (control)
L3_32 layered     32      8      3       periodic      analytic / --
L3_64 layered     64      8      3       periodic      analytic / --
L3_96 layered     96      8      3       periodic      analytic / --
L3_64_h25 layered 64      8      3       periodic      analytic / --   (interface at 0.25H)
L5_64 layered     64      8      5       periodic      analytic / --
L3_64_I layered  64      3H     3       inlet_outlet  analytic / analytic
====  ==========  ======  =====  ======  ============  ==================
"""

from __future__ import annotations

# single-phase viscosity / layered viscosities (lattice units)
NU_SINGLE = 0.1
LAYERED_M3 = (0.15, 0.05)      # nu_bottom / nu_top = 3   (tau 0.95 / 0.65)
LAYERED_M5 = (0.20, 0.04)      # nu_bottom / nu_top = 5   (tau 1.10 / 0.62)
U_PEAK = 0.05

EPS_STEADY = 1e-6
EPS_STEADY_INLET = 1e-6

ARMS = [
    # --- resolution study, single phase, periodic (TCLB developedFlow) ------ #
    dict(label="S32", family="single", height=32, nx=8, nus=(NU_SINGLE,), interface_fracs=(),
         u_peak=U_PEAK, x_boundary="periodic", init_mode="analytic",
         steps=80000, sample_every=200, tol_steady=EPS_STEADY),
    dict(label="S64", family="single", height=64, nx=8, nus=(NU_SINGLE,), interface_fracs=(),
         u_peak=U_PEAK, x_boundary="periodic", init_mode="analytic",
         steps=120000, sample_every=400, tol_steady=EPS_STEADY),
    dict(label="S96", family="single", height=96, nx=8, nus=(NU_SINGLE,), interface_fracs=(),
         u_peak=U_PEAK, x_boundary="periodic", init_mode="analytic",
         steps=160000, sample_every=600, tol_steady=EPS_STEADY),
    # --- control: no analytic initial condition ----------------------------- #
    dict(label="H32", family="single_control_init", height=32, nx=8, nus=(NU_SINGLE,),
         interface_fracs=(), u_peak=U_PEAK, x_boundary="periodic", init_mode="uniform",
         steps=80000, sample_every=200, tol_steady=EPS_STEADY),
    # --- analytic inlet BC (extension beyond TCLB's initialisation) --------- #
    dict(label="I32", family="single_inlet", height=32, nx=96, nus=(NU_SINGLE,),
         interface_fracs=(), u_peak=U_PEAK, x_boundary="inlet_outlet", init_mode="analytic",
         steps=40000, sample_every=200, tol_steady=EPS_STEADY_INLET),
    dict(label="I64", family="single_inlet", height=64, nx=192, nus=(NU_SINGLE,),
         interface_fracs=(), u_peak=U_PEAK, x_boundary="inlet_outlet", init_mode="analytic",
         steps=60000, sample_every=400, tol_steady=EPS_STEADY_INLET),
    dict(label="I32z", family="single_inlet_control", height=32, nx=96, nus=(NU_SINGLE,),
         interface_fracs=(), u_peak=U_PEAK, x_boundary="inlet_outlet", init_mode="analytic",
         inlet_from_analytic=False, steps=40000, sample_every=200,
         tol_steady=EPS_STEADY_INLET),
    # --- layered (variable-viscosity) channel ------------------------------- #
    dict(label="L3_32", family="layered", height=32, nx=8, nus=LAYERED_M3,
         interface_fracs=(0.5,), u_peak=U_PEAK, x_boundary="periodic",
         init_mode="analytic", steps=80000, sample_every=200, tol_steady=EPS_STEADY),
    dict(label="L3_64", family="layered", height=64, nx=8, nus=LAYERED_M3,
         interface_fracs=(0.5,), u_peak=U_PEAK, x_boundary="periodic",
         init_mode="analytic", steps=120000, sample_every=400, tol_steady=EPS_STEADY),
    dict(label="L3_96", family="layered", height=96, nx=8, nus=LAYERED_M3,
         interface_fracs=(0.5,), u_peak=U_PEAK, x_boundary="periodic",
         init_mode="analytic", steps=160000, sample_every=600, tol_steady=EPS_STEADY),
    dict(label="L3_64_h25", family="layered_interface", height=64, nx=8, nus=LAYERED_M3,
         interface_fracs=(0.25,), u_peak=U_PEAK, x_boundary="periodic",
         init_mode="analytic", steps=120000, sample_every=400, tol_steady=EPS_STEADY),
    dict(label="L5_64", family="layered_contrast", height=64, nx=8, nus=LAYERED_M5,
         interface_fracs=(0.5,), u_peak=U_PEAK, x_boundary="periodic",
         init_mode="analytic", steps=160000, sample_every=400, tol_steady=EPS_STEADY),
    dict(label="L3_64_I", family="layered_inlet", height=64, nx=192, nus=LAYERED_M3,
         interface_fracs=(0.5,), u_peak=U_PEAK, x_boundary="inlet_outlet",
         init_mode="analytic", steps=60000, sample_every=400, tol_steady=EPS_STEADY_INLET),
]


# Arms I32 / I64 / I32z / L3_64_I (analytic inlet as a *boundary condition* on an open
# channel) are kept in the repo but EXCLUDED from the acceptance claim: measured
# unstable (flow rate oscillates between +1.06 and -0.81 of the analytic Q, see
# docs/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET.md "failure modes").
SKIPPED_ARMS = {"I32", "I64", "I32z", "L3_64_I"}

ACTIVE_ARMS = [a for a in ARMS if a["label"] not in SKIPPED_ARMS]


def arm_by_label(label: str) -> dict:
    for a in ARMS:
        if a["label"] == label:
            return dict(a)
    raise KeyError(label)