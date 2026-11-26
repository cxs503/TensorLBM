"""Arm matrix for ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Frozen *before* the production runs (``predeclared.json``).  Every arm is run by
**both** codes unless ``codes`` says otherwise:

* ``oss`` -- the reference implementation, executed for real (lettuce
  ``PoiseuilleFlow2D`` baseline skeleton, CPU, ``ShanChen`` forcing);
* ``tl``  -- TensorLBM, in-repo D2Q9 BGK + Guo (2002) forcing + half-way
  bounce-back (``tensorlbm.verification.layered_poiseuille.solver`` reduced to a
  single layer).

====  ==========  ====  ========  ==========  ==========  ====================
arm   family      res   nu mode   force×      init        purpose
====  ==========  ====  ========  ==========  ==========  ====================
V16   verbatim    16    oss (1)   1           zeros       OSS skeleton as shipped
V32   verbatim    32    oss (2)   1           zeros       same rule, finer grid
V48   verbatim    48    oss (3)   1           zeros       same rule, finest grid
N32   refine      32    oss16=1   1           zeros       fixed nu -> pure refinement
N48   refine      48    oss16=1   1           zeros       fixed nu -> pure refinement
F2    forcing     16    oss (1)   1e2         zeros       Mach / forcing sensitivity
F3    forcing     16    oss (1)   1e3         zeros       Mach / forcing sensitivity
F4    forcing     16    oss (1)   1e4         zeros       Mach / forcing sensitivity
D1    dtype       16    oss (1)   1           zeros       float32 (oss only)
D4    dtype       16    oss (1)   1e4         zeros       float32 (oss only)
I16   init        16    oss (1)   1           analytic    analytic init (tl only)
====  ==========  ====  ========  ==========  ==========  ====================

``nu mode``: ``oss`` = the OSS rule ``nu_lu = (res-1) * cs * Ma / Re`` (so nu
grows with the grid because Re and Ma are held fixed -- this is what happens if
the OSS skeleton is simply re-run at another resolution).  ``oss16`` = nu frozen
at the res=16 value, which isolates the discretisation error from the viscosity
change (an arm that only TensorLBM can express at the same Re/Ma bookkeeping).
"""

from __future__ import annotations

from typing import Dict, List

from tensorlbm.verification.xval.poiseuille_skeleton import oss_units

# OSS baseline skeleton defaults (lettuce examples/simple_flows/Poiseuille.ipynb)
OSS_DEFAULTS: Dict[str, float] = dict(
    reynolds_number=1.0,
    mach_number=0.02,
    acceleration_pu=0.001,
    characteristic_length_pu=1.0,
    characteristic_velocity_pu=1.0,
)

NU_OSS16 = oss_units(16, **OSS_DEFAULTS).viscosity_lu   # 0.17320508075688776

# steady-state criterion, identical for both codes
STEADY_TOL = 1e-6        # max|u(n)-u(n-window)| / max|u(n)| over one window
STEADY_WINDOW = 200      # steps between profile samples
WALL_CAP_S = 900.0       # per-arm wall-clock cap (exceeding -> reported incomplete)


def _arm(label: str, family: str, resolution: int, force_scale: float = 1.0,
         nu_mode: str = "oss", init: str = "zeros", dtype: str = "float64",
         max_steps: int = 60000, codes=("oss", "tl"), purpose: str = "") -> dict:
    base = oss_units(resolution, **OSS_DEFAULTS, force_scale=force_scale)
    spec = dict(
        label=label,
        family=family,
        resolution=resolution,
        force_scale=force_scale,
        nu_mode=nu_mode,
        init=init,
        dtype=dtype,
        max_steps=max_steps,
        codes=tuple(codes),
        purpose=purpose,
    )
    spec.update(base.as_dict())
    if nu_mode == "oss16":
        spec["viscosity_lu"] = NU_OSS16
        spec["relaxation_parameter_lu"] = NU_OSS16 / (1.0 / 3.0) + 0.5
        spec["u_peak_lu"] = spec["acceleration_lu"] * spec["height_lu"] ** 2 / (8.0 * NU_OSS16)
    return spec


ARMS: List[dict] = [
    _arm("V16", "verbatim", 16, purpose="OSS skeleton as shipped (res=16, Re=1, Ma=0.02)"),
    _arm("V32", "verbatim", 32, max_steps=90000, purpose="OSS rule at res=32 (nu grows with grid)"),
    _arm("V48", "verbatim", 48, max_steps=90000, purpose="OSS rule at res=48"),
    _arm("N32", "refine", 32, nu_mode="oss16", max_steps=90000,
         purpose="nu frozen at res=16 -> pure grid refinement"),
    _arm("N48", "refine", 48, nu_mode="oss16", max_steps=120000,
         purpose="nu frozen at res=16 -> pure grid refinement"),
    _arm("F2", "forcing", 16, force_scale=1e2, max_steps=30000,
         purpose="force x100 (u_peak_lu ~1.4e-4, Ma_peak ~2.5e-4)"),
    _arm("F3", "forcing", 16, force_scale=1e3, max_steps=30000,
         purpose="force x1000 (u_peak_lu ~1.4e-3, Ma_peak ~2.5e-3)"),
    _arm("F4", "forcing", 16, force_scale=1e4, max_steps=30000,
         purpose="force x10000 (u_peak_lu ~1.4e-2, Ma_peak ~2.5e-2)"),
    _arm("D1", "dtype", 16, dtype="float32", max_steps=30000, codes=("oss",),
         purpose="float32 at the OSS skeleton's own force (round-off trap)"),
    _arm("D4", "dtype", 16, force_scale=1e4, dtype="float32", max_steps=30000, codes=("oss",),
         purpose="float32 at force x1e4 (control: does the trap disappear?)"),
    _arm("I16", "init", 16, init="analytic", max_steps=30000, codes=("tl",),
         purpose="analytic initial condition instead of zeros (steps-to-steady)"),
]


def arm_by_label(label: str) -> dict:
    for a in ARMS:
        if a["label"] == label:
            return dict(a)
    raise KeyError(label)


def dimensionless(spec: dict) -> Dict[str, float]:
    """Dimensionless numbers of one arm (reported, never used to gate)."""
    u = spec.get("u_peak_lu", float("nan"))
    nu = spec["viscosity_lu"]
    H = spec["height_lu"]
    cs = 1.0 / 3.0 ** 0.5
    return {
        "H_lu": H,
        "nu_lu": nu,
        "tau": spec["relaxation_parameter_lu"],
        "a_lu": spec["acceleration_lu"],
        "u_peak_lu": u,
        "Ma_peak": abs(u) / cs,
        "Re_H": abs(u) * H / nu,
        "Kn_lattice": nu / (H * cs),      # nu / (H c_s): lattice "Knudsen" number
    }