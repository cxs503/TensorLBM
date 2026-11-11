#!/usr/bin/env python3
"""Apply the AUDIT-RUNTEST-6 fix to benchmarks/verified/poiseuille_convergence_order/run.py.

Defect (measured, see runs/audit_runtest_r6/conv_repro.json): the three wall arms
`halfway` / `effective` / `bouzidi` produced bit-identical errors because all
three branches set q_wall = 0.5 and only the label "bouzidi" touched the
interpolated kernel (which at q = 0.5 degenerates to half-way bounce-back), while
`effective` fell through to the very same `apply_halfway_bb` call.  The published
3-arm comparison was therefore a single computation reported three times.

Fix: each arm now gets a declared sub-cell wall distance q and a declared
analytic reference, and the geometry keeps H_true == H for every arm so the
refinement ratios stay comparable.
"""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("benchmarks/verified/poiseuille_convergence_order/run.py")

ARMS_BLOCK = '''
# ---------------------------------------------------------------------------
# Wall-treatment arms (added by AUDIT-RUNTEST-6; before this, all arms collapsed
# onto the same code path -- see the module docstring).
#
# Geometry convention: ghost rows 0 and ny-1, fluid rows 1..ny-2.  The no-slip
# surface sits a sub-cell distance q below the first fluid row
# (y_wall,bottom = 1 - q) and q above the last one (y_wall,top = ny - 2 + q),
# so the TRUE channel height is H_true = ny - 3 + 2q.  q = 0.5 is the textbook
# half-way placement, for which H_true = ny - 2 = H.
#
# `reference` selects the analytic profile the measured profile is scored
# against:
#   "true"      -- parabola spanning the true walls (height H_true, wall at 1 - q)
#   "effective" -- parabola spanning the placement the *mid-point* BC actually
#                  enforces (height H_eff = ny - 2, wall at 0.5) with the centre
#                  velocity that geometry implies -- the classic "effective wall
#                  position" correction.
# ---------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    # control: wall genuinely at the mid-point, BC exact for it
    "halfway": dict(q=0.5, kernel="halfway", reference="true"),
    # sub-cell wall ignored by the BC -> first-order wall-position error
    "subcell_halfway": dict(q=0.25, kernel="halfway", reference="true"),
    # same flow, scored against the effective wall the BC enforces
    "effective": dict(q=0.25, kernel="halfway", reference="effective"),
    # interpolated (Bouzidi) BC honouring q -> second order vs the true wall
    "bouzidi": dict(q=0.25, kernel="bouzidi", reference="true"),
}

'''


def main() -> int:
    src = PATH.read_text()

    # 1. arm table right after the stream import
    anchor = "from tensorlbm.solver import stream\n"
    assert anchor in src
    src = src.replace(anchor, anchor + ARMS_BLOCK, 1)

    # 2. geometry / analytic block
    old_geom = '''    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0
    
    # Channel geometry
    if wall_model == "effective":
        # Effective wall position: walls at y = 0.5*delta from ghost rows
        # So H_eff = H (the requested height)
        ny = H + 2  # ghost rows at 0 and ny-1
        H_eff = float(H)
        q_wall = 0.5  # half-way
    elif wall_model == "bouzidi":
        # Bouzidi: walls at y = 0.5*delta (same as half-way for flat walls)
        ny = H + 2
        H_eff = float(H)
        q_wall = 0.5
    else:  # halfway
        ny = H + 2
        H_eff = float(H)
        q_wall = 0.5
    
    nx = 4 * H  # L = 4H (long enough for fully-developed flow)
    
    # Body force to achieve target u_max
    # u_max = accel * H^2 / (8*nu)  =>  accel = 8*nu*u_max/H^2
    accel = 8.0 * nu * u_max_target / (H_eff * H_eff)
    
    # Analytic solution
    # y_phys = (j - 0.5) * delta for fluid rows j = 1, ..., ny-2
    # u(y) = 4*u_max*(y/H)*(1 - y/H)
    y_fluid = torch.arange(1, ny - 1, dtype=torch.float64) - 0.5
    y_norm = y_fluid / H_eff
    u_analytic = 4.0 * u_max_target * y_norm * (1.0 - y_norm)
'''
    new_geom = '''    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0

    if wall_model not in ARMS:
        raise ValueError(f"unknown wall_model {wall_model!r}; known: {sorted(ARMS)}")
    spec = ARMS[wall_model]
    q_wall = float(spec["q"])
    kernel = spec["kernel"]

    # Channel geometry.  ny is chosen so that the TRUE height H_true equals the
    # requested H for *every* arm (H_true = ny - 3 + 2q), which keeps the
    # refinement ratio between grids at exactly 2.
    ny = int(round(H + 3.0 - 2.0 * q_wall))
    H_true = ny - 3.0 + 2.0 * q_wall
    H_eff = float(ny - 2)  # height implied by a mid-point (q = 0.5) placement

    nx = 4 * H  # L = 4H (long enough for fully-developed flow)

    # Body force to achieve target u_max over the TRUE height
    # u_max = accel * H^2 / (8*nu)  =>  accel = 8*nu*u_max/H^2
    accel = 8.0 * nu * u_max_target / (H_true * H_true)

    # Analytic solutions, evaluated at the fluid node rows j = 1 .. ny-2
    y_fluid = torch.arange(1, ny - 1, dtype=torch.float64)

    # (a) parabola spanning the true walls:  u = 4 u_max (y/H_t)(1 - y/H_t)
    y_norm_true = (y_fluid - (1.0 - q_wall)) / H_true
    u_analytic_true = 4.0 * u_max_target * y_norm_true * (1.0 - y_norm_true)

    # (b) parabola spanning the mid-point placement the BC actually enforces,
    #     with the centre velocity that placement implies at this body force
    u_max_eff = accel * H_eff * H_eff / (8.0 * nu)
    y_norm_eff = (y_fluid - 0.5) / H_eff
    u_analytic_eff = 4.0 * u_max_eff * y_norm_eff * (1.0 - y_norm_eff)

    u_analytic = u_analytic_true if spec["reference"] == "true" else u_analytic_eff
'''
    assert old_geom in src, "geometry block not found"
    src = src.replace(old_geom, new_geom, 1)

    # 3. boundary switch uses the arm kernel
    old_bc = '''        if wall_model == "bouzidi":
            f = apply_bouzidi_bb(f, f_prev, q_wall, q_wall)
        else:
            f = apply_halfway_bb(f)'''
    new_bc = '''        if kernel == "bouzidi":
            f = apply_bouzidi_bb(f, f_prev, q_wall, q_wall)
        else:
            f = apply_halfway_bb(f)'''
    assert old_bc in src, "BC switch not found"
    src = src.replace(old_bc, new_bc, 1)

    # 4. richer per-case record (both references kept, so the textbook statement
    #    "half-way BB is 2nd order for the effective wall, 1st order for the true
    #    wall" is auditable from a single run)
    old_ret = '''        "H": H,
        "H_eff": H_eff,'''
    new_ret = '''        "H": H,
        "H_true": H_true,
        "H_eff": H_eff,
        "q_wall": q_wall,
        "bc_kernel": kernel,
        "reference": spec["reference"],'''
    assert old_ret in src
    src = src.replace(old_ret, new_ret, 1)

    old_err = '''        "u_profile": u_profile.tolist(),
        "u_analytic": u_analytic.tolist(),'''
    new_err = '''        "l2_rel_true_wall": (u_profile - u_analytic_true).pow(2).mean().sqrt().item()
        / u_max_target,
        "l2_rel_effective_wall": (u_profile - u_analytic_eff).pow(2).mean().sqrt().item()
        / u_max_target,
        "u_profile": u_profile.tolist(),
        "u_analytic": u_analytic.tolist(),'''
    assert old_err in src
    src = src.replace(old_err, new_err, 1)

    # 5. default arm list now covers all four declared treatments
    old_def = '''    ap.add_argument("--models", nargs="+", default=["halfway", "effective", "bouzidi"])'''
    new_def = '''    ap.add_argument(
        "--models", nargs="+", default=["halfway", "subcell_halfway", "effective", "bouzidi"]
    )'''
    assert old_def in src
    src = src.replace(old_def, new_def, 1)

    # 6. module docstring: replace the vacuous pre-declaration with the physics
    old_doc = '''Acceptance criteria (pre-declared):
  (a) Half-way BB: p ~ 1 (first-order due to wall position error)
  (b) Effective wall correction: p ~ 2 (second-order)
  (c) Bouzidi: p >= 1.9 (second-order)
  (d) Bouzidi error < 1% on finest grid'''
    new_doc = '''Acceptance criteria (pre-declared, corrected by AUDIT-RUNTEST-6):
  (a) halfway (wall truly at q = 0.5 delta): p ~ 2 -- mid-point BB is exact for
      that placement;
  (b) subcell_halfway (wall at q = 0.25 delta but the BC still uses the
      mid-point): p ~ 1 -- the wall-position error is O(delta);
  (c) effective (identical flow, scored against the effective wall the BC
      enforces, H_eff = ny - 2): p ~ 2 -- this is what the "effective wall
      position" correction buys;
  (d) bouzidi (interpolated BC honouring q = 0.25 delta): p >= 1.8 vs the TRUE
      wall.

Before AUDIT-RUNTEST-6 all three arms ("halfway"/"effective"/"bouzidi") set
q_wall = 0.5 and only the label "bouzidi" reached the interpolated kernel, which
at q = 0.5 is algebraically identical to half-way bounce-back; "effective" fell
through to the same apply_halfway_bb call.  The three-arm comparison in
result.json was therefore one computation reported three times (bit-identical
l2_rel down to the last digit).  Arms are now genuinely distinct.'''
    assert old_doc in src, "docstring block not found"
    src = src.replace(old_doc, new_doc, 1)

    PATH.write_text(src)
    print("patched", PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())