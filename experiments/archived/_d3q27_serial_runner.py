"""D3Q27 CASCADED/CUMULANT SERIAL benchmark runner.
Runs all 8 benchmarks ONE AT A TIME on SDAA:0-3 to avoid OOM.

Benchmarks (per task spec, completed from context):
A. SHIP HULLS (attached flow, primary target):
  1. KVLCC2   200x60x60, hl=80, Re=2e6, 2000 steps (SDAA:0)
  2. Wigley   200x60x60, hl=80, Re=2e6, 2000 steps (SDAA:1)
  3. KCS      200x60x60, hl=80, Re=2e6, 2000 steps (SDAA:2)

B. CONFIRMATION (small/fast):
  4. SUBOFF bare_hull 200³, Re=2e6, 3000 steps CASCADED  (SDAA:3)
  5. SUBOFF bare_hull 200³, CUMULANT variant, 2000 steps (SDAA:0)

C. BLUFF BODIES (separated flow validation):
  6. Cylinder  Re=200, D=24, 200×80×4, 2000 steps (SDAA:0)
  7. Sphere    Re=1000, D=24, 120×60×60, 2000 steps (SDAA:1)

D. LARGER GRID:
  8. SUBOFF bare_hull 256³, Re=2e6, 2000 steps (SDAA:0)

All: D3Q27, NO Smagorinsky (Cs=0), wallfn log-law, sign-fixed code.
Uses: collide_cascaded_d3q27 + wall_function_d3q27 + stream27 + far_field_bc_27.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ── D3Q27 streaming & far-field BC (self-contained) ──────────────────────
_D3Q27_SHIFTS = [
    (cx, cy, cz) for cz in [-1, 0, 1] for cy in [-1, 0, 1] for cx in [-1, 0, 1]
]


def stream27_roll(f):
    """Memory-efficient D3Q27 streaming using torch.roll."""
    import torch
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def far_field_bc_27(f, u_in=0.06):
    """Equilibrium at inlet, zero-gradient at outlet, symmetry at sides."""
    import torch
    from tensorlbm.d3q27 import equilibrium27

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1,
        torch.full_like(rho1, u_in),
        torch.zeros_like(rho1),
        torch.zeros_like(rho1),
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet
    f[:, :, :, -1] = f[:, :, :, -2]        # outlet zero-gradient
    f[:, 0, :, :] = feq[:, 0, :, :]        # y-min
    f[:, -1, :, :] = feq[:, -1, :, :]      # y-max
    f[:, :, 0, :] = feq[:, :, 0, :]        # z-min
    f[:, :, -1, :] = feq[:, :, -1, :]      # z-max
    return f


# ── Mask builders for bluff bodies ───────────────────────────────────────
def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    import torch
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    import torch
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


# ── Collision dispatchers ────────────────────────────────────────────────
def collide_fn(op, f, tau):
    """Dispatch D3Q27 CASCADED or CUMULANT (no Smagorinsky)."""
    if op == "cascaded":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        return collide_cascaded_d3q27(f, tau)
    elif op == "cumulant":
        from tensorlbm.cumulant import collide_cumulant_d3q27
        return collide_cumulant_d3q27(f, tau)
    else:
        raise ValueError(f"Unknown collision operator: {op}")


# ── Reference values ────────────────────────────────────────────────────
REF_CT_SUBOFF = 0.00405
REF_CD_CYLINDER = {200: 1.30, 500: 1.20}
REF_CD_SPHERE = {1000: 0.47, 10000: 0.40}


def run_one(did, case, nx, ny, nz, hl, n_steps, op, u_in_override=None, re_override=None):
    """Run a single benchmark case and return result dict."""
    import torch
    from tensorlbm.d3q27 import equilibrium27, correct_mass27
    from tensorlbm.wall_model import wall_function_d3q27

    # ── Physics parameters ──────────────────────────────────────────────
    u_in = u_in_override or 0.06
    re = re_override or 2e6
    diameter = None

    if case == "cylinder":
        re = 200.0
        diameter = hl
        u_in = 0.08
    elif case == "sphere":
        re = 1000.0
        diameter = hl
        u_in = 0.08

    if case in ("cylinder", "sphere"):
        nu = u_in * diameter / re
    else:
        nu = u_in * hl / re

    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] D3Q27-{op.upper()} {case} {nx}x{ny}x{nz}"
    print(f"\n{'='*70}")
    print(f"{tag} hl={hl:.1f} u_in={u_in} Re={re:.0f} nu={nu:.2e} tau={tau:.6f}")
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    # ── Build geometry mask ────────────────────────────────────────────
    if case in ("suboff", "suboff_cumulant"):
        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
        from tensorlbm.suboff_resistance import _voxel_wetted_area

        cx_hull, cy_hull, cz_hull = nx * 0.35, ny / 2.0, nz / 2.0
        solid, _ = build_suboff_mask(
            hull_type=SuboffHullType.BARE_HULL,
            nx=nx, ny=ny, nz=nz,
            cx=cx_hull, cy=cy_hull, cz=cz_hull,
            length=hl, device=device,
        )
        S = _voxel_wetted_area(solid, 1.0)
        A_ref = S
    elif case in ("kvlcc2", "wigley", "kcs"):
        from tensorlbm.ship_cad import ShipHullType, build_hull_mask

        ht_map = {"kvlcc2": ShipHullType.KVLCC2, "wigley": ShipHullType.WIGLEY, "kcs": ShipHullType.KCS}
        ht = ht_map[case]
        solid, stats = build_hull_mask(
            hull_type=ht, nx=nx, ny=ny, nz=nz,
            length=hl, device=str(device),
        )
        S = float(solid.sum().item())
        A_ref = S
        print(f"{tag} solid_cells={S:.0f} stats={stats.get('Cb_numerical', '?')}", flush=True)
    elif case == "cylinder":
        radius_cyl = diameter / 2.0
        cx_cyl, cy_cyl = nx * 0.25, ny * 0.5
        solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, radius_cyl, device)
        A_ref = diameter * nz
    elif case == "sphere":
        radius_sph = diameter / 2.0
        cx_sph, cy_sph, cz_sph = nx * 0.25, ny * 0.5, nz * 0.5
        solid = build_sphere_mask(nx, ny, nz, cx_sph, cy_sph, cz_sph, radius_sph, device)
        A_ref = math.pi * radius_sph ** 2

    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref
    print(f"{tag} A_ref={A_ref:.0f} dyn_p={dyn_p:.6e} init done ({time.time()-t0:.1f}s)", flush=True)

    # ── Initialize distributions ───────────────────────────────────────
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    # ── Simulation loop ────────────────────────────────────────────────
    warmup = n_steps // 3
    slide = 500
    fric_hist = []
    pres_hist = []
    all_finite = True

    for step in range(1, n_steps + 1):
        try:
            f = collide_fn(op, f, tau)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at step {step}: {e}", flush=True)
            all_finite = False
            break

        f = stream27_roll(f)
        f, df, dp = wall_function_d3q27(f, solid, nu, y_val=0.5)
        f = far_field_bc_27(f, u_in=u_in)

        if step % 100 == 0:
            f = correct_mass27(f, initial_mass)

        if step > warmup and math.isfinite(df) and math.isfinite(dp):
            fric_hist.append(df)
            pres_hist.append(dp)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            all_finite = False
            break

        if step % 500 == 0 or step == n_steps:
            wf = fric_hist[-slide:] if len(fric_hist) >= slide else fric_hist
            wp = pres_hist[-slide:] if len(pres_hist) >= slide else pres_hist
            cf_w = sum(wf) / max(len(wf), 1) / dyn_p if wf else 0.0
            cp_w = sum(wp) / max(len(wp), 1) / dyn_p if wp else 0.0
            ct_w = cf_w + cp_w
            elapsed = time.time() - t0
            print(f"{tag} step={step} Ct={ct_w:.5f} (Cf={cf_w:.5f} Cp={cp_w:.5f}) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # ── Final statistics ───────────────────────────────────────────────
    cf = sum(fric_hist) / max(len(fric_hist), 1) / dyn_p if fric_hist else 0.0
    cp = sum(pres_hist) / max(len(pres_hist), 1) / dyn_p if pres_hist else 0.0
    ct_total = cf + cp

    wf = fric_hist[-slide:] if len(fric_hist) >= slide else fric_hist
    wp = pres_hist[-slide:] if len(pres_hist) >= slide else pres_hist
    cf_slide = sum(wf) / max(len(wf), 1) / dyn_p if wf else 0.0
    cp_slide = sum(wp) / max(len(wp), 1) / dyn_p if wp else 0.0
    ct_slide = cf_slide + cp_slide

    if len(wf) > 1:
        ct_slide_std = math.sqrt(sum(((f_i + p_i) / dyn_p - ct_slide) ** 2
                                     for f_i, p_i in zip(wf, wp)) / (len(wf) - 1))
    else:
        ct_slide_std = 0.0

    # Reference and error
    if case in ("suboff", "suboff_cumulant"):
        ref_ct = REF_CT_SUBOFF
        ref_label = "SUBOFF AFF-8"
        err_pct = abs(ct_total - ref_ct) / ref_ct * 100 if ref_ct > 0 else float("nan")
    elif case in ("cylinder", "sphere"):
        ref_map = REF_CD_CYLINDER if case == "cylinder" else REF_CD_SPHERE
        ref_ct = ref_map.get(int(re), float("nan"))
        ref_label = f"{case} Re={int(re)}"
        err_pct = abs(ct_total - ref_ct) / ref_ct * 100 if ref_ct and ref_ct > 0 else float("nan")
    else:
        ref_ct = float("nan")
        ref_label = f"{case} (no experimental ref)"
        err_pct = float("nan")

    result = {
        "case": case,
        "did": did,
        "lattice": "D3Q27",
        "collision": op,
        "Smagorinsky_Cs": 0.0,
        "grid": f"{nx}x{ny}x{nz}",
        "hull_length_or_D": hl,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "sliding_window": slide,
        "Ct_fric": cf,
        "Ct_pres": cp,
        "Ct_total": ct_total,
        "Ct_slide": ct_slide,
        "Ct_slide_std": ct_slide_std,
        "ref_value": ref_ct,
        "ref_label": ref_label,
        "error_pct": err_pct,
        "samples_total": len(fric_hist),
        "samples_sliding": min(len(fric_hist), slide),
        "finite": all_finite,
        "elapsed_s": elapsed,
        "mlups": (nx * ny * nz * n_steps / elapsed / 1e6) if elapsed > 0 else 0.0,
        "wall_law": "log",
        "farfield": True,
        "device": f"sdaa:{did}",
    }

    print(f"\n{tag} DONE Ct_full={ct_total:.5f} Ct_slide={ct_slide:.5f}±{ct_slide_std:.5f} "
          f"err={err_pct if math.isfinite(err_pct) else 'N/A'} "
          f"({elapsed:.0f}s)", flush=True)

    return result


# ── Benchmark matrix (8 cases, serial on SDAA:0-3) ──────────────────────
# Format: (did, case, nx, ny, nz, hl, n_steps, op)
BENCHMARKS = [
    # A. SHIP HULLS — attached flow
    (0, "kvlcc2",   200, 60, 60, 80.0,  2000, "cascaded"),
    (1, "wigley",   200, 60, 60, 80.0,  2000, "cascaded"),
    (2, "kcs",      200, 60, 60, 80.0,  2000, "cascaded"),

    # B. CONFIRMATION — SUBOFF reproducibility + operator comparison
    (3, "suboff",            200, 80, 80, 80.0,  3000, "cascaded"),
    (0, "suboff_cumulant",   200, 80, 80, 80.0,  2000, "cumulant"),

    # C. BLUFF BODIES — separated flow validation
    (0, "cylinder",  200, 80, 4,   24.0,  2000, "cascaded"),
    (1, "sphere",    120, 60, 60,  24.0,  2000, "cascaded"),

    # D. LARGER GRID — scaling validation
    (0, "suboff",    256, 103, 103, 102.4, 2000, "cascaded"),
]


def main():
    import torch
    print("=== D3Q27 CASCADED/CUMULANT SERIAL BENCHMARKS ===")
    print(f"Running {len(BENCHMARKS)} benchmarks SERIALLY on SDAA:0-3")
    print(f"PyTorch: {torch.__version__}, SDAA devices: 0-3")
    print()

    output_path = Path("/tmp/d3q27_serial_results.json")
    all_results = []

    total_start = time.time()

    for idx, (did, case, nx, ny, nz, hl, n_steps, op) in enumerate(BENCHMARKS):
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK {idx+1}/{len(BENCHMARKS)}: {case} {nx}x{ny}x{nz} {op}")
        print(f"{'#'*70}")

        try:
            result = run_one(did, case, nx, ny, nz, hl, n_steps, op)
            all_results.append(result)

            # Save incremental results
            output_path.write_text(json.dumps(all_results, indent=2))

        except Exception as e:
            print(f"FATAL ERROR in benchmark {case}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Save error result
            err_result = {
                "case": case, "did": did, "lattice": "D3Q27",
                "collision": op, "grid": f"{nx}x{ny}x{nz}",
                "finite": False, "error": str(e),
                "elapsed_s": 0.0,
            }
            all_results.append(err_result)
            output_path.write_text(json.dumps(all_results, indent=2))

    total_elapsed = time.time() - total_start

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ALL {len(BENCHMARKS)} BENCHMARKS COMPLETE in {total_elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"\n{'Case':<20s} {'Grid':<12s} {'Op':<10s} {'Ct/Cd':>8s} {'Ref':>8s} {'Err%':>7s} {'Time':>7s} {'Fin':>5s}")
    print("-" * 70)

    for r in all_results:
        case = r.get("case", "?")
        grid = r.get("grid", "?")
        op = r.get("collision", "?")
        val = r.get("Ct_total", float("nan"))
        ref = r.get("ref_value", float("nan"))
        err = r.get("error_pct", float("nan"))
        t = r.get("elapsed_s", 0)
        fin = "YES" if r.get("finite", False) else "NO"

        val_str = f"{val:>8.5f}" if isinstance(val, (int, float)) and math.isfinite(val) else f"{'N/A':>8s}"
        ref_str = f"{ref:>8.4f}" if isinstance(ref, (int, float)) and math.isfinite(ref) else f"{'N/A':>8s}"
        err_str = f"{err:>7.1f}" if isinstance(err, (int, float)) and math.isfinite(err) else f"{'N/A':>7s}"
        time_str = f"{t:>7.0f}s"

        print(f"{case:<20s} {grid:<12s} {op:<10s} {val_str} {ref_str} {err_str} {time_str} {fin:>5s}")

    print(f"\nResults saved to: {output_path}")
    print(f"Total wall time: {total_elapsed:.0f}s")

    # ── Key validation checks ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")

    suboff_results = [r for r in all_results if r.get("case") in ("suboff", "suboff_cumulant") and r.get("finite")]
    for r in suboff_results:
        err = r.get("error_pct", float("nan"))
        ct = r.get("Ct_total", 0)
        op = r.get("collision", "?")
        grid = r.get("grid", "?")
        if math.isfinite(err):
            status = "PASS (<5%)" if err < 5.0 else "FAIL (>=5%)"
            print(f"  SUBOFF {op} {grid}: Ct={ct:.5f} err={err:.1f}% — {status}")

    bluff_results = [r for r in all_results if r.get("case") in ("cylinder", "sphere") and r.get("finite")]
    for r in bluff_results:
        err = r.get("error_pct", float("nan"))
        ct = r.get("Ct_total", 0)
        case = r.get("case", "?")
        if math.isfinite(err):
            status = "PASS (<15%)" if err < 15.0 else f"WARN ({err:.1f}%)"
            print(f"  {case} Cd={ct:.4f} err={err:.1f}% (ref={r.get('ref_value','?')}) — {status}")

    ship_results = [r for r in all_results if r.get("case") in ("kvlcc2", "wigley", "kcs") and r.get("finite")]
    for r in ship_results:
        ct = r.get("Ct_total", 0)
        case = r.get("case", "?")
        finite = "YES" if r.get("finite") else "NO"
        print(f"  {case}: Ct={ct:.5f} finite={finite}")

    nonfinite = [r for r in all_results if not r.get("finite", True)]
    if nonfinite:
        print(f"\n  ⚠ {len(nonfinite)} benchmarks diverged/errored!")


if __name__ == "__main__":
    main()
