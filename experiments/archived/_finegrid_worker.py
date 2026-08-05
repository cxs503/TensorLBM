"""Fine-grid Musker+vanDriest comparison worker. Args: <did> <case> <improved:0|1>"""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

def main():
    did = int(sys.argv[1])
    case = sys.argv[2]
    improved = sys.argv[3] == "1"

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)
    tag = "improved" if improved else "base"

    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    if case.startswith("suboff"):
        from tensorlbm.suboff_cad import build_suboff_mask, SuboffHullType
        from tensorlbm.suboff_resistance import _voxel_wetted_area
        nx, ny, nz, hl, ns, cs = 320, 128, 128, 128.0, 1500, 0.05
        nu = 0.06 * hl / 2e6; tau = 3.0 * nu + 0.5
        cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
        solid, _ = build_suboff_mask(SuboffHullType.BARE_HULL, nx, ny, nz, cx, cy, cz, hl, device)
        S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * 0.06 ** 2 * S
        ref_ct = 0.00405
    elif case.startswith("flat"):
        from tensorlbm.suboff_resistance import _ittc57_friction_coefficient
        nx, ny, nz, hl, ns = 320, 128, 128, 128.0, 1500
        cs = float(case.split("_cs")[1]) if "_cs" in case else 0.05
        nu = 0.06 * hl / 2e6; tau = 3.0 * nu + 0.5
        solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
        solid[:, :, :24] = True  # plate leading edge
        S = ny * nz; dpS = 0.5 * 1.0 * 0.06 ** 2 * S
        ref_ct = _ittc57_friction_coefficient(2e6)
    elif case.startswith("kvlcc2") or case.startswith("wigley"):
        from tensorlbm.ship_cad import build_hull_mask, ShipHullType
        from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
        nx, ny, nz, hl, ns, cs = 320, 96, 96, 96.0, 2000, 0.05
        nu = 0.06 * hl / 2e6; tau = 3.0 * nu + 0.5
        hull_type = "kvlcc2" if "kvlcc2" in case else "wigley"
        solid, _ = build_hull_mask(hull_type, nx, ny, nz, cx=nx*0.3, cy=ny*0.5, cz_keel=nz*0.5, device=device)
        S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * 0.06 ** 2 * S
        cf = _ittc57_friction_coefficient(2e6)
        ff = 1.2 if "kvlcc2" in case else 1.15
        ref_ct = cf * ff
    else:
        raise ValueError(f"unknown case: {case}")

    r0 = torch.ones(nz, ny, nx, device=device)
    u0 = torch.full((nz, ny, nx), 0.06, device=device); u0[solid] = 0
    f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0), device=device)
    im = float(r0.sum().item())
    warmup = ns // 3; fric, pres = [], []; t0 = time.time()

    label = f"[SDAA:{did}] {case} {tag}"
    print(f"{label} start: {nx}x{ny}x{nz} steps={ns} improved={improved}", flush=True)

    for step in range(1, ns + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu, wall_law=("musker" if improved else "log"), use_van_driest=improved)
        f = far_field_bc_3d(f, u_in=0.06)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df): fric.append(df); pres.append(dp)

        if step % 500 == 0 or step == ns:
            cf = sum(fric) / max(len(fric), 1) / dpS if fric else 0
            cp = sum(pres) / max(len(pres), 1) / dpS if pres else 0
            err = abs(cf + cp - ref_ct) / ref_ct * 100 if ref_ct else 0
            print(f"{label} step={step} Ct={cf+cp:.5f} f={cf:.4f} p={cp:.4f} err={err:.1f}% ({time.time()-t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{label} DIV at {step}", flush=True); break

    cf = sum(fric) / max(len(fric), 1) / dpS if fric else 0
    cp = sum(pres) / max(len(pres), 1) / dpS if pres else 0
    ct = cf + cp
    err = abs(ct - ref_ct) / ref_ct * 100 if ref_ct else 0

    result = {"case": case, "improved": improved, "grid": f"{nx}x{ny}x{nz}",
              "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct, "Ct_ref": ref_ct,
              "error_pct": err, "steps": step, "finite": bool(torch.isfinite(f).all().item())}
    out = Path(f"/tmp/finegrid_results/{case}_{tag}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{label} DONE Ct={ct:.5f} err={err:.1f}%", flush=True)

if __name__ == "__main__":
    main()
