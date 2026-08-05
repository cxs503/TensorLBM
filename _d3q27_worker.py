"""D3Q27 CASCADED benchmark worker — all 8 cases, one unified script.

Usage:
    python _d3q27_worker.py <did> <case> <nx> <ny> <nz> <hl> <ns> <op>

Args:
    did  : SDAA device ID (0-7)
    case : "suboff" | "kvlcc2" | "wigley" | "kcs" | "cylinder" | "sphere" | "suboff_cumulant"
    nx   : grid points in x
    ny   : grid points in y
    nz   : grid points in z
    hl   : hull length (for ships/SUBOFF) or diameter (for bluff bodies)
    ns   : number of steps
    op   : "cascaded" | "cumulant"

All benchmarks use: D3Q27, no Smagorinsky, wallfn log-law, farfield, sliding window=500.
Output: /tmp/d3q27_bench_results.json

REFERENCE VALUES:
    SUBOFF bare_hull 200³: Ct ≈ 0.004 (AFF-8 experiment)
    KVLCC2, Wigley, KCS:   Ct ≈ 0.004-0.005
    Cylinder Re=200:       Cd ≈ 1.30
    Sphere Re=1000:        Cd ≈ 0.47
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

KAPPA = 0.41
B_CONST = 5.0

# ── Reference values ──────────────────────────────────────────────────────
REF_CT_SUBOFF = 0.00405
REF_CD_CYLINDER = {200: 1.30, 500: 1.20}
REF_CD_SPHERE = {1000: 0.47, 10000: 0.40}

# ── D3Q27 streaming & far-field BC (self-contained, no host sync) ─────────
_D3Q27_SHIFTS = [
    (cx, cy, cz) for cz in [-1, 0, 1] for cy in [-1, 0, 1] for cx in [-1, 0, 1]
]


def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    """Memory-efficient D3Q27 streaming using torch.roll."""
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def far_field_bc_27(f: torch.Tensor, u_in: float = 0.06) -> torch.Tensor:
    """Equilibrium at inlet, zero-gradient at outlet, symmetry at sides."""
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


# ── Mask builders for bluff bodies ────────────────────────────────────────
def build_cylinder_mask(
    nx: int, ny: int, nz: int, cx: float, cy: float, radius: float, device: torch.device,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_sphere_mask(
    nx: int, ny: int, nz: int,
    cx: float, cy: float, cz: float, radius: float, device: torch.device,
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


# ── Collision dispatchers ─────────────────────────────────────────────────
def collide_fn(op: str, f: torch.Tensor, tau: float) -> torch.Tensor:
    """Dispatch D3Q27 CASCADED or CUMULANT (no Smagorinsky)."""
    if op == "cascaded":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        return collide_cascaded_d3q27(f, tau)
    elif op == "cumulant":
        from tensorlbm.cumulant import collide_cumulant_d3q27
        return collide_cumulant_d3q27(f, tau)
    else:
        raise ValueError(f"Unknown collision operator: {op}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 9:
        print("Usage: _d3q27_worker.py <did> <case> <nx> <ny> <nz> <hl> <ns> <op>", flush=True)
        sys.exit(1)

    did = int(sys.argv[1])
    case = sys.argv[2]          # suboff | kvlcc2 | wigley | kcs | cylinder | sphere | suboff_cumulant
    nx = int(sys.argv[3])
    ny = int(sys.argv[4])
    nz = int(sys.argv[5])
    hl = float(sys.argv[6])     # hull length or diameter
    n_steps = int(sys.argv[7])
    op = sys.argv[8]            # cascaded | cumulant

    u_in = 0.06
    re: float
    diameter: float | None = None

    # Reynolds-specific overrides for bluff bodies
    if case == "cylinder":
        re = 200.0
        diameter = hl
        u_in = 0.08
    elif case == "sphere":
        re = 1000.0
        diameter = hl
        u_in = 0.08
    else:
        # Ship/SUBOFF
        re = 2e6

    # Kinematic viscosity and tau
    if case in ("cylinder", "sphere"):
        nu = u_in * diameter / re
    else:  # ship hulls / SUBOFF
        nu = u_in * hl / re

    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] D3Q27-{op.upper()} {case} {nx}x{ny}x{nz}"
    print(f"{tag} hl={hl:.1f} u_in={u_in} Re={re:.0f} nu={nu:.2e} tau={tau:.6f}", flush=True)

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
        # Default placement: length=hl, beam=ny*0.25, draft=nz*0.3
        solid, stats = build_hull_mask(
            hull_type=ht, nx=nx, ny=ny, nz=nz,
            length=hl, device=str(device),
        )
        S = float(solid.sum().item())  # rough — use voxel count as wetted proxy
        A_ref = S
        print(f"{tag} solid_cells={S:.0f} stats={stats.get('Cb_numerical', '?')}", flush=True)
    elif case == "cylinder":
        radius_cyl = diameter / 2.0
        cx_cyl, cy_cyl = nx * 0.25, ny * 0.5
        solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, radius_cyl, device)
        A_ref = diameter * nz   # D × span
    elif case == "sphere":
        radius_sph = diameter / 2.0
        cx_sph, cy_sph, cz_sph = nx * 0.25, ny * 0.5, nz * 0.5
        solid = build_sphere_mask(nx, ny, nz, cx_sph, cy_sph, cz_sph, radius_sph, device)
        A_ref = math.pi * radius_sph ** 2

    # Dynamic pressure reference
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref
    print(f"{tag} A_ref={A_ref:.0f} dyn_p={dyn_p:.6e} init done ({time.time()-t0:.1f}s)", flush=True)

    # ── Initialize distributions ───────────────────────────────────────
    from tensorlbm.d3q27 import equilibrium27, correct_mass27

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    # ── Simulation loop ────────────────────────────────────────────────
    from tensorlbm.wall_model import wall_function_d3q27

    warmup = n_steps // 3
    slide = 500  # sliding window
    fric_hist: list[float] = []
    pres_hist: list[float] = []
    all_finite = True

    for step in range(1, n_steps + 1):
        # 1. Collision
        try:
            f = collide_fn(op, f, tau)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at step {step}: {e}", flush=True)
            all_finite = False
            break

        # 2. Stream
        f = stream27_roll(f)

        # 3. Wall function (log-law body force + drag computation)
        f, df, dp = wall_function_d3q27(f, solid, nu, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_27(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass27(f, initial_mass)

        # Record drag after warmup
        if step > warmup and math.isfinite(df) and math.isfinite(dp):
            fric_hist.append(df)
            pres_hist.append(dp)

        # Divergence check
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            all_finite = False
            break

        # Periodically log
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
    # Full post-warmup average
    cf = sum(fric_hist) / max(len(fric_hist), 1) / dyn_p if fric_hist else 0.0
    cp = sum(pres_hist) / max(len(pres_hist), 1) / dyn_p if pres_hist else 0.0
    ct_total = cf + cp

    # Sliding window stats
    wf = fric_hist[-slide:] if len(fric_hist) >= slide else fric_hist
    wp = pres_hist[-slide:] if len(pres_hist) >= slide else pres_hist
    cf_slide = sum(wf) / max(len(wf), 1) / dyn_p if wf else 0.0
    cp_slide = sum(wp) / max(len(wp), 1) / dyn_p if wp else 0.0
    ct_slide = cf_slide + cp_slide

    # Compute std for sliding window
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
    else:  # ships
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

    print(f"{tag} DONE Ct_full={ct_total:.5f} Ct_slide={ct_slide:.5f}±{ct_slide_std:.5f} "
          f"err={err_pct if math.isfinite(err_pct) else 'N/A' if err_pct is not None else 'N/A'} "
          f"({elapsed:.0f}s)", flush=True)

    # ── Aggregate output ───────────────────────────────────────────────
    out_path = Path("/tmp/d3q27_bench_results.json")
    if out_path.exists():
        try:
            all_results = json.loads(out_path.read_text())
        except Exception:
            all_results = []
    else:
        all_results = []

    all_results.append(result)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"{tag} wrote result to {out_path}", flush=True)


if __name__ == "__main__":
    main()
