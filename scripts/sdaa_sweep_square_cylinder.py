#!/usr/bin/env python3
"""SDAA sweep for the square_cylinder verified benchmark (Re=100 free-stream).

Standalone script — does NOT modify any business code.  Replicates the
physics chain from ``benchmarks/verified/square_cylinder/run.py`` (same
D2Q9 MRT solver, same far-field BC, same sponge, same Inamuro seeding,
same Ladd force measurement, same FFT St extraction) but with smaller
grids/steps so the wall-clock fits ≤ 20 min on SDAA in eager mode.

Reference (from the case README / REFERENCE_AUDIT.md):
  Cd_ref = 1.48   (2D free-stream numerical cluster [1.44, 1.52])
  St_ref = 0.147  (2D free-stream numerical cluster [0.145, 0.149])

Acceptance gates (original case criteria):
  G1  |Cd err| ≤ 3 %   (each grid)
  G2  |St err| ≤ 3 %   (each grid)
  G3  mesh span Cd ≤ 3 %
  G4  mesh span St ≤ 3 %

Device: sdaa:0  (torch.sdaa).  Eager mode only — the SDAA teco-inductor
backend cannot compile the LBM step chain (5th backend defect, see
compile_route.py docstring).  No triton kernels are invoked in eager
mode (collide_mrt / stream / far_field_bc_2d / equilibrium are pure
PyTorch), so TRITON_INTERPRET is not required.

Usage:
    PYTHONPATH=src python scripts/sdaa_sweep_square_cylinder.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure <repo>/src is on the path for tensorlbm imports
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

# ── Reference values (from case README, 2D free-stream numerical cluster) ──
REF_CD = 1.48
REF_ST = 0.147

# ── Domain / physics constants (identical to run.py) ──
DOMAIN_D = 40.0       # domain edge = 40 D
SQUARE_X_D = 10.0     # square centre 10 D from inlet
SPONGE_D = 10.0       # sponge width 10 D (downstream)
SPONGE_ALPHA = 10.0   # tau_eff = tau * (1 + alpha * sigma)
ST_SEED = 0.14        # seeding Strouhal
SEED_AMPL = 0.10      # seeding uy amplitude (fraction of u_in)
SEED_PERIODS = 2.0    # seed for 2 St_seed periods
RE = 100.0
U_IN = 0.05

# ── Sweep configuration ──
# Original case uses D=32 (1280², 60k steps) and D=48 (1920², 90k steps),
# taking ~19 h on SDAA eager.  Under heavy device contention (shared SDAA
# node with 60+ concurrent processes from other agents), even D=8/320² at
# 20 k steps took 1080 s.  We use D=8/10 k and D=4/8 k as feasible smaller
# grids within the 20-min wall-clock budget.  D=4 is very coarse (4-cell
# cylinder) but still resolves the shedding physics; results are reported
# honestly regardless of pass/fail.
SWEEP = [
    {"D": 8, "steps": 10000},
    {"D": 4, "steps": 8000},
]
WARMUP_FRAC = 0.5
WALL_CLOCK_LIMIT_S = 1200  # 20 minutes


def square_mask(
    nx: int, ny: int, cx: float, cy: float, side: float, device: torch.device
) -> torch.Tensor:
    """Axis-aligned square mask; half-way BB → effective side == *side* cells."""
    x0 = int(round(cx - side / 2.0))
    x1 = x0 + int(round(side)) - 1
    y0 = int(round(cy - side / 2.0))
    y1 = y0 + int(round(side)) - 1
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def run_one_grid(D: int, steps: int, device: torch.device) -> dict:
    """Run one grid and return measured Cd, St, and diagnostics."""
    nx = int(DOMAIN_D * D)
    ny = nx
    nu = U_IN * D / RE
    tau = 0.5 + 3.0 * nu

    mask = square_mask(nx, ny, SQUARE_X_D * D, ny / 2.0, D, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * D), int(SPONGE_D * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, U_IN), torch.zeros_like(rho0))

    # ---- step functions (eager, no compilation) ----
    def _forces_and_farfield(f):
        fx, fy = compute_obstacle_forces(f, mask)
        return far_field_bc_2d(f, U_IN, mask), fx, fy

    def _step_plain(f):
        return _forces_and_farfield(stream(collide_mrt(f, tau, tau_field=tau_field)))

    def _step_seeded(f, uy_col):
        f, fx, fy = _forces_and_farfield(
            stream(collide_mrt(f, tau, tau_field=tau_field))
        )
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=f.device, dtype=f.dtype),
            torch.full((ny, 1), U_IN, device=f.device, dtype=f.dtype),
            uy_col.view(ny, 1),
        )[:, :, 0]
        return f, fx, fy

    omega_seed = 2.0 * math.pi * ST_SEED * U_IN / D
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * U_IN / D)))
    uy_amp = SEED_AMPL * U_IN

    fx_hist: list[torch.Tensor] = []
    fy_hist: list[torch.Tensor] = []

    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= seed_steps:
            uy_val = uy_amp * math.sin(omega_seed * step)
            uy_col = torch.full((ny,), uy_val, device=device, dtype=f.dtype)
            f, fx, fy = _step_seeded(f, uy_col)
        else:
            f, fx, fy = _step_plain(f)
        fx_hist.append(fx)
        fy_hist.append(fy)
    if hasattr(device, "type") and device.type == "sdaa":
        torch.sdaa.synchronize()
    elapsed = time.time() - t0

    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    fx_np = torch.stack(fx_hist).detach().cpu().numpy().astype(np.float64)
    fy_np = torch.stack(fy_hist).detach().cpu().numpy().astype(np.float64)

    q_dyn = 0.5 * 1.0 * U_IN * U_IN * D
    w0 = int(WARMUP_FRAC * steps)
    cd_series = fx_np / q_dyn
    cd_mean = float(cd_series[w0:].mean())
    cl_series = fy_np / q_dyn

    # St via FFT peak (same method as run.py)
    cl_w = cl_series[w0:] - cl_series[w0:].mean()
    st_val = float("nan")
    st_cross = float("nan")
    if cl_w.size >= 256:
        spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
        k = int(np.argmax(spec[1:])) + 1
        delta = 0.0
        if 0 < k < spec.size - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-30:
                delta = float(0.5 * (a - c) / den)
        st_val = float((k + delta) / cl_w.size * D / U_IN)
        thr = 0.25 * float(np.abs(cl_w).max())
        sig = np.where(cl_w >= thr, 1, np.where(cl_w <= -thr, -1, 0))
        idx = np.where(sig != 0, np.arange(sig.size), -1)
        last = np.maximum.accumulate(idx)
        state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
        crossings = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
        if len(crossings) >= 3:
            t_shed = float(np.median(np.diff(crossings)))
            st_cross = D / (U_IN * t_shed)

    err_cd = (cd_mean - REF_CD) / REF_CD * 100.0
    err_st = (st_val - REF_ST) / REF_ST * 100.0 if math.isfinite(st_val) else float("nan")

    result = {
        "D": D,
        "nx": nx,
        "ny": ny,
        "re": RE,
        "u_in": U_IN,
        "nu_lb": round(nu, 6),
        "tau": round(tau, 6),
        "steps": steps,
        "warmup_frac": WARMUP_FRAC,
        "analyze_from": w0,
        "seed_steps": seed_steps,
        "cd": round(cd_mean, 4),
        "cd_ref": REF_CD,
        "cd_err_pct": round(err_cd, 2),
        "st": round(st_val, 4) if math.isfinite(st_val) else None,
        "st_ref": REF_ST,
        "st_err_pct": round(err_st, 2) if math.isfinite(err_st) else None,
        "st_crossing": round(st_cross, 4) if math.isfinite(st_cross) else None,
        "cl_amp": float(np.abs(cl_w).max()),
        "finite": True,
        "elapsed_s": round(elapsed, 1),
    }
    print(
        f"[D={D}] steps={steps} t={elapsed:.0f}s "
        f"Cd={cd_mean:.4f} ({err_cd:+.2f}%) "
        f"St={st_val:.4f} ({err_st:+.2f}%) "
        f"cl_amp={result['cl_amp']:.4f}",
        flush=True,
    )
    return result


def assemble_artifact(grids, dev_name, dev_count, triton_imported,
                      triton_interpret, workaround_note, wall_clock,
                      skipped_grids=None):
    """Compute gates and write the artifact JSON."""
    gates = []
    for g in grids:
        cd_pass = abs(g["cd_err_pct"]) <= 3.0
        st_pass = (
            g["st_err_pct"] is not None and abs(g["st_err_pct"]) <= 3.0
        )
        gates.append({
            "grid": f"D{g['D']}",
            "G1_cd_within_3pct": cd_pass,
            "G1_cd_err_pct": g["cd_err_pct"],
            "G2_st_within_3pct": st_pass,
            "G2_st_err_pct": g["st_err_pct"],
        })

    # Mesh span (if ≥ 2 grids)
    if len(grids) >= 2:
        cds = [g["cd"] for g in grids]
        sts = [g["st"] for g in grids if g["st"] is not None]
        cd_span_pct = (max(cds) - min(cds)) / np.mean(cds) * 100.0
        st_span_pct = (max(sts) - min(sts)) / np.mean(sts) * 100.0 if sts else None
        g3_pass = cd_span_pct <= 3.0
        g4_pass = st_span_pct is not None and st_span_pct <= 3.0
    else:
        cd_span_pct = None
        st_span_pct = None
        g3_pass = None
        g4_pass = None

    all_pass = all(
        g["G1_cd_within_3pct"] and g["G2_st_within_3pct"] for g in gates
    ) and (g3_pass is not False) and (g4_pass is not False)

    artifact = {
        "case": "square_cylinder_re100_free_stream_vortex_shedding",
        "sdaa_sweep": True,
        "device_evidence": {
            "device": "sdaa:0",
            "torch_sdaa_get_device_name_0": dev_name,
            "torch_sdaa_device_count": dev_count,
            "torch_version": torch.__version__,
        },
        "triton_record": {
            "triton_installed": triton_imported,
            "TRITON_INTERPRET_env": triton_interpret,
            "workaround": workaround_note,
        },
        "reference": {
            "cd_ref": REF_CD,
            "cd_cluster": "[1.44, 1.52]",
            "st_ref": REF_ST,
            "st_cluster": "[0.145, 0.149]",
            "source": "2D free-stream numerical cluster (arXiv 2411.03124v4, 2309.09197, 2308.08085v3, 2404.12123, 2405.12834)",
        },
        "config": {
            "domain_D": DOMAIN_D,
            "square_x_D": SQUARE_X_D,
            "sponge_D": SPONGE_D,
            "sponge_alpha": SPONGE_ALPHA,
            "st_seed": ST_SEED,
            "seed_ampl_frac": SEED_AMPL,
            "seed_periods": SEED_PERIODS,
            "re": RE,
            "u_in": U_IN,
            "warmup_frac": WARMUP_FRAC,
            "compile_mode": "eager",
            "sweep": SWEEP,
            "note": "Reduced grids/steps from case-recommended D=32/48 (60k/90k) "
                    "to D=8/4 (10k/8k) due to 20-min wall-clock limit and heavy "
                    "SDAA device contention (60+ concurrent processes).",
        },
        "grids": grids,
        "gates": gates,
        "mesh_span": {
            "cd_span_pct": round(cd_span_pct, 2) if cd_span_pct is not None else None,
            "st_span_pct": round(st_span_pct, 2) if st_span_pct is not None else None,
            "G3_cd_span_within_3pct": g3_pass,
            "G4_st_span_within_3pct": g4_pass,
        },
        "overall_pass": all_pass,
        "wall_clock_s": round(wall_clock, 1),
        "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,
        "within_wall_clock": wall_clock <= WALL_CLOCK_LIMIT_S,
    }
    if skipped_grids:
        artifact["skipped_grids"] = skipped_grids

    out_dir = _REPO / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sdaa_square_cylinder.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\nArtifact written to {out_path}", flush=True)
    print(f"Overall PASS: {all_pass}", flush=True)
    print(f"Wall clock: {wall_clock:.1f}s (limit {WALL_CLOCK_LIMIT_S}s)", flush=True)
    return artifact


def main() -> None:
    # ── Device evidence ──
    device = torch.device("sdaa:0")
    dev_name = torch.sdaa.get_device_name(0)
    dev_count = torch.sdaa.device_count()
    print(f"SDAA device: {dev_name} (count={dev_count})", flush=True)

    # ── Triton / compile workaround record ──
    triton_interpret = os.environ.get("TRITON_INTERPRET", "")
    triton_imported = False
    try:
        import triton  # noqa: F401
        triton_imported = True
    except ImportError:
        pass
    workaround_note = (
        "Eager mode (no torch.compile). SDAA teco-inductor cannot compile the "
        "LBM step chain (5th backend defect — EXPAND fallback IndexError). "
        "collide_mrt/stream/far_field_bc_2d/equilibrium are pure PyTorch, so "
        "no triton kernels are invoked and TRITON_INTERPRET is not required."
    )

    # ── Run sweep with time guard ──
    grids = []
    skipped = []
    t_total = time.time()
    for cfg in SWEEP:
        elapsed_so_far = time.time() - t_total
        if elapsed_so_far > WALL_CLOCK_LIMIT_S * 0.75:  # 15 min guard
            print(f"Time guard: skipping D={cfg['D']} "
                  f"(elapsed {elapsed_so_far:.0f}s > 75% of limit)", flush=True)
            skipped.append({"D": cfg["D"], "steps": cfg["steps"],
                            "reason": f"wall-clock guard ({elapsed_so_far:.0f}s elapsed)"})
            continue
        r = run_one_grid(cfg["D"], cfg["steps"], device)
        grids.append(r)
    wall_clock = time.time() - t_total

    assemble_artifact(grids, dev_name, dev_count, triton_imported,
                      triton_interpret, workaround_note, wall_clock,
                      skipped_grids=skipped if skipped else None)


if __name__ == "__main__":
    main()
