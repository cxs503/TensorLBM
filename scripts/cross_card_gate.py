#!/usr/bin/env python3
"""Cross-card bit-exact reproducibility gate for m10 Couette 3D flow.

Mission M-P5·P1: Extend 2-card cross-card gate to 3-card and 4-card.

Methodology:
  - Uses the verified m10 Couette 3D case (D3Q19 BGK, moving-wall BC).
  - Domain decomposition along x-axis via manual slab management.
  - Card selection via SDAA_VISIBLE_DEVICES environment variable.
  - Runs the SAME initial condition on:
      (a) single-card reference (card 0 in visible set)
      (b) multi-card decomposition (all visible cards)
  - Diagnostic fingerprint = sha256 of gathered f-field after N steps.
  - Bit-exact gate: sha256(multi-card) == sha256(single-card).
  - Performance gate: step-time CV <= 2% within each run.

Usage:
  # 3-card run (cards 14, 15, 27):
  SDAA_VISIBLE_DEVICES=14,15,27 python scripts/cross_card_gate.py --n-cards 3 --steps 100

  # 4-card run (cards 14, 15, 16, 27):
  SDAA_VISIBLE_DEVICES=14,15,16,27 python scripts/cross_card_gate.py --n-cards 4 --steps 100

Output: artifacts/cross_card_<N>card.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure tensorlbm is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS2 = 1.0 / 3.0


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def get_device_info() -> list[dict]:
    """Get info about visible SDAA devices."""
    devices = []
    if hasattr(torch, "sdaa") and torch.sdaa.is_available():
        n = torch.sdaa.device_count()
        for i in range(n):
            devices.append({
                "local_id": i,
                "name": torch.sdaa.get_device_name(i),
            })
    return devices


def moving_wall_bounce_back3d(f_pre, f, wall, u_wall, device):
    """Moving wall bounce-back boundary condition for Couette flow."""
    opp = OPPOSITE.to(device)
    c = C.to(device)
    w = W_EXACT64.to(device=device, dtype=f.dtype)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(19, 1, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(19, 1, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def tensor_sha256(t: torch.Tensor) -> str:
    """Compute SHA256 of a tensor's raw bytes (on CPU, contiguous)."""
    t_cpu = t.detach().cpu().contiguous()
    raw = t_cpu.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def halo_exchange_3d_cross_card(slabs, overlap=1):
    """Halo exchange between slabs on different SDAA cards (via CPU relay)."""
    ov = overlap
    n = len(slabs)
    for i in range(n - 1):
        # Right interior of slab i -> left ghost of slab i+1
        r = slabs[i][:, :, :, -ov - 1:-1].cpu()
        lg = slabs[i + 1][:, :, :, :ov]
        lg.copy_(r.to(lg.device))

        # Left interior of slab i+1 -> right ghost of slab i
        l_ = slabs[i + 1][:, :, :, ov:2 * ov].cpu()
        rg = slabs[i][:, :, :, -ov:]
        rg.copy_(l_.to(rg.device))
    return slabs


def run_single_card(
    H: int, tau: float, U0: float, n_steps: int, device_str: str, seed: int = 42
) -> dict:
    """Run Couette 3D on a single card. Returns diagnostic dict."""
    torch.manual_seed(seed)
    dev = torch.device(device_str)
    dtype = torch.float32

    ny = H + 2
    nx = H
    nz = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0

    # Wall setup
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), dtype=dtype, device=dev)
    u_wall[:, -1, :] = U0

    # Initial condition: analytic ramp
    yy = torch.arange(ny, device=dev, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    f = equilibrium3d(
        torch.ones((nz, ny, nx), dtype=dtype, device=dev),
        u0,
        torch.zeros((nz, ny, nx), dtype=dtype, device=dev),
        torch.zeros((nz, ny, nx), dtype=dtype, device=dev),
    )

    # Time stepping
    step_times = []
    for step in range(n_steps):
        if hasattr(torch, 'synchronize'):
            torch.synchronize(dev)
        t0 = time.perf_counter()
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall, dev)
        f = stream3d(f)
        if hasattr(torch, 'synchronize'):
            torch.synchronize(dev)
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000.0)  # ms

    # Diagnostics
    rho, ux, uy, uz = macroscopic3d(f)
    f_sha = tensor_sha256(f)
    rho_sha = tensor_sha256(rho)
    ux_sha = tensor_sha256(ux)

    # Profile at mid-z, mid-x
    z0, x0 = nz // 2, nx // 2
    u_profile = ux[z0, :, x0].cpu().numpy()

    # Compute CV of step times (excluding first 10% as warmup)
    warmup = max(1, len(step_times) // 10)
    steady_times = step_times[warmup:]
    cv_pct = float(np.std(steady_times) / np.mean(steady_times) * 100.0) if np.mean(steady_times) > 0 else 0.0

    return {
        "mode": "single_card",
        "device": device_str,
        "f_sha256": f_sha,
        "rho_sha256": rho_sha,
        "ux_sha256": ux_sha,
        "u_profile": u_profile.tolist(),
        "step_times_ms": step_times,
        "mean_step_ms": float(np.mean(step_times)),
        "std_step_ms": float(np.std(step_times)),
        "cv_pct": cv_pct,
        "rho_range": [float(rho.min().item()), float(rho.max().item())],
        "ux_range": [float(ux.min().item()), float(ux.max().item())],
        "mass": float(f.sum().item()),
        "finite": bool(torch.isfinite(f).all().item()),
    }


def run_multi_card(
    H: int, tau: float, U0: float, n_steps: int, devices: list[str], seed: int = 42
) -> dict:
    """Run Couette 3D on multiple cards with domain decomposition."""
    torch.manual_seed(seed)
    dtype = torch.float32
    n_cards = len(devices)

    ny = H + 2
    nx = H
    nz = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0

    # Build global initial condition on CPU, then distribute to each card
    yy = torch.arange(ny, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    f_global_cpu = equilibrium3d(
        torch.ones((nz, ny, nx), dtype=dtype),
        u0,
        torch.zeros((nz, ny, nx), dtype=dtype),
        torch.zeros((nz, ny, nx), dtype=dtype),
    )

    # Wall mask (global, on CPU)
    wall_global_cpu = torch.zeros((nz, ny, nx), dtype=torch.bool)
    wall_global_cpu[:, 0, :] = True
    wall_global_cpu[:, -1, :] = True
    u_wall_global_cpu = torch.zeros((nz, ny, nx), dtype=dtype)
    u_wall_global_cpu[:, -1, :] = U0

    # Compute slab decomposition
    base = nx // n_cards
    rem = nx % n_cards
    slabs_x = []
    start = 0
    for i in range(n_cards):
        width = base + (1 if i < rem else 0)
        end = start + width
        slabs_x.append((start, end))
        start = end

    ov = 1  # overlap

    # Distribute f, wall, u_wall to slabs (each from CPU to its device)
    slabs_f = []
    slabs_wall = []
    slabs_uwall = []
    for i, (dev_str, (x0, x1)) in enumerate(zip(devices, slabs_x)):
        dev = torch.device(dev_str)
        x0g = max(0, x0 - ov)
        x1g = min(nx, x1 + ov)
        x_indices = torch.arange(x0g, x1g) % nx
        slab_f = f_global_cpu[:, :, :, x_indices].to(dev).contiguous()
        slab_wall = wall_global_cpu[:, :, x0g:x1g].to(dev)
        slab_uwall = u_wall_global_cpu[:, :, x0g:x1g].to(dev)
        slabs_f.append(slab_f)
        slabs_wall.append(slab_wall)
        slabs_uwall.append(slab_uwall)

    # Initial halo exchange
    halo_exchange_3d_cross_card(slabs_f, ov)

    # Time stepping
    step_times = []

    for step in range(n_steps):
        for dev_str in devices:
            if hasattr(torch, 'synchronize'):
                torch.synchronize(torch.device(dev_str))
        t0 = time.perf_counter()

        # 1. Clone pre-collision state
        slabs_f_pre = [f.clone() for f in slabs_f]

        # 2. Collide on each card
        for i in range(n_cards):
            slabs_f[i] = collide_bgk3d(slabs_f[i], tau)

        # 3. Bounce-back (moving wall) on each card
        for i in range(n_cards):
            slabs_f[i] = moving_wall_bounce_back3d(
                slabs_f_pre[i], slabs_f[i],
                slabs_wall[i], slabs_uwall[i],
                torch.device(devices[i])
            )

        # 4. Halo exchange (via CPU relay)
        halo_exchange_3d_cross_card(slabs_f, ov)

        # 5. Stream on each card
        for i in range(n_cards):
            slabs_f[i] = stream3d(slabs_f[i])

        for dev_str in devices:
            if hasattr(torch, 'synchronize'):
                torch.synchronize(torch.device(dev_str))
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000.0)

    # Gather global field (via CPU)
    dev0 = torch.device(devices[0])
    f_gathered = torch.zeros((19, nz, ny, nx), dtype=dtype)
    for i, (x0, x1) in enumerate(slabs_x):
        local_width = x1 - x0
        x0g_local = ov if x0 > 0 else 0
        slab_interior = slabs_f[i][:, :, :, x0g_local:x0g_local + local_width].cpu()
        f_gathered[:, :, :, x0:x1] = slab_interior
    f_gathered = f_gathered.to(dev0)

    # Diagnostics
    f_sha = tensor_sha256(f_gathered)
    rho, ux, uy, uz = macroscopic3d(f_gathered)
    rho_sha = tensor_sha256(rho)
    ux_sha = tensor_sha256(ux)

    z0, x0 = nz // 2, nx // 2
    u_profile = ux[z0, :, x0].cpu().numpy()

    # CV of step times
    warmup = max(1, len(step_times) // 10)
    steady_times = step_times[warmup:]
    cv_pct = float(np.std(steady_times) / np.mean(steady_times) * 100.0) if np.mean(steady_times) > 0 else 0.0

    return {
        "mode": "multi_card",
        "devices": devices,
        "n_cards": n_cards,
        "f_sha256": f_sha,
        "rho_sha256": rho_sha,
        "ux_sha256": ux_sha,
        "u_profile": u_profile.tolist(),
        "step_times_ms": step_times,
        "mean_step_ms": float(np.mean(step_times)),
        "std_step_ms": float(np.std(step_times)),
        "cv_pct": cv_pct,
        "rho_range": [float(rho.min().item()), float(rho.max().item())],
        "ux_range": [float(ux.min().item()), float(ux.max().item())],
        "mass": float(f_gathered.sum().item()),
        "finite": bool(torch.isfinite(f_gathered).all().item()),
        "slabs": [(int(s), int(e)) for s, e in slabs_x],
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-card gate for m10 Couette 3D")
    ap.add_argument("--n-cards", type=int, required=True, help="Number of cards (3 or 4)")
    ap.add_argument("--steps", type=int, default=100, help="Number of time steps")
    ap.add_argument("--H", type=int, default=32, help="Couette gap height")
    ap.add_argument("--tau", type=float, default=0.8, help="Relaxation time")
    ap.add_argument("--U0", type=float, default=0.05, help="Wall velocity")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    args = ap.parse_args()

    # Check SDAA availability
    if not (hasattr(torch, "sdaa") and torch.sdaa.is_available()):
        print("ERROR: SDAA not available", file=sys.stderr)
        sys.exit(1)

    n_visible = torch.sdaa.device_count()
    if n_visible < args.n_cards:
        print(f"ERROR: Need {args.n_cards} cards but only {n_visible} visible", file=sys.stderr)
        sys.exit(1)

    # Build device list
    devices = [f"sdaa:{i}" for i in range(args.n_cards)]
    device_info = get_device_info()
    git_sha = get_git_sha()

    print(f"=== Cross-Card Gate: m10 Couette 3D ===")
    print(f"Git SHA: {git_sha}")
    print(f"Visible devices: {n_visible}")
    print(f"Using devices: {devices}")
    for d in device_info:
        print(f"  sdaa:{d['local_id']} -> {d['name']}")
    print(f"Grid: H={args.H}, nx={args.H}, ny={args.H+2}, nz={args.H}")
    print(f"Steps: {args.steps}, tau={args.tau}, U0={args.U0}")
    print()

    # Run single-card reference
    print("--- Single-card reference (sdaa:0) ---")
    single_result = run_single_card(
        args.H, args.tau, args.U0, args.steps, "sdaa:0", args.seed
    )
    print(f"  f_sha256: {single_result['f_sha256']}")
    print(f"  mean step: {single_result['mean_step_ms']:.2f} ms (CV={single_result['cv_pct']:.2f}%)")
    print(f"  mass: {single_result['mass']:.10f}")
    print(f"  finite: {single_result['finite']}")
    print()

    # Run multi-card
    print(f"--- Multi-card ({args.n_cards} cards) ---")
    multi_result = run_multi_card(
        args.H, args.tau, args.U0, args.steps, devices, args.seed
    )
    print(f"  f_sha256: {multi_result['f_sha256']}")
    print(f"  mean step: {multi_result['mean_step_ms']:.2f} ms (CV={multi_result['cv_pct']:.2f}%)")
    print(f"  mass: {multi_result['mass']:.10f}")
    print(f"  finite: {multi_result['finite']}")
    print(f"  slabs: {multi_result['slabs']}")
    print()

    # Bit-exact comparison
    sha_match = single_result["f_sha256"] == multi_result["f_sha256"]
    rho_match = single_result["rho_sha256"] == multi_result["rho_sha256"]
    ux_match = single_result["ux_sha256"] == multi_result["ux_sha256"]

    # Profile comparison
    u_single = np.array(single_result["u_profile"])
    u_multi = np.array(multi_result["u_profile"])
    profile_max_abs_err = float(np.max(np.abs(u_single - u_multi)))
    profile_match = profile_max_abs_err == 0.0

    # Performance CV: both runs must have CV <= 2%
    perf_cv_single = single_result["cv_pct"]
    perf_cv_multi = multi_result["cv_pct"]
    perf_pass = perf_cv_single <= 2.0 and perf_cv_multi <= 2.0

    # Mass conservation
    mass_match = abs(single_result["mass"] - multi_result["mass"]) < 1e-6

    print("=== GATE RESULTS ===")
    print(f"  f_sha256 match:    {sha_match}")
    print(f"  rho_sha256 match:  {rho_match}")
    print(f"  ux_sha256 match:   {ux_match}")
    print(f"  profile exact:     {profile_match} (max_abs_err={profile_max_abs_err:.2e})")
    print(f"  mass match:        {mass_match}")
    print(f"  perf CV single:    {perf_cv_single:.2f}% (target <=2%)")
    print(f"  perf CV multi:     {perf_cv_multi:.2f}% (target <=2%)")
    print()

    bit_exact_pass = sha_match and rho_match and ux_match and profile_match
    overall_pass = bit_exact_pass and perf_pass and mass_match

    print(f"  BIT-EXACT: {'PASS' if bit_exact_pass else 'FAIL'}")
    print(f"  PERF(CV<=2%): {'PASS' if perf_pass else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # Build output JSON (ensure all numpy types are converted)
    env_visible = os.environ.get("SDAA_VISIBLE_DEVICES", "")
    output = {
        "mission": "M-P5·P1",
        "case": "m10_couette_3d",
        "git_sha": git_sha,
        "sdaa_visible_devices": env_visible,
        "n_cards_requested": int(args.n_cards),
        "n_cards_visible": int(n_visible),
        "device_info": device_info,
        "params": {
            "H": int(args.H),
            "nx": int(args.H),
            "ny": int(args.H + 2),
            "nz": int(args.H),
            "tau": float(args.tau),
            "U0": float(args.U0),
            "n_steps": int(args.steps),
            "seed": int(args.seed),
        },
        "single_card": {
            "device": "sdaa:0",
            "device_name": device_info[0]["name"] if device_info else "unknown",
            "f_sha256": single_result["f_sha256"],
            "rho_sha256": single_result["rho_sha256"],
            "ux_sha256": single_result["ux_sha256"],
            "mean_step_ms": float(single_result["mean_step_ms"]),
            "std_step_ms": float(single_result["std_step_ms"]),
            "cv_pct": float(single_result["cv_pct"]),
            "mass": float(single_result["mass"]),
            "finite": bool(single_result["finite"]),
            "rho_range": [float(x) for x in single_result["rho_range"]],
            "ux_range": [float(x) for x in single_result["ux_range"]],
        },
        "multi_card": {
            "devices": devices,
            "device_names": [d["name"] for d in device_info[:args.n_cards]],
            "n_cards": int(args.n_cards),
            "f_sha256": multi_result["f_sha256"],
            "rho_sha256": multi_result["rho_sha256"],
            "ux_sha256": multi_result["ux_sha256"],
            "mean_step_ms": float(multi_result["mean_step_ms"]),
            "std_step_ms": float(multi_result["std_step_ms"]),
            "cv_pct": float(multi_result["cv_pct"]),
            "mass": float(multi_result["mass"]),
            "finite": bool(multi_result["finite"]),
            "rho_range": [float(x) for x in multi_result["rho_range"]],
            "ux_range": [float(x) for x in multi_result["ux_range"]],
            "slabs": multi_result["slabs"],
        },
        "gate_results": {
            "f_sha256_match": bool(sha_match),
            "rho_sha256_match": bool(rho_match),
            "ux_sha256_match": bool(ux_match),
            "profile_exact_match": bool(profile_match),
            "profile_max_abs_err": float(profile_max_abs_err),
            "mass_conservation_match": bool(mass_match),
            "perf_cv_single_pct": float(perf_cv_single),
            "perf_cv_multi_pct": float(perf_cv_multi),
            "bit_exact_pass": bool(bit_exact_pass),
            "perf_pass": bool(perf_pass),
            "overall_pass": bool(overall_pass),
        },
        "u_profile_single": [float(x) for x in single_result["u_profile"]],
        "u_profile_multi": [float(x) for x in multi_result["u_profile"]],
    }

    # Write output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cross_card_{args.n_cards}card.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nOutput written to: {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
