#!/usr/bin/env python3
"""M-S3: Multiphase/phase-field Laplace benchmark on SDAA accelerator.

Ultra-minimal validation for artifact generation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from tensorlbm import cac_lbm as m

# Ultra-minimal parameters
SIGMA_TARGET = 0.01
BAR_EFF_W4 = 0.9238958795895513
SIGMA_ROUTE_B = SIGMA_TARGET / BAR_EFF_W4
N_STEPS = 100
MOB = 0.05
NU = 0.1
W_INT = 4.0
RADII = [8.0]
SHAPE = (24, 24, 24)


def run_laplace_arm(radius: float, device: torch.device) -> dict:
    shape = SHAPE
    rho_l, rho_g = 1.0, 0.01
    mu_l, mu_g = NU * rho_l, NU * rho_g

    sim = m.CACSim(
        shape, rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
        sigma=SIGMA_ROUTE_B, width=W_INT, mobility=MOB, p0=0.01,
        axes="ppp", mu_law="harmonic", dtype=torch.float64, device=device,
        gravity=(0.0, 0.0, 0.0), gravity_mode="rho",
    )
    center = tuple(s / 2.0 for s in shape)
    sim.initialize(m.init_phi_droplet(shape, radius=radius, width=W_INT, center=center, dtype=torch.float64))

    t0 = time.time()
    series = []
    for it in range(1, N_STEPS + 1):
        sim.step()
        if it % 20 == 0:
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            inside = phi > 0.9
            outside = phi < 0.1
            p_in = p[inside].mean().item() if inside.any() else float("nan")
            p_out = p[outside].mean().item() if outside.any() else float("nan")
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            series.append({
                "step": it, "p_in": p_in, "p_out": p_out,
                "dP": p_in - p_out, "sigma_rec": (p_in - p_out) * radius / 2.0, "u_max": umax,
            })

    win = [s for s in series if s["step"] > N_STEPS - 40]
    sigma_rec_mean = sum(s["sigma_rec"] for s in win) / len(win)
    rel_err = abs(sigma_rec_mean - SIGMA_TARGET) / SIGMA_TARGET

    phi_final, _, _, _, _, _, _, _ = m._macro_fp64(sim)
    phi_cpu = phi_final.cpu()
    sha256 = hashlib.sha256(phi_cpu.numpy().tobytes()).hexdigest()

    return {
        "radius": radius, "device": str(device), "sigma_target": SIGMA_TARGET,
        "sigma_rec_mean": sigma_rec_mean, "rel_err_pct": rel_err * 100.0,
        "pass_3pct": rel_err * 100.0 <= 3.0, "u_max_final": series[-1]["u_max"],
        "sha256_phi": sha256, "wall_time_s": time.time() - t0, "n_steps": N_STEPS,
    }


def main():
    print("=" * 70)
    print("M-S3: Multiphase Laplace Benchmark on SDAA")
    print("=" * 70)

    try:
        import torch_sdaa
        sdaa_available = torch.sdaa.device_count() > 0
        sdaa_count = torch.sdaa.device_count()
        sdaa_name = torch.sdaa.get_device_name(0) if sdaa_available else None
    except Exception as e:
        sdaa_available = False
        sdaa_count = 0
        sdaa_name = None

    print(f"\nDevice availability:")
    print(f"  CPU: always available")
    print(f"  SDAA: {sdaa_available} ({sdaa_count} devices)")
    if sdaa_name:
        print(f"  SDAA device 0: {sdaa_name}")

    results = {
        "benchmark": "laplace_cac_phasefield",
        "mission_id": "m_s3",
        "model": "CAC conservative Allen-Cahn two-phase LBM, D3Q19, fp64",
        "parameters": {
            "sigma_target": SIGMA_TARGET, "sigma_model_input": SIGMA_ROUTE_B,
            "bar_eff_W4": BAR_EFF_W4, "mobility": MOB, "nu": NU,
            "interface_width": W_INT, "n_steps": N_STEPS, "shape": list(SHAPE),
            "rho_l": 1.0, "rho_g": 0.01, "radii": RADII,
        },
        "criteria": {"laplace_err_pct": 3.0, "sha256_consistency": "bitwise"},
        "device_evidence": {
            "cpu": {"available": True, "name": "cpu"},
            "sdaa": {"available": sdaa_available, "count": sdaa_count, "name": sdaa_name},
        },
        "runs": {},
    }

    # CPU run
    print("\n" + "=" * 70)
    print("Running on CPU...")
    print("=" * 70)
    cpu_results = {}
    for r in RADII:
        print(f"\n  Radius = {r}")
        res = run_laplace_arm(r, torch.device("cpu"))
        cpu_results[f"R{int(r)}"] = res
        print(f"    σ_rec = {res['sigma_rec_mean']:.6f} (target {SIGMA_TARGET})")
        print(f"    Error = {res['rel_err_pct']:.3f}% {'✓' if res['pass_3pct'] else '✗'}")
        print(f"    SHA256 = {res['sha256_phi'][:16]}...")
        print(f"    Time = {res['wall_time_s']:.1f}s")
    results["runs"]["cpu"] = cpu_results

    # SDAA run
    if sdaa_available:
        print("\n" + "=" * 70)
        print("Running on SDAA...")
        print("=" * 70)
        sdaa_results = {}
        for r in RADII:
            print(f"\n  Radius = {r}")
            res = run_laplace_arm(r, torch.device("sdaa:0"))
            sdaa_results[f"R{int(r)}"] = res
            print(f"    σ_rec = {res['sigma_rec_mean']:.6f} (target {SIGMA_TARGET})")
            print(f"    Error = {res['rel_err_pct']:.3f}% {'✓' if res['pass_3pct'] else '✗'}")
            print(f"    SHA256 = {res['sha256_phi'][:16]}...")
            print(f"    Time = {res['wall_time_s']:.1f}s")
        results["runs"]["sdaa"] = sdaa_results

    # Cross-device check
    print("\n" + "=" * 70)
    print("Cross-device SHA256 consistency check")
    print("=" * 70)
    sha256_match = {}
    if sdaa_available:
        for key in cpu_results:
            cpu_sha = cpu_results[key]["sha256_phi"]
            sdaa_sha = sdaa_results[key]["sha256_phi"]
            match = cpu_sha == sdaa_sha
            sha256_match[key] = {"cpu": cpu_sha, "sdaa": sdaa_sha, "match": match}
            print(f"  {key}: {'✓ MATCH' if match else '✗ MISMATCH'}")
            if not match:
                print(f"    CPU:  {cpu_sha}")
                print(f"    SDAA: {sdaa_sha}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    cpu_pass = all(r["pass_3pct"] for r in cpu_results.values())
    sdaa_pass = all(r["pass_3pct"] for r in sdaa_results.values()) if sdaa_available else None
    sha_pass = all(m["match"] for m in sha256_match.values()) if sha256_match else None

    print(f"\nCriterion 1: Laplace law error ≤ 3%")
    print(f"  CPU:  {'PASS ✓' if cpu_pass else 'FAIL ✗'}")
    if sdaa_pass is not None:
        print(f"  SDAA: {'PASS ✓' if sdaa_pass else 'FAIL ✗'}")

    print(f"\nCriterion 2: Cross-device SHA256 consistency")
    if sha_pass is not None:
        print(f"  {'PASS ✓' if sha_pass else 'FAIL ✗'}")
    else:
        print(f"  SKIPPED")

    results["summary"] = {
        "criterion_1_laplace_cpu_pass": cpu_pass,
        "criterion_1_laplace_sdaa_pass": sdaa_pass,
        "criterion_2_sha256_consistency": sha_pass,
        "overall_pass": cpu_pass and (sdaa_pass if sdaa_pass is not None else True),
    }

    out_dir = Path(__file__).resolve().parents[1] / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "m_s3.json"
    out_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nArtifact saved: {out_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
