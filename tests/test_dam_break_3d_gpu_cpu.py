#!/usr/bin/env python3
"""M-S2·P2: Dam Break 3D GPU vs CPU comparison.

This script:
1. Runs CPU baseline with fixed seed
2. Runs GPU (SDAA) comparison with same seed
3. Computes L2 difference in liquid height field
4. Measures mass drift over time
5. Writes results to artifacts/m_s2.json
"""

import json
import math
import time
from pathlib import Path

import torch
import numpy as np

from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d


def compute_height_field(run_dir: Path, device: str = "cpu") -> torch.Tensor:
    """Extract liquid height field from final state.
    
    Returns a 2D tensor (ny, nx) with liquid height at each (x,y) column.
    """
    # Load final state
    final_file = run_dir / "final_state.pt"
    if final_file.exists():
        state = torch.load(final_file, map_location=device)
        # Assuming state contains 'f_water' or similar
        # For now, we'll use the diagnostics
        pass
    
    # Alternative: load from diagnostics CSV
    diag_file = run_dir / "diagnostics.csv"
    if not diag_file.exists():
        raise FileNotFoundError(f"No diagnostics found in {run_dir}")
    
    # Parse diagnostics to get front position and mass
    import csv
    with open(diag_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # For height field comparison, we need the actual field data
    # Let's check if there's a field dump
    field_file = run_dir / "fields_final.npz"
    if field_file.exists():
        data = np.load(field_file)
        # Assuming 'rho' or 'fill' field
        if 'rho' in data:
            rho = data['rho']
            # Height field: integrate density in z-direction
            height = rho.sum(axis=0)  # (ny, nx)
            return torch.from_numpy(height)
    
    # Fallback: use metadata
    meta_file = run_dir / "run_metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        # Extract what we can
        return None
    
    return None


def compute_mass_drift(run_dir: Path) -> dict:
    """Compute mass drift from diagnostics."""
    diag_file = run_dir / "diagnostics.csv"
    if not diag_file.exists():
        return {"error": "no diagnostics"}
    
    import csv
    with open(diag_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {"error": "empty diagnostics"}
    
    # Extract mass history
    masses = []
    for row in rows:
        if 'mass' in row:
            masses.append(float(row['mass']))
        elif 'mean_rho' in row:
            masses.append(float(row['mean_rho']))
    
    if len(masses) < 2:
        return {"error": "insufficient mass data"}
    
    # Compute drift
    m0 = masses[0]
    m_final = masses[-1]
    drift_abs = m_final - m0
    drift_rel = drift_abs / m0 if m0 > 0 else 0.0
    
    # Compute max drift over time
    max_drift = max(abs(m - m0) for m in masses)
    max_drift_rel = max_drift / m0 if m0 > 0 else 0.0
    
    return {
        "initial_mass": m0,
        "final_mass": m_final,
        "drift_abs": drift_abs,
        "drift_rel": drift_rel,
        "max_drift_abs": max_drift,
        "max_drift_rel": max_drift_rel,
        "n_samples": len(masses),
    }


def compute_l2_diff(field1: torch.Tensor, field2: torch.Tensor) -> dict:
    """Compute L2 difference between two fields."""
    if field1 is None or field2 is None:
        return {"error": "missing field data"}
    
    # Ensure same shape
    if field1.shape != field2.shape:
        return {"error": f"shape mismatch: {field1.shape} vs {field2.shape}"}
    
    diff = field1 - field2
    l2_norm = torch.sqrt(torch.sum(diff ** 2)).item()
    l2_rel = l2_norm / torch.sqrt(torch.sum(field1 ** 2)).item()
    
    # Also compute max absolute difference
    max_diff = torch.max(torch.abs(diff)).item()
    
    return {
        "l2_norm": l2_norm,
        "l2_relative": l2_rel,
        "max_abs_diff": max_diff,
        "shape": list(field1.shape),
    }


def run_comparison():
    """Run CPU baseline and GPU comparison."""
    
    # Configuration - small grid for speed
    cfg_common = {
        "nx": 80,
        "ny": 30,
        "nz": 30,
        "dam_width": 30,
        "fill_height": 29,
        "model": "cg",
        "gravity": 5e-5,
        "n_steps": 1000,  # Reduced for speed
        "output_interval": 100,
        "overwrite": True,
    }
    
    output_root = Path("/tmp/dam_break_3d_comparison")
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = {
        "mission_id": 364,
        "task": "M-S2·P2: Dam Break 3D GPU vs CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg_common,
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ===== CPU Baseline =====
    print("=" * 70)
    print("Running CPU baseline...")
    print("=" * 70)
    
    cfg_cpu = DamBreak3DConfig(
        **cfg_common,
        device="cpu",
        output_root=output_root,
        run_name="cpu_baseline",
    )
    
    t0 = time.perf_counter()
    run_dir_cpu = run_dam_break_3d(cfg_cpu)
    time_cpu = time.perf_counter() - t0
    
    print(f"CPU run completed in {time_cpu:.2f}s")
    print(f"Output: {run_dir_cpu}")
    
    results["cpu"] = {
        "device": "cpu",
        "run_dir": str(run_dir_cpu),
        "elapsed_s": time_cpu,
    }
    
    # Compute CPU mass drift
    cpu_mass = compute_mass_drift(run_dir_cpu)
    results["cpu"]["mass_drift"] = cpu_mass
    print(f"CPU mass drift: {cpu_mass}")
    
    # ===== GPU (SDAA) Comparison =====
    print("\n" + "=" * 70)
    print("Running GPU (SDAA) comparison...")
    print("=" * 70)
    
    # Check if SDAA is available
    sdaa_available = torch.cuda.is_available()  # SDAA uses CUDA interface
    if not sdaa_available:
        print("WARNING: SDAA/CUDA not available, skipping GPU comparison")
        results["gpu"] = {"status": "skipped", "reason": "SDAA not available"}
    else:
        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        cfg_gpu = DamBreak3DConfig(
            **cfg_common,
            device="sdaa",
            output_root=output_root,
            run_name="gpu_sdaa",
        )
        
        t0 = time.perf_counter()
        run_dir_gpu = run_dam_break_3d(cfg_gpu)
        time_gpu = time.perf_counter() - t0
        
        print(f"GPU run completed in {time_gpu:.2f}s")
        print(f"Output: {run_dir_gpu}")
        
        results["gpu"] = {
            "device": "sdaa",
            "run_dir": str(run_dir_gpu),
            "elapsed_s": time_gpu,
        }
        
        # Compute GPU mass drift
        gpu_mass = compute_mass_drift(run_dir_gpu)
        results["gpu"]["mass_drift"] = gpu_mass
        print(f"GPU mass drift: {gpu_mass}")
        
        # ===== Compare height fields =====
        print("\n" + "=" * 70)
        print("Comparing height fields...")
        print("=" * 70)
        
        # Try to load height fields
        try:
            height_cpu = compute_height_field(run_dir_cpu, "cpu")
            height_gpu = compute_height_field(run_dir_gpu, "sdaa")
            
            if height_cpu is not None and height_gpu is not None:
                l2_diff = compute_l2_diff(height_cpu, height_gpu)
                results["comparison"] = l2_diff
                print(f"L2 difference: {l2_diff}")
                
                # Check criterion: L2 ≤ 1%
                if "l2_relative" in l2_diff:
                    l2_rel = l2_diff["l2_relative"]
                    results["pass_l2_criterion"] = l2_rel <= 0.01
                    print(f"L2 relative: {l2_rel:.6f} (threshold: 0.01)")
                    print(f"PASS: {results['pass_l2_criterion']}")
            else:
                results["comparison"] = {"error": "could not extract height fields"}
                print("Could not extract height fields for comparison")
        except Exception as e:
            results["comparison"] = {"error": str(e)}
            print(f"Error comparing fields: {e}")
    
    # ===== Mass drift threshold =====
    print("\n" + "=" * 70)
    print("Mass drift analysis...")
    print("=" * 70)
    
    # Lock threshold based on CPU baseline
    if "mass_drift" in results["cpu"] and "max_drift_rel" in results["cpu"]["mass_drift"]:
        cpu_drift = results["cpu"]["mass_drift"]["max_drift_rel"]
        # Target: ≤0.1% per 1e4 steps
        # We ran 1000 steps, so scale accordingly
        drift_per_1e4 = cpu_drift * (10000 / cfg_common["n_steps"])
        
        results["mass_drift_threshold"] = {
            "cpu_baseline_drift": cpu_drift,
            "steps_run": cfg_common["n_steps"],
            "extrapolated_per_1e4_steps": drift_per_1e4,
            "target_threshold": 0.001,  # 0.1%
            "pass": drift_per_1e4 <= 0.001,
        }
        
        print(f"CPU baseline drift: {cpu_drift:.6f} ({cpu_drift*100:.4f}%)")
        print(f"Extrapolated to 1e4 steps: {drift_per_1e4:.6f} ({drift_per_1e4*100:.4f}%)")
        print(f"Target threshold: 0.001 (0.1%)")
        print(f"PASS: {results['mass_drift_threshold']['pass']}")
    
    # ===== Save results =====
    artifact_path = Path("artifacts/m_s2.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {artifact_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_comparison()
