#!/usr/bin/env python3
"""M-S2·P2: Dam Break 3D - CPU baseline and GPU comparison attempt.

Given environment constraints (no SDAA/CUDA available), this script:
1. Runs CPU baseline with minimal configuration
2. Attempts GPU comparison (will be skipped if unavailable)
3. Documents findings in artifacts/m_s2.json
"""

import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np


def check_device_availability():
    """Check available compute devices."""
    devices = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "sdaa": False,  # Will check via CUDA interface
    }
    
    if torch.cuda.is_available():
        devices["cuda_count"] = torch.cuda.device_count()
        devices["cuda_name"] = torch.cuda.get_device_name(0)
        # SDAA typically uses CUDA interface
        devices["sdaa"] = True
    
    return devices


def run_minimal_dam_break(device: str, output_dir: Path, n_steps: int = 50):
    """Run a minimal dam break simulation for testing."""
    from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d
    
    # Minimal configuration
    cfg = DamBreak3DConfig(
        nx=40,
        ny=20,
        nz=20,
        dam_width=15,
        fill_height=19,
        model="cg",
        gravity=5e-5,
        n_steps=n_steps,
        output_interval=25,
        device=device,
        output_root=output_dir,
        run_name=f"test_{device}",
        overwrite=True,
    )
    
    t0 = time.perf_counter()
    run_dir = run_dam_break_3d(cfg)
    elapsed = time.perf_counter() - t0
    
    return run_dir, elapsed


def analyze_run(run_dir: Path):
    """Analyze a completed run."""
    import csv
    
    results = {
        "run_dir": str(run_dir),
        "files": [],
    }
    
    # List files
    for f in sorted(run_dir.iterdir()):
        results["files"].append(f.name)
    
    # Try to load metadata
    meta_file = run_dir / "run_metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        results["metadata"] = meta
    
    # Try to load diagnostics
    diag_file = run_dir / "diagnostics.csv"
    if diag_file.exists():
        with open(diag_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if rows:
            # Extract mass history if available
            masses = []
            for row in rows:
                if 'mass' in row:
                    masses.append(float(row['mass']))
                elif 'mean_rho' in row:
                    masses.append(float(row['mean_rho']))
            
            if masses:
                m0 = masses[0]
                m_final = masses[-1]
                drift = abs(m_final - m0) / m0 if m0 > 0 else 0.0
                
                results["mass_drift"] = {
                    "initial": m0,
                    "final": m_final,
                    "relative_drift": drift,
                    "n_samples": len(masses),
                }
    
    return results


def main():
    """Main entry point."""
    print("=" * 70)
    print("M-S2·P2: Dam Break 3D GPU vs CPU Comparison")
    print("=" * 70)
    
    # Check devices
    print("\nChecking device availability...")
    devices = check_device_availability()
    print(f"  CPU: {devices['cpu']}")
    print(f"  CUDA: {devices['cuda']}")
    print(f"  SDAA: {devices['sdaa']}")
    
    if devices['cuda']:
        print(f"  CUDA devices: {devices.get('cuda_count', 0)}")
        if devices.get('cuda_name'):
            print(f"  CUDA device name: {devices['cuda_name']}")
    
    # Prepare output
    output_root = Path("/tmp/dam_break_3d_m_s2")
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = {
        "mission_id": 364,
        "task": "M-S2·P2: Dam Break 3D GPU vs CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "devices": devices,
        "status": "partial",
    }
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ===== CPU Baseline =====
    print("\n" + "=" * 70)
    print("Running CPU baseline (50 steps)...")
    print("=" * 70)
    
    try:
        run_dir_cpu, time_cpu = run_minimal_dam_break("cpu", output_root, n_steps=50)
        print(f"✓ CPU run completed in {time_cpu:.2f}s")
        print(f"  Output: {run_dir_cpu}")
        
        results["cpu"] = {
            "status": "success",
            "device": "cpu",
            "run_dir": str(run_dir_cpu),
            "elapsed_s": time_cpu,
        }
        
        # Analyze CPU run
        cpu_analysis = analyze_run(run_dir_cpu)
        results["cpu"]["analysis"] = cpu_analysis
        
        if "mass_drift" in cpu_analysis:
            print(f"  Mass drift: {cpu_analysis['mass_drift']['relative_drift']:.6f}")
        
    except Exception as e:
        print(f"✗ CPU run failed: {e}")
        results["cpu"] = {
            "status": "failed",
            "error": str(e),
        }
    
    # ===== GPU Comparison =====
    print("\n" + "=" * 70)
    print("Attempting GPU (SDAA/CUDA) comparison...")
    print("=" * 70)
    
    if not devices['cuda'] and not devices['sdaa']:
        print("⚠ No GPU available - skipping GPU comparison")
        results["gpu"] = {
            "status": "skipped",
            "reason": "No CUDA/SDAA device available in this environment",
        }
        results["comparison"] = {
            "status": "not_performed",
            "reason": "GPU not available",
        }
    else:
        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        device = "sdaa" if devices['sdaa'] else "cuda"
        
        try:
            run_dir_gpu, time_gpu = run_minimal_dam_break(device, output_root, n_steps=50)
            print(f"✓ GPU run completed in {time_gpu:.2f}s")
            print(f"  Output: {run_dir_gpu}")
            
            results["gpu"] = {
                "status": "success",
                "device": device,
                "run_dir": str(run_dir_gpu),
                "elapsed_s": time_gpu,
            }
            
            # Analyze GPU run
            gpu_analysis = analyze_run(run_dir_gpu)
            results["gpu"]["analysis"] = gpu_analysis
            
            if "mass_drift" in gpu_analysis:
                print(f"  Mass drift: {gpu_analysis['mass_drift']['relative_drift']:.6f}")
            
            # Compare if both succeeded
            if results["cpu"]["status"] == "success":
                # TODO: Implement height field comparison
                results["comparison"] = {
                    "status": "pending",
                    "note": "Height field comparison not yet implemented",
                }
                
        except Exception as e:
            print(f"✗ GPU run failed: {e}")
            results["gpu"] = {
                "status": "failed",
                "error": str(e),
            }
    
    # ===== Mass drift threshold =====
    print("\n" + "=" * 70)
    print("Mass drift threshold analysis...")
    print("=" * 70)
    
    if results.get("cpu", {}).get("analysis", {}).get("mass_drift"):
        cpu_drift = results["cpu"]["analysis"]["mass_drift"]["relative_drift"]
        n_steps = 50
        
        # Extrapolate to 1e4 steps
        drift_per_1e4 = cpu_drift * (10000 / n_steps)
        
        results["mass_drift_threshold"] = {
            "cpu_baseline_drift": cpu_drift,
            "steps_run": n_steps,
            "extrapolated_per_1e4_steps": drift_per_1e4,
            "target_threshold": 0.001,  # 0.1%
            "pass": drift_per_1e4 <= 0.001,
        }
        
        print(f"CPU baseline drift (50 steps): {cpu_drift:.6f} ({cpu_drift*100:.4f}%)")
        print(f"Extrapolated to 1e4 steps: {drift_per_1e4:.6f} ({drift_per_1e4*100:.4f}%)")
        print(f"Target threshold: 0.001 (0.1%)")
        print(f"Status: {'✓ PASS' if results['mass_drift_threshold']['pass'] else '✗ FAIL'}")
    
    # ===== Save artifact =====
    artifact_path = Path("artifacts/m_s2.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "=" * 70)
    print(f"✓ Results saved to: {artifact_path}")
    print("=" * 70)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  CPU baseline: {results.get('cpu', {}).get('status', 'unknown')}")
    print(f"  GPU comparison: {results.get('gpu', {}).get('status', 'unknown')}")
    if "mass_drift_threshold" in results:
        print(f"  Mass drift threshold: {'PASS' if results['mass_drift_threshold']['pass'] else 'FAIL'}")
    
    return results


if __name__ == "__main__":
    main()
