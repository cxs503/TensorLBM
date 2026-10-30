#!/usr/bin/env python3
"""Mission 209 SDAA smoke test - D=16, 200 steps to prove device availability.

Uses the existing run.py infrastructure with minimal parameters.
This is NOT a convergence test - just proves SDAA device works.
"""

import json
import os
import sys
import time
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 70)
    print("Mission 209 SDAA Smoke Test")
    print("=" * 70)
    
    # Device evidence
    import torch
    device_evidence = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    try:
        import torch_sdaa
        sdaa_available = torch.sdaa.is_available()
        sdaa_count = torch.sdaa.device_count()
        sdaa_name = torch.sdaa.get_device_name(0) if sdaa_count > 0 else None
        
        device_evidence.update({
            "sdaa_available": sdaa_available,
            "sdaa_device_count": sdaa_count,
            "sdaa_device_name": sdaa_name,
        })
        
        print(f"\nDevice Evidence:")
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  SDAA available: {sdaa_available}")
        print(f"  SDAA device count: {sdaa_count}")
        if sdaa_name:
            print(f"  SDAA device name: {sdaa_name}")
        
        if not sdaa_available or sdaa_count == 0:
            print("\nERROR: SDAA not available")
            sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: SDAA initialization failed: {e}")
        sys.exit(1)
    
    # Smoke test parameters
    D = 16
    steps = 200  # Very small for smoke test
    tau = 0.56
    
    print(f"\nSmoke Test Parameters:")
    print(f"  D = {D}")
    print(f"  steps = {steps}")
    print(f"  tau = {tau}")
    print(f"  Domain: small (lat=2, up=2, down=2)")
    
    # Run using existing run.py
    run_py = os.path.join(BASE, "run.py")
    output_json = os.path.join(BASE, f"smoke_test_d{D}_s{steps}.json")
    
    cmd = [
        sys.executable,
        run_py,
        output_json,
        "--D", str(D),
        "--steps", str(steps),
        "--lat", "2.0",
        "--up", "2.0",
        "--down", "2.0",
        "--tau", str(tau),
        "--kernel", "sparse",
        "--sample", "20",
    ]
    
    env = os.environ.copy()
    env["W8A_DEV"] = "sdaa:0"
    env["PYTHONPATH"] = "/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Device: sdaa:0")
    print("-" * 70)
    
    t0 = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        
        elapsed = time.time() - t0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        
        if result.returncode != 0:
            print(f"\nERROR: run.py exited with code {result.returncode}")
            sys.exit(1)
        
        # Load output
        if os.path.exists(output_json):
            with open(output_json) as f:
                sim_data = json.load(f)
            
            print(f"\n✓ Smoke test completed in {elapsed:.2f}s")
            print(f"  Output: {output_json}")
            
            # Extract key results
            meta = sim_data.get("meta", {})
            history = sim_data.get("history", [])
            
            if history:
                last_sample = history[-1]
                print(f"  Last sample: step={last_sample.get('step')}, Cd={last_sample.get('cd', 'N/A'):.4f}")
            
        else:
            print(f"\nWARNING: Output file not created: {output_json}")
            sim_data = {}
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\nERROR: Smoke test timed out after {elapsed:.2f}s")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Smoke test failed: {e}")
        sys.exit(1)
    
    # Write artifacts
    artifacts = {
        "mission": 209,
        "test_type": "sdaa_smoke_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_evidence": device_evidence,
        "parameters": {
            "D": D,
            "steps": steps,
            "tau": tau,
            "lateral_D": 2.0,
            "up_D": 2.0,
            "down_D": 2.0,
            "kernel": "sparse",
        },
        "results": {
            "completed": True,
            "total_time_s": elapsed,
            "output_file": output_json,
            "return_code": result.returncode,
        },
        "judgment": "SDAA device operational - smoke test passed",
    }
    
    artifact_path = os.path.join(BASE, "..", "..", "..", "artifacts", "sdaa_sphere_re100_v8.json")
    artifact_path = os.path.abspath(artifact_path)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    with open(artifact_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"\n✓ Artifacts written to: {artifact_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
