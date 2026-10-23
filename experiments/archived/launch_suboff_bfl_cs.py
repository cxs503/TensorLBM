#!/usr/bin/env python3
"""SUBOFF 200³ BFL + Cs Sweep Launcher (parallel).

Runs 4 simulations on SDAA 20-23 concurrently and collects results.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKER = Path("/root/TensorLBM_dev/suboff_full_hull_runner.py")
SRC = Path("/root/TensorLBM_dev/src")
OUTPUT = Path("/tmp/suboff_bfl_cs_results.json")

RUNS = [
    # (label, device, bfl, cs, output_file)
    ("staircase_Cs005",  "sdaa:20", False, 0.05, "/tmp/suboff_sdaa20_staircase.json"),
    ("bfl_Cs005",        "sdaa:21", True,  0.05, "/tmp/suboff_sdaa21_bfl.json"),
    ("bfl_Cs003",        "sdaa:22", True,  0.03, "/tmp/suboff_sdaa22_cs003.json"),
    ("bfl_Cs007",        "sdaa:23", True,  0.07, "/tmp/suboff_sdaa23_cs007.json"),
]

results_lock = threading.Lock()
results: dict = {}


def run_one(label, device, bfl, cs, out_file):
    cmd = [
        sys.executable, "-u", str(WORKER),
        "--device", device,
        "--hull-type", "bare_hull",
        "--cs", str(cs),
        "--output", out_file,
    ]
    if bfl:
        cmd.append("--bfl")

    env = {**__import__("os").environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}

    print(f"[{label}] Starting on {device} (BFL={bfl}, Cs={cs})...", flush=True)
    t0 = time.time()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    output_lines = []
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f"[{label}] {line}", flush=True)

    proc.wait()
    elapsed = time.time() - t0
    print(f"[{label}] Finished in {elapsed:.0f}s (rc={proc.returncode})", flush=True)

    # Load result from JSON file
    try:
        with open(out_file) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        result = {
            "label": label, "device": device, "status": "FAILED",
            "error": str(e), "output_tail": "\n".join(output_lines[-20:]),
        }

    result["_label"] = label
    result["_elapsed_s"] = elapsed
    result["_rc"] = proc.returncode

    with results_lock:
        results[label] = result


def main():
    print("=" * 70)
    print("SUBOFF 200³ BFL + Cs Sweep (PARALLEL)")
    print("  bare_hull, 200×80×80, Re=2e6, 3000 steps, warmup=1000")
    print("  SDAA:20 = staircase Cs=0.05")
    print("  SDAA:21 = BFL Cs=0.05")
    print("  SDAA:22 = BFL Cs=0.03")
    print("  SDAA:23 = BFL Cs=0.07")
    print("=" * 70)

    t_start = time.time()

    threads = []
    for label, device, bfl, cs, out_file in RUNS:
        t = threading.Thread(target=run_one, args=(label, device, bfl, cs, out_file))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_elapsed = time.time() - t_start

    # Build summary
    summary = {
        "config": {
            "hull_type": "bare_hull",
            "lattice": "D3Q19",
            "collision": "MRT+Smagorinsky",
            "grid": "200×80×80",
            "Re": 2_000_000,
            "hull_length": 100,
            "u_in": 0.06,
            "n_steps": 3000,
            "warmup": 1000,
        },
        "total_wall_time_s": total_elapsed,
        "results": {},
    }

    for label, _, _, _, _ in RUNS:
        r = results.get(label, {"status": "MISSING"})
        summary["results"][label] = r

    # Print summary table
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Run':<20} {'Ct_fric':>10} {'Ct_pres':>10} {'Ct_total':>10} {'Cs':>6} {'BFL':>5} {'Status':>10}"
    print(header)
    print("-" * 70)

    for label, _, bfl, cs, _ in RUNS:
        r = results.get(label, {})
        status = r.get("status")
        if status in ("FAILED", "EXCEPTION", "MISSING", "PARSE_FAILED", None):
            print(f"{label:<20} {'—':>10} {'—':>10} {'—':>10} {cs:>6.2f} {str(bfl):>5} {status or '?':>10}")
        else:
            print(f"{label:<20} {r.get('Ct_fric', 0):>10.5f} {r.get('Ct_pres', 0):>10.5f} "
                  f"{r.get('Ct_total', 0):>10.5f} {cs:>6.2f} {str(bfl):>5} {'OK':>10}")

    print("-" * 70)
    print(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Write results
    OUTPUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults written to {OUTPUT}")

    # Cs sensitivity analysis
    print()
    print("Cs SENSITIVITY ANALYSIS:")
    print("-" * 40)
    baseline = results.get("bfl_Cs005", {}).get("Ct_total")
    if baseline:
        print(f"  Baseline Cs=0.05: Ct={baseline:.6f}")
        for cs_label in ["bfl_Cs003", "bfl_Cs007"]:
            r = results.get(cs_label, {})
            ct = r.get("Ct_total")
            if ct is not None:
                delta = ct - baseline
                pct = delta / abs(baseline) * 100
                print(f"  {cs_label}: Ct={ct:.6f}  Δ={delta:+.6f}  ({pct:+.2f}%)")
    else:
        print("  Baseline not available for comparison")


if __name__ == "__main__":
    main()
