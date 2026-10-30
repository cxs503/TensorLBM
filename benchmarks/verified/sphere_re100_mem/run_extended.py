#!/usr/bin/env python3
"""Extended resolution study for sphere_re100_mem: D=80 and D=160.

This script runs the two additional resolution tiers required by m243:
- T3: D=80 with 48k steps
- T4: D=160 with 96k steps

Each run includes detailed window statistics for error bar computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sphere_re100', 'w8a'))

# Override device to SDAA
os.environ["W8A_DEV"] = "sdaa:0"

from run import run_case

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["T3", "T4", "all"], default="all")
    args = ap.parse_args()
    
    # Configuration based on existing D=40 and D=60 patterns
    # D=40: tau=0.56, steps=12k, sample=50
    # D=60: tau=0.59, steps=24k, sample=50
    # Extrapolate: D=80: tau=0.62, steps=48k; D=160: tau=0.68, steps=96k
    
    configs = {
        "T3": {
            "D": 80,
            "tau": 0.62,
            "steps": 48000,
            "sample": 50,
            "lat": 3.0,
            "up": 3.0,
            "down": 4.0,
            "out": "t3_d80big_48k.json"
        },
        "T4": {
            "D": 160,
            "tau": 0.68,
            "steps": 96000,
            "sample": 50,
            "lat": 3.0,
            "up": 3.0,
            "down": 4.0,
            "out": "t4_d160big_96k.json"
        }
    }
    
    tiers_to_run = ["T3", "T4"] if args.tier == "all" else [args.tier]
    
    for tier in tiers_to_run:
        cfg = configs[tier]
        print(f"\n{'='*60}")
        print(f"Running {tier}: D={cfg['D']}, steps={cfg['steps']}")
        print(f"{'='*60}\n")
        
        run_case(
            out_path=cfg["out"],
            kernel="sparse",
            D=cfg["D"],
            steps=cfg["steps"],
            lat=cfg["lat"],
            up=cfg["up"],
            down=cfg["down"],
            tau=cfg["tau"],
            ulb=0.05,
            treat="hard",
            sample=cfg["sample"],
            cv_tail=0,
            mem_stats=True
        )

if __name__ == "__main__":
    main()
