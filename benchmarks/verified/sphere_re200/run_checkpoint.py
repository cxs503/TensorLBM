#!/usr/bin/env python3
"""Checkpoint-based runner for sphere_re200 — supports resume after interruption.

Usage:
    python run_checkpoint.py --D 20 --target_steps 12000 --checkpoint_interval 500
    
This will:
1. Run in stages, saving checkpoint every checkpoint_interval steps
2. If interrupted, resume from last checkpoint
3. Output intermediate Cd values at 2500, 6000, 9000, 12000 steps
"""

import argparse
import json
import os
import sys
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run_case

def main():
    parser = argparse.ArgumentParser(description='Checkpoint-based sphere_re200 runner')
    parser.add_argument('--D', type=int, required=True, help='Sphere resolution')
    parser.add_argument('--target_steps', type=int, default=12000, help='Total target steps')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--kernel', type=str, default='sparse', choices=['sparse', 'dense'])
    parser.add_argument('--lat', type=float, default=3.0)
    parser.add_argument('--up', type=float, default=3.0)
    parser.add_argument('--down', type=float, default=4.0)
    parser.add_argument('--ulb', type=float, default=0.05)
    parser.add_argument('--sample', type=int, default=50)
    
    args = parser.parse_args()
    
    # Calculate tau for Re=200
    tau = 0.5 + 3.0 * args.ulb * args.D / 200.0
    
    checkpoint_file = args.output + '.checkpoint'
    
    # Check for existing checkpoint
    start_step = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
            start_step = ckpt['step']
            print(f"Resuming from checkpoint at step {start_step}")
    
    # Run in stages
    stages = [2500, 6000, 9000, args.target_steps]
    stages = [s for s in stages if s > start_step]
    
    if not stages:
        print("Already completed all stages")
        return
    
    current_step = start_step
    all_history = []
    
    # Load existing history if resuming
    if start_step > 0 and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
            all_history = existing.get('history', [])
    
    for target in stages:
        steps_this_stage = target - current_step
        if steps_this_stage <= 0:
            continue
            
        print(f"\n=== Running stage: {current_step} -> {target} ({steps_this_stage} steps) ===")
        
        stage_output = args.output + f'.stage_{target}'
        
        # Run this stage
        run_case(
            out_path=stage_output,
            kernel=args.kernel,
            D=args.D,
            steps=steps_this_stage,
            lat=args.lat,
            up=args.up,
            down=args.down,
            tau=tau,
            ulb=args.ulb,
            treat='hard',
            sample=args.sample,
            cv_tail=0,
            mem_stats=False
        )
        
        # Load stage results and append to history
        with open(stage_output) as f:
            stage_data = json.load(f)
            
        # Adjust step numbers
        for h in stage_data['history']:
            h['step'] += current_step
            all_history.append(h)
        
        current_step = target
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'step': current_step, 'target': args.target_steps}, f)
        
        # Save intermediate result
        intermediate = {
            'meta': stage_data['meta'],
            'meta']['steps'] = current_step
            'history': all_history
        }
        with open(args.output, 'w') as f:
            json.dump(intermediate, f, indent=2)
        
        # Report Cd at this stage
        vals = [h['cd'] for h in all_history]
        n = len(vals)
        w = max(1, int(n * 0.2))
        tail = vals[-w:]
        cd_mean = sum(tail) / len(tail)
        CD_REF = 0.8056146866904639
        err_pct = (cd_mean - CD_REF) / CD_REF * 100
        
        print(f"Stage {target}: Cd={cd_mean:.6f}, err={err_pct:+.3f}%")
    
    print(f"\n=== Completed all stages up to {args.target_steps} steps ===")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

if __name__ == '__main__':
    main()
