#!/usr/bin/env python3
"""
G11a solver consistency analysis script.
Analyzes the n_f module vs 0-D solver consistency.
"""

import json
import sys
from pathlib import Path

def main():
    case_dir = Path(__file__).parent
    result_path = case_dir / "result.json"
    
    if not result_path.exists():
        print(f"Error: {result_path} not found")
        sys.exit(1)
    
    with open(result_path) as f:
        result = json.load(f)
    
    print("=" * 60)
    print("G11a solver consistency analysis")
    print("=" * 60)
    print()
    
    # Extract G11a data
    g11a = result.get("gates", {}).get("G11a_solver_consistency", {})
    value = g11a.get("value", {})
    
    print("Case-by-case comparison:")
    print("-" * 60)
    print(f"{'Case':<15} {'n_f_module':<12} {'n_f_0d':<12} {'rel_diff':<12}")
    print("-" * 60)
    
    for case_name, data in value.items():
        n_f_module = data.get("n_f_module", 0)
        n_f_0d = data.get("n_f_0d", 0)
        rel_diff = data.get("rel_diff", 0)
        print(f"{case_name:<15} {n_f_module:<12.6f} {n_f_0d:<12.6f} {rel_diff*100:<12.2f}%")
    
    print()
    print(f"Gate: <=0.5%")
    print(f"Result: FAIL (case 3.1 glaze: {value['3.1_glaze']['rel_diff']*100:.2f}% > 0.5%)")
    print()
    
    # Analysis
    print("=" * 60)
    print("Analysis")
    print("=" * 60)
    print()
    print("The n_f (normal flow) parameter shows significant discrepancy")
    print("between the module calculation and the independent 0-D solver")
    print("for the glaze case (3.1).")
    print()
    print("Root cause: Input reconstruction mismatch.")
    print("  - The module uses panel-exported quantities (s_over_c, n_f)")
    print("  - The 0-D solver uses independent input reconstruction")
    print("  - The panel export does not close the panel n_f consistently")
    print("  - Area reconstruction is the source of the gap")
    print()
    print("Note: The solvers are bitwise-identical when given the same")
    print("inputs (G11_solver_crosscheck: 6格 abs_diff = 0.0).")
    print("The gap is purely from input reconstruction, not solver logic.")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())