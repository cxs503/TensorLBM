"""CMB7-CSTUNE2-R1: programmatic extraction of the measured triples.

Reads runs/*/result.json (the 6 CMB7 arms from m27175 + the 2 new R1 arms if
present) and prints a compact table.  No hand-copying of numbers.
"""
import json
import sys
from pathlib import Path

# experimental reference (Santini case 3)
EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.394

def peak_of(path):
    r = json.loads(Path(path).read_text())
    hist = r["history"]
    pk = max(hist, key=lambda x: x["depth_over_D"])
    return {
        "stop": r.get("stop_reason"),
        "peak_D": pk["depth_over_D"],
        "peak_ms": pk["time_ms"],
        "contact": r.get("contact_step"),
        "wall_s": r.get("_wall_s"),
        "C_s": r.get("C_s"),
        "rr_tau": r.get("rr_tau"),
    }

def main():
    bases = [
        Path("/root/tp-agent/work/m27175/runs"),
        Path("/root/tp-agent/work/m27389/runs/cmb7r1"),
    ]
    rows = []
    for base in bases:
        if not base.exists():
            continue
        for d in sorted(base.glob("cs0*")):
            rj = d / "result.json"
            if not rj.exists():
                continue
            p = peak_of(rj)
            err = (p["peak_D"] / EXP_PEAK_D - 1) * 100
            rows.append((d.name, p, err))
    print(f"{'arm':16} {'Cs':>6} {'stop':16} {'peak_D':>9} {'peak_ms':>9} {'err%':>8} {'wall_s':>9}")
    for name, p, err in rows:
        print(f"{name:16} {str(p['C_s']):>6} {str(p['stop']):16} {p['peak_D']:>9.4f} {p['peak_ms']:>9.3f} {err:>8.2f} {str(p['wall_s']):>9}")

if __name__ == "__main__":
    main()
