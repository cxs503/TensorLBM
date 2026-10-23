#!/usr/bin/env python3
"""Run one FD case (schur solver by default), write per-case JSON atomically."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hexref_schur import solve_case_schur

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "cases")


def main():
    lat, vf, nx, ny = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tag = "schur"
    os.makedirs(OUT, exist_ok=True)
    name = f"{lat}_vf{vf:g}_nx{nx}_ny{ny}_{tag}"
    path = os.path.join(OUT, name + ".json")
    if os.path.exists(path):
        print(f"SKIP {name}", flush=True)
        return
    res = solve_case_schur(lat, vf, nx, ny)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    print(
        f"DONE {name} f={res['f_press']:.8f} K/R2={res['K_s_over_R2']:.6e} "
        f"cg={res['cg_info']} cgres={res['cg_res']:.1e} div={res['div_max']:.1e} "
        f"fac={res['t_factor']}s cg_t={res['t_cg']}s rss={res['maxrss_gb']}G",
        flush=True,
    )


if __name__ == "__main__":
    main()
