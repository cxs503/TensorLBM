#!/usr/bin/env python3
"""Refresh checks in existing result.json to the corrected logic (no re-sim).

Adds per-check `extrapolated` flag and recomputes max_err over the
non-extrapolated checkpoints only (consistent with benchmarks/bench_fs_2d.py).
"""
import json
import sys
from pathlib import Path

MM_X = [(1.0, 1.1), (2.0, 1.8), (3.0, 2.7)]
H_REF, H_T = 0.78, 1.0


def interp(series, key, T_target):
    pts = [(s["T"], s[key]) for s in series]
    if T_target <= pts[0][0]:
        return pts[0][1]
    for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
        if t1 <= T_target <= t2:
            return v1 + (v2 - v1) * (T_target - t1) / (t2 - t1)
    return pts[-1][1]


def refresh(path: Path) -> dict:
    r = json.loads(path.read_text())
    series = r["series"]
    tmax = series[-1]["T"]
    checks = []
    for T_ref, X_ref in MM_X:
        x = interp(series, "X", T_ref)
        checks.append({
            "T": T_ref, "kind": "X", "ref": X_ref, "sim": x,
            "err_pct": abs(x - X_ref) / X_ref * 100.0,
            "extrapolated": T_ref > tmax + 1e-9,
        })
    h = interp(series, "H", H_T)
    checks.append({
        "T": H_T, "kind": "H", "ref": H_REF, "sim": h,
        "err_pct": abs(h - H_REF) / H_REF * 100.0,
        "extrapolated": H_T > tmax + 1e-9,
    })
    valid = [c["err_pct"] for c in checks if not c["extrapolated"]] or [
        max(c["err_pct"] for c in checks)]
    r["checks"] = checks
    r["max_err_pct"] = max(valid)
    path.write_text(json.dumps(r, indent=2) + "\n")
    return r


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = refresh(Path(p))
        print(p, "Tmax=%.3f" % r["series"][-1]["T"], "max_err=%.2f%%" % r["max_err_pct"])
        for c in r["checks"]:
            tag = "  (EXTRAPOLATED)" if c["extrapolated"] else ""
            print(f"   T={c['T']:.1f} {c['kind']}: sim={c['sim']:.4f} ref={c['ref']:.2f} "
                  f"err={c['err_pct']:+.2f}%{tag}")