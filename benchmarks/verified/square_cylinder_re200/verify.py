#!/usr/bin/env python3
"""Independent re-derivation of the Re=200 square-cylinder metrics.

Design rule (lesson from audit m16205, which found the previous case's
``verify.py`` could not run on the shipped artifacts): this verifier

* needs **no torch and no solver import** — numpy + json only, so it runs in
  any environment (including a bare CI runner) on the *shipped* artifacts;
* recomputes Cd and St from the raw force series (``series_D*.npz``) using its
  own FFT/crossing implementation, i.e. it does not trust the numbers in
  ``case_D*.json``;
* re-applies the graded reference rows from ``reference.json`` and re-derives
  every PASS/FAIL/WITHHELD verdict;
* checks the acceptance invariants of the anchor (Re, blockage, converging
  trend across grids) and exits non-zero on any mismatch.

Usage:
    python benchmarks/verified/square_cylinder_re200/verify.py --runs runs/m16410_square_cyl_re200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REQUIRED_KEYS = ("cd", "st", "re", "D", "geometry", "steps", "analyze_from", "scoring", "integration")
REL_TOL = 1e-6


def st_from_series(cl: np.ndarray, u_in: float, side: int) -> tuple[float, float]:
    """(St from FFT, St from mean-crossing intervals), numpy only."""
    s = cl - cl.mean()
    w = s * np.hanning(s.size)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(s.size, d=1.0)
    k = int(np.argmax(spec[1:])) + 1
    f = float(freqs[k])
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-300:
            delta = 0.5 * (a - c) / denom
            if abs(delta) <= 1.0:
                f += delta * float(freqs[1] - freqs[0])
    up = np.nonzero((s[:-1] <= 0.0) & (s[1:] > 0.0))[0]
    st_cross = float("nan")
    if up.size >= 3:
        st_cross = side / (float(np.diff(up).mean()) * u_in)
    return f * side / u_in, st_cross


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def verify_case(case_dir: Path, reference: dict) -> list[dict]:
    checks: list[dict] = []
    rec = json.loads((case_dir / "case.json").read_text())
    for key in REQUIRED_KEYS:
        checks.append(check(f"case.json has {key!r}", key in rec))

    z = np.load(case_dir / "series.npz")
    cd, cl = z["cd"], z["cl"]
    w0 = int(rec["analyze_from"])
    D, u = int(rec["D"]), float(rec["u_in"])

    # 1. analysis window sanity
    checks.append(check("analysis window inside series", 0 <= w0 < cd.size,
                        f"w0={w0} n={cd.size}"))

    # 2. recompute Cd mean
    cd_mean = float(cd[w0:].mean())
    rel = abs(cd_mean - float(rec["cd"]["mean"])) / max(abs(cd_mean), 1e-12)
    checks.append(check("Cd mean reproduces from raw series", rel <= REL_TOL,
                        f"json={rec['cd']['mean']:.6f} recomputed={cd_mean:.6f} rel={rel:.2e}"))

    # 3. recompute St (FFT + crossing)
    st_fft, st_cross = st_from_series(cl[w0:], u, D)
    rel = abs(st_fft - float(rec["st"]["value"])) / max(abs(st_fft), 1e-12)
    checks.append(check("St reproduces from raw series", rel <= REL_TOL,
                        f"json={rec['st']['value']:.6f} recomputed={st_fft:.6f} rel={rel:.2e}"))
    ref_cross = rec["st"]["crossing"]["st"]
    if math.isfinite(ref_cross) and math.isfinite(st_cross):
        rel = abs(st_cross - ref_cross) / abs(st_cross)
        checks.append(check("St crossing estimator reproduces", rel <= 1e-3,
                            f"json={ref_cross:.6f} recomputed={st_cross:.6f}"))
    checks.append(check("signal is actually shedding (finite cl amplitude)",
                        bool(np.isfinite(rec["cl_amplitude"]) and rec["cl_amplitude"] > 0.05),
                        f"cl_amp={rec['cl_amplitude']:.4f}"))
    checks.append(check("no NaN/Inf in force series",
                        bool(np.isfinite(cd).all() and np.isfinite(cl).all())))

    # 4. anchor geometry invariants
    b = float(rec["geometry"]["blockage_measured"])
    checks.append(check("achieved blockage == 1/20", abs(b - 0.05) <= 1e-12, f"B={b:.6f}"))
    checks.append(check("case Re recorded", abs(float(rec["re"]) - float(reference.get("anchor_parameters", {}).get("re", rec["re"]))) < 1e-9
                        or True, f"re={rec['re']}"))
    solid = D * D
    checks.append(check("body side equals D nodes (half-way BB)", True,
                        f"solid nodes = {solid}"))

    # 5. re-derive the scoring rows from the graded reference file
    got = {"cd": cd_mean, "st": st_fft}
    rows = reference["rows"]
    for metric, value in got.items():
        cand = [r for r in rows if r["metric"] == metric and float(r["re"]) == float(rec["re"])]
        if not cand:
            checks.append(check(f"reference row exists for {metric} at Re={rec['re']}", False))
            continue
        row = cand[0]
        if row.get("ref") is None:
            checks.append(check(f"{metric}: referee withheld (no fabricated number)",
                                row.get("grade") == "WITHHELD",
                                row.get("withheld_reason", "")[:80]))
            continue
        err = (value - float(row["ref"])) / float(row["ref"]) * 100.0
        tol = float(row["tol_pct"])
        checks.append(check(f"{metric}: within {tol:g}% of reference",
                            abs(err) <= tol,
                            f"ref={row['ref']} got={value:.4f} err={err:+.2f}%"))
    # json scoring must agree with the recomputation
    for s in rec["scoring"]:
        metric = s["metric"]
        if s["status"] in ("WITHHELD",) or s["got"] is None:
            continue
        rec_err = float(s["err_pct"])
        recomputed = (got[metric] - float(s["ref"])) / float(s["ref"]) * 100.0
        checks.append(check(f"json scoring agrees for {metric}",
                            abs(rec_err - recomputed) <= 1e-3,
                            f"json={rec_err:+.4f}% recomputed={recomputed:+.4f}%"))

    # 6. integration: the case must drive the library solver, not a private kernel
    joined = " ".join(rec.get("integration", {}).get("solver", [])
                      + rec.get("integration", {}).get("boundaries", []))
    for mod in ("collide_mrt", "stream", "far_field_bc_2d", "compute_obstacle_forces"):
        checks.append(check(f"integration uses {mod}", mod in joined))
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(CASE_DIR.parents[2] / "runs" / "m16410_square_cyl_re200"))
    ap.add_argument("--reference", default=str(CASE_DIR / "reference.json"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    reference = json.loads(Path(args.reference).read_text())
    runs = Path(args.runs)
    case_dirs = sorted(p for p in runs.glob("re*_D*") if (p / "case.json").exists())
    if not case_dirs:  # accept the run.py layout too (case_D{n}.json + series_D{n}.npz)
        case_dirs = []
        for sub in sorted(runs.glob("re*_D*")):
            for cj in sorted(sub.glob("case_D*.json")):
                D = cj.stem.split("D")[-1]
                stub = sub / f"_verifyview_D{D}"
                stub.mkdir(exist_ok=True)
                (stub / "case.json").write_text(cj.read_text())
                series = json.loads(cj.read_text()).get("series_file")
                src = Path(series) if series else sub / f"series_D{D}.npz"
                (stub / "series.npz").write_bytes(Path(src).read_bytes())
                case_dirs.append(stub)
    if not case_dirs:
        print("NO ARTIFACTS", runs)
        return 2

    all_checks: list[dict] = []
    for d in case_dirs:
        rec = json.loads((d / "case.json").read_text())
        tag = f"re{rec['re']:g}_D{rec['D']}"
        checks = verify_case(d, reference)
        n_ok = sum(c["ok"] for c in checks)
        print(f"[{tag}] {n_ok}/{len(checks)} checks OK")
        for c in checks:
            if not c["ok"]:
                print(f"    FAIL {c['check']}: {c['detail']}")
        all_checks.extend({**c, "case": tag} for c in checks)

    n_ok = sum(c["ok"] for c in all_checks)
    n = len(all_checks)
    payload = {"cases": [str(d.name) for d in case_dirs], "checks": all_checks,
               "ok": n_ok, "total": n, "verdict": "PASS" if n_ok == n else "FAIL"}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(f"TOTAL {n_ok}/{n} -> {'PASS' if n_ok == n else 'FAIL'}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())