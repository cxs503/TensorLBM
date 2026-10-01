#!/usr/bin/env python3
"""square_cylinder archive verify: recompute all gates from the summary records.

Checks (numpy/stdlib only; no torch/tensorlbm import):
 1. geometry: nx=ny=40D, solid cells = D^2, surface links = 12D-4 (formula)
 2. stored error self-consistency (cd/st vs locked reference, <=0.01 pp)
 3. force-column closure: pair-mem mean |d| <= 0.5%; wet-mem A/B recorded
 4. main gate (finest, pair): |Cd err| <= 3% and |St err| <= 3%
 5. monotonicity with the preregistered <=1% resolution-floor clause
 6. reproduction gate: mem column vs the replaced archive D32/D48, <=0.5%
 7. result.json verdict flags agree with the recomputed gates

Raw per-step traces are retained on the server staging tree; verify_traces.py
recomputes every statistic from them (window means, 12-block spread, Hann +
parabolic spectral peak, hysteretic zero crossings) and re-runs the same gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REF_CD = 1.4785
REF_ST = 0.1456
DOMAIN_D = 40.0
OLD_ARCH = {32: (1.4816, 0.1482), 48: (1.4844, 0.1480)}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "OK  " if ok else "FAIL"
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def main() -> None:
    stage = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    cases = {}
    for p in sorted(stage.glob("case_D*.json")):
        d = json.loads(p.read_text())
        cases[d["D"]] = d
    check("case files found", sorted(cases) == [32, 48, 64], f"D={sorted(cases)}")

    for D in sorted(cases):
        d = cases[D]
        check(
            f"D{D}: domain 40D x 40D",
            d["nx"] == d["ny"] == int(DOMAIN_D * D),
            f"{d['nx']}x{d['ny']}",
        )
        check(
            f"D{D}: solid cells = D^2",
            d["force_columns"]["n_solid_cells"] == D * D,
            f"{d['force_columns']['n_solid_cells']}",
        )
        check(
            f"D{D}: surface links = 12D-4",
            d["force_columns"]["n_links_wet"] == 12 * D - 4,
            f"{d['force_columns']['n_links_wet']}",
        )

        for col in ("wet", "pair", "mem"):
            s = d[col]
            ecd = (s["cd"] - REF_CD) / REF_CD * 100.0
            est = (s["st"] - REF_ST) / REF_ST * 100.0
            check(
                f"D{D}: stored err self-consistency {col}",
                abs(ecd - s["err_cd_pct"]) < 0.01 and abs(est - s["err_st_pct"]) < 0.01,
                f"cd {ecd:+.2f}%/{s['err_cd_pct']}% st {est:+.2f}%/{s['err_st_pct']}%",
            )
            # 12-block spread consistent with stored min/max
            check(
                f"D{D}: block spread {col}",
                abs(min(s["cd_blocks"]) - s["cd_block_min"]) < 1e-9
                and abs(max(s["cd_blocks"]) - s["cd_block_max"]) < 1e-9,
                f"[{s['cd_block_min']}, {s['cd_block_max']}]",
            )

        closure = d["closure_pair_vs_mem_cd_pct"]
        check(f"D{D}: closure pair-mem |d|<=0.5%", abs(closure) <= 0.5, f"{closure:+.3f}%")
        ab = d["ab_diff"]["cd_pct_of_mem"]
        check(f"D{D}: wet-mem A/B recorded (>0, diagnostic)", 0.0 < ab < 20.0, f"{ab:+.2f}%")

        if D in OLD_ARCH:
            ocd, ost = OLD_ARCH[D]
            dcd = (d["mem"]["cd"] - ocd) / ocd * 100.0
            dst = (d["mem"]["st"] - ost) / ost * 100.0
            check(
                f"D{D}: repro vs replaced archive (mem) |d|<=0.5%",
                abs(dcd) <= 0.5 and abs(dst) <= 0.5,
                f"cd {dcd:+.3f}% st {dst:+.3f}%",
            )

    ds = sorted(cases)
    errs_cd = [abs(cases[D]["pair"]["cd"] - REF_CD) / REF_CD for D in ds]
    errs_st = [abs(cases[D]["pair"]["st"] - REF_ST) / REF_ST for D in ds]
    fine = cases[ds[-1]]["pair"]
    ecd_f = (fine["cd"] - REF_CD) / REF_CD * 100.0
    est_f = (fine["st"] - REF_ST) / REF_ST * 100.0
    check(f"MAIN GATE finest D{ds[-1]} pair: |Cd err|<=3%", abs(ecd_f) <= 3.0, f"{ecd_f:+.2f}%")
    check(f"MAIN GATE finest D{ds[-1]} pair: |St err|<=3%", abs(est_f) <= 3.0, f"{est_f:+.2f}%")
    mono_cd = all(errs_cd[i + 1] <= errs_cd[i] + 1e-12 for i in range(len(errs_cd) - 1)) or all(
        e <= 0.01 for e in errs_cd
    )
    mono_st = all(errs_st[i + 1] <= errs_st[i] + 1e-12 for i in range(len(errs_st) - 1)) or all(
        e <= 0.01 for e in errs_st
    )
    check(
        "MONOTONE Cd (or floor<=1%)", bool(mono_cd), " -> ".join(f"{e * 100:.2f}%" for e in errs_cd)
    )
    check(
        "MONOTONE St (or floor<=1%)", bool(mono_st), " -> ".join(f"{e * 100:.2f}%" for e in errs_st)
    )

    res = json.loads((stage / "result.json").read_text())
    v = res["verdict_prereg"]
    check(
        "result.json verdict flags true",
        v["cd_within_3pct"] and v["st_within_3pct"] and v["cd_monotone"] and v["st_monotone"],
        "cd/st within-3pct + monotone flags",
    )

    print()
    if failures:
        print(f"VERIFY RESULT: {len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("VERIFY RESULT: ALL OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
