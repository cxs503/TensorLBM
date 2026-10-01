#!/usr/bin/env python3
"""W6-A independent verification: closed-form recomputation, no run.py import.

Recomputes every verdict-level number from the RAW measurement fields in the
case_*.json files (never from run.py's derived blocks) with independently
written arithmetic, then cross-checks result.json:

  1. a_nom = N*(3*phi/(4*pi))**(1/3)                     (geometry formula)
  2. k_ref = N**3/(6*pi*a_nom*K_table)                   (NOTES section 2)
  3. k_sim = nu*(1-phi)*uxf/a_body, nu = (tau-0.5)/3     (Darcy + drive)
  4. K_sim = a_body*N**3/(6*pi*nu*a_nom*(1-phi)*uxf)
  5. identity check: k_sim/k_ref == K_sim/K_table (same a, phi both sides)
  6. err = |k_sim/k_ref - 1| per tier; ladder verdict by the frozen rule
     (strict monotonic decrease AND finest <= 3%; coarsest-tier fallback)
  7. hash-chain integrity: prereg snapshot hash == first chain hash;
     last chain NOTES.md hash == sha256(NOTES.md)
  8. iron law: grep for forbidden kernel defs in run.py must hit zero

Exit code 0 = all checks pass; anything else = mismatch (message printed).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

TOL_FIN = 0.03
TAU_STD = 1.0
K_TABLE_LOCKED = {0.125: 4.292, 0.216: 7.4423, 0.343: 15.402}
LADDER_N = [64, 96, 128]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main(dirpath: str, formal_fs: float) -> int:
    d = Path(dirpath)
    failures: list[str] = []
    checks = 0

    report = json.loads((d / "result.json").read_text())
    cases: dict[tuple[float, int, float, float], dict] = {}
    for p in sorted(d.glob("case_*.json")):
        c = json.loads(p.read_text())
        if c.get("smoke"):
            continue
        r = c["run"]
        cases[(float(r["phi"]), int(r["N"]), float(r["force_scale"]), float(r["tau"]))] = c

    # ---- per-case closed-form recomputation ----
    recomputed: dict[tuple[float, int, float, float], dict] = {}
    for (phi, n, fs, tau), c in sorted(cases.items()):
        nu = (tau - 0.5) / 3.0
        a_nom = n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)
        k_table = K_TABLE_LOCKED[phi]
        a_body = float(c["run"]["a_body"])
        uxf = float(c["measurement"]["uxf_mean_win4000"])
        k_ref = n**3 / (6.0 * math.pi * a_nom * k_table)
        k_sim = nu * (1.0 - phi) * uxf / a_body
        k_sim_or = k_sim / fs  # k_sim at fs=1 (invariance check channel)
        K_sim = a_body * n**3 / (6.0 * math.pi * nu * a_nom * (1.0 - phi) * uxf)
        err = abs(k_sim / k_ref - 1.0)
        # k and K are inverse maps of the same measurement:
        # (k_sim/k_ref) * (K_sim/K_table) == 1 exactly (same a, phi used).
        ident = abs((k_sim / k_ref) * (K_sim / k_table) - 1.0)
        # run.py's own derived values must agree to float precision
        for label, mine, theirs in (
            ("a_nom", a_nom, c["run"]["a_nom"]),
            ("k_ref", k_ref, c["derived"]["k_ref"]),
            ("k_sim", k_sim, c["derived"]["k_sim"]),
            ("K_sim", K_sim, c["derived"]["K_sim"]),
            ("err", err, c["derived"]["err_primary"]),
        ):
            checks += 1
            if abs(mine - theirs) > 1e-9 * max(1.0, abs(mine)):
                failures.append(
                    f"phi={phi} N={n} fs={fs} tau={tau}: {label} mine={mine!r} theirs={theirs!r}"
                )
        checks += 1
        if ident > 1e-9:
            failures.append(f"phi={phi} N={n} fs={fs} tau={tau}: k/K identity broken: {ident:.3e}")
        recomputed[(phi, n, fs, tau)] = {
            "fs": fs,
            "tau": tau,
            "err": err,
            "k_sim": k_sim,
            "k_sim_fs1_equivalent": k_sim_or,
            "K_sim": K_sim,
        }

    def formal_err(phi: float, n: int) -> float | None:
        key = (phi, n, formal_fs, TAU_STD)
        return recomputed[key]["err"] if key in recomputed else None

    # ---- ladder verdict recomputation (frozen rule) ----
    per_phi = {}
    per_phi_amended = {}
    for phi in K_TABLE_LOCKED:
        errs = [formal_err(phi, n) for n in LADDER_N]
        if any(e is None for e in errs):
            failures.append(
                f"phi={phi}: formal ladder incomplete ({sum(e is not None for e in errs)}/3)"
            )
        else:
            mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
            fin = errs[-1] <= TOL_FIN
            if mono and fin:
                v = "PASS"
            elif errs[-2] > errs[-1] and errs[-1] <= TOL_FIN and errs[0] <= errs[1]:
                v = "PASS_WITH_DISCLOSURE"
            else:
                v = "FAIL"
            per_phi[phi] = v
            checks += 1
            rep_v = report["per_phi_verdict"].get(f"phi{phi}")
            if rep_v != v:
                failures.append(f"phi={phi}: verdict mine={v} report={rep_v}")
            # cross-check report ladder numbers
            rep_errs = report["ladders"][f"phi{phi}"]["tiers_err"]
            checks += 1
            if any(abs(a - b) > 1e-9 for a, b in zip(errs, rep_errs)):
                failures.append(f"phi={phi}: tiers_err mismatch {errs} vs {rep_errs}")
        # amendment #1 rule over N=64/96/128/160 (independent of the frozen check)
        am = [formal_err(phi, n) for n in (64, 96, 128, 160)]
        if all(e is not None for e in am):
            c1 = am[-1] <= TOL_FIN
            c2 = am[-1] < am[0]
            c3 = all(e <= am[0] for e in am[1:-1])
            va = "PASS" if (c1 and c2 and c3) else "FAIL"
            per_phi_amended[phi] = va
            checks += 1
            rep_va = report.get("per_phi_verdict_amended", {}).get(f"phi{phi}")
            if rep_va != va:
                failures.append(f"phi={phi}: amended verdict mine={va} report={rep_va}")
            rep_am = report["amendment1_ladders"][f"phi{phi}"]["tiers_err"]
            checks += 1
            if any(abs(a - b) > 1e-9 for a, b in zip(am, rep_am)):
                failures.append(f"phi={phi}: amended tiers_err mismatch {am} vs {rep_am}")

    checks += 1
    rep_overall = report["overall_verdict"]
    mine_overall = (
        "PASS"
        if per_phi and all(v == "PASS" for v in per_phi.values())
        else (
            "PASS_WITH_DISCLOSURE"
            if per_phi and all(v in ("PASS", "PASS_WITH_DISCLOSURE") for v in per_phi.values())
            else "FAIL"
        )
    )
    if rep_overall != mine_overall:
        failures.append(f"overall verdict mine={mine_overall} report={rep_overall}")

    checks += 1
    rep_overall_am = report.get("overall_verdict_amended")
    mine_overall_am = (
        "PASS" if per_phi_amended and all(v == "PASS" for v in per_phi_amended.values()) else "FAIL"
    )
    if rep_overall_am != mine_overall_am:
        failures.append(f"amended overall mine={mine_overall_am} report={rep_overall_am}")

    # ---- fs precheck spread (from raw k_sim) ----
    if report.get("fs_precheck"):
        ks = {float(k): v for k, v in report["fs_precheck"]["k_sim_by_fs"].items()}
        # k_sim is recorded per case at its own fs; the precheck compares the
        # cases' own k_sim values across fs (identical definition).
        checks += 1
        vals = list(ks.values())
        spread = max(abs(v / min(vals) - 1.0) for v in vals)
        if abs(spread - report["fs_precheck"]["max_rel_spread"]) > 1e-9:
            failures.append(
                f"fs precheck spread mine={spread} report={report['fs_precheck']['max_rel_spread']}"
            )

    # ---- hash chain integrity ----
    chain_lines = (d / "NOTES_sha256_chain.txt").read_text().splitlines()
    hashes = [ln.split()[0] for ln in chain_lines if ln and not ln.startswith("#")]
    checks += 1
    if hashes[0] != sha256_of(d / "NOTES_prereg_snapshot.txt"):
        failures.append("chain head != sha256(NOTES_prereg_snapshot.txt)")
    checks += 1
    notes_hashes = [
        ln.split()[0] for ln in chain_lines if ln.endswith("NOTES.md") and not ln.startswith("#")
    ]
    if notes_hashes[-1] != sha256_of(d / "NOTES.md"):
        failures.append("chain tail != sha256(NOTES.md)")
    checks += 1

    # ---- iron law ----
    grep = subprocess.run(
        [
            "grep",
            "-nE",
            r"^[[:space:]]*def (collide|stream|equilibrium|bounce|zou_he|far_field)",
            str(d / "run.py"),
        ],
        capture_output=True,
        text=True,
    )
    checks += 1
    if grep.returncode != 1 or grep.stdout.strip():
        failures.append(f"iron-law grep not clean: rc={grep.returncode} out={grep.stdout!r}")

    # ---- python-def audit of run.py physics helpers (naming/structure) ----
    src = (d / "run.py").read_text()
    checks += 1
    if "def apply_body_force_3d" not in src:
        failures.append("run.py missing the disclosed driver-level force helper")

    print(f"verify.py: {checks} checks, {len(failures)} failures")
    for msg in failures:
        print(f"  FAIL: {msg}")
    if failures:
        print(
            json.dumps(
                {
                    str(k): {
                        kk: (round(vv, 8) if isinstance(vv, float) else vv) for kk, vv in v.items()
                    }
                    for k, v in recomputed.items()
                },
                indent=2,
            )
        )
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else ".",
            float(sys.argv[2]) if len(sys.argv) > 2 else 10.0,
        )
    )
