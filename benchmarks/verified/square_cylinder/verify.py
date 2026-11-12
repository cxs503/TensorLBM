#!/usr/bin/env python3
"""square_cylinder archive verify — v2, schema-agnostic (audit m16183 remediation).

What was wrong in v1 (audit m16183, verdict reopened/request-changes)
--------------------------------------------------------------------
1. **Two reference conventions.** v1 hard-coded the superseded pair
   ``1.4785 / 0.1456`` as REF constants while ``run.py`` (the producer) and ``README`` use
   ``1.48 / 0.147``.  The same measured ``Cd``/``St`` therefore produced two
   different error percentages, i.e. "PASS" was not a well-defined claim.
   Now all constants come from ``refspec.py`` (single source of truth) and the
   stored ``err_pct``/``cd_ref`` in the archive are checked against it, so a
   silent convention change is a hard FAIL instead of an invisible fork.
2. **v1 could not run on the current tree.** It required every ``case_D*.json``
   to carry the triple force-column schema (``wet``/``pair``/``mem``) and did
   ``cases[ds[-1]]["pair"]``; but the D=32/D=48 archives are *summary* schema,
   so v1 raised ``IndexError`` — the "verified" claim was not machine-checkable.
   Now both schemas are flattened through ``refspec.normalize_case()``.
3. **The acceptance metric did not name its force column.** ``wet`` differs from
   ``pair``/``mem`` by +8.5 % in ``Cd`` at D=64, which alone exceeds the 3 %
   bar.  The main gate is now scored on the locked primary column
   (``refspec.PRIMARY_COLUMN``), ``wet`` is reported as a diagnostic, and the
   column used is printed in the header.

Gates (stdlib only; no torch/tensorlbm import)
----------------------------------------------
 1. archive present: ``case_D*.json`` found, D=32 and D=48 among them
 2. per-case geometry: nx = ny = 40D, solid cells = D^2, surface links = 12D-4
 3. reference-convention drift: stored ``cd_ref``/``st_ref`` == refspec, and
    stored ``err_pct``/``st_err_pct`` == recomputed (<=0.01 pp)
 4. force-column closure (triple schema only): |pair-mem| <= 0.5 %, wet-mem
    A/B recorded as a diagnostic; SKIP (not FAIL) when a case has no columns
 5. main gate, finest grid, primary column: |Cd err| <= 3 % and |St err| <= 3 %
 6. reference band: primary values inside the locked cluster bands
 7. monotonicity of the primary-column error with the <=1 % resolution floor
 8. result.json verdict flags agree with the recomputed gates (both the legacy
    ``verdict_prereg`` and the current ``convergence`` key are accepted)
 9. reproduction against the replaced 2026-09-29 archive, <= 0.5 % — only when
    a fresh re-run directory is supplied via ``--repro-dir``; otherwise SKIP
    with the reason (audit item 4 is open, and a SKIP is not a PASS).

Usage::

    python verify.py [STAGE_DIR] [--repro-dir DIR]

Exit code 0 only when no gate FAILs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refspec as R  # noqa: E402

failures: list[str] = []
skipped: list[str] = []
n_ok = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global n_ok
    tag = "OK  " if ok else "FAIL"
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if ok:
        n_ok += 1
    else:
        failures.append(name)


def skip(name: str, reason: str) -> None:
    print(f"[SKIP] {name} — {reason}", flush=True)
    skipped.append(name)


def geometry_checks(d: int, rec: dict) -> None:
    check(
        f"D{d}: domain 40D x 40D",
        rec["nx"] == rec["ny"] == int(R.DOMAIN_D * d),
        f"{rec['nx']}x{rec['ny']}",
    )
    if rec["schema"] == "triple":
        check(
            f"D{d}: solid cells = D^2",
            rec["n_solid_cells"] == d * d,
            f"{rec['n_solid_cells']}",
        )
        check(
            f"D{d}: surface links = 12D-4",
            rec["surface_links"] == 12 * d - 4,
            f"{rec['surface_links']}",
        )
    else:
        skip(
            f"D{d}: solid cells / surface links",
            "summary schema carries no force_columns (geometry not re-derivable from this record)",
        )


def drift_checks(d: int, rec: dict) -> None:
    """Reference-convention drift: are the stored error columns auditable?

    A stored ``err_pct`` is only meaningful together with the constants it was
    computed against.  Two conventions have existed in this case directory
    (``refspec.REF_CD/REF_ST`` and the superseded ``refspec.SUPERSEDED_REF``);
    the audit's complaint was that both were live *anonymous* scoring paths.
    Now every stored error must be attributable to a named convention, and the
    locked convention is the only one anything is scored against.
    """
    ecd = R.rel_err_pct(rec["primary_cd"], R.REF_CD)
    est = R.rel_err_pct(rec["primary_st"], R.REF_ST)
    print(
        f"       D{d} primary({rec['primary']}): Cd {rec['primary_cd']:.4f} St {rec['primary_st']:.4f}"
        f" -> locked-convention errors Cd {ecd:+.2f}% St {est:+.2f}%"
    )

    stored = rec.get("stored_err_pct") or {}
    sref = rec.get("stored_ref") or {}
    if rec["schema"] == "summary":
        check(
            f"D{d}: stored cd_ref == refspec.REF_CD",
            sref.get("cd") is not None and abs(sref["cd"] - R.REF_CD) < 1e-12,
            f"stored {sref.get('cd')} vs refspec {R.REF_CD}",
        )
        check(
            f"D{d}: stored st_ref == refspec.REF_ST",
            sref.get("st") is not None and abs(sref["st"] - R.REF_ST) < 1e-12,
            f"stored {sref.get('st')} vs refspec {R.REF_ST}",
        )

    if stored.get("cd") is None and stored.get("st") is None:
        skip(f"D{d}: stored error provenance", "record carries no stored error percentage")
        return

    conv = R.convention_for(rec["primary_cd"], rec["primary_st"], stored.get("cd"), stored.get("st"))
    check(
        f"D{d}: stored err attributable to a named convention",
        conv is not None,
        f"{conv} (stored Cd {stored.get('cd')}% St {stored.get('st')}%)",
    )
    if conv == "superseded":
        print(
            f"       NOTE D{d}: stored errors are the DECLARED LEGACY 1.4785/0.1456 convention"
            f" (see REFERENCE_AUDIT.md §6); they are not scored. Locked-convention values"
            f" above are the acceptance numbers."
        )


def column_checks(d: int, rec: dict) -> None:
    if rec["schema"] != "triple":
        skip(
            f"D{d}: force-column closure",
            "summary schema: single recorded Cd/St, no wet/pair/mem columns",
        )
        return
    closure = rec.get("closure_pair_vs_mem_pct")
    check(
        f"D{d}: closure |pair-mem| <= {R.CLOSURE_COLUMN_PAIR_PCT}%",
        closure is not None and abs(closure) <= R.CLOSURE_COLUMN_PAIR_PCT,
        f"{closure:+.3f}%" if closure is not None else "not recorded",
    )
    ab = rec.get("ab_wet_vs_mem_pct")
    check(
        f"D{d}: wet-mem A/B recorded (diagnostic, not scored)",
        ab is not None and 0.0 < ab < 20.0,
        f"{ab:+.2f}%" if ab is not None else "not recorded",
    )


def result_json_checks(stage: Path, finest_err: tuple[float, float]) -> None:
    rp = stage / "result.json"
    if not rp.exists():
        skip("result.json verdict flags", "result.json not present in stage dir")
        return
    res = json.loads(rp.read_text())
    conv = res.get("convergence") or {}
    vp = res.get("verdict_prereg") or {}
    cd_ok = conv.get("cd_within_3pct", vp.get("cd_within_3pct"))
    st_ok = conv.get("st_within_3pct", vp.get("st_within_3pct"))
    if cd_ok is None and st_ok is None:
        skip("result.json verdict flags", "no cd/st within-3pct flag in either schema")
        return
    rec_cd = abs(finest_err[0]) <= R.TOL_PCT
    rec_st = abs(finest_err[1]) <= R.TOL_PCT
    check(
        "result.json Cd flag agrees with recomputed gate",
        (cd_ok is None) or (bool(cd_ok) == rec_cd),
        f"file {cd_ok} vs recomputed {rec_cd}",
    )
    check(
        "result.json St flag agrees with recomputed gate",
        (st_ok is None) or (bool(st_ok) == rec_st),
        f"file {st_ok} vs recomputed {rec_st}",
    )


def repro_checks(repro_dir: Path | None, cases: dict[int, dict]) -> None:
    if repro_dir is None:
        skip(
            "repro vs replaced 2026-09-29 archive",
            "no --repro-dir supplied: audit item 4 (fresh full re-run) is open",
        )
        return
    fresh = R.load_archive(repro_dir)
    if not fresh:
        skip("repro vs replaced 2026-09-29 archive", f"no case_D*.json in {repro_dir}")
        return
    for d in sorted(fresh):
        if d not in R.OLD_ARCHIVE or d not in cases:
            continue
        ocd, ost = R.OLD_ARCHIVE[d]
        rcd, rst = fresh[d]["primary_cd"], fresh[d]["primary_st"]
        dcd = R.rel_err_pct(rcd, ocd)
        dst = R.rel_err_pct(rst, ost)
        check(
            f"D{d}: fresh re-run vs replaced archive |d| <= {R.REPRO_TOL_PCT}%",
            abs(dcd) <= R.REPRO_TOL_PCT and abs(dst) <= R.REPRO_TOL_PCT,
            f"cd {dcd:+.3f}% st {dst:+.3f}%",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="verify the square_cylinder Re=100 case archive")
    ap.add_argument("stage", nargs="?", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--repro-dir", default=None, help="fresh re-run dir with case_D*.json")
    args = ap.parse_args()
    stage = Path(args.stage)

    print(
        f"reference convention (refspec.py): Cd {R.REF_CD} in {list(R.CD_BAND)}, "
        f"St {R.REF_ST} in {list(R.ST_BAND)}, tol {R.TOL_PCT}%, "
        f"primary column '{R.PRIMARY_COLUMN}', diagnostics {list(R.DIAGNOSTIC_COLUMNS)}",
        flush=True,
    )

    cases = R.load_archive(stage)
    check(
        "case files found (D=32 and D=48 required)",
        bool(cases) and 32 in cases and 48 in cases,
        f"D={sorted(cases)}",
    )
    if not cases:
        print(f"\nVERIFY RESULT: {len(failures)} FAILURES: {failures}")
        sys.exit(1)

    for d in sorted(cases):
        geometry_checks(d, cases[d])
        drift_checks(d, cases[d])
        column_checks(d, cases[d])

    ds = sorted(cases)
    finest = ds[-1]
    errs_cd = [abs(R.rel_err_pct(cases[d]["primary_cd"], R.REF_CD)) for d in ds]
    errs_st = [abs(R.rel_err_pct(cases[d]["primary_st"], R.REF_ST)) for d in ds]
    ecd_f = R.rel_err_pct(cases[finest]["primary_cd"], R.REF_CD)
    est_f = R.rel_err_pct(cases[finest]["primary_st"], R.REF_ST)

    check(
        f"MAIN GATE finest D{finest} ({cases[finest]['primary']}): |Cd err| <= {R.TOL_PCT}%",
        abs(ecd_f) <= R.TOL_PCT,
        f"{ecd_f:+.2f}%",
    )
    check(
        f"MAIN GATE finest D{finest} ({cases[finest]['primary']}): |St err| <= {R.TOL_PCT}%",
        abs(est_f) <= R.TOL_PCT,
        f"{est_f:+.2f}%",
    )

    for d in ds:
        check(
            f"D{d}: Cd inside locked band {list(R.CD_BAND)}",
            R.within_band(cases[d]["primary_cd"], R.CD_BAND),
            f"{cases[d]['primary_cd']:.4f}",
        )
        check(
            f"D{d}: St inside locked band {list(R.ST_BAND)}",
            R.within_band(cases[d]["primary_st"], R.ST_BAND),
            f"{cases[d]['primary_st']:.4f}",
        )

    mono_cd = all(errs_cd[i + 1] <= errs_cd[i] + 1e-12 for i in range(len(errs_cd) - 1)) or all(
        e <= 1.0 for e in errs_cd
    )
    mono_st = all(errs_st[i + 1] <= errs_st[i] + 1e-12 for i in range(len(errs_st) - 1)) or all(
        e <= 1.0 for e in errs_st
    )
    check("MONOTONE Cd (or floor <= 1%)", bool(mono_cd), " -> ".join(f"{e:.2f}%" for e in errs_cd))
    check("MONOTONE St (or floor <= 1%)", bool(mono_st), " -> ".join(f"{e:.2f}%" for e in errs_st))

    result_json_checks(stage, (ecd_f, est_f))
    repro_checks(Path(args.repro_dir) if args.repro_dir else None, cases)

    print()
    print(f"counts: OK {n_ok} / SKIP {len(skipped)} / FAIL {len(failures)}")
    if failures:
        print(f"VERIFY RESULT: {len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("VERIFY RESULT: ALL OK")
    sys.exit(0)


if __name__ == "__main__":
    main()