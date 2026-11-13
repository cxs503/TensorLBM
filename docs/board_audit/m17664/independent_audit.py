#!/usr/bin/env python3
"""Independent board audit of mission m17603 (anchor XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE).

What this script does — and does *not* do
-----------------------------------------
It does NOT trust the shipped ``artifacts/m17603/xval_inamuro_wetnode.json``.
It re-derives every number it reports from raw behaviour:

  A1 ref authenticity  sha256 of the mission-archived OSS ``.dat`` fields vs a
                       *fresh* ``g++ -O2`` build+run of the archived upstream
                       source ``artifacts/m17603_oss_src/chapter8/film_inamuro.cpp``
                       (``/tmp/audit17664/oss/run/film_inamuro``).  Detects a
                       doctored / perturbed reference.
  A2 refusal of the shipped driver on a fresh outdir (the README "Reproduce"
                       path, executed verbatim) — reported as-is, not repaired.
  A3 fresh field run    re-execute the film problem through the production
                       module ``tensorlbm.boundary.inamuro_wetnode`` at the full
                       upstream horizon (N=50000) and recompute rel L2 / Linf
                       against the *fresh* reference fields.
  A4 shipped-number     compare the recomputed deviations with the numbers in
     cross-check        the shipped json (checkpoints 100/1000/10000 overlap).
  A5 control probe     sabotage a *covered* quantity (skip the scalar Dirichlet
                       closure) and confirm the metric explodes -> the gate has
                       detection power, i.e. 1e-11 is not a vacuous pass.
  A6 coupling probe    static: does anything under ``src/`` besides the module
                       itself reference the new closures (= solver integration)?
  A7 referee probe     machine referee row in ``tensor_benchmark_cases`` and the
                       report / verdict rows for the mission.

Run:
  python3 docs/board_audit/m17664/independent_audit.py \
      --lbm-worktree /root/tp-agent/work/m17603 \
      --oss-fields /tmp/audit17664/oss/run/film_inamuro \
      --steps 50000 --out docs/board_audit/m17664
Exit 0 always (evidence collection); the verdict lives in audit.json / AUDIT_m17664.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ANCHOR = "XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE"
CHECKPOINTS = (100, 1000, 10000, 12000)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_driver(lbm: Path):
    """Import the mission's comparison driver as a module (no copy, no edit)."""
    sys.path.insert(0, str(lbm / "src"))
    path = lbm / "benchmarks/verified/inamuro_wetnode_xval/run.py"
    spec = importlib.util.spec_from_file_location("m17603_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_ref_authenticity(lbm: Path, oss_fields: Path) -> dict:
    """A1: fresh upstream run vs the fields the mission archived."""
    rows = {}
    archived = lbm / "artifacts/m17603/oss_ref/film_inamuro"
    for step in CHECKPOINTS:
        name = f"film_inamuro{step:07d}.dat"
        fresh, ship = oss_fields / name, archived / name
        rows[name] = {
            "fresh_sha256": sha256(fresh) if fresh.is_file() else None,
            "shipped_sha256": sha256(ship) if ship.is_file() else None,
        }
        rows[name]["match"] = (
            rows[name]["fresh_sha256"] is not None
            and rows[name]["fresh_sha256"] == rows[name]["shipped_sha256"]
        )
    return {
        "probe": "A1 ref authenticity (fresh g++ -O2 build of the archived upstream source)",
        "all_match": all(r["match"] for r in rows.values()),
        "rows": rows,
    }


def probe_documented_repro(lbm: Path, tmp: Path) -> dict:
    """A2: run the README 'Reproduce' step 2 verbatim into a fresh outdir."""
    out = tmp / "readme_repro"
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)], check=False)
    env = dict(os.environ, PYTHONPATH=str(lbm / "src"))
    proc = subprocess.run(
        [sys.executable, "benchmarks/verified/inamuro_wetnode_xval/run.py",
         "--part", "film", "--steps", "12000", "--devsteps", "2000",
         "--oss-ref", str(lbm / "artifacts/m17603_oss_src"), "--out", str(out)],
        cwd=str(lbm), env=env, capture_output=True, text=True, timeout=1800,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return {
        "probe": "A2 documented repro on a FRESH outdir (README §Reproduce step 2)",
        "cmd": "PYTHONPATH=src python3 benchmarks/verified/inamuro_wetnode_xval/run.py "
               "--part film --steps 12000 --devsteps 2000 "
               "--oss-ref artifacts/m17603_oss_src --out <fresh>",
        "returncode": proc.returncode,
        "tail": tail[-4:],
        "succeeded": proc.returncode == 0,
        "note": "shipped outdir carries a hand-built oss_ref/ layout; a fresh outdir "
                "collides the compiled binary name (oss_ref/film_inamuro) with the "
                "directory the upstream binary writes into.",
    }


def probe_fresh_run(lbm: Path, oss_fields: Path, steps: int, out: Path) -> dict:
    """A3/A4: re-run the film through the production module, recompute deviations."""
    mod = load_driver(lbm)
    params = dict(mod.FILM)
    cps = tuple(s for s in CHECKPOINTS if s <= steps)
    snaps = mod.run_film(params, steps, scalar_policy="all_known", checkpoints=cps)

    got_fields, rows = {}, []
    for step in cps:
        got = snaps[step].detach().numpy()
        ref = mod.read_oss_field(oss_fields / f"film_inamuro{step:07d}.dat",
                                params["nx"], params["ny"])
        got_fields[f"got_{step}"] = got
        rows.append({
            "step": step,
            "got_mean": float(got.mean()),
            "ref_mean": float(ref.mean()),
            "rel_l2_vs_fresh_oss": float(mod.rel_l2(got, ref)),
            "rel_linf_vs_fresh_oss": float(mod.rel_linf(got, ref)),
            "pass_15pct": bool(mod.rel_l2(got, ref) < 0.15),
        })

    # A5 control: skip the scalar Dirichlet closure entirely -> must blow up
    orig = mod.inamuro_wetnode_scalar
    mod.inamuro_wetnode_scalar = lambda f, face, target, **kw: f
    try:
        sab = mod.run_film(params, 1000, scalar_policy="all_known", checkpoints=(1000,))
    finally:
        mod.inamuro_wetnode_scalar = orig
    ref1000 = mod.read_oss_field(oss_fields / "film_inamuro0001000.dat",
                                 params["nx"], params["ny"])
    sab_l2 = float(mod.rel_l2(sab[1000].detach().numpy(), ref1000))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "audit_fields.npz", **got_fields)
    return {
        "probe": "A3 fresh full-horizon run through tensorlbm.boundary.inamuro_wetnode",
        "steps": steps,
        "rows": rows,
        "control_probe_A5": {
            "mutation": "inamuro_wetnode_scalar -> identity (scalar Dirichlet closure removed)",
            "rel_l2_at_step_1000": sab_l2,
            "detected": sab_l2 > 0.15,
        },
        "fields_npz": str(out / "audit_fields.npz"),
    }


def probe_coupling(lbm: Path) -> dict:
    """A6: is the new closure wired into any solver path?"""
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-e", "inamuro_wetnode", "-e", "inamuro_zero_gradient",
         "-e", "inamuro_corner_average", "-e", "inamuro", str(lbm / "src")],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    outside = [h for h in hits if "boundary/inamuro_wetnode.py" not in h]
    return {
        "probe": "A6 solver coupling (static grep over src/)",
        "hits_inside_module": len(hits) - len(outside),
        "hits_outside_module": outside,
        "registered_in_boundary_registry": any(
            "inamuro" in (lbm / p).read_text(errors="ignore")
            for p in ("src/tensorlbm/boundary/boundary_registry.py",
                      "src/tensorlbm/boundary/__init__.py",
                      "src/tensorlbm/boundary/boundary_capability_contract.py")
        ),
    }


def probe_ledger(pg: str) -> dict:
    """A7: machine referee row + mission/report/verdict rows."""
    def q(sql: str) -> str:
        return subprocess.run(["psql", "-U", "postgres", "-d", pg, "-tAc", sql],
                              capture_output=True, text=True).stdout.strip()

    ref = q("SELECT count(*) FROM tensor_benchmark_cases "
            f"WHERE anchor_slug='{ANCHOR}'")
    rep = q("SELECT count(*) FROM tensor_reports WHERE mission_id=17603")
    mission = q("SELECT status || ' | finished=' || coalesce(finished_at::text,'-') "
                "FROM tensor_dev_missions WHERE id=17603")
    git = subprocess.run(["git", "-C", os.environ.get("LBM_WT", "."), "status",
                          "--short", "--", "src/tensorlbm/boundary/inamuro_wetnode.py",
                          "tests/test_inamuro_wetnode.py",
                          "benchmarks/verified/inamuro_wetnode_xval", "artifacts/m17603"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {
        "probe": "A7 ledger probes",
        "benchmark_referee_rows": int(ref),
        "tensor_reports_rows_mission_17603": int(rep),
        "mission_row": mission,
        "git_status_of_deliverables": git,
        "deliverables_tracked": not git,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbm-worktree", default="/root/tp-agent/work/m17603")
    ap.add_argument("--oss-fields", default="/tmp/audit17664/oss/run/film_inamuro")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--pg", default=os.environ.get("TP_PG_DB", "tensorplatform"))
    ap.add_argument("--out", default="docs/board_audit/m17664")
    ap.add_argument("--run-a2", action="store_true",
                    help="re-execute the README repro (costs ~70 s); default reuses "
                         "the recorded verbatim attempt in a2_recorded.json")
    args = ap.parse_args(argv)

    lbm = Path(args.lbm_worktree)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["LBM_WT"] = str(lbm)

    audit = {
        "mission": 17603,
        "anchor": ANCHOR,
        "audited_repo": "TensorLBM",
        "lbm_worktree": str(lbm),
        "A1_ref_authenticity": probe_ref_authenticity(lbm, Path(args.oss_fields)),
        "A2_documented_repro": (
            probe_documented_repro(lbm, out / "_tmp")
            if args.run_a2 else
            json.loads((out / "a2_recorded.json").read_text())
        ),
        "A3_A4_fresh_run": probe_fresh_run(lbm, Path(args.oss_fields), args.steps, out),
        "A6_coupling": probe_coupling(lbm),
        "A7_ledger": probe_ledger(args.pg),
    }

    shipped = json.loads((lbm / "artifacts/m17603/xval_inamuro_wetnode.json").read_text())
    ship_rows = {r["step"]: r for r in shipped["film"]["rows"] if r["policy"] == "all_known"}
    cross = []
    for r in audit["A3_A4_fresh_run"]["rows"]:
        s = ship_rows.get(r["step"])
        if s is None:
            continue
        cross.append({
            "step": r["step"],
            "shipped_rel_l2": s["rel_l2_vs_oss"],
            "audit_rel_l2": r["rel_l2_vs_fresh_oss"],
            "abs_delta": abs(s["rel_l2_vs_oss"] - r["rel_l2_vs_fresh_oss"]),
            "shipped_got_mean": s["got_mean"],
            "audit_got_mean": r["got_mean"],
            "got_mean_abs_delta": abs(s["got_mean"] - r["got_mean"]),
        })
    audit["A4_shipped_vs_audit"] = cross

    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items()
                      if k in ("A1_ref_authenticity", "A2_documented_repro",
                               "A4_shipped_vs_audit", "A6_coupling", "A7_ledger")},
                     indent=2)[:4000])
    print(json.dumps(audit["A3_A4_fresh_run"]["rows"], indent=2))
    print("control:", audit["A3_A4_fresh_run"]["control_probe_A5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())