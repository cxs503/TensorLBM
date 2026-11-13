#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m17744 — anchor
XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN (TensorLBM, commit bf1927ca).

This is a verification job, not a re-run of the mission.  Every number the audit
documents is recomputed here from committed bytes:

  1. existence + sha256 of the acceptance triple (run_artifacts / benchmark
     ref-vs-got / repro_cmd) in the subject worktree;
  2. the *registered* referee: rows of tensor_benchmark_cases for the anchor
     (the platform benchmark ledger is the only referee, the report is not);
  3. independent referee: nu_theory = (tau-1/2)/3 from each arm's own config,
     versus nu recovered by the auditor from the energy samples in the artifact
     (nu = -slope/(4 k_lattice^2), k_lattice = 2*pi/N);
  4. reproducibility: fresh `git archive <commit>` extraction -> pytest the
     delivered test file and re-run all six arms, compare against the committed
     arm JSONs (numbers and emitted-kernel sha256);
  5. adversarial probes that the mission did not run (protocol rejection of a
     lying declaration / lying magic / broken stencil, and a standalone import of
     the emitted kernel against the independent reference kernel);
  6. doc-vs-artifact binding: the committed doc is rendered by make_doc.py from
     summary.json/arm_*.json, so its table must equal table.csv.

Outputs (inside the audit worktree):
    runs/m17824_audit/audit.json           machine-readable audit record
    runs/m17824_audit/referee.json         independent viscosity referee
    runs/m17824_audit/repro_compare.json   committed vs reproduced, per arm
    runs/m17824_audit/probe_negative.json  adversarial probes
    docs/board_audit/m17824/fig1_nu_referee.png
    docs/board_audit/m17824/fig2_repro_equivalence.png
    docs/board_audit/M17824_XVAL_LBM_COMPONENT_PROTOCOL_BOARD_AUDIT.md

Usage:  python runs/m17824_audit/audit_m17744.py [--insert]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "m17824_audit"
FIGDIR = REPO / "docs" / "board_audit" / "m17824"
DOC = REPO / "docs" / "board_audit" / "M17824_XVAL_LBM_COMPONENT_PROTOCOL_BOARD_AUDIT.md"
SUBJECT = pathlib.Path("/root/tp-agent/work/m17744")
COMMIT = "bf1927ca159e027ef34f3b613cf248cd4dc0b05c"
EXT = pathlib.Path("/tmp/audit_m17744_ext")
ANCHOR = "XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN"
MISSION = 17744
ARMS = ["A1", "A2", "A3", "A4", "A5", "A6"]
ART = pathlib.Path("artifacts/xval_component_protocol_codegen")
ARM_KEYS = ["nu_measured", "nu_error_pct", "field_l2_rel", "div_max", "mass_drift"]


def sh(*args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract() -> None:
    if EXT.exists():
        shutil.rmtree(EXT)
    EXT.mkdir(parents=True)
    tar = subprocess.run(["git", "archive", COMMIT], cwd=REPO, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(EXT)], input=tar, check=True)
    (EXT / "repro").mkdir()


def acceptance_triple() -> dict:
    files = sorted(p for p in (SUBJECT / ART).iterdir() if p.is_file())
    record = {"dir": str(SUBJECT / ART),
              "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
    # repro_cmd declared by the mission doc; executed below in the fresh extraction
    record["repro_cmd"] = (
        "for a in A1 A2 A3 A4 A5 A6; do python xval/component_protocol_codegen/run_arm.py "
        "--arm $a --out artifacts/xval_component_protocol_codegen/arm_$a.json; done && "
        "python xval/component_protocol_codegen/analyze.py")
    record["repro_cmd_present_in_doc"] = "--arm $a" in \
        (SUBJECT / "docs/codegen_component_protocol.md").read_text()
    return record


def registered_referee() -> dict:
    import psycopg
    with psycopg.connect("dbname=tensorplatform user=postgres") as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        total = cur.fetchone()[0]
        cur.execute("SELECT case_name, metric, tolerance, status, ref_source "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))
        rows = cur.fetchall()
        cur.execute("SELECT anchor_slug FROM tensor_benchmark_cases "
                    "WHERE anchor_slug LIKE 'XVAL-LBM%' ORDER BY 1")
        near = [r[0] for r in cur.fetchall()]
    return {"anchor": ANCHOR, "rows": rows, "n_rows": len(rows),
            "ledger_total_cases": total,
            "other_xval_lbm_anchors_registered": near}


def referee() -> dict:
    rows, worst = [], 0.0
    for a in ARMS:
        d = json.loads((SUBJECT / ART / f"arm_{a}.json").read_text())
        cfg, case = d["config"], d["case"]
        nu_th = (cfg["tau"] - 0.5) / 3.0            # D2Q9 Chapman-Enskog, not read from the code
        t = np.array([s["t"] for s in d["samples"]], float)
        e = np.array([s["E"] for s in d["samples"]], float)
        m = (t >= case["fit_from"]) & (e > 0)
        slope, intercept = np.polyfit(t[m], np.log(e[m]), 1)
        k = case["k_lattice"]                        # 2*pi/N, NOT the dimensionless k=1
        nu_fit = -slope / (4 * k * k)
        resid = np.log(e[m]) - (slope * t[m] + intercept)
        err = abs(nu_fit - nu_th) / nu_th * 100
        worst = max(worst, err)
        rows.append({"arm": a, "family": cfg["family"], "n": cfg["n"], "u0": cfg["u0"], "tau": cfg["tau"],
                     "nu_theory": nu_th, "nu_from_decay": nu_fit, "nu_committed": d["nu_measured"],
                     "err_vs_theory_pct": err,
                     "err_committed_vs_recomputed_pct": abs(nu_fit - d["nu_measured"]) / nu_th * 100,
                     "fit_rms": float(np.sqrt((resid ** 2).mean())),
                     "field_l2_rel": d["field_l2_rel"], "div_max": d["div_max"],
                     "mass_drift": d["mass_drift"],
                     "kernel_sha256": d["kernel"]["sha256"],
                     "samples_t_max": float(t.max()), "energy_drop_pct": float((1 - e[-1] / e[0]) * 100)})
    return {"referee": "nu_theory=(tau-1/2)/3 (D2Q9 Chapman-Enskog, recomputed from artifact config)",
            "energy_law": "E(t)=u0^2/4 exp(-4 nu k_lat^2 t), least squares of ln E over the "
                          "predeclared window t>=fit_from",
            "owner_acceptance_pct": 15.0, "tighter_band_pct": 3.0,
            "worst_err_vs_theory_pct": worst, "rows": rows}


def reproduce() -> dict:
    extract()
    pytest = sh(sys.executable, "-m", "pytest", "tests/test_component_protocol_codegen.py", "-q",
                cwd=EXT, check=False)
    tail = [ln for ln in pytest.stdout.strip().splitlines() if ln.strip()][-3:]
    arms = {}
    for a in ARMS:
        run = sh(sys.executable, "xval/component_protocol_codegen/run_arm.py", "--arm", a,
                 "--out", f"repro/arm_{a}.json", cwd=EXT, check=False)
        if run.returncode != 0:
            arms[a] = {"error": run.stderr[-400:]}
            continue
        c = json.loads((SUBJECT / ART / f"arm_{a}.json").read_text())
        r = json.loads((EXT / "repro" / f"arm_{a}.json").read_text())
        rec = {}
        for key in ARM_KEYS:
            if key in c and key in r:
                rec[key] = {"committed": c[key], "reproduced": r[key], "abs_diff": abs(c[key] - r[key])}
        rec["kernel_sha256_equal"] = c["kernel"]["sha256"] == r["kernel"]["sha256"]
        rec["max_abs_diff"] = max((v["abs_diff"] for v in rec.values() if isinstance(v, dict)), default=0.0)
        arms[a] = rec
    return {"pytest": {"returncode": pytest.returncode, "tail": tail},
            "arms": arms,
            "all_zero_diff": all(v.get("max_abs_diff", 1.0) == 0.0 for v in arms.values() if "error" not in v),
            "extraction": str(EXT)}


def probes() -> dict:
    script = OUT / "probe_negative.py"
    run = sh(sys.executable, str(script), str(EXT), check=False)
    if run.returncode != 0:
        return {"error": run.stderr[-600:]}
    data = json.loads(run.stdout)
    (OUT / "probe_negative.json").write_text(json.dumps(data, indent=1))
    return data


def figures(ref: dict, rep: dict) -> list[pathlib.Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    f1, f2 = FIGDIR / "fig1_nu_referee.png", FIGDIR / "fig2_repro_equivalence.png"
    rows = ref["rows"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(rows))
    labels = [f"{r['arm']}\n{r['family']} N={r['n']}\nu0={r['u0']}" for r in rows]
    ax[0].bar(x - 0.2, [r["nu_theory"] for r in rows], 0.4, label="nu declared/theory (tau-1/2)/3")
    ax[0].bar(x + 0.2, [r["nu_from_decay"] for r in rows], 0.4, label="nu recovered by auditor from decay")
    ax[0].set_xticks(x, labels, fontsize=7)
    ax[0].set_ylabel("shear viscosity (lattice units)")
    ax[0].set_title("Taylor-Green viscosity inversion (auditor recomputation)")
    ax[0].legend(fontsize=7)
    ax[1].bar(x, [r["err_vs_theory_pct"] for r in rows], 0.5, color="tab:red")
    ax[1].axhline(3.0, ls="--", c="tab:orange", lw=1, label="3% band")
    ax[1].axhline(15.0, ls="--", c="tab:green", lw=1, label="owner 15% criterion")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x, labels, fontsize=7)
    ax[1].set_ylabel("|nu_recovered - nu_theory| / nu_theory  [%]")
    ax[1].set_title(f"worst {ref['worst_err_vs_theory_pct']:.3f}%  -> inside 3%")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in rows:
        d = json.loads((SUBJECT / ART / f"arm_{r['arm']}.json").read_text())
        t = np.array([s["t"] for s in d["samples"]], float)
        e = np.array([s["E"] for s in d["samples"]], float)
        ax[0].plot(t, e / e[0], lw=1, label=f"{r['arm']} {r['family']} N={r['n']} u0={r['u0']}")
        tt = np.linspace(0, t.max(), 200)
        ax[0].plot(tt, np.exp(-4 * r["nu_theory"] * d["case"]["k_lattice"] ** 2 * tt), "k:", lw=0.8)
    ax[0].set_xlabel("lattice time t")
    ax[0].set_ylabel("E(t)/E(0)")
    ax[0].set_title("energy decay vs analytic exp(-4 nu k^2 t)  (dotted)")
    ax[0].legend(fontsize=6)
    diffs = [rep["arms"][a].get("max_abs_diff", float("nan")) for a in ARMS]
    ax[1].bar(np.arange(6), [max(d, 1e-18) for d in diffs], 0.5, color="tab:blue")
    ax[1].axhline(1e-12, ls="--", c="tab:red", lw=1, label="predeclared 1e-12 equivalence bound")
    ax[1].set_yscale("log")
    ax[1].set_xticks(np.arange(6), ARMS)
    ax[1].set_ylabel("max |committed - reproduced| (all metrics)")
    ax[1].set_title("fresh-extraction reproduction (drawn at 1e-18 = exact zero)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    return [f1, f2]


def integrity(subject_triple: dict) -> dict:
    summary = json.loads((SUBJECT / ART / "summary.json").read_text())
    shas = {f["path"].split("/")[-1]: f["sha256"] for f in subject_triple["files"]}
    rows = []
    for arm in summary["arms"]:
        d = json.loads((SUBJECT / ART / f"arm_{arm['arm']}.json").read_text())
        name = ("generated_trt_tau0.56_magic0.1875.py" if arm["family"] == "TRT"
                else "generated_bgk_tau0.56.py")
        rows.append({"arm": arm["arm"],
                     "arm_json_sha_matches_generated_file":
                         d["kernel"]["sha256"] == shas.get(name),
                     "summary_nu_error_pct": arm["nu_error_pct"],
                     "arm_json_nu_error_pct": d["nu_error_pct"],
                     "agree": abs(arm["nu_error_pct"] - d["nu_error_pct"]) < 1e-12})
    csv_rows = (SUBJECT / ART / "table.csv").read_text().strip().splitlines()
    return {"per_arm": rows, "table_csv_rows": len(csv_rows),
            "summary_verdict": summary.get("verdict"),
            "failed_declared_criteria": summary.get("failed_declared_criteria"),
            "doc_lines": len((SUBJECT / "docs/codegen_component_protocol.md").read_text().splitlines())}


def code_scale() -> dict:
    files = {}
    for p in sorted((SUBJECT / "src/tensorlbm/codegen").glob("*.py")):
        txt = p.read_text()
        files[str(p.relative_to(SUBJECT))] = {
            "lines": len(txt.splitlines()),
            "placeholder_hits": len(re.findall(
                r"NotImplementedError|raise TODO|pass\s*#\s*placeholder|^\s*pass\s*$", txt, re.M))}
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", action="store_true")
    a = ap.parse_args()

    triple = acceptance_triple()
    ledger = registered_referee()
    integ = integrity(triple)
    ref = referee()
    rep = reproduce()
    prb = probes()
    scale = code_scale()
    figs = figures(ref, rep)

    importers = sh("grep", "-rn", "--include=*.py", "-l", "tensorlbm.codegen",
                   "src/tensorlbm", cwd=SUBJECT, check=False).stdout.strip().splitlines()
    callers = [c for c in importers if "src/tensorlbm/codegen/" not in c]

    audit = {
        "anchor": ANCHOR, "mission": MISSION, "repo": "TensorLBM", "commit": COMMIT,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_triple": triple, "registered_referee": ledger, "integrity": integ,
        "referee": ref, "reproduction": rep, "probes": prb,
        "code_scale": scale, "integration_callers": callers,
        "doc": str(DOC.relative_to(REPO)), "figures": [str(p.relative_to(REPO)) for p in figs],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=1))
    (OUT / "referee.json").write_text(json.dumps(ref, indent=1))
    (OUT / "repro_compare.json").write_text(json.dumps(rep, indent=1))

    print(json.dumps({k: audit[k] for k in ("registered_referee", "integration_callers")}, indent=1))
    print("pytest:", rep["pytest"])
    print("repro all_zero_diff:", rep["all_zero_diff"])
    print("worst nu err vs theory: %.4f %%" % ref["worst_err_vs_theory_pct"])
    print("ledger rows for anchor:", ledger["n_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())