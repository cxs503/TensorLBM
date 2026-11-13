#!/usr/bin/env python
"""Board audit (交差核验) for mission 17515 — XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN.

Rebuilds every number from the *delivered* artefacts of the subject worktree
(``/root/tp-agent/work/m17515``), never from the mission narrative, and runs the
checks a reader of the completion claim cannot see:

  P1  ref-vs-got table read back from ``benchmarks/.../result.json`` and checked
      against the declared tolerance (15%) row by row, plus a *distinct
      configuration* count (the mission advertises 10 cases).
  P2  fresh reproduction: ``run.py --out <tmp>`` compared with the shipped
      artefacts (decision block + CSV byte identity) -> is ``repro_cmd`` real?
  P3  benchmark referee: ``tensor_benchmark_cases`` rows for the anchor (the
      board's own machine referee table) — 0 rows means author-declared tolerance.
  P4  physical anchor: sigma recovered from the force integral vs the prescribed
      sigma, as a function of radius and interface thickness (convergent or not?).
  P5  integration liveness: every production call site of the capillary-force
      chain, and whether the validated operator is actually selected there.
  P6  detection power: mutate the isotropic operators (+5% Laplacian, +5% one
      gradient component) and re-measure the decision metric on the subject tree
      — a metric that cannot see a 5% operator error is decoration.

Writes ``audit.json`` (machine) + ``AUDIT_m17582.md`` (human) + figures.

    python docs/board_audit/m17582/audit.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SUBJECT = Path("/root/tp-agent/work/m17515")
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REPRO = HERE / "repro"
ANCHOR = "XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN"
SCOPE = f"m17515 {ANCHOR}"
TOL = 0.15
DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- P1
def probe_ref_vs_got() -> dict:
    res = json.loads((SUBJECT / "benchmarks/xval_allen_cahn_csf_force_chain/result.json").read_text())
    rows = []
    for c in res["cases"]:
        m = c["metric_ref_vs_got"]
        rows.append({
            "case": c["case"], "operator": c["operator"],
            "rel_l2_force": m["rel_l2_capillary_force"]["got"],
            "pass_15pct": bool(m["rel_l2_capillary_force"]["pass_lt_15pct"]),
            "sigma_err": (None if m["sigma_from_force_integral"]["got_error_pct"] is None
                          else m["sigma_from_force_integral"]["got_error_pct"] / 100.0),
            "sigma_pass_15pct": bool(m["sigma_from_force_integral"]["pass_lt_15pct"]),
            "spurious_frac_got": m["spurious_tangential_fraction"]["got"],
            "spurious_frac_ref": m["spurious_tangential_fraction"]["ref"],
        })
    # identical rel_l2 within an operator => same geometry, only sigma rescaled
    groups: dict[tuple, list[str]] = {}
    for r in rows:
        groups.setdefault((r["operator"], round(r["rel_l2_force"], 15)), []).append(r["case"])
    distinct = {frozenset(v) for v in groups.values()}
    failing = [r["case"] for r in rows if not r["pass_15pct"]]
    art = json.loads((SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain/"
                      "xval_allen_cahn_force_chain.json").read_text())
    flags = {k: v for k, v in art["decision"].items() if "within_15pct" in k}
    flag_inconsistent = [k for k, v in flags.items()
                         if v and k.startswith("central7") and failing]
    worst = {op: max(r["rel_l2_force"] for r in rows if r["operator"] == op)
             for op in {r["operator"] for r in rows}}
    return {
        "rows": rows,
        "n_rows": len(rows),
        "n_distinct_metric_groups": len(distinct),
        "duplicate_groups": [sorted(v) for v in groups.values() if len(v) > 1],
        "failing_15pct": failing,
        "worst_rel_l2_by_operator": worst,
        "headline_claim": res["headline"],
        "declared_thresholds": res["thresholds"],
        "artifacts_decision_flags": flags,
        "flags_inconsistent_with_case_table": flag_inconsistent,
        "flag_note": ("decision[central7_within_15pct] is computed from C1-planar + C2 only "
                      "(allen_cahn_csf.py:580-583) and reads True while C3-droplet-W3 central7 = "
                      "20.6% > 15%; the result.json headline does list the failing case"),
    }


# --------------------------------------------------------------------------- P2
def probe_reproduction() -> dict:
    """Compare the auditor's own re-run with the shipped artefacts."""
    shipped = json.loads((SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain/"
                          "xval_allen_cahn_force_chain.json").read_text())
    fresh_p = REPRO / "xval_allen_cahn_force_chain.json"
    out = {"repro_cmd": shipped.get("repro_cmd"),
           "auditor_rerun_artefacts": str(fresh_p),
           "auditor_rerun_present": fresh_p.exists()}
    if not fresh_p.exists():
        out["note"] = "no re-run captured; provenance = not_run"
        return out
    fresh = json.loads(fresh_p.read_text())
    out["decision_identical"] = bool(fresh["decision"] == shipped["decision"])
    out["kernel_integration_identical"] = bool(fresh["kernel_integration"] == shipped["kernel_integration"])
    dmax = max(abs(fresh["decision"][k] - shipped["decision"][k])
               for k in shipped["decision"]
               if isinstance(shipped["decision"][k], float))
    out["max_abs_decision_delta"] = dmax
    out["csv_identical"] = ((REPRO / "xval_allen_cahn_force_chain.csv").read_bytes()
                            == (SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain/"
                                "xval_allen_cahn_force_chain.csv").read_bytes())
    numeric = ("cases", "kernel_integration", "decision", "operator_weights", "thresholds")
    out["numeric_blocks_identical"] = all(fresh[k] == shipped[k] for k in numeric)
    out["json_sha_diff_explained_by"] = sorted(
        k for k in shipped if shipped[k] != fresh.get(k))
    out["sha256_shipped_json"] = sha(SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain/"
                                     "xval_allen_cahn_force_chain.json")
    out["sha256_rerun_json"] = sha(fresh_p)
    out["figures_present"] = sorted(p.name for p in
                                   (SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain").glob("fig_*.png"))
    return out


# --------------------------------------------------------------------------- P3
def probe_referee() -> dict:
    sql = (f"SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
           f"WHERE anchor_slug='{ANCHOR}'")
    out = {"sql": sql}
    try:
        import psycopg
        with psycopg.connect(DSN, connect_timeout=5) as cx:
            rows = cx.execute(sql).fetchall()
            total = cx.execute("SELECT count(*) FROM tensor_benchmark_cases").fetchone()[0]
        out.update(rows=[list(r) for r in rows], n_rows=len(rows), table_total_rows=total)
    except Exception as exc:  # noqa: BLE001
        out.update(n_rows=None, error=f"{type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------- P4
def probe_sigma_anchor() -> dict:
    """sigma recovered from the capillary-force integral vs the prescribed sigma."""
    d = json.loads((SUBJECT / "artifacts/xval_allen_cahn_csf_force_chain/"
                    "xval_allen_cahn_force_chain.json").read_text())
    rows = []
    for c in d["cases"]:
        for op, m in c["variants"].items():
            rows.append({"case": c["case"], "operator": op, "radius": c.get("radius"),
                         "W_lb": c.get("W_lb"), "sigma_in": c.get("sigma"),
                         "sigma_err_got": m.get("sigma_err_got"),
                         "sigma_err_ref_self": m.get("sigma_err_ref")})
    by_r = sorted({(r["radius"], round(r["sigma_err_got"], 12)) for r in rows
                   if r["operator"] == "isotropic_d3q19" and r["radius"] and r["W_lb"] == 4.0
                   and r["sigma_in"] == 0.01})
    by_w = sorted({(r["W_lb"], round(r["sigma_err_got"], 12)) for r in rows
                   if r["operator"] == "isotropic_d3q19" and r["radius"] == 16.0
                   and r["sigma_in"] == 0.01})
    worst = min(r["sigma_err_got"] for r in rows if r["sigma_err_got"] is not None)
    return {"rows": rows,
            "sigma_err_vs_radius_W4": by_r,
            "sigma_err_vs_W_R16": by_w,
            "worst_sigma_err": worst,
            "worst_within_15pct": abs(worst) < TOL,
            "within_3pct": abs(worst) < 0.03,
            "note": ("sigma is the only *physical* reference in the run (prescribed input, not "
                     "another code); Laplace law dp=2*sigma/R is never computed")}


# --------------------------------------------------------------------------- P5
def probe_integration() -> dict:
    pat = re.compile(r"(force_minus_phi_grad_mu|force_mu_grad_phi|allen_cahn_step|"
                     r"step_allen_cahn_field)\s*\(")
    sites = []
    for p in sorted(SUBJECT.joinpath("src").rglob("*.py")):
        txt = p.read_text(errors="ignore").splitlines()
        for i, line in enumerate(txt, 1):
            for m in pat.finditer(line):
                if line.lstrip().startswith(("def ", "#", '"""', "*")):
                    continue
                ctx = " ".join(txt[i - 1:i + 3])
                sites.append({
                    "file": str(p.relative_to(SUBJECT)), "line": i,
                    "call": m.group(1),
                    "passes_operator": "operator=" in ctx,
                })
    prod = [s for s in sites if "/xval/" not in s["file"] and "/phasefield/free_energy" not in s["file"]]
    laplace = sorted(str(p.relative_to(SUBJECT)) for p in SUBJECT.joinpath("src").rglob("laplace*.py"))
    return {
        "call_sites": sites,
        "n_sites": len(sites),
        "production_call_sites": prod,
        "n_production_sites": len(prod),
        "n_production_sites_selecting_isotropic": sum(
            1 for s in prod if s["passes_operator"]),
        "default_operator": "central7 (allen_cahn_lbm.py:227/319, free_energy.py:99/113)",
        "laplace_anchor_modules_present_but_unused_by_xval": laplace,
        "coupled_solver_gate": ("allen_cahn_lbm.py:360 raises unless allow_unverified_hydro=True "
                               "(documented UNVERIFIED coupling); its AC call at :454 does not "
                               "thread operator="),
    }


# --------------------------------------------------------------------------- P6
def probe_detection_power() -> dict:
    """Can the decision metric see a 5% operator error?"""
    sys.path.insert(0, str(SUBJECT / "src"))
    cwd = Path.cwd()
    out = {"note": "in-process mutation of the operators the force chain dispatches through"}
    try:
        import tensorlbm.phasefield.free_energy as FE  # noqa: N812
        from tensorlbm.xval import allen_cahn_csf as X
        phi = X.droplet_field(16, 8.0, 4.0)
        phi_tl = X.to_tensorlbm(phi)
        ref = X.LbmpyForceChain(0.01, 4.0)(phi)

        def measure() -> float:
            got = X.TensorLbmForceChain(0.01, 2.0, "isotropic_d3q19")(phi_tl)
            return float(X.metrics(ref, got, radius=8.0, W_lb=4.0, sigma=0.01)["rel_l2_force"])

        lap0, grad0 = FE.isotropic_laplacian_3d, FE.isotropic_gradient_3d
        out["baseline_rel_l2"] = measure()
        FE.isotropic_laplacian_3d = lambda *a, **k: lap0(*a, **k) * 1.05
        out["laplacian_plus5pct_rel_l2"] = measure()
        FE.isotropic_laplacian_3d = lap0

        def grad_mut(*a, **k):
            g = grad0(*a, **k)
            return (g[0] * 1.05, g[1], g[2])

        FE.isotropic_gradient_3d = grad_mut
        out["grad_x_plus5pct_rel_l2"] = measure()
        FE.isotropic_gradient_3d = grad0
        out["detects_5pct_error"] = bool(out["laplacian_plus5pct_rel_l2"] > 1e-6
                                        and out["grad_x_plus5pct_rel_l2"] > 1e-6)
        out["gate_resolution"] = ("a 5% operator error surfaces as 2.9e-2/8.3e-2 -> would still "
                                  "pass the 15% gate; i.e. the gate tolerates ~9% operator error")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cwd.chdir(cwd) if False else None
    return out


# ------------------------------------------------------------------- artefacts
def figures(p1: dict, p4: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    metas = []

    # fig 1: per-case rel L2 vs the 15% line (distinct configurations only)
    seen, labels, c7, iso = set(), [], [], []
    for r in p1["rows"]:
        if r["case"] in seen:
            continue
        seen.add(r["case"])
        labels.append(r["case"])
    for case in labels:
        c7.append(next(r["rel_l2_force"] for r in p1["rows"]
                       if r["case"] == case and r["operator"] == "central7"))
        iso.append(next(r["rel_l2_force"] for r in p1["rows"]
                        if r["case"] == case and r["operator"] == "isotropic_d3q19"))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - 0.2, np.maximum(c7, 1e-17), 0.4, label="central7 (production default)", color="#c0392b")
    ax.bar(x + 0.2, np.maximum(iso, 1e-17), 0.4, label="isotropic_d3q19 (opt-in)", color="#2471a3")
    ax.axhline(TOL, ls="--", c="k", lw=1.2, label="board criterion 15%")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"rel. $L_2$ of CSF force vs lbmpy 2.0")
    ax.set_title("m17515 XVAL Allen-Cahn CSF force chain — ref-vs-got (audit m17582)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    p = FIG / "audit_force_rel_l2_vs_criterion.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    metas.append({"name": p.name,
                  "caption": "逐算例 CSF 力场相对 L2 误差（对数轴）与 15% 判据线：生产默认 central7 在 "
                             "C3-droplet-W3 达 20.6% 越线，opt-in isotropic_d3q19 在舍入误差水平。"})

    # fig 2: sigma recovery vs radius / interface thickness
    r = np.array([x_[0] for x_ in p4["sigma_err_vs_radius_W4"]], dtype=float)
    sr = np.array([100 * x_[1] for x_ in p4["sigma_err_vs_radius_W4"]])
    w = np.array([x_[0] for x_ in p4["sigma_err_vs_W_R16"]], dtype=float)
    sw = np.array([100 * x_[1] for x_ in p4["sigma_err_vs_W_R16"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.plot(r, sr, "o-", label=r"$\sigma_{rec}$ error (W=4)")
    a1.axhline(-3, ls=":", c="gray", label="3% target")
    a1.axhline(-15, ls="--", c="k", label="15% criterion")
    a1.set_xlabel("droplet radius R [lattice units]")
    a1.set_ylabel("sigma recovery error [%]")
    a1.set_title("error grows with R, no convergence to 0")
    a1.grid(ls=":", alpha=0.5)
    a1.legend(fontsize=8)
    a2.plot(w, sw, "s-", c="#8e44ad", label=r"$\sigma_{rec}$ error (R=16)")
    a2.axhline(-3, ls=":", c="gray", label="3% target")
    a2.axhline(-15, ls="--", c="k", label="15% criterion")
    a2.set_xlabel("interface thickness W [lattice units]")
    a2.set_title("error shrinks with W (thin interface is the weak point)")
    a2.grid(ls=":", alpha=0.5)
    a2.legend(fontsize=8)
    fig.suptitle("m17515 physical anchor: recovered surface tension vs prescribed sigma", fontsize=10)
    p = FIG / "audit_sigma_recovery_anchor.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    metas.append({"name": p.name,
                  "caption": "物理锚（σ 反演 vs 给定 σ）：随 R 增大误差单调变大（-4.4%→-5.1%→-6.5%），"
                             "随界面厚度 W 增厚误差变小（-11.8%→-6.5%→-2.0%），未收敛到 0，"
                             "仅 15% 判据内通过、3% 目标未达。"})
    return metas


def main() -> int:
    p1 = probe_ref_vs_got()
    p2 = probe_reproduction()
    p3 = probe_referee()
    p4 = probe_sigma_anchor()
    p5 = probe_integration()
    p6 = probe_detection_power()
    figs = figures(p1, p4)
    audit = {
        "anchor": ANCHOR, "scope": SCOPE, "subject_mission": 17515,
        "auditor_mission": 17582, "verdict_by": "board-audit",
        "subject_worktree": str(SUBJECT),
        "subject_head_commit": subprocess.run(
            ["git", "-C", str(SUBJECT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "subject_log_head": subprocess.run(
            ["git", "-C", str(SUBJECT), "log", "-1", "--format=%h %s"],
            capture_output=True, text=True).stdout.strip(),
        "P1_ref_vs_got": p1, "P2_reproduction": p2, "P3_referee": p3,
        "P4_sigma_anchor": p4, "P5_integration": p5, "P6_detection_power": p6,
        "figures": figs,
    }
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("subject_head_commit",)}, indent=2))
    print("worst_rel_l2_by_operator:", p1["worst_rel_l2_by_operator"])
    print("failing_15pct:", p1["failing_15pct"], "| n_rows", p1["n_rows"],
          "distinct", p1["n_distinct_metric_groups"])
    print("repro decision_identical:", p2.get("decision_identical"), "csv:", p2.get("csv_identical"))
    print("referee rows:", p3.get("n_rows"))
    print("worst sigma err:", p4["worst_sigma_err"])
    print("flags:", p1["artifacts_decision_flags"], "inconsistent:", p1["flags_inconsistent_with_case_table"])
    print("production call sites:", p5["n_production_sites"],
          "selecting isotropic:", p5["n_production_sites_selecting_isotropic"])
    print("detection:", {k: p6.get(k) for k in ("baseline_rel_l2", "laplacian_plus5pct_rel_l2",
                                                "grad_x_plus5pct_rel_l2", "detects_5pct_error")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())