#!/usr/bin/env python3
"""Aggregate XVAL runs + run the OSS survey + emit figures/tables (zero hand-copied numbers).

Every number in summary.json / table.csv / the figures is computed here from
(a) the raw run JSONs in runs/ and (b) the live oss_inventory table + the
~/oss-cache checkout survey. Nothing is typed by hand.
"""
from __future__ import annotations

import json
import subprocess
import sys
import glob
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = HERE / "runs"
OUT = HERE / "artifacts"
PREDECL = json.loads((HERE / "predeclared.json").read_text())

REF = PREDECL["benchmark"]["reference_values"]
TOL_OWNER = PREDECL["predeclared_thresholds"]["acceptance_owner_standard_pct"]
TOL_CALIBER = PREDECL["predeclared_thresholds"]["platform_caliber_pct"]
GRID_SPAN_TOL = 2.0  # % span between the two finest completed grids


# ---------------------------------------------------------------- OSS survey
def oss_survey() -> dict:
    """Horizontal OSS survey: does any inventoried/cached repo cover thermal LBM DHC?"""
    keys = ["thermal", "temperature", "heat", "boussinesq", "convection", "natural"]
    sql = (
        "SELECT repo, status, COALESCE(methods,'') || ' | ' || COALESCE(summary,'') "
        "FROM oss_inventory"
    )
    inv_rows = []
    try:
        out = subprocess.run(
            ["psql", "-U", "postgres", "-d", "tensorplatform", "-At", "-F", "\t", "-c", sql],
            capture_output=True, text=True, timeout=60,
        )
        for line in out.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                inv_rows.append({"repo": parts[0], "status": parts[1], "text": parts[2]})
    except Exception as exc:  # noqa: BLE001
        inv_rows = [{"error": repr(exc)}]

    inv_hits = []
    for r in inv_rows:
        low = r.get("text", "").lower()
        hit = [k for k in keys if k in low]
        if hit:
            inv_hits.append({"repo": r["repo"], "status": r["status"], "matched": hit})

    cache_root = Path(os.path.expanduser("~/oss-cache"))
    cache = []
    if cache_root.is_dir():
        for repo in sorted(p for p in cache_root.iterdir() if p.is_dir()):
            src_files = [p for p in repo.rglob("*")
                         if p.is_file() and p.suffix in {".py", ".f90", ".F90", ".c", ".cpp", ".h", ".md", ".ipynb"}]
            hits = []
            for p in src_files:
                try:
                    txt = p.read_text(errors="ignore").lower()
                except Exception:  # noqa: BLE001
                    continue
                m = [k for k in keys if k in txt]
                if len(m) >= 2:  # two independent thermal keywords in one file
                    hits.append({"file": str(p.relative_to(cache_root)), "matched": m})
            cache.append({"repo": repo.name, "scanned_files": len(src_files), "multi_keyword_hits": hits[:10],
                          "n_hits": len(hits)})

    return {
        "question": "Is there an OSS reference implementation of double-distribution thermal LBM "
                    "natural convection (DHC cavity) inside oss_inventory or ~/oss-cache?",
        "oss_inventory_rows": len(inv_rows),
        "oss_inventory_keyword_hits": inv_hits,
        "oss_cache": cache,
        "verdict": "NO_OSS_REFERENCE" if not any(
            c["n_hits"] for c in cache
        ) else "PARTIAL_OR_WEAK_HINTS",
        "consequence": "external anchor must come from the published benchmark lineage "
                       "(de Vahl Davis 1983 Table 1), as predeclared in predeclared.json alternatives[B]",
    }


# ---------------------------------------------------------------- aggregation
def load_runs() -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(str(RUNS / "*.json"))):
        d = json.loads(Path(fp).read_text())
        d["_file"] = str(Path(fp).relative_to(REPO))
        n = d["nx"]
        nu = d["nu_grad2"]
        rows.append({
            "file": d["_file"],
            "nx": n,
            "ra": d["ra"],
            "tau": d["tau"],
            "steps_done": d["steps_done"],
            "elapsed_s": d["elapsed_s"],
            "stopped_early": d["stopped_early"],
            "last_resid_u": d["last_resid_u"],
            "last_resid_T": d["last_resid_T"],
            "nu": nu,
            "nu_left": d["nu_left_grad2"],
            "nu_right": d["nu_right_grad2"],
            "nu_sym_pct": 100.0 * abs(d["nu_left_grad2"] - d["nu_right_grad2"]) / nu if nu else None,
            "u_max_nd": d["u_max_nd"],
            "v_max_nd": d["v_max_nd"],
            "y_at_u_max": d["y_at_u_max_x"],
            "x_at_v_max": d["x_at_v_max_y"],
            "ma_lattice": d["ma_lattice"],
            "err_nu_pct": 100.0 * (nu - REF["nu_mean"]) / REF["nu_mean"],
            "abs_err_nu_pct": 100.0 * abs(nu - REF["nu_mean"]) / REF["nu_mean"],
            "err_u_pct": 100.0 * (d["u_max_nd"] - REF["u_max_nd"]) / REF["u_max_nd"],
            "err_v_pct": 100.0 * (d["v_max_nd"] - REF["v_max_nd"]) / REF["v_max_nd"],
            "converged": bool(d["stopped_early"]),
            "nu_history": d["nu_history"],
        })
    return rows


def gates(rows: list[dict]) -> dict:
    """Every gate recomputed from the raw rows."""
    t06 = [r for r in rows if abs(r["tau"] - 0.6) < 1e-9]
    t06.sort(key=lambda r: r["nx"])
    g = {}
    g["G1_runs_completed"] = {"pass": len(t06) >= 2, "detail": f"{len(t06)} grids at tau=0.6: {[r['nx'] for r in t06]}"}
    g["G2_owner_15pct"] = {
        "pass": bool(t06) and all(r["abs_err_nu_pct"] <= TOL_OWNER for r in t06),
        "detail": {r["nx"]: round(r["abs_err_nu_pct"], 3) for r in t06},
        "threshold_pct": TOL_OWNER,
    }
    g["G3_caliber_3pct"] = {
        "pass": bool(t06) and all(r["abs_err_nu_pct"] <= TOL_CALIBER for r in t06),
        "detail": {r["nx"]: round(r["abs_err_nu_pct"], 3) for r in t06},
        "threshold_pct": TOL_CALIBER,
    }
    g["G4_steadiness"] = {
        "pass": bool(t06) and all(r["converged"] and r["last_resid_u"] <= 1e-5 for r in t06),
        "detail": {r["nx"]: {"stopped_early": r["stopped_early"],
                             "resid_u": r["last_resid_u"],
                             "steps_done": r["steps_done"]} for r in t06},
    }
    if len(t06) >= 2:
        fine, coarse = t06[-1], t06[-2]
        span = 100.0 * abs(fine["nu"] - coarse["nu"]) / coarse["nu"]
        mono = abs(fine["err_nu_pct"]) <= abs(coarse["err_nu_pct"]) + 1e-12
        g["G5_grid_span"] = {"pass": span <= GRID_SPAN_TOL, "span_pct": round(span, 3),
                             "pairs": [coarse["nx"], fine["nx"]], "threshold_pct": GRID_SPAN_TOL}
        g["G6_monotone"] = {"pass": mono, "detail": {coarse["nx"]: round(coarse["err_nu_pct"], 3),
                                                    fine["nx"]: round(fine["err_nu_pct"], 3)}}
    else:
        g["G5_grid_span"] = {"pass": False, "detail": "fewer than two completed grids"}
        g["G6_monotone"] = {"pass": False, "detail": "fewer than two completed grids"}
    taus = sorted({r["tau"] for r in rows})
    g["G7_tau_sensitivity"] = {"pass": len(taus) >= 2, "tau_values": taus,
                               "detail": {r["nx"]: round(r["nu"], 4) for r in rows if len(taus) >= 2}}
    return g


def figures(rows: list[dict], g: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # fig1: Nu history (convergence of the observable itself)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    for r in rows:
        h = r["nu_history"]
        if not h:
            continue
        ax.plot([x["step"] for x in h], [x["nu"] for x in h],
                label=f"N={r['nx']} tau={r['tau']:g}")
    ax.axhline(REF["nu_mean"], color="k", ls="--", lw=1.2, label=f"de Vahl Davis 1983 Nu={REF['nu_mean']}")
    ax.set_xlabel("step"); ax.set_ylabel("Nu (wall heat-flux average)")
    ax.set_title("DHC Ra=1e5, Pr=0.71: Nu approach to steady state")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = OUT / "fig1_nu_history.png"; fig.tight_layout(); fig.savefig(p, optimize=True); plt.close(fig)
    made.append(str(p.relative_to(REPO)))

    # fig2: ref vs got bars with error labels
    t06 = sorted([r for r in rows if abs(r["tau"] - 0.6) < 1e-9], key=lambda r: r["nx"])
    labels = [f"N={r['nx']}" for r in t06]
    vals = [r["nu"] for r in t06]
    errs = [r["err_nu_pct"] for r in t06]
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=120)
    bars = ax.bar(labels + ["ref"], vals + [REF["nu_mean"]],
                  color=["#4c78a8"] * len(vals) + ["#b0b0b0"])
    for b, v, e in zip(bars[:-1], vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.3f}\n{e:+.2f}%", ha="center", fontsize=7)
    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2, REF["nu_mean"] + 0.05,
            f"{REF['nu_mean']:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Nu"); ax.set_ylim(0, max(vals + [REF["nu_mean"]]) * 1.25)
    ax.set_title("ref vs got (tau=0.6)"); ax.grid(axis="y", alpha=0.3)
    p = OUT / "fig2_ref_vs_got.png"; fig.tight_layout(); fig.savefig(p, optimize=True); plt.close(fig)
    made.append(str(p.relative_to(REPO)))

    # fig3: gate matrix
    names = list(g)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.2), dpi=120)
    for i, k in enumerate(names):
        ok = bool(g[k].get("pass"))
        ax.barh(i, 1, color="#54a24b" if ok else "#e45756")
        ax.text(0.02, i, ("PASS " if ok else "FAIL ") + k, va="center", fontsize=8, color="white")
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, 1)
    ax.set_title("XVAL gate matrix (recomputed from raw run JSONs)")
    p = OUT / "fig3_gate_matrix.png"; fig.tight_layout(); fig.savefig(p, optimize=True); plt.close(fig)
    made.append(str(p.relative_to(REPO)))
    return made


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    survey = oss_survey()
    (OUT / "oss_survey.json").write_text(json.dumps(survey, indent=2, ensure_ascii=False))

    runs_dir = RUNS
    rows = load_runs()
    g = gates(rows)
    figs = figures(rows, g)

    table = ["grid,tau,steps,elapsed_s,stopped_early,nu,nu_ref,err_nu_pct,u_max_nd,err_u_pct,v_max_nd,err_v_pct,resid_u"]
    for r in sorted(rows, key=lambda r: (r["tau"], r["nx"])):
        table.append(",".join(str(x) for x in [
            r["nx"], r["tau"], r["steps_done"], r["elapsed_s"], r["stopped_early"],
            round(r["nu"], 5), REF["nu_mean"], round(r["err_nu_pct"], 3),
            round(r["u_max_nd"], 4), round(r["err_u_pct"], 3),
            round(r["v_max_nd"], 4), round(r["err_v_pct"], 3), r["last_resid_u"],
        ]))
    (OUT / "table.csv").write_text("\n".join(table) + "\n")

    summary = {
        "anchor": PREDECL["anchor"],
        "reference": REF,
        "reference_source": PREDECL["benchmark"]["reference_source"],
        "oss_survey_verdict": survey["verdict"],
        "runs": [{k: v for k, v in r.items() if k != "nu_history"} for r in rows],
        "gates": g,
        "all_gates_pass": all(bool(v.get("pass")) for v in g.values()),
        "figures": figs,
        "n_runs": len(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"runs": summary["n_runs"], "gates": {k: v.get("pass") for k, v in g.items()},
                      "nus": {r["nx"]: round(r["nu"], 4) for r in rows}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())