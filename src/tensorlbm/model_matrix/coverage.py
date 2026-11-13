"""Model x case coverage driver: the CI gate and its evidence.

The driver answers three questions on every run:

1. *Coverage* -- which (model, case) pairs the registry declares, which of them
   actually executed, which are declared inapplicable and which are holes
   (``MISSING``).  This is the analogue of the per-model ``gcov`` line
   coverage of the upstream CI: a model that is compiled but not exercised is
   not covered.
2. *Correctness* -- for every executed gate cell, the reference value and the
   got value with the signed relative error, so the build can fail on physics
   instead of on presence of files.
3. *Guard strength* -- models that only need to prove they are constructible
   (LES-enabled operators whose sub-grid viscosity has no analytic
   molecular-viscosity reference) are recorded as ``GUARD`` and are counted
   separately, never mixed with gate passes.

Everything written by :func:`run_matrix` is machine-generated from the run
itself (no hand-typed numbers): ``model_matrix.json`` is the raw record,
``COVERAGE.md`` / ``BENCHMARK_REF_VS_GOT.md`` / ``matrix.csv`` are views of it.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .cases import CASES, CaseResult, Grid, case_names, run_case
from .registry import ModelSpec, get_model, model_names, registry

PASS_LIKE = {"PASS", "GUARD"}
FAIL_LIKE = {"FAIL", "ERROR"}
MARK = {
    "PASS": "PASS",
    "FAIL": "**FAIL**",
    "GUARD": "guard",
    "NOT_APPLICABLE": "n/a",
    "MISSING": "MISSING",
    "ERROR": "**ERROR**",
}


def _git_rev(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def guard_one(name: str, *, dtype=torch.float64) -> dict:
    """Construct the collider and run one step on a tiny box (compile guard)."""
    spec = get_model(name)
    grid = Grid(nx=8, ny=8, nz=1)
    from .cases import _shear_field  # noqa: PLC0415

    try:
        f = _shear_field(spec, grid, 0.01, dtype)
        collider = __import__(
            "tensorlbm.model_matrix.registry", fromlist=["build_collider"]
        ).build_collider(spec.name, 0.6)
        f2 = collider(f)
        finite = bool(torch.isfinite(f2).all())
        return {
            "model": name,
            "status": "GUARD_PASS" if finite else "GUARD_FAIL",
            "finite": finite,
            "source": spec.source,
            "reason": "" if finite else "non-finite distribution after one collision",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model": name,
            "status": "GUARD_FAIL",
            "finite": False,
            "source": spec.source,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def run_matrix(
    models: Sequence[str] | None = None,
    cases: Sequence[str] | None = None,
    *,
    grid: Grid | None = None,
    taus: Sequence[float] = (0.6, 0.8),
    dtype=torch.float64,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every declared (model, case) pair at every tau."""
    grid = grid or Grid()
    chosen_models = list(models) if models else model_names()
    chosen_cases = list(cases) if cases else case_names()
    results: list[CaseResult] = []
    for name in chosen_models:
        spec = get_model(name)
        for tau in taus:
            for case in chosen_cases:
                res = run_case(case, spec, grid, tau, dtype=dtype)
                # tau is part of the record so a cell can be traced to the
                # relaxation time it was produced at.
                res = dataclasses.replace(res, metrics=dict(res.metrics, tau=float(tau)))
                results.append(res)
                if progress:
                    print(
                        f"[matrix] {name:22s} tau={tau:.2f} {case:20s} -> {res.status}",
                        file=sys.stderr,
                    )
    return {
        "results": [r.as_dict() for r in results],
        "models": chosen_models,
        "cases": chosen_cases,
        "taus": list(taus),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz},
    }


def summarize(run: dict[str, Any], *, min_coverage: float = 0.85) -> dict[str, Any]:
    """Coverage accounting + gate verdict over a raw run record."""
    declared: list[dict[str, str]] = []
    for name in run["models"]:
        spec = get_model(name)
        for case in run["cases"]:
            if case in spec.inapplicable_reason:
                declared.append({"model": name, "case": case, "declared": "not_applicable"})
            elif case in spec.cases:
                declared.append({"model": name, "case": case, "declared": "declared"})
            else:
                declared.append({"model": name, "case": case, "declared": "undeclared"})

    by_key = {(r["model"], r["case"], float(r["metrics"].get("tau", 0.0))): r for r in run["results"]}

    covered = 0
    gate_cells: list[dict[str, Any]] = []
    for d in declared:
        if d["declared"] == "not_applicable":
            d["status"] = "NOT_APPLICABLE"
            continue
        statuses = [
            r["status"]
            for (m, c, _t), r in by_key.items()
            if m == d["model"] and c == d["case"]
        ]
        if d["declared"] == "undeclared":
            d["status"] = "MISSING"
            continue
        if not statuses:
            d["status"] = "MISSING"
            continue
        guard_only = get_model(d["model"]).enforcement == "guard"
        if any(s in FAIL_LIKE for s in statuses):
            d["status"] = "FAIL"
        elif guard_only or all(s == "GUARD" for s in statuses):
            # construct-only models never claim a gate-level pass
            d["status"] = "GUARD"
        else:
            d["status"] = "PASS"
        covered += 1
        gate_cells.append(d)

    declared_run = [d for d in declared if d["declared"] != "not_applicable"]
    n_declared = len(declared_run)
    coverage = covered / n_declared if n_declared else 0.0
    gate_failures = [d for d in gate_cells if d["status"] == "FAIL"]
    gate_models = {n for n in run["models"] if get_model(n).enforcement == "gate"}
    gate_model_cells = [d for d in gate_cells if d["model"] in gate_models]
    gate_model_pass = [d for d in gate_model_cells if d["status"] == "PASS"]
    advisory = [
        r
        for r in run["results"]
        if CASES[r["case"]].enforcement == "advisory"
    ]
    advisory_deviations = [
        {
            "model": r["model"],
            "case": r["case"],
            "tau": r["metrics"].get("tau"),
            "nu_rel_error_pct": r["metrics"].get("nu_rel_error_pct"),
            "nu_fit": r["metrics"].get("nu_fit"),
            "nu_ref": r["metrics"].get("nu_ref"),
            "u0": r["metrics"].get("u0"),
            "mach": r["metrics"].get("mach"),
            "energy_series_rel_l2_pct": r["metrics"].get("energy_series_rel_l2_pct"),
            "status": r["status"],
        }
        for r in advisory
    ]
    ok = (
        not gate_failures
        and coverage >= min_coverage
        and all(
            r["status"] not in FAIL_LIKE
            for r in run["results"]
            if CASES[r["case"]].enforcement == "gate" and get_model(r["model"]).enforcement == "gate"
        )
    )
    return {
        "coverage_fraction": coverage,
        "cells_declared": n_declared,
        "cells_covered": covered,
        "cells_not_applicable": len(declared) - n_declared,
        "missing_cells": [f"{d['model']}::{d['case']}" for d in declared if d["status"] == "MISSING"],
        "gate_model_pass_rate": (
            len(gate_model_pass) / len(gate_model_cells) if gate_model_cells else 0.0
        ),
        "gate_cells": gate_cells,
        "gate_failures": [f"{d['model']}::{d['case']}" for d in gate_failures],
        "advisory": advisory_deviations,
        "min_coverage_required": min_coverage,
        "verdict": "PASS" if ok else "FAIL",
        "cells": declared,
    }


# --------------------------------------------------------------------------- #
#  Report writers (all numbers derived from the run record)
# --------------------------------------------------------------------------- #


def write_json(run: dict[str, Any], summary: dict[str, Any], out: Path, **meta: Any) -> Path:
    payload = {
        "schema": "tensorlbm.model_matrix/1",
        "generated_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
        "run": run,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return out


def write_csv(run: dict[str, Any], out: Path) -> Path:
    rows = []
    for r in run["results"]:
        rows.append(
            {
                "model": r["model"],
                "case": r["case"],
                "tau": r["metrics"].get("tau", ""),
                "status": r["status"],
                **{k: v for k, v in r["metrics"].items() if isinstance(v, (int, float))},
            }
        )
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    return out


def write_coverage_md(run: dict[str, Any], summary: dict[str, Any], out: Path) -> Path:
    cases = run["cases"]
    taus = run["taus"]
    lines: list[str] = [
        "# Model matrix CI coverage (machine-generated)",
        "",
        "Model-as-matrix-key coverage report. One row per registry entry, one",
        "column per analytic case. Cells are the status of that (model, case)",
        "pair at every requested relaxation time.",
        "",
        "Legend: `PASS` gate case passed · `**FAIL**` gate case outside tolerance ·",
        "`guard` constructible-only model (LES: no molecular-ν reference) ·",
        "`n/a` declared inapplicable · `MISSING` coverage hole.",
        "",
        f"- coverage: **{summary['coverage_fraction'] * 100:.1f}%** "
        f"({summary['cells_covered']}/{summary['cells_declared']} declared cells executed)",
        f"- gate-model pass rate: {summary['gate_model_pass_rate'] * 100:.1f}%",
        f"- gate failures: {summary['gate_failures'] or 'none'}",
        f"- missing cells: {summary['missing_cells'] or 'none'}",
        f"- verdict: **{summary['verdict']}** (min coverage {summary['min_coverage_required']:.0%})",
        "",
        "## Matrix",
        "",
        "| model | " + " | ".join(cases) + " |",
        "|---|" + "|".join(["---"] * len(cases)) + "|",
    ]
    cell = {(c["model"], c["case"]): c for c in summary["cells"]}
    for name in run["models"]:
        spec = get_model(name)
        row = [f"`{name}` ({spec.enforcement})"]
        for case in cases:
            st = cell.get((name, case), {}).get("status", "MISSING")
            row.append(MARK.get(st, st))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Registry provenance",
        "",
        "| model | lattice | implementation | tau arg | params | enforcement |",
        "|---|---|---|---|---|---|",
    ]
    for name in run["models"]:
        spec = get_model(name)
        lines.append(
            f"| `{name}` | {spec.lattice} | `{spec.source}` | `{spec.tau_arg}` | "
            f"`{json.dumps(spec.params)}` | {spec.enforcement} |"
        )
    lines += [
        "",
        "## Case definitions (reference is a formula, not a stored table)",
        "",
        "| case | reference | enforcement | tolerance | metrics |",
        "|---|---|---|---|---|",
    ]
    for case in cases:
        cs = CASES[case]
        tol = f"{cs.tol_pct:.0f}%" if cs.tol_pct else "round-off"
        desc = cs.description.replace("|", "\\|")
        mdoc = cs.metrics_doc.replace("|", "\\|")
        lines.append(f"| `{case}` | {desc} | {cs.enforcement} | {tol} | {mdoc} |")
    lines += ["", f"Taus swept: {', '.join(f'{t:.2f}' for t in taus)}", ""]
    out.write_text("\n".join(lines) + "\n")
    return out


def write_benchmark_md(run: dict[str, Any], summary: dict[str, Any], out: Path) -> Path:
    lines = [
        "# Model matrix: reference vs got (machine-generated)",
        "",
        "Each row is a gate cell: the reference comes from the closed-form",
        "solution of the case (never a stored result), the got value is measured",
        "from the run recorded in `model_matrix.json`. Error = signed relative",
        "deviation of the fitted quantity from its analytic value.",
        "",
        "| model | case | tau | quantity | ref | got | error % | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in run["results"]:
        if CASES[r["case"]].enforcement != "gate":
            continue
        if r["status"] == "NOT_APPLICABLE":
            continue
        m = r["metrics"]
        tau = m.get("tau", "")
        if r["case"] == "feq_fixed_point":
            ref = 0.0
            got = max(
                m.get("rel_feq_fixed_point_error", float("nan")),
                m.get("max_abs_rho_error", float("nan")),
                m.get("max_abs_velocity_error", float("nan")),
            )
            err = float("nan")
            qty = "max rel. deviation from the f_eq fixed point (and rho,u conservation)"
            ref_s, got_s = "0 (exact)", f"{got:.3e}"
        else:
            ref = m.get("nu_ref", float("nan"))
            got = m.get("nu_fit", float("nan"))
            err = m.get("nu_rel_error_pct", float("nan"))
            qty = "shear viscosity nu"
            ref_s, got_s = f"{ref:.5f}", f"{got:.5f}"
        verdict = "OK" if r["status"] == "PASS" else ("X" if r["status"] in FAIL_LIKE else r["status"])
        err_s = "-" if not math.isfinite(err) else f"{err:+.2f}"
        lines.append(
            f"| `{r['model']}` | {r['case']} | {tau} | {qty} | {ref_s} | {got_s} | {err_s} | {verdict} |"
        )
    lines += [
        "",
        "## Advisory cells (finite-Mach probe; recorded, not gated)",
        "",
        "| model | tau | u0 | Ma | nu_rel_error_pct | energy_series_rel_l2_pct |",
        "|---|---|---|---|---|---|",
    ]
    for a in summary["advisory"]:
        err = a["nu_rel_error_pct"]
        if a["status"] == "NOT_APPLICABLE":
            continue
        def _f(v, fmt="{:+.2f}"):
            return "n/a" if v is None or not math.isfinite(float(v)) else fmt.format(float(v))

        lines.append(
            f"| `{a['model']}` | {a['tau']} | {_f(a.get('u0'), '{:.3f}')} | "
            f"{_f(a.get('mach'), '{:.3f}')} | {_f(a['nu_rel_error_pct'])} | "
            f"{_f(a.get('energy_series_rel_l2_pct'))} |"
        )
    out.write_text("\n".join(lines) + "\n")
    return out


def make_figures(run: dict[str, Any], summary: dict[str, Any], outdir: Path) -> list[Path]:
    """Decay curves + per-model viscosity error bars + coverage heat-map."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # (a) decay curves for a representative subset
    gate = [r for r in run["results"] if r["case"] == "shear_decay" and r.get("curves")]
    taus = run["taus"]
    tau0 = taus[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    shown = 0
    for r in gate:
        if abs(r["metrics"].get("tau", -1) - tau0) > 1e-9:
            continue
        c = r["curves"]
        ax.plot(c["t"], c["amplitude"], lw=1.2, label=f"{r['model']} (measured)")
        shown += 1
    if shown:
        c0 = next(r for r in gate if abs(r["metrics"].get("tau", -1) - tau0) < 1e-9)["curves"]
        ax.plot(c0["t"], c0["amplitude_analytic"], "k--", lw=2.0, label="analytic exp(-nu k^2 t)")
    ax.set_xlabel("time step")
    ax.set_ylabel("shear-wave amplitude |A(t)|")
    ax.set_yscale("log")
    ax.set_title(f"Periodic shear-wave decay, tau={tau0:.2f}")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p = outdir / "fig_shear_decay_curves.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # (b) viscosity error per model, one bar group per tau
    fig, ax = plt.subplots(figsize=(8, 4.0))
    names = [n for n in run["models"] if get_model(n).enforcement == "gate"]
    width = 0.8 / max(1, len(taus))
    for i, tau in enumerate(taus):
        errs = []
        for n in names:
            vals = [
                r["metrics"].get("nu_rel_error_pct")
                for r in run["results"]
                if r["model"] == n
                and r["case"] == "shear_decay"
                and abs(r["metrics"].get("tau", -1) - tau) < 1e-9
            ]
            errs.append(vals[0] if vals else float("nan"))
        xs = [j + i * width for j in range(len(names))]
        ax.bar(xs, errs, width=width, label=f"tau={tau:.2f}")
    ax.axhline(3.0, color="r", ls="--", lw=1.0, label="3% acceptance")
    ax.axhline(-3.0, color="r", ls="--", lw=1.0)
    ax.set_xticks([j + width * (len(taus) - 1) / 2 for j in range(len(names))])
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("fitted nu relative error [%]")
    ax.set_title("Model matrix: analytic-viscosity error per collision model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig_nu_error_by_model.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # (c) coverage heat-map (statuses as integer codes)
    code = {"PASS": 4, "GUARD": 3, "NOT_APPLICABLE": 2, "MISSING": 1, "FAIL": 0, "ERROR": 0}
    cell = {(c["model"], c["case"]): c for c in summary["cells"]}
    grid = [[code.get(cell.get((n, c), {}).get("status", "MISSING"), 1) for c in run["cases"]] for n in run["models"]]
    fig, ax = plt.subplots(figsize=(1.4 * len(run["cases"]) + 3.2, 0.42 * len(run["models"]) + 1.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(range(len(run["cases"])))
    ax.set_xticklabels(run["cases"], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(run["models"])))
    ax.set_yticklabels(run["models"], fontsize=7)
    ax.set_title(
        f"Model x case coverage {summary['coverage_fraction'] * 100:.0f}% "
        f"({summary['cells_covered']}/{summary['cells_declared']})"
    )
    cb = fig.colorbar(im, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["FAIL", "MISSING", "n/a", "guard", "PASS"], fontsize=7)
    fig.tight_layout()
    p = outdir / "fig_coverage_matrix.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def environment_meta(repo_root: Path) -> dict[str, Any]:
    return {
        "git_rev": _git_rev(repo_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dtype": "float64",
    }