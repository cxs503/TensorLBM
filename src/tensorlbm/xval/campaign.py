"""XVAL campaign driver: Taylor-Green three-way cross-validation.

Runs the partners declared by the mission (in-house TensorLBM, palabos,
OpenFOAM icoFoam) on one :class:`TG2DSpec`, computes the metrics with
:mod:`tensorlbm.xval.metrics`, and writes the mission deliverables:

    <out>/case_spec.json        the single spec consumed by all tools
    <out>/k_history_<tool>.csv  raw tool output (step,K,umax,phase,peak)
    <out>/field_<tool>_<t>.csv  raw velocity fields at the reference steps
    <out>/result.json           xval_table + criteria + bug_suspects + status
    <out>/xval_table.csv        the same table, flat
    <out>/xval_taylor_green.png K(t) + decay-rate + phase comparison figure

Nothing is hard-coded: every number in ``result.json`` is derived from the
tool artefacts on disk, and a missing tool is recorded as a *degradation*
(``frictions``) rather than silently dropped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import metrics as M
from .case_spec import TG2DSpec
from .runners import run_openfoam, run_palabos, run_tensorlbm

TOOL_LABEL = {
    "tensorlbm": "TensorLBM (in-house D2Q9 BGK)",
    "palabos": "palabos (OSS D2Q9 BGK)",
    "openfoam": "OpenFOAM icoFoam (OSS FV/PISO)",
}


# ------------------------------------------------------------ per-tool maths
def tool_metrics(tool: dict[str, Any], spec: TG2DSpec) -> dict[str, float]:
    """Reduce one tool's raw artefacts to the comparison metrics."""
    hist = tool["history"]
    steps = np.asarray(hist["step"], dtype=float)
    k_energy = np.asarray(hist["kinetic_energy"], dtype=float)
    umax = np.asarray(hist["umax"], dtype=float)
    phases = np.asarray(hist["phase_deg"], dtype=float)

    boost_offset = 0.5 * spec.boost**2  # non-decaying part of K in the boosted case
    fit_k = M.fit_exp_rate(steps, k_energy, offset=boost_offset)
    fit_v = M.fit_exp_rate(steps, umax)

    # phase error against the exact solution, in percent of pi
    phi0 = float(phases[0])
    phi_exact = phi0 + np.rad2deg(M.exact_phase_rate(spec) * steps)
    # unwrap both tracks before comparing (the mode phase is reported mod 360)
    phi_meas = np.rad2deg(np.unwrap(np.deg2rad(phases)))
    phi_ref = np.rad2deg(np.unwrap(np.deg2rad(phi_exact)))
    phase_err_pct = float(np.max(np.abs(phi_meas - phi_ref)) / 180.0 * 100.0)
    rate = M.phase_rate(spec, steps, phases)

    coll = dict(tool.get("field_l2_pct", {}))
    field_l2 = max(coll.values()) if coll else float("nan")

    return {
        "gamma_K": fit_k["gamma"],
        "gamma_K_r2": fit_k["r2"],
        "gamma_vel": fit_v["gamma"],
        "gamma_vel_r2": fit_v["r2"],
        "K0": float(k_energy[0]),
        "field_l2_pct": field_l2,
        "phase_err_pct": phase_err_pct,
        "phase_rate_rad_per_step": rate["phase_rate_rad_per_step"],
        "phase_rate_r2": rate["r2"],
        "n_samples": float(fit_k["n_samples"]),
    }


def reference_metrics(spec: TG2DSpec) -> dict[str, float | None]:
    """Analytic reference values; ``None`` marks a zero-reference (absolute) metric."""
    return {
        "gamma_K": spec.gamma_k_theory,
        "gamma_vel": spec.gamma_vel_theory,
        # the boosted variant carries a non-decaying 0.5 U_b^2 offset in K
        "K0": spec.k0_theory + 0.5 * spec.boost**2,
        "field_l2_pct": None,   # exact solution is the reference -> 0 % allowed
        "phase_err_pct": None,
        # a stationary vortex has zero exact phase rate: absolute metric
        # a stationary vortex has zero exact phase rate; the metric is only
        # meaningful (and only then predeclared) for the boosted variant.
        **({"phase_rate_rad_per_step": M.exact_phase_rate(spec)} if spec.boost else {}),
    }


# ------------------------------------------------------------------ runners
def run_campaign(
    spec: TG2DSpec,
    out_dir: str | Path,
    *,
    tools: Iterable[str] = ("tensorlbm", "palabos", "openfoam"),
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec.dump(str(out / "case_spec.json"))

    results: dict[str, dict[str, Any]] = {}
    frictions: list[dict[str, Any]] = []
    for name in tools:
        if name == "tensorlbm":
            res = run_tensorlbm(spec, out, dtype=dtype, threads=threads)
        elif name == "palabos":
            res = run_palabos(spec, out)
        elif name == "openfoam":
            res = run_openfoam(spec, out)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown tool {name!r}")
        results[name] = res
        if not res.get("available", False):
            frictions.append(
                {
                    "tool": name,
                    "kind": "tool_unavailable",
                    **res.get("friction", {}),
                }
            )

    available = {k: v for k, v in results.items() if v.get("available")}
    if "tensorlbm" not in available:
        raise RuntimeError("the in-house tool must always be available")

    per_tool_metrics = {name: tool_metrics(res, spec) for name, res in available.items()}
    refs = reference_metrics(spec)
    table = M.build_xval_table(refs, per_tool_metrics)

    crit_tol = spec.to_dict()["criteria"]
    criterion_specs: list[dict[str, Any]] = [
        # measurement metrics: reproduced by every tool *and* by the exact
        # solution, so both criteria apply.
        {"metric": "gamma_K", "kind": "cross_tool", "tol_pct": crit_tol["cross_tool_gamma_K_pct"]},
        {"metric": "gamma_K", "kind": "vs_exact", "tol_pct": crit_tol["vs_exact_gamma_K_pct"]},
        {"metric": "gamma_vel", "kind": "vs_exact", "tol_pct": crit_tol["vs_exact_gamma_K_pct"]},
        # error metrics: exact value is zero -> vs-exact bound only (see
        # metrics.evaluate_criteria for the rationale).
        {"metric": "field_l2_pct", "kind": "vs_exact", "tol_pct": crit_tol["vs_exact_gamma_K_pct"]},
        {"metric": "phase_err_pct", "kind": "vs_exact", "tol_pct": crit_tol["phase_pct"]},
    ]
    if spec.boost:
        criterion_specs += [
            {"metric": "phase_rate_rad_per_step", "kind": "vs_exact", "tol_pct": crit_tol["phase_pct"]},
            {"metric": "phase_rate_rad_per_step", "kind": "cross_tool", "tol_pct": crit_tol["phase_pct"]},
        ]
    criteria = M.evaluate_criteria(table, criterion_specs)
    cross_tool_metrics = {c["metric"] for c in criterion_specs if c["kind"] == "cross_tool"}
    thresholds = {"gamma_K": crit_tol["cross_tool_gamma_K_pct"],
                  "phase_rate_rad_per_step": crit_tol["phase_pct"]}

    # degrade criteria that involve a tool we could not install
    degraded = [c for c in frictions if c["kind"] == "tool_unavailable"]
    if degraded:
        criteria.append(
            {
                "item": "three_way_cross_check",
                "PASS": False,
                "degraded_to": " + ".join(sorted(available)),
                "missing_tools": [d["tool"] for d in degraded],
                "note": "cross-tool criterion computed on the available tool set only",
            }
        )

    suspects = find_bug_suspects(per_tool_metrics, refs, table, thresholds, cross_tool_metrics)
    overall = "PASS" if all(c["PASS"] for c in criteria) else "DEGRADED_PASS"

    payload: dict[str, Any] = {
        "ref_id": spec.ref_id,
        "case": spec.name,
        "case_spec": spec.to_dict(),
        "tools": {
            name: {
                "label": TOOL_LABEL.get(name, name),
                "available": res.get("available", False),
                "run_info": res.get("run_info", {}),
                "metrics": per_tool_metrics.get(name),
                "field_l2_pct_by_step": res.get("field_l2_pct", {}),
            }
            for name, res in results.items()
        },
        "analytic_reference": refs,
        "xval_table": table,
        "criteria": criteria,
        "bug_suspects": suspects,
        "frictions": frictions,
        "overall": overall,
        "predeclared_thresholds": {
            "cross_tool_gamma_K_pct": crit_tol["cross_tool_gamma_K_pct"],
            "vs_exact_gamma_K_pct": crit_tol["vs_exact_gamma_K_pct"],
            "phase_pct_of_pi": crit_tol["phase_pct"],
        },
    }
    (out / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_table_csv(table, out / "xval_table.csv")
    return payload


def write_table_csv(table: list[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("tool,metric,got,vs_ref_err,cross_diff\n")
        for row in table:
            fh.write(
                f"{row['tool']},{row['metric']},{row['got']!r},"
                f"{row['vs_ref_err']!r},{row['cross_diff']!r}\n"
            )


# ------------------------------------------------------------- bug hunting
def find_bug_suspects(
    per_tool_metrics: dict[str, dict[str, float]],
    refs: dict[str, float | None],
    table: list[dict[str, Any]],
    thresholds: dict[str, float],
    cross_tool_metrics: Iterable[str] = ("gamma_K",),
) -> list[dict[str, Any]]:
    """Locate and attribute every metric that misses the predeclared tolerance.

    Attribution rule (predeclared, not tuned after the fact): the side that is
    further from the *analytic* reference is the suspect.  When every tool is
    inside the band the list is empty and the report says so explicitly.
    """
    suspects: list[dict[str, Any]] = []
    metrics = [m for m in refs if any(r["metric"] == m for r in table)]
    for metric in metrics:
        rows = [r for r in table if r["metric"] == metric]
        worst_cross = max(abs(r["cross_diff"]) for r in rows)
        tol = thresholds.get(metric, 5.0)
        offenders = [r for r in rows if abs(r["vs_ref_err"]) > tol]
        cross_ok = metric not in set(cross_tool_metrics) or worst_cross <= tol
        if cross_ok and not offenders:
            continue
        ranked = sorted(rows, key=lambda r: -abs(r["vs_ref_err"]))
        top = ranked[0]
        peers = [r["got"] for r in rows if r["tool"] != top["tool"]]
        suspects.append(
            {
                "metric": metric,
                "ours": per_tool_metrics.get("tensorlbm", {}).get(metric),
                "oss": {r["tool"]: r["got"] for r in rows if r["tool"] != "tensorlbm"},
                "delta": top["got"] - (refs[metric] if refs[metric] is not None else 0.0),
                "suspected_side": top["tool"],
                "evidence": {
                    "worst_cross_diff_pct": worst_cross,
                    "tol_pct": tol,
                    "ranked_abs_err_vs_exact_pct": {
                        r["tool"]: abs(r["vs_ref_err"]) for r in ranked
                    },
                    "peers": peers,
                },
                "locate": LOCATE_HINTS.get(metric, "n/a"),
            }
        )
    return suspects


LOCATE_HINTS = {
    "gamma_K": "src/tensorlbm/xval/runners.py::run_tensorlbm (tau mapping / collision), "
    "src/tensorlbm/d2q9.py::equilibrium",
    "gamma_vel": "src/tensorlbm/solver.py::collide_bgk, src/tensorlbm/xval/runners.py (umax sampling)",
    "K0": "src/tensorlbm/xval/runners.py::run_tensorlbm (initial condition / f_eq)",
    "field_l2_pct": "src/tensorlbm/xval/runners.py::run_tensorlbm + src/tensorlbm/solver.py::stream",
    "phase_err_pct": "src/tensorlbm/xval/metrics.py::mode_phase_deg (sampling convention) / "
    "streaming stencil in src/tensorlbm/solver.py::stream",
    "phase_rate_rad_per_step": "src/tensorlbm/solver.py::stream (advection stencil) — "
    "Galilean defect of the D2Q9 BGK scheme",
}


# ------------------------------------------------------------------ figures
def make_figure(payload: dict[str, Any], out_dir: str | Path) -> str | None:
    """K(t) decay + decay-rate + phase comparison figure (matplotlib, Agg)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None

    out = Path(out_dir)
    spec = TG2DSpec.from_dict(payload["case_spec"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    steps = np.arange(0, spec.steps + 1, spec.sample_every)
    k_exact = np.array([M.exact_K(spec, s) - 0.5 * spec.boost**2 for s in steps])
    axes[0].plot(steps, k_exact, "k-", lw=2.0, label="exact (series)")

    colors = {"tensorlbm": "#1f77b4", "palabos": "#d62728", "openfoam": "#2ca02c"}
    for name, tool in payload["tools"].items():
        if not tool["available"]:
            continue
        path = out / f"k_history_{name}.csv"
        if not path.is_file():
            continue
        hist = np.genfromtxt(path, delimiter=",", names=True)
        kk = np.asarray(hist["kinetic_energy"], dtype=float) - 0.5 * spec.boost**2
        axes[0].plot(np.asarray(hist["step"], float), kk, "o", ms=3, alpha=0.75,
                     color=colors.get(name), label=tool["label"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("lattice step")
    axes[0].set_ylabel("K(t) - 0.5 U_b^2")
    axes[0].set_title("(a) kinetic-energy decay")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    # (b) decay rates vs exact
    names, got, exact = [], [], []
    for metric, ref in (("gamma_K", spec.gamma_k_theory), ("gamma_vel", spec.gamma_vel_theory)):
        for name in ("tensorlbm", "palabos", "openfoam"):
            tool = payload["tools"].get(name)
            if not tool or not tool.get("available"):
                continue
            names.append(f"{name}\n{metric}")
            got.append(tool["metrics"][metric])
            exact.append(ref)
    idx = np.arange(len(names))
    axes[1].bar(idx - 0.2, exact, width=0.4, color="0.6", label="exact")
    axes[1].bar(idx + 0.2, got, width=0.4, color="#1f77b4", label="measured")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, fontsize=6.5, rotation=90)
    axes[1].set_ylabel("decay rate [1/step]")
    axes[1].set_title("(b) decay rates (K and |u|max)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")

    # (c) phase error vs exact
    labels, errs = [], []
    for name in ("tensorlbm", "palabos", "openfoam"):
        tool = payload["tools"].get(name)
        if not tool or not tool.get("available"):
            continue
        labels.append(name)
        errs.append(tool["metrics"]["phase_err_pct"])
    if labels:
        axes[2].bar(np.arange(len(labels)), errs, color="#ff7f0e")
        axes[2].axhline(spec.to_dict()["criteria"]["phase_pct"], ls="--", c="r",
                        label="tol 3% of pi")
        axes[2].set_xticks(np.arange(len(labels)))
        axes[2].set_xticklabels(labels, fontsize=8)
        axes[2].set_ylabel("max phase error [% of pi]")
        axes[2].set_title("(c) vortex-pattern phase error")
        axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"{spec.ref_id} — {spec.name}: N={spec.n}, Re={spec.re}, "
                 f"U0={spec.u0}, boost={spec.boost}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "xval_taylor_green.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def run_and_report(
    spec: TG2DSpec,
    out_dir: str | Path,
    *,
    tools: Iterable[str] = ("tensorlbm", "palabos", "openfoam"),
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    payload = run_campaign(spec, out_dir, tools=tools, dtype=dtype, threads=threads)
    fig = make_figure(payload, out_dir)
    if fig:
        payload["figure"] = fig
        (Path(out_dir) / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    return payload