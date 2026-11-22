"""Reusable reference-vs-computed benchmark for two-component SC layered Poiseuille.

This module is the *shared* scoring layer of the anchor
``XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM``.  It does not contain any
physics of its own: it drives :func:`.run_two_component_sc_validation.run_case_spec`
(which owns the solver + the exact layered reference) and reshapes the result
into an explicit ``ref`` / ``got`` / ``err`` / ``threshold`` / ``pass`` table so
that the numbers can be quoted, machine-checked and plotted without hand
transcription.

Callers:

* ``python -m tensorlbm.verification.layered_poiseuille.benchmark`` -- CLI entry
  that writes ``benchmark.json`` / ``benchmark.csv`` / ``benchmark.md``;
* ``tools/paper_benchmarks/run_all_benchmarks.py`` --
  ``run_two_component_sc_layered_poiseuille()`` is registered in the paper
  evidence harness next to the single-component layered-Poiseuille benchmark;
* ``tensorlbm.verification.xval.cases`` -- the case is registered in the
  cross-validation registry (``XValRegistry``).

Gates are the pre-registered ones from the validation entry point: a control
run (one component, ``G = 0``) must reproduce the exact single-fluid
Poiseuille profile to 3 %; a layered run must stay inside the owner standard of
15 % on peak / flux / ``L2`` / ``Linf`` and 5 % of ``H`` on the interface
position; a static run (``g = 0``) must keep spurious currents below 15 % of the
driven peak.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .run_two_component_sc_validation import (CASE_MATRIX, GATE_OWNER,
                                             GATE_TARGET, run_case_spec)

#: cheap-but-physical matrix used by the CLI entry and by the callers below.
#: ``H = 22`` fluid cells, 3000 steps: same physics as the ``core`` matrix
#: (demixing, viscosity contrast, wall bounce-back) at ~4x lower cost.
LIGHT_CASES: Dict[str, Dict[str, Any]] = {
    "control_single_light": dict(
        kind="control",
        cfg=dict(ny=24, nx=1, tau_a=0.8, tau_b=0.8, G=0.0, g_x=1e-4, g_y=0.0, steps=3000,
                 init_rho_a=2.0, init_rho_b=0.0, init_layer_frac=1.0, interface_width=1.0),
        note="light control: one component, nu=0.1, exact single-fluid Poiseuille",
    ),
    "layered_r167_light": dict(
        kind="layered",
        cfg=dict(ny=24, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=5e-5, g_y=0.0, steps=3000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=1000),
        note="light layered: nu_A=0.1000 (bottom) / nu_B=0.1667 (top), ratio 1.67",
    ),
    "layered_r233_light": dict(
        kind="layered",
        cfg=dict(ny=24, nx=1, tau_a=0.8, tau_b=1.2, G=2.5, g_x=5e-5, g_y=0.0, steps=3000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=1000),
        note="light layered: nu_A=0.1000 (bottom) / nu_B=0.2333 (top), ratio 2.33",
    ),
    "layered_static_light": dict(
        kind="static",
        cfg=dict(ny=24, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=0.0, g_y=0.0, steps=3000,
                 init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5, interface_width=2.0,
                 g_ramp_steps=1000),
        note="light static interface: spurious currents must stay << driven peak",
    ),
}

#: matrices, cheapest first.  ``light`` is the CI/smoke matrix (H = 22, 3000
#: steps); ``core``/``full`` are the pre-registered matrices of the validation
#: entry point (H = 32, 12000 steps; ``full`` adds the H = 48 resolution check).
MATRICES: Dict[str, List[str]] = {
    "light": list(LIGHT_CASES),
    "core": list(CASE_MATRIX)[:5],
    "full": list(CASE_MATRIX),
}

#: minimum fluid height for the layer-curvature (nu) KPI.  The fit windows are
#: ``[2, H/2-4]`` and ``[H/2+4, H-2]`` cells; below ``H_MIN_NU_FIT`` they hold
#: <=5 points, i.e. the quadratic is not determined well enough to be quoted as
#: a viscosity measurement.  Such rows are *reported as skipped with a reason*
#: (``kpi_skipped``), never silently dropped.
H_MIN_NU_FIT = 32.0

#: KPI rows per case kind: ``(kpi, ref_key, got_key, err_key, err_name, gate)``
#: ``err_key`` may be ``None`` when the metric is itself an error (``eps_*``).
_KPI_ROWS: Dict[str, List[tuple]] = {
    "control": [
        ("u_peak", "u_peak_ref", "u_peak_num", "u_peak_err", "relerr", GATE_TARGET),
        ("eps_inf", None, None, "eps_inf", "relerr", GATE_TARGET),
        ("Q_total", None, None, "Q_err", "relerr", GATE_TARGET),
    ],
    "layered": [
        ("u_peak", "u_peak_ref", "u_peak_num", "u_peak_err", "relerr", GATE_OWNER),
        ("Q_total", None, None, "Q_err", "relerr", GATE_OWNER),
        ("eps_l2_rel", None, None, "eps_l2_rel", "relerr", GATE_OWNER),
        ("eps_inf", None, None, "eps_inf", "relerr", GATE_OWNER),
        ("y_iface", "y_iface_ref", "y_iface", "y_iface_err", "relH", 0.05),
        ("nu_lower", None, "nu_fit_lower", "nu_fit_err_lower", "relerr", GATE_OWNER),
        ("nu_upper", None, "nu_fit_upper", "nu_fit_err_upper", "relerr", GATE_OWNER),
    ],
    "static": [
        ("spurious_ratio", None, None, "spurious_ratio", "relref", GATE_OWNER),
    ],
}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", "..", ".."))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git unavailable in a tarball checkout
        return "unknown"


def kpi_table(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one run record into ``ref``/``got``/``err``/``gate`` rows.

    ``record`` is what :func:`.run_two_component_sc_validation.run_case_spec`
    returns (``metrics`` in SI-free lattice units, ``gates`` already evaluated).
    """
    m = record.get("metrics", {})
    rows: List[Dict[str, Any]] = []
    for kpi, ref_key, got_key, err_key, err_name, gate in _KPI_ROWS.get(record["kind"], []):
        err = m.get(err_key)
        if err is None:
            continue
        if kpi.startswith("nu_") and float(record.get("height", 0.0)) < H_MIN_NU_FIT:
            rows.append({
                "case": record["case"], "kind": record["kind"], "kpi": kpi,
                "ref": None, "got": None, "err": None, "err_kind": err_name,
                "threshold": float(gate), "pass": None,
                "skipped": f"height {record.get('height')} < H_MIN_NU_FIT={H_MIN_NU_FIT}: "
                           "layer-curvature fit under-resolved",
            })
            continue
        ref = m.get(ref_key) if ref_key else None
        got = m.get(got_key) if got_key else None
        if kpi == "Q_total":
            ref = float(np.sum(m["Q_ref"])) if "Q_ref" in m else None
            got = float(np.sum(m["Q_num"])) if "Q_num" in m else None
        elif kpi == "nu_lower":
            ref = record.get("nu_a")
        elif kpi == "nu_upper":
            ref = record.get("nu_b")
        rows.append({
            "case": record["case"],
            "kind": record["kind"],
            "kpi": kpi,
            "ref": None if ref is None else float(ref),
            "got": None if got is None else float(got),
            "err": float(err),
            "err_kind": err_name,
            "threshold": float(gate),
            "pass": bool(np.isfinite(err) and err <= gate),
        })
    return rows


def run_benchmark(
    matrix: str = "light",
    *,
    cases: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a benchmark matrix and return the ``ref vs got`` payload.

    Parameters
    ----------
    matrix : {'light', 'core', 'full'}
        ``light``/``core`` = the four cheap cases in :data:`LIGHT_CASES`;
        ``full`` = the six pre-registered cases of
        :data:`.run_two_component_sc_validation.CASE_MATRIX`.
    cases : sequence of str, optional
        Explicit case names (overrides ``matrix``).
    """
    names = list(cases) if cases else MATRICES[matrix]
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for name in names:
        spec = LIGHT_CASES.get(name) or CASE_MATRIX[name]
        rec = run_case_spec(spec, name=name, verbose=verbose)
        records.append(rec)
    wall = time.time() - t0

    rows: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("status") != "RAN":
            continue
        rows.extend(kpi_table(rec))
    live = [r for r in rows if r.get("pass") is not None]
    skipped = [r for r in rows if r.get("pass") is None]

    payload: Dict[str, Any] = {
        "benchmark": "two_component_sc_layered_poiseuille",
        "anchor": "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM",
        "matrix": matrix,
        "model": {
            "kind": "two-component Shan-Chen (MCMP) D2Q9, one distribution per component",
            "force_scheme": "EDM (Kupershtokh 2004) at the viscosity-weighted common velocity",
            "reference": "exact piecewise-quadratic layered solution (analytic.solve_layered), "
                         "independent of the lattice solver",
        },
        "gates": {"owner_standard": GATE_OWNER, "self_target": GATE_TARGET},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "git_commit": _git_commit(),
            "cwd": os.getcwd(),
        },
        "repro_cmd": _repro_cmd(matrix, cases),
        "kpi_table": rows,
        "records": records,
        "summary": {
            "n_cases": len(records),
            "n_diverged": sum(1 for r in records if r.get("status") != "RAN"),
            "n_kpi": len(live),
            "n_kpi_pass": sum(1 for r in live if r["pass"]),
            "n_kpi_fail": sum(1 for r in live if not r["pass"]),
            "n_kpi_skipped": len(skipped),
            "kpi_skipped": [{"case": r["case"], "kpi": r["kpi"], "reason": r.get("skipped")}
                            for r in skipped],
            "worst_err_by_kpi": _worst(live),
            "wall_seconds": round(wall, 1),
            "all_gates_pass": all(bool(r.get("gate_pass")) for r in records),
        },
    }
    return payload


def _worst(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("err") is None:
            continue
        cur = out.get(r["kpi"])
        if cur is None or r["err"] > cur["err"]:
            out[r["kpi"]] = {"case": r["case"], "err": r["err"], "threshold": r["threshold"]}
    return out


def _repro_cmd(matrix: str, cases: Optional[Sequence[str]]) -> str:
    command = (
        "PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.benchmark "
        f"--matrix {matrix}"
    )
    if cases:
        command += f" --cases {','.join(cases)}"
    return command


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def to_csv(payload: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case", "kind", "kpi", "ref", "got", "err", "err_kind", "threshold", "pass"])
    for r in payload["kpi_table"]:
        verdict = "SKIP" if r.get("pass") is None else ("PASS" if r["pass"] else "FAIL")
        w.writerow([r["case"], r["kind"], r["kpi"],
                    "" if r["ref"] is None else f"{r['ref']:.6g}",
                    "" if r["got"] is None else f"{r['got']:.6g}",
                    "" if r["err"] is None else f"{r['err']:.6g}",
                    r["err_kind"], f"{r['threshold']:.4g}", verdict])
    return buf.getvalue()


def to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# Two-component SC layered Poiseuille — ref vs got ({payload['matrix']} matrix)",
        "",
        f"anchor: `{payload['anchor']}` · git `{payload['environment']['git_commit'][:12]}` · "
        f"torch {payload['environment']['torch']} · wall {payload['summary']['wall_seconds']} s",
        "",
        "| case | KPI | ref (exact) | got | error | gate | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in payload["kpi_table"]:
        ref = "—" if r["ref"] is None else f"{r['ref']:.6g}"
        got = "—" if r["got"] is None else f"{r['got']:.6g}"
        err = "skip" if r["err"] is None else f"{100*r['err']:.2f} %"
        verdict = "SKIP" if r.get("pass") is None else ("PASS" if r["pass"] else "FAIL")
        lines.append(
            f"| `{r['case']}` | {r['kpi']} | {ref} | {got} | {err} "
            f"| {100*r['threshold']:.0f} % | {verdict} |"
        )
    s = payload["summary"]
    lines += [
        "",
        f"KPI rows: {s['n_kpi']} ({s['n_kpi_pass']} pass / {s['n_kpi_fail']} fail), "
        f"{s['n_kpi_skipped']} skipped, diverged runs: {s['n_diverged']}.",
        "",
        "```bash",
        payload["repro_cmd"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(payload: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """Write ``benchmark.json`` / ``benchmark.csv`` / ``benchmark.md`` + sha256."""
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    js = json.dumps(payload, indent=1)
    paths["json"] = os.path.join(out_dir, "benchmark.json")
    with open(paths["json"], "w") as fh:
        fh.write(js)
    paths["csv"] = os.path.join(out_dir, "benchmark.csv")
    with open(paths["csv"], "w") as fh:
        fh.write(to_csv(payload))
    paths["md"] = os.path.join(out_dir, "benchmark.md")
    with open(paths["md"], "w") as fh:
        fh.write(to_markdown(payload))
    digest = hashlib.sha256(js.encode()).hexdigest()
    paths["sha256"] = os.path.join(out_dir, "benchmark.sha256")
    with open(paths["sha256"], "w") as fh:
        fh.write(f"{digest}  benchmark.json\n")
    payload["sha256"] = digest
    return paths


# --------------------------------------------------------------------------- #
# single-case convenience entry used by the paper-evidence harness and the
# cross-validation registry (keeps both callers on this one code path)
# --------------------------------------------------------------------------- #
def benchmark_single_case(case: str = "layered_r167_light", *, verbose: bool = False) -> Dict[str, Any]:
    """Run one case and return its record plus its ``ref vs got`` rows."""
    spec = LIGHT_CASES.get(case) or CASE_MATRIX[case]
    rec = run_case_spec(spec, name=case, verbose=verbose)
    return {"case": case, "record": rec, "kpi_table": kpi_table(rec)}


if __name__ == "__main__":  # pragma: no cover - command-line entry
    import argparse

    ap = argparse.ArgumentParser(description="two-component SC layered Poiseuille benchmark")
    ap.add_argument("--matrix", default="light", choices=sorted(MATRICES))
    ap.add_argument("--cases", help="Comma-separated case names overriding --matrix")
    ap.add_argument("--out", default="artifacts/xval_two_component_sc_layered_poiseuille_edm")
    a = ap.parse_args()
    cases = None
    if a.cases is not None:
        cases = [name.strip() for name in a.cases.split(",")]
        unknown = [name for name in cases if name not in LIGHT_CASES and name not in CASE_MATRIX]
        if unknown:
            ap.error(f"unknown benchmark case(s): {', '.join(repr(name) for name in unknown)}")
    p = run_benchmark(a.matrix, cases=cases)
    for path in write_artifacts(p, a.out).values():
        print("wrote", path)
    print(to_markdown(p))
    raise SystemExit(0 if p["summary"]["all_gates_pass"] else 1)
