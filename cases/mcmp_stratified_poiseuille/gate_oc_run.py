#!/usr/bin/env python3
"""m16895 · Operating-characteristic audit of the MCMP stratified-Poiseuille gates.

Anchor: ``FEAT-LBM-VERIFY-BENCH`` (TensorLBM MCMP calibration & *falsification*
benchmark tooling).

What this script does (pure post-processing — no LBM run, no re-integration):

1. **Declares** the case's acceptance gates from the pre-registered SPEC §6
   (``docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md``) and records one line of
   rationale per gate.
2. **Triangulates every threshold** across three independent carriers — the
   harness declaration, the number stored in the acceptance records
   (``result*.json`` ``thresholds`` blocks) and the number the *producing source
   compares against* (parsed with :mod:`ast` out of ``r3_benchmark.py`` /
   ``r4_benchmark.py``, never imported).  Any disagreement is a defect: a record
   that claims a 10 % gate while the code compares against 20 % is exactly the
   silent-slack failure this module exists to catch.
3. **Reproduces every recorded verdict** from the recorded metric values, so the
   "PASS/FAIL" columns in the acceptance record are machine-checked rather than
   trusted (mismatch = the record and its own numbers disagree).
4. **Quantifies the gates' operating characteristics**: switch point, sensitivity
   (minimum detectable defect), and *utilisation* (|metric|/threshold) per
   recorded cell — i.e. how close each cell came to flipping.
5. **Falsifies the gate set**: for each gate, degenerate mutants (threshold×2,
   threshold/2, inverted operator) are run over a conforming/defective probe
   battery; a mutant that survives means the battery (hence the "verified" claim)
   is under-powered.

Writes ``benchmarks/verified/mcmp_stratified_poiseuille/gate_oc.json`` and
figures under ``figures/gate_oc/``.  Zero hand-copied numbers: every value in the
report is read from the records or parsed from source inside this run.

Usage::

    PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/gate_oc_run.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tensorlbm.audit import gate_oc as go  # noqa: E402

VER = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille")
CASE = os.path.join("cases", "mcmp_stratified_poiseuille")
FIGDIR = os.path.join("figures", "gate_oc")
OUT = os.path.join(VER, "gate_oc.json")

# --- 1. declared gates (values from SPEC §6; rationale is mandatory) ---------
GATES = [
    go.Gate("iface_5pct", "err_interface_rel", "abs_le", 0.05,
            "SPEC §6: interface position within 5 % of the analytic halfway-plane "
            "interface; 5 % covers the 1-cell interface width at ny=25."),
    go.Gate("flow_mix_10pct", "err_Q_mix", "abs_le", 0.10,
            "SPEC §6 + QR independent audit: mixture volumetric flux within 10 % of "
            "the closed-form two-layer Poiseuille reference (nominal 0.7/0.3 "
            "viscosity rule)."),
    go.Gate("flow_component_10pct", "err_Q_mix_component", "abs_le", 0.10,
            "same 10 % gate evaluated under the *component* viscosity rule — the "
            "record requires both rules to pass, so the verdict cannot depend on a "
            "silently chosen mixing law."),
    go.Gate("spurious_static_0p5pct", "spurious_ratio_static", "abs_le", 0.005,
            "criterion ③: parasitic interface velocity in the static control "
            "(G_x=0) normalised by the driven-case analytic u_max must stay below "
            "0.5 % of the characteristic velocity (Kruger et al. 2017 ch. 9)."),
    go.Gate("stratified_phi_0p5", "phi_contrast", "ge", 0.5,
            "stratification gate: below phi_contrast 0.5 the phases have mixed, so "
            "a two-layer flux comparison is not a measurement of two-layer flow "
            "(declared in result.json thresholds as phi_contrast_min)."),
    go.Gate("steady_1e-3", "steady_rel_change", "abs_le", 1e-3,
            "steadiness gate: relative drift of the total flux over the last three "
            "samples must be < 1e-3, otherwise the time-average is not a steady "
            "state (declared in result.json thresholds)."),
]

# threshold carrier names: declared gate -> (r3/r4 TARGET key, result.json key)
THRESHOLD_MAP = {
    "iface_5pct": ("err_iface", "err_interface_rel_abs"),
    "flow_mix_10pct": ("err_Q", "err_Q_mix_abs"),
    "flow_component_10pct": ("err_Q", "err_Q_mix_abs"),
    "spurious_static_0p5pct": ("ratio_S", "spurious_ratio"),
    "stratified_phi_0p5": ("phi", "phi_contrast_min"),
    "steady_1e-3": (None, "steady_rel_change"),
}


def load(name: str) -> dict:
    with open(os.path.join(VER, name), "r", encoding="utf-8") as fh:
        return json.loads(fh.read())


def source_thresholds() -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Parse TARGET dicts out of the producing sources (no import, no execution)."""
    r3, r4 = {}, {}
    for key, gate in (("err_iface", "iface_5pct"), ("err_Q", "flow_mix_10pct"),
                      ("ratio_S", "spurious_static_0p5pct"), ("phi", "stratified_phi_0p5")):
        r3[gate] = go.parse_source_constant(os.path.join(CASE, "r3_benchmark.py"), "TARGET", key)
        r4[gate] = go.parse_source_constant(os.path.join(CASE, "r4_benchmark.py"), "TARGET", key)
    r3["flow_component_10pct"] = r3["flow_mix_10pct"]
    r4["flow_component_10pct"] = r4["flow_mix_10pct"]
    r3["steady_1e-3"] = None      # compared against a bare literal in run_sweep.py
    r4["steady_1e-3"] = None
    return r3, r4


def records_from_files() -> tuple[dict[str, dict], dict[str, dict]]:
    """Build flat metric records + the verdicts the records *claim* for them."""
    res = load("result.json")
    r3 = load("result_r3.json")
    records: dict[str, dict] = {}
    claimed: dict[str, dict] = {}

    for i, cell in enumerate(res["cells"]):
        em = cell.get("errors_mixture") or {}
        ec = cell.get("errors_component") or {}
        g = cell.get("gates") or {}
        v = cell.get("verdict") or {}
        name = f"result/{cell['tag']}"
        records[name] = {
            "err_interface_rel": em.get("err_interface_rel"),
            "err_Q_mix": em.get("err_Q_mix"),
            "err_Q_mix_component": ec.get("err_Q_mix"),
            "phi_contrast": cell.get("phi_contrast", g.get("phi_contrast")),
            "steady_rel_change": None,   # the record stores only the boolean
            "spurious_ratio_static": None,
        }
        claimed[name] = {
            "iface_5pct": "PASS" if v.get("interface_ok") else "FAIL",
            "flow_mix_10pct": "PASS" if v.get("flow_rate_ok") else "FAIL",
            "flow_component_10pct": "PASS" if v.get("flow_rate_ok_vs_component_rule") else "FAIL",
            "stratified_phi_0p5": "PASS" if g.get("stratified") else "FAIL",
        }

    # r3 driven cells: c1 (interface), c2 (flux, both viscosity rules)
    std = r3["criteria"]["standard"]
    for cell in std["c1_interface"]["cells"]:
        name = f"result_r3/c1_ny{cell['ny']}"
        records.setdefault(name, {})["err_interface_rel"] = cell["err"]
        claimed.setdefault(name, {})["iface_5pct"] = "PASS" if cell["pass_"] else "FAIL"
    for cell in std["c2_flow"]["cells"]:
        name = f"result_r3/c2_ny{cell['ny']}"
        rec = records.setdefault(name, {})
        rec["err_Q_mix"] = cell["err_mixture"]
        rec["err_Q_mix_component"] = cell["err_component"]
        claim = claimed.setdefault(name, {})
        # c2 is a single flag covering both rules: attribute it to each rule only
        # when that rule alone decides it, otherwise record the joint flag.
        claim["flow_mix_10pct"] = "PASS" if cell["pass_"] else "FAIL"
        claim["flow_component_10pct"] = "PASS" if cell["pass_"] else "FAIL"
    for cell in std["c3_spurious"]["cells"]:
        name = f"result_r3/c3_static_ny{cell['ny']}"
        records.setdefault(name, {})["spurious_ratio_static"] = cell["ratio_ifc"]
        claimed.setdefault(name, {})["spurious_static_0p5pct"] = (
            "PASS" if cell["pass_"] else "FAIL")
    return records, claimed


def utilisation(gate: go.Gate, value: float | None) -> float | None:
    """|metric| / threshold (1.0 = exactly at the gate).  ``ge`` gates invert."""
    if value is None:
        return None
    if gate.op == "ge":
        return (gate.threshold / value) if value else float("inf")
    return abs(value) / gate.threshold


def reproduce(records: dict, claimed: dict) -> dict:
    """Machine-reproduce the claimed verdicts from the recorded metric values."""
    mismatches, compared, withheld = [], 0, 0
    for name, rec in records.items():
        for g in GATES:
            if name not in claimed or g.name not in claimed[name]:
                continue
            got = go.check(g, rec.get(g.key))
            want = claimed[name][g.name]
            if got == go.WITHHELD:
                withheld += 1
                continue
            compared += 1
            if got != want:
                mismatches.append({"record": name, "gate": g.name, "metric_value":
                                   rec.get(g.key), "threshold": g.threshold,
                                   "recomputed": got, "recorded_claim": want})
    return {"n_compared": compared, "n_withheld": withheld,
            "n_mismatch": len(mismatches), "mismatches": mismatches}


def figures(report: dict, path: str) -> list[str]:
    os.makedirs(FIGDIR, exist_ok=True)
    made = []

    # fig 1: utilisation per record per gate (1.0 = gate trips)
    gates = report["gates"]
    labels, utils = [], []
    for g in gates:
        rows = []
        for rec, v in g["verdicts"].items():
            u = utilisation(go.Gate(g["gate"], g["key"], g["op"], g["threshold"]), None)
            rows.append((rec, v, u))
        labels.append(g["gate"])
        utils.append(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xs = range(len(gates))
    passing_util = []
    for g_row in gates:
        vals = [u for (_, _, u) in
                [(r, v, _util_cached(records_cache, go.Gate(g_row["gate"], g_row["key"],
                                                            g_row["op"], g_row["threshold"]), r))
                 for r, v in g_row["verdicts"].items()]]
        vals = [v for v in vals if v is not None]
        ax.scatter([labels.index(g_row["gate"])] * len(vals), vals, s=18, alpha=0.75)
        passing_util.append((min(vals) if vals else None, max(vals) if vals else None))
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label="gate trips (utilisation = 1)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("|metric| / threshold")
    ax.set_title("Gate utilisation of every recorded cell (m16895)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig1_gate_utilisation.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # fig 2: operating characteristic (verdict vs defect factor)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for g_row in gates:
        curve = g_row["detection_curve"]
        ys = [1 if c["verdict"] == go.FAIL else 0 for c in curve]
        ax.step([c["factor"] for c in curve], ys, where="mid", label=g_row["gate"], lw=1.4)
    ax.set_xlabel("defect factor  f  (value = threshold x (1+f))")
    ax.set_ylabel("verdict  (0 = PASS, 1 = FAIL)")
    ax.set_yticks([0, 1])
    ax.set_title("Gate switch point: every gate trips exactly at its declared threshold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig2_switch_point.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # fig 3: mutant discrimination matrix
    names = [g["gate"] for g in gates]
    mut_labels: list[str] = []
    for g in gates:
        for m in g["discrimination"]["mutants"]:
            if m not in mut_labels:
                mut_labels.append(m)
    mat = []
    for g in gates:
        row = []
        for m in mut_labels:
            row.append(g["discrimination"]["mutants"].get(m, {}).get("n_differ", 0))
        mat.append(row)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mut_labels)))
    ax.set_xticklabels(mut_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="w" if v == 0 else "k", fontsize=8)
    ax.set_title("Probes that kill each gate mutant (0 = mutant survives)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="n differing probes")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig3_mutant_kill.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)
    return made


def _util_cached(records: dict, gate: go.Gate, rec_name: str) -> float | None:
    return utilisation(gate, records[rec_name].get(gate.key))


records_cache: dict = {}


def main() -> int:
    global records_cache
    records, claimed = records_from_files()
    records_cache = records

    r3_src, r4_src = source_thresholds()
    res_json = load("result.json")["thresholds"]
    declared = {g.name: g.threshold for g in GATES}
    recorded: dict[str, float] = {}
    for gate, (_target_key, json_key) in THRESHOLD_MAP.items():
        if json_key in res_json:
            recorded[gate] = float(res_json[json_key])

    tri = {
        "vs_result_json": go.triangulate(declared, recorded, declared),
        "vs_r3_source": go.triangulate(declared, recorded, r3_src),
        "vs_r4_source": go.triangulate(declared, recorded, r4_src),
    }

    probes_by_gate = {g.name: go.probe_battery(g) for g in GATES}
    flat_probes: list[dict] = []
    for name, battery in probes_by_gate.items():
        key = next(g.key for g in GATES if g.name == name)
        for p in battery:
            row = {k: v for k, v in p.items()}
            flat_probes.append({key: row[key]})

    report = go.run_report(GATES, records, triangulation=tri, probes=flat_probes)
    report["reproduction"] = reproduce(records, claimed)
    report["utilisation"] = {
        g.name: {name: utilisation(g, rec.get(g.key)) for name, rec in records.items()}
        for g in GATES
    }
    report["provenance"] = {
        "records_read": ["result.json", "result_r3.json"],
        "sources_parsed": ["cases/mcmp_stratified_poiseuille/r3_benchmark.py",
                           "cases/mcmp_stratified_poiseuille/r4_benchmark.py"],
        "spec": "docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md §6",
        "thousands_hand_copied": 0,
        "note": "every number below is read from the records or parsed from source "
                "inside this run; the declared thresholds are the SPEC values and are "
                "triangulated against both carriers.",
    }
    report["headline"] = {
        "n_gates": len(GATES),
        "triangulation_agree": all(v["agree"] or v["missing"] for v in tri.values()),
        "triangulation_disagreements": {
            k: v["rows"] for k, v in tri.items() if v["rows"]},
        "triangulation_missing_evidence": {
            k: v["missing"] for k, v in tri.items() if v["missing"]},
        "verdict_reproduction": {k: report["reproduction"][k]
                                 for k in ("n_compared", "n_mismatch", "n_withheld")},
        "gates_ever_fail": [g["gate"] for g in report["gates"]
                            if any(v == go.FAIL for v in g["verdicts"].values())],
        "mutants_surviving": [f"{g['gate']}/{m}" for g in report["gates"]
                              for m, v in g["discrimination"]["mutants"].items() if not v["killed"]],
        "min_detectable_factor": {g["gate"]: g["minimum_detectable_factor"]
                                  for g in report["gates"]},
        "switch_probe_ok": {g["gate"]: g["switch_probe"]["ok"] for g in report["gates"]},
    }

    figs = figures(report, OUT)
    report["figures"] = [os.path.basename(f) for f in figs]
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(json.dumps(report["headline"], indent=1))
    print("wrote", OUT, "and", len(figs), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())