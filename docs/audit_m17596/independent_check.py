#!/usr/bin/env python3
"""Independent board-audit probes for mission m17596
(XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP, repo TensorLBM).

Three probes, all run against the *audited* worktree (read-only):

  --repro   re-execute the deterministic experiments E1/E2/E3/E4/E6 of
            benchmarks/xval_blockforest_metis_multistream/run_xval.py and diff
            every number against the shipped artifacts/.../result.json.
            E5 (wall-clock overlap timing) is excluded by design: the shipped
            report itself declares +-10 % run-to-run spread.

  --floor   compute the *quantized* optimum-workload-imbalance floor for the
            refined forests of E3 (block weight is 1x coarse / 8x level-1, and
            blocks are indivisible) and compare it with (a) the predeclared 5 %
            D2 threshold and (b) the shipped `ours_imbalance`.

  --figure  render docs/audit_m17596/fig_audit_m17596.png from the two JSONs.

Usage
-----
    python3 docs/audit_m17596/independent_check.py --floor
    python3 docs/audit_m17596/independent_check.py --repro      # ~2 min
    python3 docs/audit_m17596/independent_check.py --figure
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

WT = Path("/root/tp-agent/work/m17596")
OUT = Path(__file__).resolve().parent
D2_THRESHOLD = 0.05

REFINED_PATTERNS = {  # identical masks to run_xval.REFINED_PATTERNS
    "refA": ((4, 4, 4), tuple(1 if (i // 2 + j // 2 + k // 2) % 2 else 0
                              for i in range(4) for j in range(4) for k in range(4))),
    "refB": ((4, 4, 4), tuple(1 if (j * 4 + k) % 3 == 0 else 0
                              for i in range(4) for j in range(4) for k in range(4))),
    "refC": ((3, 3, 3), tuple(1 if (i + j + k) % 2 else 0
                              for i in range(3) for j in range(3) for k in range(3))),
}


def load_run_xval():
    spec = importlib.util.spec_from_file_location(
        "run_xval", WT / "benchmarks/xval_blockforest_metis_multistream/run_xval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_xval"] = mod
    spec.loader.exec_module(mod)
    return mod


def shipped():
    return json.loads(
        (WT / "artifacts/xval_blockforest_metis_multistream/result.json").read_text())


# --------------------------------------------------------------------- repro
def _cmp(exp, got, path, diffs):
    if isinstance(exp, dict):
        for k in exp:
            if k in got:
                _cmp(exp[k], got[k], f"{path}.{k}", diffs)
    elif isinstance(exp, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(exp) != len(got):
            diffs.append([path, "len", len(exp), len(got)])
        else:
            for i, (a, b) in enumerate(zip(exp, got)):
                _cmp(a, b, f"{path}[{i}]", diffs)
    elif isinstance(exp, (int, float)) and not isinstance(exp, bool) \
            and isinstance(got, (int, float)):
        if exp != got:
            rel = abs(got - exp) / max(abs(exp), 1e-300)
            if abs(got - exp) > 1e-12 and rel > 1e-12:
                diffs.append([path, exp, got, rel])
    elif exp != got:
        diffs.append([path, exp, got])


def probe_repro():
    mod = load_run_xval()
    orig = shipped()
    out = {
        "E1": mod.e1_exact(),
        "E2": mod.e2_uniform_reference(fast=False),
        "E3": mod.e3_balance(),
        "E4": mod.e4_physics(steps=400),
        "E6": mod.e6_sensitivity(),
    }
    diffs = []
    for k in out:
        _cmp(orig[k], out[k], k, diffs)
    timing_only = all("_s" in d[0] or "time" in d[0] for d in diffs)
    summary = {
        "audited_worktree": str(WT),
        "shipped_verdict": orig["verdict"],
        "n_number_diffs": len(diffs),
        "all_diffs_are_wall_clock": timing_only,
        "diffs": diffs[:40],
        "recomputed": {
            "E1_pass_D1_exact": out["E1"]["pass_D1_exact"],
            "E2_pass": {k: v for k, v in out["E2"].items() if k.startswith("pass")},
            "E3_pass": {k: v for k, v in out["E3"].items() if k.startswith("pass")},
            "E4": {k: out["E4"][k] for k in
                   ("steps", "rate_ref", "rate_decomposed_overlapped",
                    "rel_error_decomposed", "max_field_diff", "pass_D3", "pass_D4")},
        },
        "note": "E5 (wall-clock latency injection) not re-run: not reproducible by "
                "construction (shared CPU, +-10 % declared spread in report sec.5).",
    }
    (OUT / "repro_diff_m17596.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_number_diffs", "all_diffs_are_wall_clock", "recomputed")}, indent=2))
    return summary


# --------------------------------------------------------------------- floor
def _int_partitions(n, maxpart=None, prefix=()):
    if n == 0:
        yield prefix
        return
    maxpart = n if maxpart is None or maxpart > n else maxpart
    for i in range(maxpart, 0, -1):
        yield from _int_partitions(n - i, i, prefix + (i,))


def floor_imbalance(weights, k):
    """Exact minimum achievable `max(part weight)/mean - 1` for indivisible blocks
    with at most two distinct workloads (coarse 1x / refined 8x).

    Partitions are symmetric, so enumerate integer partitions of the heavy-block
    count; distribute the light (unit) blocks greedily to the least loaded part,
    which is optimal for identical unit items minimising the maximum.
    """
    distinct = sorted(set(weights))
    assert len(distinct) <= 2, distinct
    light, heavy = distinct[0], distinct[-1]
    n_heavy = sum(1 for w in weights if w == heavy)
    n_light = len(weights) - n_heavy
    best = math.inf
    for p in _int_partitions(n_heavy):
        if len(p) > k:
            continue
        parts = [x * heavy for x in p] + [0.0] * (k - len(p))
        for _ in range(n_light):
            i = min(range(k), key=lambda j: parts[j])
            parts[i] += light
        best = min(best, max(parts))
    mean = sum(weights) / k
    return best, best / mean - 1.0


def probe_floor():
    sys.path.insert(0, str(WT / "src"))
    from tensorlbm.blockforest import BlockForest  # noqa: E402

    res = shipped()
    rows = []
    for name, (dims, mask) in REFINED_PATTERNS.items():
        f = BlockForest(nblocks=dims, block_cells=(8, 8, 8), refinement=mask)
        w = [f.workload_at(i) for i in range(f.n_blocks)]
        for k in (2, 4, 8):
            mw, imb = floor_imbalance(w, k)
            r = next(x for x in res["E3"]["rows"]
                     if x["instance"] == name and x["k"] == k)
            rows.append({
                "instance": name, "k": k, "n_blocks": len(w),
                "n_refined": sum(1 for x in w if x == max(w)),
                "total_weight": sum(w),
                "ours_imbalance": round(r["ours_imbalance"], 6),
                "ours_cut": r["ours_cut"], "best_known_cut": r["best_known_cut"],
                "cut_rel_error_vs_best_known": round(r["rel_error_vs_best_known"], 6),
                "quantized_floor_min_max_weight": mw,
                "quantized_floor_imbalance": round(imb, 6),
                "ours_excess_over_floor": round(r["ours_imbalance"] - imb, 6),
                "d2_threshold_achievable": bool(imb <= D2_THRESHOLD),
            })
    payload = {
        "definition": "max(part weight)/mean(part weight) - 1, blocks indivisible, "
                      "workload 1x coarse / 8x level-1",
        "d2_threshold": D2_THRESHOLD,
        "rows": rows,
        "infeasible_cases": [f"{r['instance']}/k={r['k']}" for r in rows
                             if not r["d2_threshold_achievable"]],
    }
    (OUT / "imbalance_floor.json").write_text(json.dumps(payload, indent=2))
    for r in rows:
        print(json.dumps(r))
    return payload


# -------------------------------------------------------------------- figure
def probe_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = json.loads((OUT / "imbalance_floor.json").read_text())
    res = shipped()
    rows = floor["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    labels = [f"{r['instance']}\nk={r['k']}" for r in rows]
    x = range(len(rows))
    ours = [100 * r["ours_imbalance"] for r in rows]
    flr = [100 * r["quantized_floor_imbalance"] for r in rows]
    ax1.bar([i - 0.2 for i in x], ours, width=0.4, label="shipped partition",
            color="#c0392b")
    ax1.bar([i + 0.2 for i in x], flr, width=0.4, label="quantized floor (exact)",
            color="#7f8c8d")
    ax1.axhline(5.0, color="k", ls="--", lw=1.4, label="predeclared D2 cap 5 %")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("workload imbalance  max/mean - 1  [%]")
    ax1.set_title("D2 on E3 refined forests: shipped vs indivisible-block floor")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    e1 = res["E1"]["rows"]
    e2 = res["E2"]["rows"]
    e3 = res["E3"]["rows"]
    cat = ([f"E1 {r['instance']}" for r in e1]
           + [f"E2 {r['instance']} k={r['k']}" for r in e2]
           + [f"E3 {r['instance']} k={r['k']}" for r in e3])
    err = ([100 * r["ours_rel_error"] for r in e1]
           + [100 * r["rel_error_vs_kl"] for r in e2]
           + [100 * r["rel_error_vs_best_known"] for r in e3])
    ax2.bar(range(len(err)), err, color=["#27ae60" if abs(v) < 1e-9 else "#2980b9"
                                         for v in err])
    ax2.axhline(15.0, color="k", ls="--", lw=1.4, label="predeclared D1 cap 15 %")
    ax2.set_xticks(range(len(err)))
    ax2.set_xticklabels(cat, rotation=90, fontsize=6)
    ax2.set_ylabel("cut rel. error vs reference [%]")
    ax2.set_title("D1 cut quality vs exhaustive optimum (E1), networkx KL (E2),"
                  " best-of-64 (E3)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = OUT / "fig_audit_m17596.png"
    fig.savefig(p, dpi=130)
    print("figure:", p)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if a.repro:
        probe_repro()
    if a.floor:
        probe_floor()
    if a.figure:
        probe_figure()
    if not (a.repro or a.floor or a.figure):
        ap.print_help()