"""XVAL: TensorLBM gate policy vs the OSS lbmpy staged-gate CI practice.

Runs the pre-declared experiment in ``predeclared.json``:

1. normalise both CI definitions into one gate-policy schema (``ci_spec``)
2. evaluate each policy on TensorLBM's own pytest corpus (``gate_sim``)
3. validate the simulator against pytest's own selection counts
4. test the declared hypotheses (functional parity H1, vocabulary portability H2)
5. write ``summary.json``, ``table.csv``, figures and print the decision table

Every number in the output is derived from the artefacts under ``oss_refs/``
(upstream bytes + sha256) and ``data/`` (corpus snapshot, measured selection
counts); nothing is transcribed.

Usage::

    PYTHONPATH=src python xval/gitlab_ci_staged_gates/run_xval.py [--refresh-oss]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import ci_spec, gate_sim, oss_refs

HERE = Path(__file__).resolve().parent
OUT = HERE

# Declared in predeclared.json (stopping rule): the four expressions the
# simulator is validated against, and the vocabulary port used by ALT-2.
VALIDATION_EXPRS = ("not slow", "slow", "unit", "not longrun")
VOCAB_RENAME = {"longrun": "slow"}  # ALT-1 -> ALT-2 declared port
OURS_OPS = {"tests": ["tests"], "app/tests": ["app/tests"]}


def load_oss_spec() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = oss_refs.fetch()
    spec = ci_spec.parse_gitlab_ci(oss_refs.LOCAL["gitlab_ci"])
    nox = oss_refs.resolve_noxfile(spec, oss_refs.LOCAL["noxfile"])
    inventory = oss_refs.nox_gate_inventory(oss_refs.LOCAL["noxfile"])
    return spec, nox, {"manifest": manifest, "nox": nox, "inventory": inventory}


def transplant(spec: dict[str, Any], name: str, rename: dict[str, str] | None) -> dict[str, Any]:
    """Re-label the OSS policy for evaluation on TensorLBM's corpus.

    The declared scope of the OSS test gate is ``pytest`` over the project's
    own testpaths; on our corpus that means ``tests`` + ``app/tests``.  The
    adapter is part of ALT-2 and is *declared*, not inferred: the verbatim arm
    (ALT-1) keeps the OSS marker vocabulary, which is the quantity H2 measures.
    """
    import copy

    out = copy.deepcopy(spec)
    out["name"] = name
    if rename:
        for job in out["jobs"]:
            expr = job["selection"].get("marker_expr")
            if expr:
                for old, new in rename.items():
                    expr = expr.replace(old, new)
                job["selection"]["marker_expr"] = expr
    return out


def validate_simulator(
    corpus: gate_sim.Corpus, measurements: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for rec in measurements["rows"]:
        expr = None if rec["expr"] == "<all>" else rec["expr"]
        sim = len(corpus.ids(expr))
        measured = rec["n_collected"]
        err = abs(sim - measured) / max(1, measured)
        rows.append(
            {
                "expr": rec["expr"],
                "simulated": sim,
                "measured_pytest": measured,
                "abs_diff": abs(sim - measured),
                "rel_error_pct": 100.0 * err,
                "pass_15pct": err < 0.15,
                "cpu_s": rec.get("cpu_s"),
                "wall_s": rec.get("wall_s"),
            }
        )
    return rows


def leakage(
    corpus: gate_sim.Corpus, spec: dict[str, Any], rename: dict[str, str] | None
) -> dict[str, Any]:
    """Fraction of the repo's long-tier tests that a transplanted fast gate still runs."""
    arm = transplant(spec, "leakage-arm", rename)
    runs = gate_sim.apply_policy(
        arm,
        "mr",
        corpus,
        scope_override={"testsuite-cuda-py3.10": ["tests"], "testsuite-cpu-py3.13": ["tests"]},
    )
    fast_gates = [r for r in runs if not r.skipped and r.tier == "blocking" and r.selected]
    fast_selected: set[str] = set()
    for r in fast_gates:
        fast_selected |= set(r.selected)
    long_ids = corpus.ids("slow")
    leaked = sorted(long_ids & fast_selected)
    return {
        "n_long_tier": len(long_ids),
        "n_leaked": len(leaked),
        "leak_fraction": len(leaked) / max(1, len(long_ids)),
        "example_leaks": leaked[:5],
    }


def parity(row_ours: dict[str, Any], row_oss: dict[str, Any]) -> dict[str, Any]:
    """Compare the two policies on one event.

    ``guard_fraction`` is compared as a relative difference (the quantity is
    non-degenerate: both arms gate most of the corpus).  ``escape_fraction``
    is compared as an absolute difference in percentage points, because a
    relative difference is undefined when one policy escapes nothing - the
    convention is fixed here and applied to every event.
    """

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(1e-12, abs(b)) * 100.0

    g = rel(row_ours["guard_fraction"], row_oss["guard_fraction"])
    e_abs = abs(row_ours["escape_fraction"] - row_oss["escape_fraction"]) * 100.0
    return {
        "event": row_ours["event"],
        "guard_fraction_ours": row_ours["guard_fraction"],
        "guard_fraction_oss": row_oss["guard_fraction"],
        "guard_rel_diff_pct": g,
        "escape_abs_diff_pp": e_abs,
        "escape_fraction_ours": row_ours["escape_fraction"],
        "escape_fraction_oss": row_oss["escape_fraction"],
        "parity_pass_15pct": g < 15.0 and e_abs < 15.0,
    }


def build_table(rows: list[dict[str, Any]]) -> str:
    head = "arm,event,n_active_jobs,n_blocking_gates,n_guarded,guard_fraction,n_escaped,escape_fraction,n_tests_executed"
    lines = [head]
    for r in rows:
        lines.append(
            ",".join(
                str(x)
                for x in (
                    r["arm"],
                    r["event"],
                    r["n_active_jobs"],
                    r["n_blocking_gates"],
                    r["n_guarded"],
                    f"{r['guard_fraction']:.4f}",
                    r["n_escaped"],
                    f"{r['escape_fraction']:.4f}",
                    r["n_tests_executed"],
                )
            )
        )
    return "\n".join(lines) + "\n"


def figures(summary: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, str]] = []

    # 1. guard fraction per arm x event
    events = [e for e in ci_spec.EVENTS]
    arms = list(summary["kpis"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        vals = [summary["kpis"][arm][e]["guard_fraction"] * 100 for e in events]
        ax.bar([x + i * width for x in range(len(events))], vals, width, label=arm)
    ax.set_xticks([x + 0.4 for x in range(len(events))])
    ax.set_xticklabels(events)
    ax.set_ylabel("gated corpus fraction [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Gate coverage per policy arm and event")
    ax.legend(fontsize=7)
    p1 = out_dir / "fig1_guard_fraction.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    made.append(
        {
            "name": "fig1_guard_fraction.png",
            "caption": "gated corpus fraction per policy arm and event",
        }
    )

    # 2. ladder structure: declared vs active stages and tier mix
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(summary["ladder"])
    tiers = ["blocking", "advisory", "conditional", "manual"]
    bottom = [0] * len(names)
    for t in tiers:
        vals = [summary["ladder"][n]["tier_counts"].get(t, 0) for n in names]
        ax.bar(names, vals, bottom=bottom, label=t)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, n in enumerate(names):
        lad = summary["ladder"][n]
        ax.text(
            i,
            bottom[i] + 0.05,
            f"stages {lad['n_stages_active']}/{lad['n_stages_declared']}\nneeds {lad['needs_edges']}",
            ha="center",
            fontsize=7,
        )
    ax.set_ylabel("jobs by tier")
    ax.set_title("Gate ladder structure (declared stages vs active)")
    ax.legend(fontsize=7)
    p2 = out_dir / "fig2_ladder_structure.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    made.append(
        {
            "name": "fig2_ladder_structure.png",
            "caption": "tier mix and declared/active stage counts per arm",
        }
    )

    # 3. simulator fidelity vs pytest
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sims = [r["simulated"] for r in summary["fidelity"]]
    meas = [r["measured_pytest"] for r in summary["fidelity"]]
    ax.scatter(meas, sims, zorder=3)
    lim = max(meas + sims) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.fill_between(
        [0, lim],
        [0, lim * 0.85],
        [0, lim * 1.15],
        color="tab:green",
        alpha=0.15,
        label="+/-15% band",
    )
    for r in summary["fidelity"]:
        ax.annotate(
            r["expr"],
            (r["measured_pytest"], r["simulated"]),
            fontsize=6,
            xytext=(4, -8),
            textcoords="offset points",
        )
    ax.set_xlabel("pytest --collect-only (selected items)")
    ax.set_ylabel("simulator selection size")
    ax.set_title("Simulator vs pytest selection size")
    ax.legend(fontsize=7)
    p3 = out_dir / "fig3_simulator_fidelity.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    made.append(
        {
            "name": "fig3_simulator_fidelity.png",
            "caption": "simulated vs pytest-reported selection sizes with the 15% band",
        }
    )

    # 4. vocabulary leakage
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    labs = ["verbatim\n(ALT-1)", "adapted\n(ALT-2)"]
    vals = [
        summary["leakage"]["verbatim"]["leak_fraction"] * 100,
        summary["leakage"]["adapted"]["leak_fraction"] * 100,
    ]
    bars = ax.bar(labs, vals, color=["tab:red", "tab:green"])
    ax.axhline(15, ls="--", lw=1, color="k", label="15% acceptance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("long-tier tests leaked into fast gate [%]")
    ax.set_ylim(0, 115)
    ax.set_title("Marker-vocabulary portability")
    ax.legend(fontsize=7)
    p4 = out_dir / "fig4_vocabulary_leakage.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    made.append(
        {
            "name": "fig4_vocabulary_leakage.png",
            "caption": "long-tier leakage of the verbatim vs adapted marker vocabulary",
        }
    )

    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--refresh-oss", action="store_true", help="refetch the upstream OSS references"
    )
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.refresh_oss:
        oss_refs.fetch(refresh=True)

    corpus = gate_sim.Corpus.load(HERE / "data" / "corpus_markers.json")
    measurements = json.loads((HERE / "data" / "selection_measurements.json").read_text())
    ours = ci_spec.parse_github_actions(HERE.parents[1] / ".github" / "workflows" / "ci.yml")
    oss, nox_resolution, oss_meta = load_oss_spec()

    arms = {
        "ALT-0_ours": ours,
        "ALT-1_verbatim": transplant(oss, "ALT-1_verbatim", None),
        "ALT-2_adapted": transplant(oss, "ALT-2_adapted", VOCAB_RENAME),
    }
    # ALT-1/ALT-2 are the OSS ladder evaluated on our corpus; the test gates get
    # the declared scope mapping (OSS pytest scope -> our tests/ + app/tests).
    scope_override = {
        "testsuite-cuda-py3.10": ["tests", "app/tests"],
        "testsuite-cpu-py3.13": ["tests", "app/tests"],
        "nightly-sympy": ["tests", "app/tests"],
    }

    kpi_table: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    for arm_name, spec in arms.items():
        kpi_table[arm_name] = {}
        for event in ci_spec.EVENTS:
            row = gate_sim.kpis(spec, event, corpus, None, scope_override)
            row.pop("guard_vector", None)
            row["arm"] = arm_name
            kpi_table[arm_name][event] = row
            flat_rows.append(row)

    ladder = {name: gate_sim.ladder_structure(spec) for name, spec in arms.items()}
    fidelity = validate_simulator(corpus, measurements)
    leak_v = leakage(corpus, oss, None)
    leak_a = leakage(corpus, oss, VOCAB_RENAME)
    parity_rows = [
        parity(kpi_table["ALT-0_ours"][e], kpi_table["ALT-2_adapted"][e]) for e in ci_spec.EVENTS
    ]

    h1 = all(r["parity_pass_15pct"] for r in parity_rows)
    h2 = leak_v["leak_fraction"] >= 0.15 and leak_a["leak_fraction"] < 0.15
    fidelity_ok = all(r["pass_15pct"] for r in fidelity)

    decision = _decide(h1, h2, fidelity_ok, parity_rows, leak_v, leak_a, ladder)
    summary = {
        "decision": decision,
        "hypotheses": {
            "H1_functional_parity": {"holds": h1, "rows": parity_rows},
            "H2_vocabulary_portability": {"holds": h2, "verbatim": leak_v, "adapted": leak_a},
            "fidelity_validated": fidelity_ok,
        },
        "kpis": kpi_table,
        "ladder": ladder,
        "fidelity": fidelity,
        "leakage": {"verbatim": leak_v, "adapted": leak_a},
        "oss_refs": oss_meta,
        "corpus": {"n_items": len(corpus), "path": str(HERE / "data" / "corpus_markers.json")},
        "predeclared": json.loads((HERE / "predeclared.json").read_text()),
    }

    if not args.no_figures:
        summary["figures"] = figures(summary, HERE / "figures")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "table.csv").write_text(build_table(flat_rows))

    _print_report(summary)
    return 0 if (h1 and fidelity_ok) else 1


def _decide(
    h1: bool,
    h2: bool,
    fidelity_ok: bool,
    parity_rows: list[dict[str, Any]],
    leak_v: dict[str, Any],
    leak_a: dict[str, Any],
    ladder: dict[str, Any],
) -> dict[str, Any]:
    if not fidelity_ok:
        chosen, why = (
            "WITHHELD",
            "simulator fidelity >= 15% on at least one expression: policy KPIs are unvalidated",
        )
    elif h1 and h2:
        chosen = "ALT-2"
        why = (
            "functional coverage already matches the OSS policy within 15% on every event, so the ladder itself "
            f"carries no coverage gain; the OSS marker vocabulary leaks {leak_v['leak_fraction'] * 100:.1f}% of the "
            f"long tier when copied verbatim and {leak_a['leak_fraction'] * 100:.1f}% once adapted, so only the "
            "declarative elements are portable"
        )
    elif h1 and not h2:
        chosen = "ALT-0"
        why = "parity holds and the vocabulary port is not needed (leak below 15%)"
    else:
        chosen = "ALT-2 (coverage gap)"
        why = "coverage parity fails: at least one event leaves corpus items ungated relative to the OSS policy"
    return {
        "chosen_alternative": chosen,
        "reason": why,
        "parity_passed_events": [r["event"] for r in parity_rows if r["parity_pass_15pct"]],
        "parity_failed_events": [r["event"] for r in parity_rows if not r["parity_pass_15pct"]],
        "structural_diff": {
            name: {
                "n_stages_declared": ladder[name]["n_stages_declared"],
                "n_stages_active": ladder[name]["n_stages_active"],
                "needs_edges": ladder[name]["needs_edges"],
                "test_gate_configs": ladder[name]["test_gate_configs"],
                "artifact_contracts": ladder[name]["artifact_contracts"],
                "tier_counts": ladder[name]["tier_counts"],
            }
            for name in ladder
        },
    }


def _print_report(summary: dict[str, Any]) -> None:
    print("== simulator fidelity (sim vs pytest, acceptance <15%) ==")
    for r in summary["fidelity"]:
        flag = "OK " if r["pass_15pct"] else "FAIL"
        print(
            f"  {flag} {r['expr']:<13} sim={r['simulated']:<6} pytest={r['measured_pytest']:<6} err={r['rel_error_pct']:.3f}%"
        )
    print("\n== gate coverage (guarded corpus fraction) ==")
    for arm, per in summary["kpis"].items():
        print(f"  {arm}")
        for event, row in per.items():
            print(
                f"    {event:<12} gates={row['n_blocking_gates']} guarded={row['n_guarded']}/{summary['corpus']['n_items']}"
                f" ({row['guard_fraction'] * 100:.2f}%) escaped={row['n_escaped']}"
            )
    print("\n== ladder structure ==")
    for arm, lad in summary["ladder"].items():
        print(
            f"  {arm:<16} jobs={lad['n_jobs']} stages={lad['n_stages_active']}/{lad['n_stages_declared']} "
            f"needs={lad['needs_edges']} configs={lad['test_gate_configs']} artifacts={lad['artifact_contracts']} tiers={lad['tier_counts']}"
        )
    print("\n== H2 vocabulary portability ==")
    print(
        f"  verbatim leak={summary['leakage']['verbatim']['leak_fraction'] * 100:.2f}%  adapted leak={summary['leakage']['adapted']['leak_fraction'] * 100:.2f}%"
    )
    print("\n== decision ==")
    d = summary["decision"]
    print(f"  chose {d['chosen_alternative']}: {d['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
