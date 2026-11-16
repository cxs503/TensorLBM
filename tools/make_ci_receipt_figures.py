#!/usr/bin/env python3
"""Figures + metrics for the staged-gate receipt freshness work.

Everything here is derived, nothing is hand-transcribed:

* the ladder inventory and gate power come from the committed receipt, which in
  turn is generated from ``.gitlab-ci.yml``;
* the *stale* claim counts come from the previous revision of the design
  document (``git show <rev>:docs/ci/gitlab_ci_staged_gates.md``), parsed out of
  its hand-written ladder code block;
* the staleness-gate verdicts are produced by actually running the gate on
  fresh and mutated inputs.

    python tools/make_ci_receipt_figures.py --out artifacts/ci_receipt_report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOOL = REPO_ROOT / "tools" / "verify_ci_staged_gates.py"
CI = REPO_ROOT / ".gitlab-ci.yml"
RECEIPT = REPO_ROOT / "docs" / "ci" / "staged_gate_semantics_receipt.json"
DESIGN_DOC = REPO_ROOT / "docs" / "ci" / "gitlab_ci_staged_gates.md"
BASE_REV = "ea9ca221f"  # revision at which the ladder receipt was first committed


def run(*args: str) -> int:
    return subprocess.run(
        [sys.executable, str(TOOL), *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).returncode


def old_claims() -> dict[str, int]:
    """Active-job counts as claimed by the *previous* design document.

    The previous revision carried a hand-written ``| Stage | Purpose | Tier |
    Jobs |`` table; its ``Jobs`` cell is the claim.  "*(commented out)*" means
    the stage was declared but empty (0 active jobs).
    """
    text = subprocess.run(
        ["git", "show", f"{BASE_REV}:docs/ci/gitlab_ci_staged_gates.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    section = re.search(r"### Stage Descriptions(.*?)### Gate-Tier", text, re.S)
    if not section:
        raise SystemExit("could not locate the hand-written stage table")
    stages: dict[str, int] = {}
    for row in section.group(1).splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[0].startswith("**"):
            continue
        stage = cells[0].strip("* ")
        jobs_cell = cells[3]
        if "commented out" in jobs_cell:
            stages[stage] = 0
            continue
        jobs_cell = re.sub(r"\s*\(advisory\)", "", jobs_cell)
        stages[stage] = len([j for j in jobs_cell.split(",") if j.strip()])
    if not stages:
        raise SystemExit("stage table parsed empty")
    return stages


def staleness_verdicts(tmp: Path) -> list[dict[str, object]]:
    """Run the gate on a fresh and four deliberately broken states."""
    cases: list[dict[str, object]] = []

    cases.append(
        {
            "case": "committed receipt / doc",
            "rc": run("--check", str(RECEIPT), "--check-doc", str(DESIGN_DOC)),
        }
    )

    nightly = (
        'nightly-full:\n  stage: nightly\n  rules:\n    - if: $CI_PIPELINE_SOURCE == "schedule"\n'
    )
    text = CI.read_text()
    if nightly not in text:
        raise SystemExit("nightly-full rules block moved; update the mutation")
    mutated = tmp / "ci_schedule_gate_removed.yml"
    mutated.write_text(text.replace(nightly, "nightly-full:\n  stage: nightly\n", 1))
    cases.append(
        {
            "case": "event scope changed\n(schedule gate made unconditional)",
            "rc": run("--ci", str(mutated), "--check", str(RECEIPT)),
        }
    )

    marker = "typecheck-core:\n  extends: .quality-base\n"
    start = text.index(marker)
    stop = text.index("allow_failure: false", start)
    mutated2 = tmp / "ci_tier_demoted.yml"
    mutated2.write_text(
        text[:stop] + "allow_failure: true" + text[stop + len("allow_failure: false") :]
    )
    cases.append(
        {
            "case": "gate tier changed\n(blocking -> advisory)",
            "rc": run("--ci", str(mutated2), "--check", str(RECEIPT)),
        }
    )

    doc = tmp / "design_edited.md"
    doc.write_text(DESIGN_DOC.read_text().replace("| `test` | 5 |", "| `test` | 4 |", 1))
    cases.append({"case": "hand-edited ladder table", "rc": run("--check-doc", str(doc))})

    cases.append({"case": "receipt deleted", "rc": run("--check", str(tmp / "absent.json"))})
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "ci_receipt_report")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    receipt = json.loads(RECEIPT.read_text())
    stages = [s["stage"] for s in receipt["ladder"]]
    measured = {s["stage"]: len(s["jobs"]) for s in receipt["ladder"]}
    claimed = old_claims()

    # ---- fig1: gate power per event -------------------------------------
    events = list(receipt["events"])
    gates = receipt["gates"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(events)
    for k, event in enumerate(events):
        power = receipt["events"][event]["gate_power"]
        xs = [i + k * width for i in range(len(gates))]
        ax.bar(xs, [power.get(g, 0) for g in gates], width=width, label=event)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(gates))])
    ax.set_xticklabels(gates, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("downstream jobs pruned\nwhen this gate fails alone")
    ax.set_title("Staged-gate ladder: gate power per event (generated from .gitlab-ci.yml)")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(args.out / "fig1_gate_power.png", dpi=150)
    plt.close(fig)

    # ---- fig2: documented vs measured inventory --------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = range(len(stages))
    ax.bar(
        [x - 0.2 for x in xs],
        [claimed.get(s, 0) for s in stages],
        width=0.4,
        label="previous doc (hand-written)",
    )
    ax.bar(
        [x + 0.2 for x in xs],
        [measured[s] for s in stages],
        width=0.4,
        label="measured (generated)",
    )
    for x, s in zip(xs, stages):
        ax.text(x - 0.2, claimed.get(s, 0) + 0.05, str(claimed.get(s, 0)), ha="center", fontsize=8)
        ax.text(x + 0.2, measured[s] + 0.05, str(measured[s]), ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(stages, rotation=20)
    ax.set_ylabel("active jobs in stage")
    ax.set_title("Ladder inventory: documented claims vs measured (.gitlab-ci.yml)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "fig2_inventory_drift.png", dpi=150)
    plt.close(fig)

    # ---- fig3: staleness gate verdicts -----------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = staleness_verdicts(Path(tmpdir))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["#2e7d32" if c["rc"] == 0 else "#c62828" for c in cases]
    ax.barh([c["case"] for c in cases], [1 for _ in cases], color=colors)
    for i, c in enumerate(cases):
        ax.text(
            0.5,
            i,
            "FRESH (rc=0)" if c["rc"] == 0 else "STALE (rc=1)",
            va="center",
            ha="center",
            color="w",
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Anti-staleness gate: verdict on fresh input and four mutations")
    fig.tight_layout()
    fig.savefig(args.out / "fig3_staleness_gate.png", dpi=150)
    plt.close(fig)

    metrics = {
        "source": receipt["source"],
        "source_sha256": receipt["source_sha256"],
        "n_stages": len(stages),
        "n_jobs": receipt["n_jobs"],
        "n_blocking": len(receipt["gates"]),
        "n_advisory": len(receipt["advisory"]),
        "declared_but_empty": [s for s, n in measured.items() if n == 0],
        "documented_vs_measured": {
            s: {"documented": claimed.get(s, 0), "measured": measured[s]} for s in stages
        },
        "stages_where_doc_was_stale": sorted(s for s in stages if claimed.get(s, 0) != measured[s]),
        "stages_missing_from_doc": sorted(s for s in stages if s not in claimed),
        "staleness_gate": cases,
        "claims_checked": receipt["claims"]["n_checks"],
        "claims_failed": receipt["claims"]["n_failed"],
        "structural_errors": receipt["n_error"],
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
