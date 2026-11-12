"""Figure for docs/INPUT_CONTRACT_GUARDS_LATTICE_20261112.md.

Reads the two probe JSONs produced by
``scripts/contract_guards/lattice_feature_probe.py`` and plots the *measured*
fail-fast metric (no hand-typed numbers):

  panel A — bytes of ``f`` allocated before the rejection, per unsupported combo
  panel B — rejection stage (setup vs run) per unsupported combo

Usage::

    python scripts/contract_guards/plot_lattice_probe.py \
        --before artifacts/contract_guards/lattice_probe_before.json \
        --after  artifacts/contract_guards/lattice_probe_after.json \
        --out    figures/lattice_contract_probe.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    names = [r["probe"] for r in before["rows"] if r["expect"] == "reject"]
    b_map = {r["probe"]: r for r in before["rows"]}
    a_map = {r["probe"]: r for r in after["rows"]}

    mb = [(b_map[n].get("field_bytes") or 0) / 1e6 for n in names]
    ma = [(a_map[n].get("field_bytes") or 0) / 1e6 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], mb, height=0.38, color="#c0392b", label="before (HEAD)")
    ax.barh([i - 0.2 for i in y], ma, height=0.38, color="#27ae60", label="after (guard)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("bytes of f allocated before the configuration was refused  [MB]")
    ax.set_title("A. Work spent before rejection (D3Q27 unsupported combos)")
    ax.legend(fontsize=8)
    for i, v in enumerate(mb):
        ax.text(v + 0.05, i + 0.2, f"{v:.2f} MB", va="center", fontsize=7)
    for i, v in enumerate(ma):
        ax.text(v + 0.05, i - 0.2, "0.00 MB", va="center", fontsize=7)
    ax.set_xlim(0, max(mb + [0.1]) * 1.35)

    ax = axes[1]
    stages = {"config": 0, "setup": 1, "run": 2, "ran": 3}
    ax.scatter(
        [stages[b_map[n].get("stage", "ran")] for n in names],
        list(y),
        s=70,
        marker="o",
        color="#c0392b",
        label="before",
    )
    ax.scatter(
        [stages[a_map[n].get("stage", "ran")] for n in names],
        list(y),
        s=70,
        marker="X",
        color="#27ae60",
        label="after",
    )
    for i, n in enumerate(names):
        ax.plot(
            [stages[b_map[n].get("stage", "ran")], stages[a_map[n].get("stage", "ran")]],
            [i, i],
            color="#7f8c8d",
            lw=1,
            ls=":",
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(list(stages.values()))
    ax.set_xticklabels(list(stages.keys()))
    ax.set_xlim(-0.4, 3.4)
    ax.set_xlabel("stage at which NotImplementedError surfaced")
    ax.set_title("B. Rejection moved from run() to setup()")
    ax.legend(fontsize=8)

    fig.suptitle(
        "GeneralSimEngine lattice/feature contract: fail-fast guard "
        f"(unsupported combos {before['summary']['unsupported_combos']}, "
        f"positive controls {after['summary']['positive_controls_ok']})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
