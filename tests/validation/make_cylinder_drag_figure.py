"""Figure for docs/cylinder_drag_benchmark_r1.md — built from the registry + samples.

Every input is in the repository, so the committed PNG is reproducible with one
command::

    python -m tests.validation.make_cylinder_drag_figure

Sample sources, in order of preference:

1. ``--csv PATH`` (a fresh ``--emit-dir`` run of the benchmark harness);
2. ``artifacts/cylinder_drag/samples.csv`` (gitignored scratch output of such a run);
3. ``tests/validation/data/cylinder_drag_r1_samples.csv`` — the committed
   decimated (every 10th step) sample tail of the ``dfg2d2_simple`` run, which is
   what makes the figure reproducible from a clean checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
REG = REPO / "docs/cylinder_drag_registry.json"
OUT = REPO / "docs/assets/cylinder_drag_benchmark.png"
SAMPLE_SOURCES = (
    REPO / "artifacts/cylinder_drag/samples.csv",
    REPO / "tests/validation/data/cylinder_drag_r1_samples.csv",
)


def _sample_path(explicit: str | None) -> pathlib.Path | None:
    """First existing sample CSV (explicit path wins, then repo candidates)."""
    if explicit:
        path = pathlib.Path(explicit)
        assert path.exists(), f"no such sample CSV: {path}"
        return path
    for cand in SAMPLE_SOURCES:
        if cand.exists():
            return cand
    return None


def _read_samples(path: pathlib.Path) -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["run"], []).append((float(r["cd"]), float(r["cl"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="sample CSV from --emit-dir (optional)")
    ap.add_argument("--registry", default=str(REG))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    reg = json.loads(pathlib.Path(args.registry).read_text())
    out = pathlib.Path(args.out)
    samples = _sample_path(args.csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if samples is not None:
        for name, seq in _read_samples(samples).items():
            ax.plot(range(len(seq)), [c for c, _ in seq], lw=0.8, label=f"{name} Cd")
        ax.set_title(f"Drag history (real LBM run, source: {samples.name})", fontsize=9)
    else:
        ax.set_title("Drag history (no sample CSV found)", fontsize=9)
    ax.set_xlabel("step in statistics window")
    ax.set_ylabel("Cd")
    ax.legend(fontsize=7)

    ax = axes[1]
    runs = reg["runs"]
    xs = [r["derived"]["blockage"] for r in runs]
    ys = [r["derived"]["cd_mean"] for r in runs]
    ax.plot(xs, ys, "o-", label="measured Cd")
    ext = reg.get("extrapolation") or {}
    if "cd_b0" in ext:
        ax.axhline(ext["cd_b0"], ls="--", color="C1", label=f"B->0 fit: {ext['cd_b0']:.3f}")
    rng = reg["anchors"]["unconfined_re100_cd"]["range"]
    ax.axhspan(
        rng[0], rng[1], color="C2", alpha=0.25, label=f"unconfined anchor {rng[0]:.2f}-{rng[1]:.2f}"
    )
    dfg = reg["anchors"]["dfg2d2_cd"]["range"]
    ax.axhspan(
        dfg[0], dfg[1], color="C3", alpha=0.25, label=f"DFG 2D-2 anchor {dfg[0]:.2f}-{dfg[1]:.2f}"
    )
    ax.set_xlabel("blockage B = D/H")
    ax.set_ylabel("Cd")
    ax.set_title("Blockage trend vs literature anchors", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out, f"({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
