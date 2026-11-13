"""T2 core path — precision-tier admissibility sweep (measured, not assumed).

Why this benchmark exists
-------------------------
The T2 report advertised four compute/store precision tiers and claimed the
``FP32FP16`` store tier costs "~2e-4 relative error" while halving the
bandwidth.  Round-2 audit (``docs/audits/m15939``, finding F4) showed the
store dtype never entered the marching loop, so ``FP32FP16`` was bit-identical
to ``FP32FP32``: the tier was declared but never exercised.  The store tier is
now wired into :class:`tensorlbm.gpu_core_path.GPUCorePathSolver` (cast at step
entry/exit), which makes the claim *measurable* — and this script measures it.

Decision this experiment supports
---------------------------------
Which precision tiers may be used for production runs of the core path, and
which must carry a warning.  Alternatives: (A) keep every tier silently
available, (B) drop the float16 tiers, (C) keep them, wired and measured, with
the measured verdict documented and a construction-time warning for lossy
tiers.  We test (C); (B) is the fallback if a lossy tier turns out unusable
even as a screening mode.

Pre-declared judgement (fixed before looking at any result)
----------------------------------------------------------
For every tier and every anchor case::

    PASS      worst relative error on the anchor  <= 3%     (platform line)
    MARGINAL  3% <  worst relative error  <= 15%           (screening only)
    REJECT    worst relative error > 15%                    (unacceptable)

Anchors (closed-form, no fitted freedom):

* body-force channel: ``u(y) = a/(2 nu) y (H - y)`` — metrics ``u_max_rel_err``
  and ``profile_rel_l2``;
* Taylor-Green vortex: ``KE(t) = u0^2/4 exp(-4 nu k^2 t)`` — metric
  ``max_rel_err`` over the sampled instants.

The tier's verdict is the worst verdict over its metrics; the control arm is
``FP64FP64`` (same geometry, same seed, ``perturbation_amplitude=0``), so every
tier differs from the control in exactly one variable: the compute/store dtype
pair.  Runs are deterministic (no perturbation, fixed seed), so one run per
cell suffices; timing is reported as a single run with the thread count pinned
(labelled as such, not as a performance claim).

Usage
-----
::

    PYTHONPATH=src python benchmarks/bench_gpu_core_path_precision.py \
        --tiers FP64FP64 FP64FP32 FP32FP32 FP32FP16 \
        --out artifacts/gpu_core_path_precision/results.json
    python benchmarks/bench_gpu_core_path_precision.py --plot \
        --out artifacts/gpu_core_path_precision/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Deterministic, single-threaded execution: the point of the sweep is the
# arithmetic, and thread scheduling must not enter the comparison.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tensorlbm.gpu_core_path import (  # noqa: E402
    GPUCorePathConfig,
    GPUCorePathSolver,
    run_channel_case,
    run_taylor_green_case,
)

# --- pre-declared thresholds (do not tune after seeing results) -------------
PASS_REL_ERR = 0.03
REJECT_REL_ERR = 0.15
TIERS = ("FP64FP64", "FP64FP32", "FP32FP32", "FP32FP16")

#: Channel configurations: (label, ny, steps).  ``ny=34`` is the resolution
#: check and only runs for the control and the most aggressive tier.
CHANNEL_CASES = (("channel_ny18", 18, 3000), ("channel_ny34", 34, 6000))
CHANNEL_TIERS = {
    "channel_ny18": TIERS,
    "channel_ny34": ("FP64FP64", "FP32FP16"),
}
#: Taylor–Green configurations: (label, nx, steps, u0).
TG_CASES = (("tg_nx32", 32, 400, 0.02), ("tg_nx64", 64, 800, 0.02))


def classify(rel_err: float) -> str:
    """Verdict for one metric under the pre-declared thresholds."""
    if rel_err <= PASS_REL_ERR:
        return "PASS"
    if rel_err <= REJECT_REL_ERR:
        return "MARGINAL"
    return "REJECT"


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "threads": torch.get_num_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "backend": "eager",
        "deterministic": True,
    }


def run_tier(tier: str) -> dict[str, Any]:
    """All anchor cases for one precision tier."""
    out: dict[str, Any] = {"tier": tier, "channel": {}, "taylor_green": {}}

    for label, ny, steps in CHANNEL_CASES:
        if tier not in CHANNEL_TIERS[label]:
            continue
        cfg = GPUCorePathConfig(
            nx=32,
            ny=ny,
            nz=8,
            tau=0.8,
            u_max_target=0.05,
            steps=steps,
            precision=tier,
            backend="eager",
            allow_lossy_store=True,   # the sweep is *about* the lossy tier
            label=f"{label}_{tier}",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = run_channel_case(cfg, check_every=500, steady_tol=2e-5)
        row["warnings"] = [str(w.message) for w in caught]
        out["channel"][label] = row

    for label, nx, steps, u0 in TG_CASES:
        cfg = GPUCorePathConfig(
            nx=nx,
            ny=nx,
            nz=4,
            tau=0.6,
            init="taylor_green",
            tg_u0=u0,
            steps=steps,
            precision=tier,
            backend="eager",
            allow_lossy_store=True,   # the sweep is *about* the lossy tier
            label=f"{label}_{tier}",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = run_taylor_green_case(cfg, samples=4)
        row["warnings"] = [str(w.message) for w in caught]
        out["taylor_green"][label] = row

    solver = GPUCorePathSolver(
        GPUCorePathConfig(ny=18, steps=1, precision=tier, backend="eager",
                          allow_lossy_store=True)
    )
    out["memory"] = solver.memory_report()
    return out


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn raw rows into the verdict table (one line per tier x metric)."""
    rows: list[dict[str, Any]] = []
    for res in results:
        tier = res["tier"]
        for label, row in res["channel"].items():
            for metric in ("u_max_rel_err", "profile_rel_l2"):
                val = float(row[metric])
                rows.append(
                    {
                        "tier": tier,
                        "case": label,
                        "metric": metric,
                        "value": val,
                        "verdict": classify(val),
                        "steps": row["steps"],
                        "steady_at_step": row.get("steady_at_step"),
                    }
                )
        for label, row in res["taylor_green"].items():
            val = float(row["max_rel_err"])
            rows.append(
                {
                    "tier": tier,
                    "case": label,
                    "metric": "ke_max_rel_err",
                    "value": val,
                    "verdict": classify(val),
                    "steps": row["steps"],
                    "steady_at_step": None,
                }
            )
    order = {"PASS": 0, "MARGINAL": 1, "REJECT": 2}
    by_tier: dict[str, dict[str, Any]] = {}
    for row in rows:
        slot = by_tier.setdefault(
            row["tier"], {"worst": 0.0, "verdict": "PASS", "cases": 0}
        )
        slot["worst"] = max(slot["worst"], row["value"])
        if order[row["verdict"]] > order[slot["verdict"]]:
            slot["verdict"] = row["verdict"]
        slot["cases"] += 1
    admissible = sorted(t for t, v in by_tier.items() if v["verdict"] == "PASS")
    return {
        "rows": rows,
        "by_tier": by_tier,
        "admissible_tiers": admissible,
        "thresholds": {"pass": PASS_REL_ERR, "reject": REJECT_REL_ERR},
    }


def markdown(payload: dict[str, Any]) -> str:
    thr = payload["summary"]["thresholds"]
    lines = [
        "# T2 core path — precision tier verdicts",
        "",
        f"Pre-declared: PASS <= {100 * thr['pass']:.0f}%, "
        f"MARGINAL <= {100 * thr['reject']:.0f}%, REJECT above. "
        "Control arm = FP64FP64.",
        "",
        "| tier | case | metric | rel. error | verdict |",
        "|---|---|---|---|---|",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['tier']} | {row['case']} | {row['metric']} | "
            f"{100 * row['value']:.4f}% | {row['verdict']} |"
        )
    lines += ["", "## Per-tier verdict (worst metric)", "", "| tier | worst | verdict | memory (B/cell) |", "|---|---|---|---|"]
    for res in payload["results"]:
        tier = res["tier"]
        slot = payload["summary"]["by_tier"][tier]
        mem = res["memory"]
        lines.append(
            f"| {tier} | {100 * slot['worst']:.4f}% | {slot['verdict']} | "
            f"state {mem['persistent_state_bytes_per_cell']:.0f} / "
            f"total {mem['total_bytes_per_cell']:.0f} |"
        )
    lines.append("")
    lines.append(
        "Admissible (PASS on every anchor): "
        + (", ".join(payload["summary"]["admissible_tiers"]) or "none")
    )
    return "\n".join(lines) + "\n"


def plot(payload: dict[str, Any], outdir: Path) -> list[str]:
    """Comparison curves: channel profile per tier vs analytic, TGV decay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ref_drawn = False
    for res in payload["results"]:
        row = res["channel"].get("channel_ny18")
        if row is None:
            continue
        ax.plot(row["u_got"], row["y"], marker="o", ms=3, lw=1.2,
                label=f"{res['tier']} ({100 * row['profile_rel_l2']:.2f}% L2)")
        if not ref_drawn:
            ax.plot(row["u_ref"], row["y"], "k--", lw=1.2, label="analytic")
            ref_drawn = True
    ax.set_xlabel("u_x (lattice)")
    ax.set_ylabel("y (lattice)")
    ax.set_title("Poiseuille channel, ny=18, 3000 steps")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for res in payload["results"]:
        row = res["taylor_green"].get("tg_nx32")
        if row is None:
            continue
        steps = [s["step"] for s in row["samples"]]
        ke = [s["ke_got"] / s["ke_ref"] for s in row["samples"]]
        ax.plot(steps, ke, marker="o", ms=3, lw=1.2,
                label=f"{res['tier']} (max {100 * row['max_rel_err']:.2f}%)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="exact decay")
    ax.set_xlabel("step")
    ax.set_ylabel("KE(t) / KE_exact(t)")
    ax.set_title("Taylor-Green decay, nx=32")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "precision_tiers.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path))

    # Field snapshot: streamwise velocity of the channel mid-plane per tier.
    fig, axes = plt.subplots(1, len(payload["results"]), figsize=(3.1 * len(payload["results"]), 2.6),
                             squeeze=False)
    for ax, res in zip(axes[0], payload["results"]):
        row = res["channel"].get("channel_ny18")
        if row is None:
            ax.set_axis_off()
            continue
        prof = torch.tensor(row["u_got"]).view(1, -1).repeat(24, 1)
        im = ax.imshow(prof, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(res["tier"], fontsize=8)
        ax.set_xlabel("y", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("u_x(y) field rebuilt from the measured profile (ny=18)", fontsize=9)
    fig.tight_layout()
    path = outdir / "precision_fields.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    ap.add_argument("--out", default="artifacts/gpu_core_path_precision/results.json")
    ap.add_argument("--plot", action="store_true", help="render figures from an existing --out")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge per-tier result files (one --tiers run each) into --out")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if args.merge:
        results = []
        envs = []
        for path in args.merge:
            part = json.loads(Path(path).read_text())
            results.extend(part["results"])
            envs.append(part["environment"])
        order = {t: i for i, t in enumerate(TIERS)}
        results.sort(key=lambda r: order.get(r["tier"], 99))
        payload = {
            "environment": envs[0],
            "environments_per_arm": envs,
            "thresholds": {"pass": PASS_REL_ERR, "reject": REJECT_REL_ERR},
            "results": results,
            "summary": summarise(results),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        md = markdown(payload)
        (out_path.parent / "tiers.md").write_text(md)
        print(md)
        return 0
    if args.plot:
        payload = json.loads(out_path.read_text())
        figs = plot(payload, out_path.parent)
        print("figures:", *figs, sep="\n  ")
        return 0

    torch.set_num_threads(1)
    results = []
    for tier in args.tiers:
        t0 = time.perf_counter()
        print(f"[tier {tier}] running anchors ...", flush=True)
        results.append(run_tier(tier))
        print(f"[tier {tier}] done in {time.perf_counter() - t0:.1f}s", flush=True)

    payload = {
        "environment": environment(),
        "thresholds": {"pass": PASS_REL_ERR, "reject": REJECT_REL_ERR},
        "results": results,
        "summary": summarise(results),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    md = markdown(payload)
    (out_path.parent / "tiers.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())