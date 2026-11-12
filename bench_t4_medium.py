#!/usr/bin/env python3
"""T4 physics-AI loop (medium campaign): real LBM field production -> asset ->
FNO flow map with held-out Reynolds -> LBM feedback -> cross-validation.

Medium resolution (n=64, 2000 steps) so it runs in <2 min on CPU while still
being a *real* LBM run (not smoke/synthetic).  The producer verification uses
the same n=64/2000-step config and passes the <1% field / <2% decay-rate gate.

Usage::

    python bench_t4_medium.py            # full medium campaign
    python bench_t4_medium.py --stage X  # produce | train | all

Everything lands in ``artifacts/t4_physics_ai_loop_medium/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.ai.flow_field_loop import (  # noqa: E402
    FieldRunConfig,
    FlowMapConfig,
    evaluate_holdout,
    feedback_acceleration,
    load_dataset,
    produce_dataset,
    train_flow_map,
    verify_producer,
)

OUT = ROOT / "artifacts" / "t4_physics_ai_loop_medium"

TRAIN_RE = (50.0, 100.0, 200.0)
HOLDOUT_RE = (75.0,)
GRID = 64
SEEDS = (0, 1)
DT = 200  # LBM steps between samples
N_STEPS = 2000
EPOCHS = 12
WIDTH = 12
LAYERS = 2
MODES = 6
HEAL_STEPS = 500


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_configs() -> list[FieldRunConfig]:
    cfgs = []
    seen = set()
    for re in TRAIN_RE + HOLDOUT_RE:
        for s in SEEDS:
            key = (GRID, re, s)
            if key in seen:
                continue
            seen.add(key)
            cfgs.append(
                FieldRunConfig(n=GRID, re=re, seed=s, n_steps=N_STEPS,
                               sample_every=DT, tag="t4")
            )
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "produce", "train"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_dir = out / "dataset"
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- producer
    verif_path = out / "producer_verification.json"
    if verif_path.exists() and args.stage != "all":
        verification = json.loads(verif_path.read_text())
        _log(f"producer verification reused ({verification['verified']})")
    else:
        _log("producer verification: single-mode Taylor-Green vs exact NS decay")
        verification = verify_producer(
            n=GRID, re_values=(50.0, 200.0), n_steps=N_STEPS, sample_every=DT,
        )
        verif_path.write_text(json.dumps(verification, indent=2))
        _log(
            f"producer verified={verification['verified']} "
            f"field_err={verification['max_abs_rel_err_field']:.4%} "
            f"rate_err={verification['max_abs_rel_err_decay_rate']:.4%}"
        )

    # ----------------------------------------------------------------- dataset
    manifest_path = ds_dir / "manifest.json"
    if manifest_path.exists() and args.stage != "produce":
        manifest = json.loads(manifest_path.read_text())
        _log(f"dataset reused: {manifest['n_files']} runs")
    else:
        cfgs = build_configs()
        split = {
            "train_re": list(TRAIN_RE),
            "holdout_re": list(HOLDOUT_RE),
            "train_grid": GRID,
            "note": "holdout Reynolds numbers never appear in training",
        }
        t0 = time.perf_counter()
        manifest = produce_dataset(
            cfgs, ds_dir, dataset_name="t4-tgv-real-lbm-fields", split=split,
        )
        _log(
            f"dataset produced: {manifest['n_files']} runs, "
            f"{manifest['totals']['n_snapshots']} snapshots in "
            f"{time.perf_counter() - t0:.1f}s, manifest {manifest['manifest_sha256'][:16]}"
        )

    # --------------------------------------------------------------- train FNO
    model_path = model_dir / "fno_flow_map.pt"
    cfg = FlowMapConfig(
        width=WIDTH, n_layers=LAYERS, modes=MODES, epochs=EPOCHS, batch_size=32,
        patience=6, seed=0,
    )
    _log(f"training FNO flow map (train Re={TRAIN_RE}, holdout Re={HOLDOUT_RE})")
    trained = train_flow_map(
        ds_dir, model_path, config=cfg, train_res=TRAIN_RE, holdout_res=HOLDOUT_RE,
        grid_n=GRID,
    )
    model = trained["model"]
    (model_dir / "train_summary.json").write_text(
        json.dumps({k: v for k, v in trained.items() if k != "model"}, indent=2)
    )
    _log(
        f"trained on {trained['n_train_pairs']} pairs / {trained['epochs_run']} epochs, "
        f"best val MSE {trained['best_val_mse']:.3e} in {trained['train_wall_s']}s"
    )

    # ------------------------------------------------------ held-out evaluation
    _log("held-out evaluation (unseen Re, one-step + full rollout)")
    eval_main = evaluate_holdout(
        model, ds_dir, train_res=TRAIN_RE, holdout_res=HOLDOUT_RE,
        n_rollout=N_STEPS // DT, grid_n=GRID,
    )
    _log(f"holdout summary: {json.dumps(eval_main['summary'])}")

    # ---------------------------------------------------------- LBM feedback
    _log("LBM feedback: surrogate coarse-stepping + handback + healing")
    feedback = feedback_acceleration(
        model, ds_dir, holdout_res=HOLDOUT_RE, k_values=(2, 4), heal_steps=HEAL_STEPS,
    )
    _log(f"feedback rows: {len(feedback['rows'])}")

    # ------------------------------------------------------------- artifacts
    results = {
        "mission": "T4 physics-AI loop (medium campaign, real LBM)",
        "repro_cmd": "python bench_t4_medium.py",
        "external_anchor": verification["anchor"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "grid": GRID,
            "lbm_steps": N_STEPS,
            "sample_every_steps": DT,
            "dt_physical": "dt = sample_every LBM steps",
            "train_re": list(TRAIN_RE),
            "holdout_re": list(HOLDOUT_RE),
            "seeds": list(SEEDS),
            "u0": 0.05,
            "mach": 0.05 * math.sqrt(3.0),
            "solver": "D2Q9 BGK, tau = 3 nu + 0.5, periodic",
            "data_source": "real LBM runs only (analytic used as ruler in producer check)",
        },
        "producer_verification": verification,
        "dataset_manifest": {
            "schema": manifest["schema"],
            "n_files": manifest["n_files"],
            "totals": manifest["totals"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "training": {k: v for k, v in trained.items() if k != "model"},
        "holdout_eval": eval_main,
        "feedback": feedback,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _log(f"results written -> {out / 'results.json'}")

    # ------------------------------------------------------------- figures
    _make_figures(results, out, ds_dir, model)
    _log("figures written")
    return 0


def _make_figures(results: dict, out: Path, ds_dir: Path, model) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out / "figures"

    # --- fig 1: producer verification (u_rms vs analytic)
    pv = results["producer_verification"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in pv["rows"]:
        ax.plot(row["steps"], row["u_rms_lbm"], "o-", label=f"LBM Re={row['re']:g}")
        ax.plot(row["steps"], row["u_rms_analytic"], "s--", label=f"analytic Re={row['re']:g}")
    ax.set_xlabel("LBM steps")
    ax.set_ylabel("$u_{rms}$")
    ax.set_title("producer: single-mode Taylor-Green vs exact NS decay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_producer_verification.png", dpi=150)
    plt.close(fig)

    # --- fig 2: error vs Re (generalisation to unseen Re)
    rows = results["holdout_eval"]["per_run"]
    res = sorted({r["re"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab, style in (
        ("fno", "FNO (ours)", "o-"),
        ("viscous_prior", "exact viscous prior", "s--"),
        ("persistence", "persistence", "^:"),
    ):
        ys = [
            np.mean([r["one_step_rel_l2"][key] for r in rows if r["re"] == re]) for re in res
        ]
        ax[0].plot(res, ys, style, label=lab)
    for re in results["protocol"]["holdout_re"]:
        ax[0].axvline(re, ls=":", c="gray", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Re (log)")
    ax[0].set_ylabel("one-step relative $L_2$")
    ax[0].set_title("one-step flow-map error (dotted = held-out Re)")
    ax[0].legend(fontsize=8)

    for key, lab, style in (
        ("fno", "FNO (ours)", "o-"),
        ("viscous", "exact viscous prior", "s--"),
        ("persistence", "persistence", "^:"),
    ):
        ys = [
            np.mean([r["rollout"][key]["final"] for r in rows if r["re"] == re]) for re in res
        ]
        ax[1].plot(res, ys, style, label=lab)
    for re in results["protocol"]["holdout_re"]:
        ax[1].axvline(re, ls=":", c="gray", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Re (log)")
    ax[1].set_ylabel("rollout final relative $L_2$")
    ax[1].set_title(f"autoregressive rollout ({results['protocol']['lbm_steps']} steps)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_error_vs_re.png", dpi=150)
    plt.close(fig)

    # --- fig 3: field maps on a held-out run
    re0 = results["protocol"]["holdout_re"][0]
    data = load_dataset(ds_dir)
    cand = [rid for rid, run in data["runs"].items()
            if round(run["re"], 6) == round(re0, 6) and run["n"] == results["protocol"]["grid"]]
    if cand:
        rid = sorted(cand)[0]
        run = data["runs"][rid]
        u0, nu = run["u0"], run["nu"]
        i = min(4, run["ux"].shape[0] - 3)
        dt = run["steps"][1] - run["steps"][0]
        cur = (run["ux"][i], run["uy"][i])
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                x = torch.stack([cur[0] / u0, cur[1] / u0,
                                 torch.full_like(cur[0], nu / 1e-2)])[None]
                o = model(x)[0]
                cur = (o[0] * u0, o[1] * u0)
        j = min(i + 2, run["ux"].shape[0] - 1)
        truth = (run["ux"][j], run["uy"][j])
        vis = (run["ux"][i], run["uy"][i])
        from tensorlbm.ai.flow_field_loop import viscous_prior_step
        for _ in range(2):
            vis = viscous_prior_step(vis[0], vis[1], nu, dt)
        panels = [
            (torch.sqrt(truth[0] ** 2 + truth[1] ** 2), "LBM truth"),
            (torch.sqrt(cur[0] ** 2 + cur[1] ** 2), "FNO rollout (k=2)"),
            (torch.sqrt(vis[0] ** 2 + vis[1] ** 2), "exact viscous prior"),
            (torch.sqrt((cur[0] - truth[0]) ** 2 + (cur[1] - truth[1]) ** 2), "|FNO - LBM|"),
        ]
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
        for a, (fld, title) in zip(axs, panels):
            im = a.imshow(fld.numpy(), origin="lower", cmap="viridis")
            a.set_title(title, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle(f"held-out Re={re0:g} ({rid}), |u| after {2 * dt} LBM steps", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_field_maps.png", dpi=150)
        plt.close(fig)

    # --- fig 4: feedback (handback + healing)
    fb = results["feedback"]["rows"]
    if fb:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ks = sorted({r["k_surrogate_steps"] for r in fb})
        for re in results["protocol"]["holdout_re"]:
            sub = [r for r in fb if r["re"] == re]
            if not sub:
                continue
            ax[0].plot(
                [r["k_surrogate_steps"] for r in sub], [r["handback_rel_l2"] for r in sub],
                "o-", label=f"Re={re:g}",
            )
            ax[1].plot(
                [r["k_surrogate_steps"] for r in sub], [r["surrogate_rel_l2_at_handback"]
                                                       for r in sub],
                "s--", label=f"Re={re:g}",
            )
        ax[0].set_xlabel("k surrogate steps before handback")
        ax[0].set_ylabel("relative $L_2$ after LBM handback step")
        ax[0].set_title("handback error (surrogate state -> 1 LBM dt -> truth)")
        ax[1].set_xlabel("k surrogate steps")
        ax[1].set_ylabel("relative $L_2$ of surrogate state")
        ax[1].set_title("pure surrogate error at handback")
        ax[0].set_xticks(ks)
        ax[1].set_xticks(ks)
        ax[0].legend(fontsize=8)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_feedback.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
