#!/usr/bin/env python3
"""T4 physics-AI loop driver: real LBM field production -> dataset asset ->
FNO flow map with held-out Reynolds numbers -> LBM feedback -> cross-validation.

Usage (CPU, ~10 min)::

    python bench_t4_physics_ai_loop.py                # full campaign
    python bench_t4_physics_ai_loop.py --quick        # 1/8-size smoke run
    python bench_t4_physics_ai_loop.py --stage train  # reuse the dataset

Everything lands in ``artifacts/t4_physics_ai_loop/``:

    producer_verification.json   single-mode Taylor-Green vs exact NS decay
    dataset/                     npz fields + manifest.json (sha256) + catalog.sqlite
    model/fno_flow_map.pt        trained FNO flow map (+ arch json)
    results.json                 every number quoted in the report
    figures/*.png                field maps, error curves, sensitivity
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
    FlowMapConfig,
    FieldRunConfig,
    _rel_l2,
    evaluate_holdout,
    feedback_acceleration,
    load_dataset,
    produce_dataset,
    run_case,
    train_flow_map,
    verify_producer,
    viscous_prior_step,
)

OUT = ROOT / "artifacts" / "t4_physics_ai_loop"

TRAIN_RE = (50.0, 100.0, 200.0, 400.0)
HOLDOUT_RE = (75.0, 300.0)
GRID = 128
GRID_COARSE = 64
SEEDS = (0, 1, 2)
DT = 250  # LBM steps between samples (the flow-map dt)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_configs(*, n_main: int, n_coarse: int, n_steps: int, dt: int, seeds,
                  coarse_seeds) -> list[FieldRunConfig]:
    """Production matrix: (Re, seed) x {main grid, coarse grid}, no duplicates."""
    cfgs: list[FieldRunConfig] = []
    seen: set[tuple[int, float, int]] = set()
    for n, seed_list in ((n_main, seeds), (n_coarse, coarse_seeds)):
        for re in TRAIN_RE + HOLDOUT_RE:
            for s in seed_list:
                key = (n, re, s)
                if key in seen:
                    continue
                seen.add(key)
                cfgs.append(
                    FieldRunConfig(n=n, re=re, seed=s, n_steps=n_steps, sample_every=dt,
                                   tag="t4")
                )
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke campaign")
    ap.add_argument("--stage", default="all", choices=("all", "produce", "train"))
    ap.add_argument("--threads", type=int, default=8)
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

    if args.quick:
        grid, grid_coarse, n_steps, dt = 32, 32, 500, 250
        seeds, coarse_seeds = (0,), (0,)
        train_re, holdout_re = (50.0, 100.0), (75.0,)
        epochs, width, layers, modes = 8, 16, 2, 6
        heal_steps = 500
    else:
        grid, grid_coarse, n_steps, dt = GRID, GRID_COARSE, 5000, DT
        seeds, coarse_seeds = SEEDS, (0,)
        train_re, holdout_re = TRAIN_RE, HOLDOUT_RE
        epochs, width, layers, modes = 12, 12, 2, 6
        heal_steps = 500
    del grid, grid_coarse

    grid_n = (32 if args.quick else 128)
    n_coarse = (32 if args.quick else 64)

    # ---------------------------------------------------------------- producer
    verif_path = out / "producer_verification.json"
    if verif_path.exists() and args.stage != "all":
        verification = json.loads(verif_path.read_text())
        _log(f"producer verification reused ({verification['verified']})")
    else:
        _log("producer verification: single-mode Taylor-Green vs exact NS decay")
        verification = verify_producer(
            n=min(128, 128 if not args.quick else 64),
            re_values=(50.0, 200.0),
            n_steps=2000 if not args.quick else 400,
            sample_every=200 if not args.quick else 100,
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
        cfgs = build_configs(
            n_main=grid_n, n_coarse=(grid_n if args.quick else GRID_COARSE),
            n_steps=n_steps, dt=dt, seeds=seeds, coarse_seeds=coarse_seeds,
        )
        split = {
            "train_re": list(train_re),
            "holdout_re": list(holdout_re),
            "train_grid": grid_n,
            "coarse_grid": n_coarse,
            "note": "holdout Reynolds numbers never appear in training",
        }
        t0 = time.perf_counter()
        manifest = produce_dataset(
            cfgs,
            ds_dir,
            dataset_name="t4-tgv-real-lbm-fields",
            split=split,
        )
        _log(
            f"dataset produced: {manifest['n_files']} runs, "
            f"{manifest['totals']['n_snapshots']} snapshots in "
            f"{time.perf_counter() - t0:.1f}s, manifest {manifest['manifest_sha256'][:16]}"
        )

    # --------------------------------------------------------------- train FNO
    model_path = model_dir / "fno_flow_map.pt"
    cfg = FlowMapConfig(
        width=width, n_layers=layers, modes=modes, epochs=epochs, batch_size=32,
        patience=6, seed=0,
    )
    _log(f"training FNO flow map (train Re={train_re}, holdout Re={holdout_re})")
    trained = train_flow_map(
        ds_dir, model_path, config=cfg, train_res=train_re, holdout_res=holdout_re,
        grid_n=grid_n,
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
        model, ds_dir, train_res=train_re, holdout_res=holdout_re,
        n_rollout=n_steps // dt, grid_n=grid_n,
    )
    _log(f"holdout summary: {json.dumps(eval_main['summary'])}")

    eval_coarse = None
    if not args.quick:
        # zero-shot cross-resolution: the same model on the 64^2 holdout runs
        _log(f"cross-resolution check: same model on {n_coarse}^2 fields")
        eval_coarse = evaluate_holdout(
            model, ds_dir, train_res=train_re, holdout_res=holdout_re,
            n_rollout=n_steps // dt, grid_n=n_coarse,
        )
        _log(f"64^2 summary: {json.dumps(eval_coarse['summary'])}")

    # ---------------------------------------------------------- LBM feedback
    _log("LBM feedback: surrogate coarse-stepping + handback + healing")
    feedback = feedback_acceleration(
        model, ds_dir, holdout_res=holdout_re, k_values=(2, 4), heal_steps=heal_steps,
    )
    _log(f"feedback rows: {len(feedback['rows'])}")

    # ------------------------------------------------------------- artifacts
    results = {
        "mission": "T4 physics-AI loop (real LBM field production -> asset -> FNO -> feedback)",
        "repro_cmd": (
            "cd TensorLBM && python bench_t4_physics_ai_loop.py"
            + ("  # or --quick" if args.quick else "")
        ),
        "external_anchor": verification["anchor"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "grid": grid_n,
            "grid_coarse": None if args.quick else 64,
            "lbm_steps": n_steps,
            "sample_every_steps": dt,
            "dt_physical": "dt = sample_every LBM steps",
            "train_re": list(train_re),
            "holdout_re": list(holdout_re),
            "seeds": list(seeds),
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
        "holdout_eval_128": eval_main,
        "holdout_eval_64": eval_coarse,
        "feedback": feedback,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _log(f"results written -> {out / 'results.json'}")

    make_figures(results, out, ds_dir, model)
    _log("figures written")
    return 0


def make_figures(results: dict, out: Path, ds_dir: Path, model) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out / "figures"

    # --- fig 1: producer verification
    pv = results["producer_verification"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for row in pv["rows"]:
        ax[0].plot(row["steps"], row["u_rms_analytic"], "-",
                   label=f"analytic Re={row['re']:g}")
        ax[0].plot(row["steps"], row["u_rms_lbm"], "o", ms=4,
                   label=f"LBM Re={row['re']:g}")
        ax[1].plot(row["steps"], [abs(e) * 100 for e in row["rel_err"]], "o-", ms=4,
                   label=f"Re={row['re']:g}")
    ax[0].set_xlabel("LBM steps")
    ax[0].set_ylabel(r"$u_{\rm rms}$ (lu)")
    ax[0].set_title("single-mode Taylor-Green decay: LBM vs exact NS")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("LBM steps")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title(r"$u_{\rm rms}(t)$ error (threshold 1%)")
    ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_producer_verification.png", dpi=150)
    plt.close(fig)

    # --- fig 2: error vs Re (generalisation to unseen Re)
    rows = results["holdout_eval_128"]["per_run"]
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

    # --- fig 3: rollout curves on a held-out Re
    data = load_dataset(ds_dir)
    for re in results["protocol"]["holdout_re"]:
        rr = [r for r in rows if r["re"] == re][:1]
        if not rr:
            continue
        r0 = rr[0]
        steps = np.arange(1, len(r0["rollout"]["fno"]["rel_l2_series"]) + 1) * r0["dt_steps"]
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for key, lab, style in (
            ("fno", "FNO (ours)", "o-"),
            ("viscous", "exact viscous prior", "s--"),
            ("persistence", "persistence", "^:"),
        ):
            ax.plot(steps, r0["rollout"][key]["rel_l2_series"], style, label=lab)
        ax.set_xlabel("LBM steps")
        ax.set_ylabel("relative $L_2$ vs LBM")
        ax.set_title(f"rollout on held-out Re={re:g} (n={r0['run_id']})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig3_rollout_re{re:g}.png", dpi=150)
        plt.close(fig)

    # --- fig 4: field maps + error on a held-out run
    re0 = results["protocol"]["holdout_re"][0]
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
            for _ in range(2):  # two surrogate steps = the k=2 feedback path
                x = torch.stack([cur[0] / u0, cur[1] / u0,
                                 torch.full_like(cur[0], nu / 1e-2)])[None]
                o = model(x)[0]
                cur = (o[0] * u0, o[1] * u0)
        j = min(i + 2, run["ux"].shape[0] - 1)
        truth = (run["ux"][j], run["uy"][j])
        vis = (run["ux"][i], run["uy"][i])
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
        fig.savefig(fig_dir / "fig4_field_maps.png", dpi=150)
        plt.close(fig)

    # --- fig 5: feedback (handback + healing)
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
        fig.savefig(fig_dir / "fig5_feedback.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())