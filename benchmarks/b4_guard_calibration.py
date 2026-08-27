"""B4-P1d · guardrail + UQ calibration over archived v4 predictions.

Sweeps extrapolation-guard thresholds against per-point errors of the
archived LOHO folds (``preds_v4.npz`` of the B4-v4 run) and picks a
default threshold from the flag-coverage / large-error-capture trade-off.
Also reports deep-ensemble UQ quality (PICP of the member min-max band,
Spearman(std, |err|)) per fold and probes the documented manual-feature
blind spot (the bare corner: hull family is not a condition_v3 channel).

Analysis protocol (byte-identical to ``train_fno_v4.py``): splits are
re-derived from ``cache_v4.npz`` with the same param-key grouping and
seeds, so ``fit`` rows here are exactly the rows each fold's model (and
therefore each fold's deployed guard) trained on.

Usage::

    PYTHONPATH=src python benchmarks/b4_guard_calibration.py \
        --run-dir /nfs/wangxi/runs/b4_v4_20260824 \
        --out-dir /nfs/wangxi/runs/b4_serve_20260824

Writes ``guard_calibration.json`` + ``guard_calibration.md`` into
``--out-dir`` (created).  Exits with code 2 ("skip") when the run dir is
absent, so the benchmark is a no-op on hosts without the /nfs artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import tensorlbm
from tensorlbm.ai.drag_cond import condition_v3
from tensorlbm.ai.inference_service import (
    EnvelopeMahalanobisGuardrail,
    choose_threshold,
    ensemble_picp,
    error_std_spearman,
    guard_threshold_sweep,
)

# -- byte-identical split machinery (train_fno_v4.py) ------------------------
HULLS = ("bare_hull", "with_sail", "full")
VAL_SEED = 1
VAL_FRAC = 0.15

FOLDS = ("loho::bare_hull", "loho::with_sail", "loho::full")
ARM = "C_full"
#: relative-error levels counted as "large" for the capture curves
LARGE_ERRORS = (0.02, 0.05, 0.10)
TARGET_CAPTURE = 0.8


def param_keys(d: dict) -> list[str]:
    return [
        f"{r:.6g}|{u:.6g}|{s:.6g}|{f:.6g}|{HULLS[h]}"
        for r, u, s, f, h in zip(d["re"], d["uin"], d["sail"], d["fin"], d["hull"])
    ]


def _groups(d: dict, idx: list[int]) -> dict[str, list[int]]:
    keys = param_keys(d)
    g: dict[str, list[int]] = {}
    for i in idx:
        g.setdefault(keys[i], []).append(i)
    return g


def carve_val(d: dict, train: list[int]) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(VAL_SEED)
    fit, val = [], []
    groups = _groups(d, train)
    for ss in sorted({d["dsi"][min(v)] for v in groups.values()}):
        g = {k: v for k, v in groups.items() if d["dsi"][min(v)] == ss}
        ks = sorted(g)
        rng.shuffle(ks)
        n_val = max(1, int(round(VAL_FRAC * len(ks))))
        for k in ks[:n_val]:
            val += g[k]
        for k in ks[n_val:]:
            fit += g[k]
    return sorted(fit), sorted(val)


def split_loho(d: dict, held: str) -> dict:
    h = HULLS.index(held)
    test = [i for i in range(len(d["cd"])) if d["hull"][i] == h]
    train = [i for i in range(len(d["cd"])) if d["hull"][i] != h]
    fit, val = carve_val(d, sorted(train))
    return {"train": sorted(train), "fit": fit, "val": val, "test": sorted(test)}


def load_run(run_dir: Path) -> dict:
    z = np.load(run_dir / "cache_v4.npz")
    d = {k: z[k] for k in ("dsi", "re", "uin", "sail", "fin", "hull", "cd", "geo", "mask_bit_eq")}
    d["cond"] = condition_v3(d["re"], d["uin"], d["sail"], d["fin"], d["geo"])
    return d


def load_members(preds: Any, fold: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(M, N_test) member predictions + (N_test,) truth for one fold."""
    tags = [""]
    k = 1
    while f"{fold}::{arm}::s{k}::pred" in preds.files:
        tags.append(f"s{k}")
        k += 1
    mat = np.stack(
        [preds[f"{fold}::{arm}::pred" if t == "" else f"{fold}::{arm}::{t}::pred"] for t in tags],
        axis=0,
    ).astype(np.float64)
    truth = np.asarray(preds[f"{fold}::{arm}::true"], dtype=np.float64)
    return mat, truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default="/nfs/wangxi/runs/b4_v4_20260824")
    ap.add_argument("--out-dir", default="/nfs/wangxi/runs/b4_serve_20260824")
    ap.add_argument("--arm", default=ARM)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if not (run_dir / "cache_v4.npz").is_file():
        print(f"[skip] no cache_v4.npz under {run_dir} — calibration needs the archived B4-v4 run")
        return 2

    print(f"tensorlbm: {tensorlbm.__file__}")
    d = load_run(run_dir)
    preds = np.load(run_dir / "preds_v4.npz")
    n = len(d["cd"])
    print(f"corpus: {n} points | cond {d['cond'].shape} | arm {args.arm}")

    folds_report: dict[str, dict[str, Any]] = {}
    sweep_report: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        tensorlbm=tensorlbm.__file__,
        run_dir=str(run_dir),
        arm=args.arm,
        folds=folds_report,
        sweep=sweep_report,
        blind_spot={},
    )
    all_scores, all_errors = [], []
    blind_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        held = fold.split("::")[1]
        split = split_loho(d, held)
        fit_idx = np.asarray(split["fit"])
        te = np.asarray(preds[f"{fold}::{args.arm}::idx"], dtype=np.int64)
        mat, truth = load_members(preds, fold, args.arm)

        guard = EnvelopeMahalanobisGuardrail(d["cond"][fit_idx])
        scores = guard.row_scores(d["cond"][te])
        ens = mat.mean(axis=0)
        rel = np.abs(ens - truth) / truth

        all_scores.append(scores)
        all_errors.append(rel)

        picp = ensemble_picp(mat, truth)
        rho = error_std_spearman(mat.std(axis=0, ddof=1), rel)
        frac_flagged = {
            lev: float(np.mean(scores >= t))
            for lev, t in (
                ("review", guard.review_threshold),
                ("reject", guard.mahal_threshold),
            )
        }
        folds_report[fold] = dict(
            n_fit=int(fit_idx.size),
            n_test=int(te.size),
            n_members=int(mat.shape[0]),
            mape_pct=float(rel.mean() * 100),
            score_q50=float(np.quantile(scores, 0.50)),
            score_q90=float(np.quantile(scores, 0.90)),
            score_max=float(scores.max()),
            frac_flagged=frac_flagged,
            uq=dict(picp=picp, spearman_std_err=rho),
            err_q50=float(np.quantile(rel, 0.50)),
            err_q90=float(np.quantile(rel, 0.90)),
            err_max=float(rel.max()),
        )
        print(
            f"{fold:18s} fit={fit_idx.size:3d} test={te.size:3d} "
            f"members={mat.shape[0]}  MAPE={rel.mean() * 100:5.2f}%  "
            f"score[q50,q90,max]=[{np.quantile(scores, 0.5):.2f}, "
            f"{np.quantile(scores, 0.9):.2f}, {scores.max():.2f}]  "
            f"flagged(rej)={frac_flagged['reject'] * 100:4.1f}%  "
            f"PICP={picp:.2f} rho(std,err)={rho:+.2f}"
        )

        if held == "bare_hull":
            sail = d["sail"][te]
            fin = d["fin"][te]
            bare_mask = d["hull"][te] == HULLS.index("bare_hull")
            bit_eq = np.asarray(d["mask_bit_eq"][te], dtype=bool)
            for label, m in (
                ("bare_test_rows", bare_mask),
                ("mask_bit_eq_rows", bit_eq),
                ("near_bare_sail", (~bare_mask) & (sail <= 0.40) & (fin <= 1.01)),
            ):
                if not m.any():
                    continue
                blind_rows.append(
                    dict(
                        fold=fold,
                        subset=label,
                        n=int(m.sum()),
                        sail_range=[float(sail[m].min()), float(sail[m].max())],
                        in_envelope_frac=float(np.mean(scores[m] < guard.review_threshold)),
                        score_max=float(scores[m].max()),
                        mape_pct=float(rel[m].mean() * 100),
                        err_max=float(rel[m].max()),
                    )
                )

    scores = np.concatenate(all_scores)
    errors = np.concatenate(all_errors)
    for lev in LARGE_ERRORS:
        rows = guard_threshold_sweep(scores, errors, large_error=lev)
        pick = choose_threshold(rows, target_capture=TARGET_CAPTURE)
        sweep_report[f"large_error_{lev:g}"] = dict(
            n=int(scores.size),
            n_large=int((errors >= lev).sum()),
            table=[r.__dict__ for r in rows],
            chosen=dict(
                threshold=pick.threshold,
                capture_rate=pick.capture_rate,
                flagged_frac=pick.flagged_frac,
                precision=pick.precision,
            ),
        )
        print(
            f"sweep @err>={lev:g}: n_large={(errors >= lev).sum()}/{scores.size} "
            f"-> default threshold {pick.threshold:.2f} "
            f"(capture {pick.capture_rate * 100:.0f}%, "
            f"flagged {pick.flagged_frac * 100:.0f}%, "
            f"precision {pick.precision * 100:.0f}%)"
        )

    report["blind_spot"] = dict(
        note="hull family is not a condition_v3 channel: bare-hull rows "
        "carry nominal sail/fin scales, so a held-out hull can sit "
        "inside the manual envelope while the model is biased — the "
        "guard cannot see it by construction (SDF-latent guard is the "
        "planned replacement; see docs/inference_service_20260824.md)",
        rows=blind_rows,
    )
    for r in blind_rows:
        print(
            f"blind-spot {r['fold']}::{r['subset']}: n={r['n']} "
            f"in-envelope={r['in_envelope_frac'] * 100:.0f}% "
            f"MAPE={r['mape_pct']:.2f}% err_max={r['err_max'] * 100:.1f}%"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guard_calibration.json").write_text(json.dumps(report, indent=2))
    write_markdown(out_dir / "guard_calibration.md", report)
    print(f"wrote {out_dir / 'guard_calibration.json'}")
    print(f"wrote {out_dir / 'guard_calibration.md'}")
    return 0


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# B4 guardrail + UQ calibration (offline, archived v4 preds)",
        "",
        f"generated: {report['generated']}  ",
        f"run: `{report['run_dir']}`  arm: `{report['arm']}`",
        "",
        "Guard: `EnvelopeMahalanobisGuardrail` over condition_v3 (8 dims), "
        "fit per-fold on the fold's `fit` rows (byte-identical split "
        "re-derivation), thresholds chi2-calibrated "
        "(review=sqrt(chi2_0.99), reject=sqrt(chi2_0.999)).",
        "",
        "## Per-fold summary",
        "",
        "| fold | fit | test | members | MAPE % | score q50 | q90 | max | "
        "flagged(reject) | PICP | rho(std,err) | err q90 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fold, f in report["folds"].items():
        lines.append(
            f"| {fold} | {f['n_fit']} | {f['n_test']} | {f['n_members']} "
            f"| {f['mape_pct']:.2f} | {f['score_q50']:.2f} | {f['score_q90']:.2f} "
            f"| {f['score_max']:.2f} | {f['frac_flagged']['reject'] * 100:.1f}% "
            f"| {f['uq']['picp']:.2f} | {f['uq']['spearman_std_err']:+.2f} "
            f"| {f['err_q90'] * 100:.2f}% |"
        )
    for key, s in report["sweep"].items():
        lev = key.rsplit("_", 1)[1]
        lines += [
            "",
            f"## Threshold sweep ({key}, n={s['n']}, large={s['n_large']})",
            "",
            "| threshold | flagged | flagged % | large captured | capture % | precision % |",
            "|---|---|---|---|---|---|",
        ]
        for r in s["table"]:
            lines.append(
                f"| {r['threshold']:.2f} | {r['n_flagged']} "
                f"| {r['flagged_frac'] * 100:.1f} | {r['large_captured']} "
                f"| {r['capture_rate'] * 100:.1f} | {r['precision'] * 100:.1f} |"
            )
        c = s["chosen"]
        lines += [
            "",
            f"**default** = {c['threshold']:.2f} "
            f"(capture {c['capture_rate'] * 100:.0f}% of large errors while "
            f"flagging {c['flagged_frac'] * 100:.0f}% of queries, "
            f"precision {c['precision'] * 100:.0f}%)",
        ]
    lines += ["", "## Manual-feature blind spot", "", report["blind_spot"]["note"], ""]
    if report["blind_spot"]["rows"]:
        lines += [
            "| fold | subset | n | sail range | in-envelope % | score max | MAPE % | err max % |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in report["blind_spot"]["rows"]:
            sr = r["sail_range"]
            lines.append(
                f"| {r['fold']} | {r['subset']} | {r['n']} | "
                f"[{sr[0]:.2f}, {sr[1]:.2f}] | {r['in_envelope_frac'] * 100:.0f}% "
                f"| {r['score_max']:.2f} | {r['mape_pct']:.2f} | "
                f"{r['err_max'] * 100:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
