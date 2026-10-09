"""Closure calibration against a *real* campaign drag dataset (B3-real).

The verification example (:file:`examples/closure_calibration.py`) recovers
a closure from synthetic observations produced by the same solver.  This
example upgrades every ingredient to the real thing:

* **geometry** — :class:`tensorlbm.autograd_calib.HullCase` builds the
  *production* SUBOFF bare-hull voxel mask (``suboff_n128``: 64x64x128,
  hull centred at ``cx = 0.35 nx``, length ``0.6 nx``, 4093 solid cells),
  keeps the production tau relation ``tau = 0.5 + 23.04/Re`` at
  ``u_in = 0.1`` and closes the lateral planes with free-stream faces, the
  campaign far-field condition;
* **observations** — :func:`tensorlbm.autograd_calib.drag_targets_from_sidecars`
  reduces ``drag_history.json`` sidecars (``tensorlbm.drag-history/v1``,
  exact discrete-kinetic control-volume force) to ``C_D = 2F/(rho u^2 S_proj)``
  with ``S_proj = 69`` cells^2 — the projected area of the very mask that is
  simulated, so calibration and data sides normalise identically;
* **rollout length** — decided from the data, not guessed: the campaign
  histories show windowed C_D at ``[1000, 1200)`` within 0.3-0.5% of the
  4000-step tail mean across the whole Re sweep, so the calibration rolls
  out 1200 steps and samples the last 200.

Training Re stay in the identifiable band of the B3 sphere study
(tau <= 0.58 <=> Re >= 288 at u_in = 0.1): {305, 437.8, 800}; held-out
{628.6, 148} — the second one deliberately in the non-identifiable regime.

Two references frame the result:

* a pure power law ``C_D = a Re^b`` fitted to the same training points
  (the zero-solver physical baseline);
* the C_s -> 0 BGK limit of the calibration path itself (the best the
  Smagorinsky family can do: any C_s > 0 only *adds* drag).

Usage
-----
    python examples/closure_calibration_real.py --dataset DIR     # full run
    python examples/closure_calibration_real.py --device cuda
    python examples/closure_calibration_real.py --quick           # no fits
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.autograd_calib import (
    DragTarget,
    HullCase,
    bounded_drag,
    calibrate,
    drag_targets_from_sidecars,
    evaluate,
    load_drag_history,
    windowed_cd,
)

#: Campaign dataset (2026-08-21): 24 log-spaced Re 50..800, exact drag sidecars.
DEFAULT_DATASET = "/nfs/wangxi/datasets/scan_suboff_re_drag_20260821"

#: Identifiable band (B3): tau = 0.5 + 23.04/Re <= 0.58 <=> Re >= 288.
TRAIN_RE = (305.0, 437.8, 800.0)
HELDOUT_RE = (628.6, 148.0)

#: Window-convergence probe offsets (steps) against the 4000-step tail.
CONV_WINDOWS = ((400, 500), (700, 800), (1000, 1200), (2000, 2400))


def load_targets(dataset: str, u_in: float, ref_area: float) -> list[DragTarget]:
    paths = sorted(glob.glob(f"{dataset}/points/p*/drag_history.json"))
    if not paths:
        raise SystemExit(f"no drag_history.json sidecars under {dataset}/points/")
    return drag_targets_from_sidecars(paths, u_in=u_in, ref_area=ref_area)


def convergence_table(dataset: str, u_in: float, ref_area: float) -> list[dict]:
    """How fast does windowed C_D converge?  (rollout-length justification)"""
    rows = []
    for path in sorted(glob.glob(f"{dataset}/points/p*/drag_history.json")):
        status = json.loads(Path(path).with_name("status.json").read_text())
        history = load_drag_history(path)
        tail = windowed_cd(
            history, int(history.steps[-1] * 0.75), int(history.steps[-1]) + 1, u_in, ref_area
        )
        row = {"re": float(status["params"]["re"]), "cd_tail": tail, "n": len(history.steps)}
        for lo, hi in CONV_WINDOWS:
            row[f"dev_{lo}"] = 100.0 * (windowed_cd(history, lo, hi, u_in, ref_area) / tail - 1.0)
        rows.append(row)
    rows.sort(key=lambda r: r["re"])
    print("C_D window convergence vs the 4000-step tail mean (deviation, %):")
    print("  Re      tail C_D  " + "  ".join(f"[{lo},{hi})" for lo, hi in CONV_WINDOWS))
    for r in rows[::4]:  # every 4th point keeps the table readable
        print(
            f"  {r['re']:6.1f}  {r['cd_tail']:9.4f}  "
            + "  ".join(f"{r[f'dev_{lo}']:+8.2f}%" for lo, _ in CONV_WINDOWS)
        )
    early = max(abs(r[f"dev_{lo}"]) for r in rows for lo, _ in CONV_WINDOWS[:2])
    late = max(abs(r["dev_1000"]) for r in rows)
    print(f"  worst deviation: early windows {early:.2f}%  vs [1000,1200) {late:.2f}%")
    return rows


def powerlaw_baseline(train: list[DragTarget], heldout: list[DragTarget]) -> dict:
    """C_D = a Re^b by log-log least squares on the training points only."""
    x = np.log10([t.re for t in train])
    y = np.log10([t.cd for t in train])
    b, log_a = np.polyfit(x, y, 1)
    a = 10.0**log_a

    def pred(re: float) -> float:
        return float(a * re**b)

    rows = [
        {
            "re": t.re,
            "target": t.cd,
            "pred": pred(t.re),
            "rel_err_pct": 100.0 * abs(pred(t.re) - t.cd) / t.cd,
        }
        for t in train
    ]
    held = [
        {
            "re": t.re,
            "target": t.cd,
            "pred": pred(t.re),
            "rel_err_pct": 100.0 * abs(pred(t.re) - t.cd) / t.cd,
        }
        for t in heldout
    ]
    print(f"power-law baseline (train-only fit): C_D = {a:.4f} Re^{b:+.4f}")
    for tag, rr in (("train", rows), ("held-out", held)):
        print("  " + "  ".join(f"Re {r['re']:g}: {r['rel_err_pct']:.2f}%" for r in rr))
    return {"a": float(a), "b": float(b), "train": rows, "heldout": held}


def bgk_floor(case: HullCase, mask: torch.Tensor, targets: list[DragTarget]) -> list[dict]:
    """C_s -> 0 limit of the calibration path: the best Smagorinsky can do."""
    rows = []
    with torch.no_grad():
        for t in targets:
            cd = float(bounded_drag(case, re=t.re, cs=None, mask=mask))
            rows.append(
                {
                    "re": t.re,
                    "target": t.cd,
                    "cd_bgk": cd,
                    "rel_err_pct": 100.0 * (cd - t.cd) / t.cd,
                }
            )
    print("BGK floor (C_s = 0, no SGS term):")
    for r in rows:
        print(
            f"  Re {r['re']:6.1f}: BGK {r['cd_bgk']:7.4f} vs campaign {r['target']:7.4f} "
            f"-> {r['rel_err_pct']:+.2f}%"
        )
    return rows


def sensitivity(case: HullCase, mask: torch.Tensor, closure, res: tuple[float, ...]) -> list[dict]:
    """dC_D/dC_s at the identified closure, central differences (no grad)."""
    rows = []
    with torch.no_grad():
        for re in res:
            cs = float(closure(re))
            eps = max(1e-3, 0.1 * cs)
            up = float(bounded_drag(case, re=re, cs=cs + eps, mask=mask))
            dn = float(bounded_drag(case, re=re, cs=max(1e-6, cs - eps), mask=mask))
            grad = (up - dn) / ((cs + eps) - max(1e-6, cs - eps))
            mid = 0.5 * (up + dn)
            rows.append(
                {
                    "re": re,
                    "cs": cs,
                    "dcd_dcs": grad,
                    "elasticity": grad * cs / mid if mid else float("nan"),
                }
            )
    print("dC_D/dC_s at the identified closure (finite differences):")
    for r in rows:
        print(
            f"  Re {r['re']:6.1f}: C_s {r['cs']:.5f}  dC_D/dC_s {r['dcd_dcs']:+8.4f}  "
            f"dlnC_D/dlnC_s {r['elasticity']:+.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--window-start", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--cs0", type=float, default=0.1)
    parser.add_argument("--kinds", default="power,scalar", help="comma list: power,scalar")
    parser.add_argument("--checkpoint-block", type=int, default=25)
    parser.add_argument("--quick", action="store_true", help="data tables only, no fits")
    parser.add_argument("--out", default="", help="write all metrics to this json")
    args = parser.parse_args()

    u_in = 0.1
    case = HullCase(
        nz=64,
        ny=64,
        nx=128,
        u_in=u_in,
        steps=args.steps,
        window_start=args.window_start,
        checkpoint_block=args.checkpoint_block,
        device=args.device,
    )
    mask = case.make_mask()
    ref_area = case.ref_area(mask)
    print(
        f"hull {case.nz}x{case.ny}x{case.nx} L={case.hull_length} solid={int(mask.sum())} "
        f"S_proj={ref_area:.0f} tau(Re)=0.5+{3 * u_in * case.hull_length:.2f}/Re"
    )
    print(f"rollout {args.steps} steps, window [{args.window_start}, {args.steps})")

    report: dict = {
        "dataset": args.dataset,
        "train_re": TRAIN_RE,
        "heldout_re": HELDOUT_RE,
        "steps": args.steps,
        "window_start": args.window_start,
    }

    report["convergence"] = convergence_table(args.dataset, u_in, ref_area)

    targets = load_targets(args.dataset, u_in, ref_area)
    by_re = {t.re: t for t in targets}
    missing = [re for re in TRAIN_RE + HELDOUT_RE if re not in by_re]
    if missing:
        raise SystemExit(f"dataset lacks the requested Re points: {missing}")
    train = [by_re[re] for re in TRAIN_RE]
    heldout = [by_re[re] for re in HELDOUT_RE]
    print("train:", ", ".join(f"Re {t.re:g} -> C_D {t.cd:.4f}" for t in train))
    print("held-out:", ", ".join(f"Re {t.re:g} -> C_D {t.cd:.4f}" for t in heldout))

    report["powerlaw"] = powerlaw_baseline(train, heldout)
    if args.quick:
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        return

    report["bgk_floor"] = bgk_floor(case, mask, train + heldout)

    report["fits"] = {}
    for kind in args.kinds.split(","):
        t0 = time.time()
        result = calibrate(
            train,
            case,
            kind=kind,
            cs0=args.cs0,
            iters=args.iters,
            lr=args.lr,
            log_every=max(1, args.iters // 8),
        )
        elapsed = time.time() - t0
        ev_train = evaluate(result, train, case)
        ev_held = evaluate(result, heldout, case)
        print(
            f"[{kind}] identified {result.params} (re_ref {result.re_ref:.1f}) "
            f"loss {result.loss_history[0]:.4e} -> {result.loss_history[-1]:.4e} "
            f"in {elapsed:.0f}s"
        )
        for tag, ev in (("train", ev_train), ("held-out", ev_held)):
            for re, row in ev.items():
                print(
                    f"  {tag} Re {re:>6}: target {row['target']:.4f} pred {row['pred']:.4f} "
                    f"err {row['rel_err_pct']:.2f}%"
                )
        report["fits"][kind] = {
            "params": result.params,
            "re_ref": result.re_ref,
            "loss_history": result.loss_history,
            "elapsed_s": elapsed,
            "train": ev_train,
            "heldout": ev_held,
            "sensitivity": sensitivity(case, mask, result.closure, TRAIN_RE),
            "cs_curve": [
                {"re": re, "cs": float(result.closure(re).detach())}
                for re in (50.0, 100.0, 150.0) + TRAIN_RE + HELDOUT_RE + (1000.0,)
            ],
        }
        print(
            "  C_s(Re): "
            + ", ".join(f"{r['re']:g}->{r['cs']:.5f}" for r in report["fits"][kind]["cs_curve"])
        )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
