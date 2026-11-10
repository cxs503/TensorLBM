#!/usr/bin/env python3
"""Run the LBM-LES channel verification campaign (a short grid ladder).

Each ``--case`` is ``h:steps:warmup:strain_source:cs``; every case writes its
own artifact (``case_<label>.json``) **as soon as it finishes**, so a later
failure or wall-clock kill cannot destroy earlier evidence.  A case that
diverges is recorded as ``honest_fail`` with the exception text; the campaign
keeps running (a failed coarse case does not invalidate a finer one).

    PYTHONPATH=src python benchmarks/p84w3_n05_les_channel/run_campaign.py \
        --case 8:8000:2000:forward_gradient:0.12 \
        --case 12:14000:4000:forward_gradient:0.12 \
        --case 16:30000:10000:forward_gradient:0.12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.turbulence.les_channel_verification import run_case  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", action="append", required=True,
                    help="h:steps:warmup:strain_source:cs")
    ap.add_argument("--u-tau", type=float, default=0.01)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--sample-interval", type=int, default=20)
    ap.add_argument("--report-interval", type=int, default=1000)
    ap.add_argument("--wall-model", default="none")
    ap.add_argument("--no-van-driest", action="store_true")
    ap.add_argument("--tau-minus-max", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "artifacts" / "p84w3_n05_les_channel")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    register = {"cases": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for spec in args.case:
        h, steps, warmup, strain, cs = spec.split(":")
        label = f"re{int(args.re_tau)}_h{h}_{strain}_cs{float(cs):.2f}"
        path = args.outdir / f"case_{label}.json"
        t0 = time.time()
        record = {"spec": spec, "label": label, "artifact": str(path)}
        try:
            art = run_case(
                half_height=int(h),
                steps=int(steps),
                warmup_steps=int(warmup),
                u_tau=args.u_tau,
                re_tau=args.re_tau,
                cs=float(cs),
                strain_source=strain,
                wall_model=args.wall_model,
                van_driest=not args.no_van_driest,
                tau_minus_max=args.tau_minus_max,
                sample_interval=args.sample_interval,
                report_interval=args.report_interval,
                label=label,
                verbose=True,
            )
            art["command"] = {"argv": sys.argv[1:], "case": spec,
                              "wall_clock_seconds": time.time() - t0}
            path.write_text(json.dumps(art, indent=1))
            record.update({
                "status": "completed",
                "summary": art["summary"],
                "criteria": {k: {"status": v["status"], "metric": v.get("metric")}
                             for k, v in art["criteria"]["checks"].items()},
                "wall_clock_seconds": time.time() - t0,
            })
        except Exception as exc:  # honest FAIL is data, not a crash
            record.update({
                "status": "honest_fail",
                "error": f"{type(exc).__name__}: {exc}",
                "trace_tail": traceback.format_exc()[-1200:],
                "wall_clock_seconds": time.time() - t0,
            })
        register["cases"].append(record)
        (args.outdir / "campaign_register.json").write_text(json.dumps(register, indent=1))
        print(json.dumps(record, indent=1, default=str)[:2000], flush=True)

    print("CAMPAIGN DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
