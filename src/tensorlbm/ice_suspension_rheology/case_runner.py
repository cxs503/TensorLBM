"""Single-case worker: run one shear-cell (or Stokes) case in a fresh process.

The campaign dispatches one of these per case with ``subprocess``.  Using a
process per case is deliberate:

* torch's thread pool plus ``multiprocessing.fork`` deadlocks, and ``spawn``
  re-imports the driver (fragile under the harness), so the worker is a plain
  CLI invoked with ``python -m``;
* a failed case leaves a JSON record with its exception instead of killing the
  campaign;
* the CPU budget per case is explicit (``--threads``).

Usage::

    python -m tensorlbm.ice_suspension_rheology.case_runner --out case.json \\
        --mode shear --spec '{"phi": 0.05, "seed": 1, "nx": 48, ...}'

    python -m tensorlbm.ice_suspension_rheology.case_runner --out gate.json \\
        --mode stokes --spec '{"d": 12, "re_p": 0.1, "steps": 3000}'
"""

from __future__ import annotations

import argparse
import json
import traceback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="shear", choices=("shear", "stokes", "drag"))
    ap.add_argument("--spec", default="{}", help="JSON dict of keyword arguments")
    ap.add_argument("--spec-file", default=None)
    args = ap.parse_args(argv)
    spec = json.loads(args.spec) if not args.spec_file else json.load(open(args.spec_file))
    threads = int(spec.pop("threads", 2))
    import torch

    torch.set_num_threads(max(1, threads))
    try:
        if args.mode == "drag":
            from .calibrate import run_drag_case

            out = run_drag_case(**spec)
        elif args.mode == "shear":
            from .calibrate import run_case

            out = run_case(**spec)
        else:
            from .calibrate import _stokes_case

            out = _stokes_case(
                spec.get("d", 8), spec.get("re_p", 0.1), spec.get("steps", 3000),
                spec.get("domain_factor", 4.0), threads,
            )
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        out = {"error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(), "spec": spec}
    with open(args.out, "w") as fh:
        json.dump(out, fh, default=float)
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())