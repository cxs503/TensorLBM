#!/usr/bin/env python3
"""Run one numerical-wave-tank case and write its acceptance metrics as JSON.

Usage::

    python benchmarks/nwt_stokes2_2d/run_case.py \
        --out runs/nwt/kh3_eps002/case.json \
        --set wavelength=69 depth=33 steepness=0.02

``--set`` accepts ``key=value`` pairs of :class:`tensorlbm.nwt.NWTConfig`
fields (values are parsed with ``json.loads``, falling back to a string).
Probe time series are stored next to the metrics in ``probes.npz`` so figures
can be regenerated without re-running the solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorlbm.nwt as nwt  # noqa: E402
from nwt_metrics import evaluate_case  # noqa: E402


_BOOL_STRINGS = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}


def _coerce(key: str, raw: str, defaults: dict):
    """Parse a ``--set`` value against the declared NWTConfig field.

    ``json.loads`` alone is not enough: a shell-friendly ``key=False`` would fail
    JSON parsing, silently become the *string* ``"False"`` (truthy) and leave the
    flag switched on.  Booleans are therefore parsed explicitly and every value
    is cast to the declared field type, so a typo raises instead of silently
    changing the experiment.
    """
    if key not in defaults:
        raise SystemExit(f"unknown NWTConfig field: {key}")
    default = defaults[key]
    low = raw.strip()
    if isinstance(default, bool):
        if low.lower() not in _BOOL_STRINGS:
            raise SystemExit(f"{key}: expected a boolean, got {raw!r}")
        return _BOOL_STRINGS[low.lower()]
    if isinstance(default, int) and not isinstance(default, bool):
        return int(low)
    if isinstance(default, float):
        return float(low)
    if low.startswith('"') or low.startswith("'"):
        return json.loads(low.replace("'", '"'))
    try:
        return json.loads(low)
    except json.JSONDecodeError:
        return raw


def _parse_set(pairs: list[str]) -> dict:
    defaults = {k: getattr(nwt.NWTConfig(), k) for k in nwt.NWTConfig().__dataclass_fields__}
    out = {}
    for item in pairs:
        key, _, raw = item.partition("=")
        out[key.strip()] = _coerce(key.strip(), raw, defaults)
    return out


def _resolved_threads() -> int:
    """One thread per case: the 2-D Körner step is memory-bound and loses
    throughput under intra-process threading (measured 48 ms -> 828 ms per step
    from 1 to 16 threads at 7.4 k cells on loong64)."""
    try:
        return int(os.environ.get("NWT_TORCH_THREADS", "1"))
    except ValueError:
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="metrics JSON path")
    ap.add_argument("--set", nargs="*", default=[], help="NWTConfig overrides key=value")
    ap.add_argument("--save-probes", action="store_true", help="store probe series in probes.npz")
    args = ap.parse_args(argv)

    torch.set_num_threads(_resolved_threads())
    overrides = _parse_set(args.set)
    if "device" in overrides and overrides["device"] == "auto":
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = nwt.NWTConfig(**overrides)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = nwt.run_nwt(cfg)
    metrics = evaluate_case(result)
    metrics["solver"] = "tensorlbm.free_surface_lbm_2d.koerner_step_2d"
    metrics["torch_threads"] = torch.get_num_threads()
    metrics["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(metrics, indent=1, default=float))

    if args.save_probes:
        import numpy as np

        np.savez_compressed(
            out.parent / "probes.npz",
            eta=result.eta_probes.numpy(),
            probe_x=np.asarray(result.probe_x),
            inventory=result.inventory.numpy(),
            time=result.time.numpy(),
            eta_snap=result.eta_snapshots.numpy(),
            snap_steps=np.asarray(result.snapshot_steps),
            omega=cfg.omega,
            dt=1.0,
        )

    print(
        "case kh=%.3f eps=%.3f sponge=%s | C1 %.2e | C2 A1 %.2e A2 %.2e | C3 KR=%.4f | C4 %.3e/period | %.1f s"
        % (
            metrics["case"]["kh"],
            cfg.steepness,
            cfg.sponge_enabled,
            metrics["dispersion"]["relative_error"],
            metrics["waveform"]["A1_relative_error"],
            metrics["waveform"]["A2_relative_error"],
            metrics["reflection"]["KR_two_point"],
            metrics["mass"]["max_per_period_drift"],
            time.perf_counter() - t0,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())