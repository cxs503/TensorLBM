#!/usr/bin/env python3
"""Generate the Saint-Venant (SWE) wall-pressure anchor for the LBM calibration.

The low-order sloshing chain lives in the ``ship-performance-platform``
repository (IN48-N07, ``backend/physics/sloshing/impact.py``).  To keep this
repository free of a cross-repository import dependency, the SWE time histories
are *derived once* here (the source module is imported read-only from
``$M5744_ROOT``) and stored as a data artifact with full provenance
(repository commit, module path, parameters, solver settings).

Cases (both on the IN48-N07 tank: L = 20 m, h_l = 4 m, rho = 1025, g = 9.80665):

``surge``
    ``dam_break_case(h0=4, wet_length=5, dry_length=15)`` - the surge-release
    impact configuration that the LBM impact case reproduces.
``forced``
    ``forced_tank_impact`` with ``regular_excitation(omega_1, A=0.25 m)`` -
    the IN48-N07 forced anchor (standing-wave wall pressure history).

Usage
-----
    M5744_ROOT=/path/to/ship-performance-platform \
    python scripts/gen_swe_reference_anchor.py --out artifacts/sloshing_impact_lbm/sw_swe_reference.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_impact_calibration import TankAnchor  # noqa: E402

DEFAULT_M5744 = Path("/root/tp-agent/work/m5744")


def _load_impact_module(root: Path):
    path = root / "backend" / "physics" / "sloshing" / "impact.py"
    if not path.exists():
        raise SystemExit(
            f"SWE anchor module not found at {path}; set M5744_ROOT to the "
            "ship-performance-platform checkout"
        )
    spec = importlib.util.spec_from_file_location("_m5744_sloshing_impact", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)  # read-only: never written back
    return mod, path


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or None
    except Exception:  # pragma: no cover - provenance only
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m5744-root", default=os.environ.get("M5744_ROOT", str(DEFAULT_M5744)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=1200)
    ap.add_argument("--t-end", type=float, default=6.0)
    ap.add_argument("--periods", type=float, default=6.0)
    args = ap.parse_args(argv)

    root = Path(args.m5744_root)
    mod, path = _load_impact_module(root)
    anchor = TankAnchor()

    # --- (1) surge release: column 5 m long, far wall at x = 20 m ----------
    t0 = time.time()
    surge = mod.dam_break_case(
        h0=anchor.depth, dry_length=15.0, wet_length=5.0,
        nx=args.nx, t_end=args.t_end, sample_every=4,
    )
    p_surge = anchor.rho * anchor.g * np.asarray(surge.wall_right)

    # --- (2) forced standing wave (IN48-N07 regular anchor) ---------------
    exc = mod.regular_excitation(anchor.omega1, 0.25)
    forced = mod.forced_tank_impact(
        anchor.length, anchor.depth, exc, n_periods=args.periods,
        nx=600, sample_every=20,
    )
    p_forced = anchor.rho * anchor.g * np.asarray(forced.wall_right)
    wall_s = time.time() - t0

    payload = {
        "case": "swe_reference",
        "source": {
            "repository": "ship-performance-platform (IN48-N07 sloshing chain)",
            "module": str(path),
            "git_commit": _git_commit(root),
            "note": "imported read-only; only derived time histories are stored",
        },
        "anchor": {
            "length": anchor.length, "depth": anchor.depth, "rho": anchor.rho,
            "g": anchor.g, "omega1": anchor.omega1, "period1": anchor.period1,
        },
        "solver": {"nx": args.nx, "muscl": True, "cfl": 0.35,
                   "samples_surge": int(np.asarray(surge.t).size),
                   "samples_forced": int(np.asarray(forced.t).size)},
        "surge": {
            "case": "surge",
            "params": {"h0": anchor.depth, "wet_length": 5.0, "dry_length": 15.0,
                       "t_end": args.t_end},
            "t": np.asarray(surge.t).tolist(),
            "h_wall": np.asarray(surge.wall_right).tolist(),
            "p_wall": p_surge.tolist(),
            "p_peak": float(p_surge.max()),
            "p_peak_over_rho_g_h": float(p_surge.max() / anchor.p_hydrostatic),
            "t_impact": float(np.asarray(surge.t)[int(np.argmax(p_surge))]),
        },
        "forced": {
            "case": "forced",
            "params": {"amplitude_m": 0.25, "omega": anchor.omega1,
                       "periods": args.periods, "nx": 600},
            "t": np.asarray(forced.t).tolist(),
            "h_wall": np.asarray(forced.wall_right).tolist(),
            "p_wall": p_forced.tolist(),
            "p_peak": float(p_forced.max()),
            "p_peak_over_rho_g_h": float(p_forced.max() / anchor.p_hydrostatic),
        },
        "wall_seconds": round(wall_s, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {out}: surge P_peak/(rho g h)={payload['surge']['p_peak_over_rho_g_h']:.4f} "
          f"t_impact={payload['surge']['t_impact']:.3f}s; "
          f"forced P_peak/(rho g h)={payload['forced']['p_peak_over_rho_g_h']:.4f} "
          f"({payload['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())