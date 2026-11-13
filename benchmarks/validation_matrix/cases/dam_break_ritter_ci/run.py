"""CI-tier dam-break validation against the Ritter (1892) self-similar solution.

This is the matrix slot-8 runner (``dam_break``). It is deliberately small: two
grids (nx = 200 / 400) at ``t_end = 5 sqrt(h0/g)``, which costs well under a
minute on CPU, so the slot can be reproduced inside CI. The standard tier
(nx up to 1600, self-similarity collapse, two independent front estimators)
stays in ``benchmarks/dam_break_ritter``.

Anchor (zero hand-copy): the analytic front law ``x_f(t) = 2 sqrt(g h0) t`` and
the depth profile ``h = (2c0 - xi)^2 / 9g`` are taken from the library module
``tensorlbm.shallow_water`` — the same module that integrates the solution, so
the runner cannot drift from the anchor it claims to use.

Metrics per grid
----------------
``front_err_pct``   signed error of the mass-recovered front (integral form of
                    the Ritter law, immune to dry-front smearing) vs the law.
``profile_l2_pct``  relative L2 error of the depth profile inside the wet fan.
``mass_drift_pct``  total fluid volume drift (the FV scheme is flux-form).

Scheme (2026-11-13, m17112): the judged levels run the second-order MUSCL
(minmod) + local-speed HLL + SSP-RK2 scheme of ``tensorlbm.shallow_water``
(``scheme="muscl2"``).  The legacy first-order global-wave-speed Rusanov flux is
still selectable with ``--scheme rusanov`` and is kept as a regression anchor
(``tests/test_shallow_water_muscl2.py``).  Rationale: the first-order scheme's
numerical viscosity is ``~ (|u|+c) dx`` irrespective of the local gradient, which
put ``profile_l2_pct`` at 9.82% / 6.29% (nx=200/400) -- an under-resolution of the
profile *metric*, not a mis-calibrated gate.

Usage::

    python benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from tensorlbm.shallow_water import (  # noqa: E402
    ShallowWaterConfig,
    front_from_mass,
    ritter_depth,
    ritter_front,
    run_ritter_dam_break,
)

CASE_ID = "dam_break_ritter_ci"
TIER = "ci"


def measure_level(nx: int, t_end: float, cfl: float, scheme: str = "muscl2") -> dict:
    cfg = ShallowWaterConfig(
        scheme=scheme,
        nx=nx,
        x_min=-4.0,
        x_max=12.0,
        dam_x=0.0,
        h0=1.0,
        g=1.0,
        cfl=cfl,
        t_end=t_end,
        sample_every=max(1, nx // 8),
    )
    t0 = time.perf_counter()
    run = run_ritter_dam_break(cfg)
    elapsed = time.perf_counter() - t0

    t_final = run.time[-1]
    vol_down = run.volume_downstream[-1]
    front_num = front_from_mass(vol_down, cfg.h0)
    front_ref = ritter_front(t_final, cfg.h0, cfg.g)
    front_err_pct = (front_num - front_ref) / front_ref * 100.0

    # depth-profile L2 error on the wet fan 0 <= x <= x_f
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=torch.float64)
    h_num = torch.as_tensor(run.profiles[-1][1], dtype=torch.float64)
    c0 = math.sqrt(cfg.g * cfg.h0)
    x_f = 2.0 * c0 * t_final
    mask = (x >= 0.0) & (x <= x_f)
    h_ref = torch.tensor(
        [ritter_depth(float(xi), t_final, cfg.h0, cfg.g) for xi in x[mask]],
        dtype=torch.float64,
    )
    num = float(torch.sqrt(torch.sum((h_num[mask] - h_ref) ** 2) / torch.sum(h_ref**2)))
    profile_l2_pct = num * 100.0

    mass_drift_pct = abs(run.mass_final - run.mass_initial) / run.mass_initial * 100.0

    return {
        "level": f"nx={nx}",
        "resolution": {"nx": nx, "dx": cfg.dx},
        "metrics": {
            "front_err_pct": front_err_pct,
            "profile_l2_pct": profile_l2_pct,
            "mass_drift_pct": mass_drift_pct,
        },
        "detail": {
            "t_final": t_final,
            "front_numeric": front_num,
            "front_analytic": front_ref,
            "volume_downstream": vol_down,
            "steps": run.steps,
            "dt": run.dt,
            "elapsed_s": elapsed,
            "lattice": "SWE finite volume (1-D): " + (
            "MUSCL-Hancock/minmod reconstruction + local-speed HLL flux + SSP-RK2"
            if scheme == "muscl2" else
            "1st-order global-wave-speed Rusanov/HLL flux"),
        "scheme": scheme,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", default="200,400", help="comma separated grid sizes")
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--ladder", default="800,1600",
                    help="extra grids reported as a resolution ladder (diagnostic, "
                         "not part of the judged CI levels)")
    ap.add_argument("--scheme", default="muscl2", choices=["muscl2", "rusanov"],
                    help="SWE flux scheme; muscl2 = 2nd-order (default), "
                         "rusanov = legacy 1st-order global-speed flux")
    ap.add_argument("--out", default=str(CASE_DIR))
    args = ap.parse_args()

    grids = [int(s) for s in args.nx.split(",")]
    levels = [measure_level(nx, args.t_end, args.cfl, args.scheme) for nx in grids]
    ladder = [measure_level(int(s), args.t_end, args.cfl, args.scheme)
              for s in args.ladder.split(",") if s]

    result = {
        "case": CASE_ID,
        "tier": TIER,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": {
            "id": "ritter_front_law",
            "law_front": "x_f(t) = 2 sqrt(g h0) t",
            "law_depth": "h(x,t) = (2c0 - xi)^2 / 9g,  xi = x/t",
            "source": "src/tensorlbm/shallow_water.py (docstring + ritter_front/ritter_depth)",
            "origin": "Ritter 1892",
        },
        "config": {"t_end": args.t_end, "cfl": args.cfl, "h0": 1.0, "g": 1.0,
                   "scheme": args.scheme},
        "levels": levels,
        "resolution_ladder": [
            {"level": lv["level"], "metrics": lv["metrics"],
             "detail": lv["detail"]}
            for lv in ladder
        ],
        "tolerances": {
            "front_err_pct": 5.0,
            "profile_l2_pct": 5.0,
            "mass_drift_pct": 1.0,
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for lv in levels + ladder:
        m = lv["metrics"]
        print(
            f"{lv['level']:>8}: front_err={m['front_err_pct']:+.3f}% "
            f"profile_l2={m['profile_l2_pct']:.3f}% "
            f"mass_drift={m['mass_drift_pct']:.2e}% "
            f"({lv['detail']['elapsed_s']:.2f}s, {lv['detail']['steps']} steps)"
        )
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())