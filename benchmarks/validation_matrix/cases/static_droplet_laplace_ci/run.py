"""CI-tier static-droplet (Young-Laplace) validation for the matrix slot 6.

The physics is *not* reimplemented here: the stepper, the steady-state
measurement and the Laplace fit are imported from the verified standard-tier
case ``benchmarks/verified/laplace_droplet/run.py`` (SCMP SC94, psi=1-exp(-rho),
physical G_eff=-5 -> library G_lib=+5, tau=1).  What this runner adds is the
part the matrix needs:

* a CI-scale radius set (R = 8/12/16 for the coarse level, 12/16/20 for the
  fine level) that finishes in a couple of minutes on CPU;
* a two-level resolution check of the fitted surface tension sigma_eff, which
  the standard tier (R = 15/25/40) cannot provide inside CI.

Anchor (zero hand-copy): the 2-D Young-Laplace law ``Delta p = sigma / R`` is
read from the verified case README; the fit itself is the imported
``fit_laplace`` (free-intercept line through the origin test, R^2 >= 0.999,
per-radius sigma within 3%).

Usage::

    python benchmarks/validation_matrix/cases/static_droplet_laplace_ci/run.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
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

LAPLACE_RUNNER = REPO_ROOT / "benchmarks" / "verified" / "laplace_droplet" / "run.py"
CASE_ID = "static_droplet_laplace_ci"
TIER = "ci"


def load_laplace_case():
    spec = importlib.util.spec_from_file_location("laplace_droplet_case", LAPLACE_RUNNER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {LAPLACE_RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_level(mod, radii: list[float], max_steps: int, min_steps: int,
                  sample_interval: int, device: torch.device) -> dict:
    rows = []
    t0 = time.perf_counter()
    for R in radii:
        L = int(round(4.0 * R))
        final, _hist = mod.run_droplet(
            2, R, L, device, max_steps, min_steps, sample_interval,
            compile_mode="eager",
        )
        rows.append(
            {
                "R_init": R,
                "R_eq": final["R_eq"],
                "dp": final["dp"],
                "p_in": final["p_in"],
                "p_out": final["p_out"],
                "rho_in": final["rho_in"],
                "rho_out": final["rho_out"],
                "max_u": final["max_u"],
                "mass_drift": final["mass_drift"],
                "converged": final["converged"],
                "step": final["step"],
                "elapsed_s": final["dt_s"],
            }
        )
    elapsed = time.perf_counter() - t0

    fit = mod.fit_laplace(rows, 2)
    return {
        "level": "R=" + "/".join(f"{r:g}" for r in radii),
        "resolution": {"R": radii, "L": [int(round(4 * r)) for r in radii]},
        "metrics": {
            "laplace_max_dev_pct": fit["max_dev_sigma_pct"],
            "sigma_eff": fit["sigma_eff_fit"],
            "R2": fit["R2"],
            "intercept_frac_pct": fit["intercept_frac_pct"],
        },
        "detail": {
            "pass_laplace": fit["pass_laplace"],
            "per_radius": fit["per_radius"],
            "elapsed_s": elapsed,
            "solver": "tensorlbm.multiphase.collide_sc_single_component + solver.stream",
            "model": "SCMP SC94 psi_exp, G_lib=+5 (physical G_eff=-5), tau=1.0",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radii-coarse", default="12,16,20",
                    help="gated level 0 (must be above the resolvable radius: "
                         "R >= 3 interface widths, W=4 -> R >= 12)")
    ap.add_argument("--radii-fine", default="16,20,24",
                    help="gated level 1")
    ap.add_argument("--radii-ladder", default="8,12,16",
                    help="diagnostic resolution ladder (reported, not gated): "
                         "R=8 is 2 interface widths and is expected to violate "
                         "the Laplace gate, which is the point of keeping it")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--min-steps", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(CASE_DIR))
    args = ap.parse_args()

    mod = load_laplace_case()
    device = torch.device(args.device)
    coarse = [float(x) for x in args.radii_coarse.split(",")]
    fine = [float(x) for x in args.radii_fine.split(",")]
    ladder_radii = [float(x) for x in args.radii_ladder.split(",") if x]

    levels = [
        measure_level(mod, coarse, args.max_steps, args.min_steps,
                      args.sample_interval, device),
        measure_level(mod, fine, args.max_steps, args.min_steps,
                      args.sample_interval, device),
    ]
    ladder = (
        [measure_level(mod, ladder_radii, args.max_steps, args.min_steps,
                       args.sample_interval, device)]
        if ladder_radii else []
    )

    # two-level resolution consistency of the fitted surface tension
    sig = [lv["metrics"]["sigma_eff"] for lv in levels]
    pair_mean = sum(sig) / 2.0
    for lv in levels:
        lv["metrics"]["sigma_pair_dev_pct"] = (
            abs(lv["metrics"]["sigma_eff"] - pair_mean) / pair_mean * 100.0
        )

    result = {
        "case": CASE_ID,
        "tier": TIER,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": {
            "id": "laplace_law_2d",
            "law": "Delta p = sigma / R (2-D circular droplet)",
            "source": "benchmarks/verified/laplace_droplet/README.md",
            "origin": "Young-Laplace",
            "note": "sigma_eff is fitted from the simulation itself; the law's "
                    "linearity (sigma independent of R) is the anchor, not a "
                    "tabulated sigma.",
        },
        "config": {
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "sample_interval": args.sample_interval,
            "conv_dp": 2e-4,
            "conv_r": 2e-3,
            "device": args.device,
            "compile_mode": "eager",
        },
        "levels": levels,
        "resolution_ladder": [
            {
                "level": lv["level"],
                "metrics": lv["metrics"],
                "detail": lv["detail"],
                "gated": False,
                "note": "diagnostic only: R=8 is 2 interface widths (W=4) and "
                        "is below the resolvable-radius requirement R >= 3W; "
                        "its violation of the Laplace gate is recorded, not hidden",
            }
            for lv in ladder
        ],
        "tolerances": {
            "laplace_max_dev_pct": 3.0,
            "sigma_pair_dev_pct": 3.0,
            "R2": 0.999,
            "intercept_frac_pct": 3.0,
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for lv in levels + ladder:
        m = lv["metrics"]
        print(
            f"{lv['level']:<16}: sigma_eff={m['sigma_eff']:.6f} "
            f"max_dev={m['laplace_max_dev_pct']:.3f}% R2={m['R2']:.6f} "
            f"pair_dev={m.get('sigma_pair_dev_pct', float('nan')):.3f}% "
            f"({lv['detail']['elapsed_s']:.1f}s)"
        )
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())