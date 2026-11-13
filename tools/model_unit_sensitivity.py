#!/usr/bin/env python
"""Resolution / parameter sensitivity of the model-unit CI benchmarks.

The unit manifests fix one parameter set per directory, so a single CI verdict
says nothing about whether that verdict is near a discretisation limit or a
lucky point.  This driver re-runs the *same* kernels through the *same* CI path
(``run_ci`` on a modified ``ModelUnit``) over

* channel grid refinement ``ny`` in {16, 24, 32} (time to steady scales with
  ``ny^2``, so ``steps`` is scaled with it), for BGK and TRT; and
* shear-wave relaxation time ``tau`` in {0.60, 0.65, 0.80, 1.00}, for BGK and
  MRT, checking that the recovered viscosity follows ``nu=(tau-1/2)/3`` rather
  than being right at a single tau.

Outputs ``artifacts/model_units/sensitivity.json`` and ``sensitivity.md``.

    python tools/model_unit_sensitivity.py --out artifacts/model_units
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.models.ci_unit import load_manifest, run_ci  # noqa: E402

UNIT_ROOT = REPO_ROOT / "models" / "units"
BASE_STEPS = 24000
BASE_NY = 32


def _scaled_params(params: dict, ny: int) -> dict:
    steps = int(round(BASE_STEPS * (ny / BASE_NY) ** 2))
    return {**params, "ny": ny, "steps": steps, "avg_start": int(round(0.75 * steps))}


def channel_sweep() -> list[dict]:
    rows: list[dict] = []
    for unit_id in ("d2q9_bgk_channel_poiseuille", "d2q9_trt_channel_poiseuille"):
        unit = load_manifest(UNIT_ROOT / unit_id / "model.json")
        for ny in (16, 24, 32):
            params = _scaled_params(dict(unit.ci.params), ny)
            variant = replace(unit, ci=replace(unit.ci, params=params))
            record = run_ci(variant)
            rows.append(
                {
                    "unit_id": unit_id,
                    "collision": unit.composition.collision,
                    "kernel": unit.ci.kernel,
                    "variant": {"ny": ny, "steps": params["steps"]},
                    "metrics": record.metrics,
                    "ok": record.ok,
                    "seconds": record.seconds,
                    "error": record.error,
                }
            )
            print(
                f"channel {unit_id} ny={ny:3d} steps={params['steps']:6d} "
                f"l2={record.metrics.get('profile_l2_rel_err', float('nan')):.3e} "
                f"umax={record.metrics.get('umax_rel_err', float('nan')):.3e} "
                f"({record.seconds:.1f}s)",
                flush=True,
            )
    return rows


def shear_sweep() -> list[dict]:
    rows: list[dict] = []
    for unit_id in ("d2q9_bgk_shear_wave", "d2q9_mrt_shear_wave"):
        unit = load_manifest(UNIT_ROOT / unit_id / "model.json")
        for tau in (0.60, 0.65, 0.80, 1.00):
            params = {**dict(unit.ci.params), "tau": tau}
            variant = replace(unit, ci=replace(unit.ci, params=params))
            record = run_ci(variant)
            rows.append(
                {
                    "unit_id": unit_id,
                    "collision": unit.composition.collision,
                    "kernel": unit.ci.kernel,
                    "variant": {"tau": tau},
                    "metrics": record.metrics,
                    "reference": record.reference,
                    "ok": record.ok,
                    "seconds": record.seconds,
                    "error": record.error,
                }
            )
            print(
                f"shear {unit_id} tau={tau:.2f} "
                f"nu_err={record.metrics.get('nu_rel_err', float('nan')):.3e} "
                f"nu_ref={record.reference.get('nu', float('nan')):.6e} "
                f"({record.seconds:.1f}s)",
                flush=True,
            )
    return rows


def markdown(channel: list[dict], shear: list[dict]) -> str:
    def f(value: float | None, digits: int = 3) -> str:
        if value is None:
            return "-"
        return f"{value:.{digits}e}"

    lines = ["# Model-unit CI sensitivity", ""]
    lines.append("## Channel (Poiseuille) grid refinement")
    lines.append("")
    lines.append("| unit | ny | steps | profile L2 rel err | u_max rel err | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for row in channel:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row["unit_id"],
                row["variant"]["ny"],
                row["variant"]["steps"],
                f(row["metrics"].get("profile_l2_rel_err")),
                f(row["metrics"].get("umax_rel_err")),
                "pass" if row["ok"] else "FAIL",
            )
        )
    lines.append("")
    lines.append("## Shear wave: relaxation-time sweep (nu = (tau-1/2)/3)")
    lines.append("")
    lines.append("| unit | tau | nu (analytic) | nu rel err | verdict |")
    lines.append("|---|---|---|---|---|")
    for row in shear:
        lines.append(
            "| `{}` | {:.2f} | {} | {} | {} |".format(
                row["unit_id"],
                row["variant"]["tau"],
                f(row["reference"].get("nu")),
                f(row["metrics"].get("nu_rel_err")),
                "pass" if row["ok"] else "FAIL",
            )
        )
    lines.append("")
    lines.append(
        "Observed order (channel, BGK): error should fall with ny; the TRT arm "
        "uses the same grid and force, so the difference between the arms is "
        "the wall-placement error only."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/model_units")
    args = parser.parse_args(argv)

    channel = channel_sweep()
    shear = shear_sweep()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "channel_grid_refinement": channel,
        "shear_tau_sweep": shear,
        "repro_cmd": "python tools/model_unit_sensitivity.py --out artifacts/model_units",
    }
    (out / "sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "sensitivity.md").write_text(markdown(channel, shear))
    print(f"wrote {out / 'sensitivity.json'}")
    failed = [row for row in channel + shear if not row["ok"]]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
