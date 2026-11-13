"""Scheme-upgrade evidence bundle for matrix slot 8 (``dam_break``).

Produces, under ``artifacts/validation_matrix/dam_break_scheme_upgrade/``:

* ``figure_dam_break_scheme.png`` -- (a) depth profile at t=5 sqrt(h0/g) against
  the Ritter (1892) self-similar solution for the legacy first-order flux and the
  new second-order scheme at nx=200 and nx=400; (b) profile-L2 error vs grid with
  the matrix 5% gate and the frozen baseline.
* ``baseline_vs_after.json`` -- the numbers behind the figure, with the baseline
  read from the git blob of the pre-change artifact (zero hand-copy).

Run: ``PYTHONPATH=src python benchmarks/validation_matrix/cases/dam_break_ritter_ci/make_scheme_figure.py``
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.shallow_water import (  # noqa: E402
    ShallowWaterConfig,
    front_from_mass,
    ritter_depth,
    ritter_front,
    run_ritter_dam_break,
)

OUT = REPO_ROOT / "artifacts" / "validation_matrix" / "dam_break_scheme_upgrade"
T_END, CFL = 5.0, 0.4
GATE = 5.0
BASELINE_BLOB = "HEAD:benchmarks/validation_matrix/cases/dam_break_ritter_ci/result.json"


def measure(nx: int, scheme: str) -> dict:
    cfg = ShallowWaterConfig(nx=nx, scheme=scheme, t_end=T_END, cfl=CFL,
                             sample_every=max(1, nx // 8))
    run = run_ritter_dam_break(cfg)
    t_final = run.time[-1]
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=torch.float64)
    h = torch.as_tensor(run.profiles[-1][1], dtype=torch.float64)
    x_f = ritter_front(t_final, cfg.h0, cfg.g)
    mask = (x >= 0.0) & (x <= x_f)
    h_ref = torch.tensor([ritter_depth(float(xi), t_final, cfg.h0, cfg.g) for xi in x[mask]],
                         dtype=torch.float64)
    l2 = float(torch.sqrt(torch.sum((h[mask] - h_ref) ** 2) / torch.sum(h_ref ** 2))) * 100.0
    front_num = front_from_mass(run.volume_downstream[-1], cfg.h0)
    return {
        "nx": nx, "scheme": scheme,
        "profile_l2_pct": l2,
        "front_err_pct": (front_num / ritter_front(t_final, cfg.h0, cfg.g) - 1.0) * 100.0,
        "mass_drift_pct": abs(run.mass_final - run.mass_initial) / run.mass_initial * 100.0,
        "t_final": t_final, "steps": run.steps,
        "x": x.tolist(), "h": run.profiles[-1][1],
        "h_ref_mask": h_ref.tolist(), "mask": mask.tolist(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grids = [200, 400, 800]
    data = {s: [measure(nx, s) for nx in grids] for s in ("rusanov", "muscl2")}

    baseline = json.loads(subprocess.run(
        ["git", "show", BASELINE_BLOB], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout)

    bundle = {
        "case": "dam_break_ritter_ci",
        "slot": "dam_break",
        "mission": "m17112",
        "gate_pct": GATE,
        "reference": {"law": "h=(2c0-xi)^2/9g, x_f=2c0 t; Ritter 1892",
                      "source": "src/tensorlbm/shallow_water.py::ritter_depth/ritter_front"},
        "baseline_pre_change_artifact": {
            "git_blob": BASELINE_BLOB,
            "levels": {lv["level"]: lv["metrics"] for lv in baseline["levels"]},
            "ladder": {lv["level"]: lv["metrics"] for lv in baseline["resolution_ladder"]},
        },
        "after_change": {
            s: [{k: r[k] for k in ("nx", "scheme", "profile_l2_pct", "front_err_pct",
                                   "mass_drift_pct", "t_final", "steps")} for r in rows]
            for s, rows in data.items()
        },
    }
    (OUT / "baseline_vs_after.json").write_text(json.dumps(bundle, indent=2) + "\n")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ref = data["muscl2"][0]
    xs = [xx for xx, m in zip(ref["x"], ref["mask"]) if m]
    ax0.plot(xs, ref["h_ref_mask"], "k-", lw=1.6, label="Ritter 1892 (analytic)")
    ax0.plot(ref["x"], ref["h"], "o", ms=2.4, color="tab:red",
             label="1st-order Rusanov, nx=200 (L2=%.2f%%)" % data["rusanov"][0]["profile_l2_pct"])
    ax0.plot(ref["x"], ref["h"] if False else data["muscl2"][0]["h"], "-", lw=1.3, color="tab:blue",
             label="MUSCL2+HLL, nx=200 (L2=%.2f%%)" % data["muscl2"][0]["profile_l2_pct"])
    ax0.set_xlim(0, 14)
    ax0.set_ylim(0, 1.08)
    ax0.set_xlabel(r"$x/h_0$")
    ax0.set_ylabel(r"$h/h_0$")
    ax0.set_title(r"dam break, $t=5\sqrt{h_0/g}$")
    ax0.legend(fontsize=7.5)
    ax0.grid(alpha=0.3)

    ax1.semilogy(grids, [r["profile_l2_pct"] for r in data["rusanov"]], "s--", label="1st-order Rusanov")
    ax1.semilogy(grids, [r["profile_l2_pct"] for r in data["muscl2"]], "o-", label="MUSCL2 + local HLL")
    ax1.axhline(GATE, color="tab:green", ls=":", lw=1.5, label="matrix gate 5%")
    ax1.set_xticks(grids)
    ax1.set_xlabel("nx")
    ax1.set_ylabel("depth-profile $L_2$ error [%]")
    ax1.set_title("profile error vs grid (gated levels 200/400)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "figure_dam_break_scheme.png", dpi=150)

    for s, rows in data.items():
        print(s, [(r["nx"], round(r["profile_l2_pct"], 3), round(r["front_err_pct"], 3)) for r in rows])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())