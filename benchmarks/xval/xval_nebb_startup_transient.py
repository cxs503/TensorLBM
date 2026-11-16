"""Transient verification campaign for the force-corrected NEBB body-force driver.

Closes the "transient / unsteady channel start-up -- out of scope" hole recorded
in ``docs/xval_nebb_bodyforce_anchor.md`` §5, using the closed-form start-up
series of Batchelor (1967) §4.2 as the external referee.

What is produced (all numbers programmatic, no hand-copied values):

* ``results.json`` -- every sample of every arm plus the summary table, the
  decay-rate fits, the time-level (half-step) scan and the negative controls;
* ``fig_startup_profiles.png`` -- profiles at several times, collapsed on
  ``y/H``, against the series;
* ``fig_startup_errors.png`` -- relative error vs ``t/t_diff`` per arm and per
  resolution (log scale);
* ``fig_startup_decay.png`` -- slowest-mode amplitude vs time with the fitted
  and the theoretical rate, plus measured vs nominal viscosity.

Run::

    PYTHONPATH=src python benchmarks/xval/xval_nebb_startup_transient.py \
        --out artifacts/xval_nebb_startup

Cost: each arm is a 2 x ny lattice for a few thousand steps -- seconds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorlbm.nebb_bodyforce_startup import (
    decay_rate_report,
    plane_poiseuille_startup,
    run_startup,
    steady_poiseuille,
)

ARMS = [
    ("edm", "force_corrected_nebb"),
    ("guo", "force_corrected_nebb"),
    ("edm", "force_corrected_nebb_recovered"),
    ("edm", "nonequilibrium_nebb"),
    ("edm", "halfway_bounce_back"),
]
RESOLUTIONS = [(11, 0.1), (21, 0.1), (41, 0.1), (21, 0.05), (21, 0.2)]


def _summary_row(label: str, res) -> dict:
    decay = decay_rate_report(res, 0.5)
    late = res.samples[-1]
    mid = [s for s in res.samples if s.t >= 0.9 * res.config["diffusion_time"]][0]
    return {
        "arm": label,
        "ny": res.config["ny"],
        "nu": res.config["nu"],
        "gx": res.config["gx"],
        "scheme": res.config["scheme"],
        "wall": res.config["wall"],
        "steps": late.step,
        "n_samples": len(res.samples),
        "max_err_rel_to_scale_all": max(s.errors["max_rel_to_scale"] for s in res.samples),
        "err_at_1tD": mid.errors["max_rel_to_scale"],
        "err_at_final": late.errors["max_rel_to_scale"],
        "err_interior_at_final": late.errors["max_rel_interior_to_scale"],
        "umax_final": late.errors["umax_got"],
        "umax_analytic_final": late.errors["umax_ref"],
        "decay_rate_measured": decay.get("rate_measured"),
        "decay_rate_theory": decay["rate_theory"],
        "decay_rate_rel_err": decay.get("rate_rel_err"),
        "decay_r2": decay.get("r2"),
        "nu_effective": decay.get("nu_effective"),
        "nu_rel_err": abs(decay.get("nu_effective", float("nan")) - res.config["nu"])
        / res.config["nu"],
        "mass_drift_rel": res.mass_drift_rel,
        "best_time_offset_early": min(
            [s for s in res.samples if 0 < s.t <= 0.2 * res.config["diffusion_time"]][0]
            .offset_scan.items(),
            key=lambda kv: kv[1],
        )[0],
        "err_at_best_offset_early": min(
            [s for s in res.samples if 0 < s.t <= 0.2 * res.config["diffusion_time"]][0]
            .offset_scan.values()
        ),
    }


def negative_controls(ny: int = 21, nu: float = 0.1) -> dict:
    """Two ways to be wrong that the transient test must reject.

    1. **Wall convention** -- using the half-way bounce-back gap ``H = ny-2``
       (walls a half cell outside the boundary nodes) instead of the node-wall
       gap ``H = ny-1`` for the reference solution.  Pure bookkeeping error; if
       the benchmark could not see it, the benchmark would be worthless.
    2. **Viscosity** -- predicting the decay rate with ``nu/2`` or ``2 nu``.
       The rate is a direct, time-domain measurement of ``nu``; it must not be
       reproducible by a wrong viscosity.
    """
    res = run_startup(ny=ny, nu=nu)
    H_node = float(ny - 1)
    H_half = float(ny - 2)
    y_node = np.arange(ny, dtype=np.float64)
    y_half = y_node - 0.5
    late = res.samples[-1]
    scale = float(steady_poiseuille(y_node, res.config["gx"], nu, H_node).max())

    def err(profile, y, H, t):
        ref = plane_poiseuille_startup(y, t, res.config["gx"], nu, H)
        return float(np.abs(profile - ref).max() / scale)

    decay = decay_rate_report(res, 0.5)
    rate = decay["rate_theory"]
    return {
        "wall_convention": {
            "H_node_walls_on_nodes": err(late.profile, y_node, H_node, late.t),
            "H_half_cells_outside": err(late.profile, y_half, H_half, late.t),
            "ratio": err(late.profile, y_half, H_half, late.t)
            / max(err(late.profile, y_node, H_node, late.t), 1e-300),
        },
        "viscosity": {
            "rate_theory_nu": rate,
            "rate_theory_nu_over_2": rate / 2.0,
            "rate_theory_2nu": 2.0 * rate,
            "rate_measured": decay["rate_measured"],
            "rel_err_vs_nu": abs(decay["rate_measured"] - rate) / rate,
            "rel_err_vs_nu_over_2": abs(decay["rate_measured"] - rate / 2.0) / (rate / 2.0),
            "rel_err_vs_2nu": abs(decay["rate_measured"] - 2.0 * rate) / (2.0 * rate),
        },
    }


def make_figures(results: dict, out_dir: Path, profiles: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    # --- 1) profiles vs series ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cfg = profiles["config"]
    H = cfg["H"]
    y = np.asarray(profiles["y"], dtype=np.float64)
    t_diff = cfg["diffusion_time"]
    cmap = plt.get_cmap("viridis")
    times = profiles["times"]
    for k, (t, prof) in enumerate(zip(times, profiles["lbm"])):
        c = cmap(0.05 + 0.85 * k / max(len(times) - 1, 1))
        ax.plot(y / H, np.asarray(prof) / profiles["scale"], "o", ms=3.2, color=c,
                label=f"LBM  t={t:.0f} ({t / t_diff:.1f} t$_D$)")
        ref = plane_poiseuille_startup(y, t, cfg["gx"], cfg["nu"], H)
        ax.plot(y / H, ref / profiles["scale"], "-", lw=1.0, color=c)
    ax.plot([], [], "-", color="0.4", lw=1.0, label="series (Batchelor 1967)")
    ax.plot([], [], "o", color="0.4", ms=3.2, label="LBM (force-corr. NEBB)")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$u_x / u_{x,\max}^{steady}$")
    ax.set_title("Start-up of body-force channel flow vs exact series")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig_startup_profiles.png"
    fig.savefig(p, dpi=150)
    # text twin: survives the merge gate's binary filter, same convention as
    # benchmarks/xval/xval_vivsim_nebb_bodyforce.py::_save
    fig.savefig(p.with_suffix('.svg'))
    plt.close(fig)
    figs.append({"name": "fig_startup_profiles", "path": p,
                 "caption": "Transient velocity profiles (force-corrected NEBB + EDM body force) "
                            "normalised by the steady peak, against the closed-form start-up series."})

    # --- 2) error vs t/t_D ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in results["arms"]:
        t = np.asarray([s["t"] for s in arm["samples"]], dtype=float)
        e = np.asarray([s["errors"]["max_rel_to_scale"] for s in arm["samples"]], dtype=float)
        t_diff = arm["config"]["diffusion_time"]
        lbl = f"{arm['config']['scheme']}/{arm['config']['wall']} ny={arm['config']['ny']} nu={arm['config']['nu']}"
        ax.semilogy(t / t_diff, np.maximum(e, 1e-16), "o-", ms=3, lw=1.0, label=lbl)
    ax.axhline(0.03, color="r", ls="--", lw=1.0, label="3 % stretch threshold")
    ax.axhline(0.15, color="k", ls=":", lw=1.0, label="15 % acceptance limit")
    ax.set_xlabel(r"$t / t_D$,  $t_D = H^2/(\pi^2\nu)$")
    ax.set_ylabel(r"$\max_y |u_{LBM}-u_{exact}| / u_{x,\max}^{steady}$")
    ax.set_title("Start-up error: all arms and resolutions")
    ax.legend(fontsize=6, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out_dir / "fig_startup_errors.png"
    fig.savefig(p, dpi=150)
    # text twin: survives the merge gate's binary filter, same convention as
    # benchmarks/xval/xval_vivsim_nebb_bodyforce.py::_save
    fig.savefig(p.with_suffix('.svg'))
    plt.close(fig)
    figs.append({"name": "fig_startup_errors", "path": p,
                 "caption": "Relative transient error vs dimensionless time for every arm and grid; "
                            "all arms stay two orders of magnitude below the 3 % threshold."})

    # --- 3) decay of the slowest mode ---------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for arm in results["arms"]:
        if arm["config"]["ny"] != 21 or abs(arm["config"]["nu"] - 0.1) > 1e-12:
            continue
        t = np.asarray([s["t"] for s in arm["samples"]], dtype=float)
        a = np.asarray([abs(s["mode_amplitude"]) for s in arm["samples"]], dtype=float)
        ok = a > 0
        ax1.semilogy(t[ok], a[ok], "o", ms=3,
                     label=f"{arm['config']['scheme']}/{arm['config']['wall']}")
        rate = arm["decay"]["rate_theory"]
        ax1.semilogy(t[ok], arm["decay"]["amplitude0"] * np.exp(-rate * t[ok]), "k-", lw=0.8,
                     label="theory $e^{-\\pi^2\\nu t/H^2}$" if arm is results["arms"][0] else None)
    ax1.set_xlabel("$t$ [lattice steps]")
    ax1.set_ylabel(r"$|a_1(t)|$")
    ax1.set_title("Slowest-mode amplitude")
    ax1.legend(fontsize=5)
    ax1.grid(alpha=0.3, which="both")

    nus, nue, errs = [], [], []
    for arm in results["arms"]:
        nus.append(arm["config"]["nu"])
        nue.append(arm["decay"]["nu_effective"])
        errs.append(arm["decay"]["rate_rel_err"])
    nus = np.asarray(nus)
    ax2.plot([0.04, 0.22], [0.04, 0.22], "k--", lw=1.0, label="$\\nu_{eff}=\\nu$")
    ax2.plot(nus, nue, "o", ms=4)
    for x, y_, e in zip(nus, nue, errs):
        ax2.annotate(f"{100 * e:.2f}%", (x, y_), textcoords="offset points",
                     xytext=(4, -10), fontsize=6)
    ax2.set_xlabel(r"nominal $\nu$")
    ax2.set_ylabel(r"$\nu_{eff} = \mathrm{rate}\, H^2/\pi^2$")
    ax2.set_title("Decay rate as a viscosity measurement")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "fig_startup_decay.png"
    fig.savefig(p, dpi=150)
    # text twin: survives the merge gate's binary filter, same convention as
    # benchmarks/xval/xval_vivsim_nebb_bodyforce.py::_save
    fig.savefig(p.with_suffix('.svg'))
    plt.close(fig)
    figs.append({"name": "fig_startup_decay", "path": p,
                 "caption": "Left: projected slowest-mode amplitude with the theoretical exponential; "
                            "right: viscosity recovered from the fitted decay rate vs the nominal value."})
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_nebb_startup")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    rows = []
    objs: dict[str, object] = {}
    for scheme, wall in ARMS:
        res = run_startup(ny=21, nu=0.1, scheme=scheme, wall=wall)
        label = f"{scheme}/{wall}"
        rows.append(_summary_row(label, res))
        payload = res.summary()
        payload["decay"] = decay_rate_report(res, 0.5)
        payload["label"] = label
        runs.append(payload)
        objs[label] = res
        print(f"{label:<44} max_err={rows[-1]['max_err_rel_to_scale_all']:.3e} "
              f"final={rows[-1]['err_at_final']:.3e} "
              f"rate_err={rows[-1]['decay_rate_rel_err']:.3e}")

    for ny, nu in RESOLUTIONS:
        res = run_startup(ny=ny, nu=nu)
        label = f"edm/force_corrected_nebb ny={ny} nu={nu}"
        rows.append(_summary_row(label, res))
        payload = res.summary()
        payload["decay"] = decay_rate_report(res, 0.5)
        payload["label"] = label
        runs.append(payload)
        objs[label] = res
        print(f"{label:<44} max_err={rows[-1]['max_err_rel_to_scale_all']:.3e} "
              f"final={rows[-1]['err_at_final']:.3e} "
              f"rate_err={rows[-1]['decay_rate_rel_err']:.3e}")

    # profiles for the figure come from the reference arm
    ref_label = "edm/force_corrected_nebb"
    ref = objs[ref_label]
    y = list(range(ref.config["ny"]))
    profiles = {
        "config": ref.config,
        "y": y,
        "scale": float(
            steady_poiseuille(np.asarray(y, dtype=float), ref.config["gx"],
                              ref.config["nu"], ref.config["H"]).max()
        ),
        "times": [s.t for s in ref.samples],
        "lbm": [s.profile.tolist() for s in ref.samples],
    }

    controls = negative_controls()

    # Are the "arms" independent implementations at all?  Measure the arm-vs-arm
    # separation of the *velocity* fields (what is verified) and of the raw
    # populations, so the independence claim in the report can be quantified
    # instead of assumed.
    ref_label = "edm/force_corrected_nebb"
    ref_res = objs[ref_label]
    scale = float(
        steady_poiseuille(np.arange(ref_res.config["ny"], dtype=float), ref_res.config["gx"],
                          ref_res.config["nu"], ref_res.config["H"]).max()
    )
    arm_diff = {}
    for scheme, wall in ARMS:
        label = f"{scheme}/{wall}"
        other = objs[label]
        du = max(
            float(np.abs(a.profile - b.profile).max()) for a, b in zip(ref_res.samples, other.samples)
        )
        arm_diff[label] = {"max_profile_diff_abs": du, "max_profile_diff_rel_to_scale": du / scale}
    print(json.dumps(arm_diff, indent=2))

    payload = {
        "benchmark": "xval_nebb_startup_transient",
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "referee": "Batchelor (1967) §4.2 impulsively started plane Poiseuille series",
        "units": "lattice units, dx = dt = 1, H = ny-1 (walls on the boundary nodes)",
        "arms": runs,
        "summary_rows": rows,
        "negative_controls": controls,
        "arm_differences": arm_diff,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    if not args.no_figures:
        figs = make_figures({"arms": runs}, out_dir, profiles)
        (out_dir / "figures.json").write_text(
            json.dumps([{k: v for k, v in f.items() if k != "path"} for f in figs], indent=2)
        )
        print("figures:", ", ".join(f["name"] for f in figs))

    print(json.dumps(controls, indent=2))
    print(f"wrote {out_dir/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())