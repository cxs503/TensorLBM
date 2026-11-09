#!/usr/bin/env python
"""Run the TensorLBM MHD physics-surface campaign (Hartmann ladder + thruster ledger).

Writes, under ``--outdir`` (default ``artifacts/mhd_hartmann``):

* ``hartmann_ladder.json``       - every run, GCI table, criteria ``c1..c4``, ``c6``
* ``hartmann_profiles.csv``      - node-wise profiles vs the closed form
* ``hartmann_ladder_summary.csv``- per-run scalars
* ``thruster_ledger.json``       - thrust (two mouths), powers, efficiencies
* ``criteria.json``              - machine-checkable acceptance verdicts ``c1..c7``
* ``fig1_profiles.png`` ...      - figures used by the report

Usage::

    python scripts/run_mhd_campaign.py --outdir artifacts/mhd_hartmann --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm.mhd.hartmann import (  # noqa: E402
    HartmannConfig,
    hartmann_wall_shear_coefficient,
    run_hartmann_channel,
)
from tensorlbm.mhd.ladder import (  # noqa: E402
    DEFAULT_HA_SWEEP,
    DEFAULT_RESOLUTIONS,
    THRESHOLDS,
    evaluate_ladder_criteria,
    gci,
)
from tensorlbm.mhd.thruster import (  # noqa: E402
    MHDThrusterConfig,
    run_thruster_ledger,
)

NU_LATTICE = 0.1
U_MEAN_LATTICE = 0.01
STEP_CAP = 400_000
CHECK_EVERY = 1000
STEADY_TOL = 1e-10


def _run_one_ha(args: tuple[float, list[int], int]) -> tuple[float, list[dict]]:
    """Run all resolutions for one Hartmann number with grid sequencing."""
    ha, ny_list, step_cap = args
    torch.set_num_threads(1)
    out: list[dict] = []
    seed: tuple[np.ndarray, np.ndarray] | None = None
    for ny in ny_list:
        u_init = None
        if seed is not None:
            geom = HartmannConfig(ny=ny, ha=ha, u_mean=U_MEAN_LATTICE, nu=NU_LATTICE)
            u_init = tuple(float(v) for v in np.interp(geom.y_hat(), seed[0], seed[1]))
        cfg = HartmannConfig(
            ny=ny,
            ha=ha,
            u_mean=U_MEAN_LATTICE,
            nu=NU_LATTICE,
            drive=(None if ha == 0.0 else 0.0),
            steps=step_cap,
            check_every=CHECK_EVERY,
            tol=STEADY_TOL,
            u_init=u_init,
        )
        res = run_hartmann_channel(cfg)
        seed = (res.y_hat, res.u_profile)
        out.append(res.to_dict() | {"profile": res.profile_rows()})
        print(
            f"[mhd] Ha={ha:5.2f} ny={ny:4d} steps={res.steps:6d} conv={res.converged} "
            f"rel_linf={res.rel_linf:.2e} local={res.rel_local_core:.2e} "
            f"closure={res.closure_residual:.1e}",
            flush=True,
        )
    return ha, out


def load_ladder_artifacts(
    outdir: Path,
) -> tuple[dict[tuple[float, int], dict], dict[int, list[int]]]:
    """Reload a previously written ladder (run scalars + profiles) from ``outdir``.

    Used by ``--reuse-ladder`` so that re-evaluating the acceptance criteria does
    not re-run the (expensive) lattice integrations.
    """
    ladder_json = json.loads((outdir / "hartmann_ladder.json").read_text())
    runs: dict[tuple[float, int], dict] = {}
    for run in ladder_json["runs"]:
        ha = float(run["ha"])
        ny = int(run["ny"])
        runs[(ha, ny)] = {k: v for k, v in run.items() if k not in {"ha", "ny"}}
        runs[(ha, ny)]["profile"] = []
    profiles_path = outdir / "hartmann_profiles.csv"
    with profiles_path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            ha_s, ny_s, y, u_lbm, u_an, abs_e, rel_e = line.strip().split(",")
            key = (float(ha_s), int(ny_s))
            runs[key]["profile"].append(
                {
                    "y_over_H": float(y),
                    "u_lbm": float(u_lbm),
                    "u_analytic": float(u_an),
                    "abs_error": float(abs_e),
                    "rel_error": float(rel_e),
                }
            )
    res_by_ha: dict[int, list[int]] = {
        int(float(ha)): sorted(int(n) for (h, n) in runs if h == float(ha))
        for ha in ladder_json["ha_values"]
    }
    return runs, res_by_ha


def run_campaign(
    outdir: Path,
    workers: int = 4,
    step_cap: int = STEP_CAP,
    reuse_ladder: bool = False,
) -> dict:
    """Run the full campaign and write every artefact; returns the criteria dict."""
    outdir.mkdir(parents=True, exist_ok=True)
    ha_list = list(DEFAULT_HA_SWEEP)
    ny_list = list(DEFAULT_RESOLUTIONS)
    # Ha = 20 needs a finer sequence: the Hartmann layer is H/Ha = 6.4 cells at
    # ny = 256, which is not enough for the 1e-3 profile criterion.
    res_by_ha: dict[float, list[int]] = {float(ha): list(ny_list) for ha in ha_list}
    res_by_ha[20.0] = [128, 256, 512]

    t0 = time.time()
    if reuse_ladder:
        runs, res_by_ha = load_ladder_artifacts(outdir)
        elapsed_ladder = 0.0
    else:
        payload = [(ha, res_by_ha[float(ha)], step_cap) for ha in ha_list]
        ctx = mp.get_context("fork")
        results: dict[float, list[dict]] = {}
        with ctx.Pool(processes=min(workers, len(payload))) as pool:
            for ha, run_items in pool.imap_unordered(_run_one_ha, payload):
                results[ha] = run_items
        elapsed_ladder = time.time() - t0
        runs = {}
        for ha, items in results.items():
            for item in items:
                runs[(float(ha), int(item["config"]["ny"]))] = item

    # GCI table on the dimensionless wall-shear coefficient and the L∞ error
    gci_table: dict[str, dict] = {}
    for ha in ha_list:
        ny_ha = res_by_ha[float(ha)][-3:]
        vals = [runs[(float(ha), int(ny))]["wall_shear_coefficient"] for ny in ny_ha]
        gci_table[f"wall_shear_coefficient_Ha{ha:g}"] = gci(vals[2], vals[1], vals[0], r=2.0) | {
            "values": {"fine": vals[2], "medium": vals[1], "coarse": vals[0]},
            "analytic": hartmann_wall_shear_coefficient(float(ha)),
        }
        errs = [runs[(float(ha), int(ny))]["rel_linf"] for ny in ny_ha]
        gci_table[f"profile_linf_error_Ha{ha:g}"] = gci(errs[2], errs[1], errs[0], r=2.0) | {
            "values": {"fine": errs[2], "medium": errs[1], "coarse": errs[0]}
        }

    ladder = _LadderView(ha_list, res_by_ha, runs, gci_table)
    ladder_criteria = evaluate_ladder_criteria(ladder)  # type: ignore[arg-type]

    # --- thruster ledger ----------------------------------------------------
    t1 = time.time()
    ledger = run_thruster_ledger(
        MHDThrusterConfig(ny=256, nu_lattice=NU_LATTICE, steps=300_000, check_every=1000),
        run_lattice=True,
    )
    elapsed_thruster = time.time() - t1
    thruster = ledger.to_dict()

    # --- thrust map over the jet-excess ratio (closed form, cheap) ----------
    sweep = []
    for theta in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        cfg = MHDThrusterConfig(jet_excess_ratio=theta)
        led = run_thruster_ledger(cfg, run_lattice=False)
        sweep.append(
            {
                "jet_excess_ratio": theta,
                "thrust_N": led.thrust_lorentz_n,
                "P_electric_W": led.power_electric_w,
                "eta": led.efficiency,
                "eta_jet_ideal": led.efficiency_jet_ideal,
                "eta_propeller_ideal": led.efficiency_propeller_ideal,
                "CT": led.disk_loading_ct,
            }
        )

    # --- criteria -----------------------------------------------------------
    criteria: dict[str, dict] = dict(ladder_criteria)
    criteria["c5_thrust"] = {
        "id": "c5_thrust",
        "value": ledger.thrust_closure_residual,
        "threshold": THRESHOLDS["c5_thrust_closure"],
        "passed": ledger.thrust_closure_residual <= THRESHOLDS["c5_thrust_closure"],
        "note": "|T_lorentz_volume - T_momentum_flux| / T_lorentz_volume",
        "lattice_value": ledger.lattice.get("closure_residual_lbm"),
        "thrust_lorentz_N": ledger.thrust_lorentz_n,
        "thrust_momentum_N": ledger.thrust_momentum_n,
    }
    criteria["c7_energy"] = {
        "id": "c7_energy",
        "value": ledger.energy_residual,
        "threshold": THRESHOLDS["c7_energy_guard"],
        "passed": ledger.energy_residual <= THRESHOLDS["c7_energy_guard"],
        "note": "|P_electric - P_joule - P_viscous - P_hydraulic| / P_electric",
        "lattice_value": ledger.lattice.get("energy_residual_lattice"),
        "efficiency": ledger.efficiency,
        "efficiency_below_one": ledger.efficiency < 1.0,
    }
    criteria["c7_energy"]["passed"] = bool(
        criteria["c7_energy"]["passed"] and ledger.efficiency < 1.0
    )

    all_passed = all(
        bool(c.get("passed")) for key, c in criteria.items() if not c.get("supplementary")
    )
    verdict = {
        "all_passed": all_passed,
        "criteria": criteria,
        "thresholds": THRESHOLDS,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "nu_lattice": NU_LATTICE,
            "u_mean_lattice": U_MEAN_LATTICE,
            "step_cap": step_cap,
            "steady_tol": STEADY_TOL,
            "check_every": CHECK_EVERY,
            "scheme": "D2Q9 BGK + Guo forcing, half-way bounce-back walls, float64",
            "elapsed_ladder_s": elapsed_ladder,
            "elapsed_thruster_s": elapsed_thruster,
        },
    }

    # --- artefacts ----------------------------------------------------------
    (outdir / "hartmann_ladder.json").write_text(
        json.dumps(
            {
                "ha_values": ha_list,
                "resolutions_by_ha": {f"{ha:g}": list(v) for ha, v in res_by_ha.items()},
                "runs": [
                    {k: v for k, v in run.items() if k != "profile"} | {"ha": ha, "ny": ny}
                    for (ha, ny), run in runs.items()
                ],
                "gci": gci_table,
                "criteria": ladder_criteria,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (outdir / "hartmann_profiles.csv").open("w", encoding="utf-8") as fh:
        fh.write("ha,ny,y_over_H,u_lbm,u_analytic,abs_error,rel_error\n")
        for (ha, ny), run in sorted(runs.items()):
            for row in run["profile"]:
                fh.write(
                    f"{ha:g},{ny},{row['y_over_H']:.10f},{row['u_lbm']:.12e},"
                    f"{row['u_analytic']:.12e},{row['abs_error']:.6e},{row['rel_error']:.6e}\n"
                )
    with (outdir / "hartmann_ladder_summary.csv").open("w", encoding="utf-8") as fh:
        keys = [
            "ha",
            "ny",
            "steps",
            "converged",
            "u_mean",
            "u_mean_analytic",
            "rel_linf",
            "rel_l2",
            "rel_local_core",
            "rel_max_pointwise",
            "wall_shear_coefficient",
            "closure_residual",
            "max_mach",
        ]
        fh.write(",".join(keys) + "\n")
        for (ha, ny), run in sorted(runs.items()):
            fh.write(",".join([f"{ha:g}", str(ny)] + [str(run[k]) for k in keys[2:]]) + "\n")
    (outdir / "thruster_ledger.json").write_text(
        json.dumps({"ledger": thruster, "sweep": sweep}, indent=2), encoding="utf-8"
    )
    (outdir / "criteria.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    _plot_profiles(runs, ha_list, res_by_ha, outdir)
    _plot_convergence(runs, ha_list, res_by_ha, outdir)
    _plot_field(runs, outdir)
    _plot_thruster(sweep, thruster, outdir)

    print(json.dumps({k: (v.get("value"), v.get("passed")) for k, v in criteria.items()}, indent=1))
    print(f"verdict all_passed={all_passed}")
    return verdict


class _LadderView:
    """Adapter exposing the attributes :func:`evaluate_ladder_criteria` needs."""

    def __init__(self, ha_values, res_by_ha, runs, gci_table):
        self.ha_values = ha_values
        self.res_by_ha = res_by_ha
        self.resolutions = sorted({ny for vals in res_by_ha.values() for ny in vals})
        self._runs = runs
        self.gci_table = gci_table

    def finest(self, ha: float):
        run = self._runs[(float(ha), int(self.res_by_ha[float(ha)][-1]))]
        return _RunView(run)


class _RunView:
    """Attribute view over a run dict."""

    def __init__(self, run: dict):
        self._run = run

    def __getattr__(self, name):
        if name == "config":
            return _CfgView(self._run["config"])
        if name == "u_mean":
            return self._run["u_mean"]
        if name == "u_mean_analytic":
            return self._run["u_mean_analytic"]
        if name == "rel_max_pointwise":
            return self._run["rel_max_pointwise"]
        if name == "rel_linf":
            return self._run["rel_linf"]
        if name == "rel_l2":
            return self._run["rel_l2"]
        if name == "rel_local_core":
            return self._run["rel_local_core"]
        if name == "u_profile_analytic":
            return np.array([r["u_analytic"] for r in self._run["profile"]])
        if name == "u_profile":
            return np.array([r["u_lbm"] for r in self._run["profile"]])
        raise AttributeError(name)


class _CfgView:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    @property
    def ny(self):
        return self._cfg["ny"]

    @property
    def ha(self):
        return self._cfg["ha"]


def _plot_profiles(runs, ha_list, res_by_ha, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=140)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(ha_list)))
    for color, ha in zip(colors, ha_list):
        run = runs[(float(ha), int(res_by_ha[float(ha)][-1]))]
        y = np.array([r["y_over_H"] for r in run["profile"]])
        u = np.array([r["u_lbm"] for r in run["profile"]])
        ua = np.array([r["u_analytic"] for r in run["profile"]])
        ax.plot(u / run["u_mean"], y, "-", color=color, lw=2.0, label=f"LBM Ha={ha:g}")
        ax.plot(ua / run["u_mean_analytic"], y, "o", ms=3.2, mfc="none", mec=color, mew=0.9)
    ax.set_xlabel(r"$u/\bar u$ (-)")
    ax.set_ylabel(r"$y/H$ (-)")
    ax.set_title("Hartmann duct: LBM (lines, finest grid per Ha) vs closed form (circles)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_profiles.png")
    plt.close(fig)


def _plot_convergence(runs, ha_list, res_by_ha, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=140)
    for ha in ha_list:
        nys = res_by_ha[float(ha)]
        errs = [runs[(float(ha), int(ny))]["rel_linf"] for ny in nys]
        ax.loglog(nys, errs, "o-", label=f"Ha={ha:g}")
    ref = np.array(sorted({ny for v in res_by_ha.values() for ny in v}), dtype=float)
    ax.loglog(ref, 3.0 * ref**-2.0, "k--", lw=1.0, label=r"$O(h^2)$ reference")
    ax.set_xlabel("ny (fluid rows across channel)")
    ax.set_ylabel(r"$\|u_{LBM}-u_{exact}\|_\infty/\bar u$")
    ax.set_title("Grid convergence of the Hartmann duct profile error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_grid_convergence.png")
    plt.close(fig)


def _plot_field(runs, outdir: Path) -> None:
    ha = 10.0 if (10.0, 256) in runs else sorted(runs)[0][0]
    ny = max(ny for (h, ny) in runs if h == ha)
    # rebuild the 2D field from the 1D profile of that run is not stored in JSON;
    # show the profile-based field map (streamwise uniform, which the solver
    # enforces: the run reports max deviation from x-uniformity below 1e-12).
    run = runs[(ha, ny)]
    u = np.array([r["u_lbm"] for r in run["profile"]]) / run["u_mean"]
    field = np.tile(u[:, None], (1, 8))
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    mesh = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=(0.0, float(field.shape[1]), -1.0, 1.0),
        cmap="magma",
    )
    fig.colorbar(mesh, ax=ax, label=r"$u/\bar u$ (-)")
    ax.set_xlabel("periodic streamwise cell index")
    ax.set_ylabel(r"$y/H$ (-)")
    ax.set_title(f"Steady velocity field of the MHD duct (Ha={ha:g}, ny={ny})")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_field_Ha10.png")
    plt.close(fig)


def _plot_thruster(sweep, ledger: dict, outdir: Path) -> None:
    thetas = [s["jet_excess_ratio"] for s in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=140)
    ax1.plot(thetas, [s["eta"] for s in sweep], "o-", label=r"MHD total $\eta$")
    ax1.plot(thetas, [s["eta_jet_ideal"] for s in sweep], "s--", label="ideal jet (Froude)")
    ax1.plot(
        thetas,
        [s["eta_propeller_ideal"] for s in sweep],
        "^:",
        label="same-size propeller (actuator disk)",
    )
    ax1.set_xlabel(r"jet excess ratio $\theta=(u_e-U)/U$")
    ax1.set_ylabel(r"efficiency (-)")
    ax1.set_title("MHD duct thruster vs propeller at the same thrust")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    energy = ledger["energy"]
    labels = [r"$P_{elec}$", r"$P_{joule}$", r"$P_{visc}$", r"$P_{hyd}$"]
    values = [
        energy["P_electric_W"],
        energy["P_joule_W"],
        energy["P_viscous_W"],
        energy["P_hydraulic_W"],
    ]
    colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a"]
    ax2.bar(labels, values, color=colors)
    ax2.set_ylabel("power (W)")
    ax2.set_title("Energy ledger (seawater reference design)")
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(values):
        ax2.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_thruster_ledger.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/mhd_hartmann")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument(
        "--reuse-ladder",
        action="store_true",
        help="re-evaluate criteria/figures from the existing ladder artefacts",
    )
    args = parser.parse_args()
    verdict = run_campaign(
        Path(args.outdir),
        workers=args.workers,
        step_cap=args.step_cap,
        reuse_ladder=args.reuse_ladder,
    )
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
