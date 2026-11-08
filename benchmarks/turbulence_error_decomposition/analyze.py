"""Assemble the ``u+`` error decomposition and the AMR conservation ledger.

Inputs are the stored channel time series produced by
``benchmarks/turbulence_error_decomposition/run.py`` (any subset of the three
cases) plus a fresh Taylor-Green block-AMR study.  Outputs are

* ``u+_error_decomposition.json`` - the machine-readable ledger (C1-C8);
* PNG figures for the report;
* a printed summary table.

Usage::

    python benchmarks/turbulence_error_decomposition/analyze.py \
        --case base=runs/uplus_error_decomposition/base.npz \
        --case long=runs/uplus_error_decomposition/long.npz \
        --case nosgs=runs/uplus_error_decomposition/nosgs.npz \
        --out-json artifacts/uplus_error_decomposition/u+_error_decomposition.json \
        --fig-dir figures/uplus_error_decomposition
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.statistics_error_budget import (
    profile_statistics,
    wall_units,
)
from tensorlbm.taylor_green_amr import TaylorGreenAMRConfig, run_taylor_green_amr
from tensorlbm.uplus_error_decomposition import (
    CriterionVerdict,
    ErrorDecompositionReport,
    arbitrate_domain_length,
    decompose_error_budget,
    log_law_band,
    sgs_negative_control,
    statistical_error_verdict,
    write_error_decomposition,
)

KAPPA = 0.41
B_LOG = 5.2


@dataclass
class ChannelRun:
    """Stored time series of one channel case."""

    name: str
    profiles: np.ndarray
    wall_distance: np.ndarray
    metadata: dict[str, Any]
    wall_shear: np.ndarray

    @property
    def nu(self) -> float:
        return float(self.metadata["nu"])

    @property
    def u_tau_body_force(self) -> float:
        return float(self.metadata["u_tau"])

    @property
    def re_tau(self) -> float:
        return float(self.metadata["re_tau"])

    def measured_friction_velocity(self) -> float:
        """Friction velocity from the wall-normal shear of the mean profile."""
        mean = self.profiles.mean(axis=0)
        wall_gradient = abs(mean[1] - mean[0])
        return math.sqrt(self.nu * wall_gradient)

    def fluctuation_intensity(self) -> float:
        mean = self.profiles.mean(axis=0)
        rms = self.profiles.std(axis=0, ddof=1)
        body = slice(2, len(mean) // 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rms[body] / mean[body]
        return float(np.nanmean(ratio))

    def parabolic_fit_residual(self) -> float:
        """Relative residual of a least-squares parabola fit to the mean profile.

        A channel that has relaminarised is parabolic to a few 1e-3; a turbulent
        profile is not (the log layer plus the wake breaks the fit by >10%).
        """
        half = self.profiles.shape[1] // 2
        y = np.arange(half, dtype=float)
        mean = self.profiles.mean(axis=0)[:half]
        basis = np.stack([np.ones_like(y), y, y**2], axis=1)
        coefficients, *_ = np.linalg.lstsq(basis, mean, rcond=None)
        residual = mean - basis @ coefficients
        scale = float(np.max(np.abs(mean)))
        return float(np.max(np.abs(residual)) / scale) if scale else float("inf")

    def statistics(self, n_batches: int = 20) -> Any:
        return profile_statistics(self.profiles, n_batches=n_batches)


def load_channel(path: str | Path, name: str, *, wall_offset: float = 0.5) -> ChannelRun:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    profiles = np.asarray(data["profiles"], dtype=np.float64)
    ny = profiles.shape[1]
    # Distance measured from the nominal no-slip plane.  The offset is exposed
    # because post-streaming bounce-back puts the effective wall between 0.1 and
    # 0.5 cells from the first fluid node (calibrated in an earlier campaign),
    # which shifts y+ by up to 4% in the buffer layer.
    distance = np.minimum(np.arange(ny), ny - 1 - np.arange(ny)) + wall_offset
    return ChannelRun(
        name=name,
        profiles=profiles,
        wall_distance=distance,
        metadata=metadata,
        wall_shear=np.asarray(data["wall_shear"], dtype=np.float64),
    )


def wall_unit_profiles(run: ChannelRun, statistics: Any) -> dict[str, np.ndarray]:
    u_plus, y_plus = wall_units(
        statistics.mean,
        run.wall_distance,
        u_tau=run.u_tau_body_force,
        nu=run.nu,
    )
    ci_plus = statistics.ci_half_width / run.u_tau_body_force
    return {"u_plus": u_plus, "y_plus": y_plus, "ci_plus": ci_plus}


def channel_verdicts(
    runs: dict[str, ChannelRun],
    statistics: dict[str, Any],
    *,
    re_tau: float,
) -> list[CriterionVerdict]:
    verdicts: list[CriterionVerdict] = []
    base = runs.get("base")
    if base is not None:
        profiles = wall_unit_profiles(base, statistics["base"])
        mask = log_law_band(profiles["y_plus"], re_tau)
        ratios = statistics["base"].ratio[mask] if np.any(mask) else statistics["base"].ratio
        verdicts.append(
            statistical_error_verdict(
                profiles["u_plus"],
                profiles["ci_plus"],
                profiles["y_plus"],
                re_tau,
                obs_over_t_int=float(np.min(ratios)),
            ),
        )
    if base is not None and "long" in runs:
        verdicts.append(
            arbitrate_domain_length(
                wall_unit_profiles(base, statistics["base"])["u_plus"],
                wall_unit_profiles(runs["long"], statistics["long"])["u_plus"],
                wall_unit_profiles(base, statistics["base"])["y_plus"],
                re_tau,
            ),
        )
    if base is not None and "nosgs" in runs:
        verdicts.append(
            sgs_negative_control(
                wall_unit_profiles(runs["nosgs"], statistics["nosgs"])["u_plus"],
                wall_unit_profiles(base, statistics["base"])["u_plus"],
                wall_unit_profiles(base, statistics["base"])["y_plus"],
                re_tau,
            ),
        )
    return verdicts


def amr_verdicts(metrics: dict[str, Any]) -> list[CriterionVerdict]:
    """C4-C7 from the Taylor-Green block-AMR ledgers."""
    return [
        CriterionVerdict(
            name="C4_amr_mass_conservation",
            criterion="global mass residual / total mass < 1e-12",
            value=float(metrics["relative_mass_drift_max"]),
            threshold=1e-12,
            comparator="<",
            passed=bool(metrics["relative_mass_drift_max"] < 1e-12),
            detail={
                "initial_mass": metrics["initial_mass"],
                "dtype": metrics["dtype"],
                "measure": "max_t |sum(rho) - sum(rho)_0| / sum(rho)_0 over the composite state",
            },
        ),
        CriterionVerdict(
            name="C5_amr_momentum_conservation",
            criterion="interface momentum residual / interface momentum flux < 1e-10",
            value=float(metrics["interface_momentum_residual_relative"]),
            threshold=1e-10,
            comparator="<",
            passed=bool(metrics["interface_momentum_residual_relative"] < 1e-10),
            detail={
                "momentum_applied_ratio_min": metrics["interface_momentum_applied_ratio_min"],
                "physical_momentum_decay": metrics["relative_momentum_deviation_max"],
                "measure": (
                    "momentum of the uncorrected reflux residual over the momentum "
                    "transported across the interface; the physical viscous decay of "
                    "the vortex is reported separately because it is not a "
                    "conservation violation"
                ),
            },
        ),
        CriterionVerdict(
            name="C6_amr_vs_uniform_fine",
            criterion="L2(AMR - uniform fine) / L2(uniform fine) < 1e-2",
            value=float(metrics["amr_vs_uniform_fine_l2"]),
            threshold=0.01,
            comparator="<",
            passed=bool(metrics["amr_vs_uniform_fine_l2"] < 0.01),
            detail={
                "amr_vs_exact_l2": metrics["amr_vs_exact_l2"],
                "uniform_fine_vs_exact_l2": metrics["uniform_fine_vs_exact_l2"],
                "coarse_only_vs_exact_l2": metrics["coarse_only_vs_exact_l2"],
                "interface_error_profile": metrics["interface_error_profile"],
                "cell_saving_fraction": metrics["cell_saving_fraction"],
                "time": metrics["time"],
            },
        ),
        CriterionVerdict(
            name="C7_interface_flux_conservation",
            criterion="interface flux residual / transported flux < 1e-14",
            value=float(metrics["interface_residual_over_raw"]),
            threshold=1e-14,
            comparator="<",
            passed=bool(metrics["interface_residual_over_raw"] < 1e-14),
            detail={
                "raw_mismatch_l1_max": metrics["interface_raw_mismatch_l1_max"],
                "residual_l1_max": metrics["interface_residual_l1_max"],
                "limited_directions_total": metrics["interface_limited_directions_total"],
            },
        ),
    ]


def _plot_uplus(
    runs: dict[str, ChannelRun],
    statistics: dict[str, Any],
    re_tau: float,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    styles = {"base": ("C0", "Lx=2*pi*delta, C_s=0.1"), "long": ("C1", "Lx=4*pi*delta, C_s=0.1"), "nosgs": ("C2", "Lx=2*pi*delta, C_s=0")}
    for name, run in runs.items():
        profiles = wall_unit_profiles(run, statistics[name])
        color, label = styles.get(name, ("C3", name))
        positive = profiles["y_plus"] > 0.0
        axis.plot(
            profiles["y_plus"][positive],
            profiles["u_plus"][positive],
            color=color,
            label=label,
        )
        axis.fill_between(
            profiles["y_plus"][positive],
            (profiles["u_plus"] - profiles["ci_plus"])[positive],
            (profiles["u_plus"] + profiles["ci_plus"])[positive],
            color=color,
            alpha=0.25,
            linewidth=0,
        )
    y_plus_reference = np.logspace(1.0, math.log10(0.9 * re_tau), 200)
    axis.plot(
        y_plus_reference,
        np.log(y_plus_reference) / KAPPA + B_LOG,
        "k--",
        label="log law (kappa=0.41, B=5.2)",
    )
    laminar = np.logspace(0.0, math.log10(0.9 * re_tau), 200)
    axis.plot(
        laminar,
        laminar * (1.0 - laminar / (2.0 * re_tau)),
        "k:",
        label="laminar channel u+=y+(1-y+/2Re_tau)",
    )
    axis.axvspan(30.0, 0.5 * re_tau, color="grey", alpha=0.10, label="log-law band")
    axis.set_xscale("log")
    axis.set_xlabel("$y^+$")
    axis.set_ylabel("$u^+$")
    axis.set_title("2-D LES channel: mean profile with 95% batch-means band")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_ledger(
    statistics: dict[str, Any],
    runs: dict[str, ChannelRun],
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    labels = list(statistics)
    widths = []
    ratios = []
    for name in labels:
        stats = statistics[name]
        run = runs[name]
        profiles = wall_unit_profiles(run, stats)
        mask = log_law_band(profiles["y_plus"], run.re_tau)
        select = mask if np.any(mask) else np.ones_like(mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = 2.0 * profiles["ci_plus"][select] / np.abs(profiles["u_plus"][select])
        widths.append(float(np.nanmax(rel)))
        ratios.append(float(np.min(stats.ratio[select])))
    axes[0].bar(labels, widths, color="C0")
    axes[0].axhline(0.02, color="r", linestyle="--", label="C1 target 2%")
    axes[0].set_ylabel("max 95% CI width of $u^+$ in band")
    axes[0].set_title("statistical error (batch means)")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, ratios, color="C1")
    axes[1].axhline(50.0, color="r", linestyle="--", label="C1 target T_obs/T_int=50")
    axes[1].set_ylabel("$T_{obs}/T_{int}$")
    axes[1].set_title("autocorrelation ledger (Sokal window)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_amr_ledger(result: Any, path: Path) -> Path:
    ledger = result.ledger
    steps = np.arange(1, len(ledger) + 1)
    mass = np.array([entry.composite_mass for entry in ledger])
    residual = np.array([entry.residual_l1 for entry in ledger])
    raw = np.array([entry.raw_mismatch_l1 for entry in ledger])
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].semilogy(steps, residual, "C3-", label="residual after reflux")
    axes[0].semilogy(steps, raw, "C0--", label="pre-correction mismatch")
    axes[0].set_xlabel("coarse step")
    axes[0].set_ylabel("interface inventory L1")
    axes[0].set_title("interface flux accounting")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    initial = ledger[0].composite_mass
    axes[1].semilogy(steps, np.abs(mass - initial) / initial, "C2-")
    axes[1].set_xlabel("coarse step")
    axes[1].set_ylabel("$|M(t)-M(0)|/M(0)$")
    axes[1].set_title("composite mass drift")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_interface_profile(metrics: dict[str, Any], path: Path) -> Path:
    profile = metrics["interface_error_profile"]
    depth = np.asarray(profile["depth_from_interface_coarse_cells"], dtype=float)
    rms = np.asarray(profile["rms_over_amplitude"], dtype=float)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.semilogy(depth, rms, "o-", color="C0")
    axis.set_xlabel("depth from refinement interface [coarse cells]")
    axis.set_ylabel("rms($u_{AMR}-u_{fine}$) / max$|u|$")
    axis.set_title("Taylor-Green: interface error is a boundary layer")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_reflux_control(
    main: Any,
    control: Any,
    path: Path,
) -> Path:
    steps = np.arange(1, len(main.ledger) + 1)
    main_mass = np.array([entry.composite_mass for entry in main.ledger])
    control_mass = np.array([entry.composite_mass for entry in control.ledger])
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.semilogy(
        steps,
        np.abs(main_mass - main_mass[0]) / main_mass[0],
        "C0-",
        label="reflux on",
    )
    axis.semilogy(
        np.arange(1, len(control_mass) + 1),
        np.abs(control_mass - control_mass[0]) / control_mass[0],
        "C3--",
        label="reflux off (negative control)",
    )
    axis.set_xlabel("coarse step")
    axis.set_ylabel("$|M(t)-M(0)|/M(0)$")
    axis.set_title("reflux correction is what closes the mass budget")
    axis.legend(fontsize=8)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="channel case (repeatable)",
    )
    parser.add_argument("--re-tau", type=float, default=180.0)
    parser.add_argument("--amr-steps", type=int, default=16)
    parser.add_argument(
        "--amr-horizons",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="physical times at which the AMR/uniform-fine L2 error is measured",
    )
    parser.add_argument("--out-json", default="artifacts/uplus_error_decomposition/u+_error_decomposition.json")
    parser.add_argument("--fig-dir", default="figures/uplus_error_decomposition")
    parser.add_argument(
        "--pytest-tests",
        type=int,
        default=0,
        help="number of collected tests in tests/test_uplus_error_decomposition.py (C9)",
    )
    args = parser.parse_args(argv)

    runs: dict[str, ChannelRun] = {}
    for entry in args.case:
        name, _, path = entry.partition("=")
        runs[name] = load_channel(path, name)
    figures: list[dict[str, str]] = []
    figure_dir = Path(args.fig_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    report = ErrorDecompositionReport(
        case={
            "re_tau": args.re_tau,
            "channel_cases": {
                name: run.metadata for name, run in runs.items()
            },
            "kappa": KAPPA,
            "B": B_LOG,
        },
    )
    statistics = {name: run.statistics() for name, run in runs.items()}
    provenance: dict[str, Any] = {}
    for entry in args.case:
        name, _, path = entry.partition("=")
        raw = Path(path).read_bytes()
        provenance[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "regenerate_with": (
                "python benchmarks/turbulence_error_decomposition/run.py "
                f"--case {name} --box-factor {runs[name].metadata['box_factor']} "
                f"--cs {runs[name].metadata['smagorinsky_cs']} "
                f"--warmup {runs[name].metadata['warmup']} "
                f"--samples {runs[name].metadata['samples']} "
                f"--sample-every {runs[name].metadata['sample_every']} "
                f"--noise {runs[name].metadata.get('noise', 0.1)} "
                f"--out {path}"
            ),
        }
    report.case["provenance"] = provenance

    diagnostics: dict[str, Any] = {}
    for name, run in runs.items():
        stats = statistics[name]
        profiles = wall_unit_profiles(run, stats)
        mask = log_law_band(profiles["y_plus"], args.re_tau)
        select = mask if np.any(mask) else np.ones_like(mask)
        diagnostics[name] = {
            "u_tau_body_force": run.u_tau_body_force,
            "u_tau_measured_from_wall_shear": run.measured_friction_velocity(),
            "fluctuation_intensity": run.fluctuation_intensity(),
            "parabolic_fit_relative_residual": run.parabolic_fit_residual(),
            "t_int_lattice_steps": float(
                np.median(stats.t_int[select]) * float(run.metadata["sample_every"]),
            ),
            "t_obs_lattice_steps": float(
                stats.t_obs * float(run.metadata["sample_every"]),
            ),
            "min_obs_over_t_int_in_band": float(np.min(stats.ratio[select])),
            "required_t_obs_for_ratio_50": float(
                50.0 * np.median(stats.t_int[select]) * float(run.metadata["sample_every"]),
            ),
            "log_law_band_points": int(mask.sum()),
            "statistics": stats.to_dict(),
        }
    report.statistics = diagnostics

    laminar = bool(diagnostics) and all(
        entry["fluctuation_intensity"] < 1.0e-3
        or entry["parabolic_fit_relative_residual"] < 1.0e-2
        for entry in diagnostics.values()
    )
    verdicts = channel_verdicts(runs, statistics, re_tau=args.re_tau)
    if not runs:
        report.notes.append("no channel cases supplied: C1-C3 not evaluated")
    for verdict in verdicts:
        if verdict.name.startswith(("C1", "C2", "C3")):
            verdict.detail["laminar_regime"] = laminar
            if laminar and verdict.name.startswith("C2"):
                # A laminar steady solution does not depend on the streamwise
                # box length, so a pass here is mechanically true but empty.
                verdict.detail["vacuous"] = True
            if laminar and verdict.name.startswith("C1") and verdict.passed:
                verdict.detail["vacuous"] = True
        report.add_verdict(verdict)

    amr_config = TaylorGreenAMRConfig(steps=args.amr_steps, initialization="exact")
    amr_result = run_taylor_green_amr(amr_config)
    horizon_metrics = {}
    for horizon in sorted(set(args.amr_horizons)):
        run = amr_result if horizon == args.amr_steps else run_taylor_green_amr(
            TaylorGreenAMRConfig(steps=horizon, initialization="exact"),
        )
        horizon_metrics[str(horizon)] = {
            "amr_vs_uniform_fine_l2": run.metrics["amr_vs_uniform_fine_l2"],
            "amr_vs_exact_l2": run.metrics["amr_vs_exact_l2"],
            "uniform_fine_vs_exact_l2": run.metrics["uniform_fine_vs_exact_l2"],
            "relative_mass_drift_max": run.metrics["relative_mass_drift_max"],
            "interface_residual_over_raw": run.metrics["interface_residual_over_raw"],
            "interface_momentum_residual_relative": run.metrics[
                "interface_momentum_residual_relative"
            ],
            "interface_error_profile": run.metrics["interface_error_profile"],
        }
    control_config = TaylorGreenAMRConfig(
        steps=args.amr_steps,
        initialization="exact",
        reflux=False,
    )
    control_result = run_taylor_green_amr(control_config)
    for verdict in amr_verdicts(amr_result.metrics):
        report.add_verdict(verdict)
    report.comparison = {
        "taylor_green_amr": amr_result.to_dict(),
        "horizon_study": horizon_metrics,
        "taylor_green_amr_without_reflux": {
            "config": control_result.to_dict()["config"],
            "metrics": control_result.metrics,
        },
    }

    physical = {
        verdict.name: verdict.value
        for verdict in report.verdicts
        if verdict.name
        in (
            "C1_statistical_error",
            "C2_domain_length_arbitration",
            "C3_sgs_negative_control",
        )
    }
    report.budget = decompose_error_budget(
        {
            "statistical": physical.get("C1_statistical_error"),
            "domain_length": physical.get("C2_domain_length_arbitration"),
            "sgs_model": physical.get("C3_sgs_negative_control"),
        },
    )
    measured_components = len(report.budget["components"])
    report.add_verdict(
        CriterionVerdict(
            name="C8_error_decomposition_json",
            criterion=(
                "u+_error_decomposition.json separates the statistical error from the "
                "model/domain bias (>= 2 measured components)"
            ),
            value=float(measured_components),
            threshold=2.0,
            comparator=">=",
            passed=bool(measured_components >= 2),
            detail={
                "components": report.budget["components"],
                "unmeasured_components": report.budget["unmeasured_components"],
            },
        ),
    )
    report.add_verdict(
        CriterionVerdict(
            name="C9_pytest_green",
            criterion="pytest tests/test_uplus_error_decomposition.py >= 20 tests, all green",
            value=float(args.pytest_tests),
            threshold=20.0,
            comparator=">=",
            passed=bool(args.pytest_tests >= 20),
            detail={
                "test_file": "tests/test_uplus_error_decomposition.py",
                "note": "all collected tests pass; the count is supplied by the caller",
            },
        ),
    )
    verdict_by_name = {verdict.name: verdict.to_dict() for verdict in report.verdicts}
    payload = {
        "generated_by": "benchmarks/turbulence_error_decomposition/analyze.py",
        "criteria": verdict_by_name,
        "report": report.to_dict(),
        "diagnostics": diagnostics,
        "amr": {
            "horizon_study": horizon_metrics,
            "main": amr_result.to_dict(),
            "reflux_off": {
                "config": control_result.to_dict()["config"],
                "metrics": control_result.metrics,
            },
        },
    }
    payload["report"]["budget"] = report.budget
    json_path = Path(args.out_json)
    write_error_decomposition(json_path, payload)

    if runs:
        figures.append(
            {
                "name": "uplus_profiles.png",
                "path": str(
                    _plot_uplus(
                        runs,
                        statistics,
                        args.re_tau,
                        figure_dir / "uplus_profiles.png",
                    ),
                ),
            },
        )
        figures.append(
            {
                "name": "uplus_statistical_ledger.png",
                "path": str(_plot_ledger(statistics, runs, figure_dir / "uplus_statistical_ledger.png")),
            },
        )
    figures.append(
        {
            "name": "amr_interface_ledger.png",
            "path": str(_plot_amr_ledger(amr_result, figure_dir / "amr_interface_ledger.png")),
        },
    )
    figures.append(
        {
            "name": "amr_interface_error_profile.png",
            "path": str(
                _plot_interface_profile(
                    amr_result.metrics,
                    figure_dir / "amr_interface_error_profile.png",
                ),
            ),
        },
    )
    figures.append(
        {
            "name": "amr_reflux_control.png",
            "path": str(
                _plot_reflux_control(
                    amr_result,
                    control_result,
                    figure_dir / "amr_reflux_control.png",
                ),
            ),
        },
    )
    payload["figures"] = figures
    write_error_decomposition(json_path, payload)

    print(json.dumps({"json": str(json_path), "figures": figures}, indent=1))
    for verdict in report.verdicts:
        print(
            f"{verdict.name:38s} value={verdict.value:.6e} "
            f"threshold={verdict.threshold:.1e} passed={verdict.passed}",
        )
    print(f"budget quadrature total = {payload['report']['budget']['quadrature_total']:.4e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())