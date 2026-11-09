"""p128w3-n09 CFD maturity: turbulent-channel wall-law ladder (TensorLBM).

Three friction Reynolds numbers (180 / 395 / 590) are run with the repository's
body-force-driven channel LES; the mean profile is reduced to wall units and
compared with the two analytic anchors of the mission (viscous sublayer and the
log law), and the Reynolds shear stress is taken from the accumulated second
moments.

Outputs ``artifacts/p128w3_cfd_maturity/channel_ladder.json`` plus figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.turbulent_channel import TurbulentChannelConfig, run_turbulent_channel

OUT = Path("artifacts/p128w3_cfd_maturity")
FIG = OUT / "figures"
KAPPA_THEORY = 0.41
B_LOG = 5.2


def log_law(y_plus: np.ndarray) -> np.ndarray:
    return 2.44 * np.log(y_plus) + B_LOG


def analyse(metadata: dict, nu: float, u_tau: float) -> dict:
    profile = metadata["turbulence_statistics_runtime"]["wall_profile"]
    mean_u = np.asarray(profile["mean_u"], dtype=float)
    uv = np.asarray(profile["uv"], dtype=float)
    ny = mean_u.size
    rows = np.arange(ny, dtype=float) - 0.5  # distance from the lower wall centre
    y_plus = rows * u_tau / nu
    fluid = (y_plus > 0.5) & (y_plus < 0.95 * metadata["derived"]["re_tau"])
    u_plus = np.where(fluid, mean_u / u_tau, np.nan)
    sub = fluid & (y_plus < 5.0)
    log_region = fluid & (y_plus > 30.0) & (y_plus < 100.0)
    out: dict[str, float] = {}
    if sub.sum() >= 2:
        err = (u_plus[sub] - y_plus[sub]) / y_plus[sub]
        out["sublayer_rms_error_pct"] = float(100.0 * np.sqrt(np.mean(err**2)))
        out["sublayer_max_error_pct"] = float(100.0 * np.max(np.abs(err)))
        out["sublayer_points"] = int(sub.sum())
    if log_region.sum() >= 4:
        ref = log_law(y_plus[log_region])
        err = (u_plus[log_region] - ref) / ref
        out["log_law_rms_error_pct"] = float(100.0 * np.sqrt(np.mean(err**2)))
        slope, intercept = np.polyfit(np.log(y_plus[log_region]), u_plus[log_region], 1)
        out["kappa_fit"] = float(1.0 / slope) if slope else float("nan")
        out["intercept_fit"] = float(intercept)
        out["log_region_points"] = int(log_region.sum())
    # Reynolds shear stress in wall units, from the accumulated second moment
    uv_plus = np.where(fluid, -uv / u_tau**2, np.nan)
    valid = fluid & np.isfinite(uv_plus)
    if valid.any():
        idx = int(np.nanargmax(uv_plus))
        out["uv_peak_plus"] = float(uv_plus[idx])
        out["uv_peak_y_plus"] = float(y_plus[idx])
        # analytic wall-layer momentum budget: 1 - y/h - dU+/dy+ ~ 1 - y/h - 1/(kappa y+)
        budget = 1.0 - y_plus[valid] / (u_tau / nu * (ny - 1) / 2.0) - 1.0 / (
            KAPPA_THEORY * y_plus[valid]
        )
        if budget.size:
            j = int(np.argmax(budget))
            out["uv_peak_budget_plus"] = float(budget[j])
            out["uv_peak_budget_y_plus"] = float(y_plus[valid][j])
            out["uv_peak_error_pct"] = float(
                100.0 * abs(out["uv_peak_plus"] - budget[j]) / abs(budget[j])
            )
    out["u_plus_center"] = float(mean_u[fluid][-1] / u_tau)
    return out, {"y_plus": y_plus, "u_plus": u_plus, "uv_plus": uv_plus}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cases = {180: 66, 395: 130, 590: 200}
    results: dict[str, object] = {}
    profiles: dict[str, dict] = {}
    for re_tau, ny in cases.items():
        cfg = TurbulentChannelConfig(
            nx=192,
            ny=ny,
            re_tau=float(re_tau),
            u_tau=0.01,
            smagorinsky_cs=0.1,
            n_steps=20000,
            averaging_start=8000,
            output_interval=5000,
            output_root=OUT / "channel_runs",
            run_name=f"re_tau_{re_tau}",
            seed=7,
            overwrite=True,
        )
        print(f"[channel] Re_tau={re_tau} ny={ny} u_tau={cfg.u_tau} nu={cfg.nu:.3e}",
              flush=True)
        run_dir = run_turbulent_channel(
            cfg,
            turbulence_statistics={"enabled": True, "start_step": 8000, "sample_every": 2},
        )
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        metrics, prof = analyse(metadata, cfg.nu, cfg.u_tau)
        metrics["steps"] = int(cfg.n_steps)
        metrics["averaging_samples"] = int(metadata.get("averaging_samples", 0))
        metrics["statistics_samples"] = int(
            metadata["turbulence_statistics_runtime"]["n_samples"]
        )
        results[str(re_tau)] = metrics
        profiles[str(re_tau)] = prof
        print(f"[channel] Re_tau={re_tau} -> {metrics}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    grid = np.logspace(0, 2.8, 200)
    for re_tau in cases:
        prof = profiles[str(re_tau)]
        axes[0].semilogx(prof["y_plus"], prof["u_plus"], "-", label=f"Re_tau={re_tau}")
        axes[1].plot(prof["y_plus"], prof["uv_plus"], "-", label=f"Re_tau={re_tau}")
    axes[0].semilogx(grid, grid, "k:", label="U+=y+")
    axes[0].semilogx(grid, log_law(grid), "k--", label="2.44 ln y+ + 5.2")
    axes[0].set_xlim(0.5, 600)
    axes[0].set_ylim(0, 26)
    axes[0].set_xlabel("y+")
    axes[0].set_ylabel("U+")
    axes[0].legend(fontsize=7)
    axes[0].set_title("mean velocity in wall units")
    axes[1].set_xlim(0, 300)
    axes[1].set_xlabel("y+")
    axes[1].set_ylabel(r"-$\langle u'v'\rangle$+")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Reynolds shear stress")
    fig.tight_layout()
    fig.savefig(FIG / "channel_wall_units.png", dpi=140)
    plt.close(fig)

    (OUT / "channel_ladder.json").write_text(
        json.dumps({"cases": results, "kappa_theory": KAPPA_THEORY}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("channel ladder done", flush=True)


if __name__ == "__main__":
    main()