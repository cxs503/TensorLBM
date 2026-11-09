"""p128w3-n09 CFD maturity: VOF free-surface three-tier ladder (L1/L2/L3).

Writes ``artifacts/p128w3_cfd_maturity/vof_ladder.json`` and the figures used by
the mission report.  Every number in the JSON is measured on the simulated
fields; the analytic anchors are evaluated in-process from their closed forms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.vof_free_surface.ladder import (
    capillary_damping_lamb,
    dam_break_ritter_profile,
    run_capillary_damping,
    run_dam_break_ritter,
    run_stokes2_wave,
    stokes2_free_surface,
)

OUT = Path("artifacts/p128w3_cfd_maturity")
FIG = OUT / "figures"


def figure_dam_break(result: dict, path: Path) -> None:
    rows = result["samples"]
    t = np.array([r["t"] for r in rows])
    sim = np.array([r["x_front_sim"] for r in rows])
    ref = np.array([r["x_front_ritter"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(t, ref, "k--", label="Ritter (1892)")
    axes[0].plot(t, sim, "o-", label="VOF-NS")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("front position [m]")
    axes[0].legend()
    axes[0].set_title("L1 dam-break front")
    axes[1].plot(t, [r["front_error_pct"] for r in rows], "o-")
    axes[1].axhline(3.0, color="r", ls=":", label="3 % criterion")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("front error [%]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_stokes(result: dict, path: Path, eta_history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.asarray(eta_history["x"])
    eta = np.asarray(eta_history["eta"])
    eta_ref = np.asarray(eta_history["eta_ref"])
    axes[0].plot(x, eta_ref, "k--", label="Stokes 2nd order")
    axes[0].plot(x, eta, "b-", label="VOF-NS")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("eta [m]")
    axes[0].legend()
    axes[0].set_title(f"L2 wave profile at t = {eta_history['t']:.2f} s")
    t = np.asarray(eta_history["t_samples"])
    err = 100.0 * np.asarray(eta_history["eta_error"]) / eta_history["amplitude"]
    axes[1].plot(t, err, "o-")
    axes[1].axhline(8.0, color="r", ls=":", label="8 % criterion")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("eta RMS error / a [%]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_capillary(result: dict, path: Path) -> None:
    t = np.asarray(result["samples"]["t"])
    amp = np.asarray(result["samples"]["amplitude"])
    lam = result["metrics"]["decay_lamb"]
    a0 = amp[0] * np.exp(lam * t[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogy(t, amp, "o", label="VOF-NS")
    ax.semilogy(t, a0 * np.exp(-lam * t), "k--", label="Lamb (2 nu k^2)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("wave amplitude [m]")
    ax.legend()
    ax.set_title("L3 capillary-wave damping")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict[str, object] = {}

    print("[L1] dam break", flush=True)
    l1 = run_dam_break_ritter(nx=200, h0=0.10, aspect=20.0, n_samples=6)
    results["L1"] = l1
    figure_dam_break(l1, FIG / "L1_dam_break.png")
    print("[L1] metrics", l1["metrics"], flush=True)

    print("[L2] Stokes II", flush=True)
    l2 = run_stokes2_wave(
        n_wavelengths=4.0, wavelength=2.0, amplitude=0.05, depth=2.0,
        periods=3.0, n_samples=20, cells_per_wavelength=50.0,
    )
    results["L2"] = l2
    print("[L2] metrics", l2["metrics"], flush=True)

    print("[L3] capillary damping", flush=True)
    l3 = run_capillary_damping(
        wavelength=0.015, amplitude=0.0005, depth=0.05,
        n_wavelengths=4.0, cells_per_wavelength=40.0,
        viscosity=2.0e-5, periods=4.0, n_samples=25,
    )
    results["L3"] = l3
    figure_capillary(l3, FIG / "L3_capillary.png")
    print("[L3] metrics", l3["metrics"], flush=True)

    results["wall_seconds"] = time.time() - started
    (OUT / "vof_ladder.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("done", results["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()