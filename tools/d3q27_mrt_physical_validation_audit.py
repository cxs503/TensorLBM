#!/usr/bin/env python3
"""Run the D3Q27 MRT physical-validation audit and regenerate the doc tables.

Produces (all paths relative to the repository root):

* ``artifacts/d3q27_mrt_physical_validation/audit.json`` — full machine-readable
  artifact (checks, measurements, self hash).
* ``artifacts/d3q27_mrt_physical_validation/checks_table.md`` — the check table
  injected into the doc between the ``<!-- BEGIN GENERATED ... -->`` markers, so
  no number in the documentation is hand-copied.
* ``artifacts/d3q27_mrt_physical_validation/figures/*.png`` — decay curves
  against the analytic solutions, the k-space eigenvalue spectrum, and the
  wall-offset inference.

Usage::

    python tools/d3q27_mrt_physical_validation_audit.py            # full run
    python tools/d3q27_mrt_physical_validation_audit.py --quick    # CI smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "d3q27_mrt_physical_validation"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "d3q27_mrt_physical_validation_r1.md"

BEGIN = "<!-- BEGIN GENERATED BLOCK (tools/d3q27_mrt_physical_validation_audit.py) -->"
END = "<!-- END GENERATED BLOCK -->"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.4f} %"


def _table(checks: list[dict]) -> str:
    lines = [
        "| check | aspect | measured | reference | rel. error | tolerance | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            "| `{}` | {} | {:,.6g} | {:,.6g} | {} | {} | **{}** |".format(
                check["id"],
                check["aspect"],
                check["measured"],
                check["reference"],
                _fmt_pct(check["rel_error"]),
                _fmt_pct(check["tolerance"]),
                check["status"].upper(),
            )
        )
    return "\n".join(lines)


def _build_figures() -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tensorlbm.d3q27_mrt_physical_validation import (
        measure_shear_wave_viscosity,
        measure_wall_mode_decay,
        spectral_step_eigenvalues,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- figure 1: shear-wave energy decay vs analytic on the grid ladder
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in (16, 24, 32):
        row = measure_shear_wave_viscosity(n, 0.65, steps=300, collect_series=True)
        times = np.asarray(row["series"]["times"], dtype=float)
        energies = np.asarray(row["series"]["energies"], dtype=float)
        k = row["lattice_wavenumber"]
        nu = row["analytic_viscosity"]
        axes[0].semilogy(times, energies / energies[0], "o", ms=3, alpha=0.6, label=f"LBM L={n}")
        axes[0].semilogy(
            times,
            np.exp(-2.0 * nu * k * k * times),
            "-",
            lw=1.0,
            color=axes[0].lines[-1].get_color(),
            label=f"analytic L={n}",
        )
        axes[1].plot(
            times,
            np.abs(energies / energies[0] - np.exp(-2.0 * nu * k * k * times)),
            "o-",
            ms=3,
            lw=1.0,
            label=f"L={n} |dev| (nu err {row['rel_error'] * 100:.2f} %)",
        )
    axes[0].set_xlabel("lattice step")
    axes[0].set_ylabel(r"$E(t)/E(0)$")
    axes[0].set_title("Shear-wave decay: LBM (markers) vs $e^{-2\\nu k^2 t}$ (lines)")
    axes[0].legend(fontsize=7, ncol=1)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("lattice step")
    axes[1].set_ylabel(r"$|E/E_0 - e^{-2\nu k^2 t}|$")
    axes[1].set_title("Deviation from the exact unsteady NSE solution")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG / "shear_wave_decay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["shear_wave_decay"] = path

    # --- figure 2: k-space eigenvalue spectrum (hybrid: time domain + spectral)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in (16, 32):
        row = spectral_step_eigenvalues(n, 0.65)
        # recompute the full spectrum for the plot
        from tensorlbm.d3q27 import C, collide_mrt27, equilibrium27

        rho = torch.ones((1, 1, 1), dtype=torch.float64)
        zero = torch.zeros_like(rho)
        base = equilibrium27(rho, zero, zero, zero).reshape(27).clone()

        def collide_vector(raw: torch.Tensor, tau: float = 0.65) -> torch.Tensor:
            return collide_mrt27(raw.reshape(27, 1, 1, 1), tau).reshape(27)

        jac = torch.autograd.functional.jacobian(collide_vector, base.clone(), vectorize=True)
        k = row["lattice_wavenumber"]
        phase = torch.exp(-1j * k * C.to(torch.float64)[:, 0]).to(torch.complex128)
        lam = torch.linalg.eigvals(torch.diag(phase) @ jac.to(torch.complex128))
        axes[0].plot(lam.real, lam.imag, "o", ms=4, label=f"k=2π/{n}")
        axes[1].plot(
            [n] * 4,
            [
                row["spectral_viscosity"],
                row["spectral_sound_speed"],
                row["spectral_attenuation"],
                abs(row["shear_eigenvalue"][1]),
            ],
            "o",
        )
    axes[0].set_xlabel(r"Re $\lambda$")
    axes[0].set_ylabel(r"Im $\lambda$")
    axes[0].set_title("Linearised one-step operator spectrum (D3Q27 MRT, τ=0.65)")
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("grid n (k = 2π/n)")
    axes[1].set_ylabel("value")
    axes[1].set_title("Spectral observables: ν, $c_s$, attenuation, Im λ")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG / "eigenvalue_spectrum.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["eigenvalue_spectrum"] = path

    # --- figure 3: wall-offset inference
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    offsets, errors, heights = [], [], []
    for ny in (16, 24, 32, 40):
        row = measure_wall_mode_decay(ny, 0.65, steps=400, collect_series=True)
        offsets.append(row["implied_wall_offset"])
        errors.append(row["rel_error"] * 100.0)
        heights.append(row["ny"])
        times = np.asarray(row["series"]["times"], dtype=float)
        energies = np.asarray(row["series"]["energies"], dtype=float)
        nu, height = 0.05, row["channel_width"]
        axes[0].semilogy(times, energies / energies[0], "o", ms=3, alpha=0.6, label=f"LBM ny={ny}")
        axes[0].semilogy(
            times,
            np.exp(-2.0 * nu * (math.pi / height) ** 2 * times),
            "-",
            lw=1.0,
            color=axes[0].lines[-1].get_color(),
            label=f"analytic ny={ny}",
        )
    axes[0].set_xlabel("lattice step")
    axes[0].set_ylabel("no-slip mode energy / E(0)")
    axes[0].set_title("Wall-bounded eigenmode: bounce-back vs exact no-slip decay")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(heights, offsets, "o-", label="implied wall offset [lattice]")
    axes[1].axhline(0.5, color="r", ls="--", lw=1.0, label="full-way bounce-back convention (0.5)")
    axes[1].set_xlabel("channel cells ny")
    axes[1].set_ylabel("implied offset from solid cell centre")
    axes[1].set_title("Effective no-slip wall position inferred from the decay rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG / "wall_offset_inference.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["wall_offset_inference"] = path

    return paths


def _inject(table: str) -> None:
    if not DOC.exists():
        print(f"[warn] doc not found: {DOC}", file=sys.stderr)
        return
    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        print("[warn] generated-block markers missing; doc left untouched", file=sys.stderr)
        return
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="single-grid smoke run (no figures)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))

    from tensorlbm.d3q27_mrt_physical_validation import (
        run_d3q27_mrt_physical_validation_probe,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    if args.quick:
        artifact = run_d3q27_mrt_physical_validation_probe(grids=(16,), wall_grids=(16,))
    else:
        artifact = run_d3q27_mrt_physical_validation_probe()

    (OUT / "audit.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
    table = _table(artifact["checks"])
    (OUT / "checks_table.md").write_text(table + "\n")
    _inject(table)

    print(json.dumps(artifact["summary"], indent=2))
    print("artifact sha256:", artifact["artifact_sha256"])
    print("evidence tier:", artifact["evidence_tier"])

    if not args.quick:
        for name, path in _build_figures().items():
            print(f"figure {name}: {path.relative_to(ROOT)}")
    return 0 if artifact["summary"]["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
