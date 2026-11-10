"""Reproduce the SCMC/CG Laplace-law accounting numbers (artifacts/scmc_laplace_20261108).

Produces ``results.json`` (programmatic, no hand-copied numbers) and two figures.
Runtime is kept small on purpose: the control node is loaded (load15 ~ 128 on
128 cores), so every experiment here is a few thousand steps on <= 64^2 cells,
single-threaded.

Usage
-----
    python3 tools/scmc_laplace_accounting.py [--out artifacts/scmc_laplace_20261108]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm.multiphase_benchmarks import (  # noqa: E402
    StaticDropletConfig,
    run_static_droplet,
)
from tensorlbm.scmc_pressure_tensor import (  # noqa: E402
    SCMCPressureConfig,
    droplet_pressure_jump,
    flat_interface_pressure_profile,
    simulate_flat_interface,
    simulate_scmc_droplet,
)

G_SCAN = (-0.9, -2.0, -2.4, -3.0, -3.5, -4.0)


def _interface_contrast(f1: torch.Tensor, f2: torch.Tensor) -> float:
    """max |rho1 - rho2| — 0 means a homogeneous mixture, >0 a real interface."""
    return float((f1.sum(0) - f2.sum(0)).abs().max().item())


def regime_scan(out: Path) -> dict[str, object]:
    rows = []
    profiles = {}
    for G in G_SCAN:
        cfg = SCMCPressureConfig(ny=32, nx=8, n_steps=3000, G12=G)
        f1, f2 = simulate_flat_interface(cfg)
        finite = bool(torch.isfinite(f1).all().item() and torch.isfinite(f2).all().item())
        row: dict[str, object] = {"G12": G, "finite": finite}
        if finite:
            prof = flat_interface_pressure_profile(f1, f2, G, "sym")
            row.update(
                contrast=_interface_contrast(f1, f2),
                flatness_mechanical=float(prof["flatness"]),  # type: ignore[arg-type]
                flatness_proxy=float(prof["proxy_flatness"]),  # type: ignore[arg-type]
                sigma_flat=float(prof["sigma_flat"]),  # type: ignore[arg-type]
                mean_normal=float(prof["mean_normal"]),  # type: ignore[arg-type]
            )
            if abs(G + 3.0) < 1e-9:
                profiles["G-3.0"] = {
                    "p_normal": prof["p_normal"].tolist(),  # type: ignore[union-attr]
                    "p_tangential": prof["p_tangential"].tolist(),  # type: ignore[union-attr]
                    "rho1": prof["rho1"].tolist(),  # type: ignore[union-attr]
                    "rho2": prof["rho2"].tolist(),  # type: ignore[union-attr]
                }
        else:
            row.update(contrast=float("nan"), flatness_mechanical=float("nan"),
                       flatness_proxy=float("nan"), sigma_flat=float("nan"),
                       mean_normal=float("nan"))
        rows.append(row)
    (out / "flat_interface_profiles.json").write_text(json.dumps(profiles, indent=1), encoding="utf-8")
    return {"scan": rows}


def scmc_droplet_mechanical(out: Path) -> dict[str, object]:
    """Mechanical-vs-proxy pressure jump of the SCMC droplet at the shipped defaults."""
    cfg = SCMCPressureConfig(G12=-0.9, tau=1.0, rho_heavy=0.7, rho_light=0.3)
    rows = []
    for r in (8.0, 12.0):
        f1, f2 = simulate_scmc_droplet(r, cfg, nx=64, ny=64, n_steps=4000)
        row = droplet_pressure_jump(f1, f2, r, cfg.G12, "sym")
        row["contrast_inside"] = _interface_contrast_inside(f1, f2, r, 64)
        rows.append(row)
    return {"per_radius": rows, "config": {"G12": cfg.G12, "tau": cfg.tau,
                                           "rho_heavy": cfg.rho_heavy, "rho_light": cfg.rho_light,
                                           "n_steps": 4000, "nx": 64, "ny": 64}}


def _interface_contrast_inside(f1: torch.Tensor, f2: torch.Tensor, r: float, n: int) -> float:
    idx = torch.arange(n, dtype=torch.float32)
    rr = (idx.view(-1, 1) - n / 2.0) ** 2 + (idx.view(1, -1) - n / 2.0) ** 2
    mask = rr <= (0.5 * r) ** 2
    return float(((f1.sum(0) - f2.sum(0))[mask]).abs().mean().item())


def cg_laplace(out: Path, *, nx: int = 48, radii: tuple[float, ...] = (6.0, 9.0, 12.0),
               n_steps: int = 2500, tag: str = "cg_droplet_runs") -> dict[str, object]:
    """Shipped droplet benchmark: ΔP·R consistency across radii (both models)."""
    rep = run_static_droplet(
        StaticDropletConfig(
            nx=nx, ny=nx, radii=radii, n_steps=n_steps,
            output_root=out / tag, overwrite=True, device="cpu",
        )
    )
    data = {}
    for model in ("scmc", "cg"):
        per = rep["results"][model]["per_radius"]  # type: ignore[index]
        sig = [float(d["sigma_eff"]) for d in per]  # type: ignore[index]
        data[model] = {
            "radii": [float(d["r"]) for d in per],  # type: ignore[index]
            "delta_p": [float(d["delta_p"]) for d in per],  # type: ignore[index]
            "sigma_eff": sig,
            "spurious_u": [float(d["max_spurious_u"]) for d in per],  # type: ignore[index]
            "sigma_eff_fit": float(rep["results"][model]["sigma_eff_fit"]),  # type: ignore[index]
            "sigma_spread": (max(sig) - min(sig)) / max(abs(sum(sig) / len(sig)), 1e-30),
            "label": rep["results"][model]["laplace_physical_validation"],  # type: ignore[index]
        }
    return data


def figures(out: Path, regime: dict[str, object], cg: dict[str, object]) -> list[str]:
    made = []
    scan = [r for r in regime["scan"] if r["finite"]]  # type: ignore[index]
    gs = [r["G12"] for r in scan]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(gs, [r["contrast"] for r in scan], "o-")
    ax[0].set_xlabel(r"$G_{12}$")
    ax[0].set_ylabel(r"max$|\rho_1-\rho_2|$")
    ax[0].set_title("SCMC interface contrast after 3000 steps")
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(gs, [r["flatness_mechanical"] for r in scan], "s-", label="mechanical $P_{NN}$")
    ax[1].semilogy(gs, [r["flatness_proxy"] for r in scan], "^--", label=r"proxy $c_s^2\rho$")
    ax[1].set_xlabel(r"$G_{12}$")
    ax[1].set_ylabel("flatness residual")
    ax[1].set_title("normal-stress constancy (flat interface)")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    f1 = out / "scmc_regime.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    made.append(str(f1))

    prof = json.loads((out / "flat_interface_profiles.json").read_text(encoding="utf-8"))
    if "G-3.0" in prof:
        d = prof["G-3.0"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].plot(d["p_normal"], label="$P_{NN}$")
        ax[0].plot(d["p_tangential"], label="$P_{TT}$")
        ax[0].set_xlabel("y (lattice)")
        ax[0].set_ylabel("pressure")
        ax[0].set_title(r"$G_{12}=-3.0$: normal stress is NOT flat")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(d["rho1"], label=r"$\rho_1$")
        ax[1].plot(d["rho2"], label=r"$\rho_2$")
        ax[1].set_xlabel("y (lattice)")
        ax[1].set_ylabel("component density")
        ax[1].set_title("component density swap (interface width ~1 cell)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        f2 = out / "flat_interface_profile.png"
        fig.savefig(f2, dpi=130)
        plt.close(fig)
        made.append(str(f2))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for model, mk in (("scmc", "o-"), ("cg", "s-")):
        d = cg[model]
        ax.plot([1.0 / r for r in d["radii"]], d["delta_p"], mk, label=f"{model.upper()} measured")
        s = d["sigma_eff_fit"]
        xs = [1.0 / r for r in d["radii"]]
        ax.plot(xs, [s * x for x in xs], "--", alpha=0.6, label=f"{model.upper()} fit σ={s:.4f}")
    ax.set_xlabel("1/R")
    ax.set_ylabel(r"$\Delta P$")
    ax.set_title("Young-Laplace: ΔP vs 1/R")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = out / "laplace_fit.png"
    fig.savefig(f3, dpi=130)
    plt.close(fig)
    made.append(str(f3))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/scmc_laplace_20261108")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    regime = regime_scan(out)
    print("regime scan done", flush=True)
    scmc = scmc_droplet_mechanical(out)
    print("scmc droplet done", flush=True)
    cg = cg_laplace(out)
    print("cg droplet done", flush=True)
    cg_conv = cg_laplace(out, nx=64, radii=(8.0, 12.0), n_steps=4000, tag="cg_droplet_converged")
    print("cg converged droplet done", flush=True)
    figs = figures(out, regime, cg)

    payload = {
        "meta": {
            "module": "tensorlbm.scmc_pressure_tensor",
            "torch_threads": 1,
            "note": "all numbers produced programmatically by this script",
        },
        "flat_interface_regime": regime["scan"],
        "scmc_droplet_mechanical": scmc,
        "droplet_benchmark": cg,
        "droplet_benchmark_converged": cg_conv,
        "figures": figs,
        "checks": {},
    }
    # derived checks (fail-closed booleans, exported for the report)
    mixed = [r for r in regime["scan"] if r["finite"] and abs(r["G12"]) <= 2.0]
    payload["checks"]["scmc_default_mixed"] = bool(mixed) and all(
        r["contrast"] < 0.05 for r in mixed
    )
    interf = [r for r in regime["scan"] if r["finite"] and 2.4 <= abs(r["G12"]) <= 3.5]
    payload["checks"]["interface_window_exists"] = bool(interf) and all(
        r["contrast"] > 0.5 for r in interf
    )
    payload["checks"]["scmc_blowup_at_G4_ny32"] = any(
        (not r["finite"]) and abs(r["G12"]) >= 4.0 for r in regime["scan"]
    )
    payload["checks"]["normal_stress_not_flat_in_window"] = bool(interf) and all(
        r["flatness_mechanical"] > 0.10 for r in interf
    )
    # honest verdicts: report the measured spread and whether the <5 % band holds
    payload["checks"]["cg_sigma_spread_48sq_2500"] = cg["cg"]["sigma_spread"]
    payload["checks"]["cg_sigma_spread_64sq_4000"] = cg_conv["cg"]["sigma_spread"]
    payload["checks"]["cg_laplace_linearity_verdict"] = (
        "VERIFIED(ΔP·R spread <5%)"
        if cg_conv["cg"]["sigma_spread"] < 0.05
        else "NOT_CONVERGED(spread >=5%)"
    )
    payload["checks"]["scmc_default_sigma_eff_near_zero"] = (
        cg["scmc"]["sigma_eff_fit"] < 1e-3
    )
    payload["checks"]["scmc_sigma_not_laplace_consistent"] = (
        cg["scmc"]["sigma_spread"] > 0.30
    )
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload["checks"], indent=1), flush=True)
    print(f"wrote {out/'results.json'}", flush=True)


if __name__ == "__main__":
    main()