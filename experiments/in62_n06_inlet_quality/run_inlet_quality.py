#!/usr/bin/env python3
"""IN62-N06 — LBM divergence-free synthetic inflow: inlet quality + development section.

Stages
------
``plane``   inlet-plane quality: Reynolds-stress recovery, divergence diagnostics,
            projection Pareto front, digital-filter/Lund control.
``duct``    uniform-stream development section (D3Q19): ``R_ij(x)`` recovery and
            decay law, self-preserving development length ``L_dev``, and the
            spurious-pressure negative control (same stresses, no projection).
``decay``   periodic isotropic decay vs the Comte-Bellot & Corrsin (1971)
            decay exponent.

Every number written to the JSON files is produced by the run itself (no manual
transcription); figures are made from the same dictionaries.

Usage::

    PYTHONPATH=src python experiments/in62_n06_inlet_quality/run_inlet_quality.py \
        --stage all --out runs/in62_n06_inlet_quality --threads 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from tensorlbm.turbulence.dfsem_inflow import (
    DFSEMInlet,
    DigitalFilterInlet,
    discrete_divergence_rms,
    divergence_ratio,
    divergence_rms,
    sample_stress,
    stress_tensor,
)
from tensorlbm.turbulence.inlet_development import (
    DevelopmentSectionConfig,
    check_kernels_against_library,
    decay_exponent,
    development_length,
    run_development_section,
    run_isotropic_decay,
    slab_stress,
)

# ---------------------------------------------------------------------------
# Predeclared target state and tolerances
# ---------------------------------------------------------------------------

TARGET = dict(uu=4.0e-4, vv=2.0e-4, ww=2.0e-4, uv=0.0, uw=0.0, vw=0.0)
# isotropic variant: the only inlet state for which a self-preserving decay can
# exist in a uniform stream (an anisotropic inlet necessarily relaxes towards
# isotropy, so its normalised anisotropy shape cannot be stationary)
TARGET_ISO = dict(uu=4.0e-4, vv=4.0e-4, ww=4.0e-4, uv=0.0, uw=0.0, vw=0.0)
UU = TARGET["uu"]

GATES = {
    "G1_plane_stress_error_max": 0.05,      # |R_ij - R_ij*| / R_uu*, projected DFSEM
    "G2_plane_div_ratio_projected_max": 1.0e-3,
    "G2_plane_div_reduction_min": 30.0,     # unprojected / projected
    "G3_negative_control_div_ratio_min": 0.05,
    "G4_dfm_lund_stress_error_max": 1.0e-6,
    "G4_dfm_lund_gain_min": 10.0,           # raw DFM error / Lund error
    "G5_duct_inlet_recovery_error_max_anisotropic": 0.15,
    "G6_L_dev_max_fraction": 0.75,          # of nx
    "G7_pressure_ratio_min": 2.0,           # SEM raw / DFSEM projected, near inlet
    "G8_decay_exponent_rel_error_max": 0.15,
    "G8_n_cbc": 1.30,
    "G9_kernel_equivalence_max": 1.0e-6,  # float32 round-off
}


def _hash(arr: torch.Tensor) -> str:
    a = arr.detach().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()[:16]


def _stress_error(R_hat: torch.Tensor, R_target: torch.Tensor, scale: float) -> float:
    return float(((R_hat - R_target) / scale).abs().amax())


# ---------------------------------------------------------------------------
# Stage 1: inlet-plane quality
# ---------------------------------------------------------------------------


def stage_plane(out: dict, n_snap: int = 64, ny: int = 48, nz: int = 48) -> dict:
    R_target = stress_tensor(**TARGET)
    rows: dict[str, dict] = {}

    cases = {
        "dfsem_projected": dict(projection=True, divergence_free=True),
        "dfsem_unprojected": dict(projection=False, divergence_free=True),
        "sem_projected": dict(projection=True, divergence_free=False),
        "sem_unprojected": dict(projection=False, divergence_free=False),
    }
    for name, flags in cases.items():
        t0 = time.time()
        gen = DFSEMInlet(
            ny=ny,
            nz=nz,
            stress=R_target,
            length_scale=8.0,
            n_eddies=300,
            seed=11,
            n_stress_iterations=1,
            **flags,
        )
        seq = gen.sequence(n_snap)
        ens = seq.permute(1, 0, 2, 3).reshape(3, -1)
        R_hat = sample_stress(ens)
        d = gen.diagnostics
        rows[name] = {
            "stress_error_ensemble": _stress_error(R_hat, R_target, UU),
            "R_ij_ensemble": R_hat.tolist(),
            "div_ratio_mean": float(sum(x["div_ratio"] for x in d) / len(d)),
            "div_ratio_unprojected_mean": float(
                sum(x["div_ratio_raw"] for x in d) / len(d)
            ),
            "div_rms_mean": float(sum(float(divergence_rms(s.float())) for s in seq[:8]) / 8),
            "discrete_div_rms_mean": float(
                sum(float(discrete_divergence_rms(s.float())) for s in seq[:8]) / 8
            ),
            "rms_u": float(seq[:, 0].std()),
            "rms_v": float(seq[:, 1].std()),
            "rms_w": float(seq[:, 2].std()),
            "seconds": time.time() - t0,
            "sequence_hash": _hash(seq[:4]),
        }
        print(f"[plane] {name}: stress_err={rows[name]['stress_error_ensemble']:.3e} "
              f"div={rows[name]['div_ratio_mean']:.3e}", flush=True)

    # digital filter (Klein) with and without the Lund layer
    for name, lund in (("dfm_lund", True), ("dfm_raw", False)):
        gen = DigitalFilterInlet(
            ny=ny, nz=nz, stress=R_target, length_scale=8.0, time_scale=16.0,
            n_steps=256, seed=5, apply_lund=lund,
        )
        seq = gen.sequence()
        R_hat = sample_stress(seq.permute(1, 0, 2, 3).reshape(3, -1))
        rows[name] = {
            "stress_error_ensemble": _stress_error(R_hat, R_target, UU),
            "R_ij_ensemble": R_hat.tolist(),
            "div_ratio_mean": float(divergence_ratio(seq[0].float())),
            "rms_u": float(seq[:, 0].std()),
            "rms_v": float(seq[:, 1].std()),
            "rms_w": float(seq[:, 2].std()),
            "sequence_hash": _hash(seq[:4]),
        }
        print(f"[plane] {name}: stress_err={rows[name]['stress_error_ensemble']:.3e}", flush=True)

    # projection/restoration Pareto front for a *shear-carrying* target: a
    # frozen (x-invariant) field cannot be both divergence-free in-plane and
    # carry R_uv, so the trade-off is measured instead of hidden.
    shear = dict(uu=4.0e-4, vv=2.0e-4, ww=2.0e-4, uv=1.0e-4, uw=0.0, vw=0.0)
    R_shear = stress_tensor(**shear)
    pareto = []
    for nit in (0, 1, 2, 4, 8):
        gen = DFSEMInlet(
            ny=ny, nz=nz, stress=R_shear, length_scale=8.0, n_eddies=300,
            seed=13, projection=True, n_stress_iterations=nit,
        )
        seq = gen.sequence(16)
        R_hat = sample_stress(seq.permute(1, 0, 2, 3).reshape(3, -1))
        pareto.append(
            {
                "n_stress_iterations": nit,
                "stress_error_ensemble": _stress_error(R_hat, R_shear, shear["uu"]),
                "div_ratio_mean": float(sum(x["div_ratio"] for x in gen.diagnostics) / len(gen.diagnostics)),
            }
        )
        print(f"[plane] pareto nit={nit}: {pareto[-1]}", flush=True)

    # reproducibility: two identically seeded generators must agree bit-for-bit
    a = DFSEMInlet(ny=16, nz=16, stress=R_target, length_scale=8.0, n_eddies=100, seed=7).sequence(4)
    b = DFSEMInlet(ny=16, nz=16, stress=R_target, length_scale=8.0, n_eddies=100, seed=7).sequence(4)
    repro = {"hash_a": _hash(a), "hash_b": _hash(b), "identical": bool(torch.equal(a, b))}

    gates = {
        "G1_plane_stress_error_max": rows["dfsem_projected"]["stress_error_ensemble"],
        "G1_pass": rows["dfsem_projected"]["stress_error_ensemble"] <= GATES["G1_plane_stress_error_max"],
        "G2_plane_div_ratio_projected": rows["dfsem_projected"]["div_ratio_mean"],
        "G2_pass": rows["dfsem_projected"]["div_ratio_mean"] <= GATES["G2_plane_div_ratio_projected_max"],
        "G2_plane_div_reduction": rows["dfsem_unprojected"]["div_ratio_mean"]
        / max(rows["dfsem_projected"]["div_ratio_mean"], 1e-30),
        "G2_reduction_pass": (
            rows["dfsem_unprojected"]["div_ratio_mean"]
            / max(rows["dfsem_projected"]["div_ratio_mean"], 1e-30)
        )
        >= GATES["G2_plane_div_reduction_min"],
        "G3_negative_control_div_ratio": max(
            rows["dfsem_unprojected"]["div_ratio_mean"], rows["sem_unprojected"]["div_ratio_mean"]
        ),
        "G3_pass": max(
            rows["dfsem_unprojected"]["div_ratio_mean"], rows["sem_unprojected"]["div_ratio_mean"]
        )
        >= GATES["G3_negative_control_div_ratio_min"],
        "G4_dfm_lund_stress_error": rows["dfm_lund"]["stress_error_ensemble"],
        "G4_pass": rows["dfm_lund"]["stress_error_ensemble"] <= GATES["G4_dfm_lund_stress_error_max"],
        "G4_dfm_gain": rows["dfm_raw"]["stress_error_ensemble"]
        / max(rows["dfm_lund"]["stress_error_ensemble"], 1e-30),
        "G4_gain_pass": (
            rows["dfm_raw"]["stress_error_ensemble"]
            / max(rows["dfm_lund"]["stress_error_ensemble"], 1e-30)
        )
        >= GATES["G4_dfm_lund_gain_min"],
        "G9_kernel_equivalence": check_kernels_against_library(),
        # float32 round-off tolerance, not bitwise equality: the two kernels
        # differ only by association order, which is not part of the contract
        "G9_pass": check_kernels_against_library()["collision_max_abs_diff"]
        <= GATES["G9_kernel_equivalence_max"],
        "G10_reproducibility": repro,
        "G10_pass": repro["identical"],
    }
    return {"cases": rows, "pareto_shear_target": pareto, "gates": gates, "target": TARGET}


# ---------------------------------------------------------------------------
# Stage 2: development section
# ---------------------------------------------------------------------------


SIGMA = 8.0


def _ducted_sequence(name: str, n_steps: int, ny: int, nz: int, seed: int = 21):
    iso = name.endswith("_iso")
    R_target = stress_tensor(**(TARGET_ISO if iso else TARGET))
    base = name.replace("_iso", "")
    if base == "dfm":
        # Klein filter width giving the same integral scale as the eddy field
        # (L_int = sigma/sqrt(2) for the Gaussian kernel used here)
        return DigitalFilterInlet(
            ny=ny, nz=nz, stress=R_target, length_scale=SIGMA / 2.0**0.5,
            time_scale=SIGMA, n_steps=n_steps, seed=seed,
        ).sequence()
    sem = base == "sem"
    proj = base != "sem"
    gen = DFSEMInlet(
        ny=ny, nz=nz, stress=R_target, length_scale=SIGMA, n_eddies=400,
        seed=seed, projection=proj, divergence_free=not sem, n_stress_iterations=1,
    )
    return gen.sequence(n_steps)


def stage_duct(out: dict, cfg: DevelopmentSectionConfig, cases=("dfsem", "sem", "dfm"),
               outdir: Path | None = None) -> dict:
    """Run (or reuse) one duct case per call; cases are written to separate files
    so the three inlet types can be computed in parallel on different nodes."""
    results: dict[str, dict] = {}
    if outdir is not None:
        for name in cases:
            f = outdir / f"results_duct_{name}.json"
            if f.exists():
                try:
                    results[name] = json.loads(f.read_text())
                except json.JSONDecodeError:
                    pass
    todo = [c for c in cases if c not in results]
    for name in todo:
        t0 = time.time()
        # the isotropic case must be scored against the isotropic target
        R_target = stress_tensor(**(TARGET_ISO if name.endswith("_iso") else TARGET))
        seq = _ducted_sequence(name, cfg.n_steps, cfg.ny, cfg.nz)
        run = run_development_section(seq, cfg)
        R = run["R"]
        rec = {
            "wall_seconds": time.time() - t0,
            "samples": run["samples"],
            "R_profile": R.tolist(),
            "R_stderr_profile": run["R_stderr"].tolist(),
            "tke_profile": run["tke"].tolist(),
            "mean_u_profile": run["mean_velocity"].tolist(),
            "mean_pressure_profile": run["mean_pressure"].tolist(),
            "pressure_rms_profile": run["pressure_rms"].tolist(),
            "pressure_rms_stderr_profile": run["pressure_rms_stderr"].tolist(),
            "inlet_recovery": {
                f"{i}{j}": float((slab_stress(R, 2, 5)[a, b] - R_target[a, b]) / R_target[0, 0])
                for (i, j), (a, b) in {
                    ("u", "u"): (0, 0), ("v", "v"): (1, 1), ("w", "w"): (2, 2),
                    ("u", "v"): (0, 1), ("u", "w"): (0, 2), ("v", "w"): (1, 2),
                }.items()
            },
            "inlet_recovery_error": float(
                ((slab_stress(R, 2, 5) - R_target) / R_target[0, 0]).abs().amax()
            ),
            "config": run["config"],
        }
        rec["development_length"] = development_length(R, x_start=2, x_stop=cfg.nx - 2)
        # campaign-faithful definition: 95 % self-preservation of the anisotropy
        # *shape*.  The rate condition above is stricter and relaxes only
        # logarithmically, so it is kept as a separate diagnostic.
        rec["development_length_shape_only"] = development_length(
            R, x_start=2, x_stop=cfg.nx - 2, require_rate=False
        )
        results[name] = rec
        if outdir is not None:
            (outdir / f"results_duct_{name}.json").write_text(json.dumps(rec, indent=2))
        print(
            f"[duct] {name}: {rec['wall_seconds']:.0f}s samples={rec['samples']} "
            f"inlet_err={rec['inlet_recovery_error']:.3f} "
            f"L_dev={rec['development_length_shape_only']['L_dev']} "
            f"p_inlet={rec['pressure_rms_profile'][3]:.3e}",
            flush=True,
        )

    def _pwin(name: str) -> float:
        prof = results[name]["pressure_rms_profile"]
        return float(sum(prof[2:8]) / len(prof[2:8]))

    gates: dict = {}
    ldev_case = "dfsem_iso" if "dfsem_iso" in results else "dfsem"
    if "dfsem" in results:
        gates.update(
            {
                "G5_duct_inlet_recovery_error": results["dfsem"]["inlet_recovery_error"],
                "G5_pass": results["dfsem"]["inlet_recovery_error"]
                <= GATES["G5_duct_inlet_recovery_error_max_anisotropic"],
            }
        )
    if ldev_case in results:
        gates.update(
            {
                "G6_L_dev": results[ldev_case]["development_length"]["L_dev"],
                "G6_case": ldev_case,
                "G6_pass": (
                    results[ldev_case]["development_length"]["L_dev"]
                    == results[ldev_case]["development_length"]["L_dev"]
                    and results[ldev_case]["development_length"]["L_dev"]
                    <= GATES["G6_L_dev_max_fraction"] * cfg.nx
                ),
                "G6_shape_drift_anisotropic_case": (
                    max(results["dfsem"]["development_length"]["shape_deviation"])
                    if "dfsem" in results
                    else None
                ),
                "G6_shape_drift_L_dev_case": max(
                    results[ldev_case]["development_length"]["shape_deviation"]
                ),
            }
        )
    if "dfsem" in results and "sem" in results:
        p_dfsem, p_sem = _pwin("dfsem"), _pwin("sem")
        gates.update(
            {
                "G7_pressure_rms_dfsem": p_dfsem,
                "G7_pressure_rms_sem": p_sem,
                "G7_ratio": p_sem / max(p_dfsem, 1e-30),
                "G7_pass": p_sem / max(p_dfsem, 1e-30) >= GATES["G7_pressure_ratio_min"],
            }
        )
        if "dfm" in results:
            gates["G7_pressure_rms_dfm"] = _pwin("dfm")
    return {
        "cases": results,
        "config": next(iter(results.values()))["config"] if results else {},
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Stage 3: isotropic decay vs Comte-Bellot & Corrsin
# ---------------------------------------------------------------------------


def stage_decay(resolutions=(64, 48), n_steps: int = 2000, sample_every: int = 25,
                tau: float = 0.505, u_rms: float = 0.05,
                k_cut: float | None = 4.0, k_min: float = 1.0) -> dict:
    out: dict[str, dict] = {}
    for n in resolutions:
        t0 = time.time()
        run = run_isotropic_decay(
            n=n, tau=tau, k_peak=4.0, u_rms=u_rms, k_cut=k_cut, k_min=k_min,
            n_steps=n_steps, sample_every=sample_every, seed=3,
        )
        fit = decay_exponent(run["t"], run["tke"])
        window_sweep = {
            f"{int(100 * frac)}pct": decay_exponent(
                run["t"], run["tke"], n_fit=max(6, int(frac * len(run["t"])))
            )["n"]
            for frac in (0.3, 0.5, 0.7, 1.0)
        }
        u0 = run["tke"][0] ** 0.5
        nu = run["config"]["viscosity"]
        # Taylor microscale from the dissipation at the first sample
        eps0 = max(run["dissipation"][0], 1e-30)
        lam = (15.0 * nu * float(run["tke"][0]) / eps0) ** 0.5
        out[f"n{n}"] = {
            "fit": fit,
            "window_sweep": window_sweep,
            "diverged": run.get("diverged", False),
            "t": run["t"],
            "tke": run["tke"],
            "dissipation": run["dissipation"],
            "spectra": run["spectra"],
            "config": run["config"],
            "re_lambda_initial": float(u0 * lam / nu) if lam == lam else None,
            "taylor_microscale": float(lam),
            "wall_seconds": time.time() - t0,
        }
        print(f"[decay] n={n}: n_exp={fit['n']:.3f} range={fit['n_range']} "
              f"Re_lambda~{out[f'n{n}']['re_lambda_initial']:.1f}", flush=True)
    primary = out["n64"] if "n64" in out else out[f"n{resolutions[0]}"]
    rel = abs(primary["fit"]["n"] - GATES["G8_n_cbc"]) / GATES["G8_n_cbc"]
    gates = {
        "G8_n_from_lbm": primary["fit"]["n"],
        "G8_n_range": primary["fit"]["n_range"],
        "G8_n_cbc": GATES["G8_n_cbc"],
        "G8_rel_error": rel,
        "G8_pass": rel <= GATES["G8_decay_exponent_rel_error_max"],
    }
    return {"runs": out, "gates": gates}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(outdir: Path, plane: dict, duct: dict, decay: dict) -> list[str]:
    try:  # figures need matplotlib; compute nodes may not have it
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("[figures] matplotlib unavailable, skipping", flush=True)
        return []

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    # 1) inlet-plane quality: stress components and divergence
    cases = list(plane["cases"].keys())
    comps = ["uu", "vv", "ww", "uv", "uw", "vw"]
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.12
    for k, (cname, (a, b)) in enumerate(zip(comps, idx)):
        vals = [plane["cases"][c][idx_name][a][b] / UU for c in cases for idx_name in ["R_ij_ensemble"]]
        axes[0].bar([i + k * width for i in range(len(cases))], vals, width=width, label=cname)
    axes[0].set_xticks([i + 2.5 * width for i in range(len(cases))])
    axes[0].set_xticklabels(cases, rotation=25, ha="right", fontsize=8)
    axes[0].axhline(1.0, color="k", lw=0.6, ls="--")
    axes[0].set_ylabel(r"$R_{ij}/R_{uu}^{\ast}$")
    axes[0].set_title("Inlet-plane Reynolds stresses (target = 1)")
    axes[0].legend(fontsize=7, ncol=2)
    names = cases + ["dfm_lund", "dfm_raw"]
    divs = [plane["cases"][n]["div_ratio_mean"] for n in names]
    axes[1].bar(range(len(names)), divs, color=["C0", "C1", "C2", "C3", "C4", "C5"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel(r"$\|\nabla\cdot u'\|_{rms}/\|\partial_i u_i\|_{rms}$")
    axes[1].set_title("Measured inlet divergence")
    fig.tight_layout()
    p = figdir / "fig1_inlet_plane_quality.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # 2) duct: R_ij(x) and decay law
    if duct:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for cname in duct["cases"]:
            R = torch.tensor(duct["cases"][cname]["R_profile"])
            x = torch.arange(R.shape[0]) + 0.5
            axes[0].semilogy(x, R[:, 0, 0] / UU, label=f"{cname} $R_{{uu}}$")
            axes[0].semilogy(x, R[:, 1, 1] / UU, "--", label=f"{cname} $R_{{vv}}$")
            axes[1].semilogy(x, torch.tensor(duct["cases"][cname]["pressure_rms_profile"]),
                             label=cname)
        axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
        axes[0].set_xlabel("$x$ [cells]")
        axes[0].set_ylabel(r"$R_{ij}(x)/R_{uu}^{\ast}$")
        axes[0].set_title("Development section: Reynolds stresses")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("$x$ [cells]")
        axes[1].set_ylabel(r"$p_{rms}$ [lattice units]")
        axes[1].set_title("Spurious pressure fluctuations")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = figdir / "fig2_development_section.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 3) decay: TKE history with power-law fit + spectra
    if decay:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for key, run in decay["runs"].items():
            t = torch.tensor(run["t"], dtype=torch.float64)
            k = torch.tensor(run["tke"], dtype=torch.float64)
            fit = run["fit"]
            axes[0].loglog(t[1:] + max(fit["t0"], 0.0), k[1:], "o", ms=3, label=f"{key} LBM")
            tt = torch.linspace(float(t[1]), float(t[-1]), 50, dtype=torch.float64)
            axes[0].loglog(tt + max(fit["t0"], 0.0), fit["A"] * (tt + max(fit["t0"], 0.0)) ** (-fit["n"]),
                           "-", lw=1.2, label=f"{key} fit n={fit['n']:.2f}")
            sp = run["spectra"]
            axes[1].loglog(sp[0]["k"][1:], sp[0]["E"][1:], "-", lw=1, label=f"{key} t={sp[0]['t']}")
            axes[1].loglog(sp[-1]["k"][1:], sp[-1]["E"][1:], "--", lw=1,
                           label=f"{key} t={sp[-1]['t']}")
        axes[0].set_xlabel(r"$t+t_0$ [LBM steps]")
        axes[0].set_ylabel(r"$K$")
        axes[0].set_title("Isotropic decay vs CBC71 ($n\\approx1.30$)")
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel(r"$k$")
        axes[1].set_ylabel(r"$E(k)$")
        axes[1].set_title("Energy spectra")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = figdir / "fig3_isotropic_decay.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 4) field clouds: fluctuation components and the divergence source term
    R_target = stress_tensor(**TARGET)
    R_iso = stress_tensor(**TARGET_ISO)
    fields = {}
    for name, stress, proj in (
        ("projected", R_target, True),
        ("unprojected", R_target, False),
    ):
        gen = DFSEMInlet(ny=64, nz=64, stress=stress, length_scale=8.0,
                         n_eddies=300, seed=11, projection=proj, dtype=torch.float64)
        fields[name] = gen.sequence(1)[0].to(torch.float64)
    div_maps = {}
    for name, u in fields.items():
        vh = torch.fft.rfft2(u[1])
        wh = torch.fft.rfft2(u[2])
        ky = 2.0 * math.pi * torch.fft.fftfreq(u.shape[-2], d=1.0).to(torch.float64).view(-1, 1)
        kz = 2.0 * math.pi * torch.fft.rfftfreq(u.shape[-1], d=1.0).to(torch.float64).view(1, -1)
        div_maps[name] = torch.fft.irfft2(ky * vh + kz * wh, s=u.shape[-2:])

    p = figdir / "fig4_inlet_field_clouds.png"
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    up = fields["unprojected"]
    pr = fields["projected"]
    for ax, (data, title, cmap) in zip(
        axes[0],
        (
            (pr[0], r"projected $u'$", "RdBu_r"),
            (pr[1], r"projected $v'$", "RdBu_r"),
            (pr[2], r"projected $w'$", "RdBu_r"),
        ),
    ):
        lim = float(data.abs().max())
        im = ax.imshow(data.numpy(), origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z [cells]")
        ax.set_ylabel("y [cells]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax, (name, title) in zip(
        axes[1, :2],
        (
            ("unprojected", r"unprojected $\nabla\cdot u'$ (spurious source)"),
            ("projected", r"projected $\nabla\cdot u'$"),
        ),
    ):
        data = div_maps[name]
        lim = float(div_maps["unprojected"].abs().max())
        im = ax.imshow(data.numpy(), origin="lower", cmap="viridis", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z [cells]")
        ax.set_ylabel("y [cells]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ratio_iso = (
        float(divergence_ratio(fields["unprojected"].float()))
        / max(float(divergence_ratio(fields["projected"].float())), 1e-30)
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.5,
        "inlet-plane divergence diagnostic\n"
        f"D(unprojected) = {float(divergence_ratio(fields['unprojected'].float())):.3e}\n"
        f"D(projected)   = {float(divergence_ratio(fields['projected'].float())):.3e}\n"
        f"ratio = {ratio_iso:.3e}\n\n"
        f"iso target $R_{{uu}}$ = {float(R_iso[0, 0]):.2e}",
        fontsize=9, va="center", family="monospace",
    )
    fig.suptitle("IN62-N06 inlet plane: fluctuation clouds and the divergence source term",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["plane", "duct", "decay", "all", "figures"])
    ap.add_argument("--out", default="runs/in62_n06_inlet_quality")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--decay-n", type=int, nargs="+", default=[64, 48])
    ap.add_argument("--decay-tau", type=float, default=0.505)
    ap.add_argument("--duct-tau", type=float, default=0.505)
    ap.add_argument("--ramp", type=int, default=4)
    ap.add_argument("--cases", nargs="+",
                    default=["dfsem", "sem", "dfm", "dfsem_iso"])
    ap.add_argument("--sigma", type=float, default=8.0,
                    help="synthetic-inflow eddy length scale (cells)")
    ap.add_argument("--decay-steps", type=int, default=2000)
    ap.add_argument("--decay-u-rms", type=float, default=0.05)
    ap.add_argument("--decay-kmin", type=float, default=1.0)
    ap.add_argument("--decay-kcut", type=float, default=4.0,
                    help="spectral cutoff of the initial spectrum (rad/cell)")
    ap.add_argument("--decay-sample-every", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    global SIGMA
    SIGMA = float(args.sigma)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": args.threads,
        "target_stress": TARGET,
        "target_isotropic": TARGET_ISO,
        "gates_thresholds": GATES,
        "module_commit": _repo_commit(),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    plane: dict = {}
    duct: dict = {}
    decay: dict = {}

    if args.stage in ("plane", "all"):
        plane = stage_plane(plane)
        (outdir / "results_plane.json").write_text(json.dumps(plane, indent=2))
    if args.stage in ("duct", "all"):
        cfg = DevelopmentSectionConfig(
            nx=args.nx, ny=args.ny, nz=args.nz, tau=args.duct_tau, n_steps=args.steps,
            n_warmup=args.warmup, sample_every=args.sample_every,
            ramp_cells=args.ramp,
        )
        duct = stage_duct(duct, cfg, cases=tuple(args.cases), outdir=outdir)
        (outdir / "results_duct.json").write_text(json.dumps(duct, indent=2))
    if args.stage in ("decay", "all"):
        decay = stage_decay(resolutions=tuple(args.decay_n), n_steps=args.decay_steps,
                            tau=args.decay_tau, u_rms=args.decay_u_rms,
                            k_cut=args.decay_kcut, k_min=args.decay_kmin,
                            sample_every=args.decay_sample_every)
        (outdir / "results_decay.json").write_text(json.dumps(decay, indent=2))

    # merge whatever exists on disk so figures/report can be built in any order
    for name, target in (("plane", plane), ("duct", duct), ("decay", decay)):
        f = outdir / f"results_{name}.json"
        if f.exists() and not target:
            try:
                loaded = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if name == "plane":
                plane = loaded
            elif name == "duct":
                duct = loaded
            else:
                decay = loaded

    figs = make_figures(outdir, plane or {"cases": {}}, duct, decay)

    ldev = {}
    ldev_path = outdir / "results_ldev.json"
    if ldev_path.exists():
        try:
            ldev = json.loads(ldev_path.read_text())
        except json.JSONDecodeError:
            ldev = {}

    report = {
        "node": "IN62-N06",
        "repo": "TensorLBM",
        "title": "LBM divergence-free synthetic inflow (DFSEM / digital filter) and the development-section self-preservation law",
        "target_stress": TARGET,
        "target_isotropic": TARGET_ISO,
        "gates_thresholds": GATES,
        "gates": {
            **(plane.get("gates", {}) if plane else {}),
            **(duct.get("gates", {}) if duct else {}),
            **(decay.get("gates", {}) if decay else {}),
        },
        "artifacts": {
            "results_plane": str(outdir / "results_plane.json"),
            "results_duct": str(outdir / "results_duct.json"),
            "results_decay": str(outdir / "results_decay.json"),
            "figures": figs,
        },
        "ldev_sweep": ldev,
        "meta": meta,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gates"], indent=2))
    return 0


def _repo_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())