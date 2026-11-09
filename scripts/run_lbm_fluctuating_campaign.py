"""Campaign driver for the fluctuating-LBM statistical-physics anchors.

Writes ``artifacts/lbm_fluc/result.json`` and ``artifacts/lbm_fluc/figures/*.png``.

Acceptance criteria (campaign node p254w1-n06-lbm-fluctuating-hydro):
  J1  Stokes-Einstein  D = kT/(6 pi eta a)                      rel. error <= 10 %
  J2  Green-Kubo       eta = (kT V)^-1 int <P_xy P_xy> dt        rel. error <= 10 %
  J3  equipartition    <|u_perp(k)|^2> = kT/rho per mode         rel. error <= 10 %
  J4  kT = 0 march bitwise identical to plain BGK
  J5  negative controls: noise off == standard LBM (bitwise), noise ~ sqrt(kT)
  J6  relative standard error of the reported means <= 5 %
  J7  discrete sensitivity (box size / tau) <= 5 %

Run ``--stage tasks`` (heavy, shardable over many workers) then ``--stage
aggregate``; the per-task cache in ``artifacts/lbm_fluc/cache/`` makes the task
stage resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tensorlbm.fluctuating_lbm import (
    FluctuatingLBM,
    fdt_covariance,
    gk_viscosity_mode,
    run_brownian_particle,
    traceless_stress,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "lbm_fluc"
CACHE = OUT / "cache"
FIG = OUT / "figures"

BASE = {"nu": 0.1, "kT": 1.0e-3}


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def task_gk(
    seed: int, n: int = 16, nu: float = 0.1, kT: float = 1e-3, n_steps: int = 4000, k_mode: int = 1
) -> dict:
    lbm = FluctuatingLBM((n, n, n), nu=nu, kT=kT, seed=seed)
    r = gk_viscosity_mode(lbm, k_mode=k_mode, n_steps=n_steps, n_equil=max(400, n_steps // 5))
    return {
        "seed": seed,
        "n": n,
        "nu": nu,
        "kT": kT,
        "k_mode": k_mode,
        "k": r.k,
        "eq_ratio": r.eq_ratio,
        "c0": r.c0_lattice,
        "eta_raw": r.eta_gk_raw,
        "eta": r.eta_gk,
        "eta_stderr_block": r.eta_stderr,
        "tau_steps": r.tau_steps,
        "acf0_over_kTV": r.acf0_over_kTV,
        "n_samples": r.n_samples,
    }


def task_brownian(seed: int, n: int = 16, steps: int = 6000) -> dict:
    r = run_brownian_particle(n=n, nu=0.05, kT=1.0e-3, radius=1.5, n_steps=steps, seed=seed)
    return {
        "seed": seed,
        "radius": r.radius,
        "gamma": r.gamma_stokes,
        "eta": r.eta,
        "kT": r.kT,
        "mass": r.mass,
        "D": r.diffusion,
        "D_se": r.d_stokes_einstein,
        "D_stderr": r.diffusion_se,
        "rel_error": r.rel_error,
        "D_vacf": r.vacf_diffusion,
        "n_steps": r.n_steps,
        "fit_window": list(r.fit_window),
        # store a coarse MSD curve for the figure (every 10th lag)
        "msd_t": r.times[::10].tolist(),
        "msd_v": r.msd[::10].tolist(),
    }


def task_controls() -> dict:
    out: dict = {}
    n, steps = 16, 60
    a = FluctuatingLBM((n, n), nu=0.1, kT=0.0, seed=1)
    b = FluctuatingLBM((n, n), nu=0.1, kT=0.0, seed=1)
    a.step(steps)
    b.step_reference_bgk(steps)
    out["kT_zero_max_abs_diff"] = float(np.abs(a.f - b.f).max())
    out["kT_zero_bitwise_identical"] = bool(np.array_equal(a.f, b.f))
    c = FluctuatingLBM((n, n), nu=0.1, kT=1e-3, seed=2)
    d = FluctuatingLBM((n, n), nu=0.1, kT=1e-3, seed=2)
    c.step(steps, noise=False)
    d.step_reference_bgk(steps)
    out["noise_off_bitwise_identical"] = bool(np.array_equal(c.f, d.f))
    out["noise_off_max_abs_diff"] = float(np.abs(c.f - d.f).max())

    kTs = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
    amps = [FluctuatingLBM((12, 12), nu=0.1, kT=x, seed=5).noise_amplitude(20) for x in kTs]
    slope = float(
        np.polyfit(np.log2(np.array(kTs) / kTs[0]), np.log2(np.array(amps) / amps[0]), 1)[0]
    )
    out["sqrtT"] = {
        "kT": kTs,
        "noise_rms": amps,
        "log_log_slope": slope,
        "ratio_over_sqrt_kT": (np.array(amps) / np.sqrt(kTs)).tolist(),
    }

    for dim in (2, 3):
        rng = np.random.default_rng(0)
        ns = 200000
        s = traceless_stress(rng, (ns,), dim, 1e-3, 0.1)
        emp = np.einsum("abn,cdn->abcd", s, s) / ns
        ana = fdt_covariance(dim, 1e-3, 0.1)
        out[f"fdt_covariance_{dim}d"] = {
            "max_abs_dev": float(np.abs(emp - ana).max()),
            "max_rel_dev": float(np.abs(emp - ana).max() / np.abs(ana).max()),
            "cov_xyxy_empirical": float(emp[0, 1, 0, 1]),
            "cov_xyxy_analytic": float(ana[0, 1, 0, 1]),
            "trace_max": float(np.abs(sum(s[a, a] for a in range(dim))).max()),
        }
    return out


def task_spectrum(seed: int = 21, n: int = 32, steps: int = 6000) -> dict:
    """2D shear-mode spectrum + real-space variance (UV suppression demo)."""
    lbm = FluctuatingLBM((n, n), nu=0.1, kT=1e-3, seed=seed)
    lbm.step(1000)
    grids = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ms = (1, 2, 3, 4, 6, 8)
    ph = {m: np.exp(-2j * np.pi * m * grids[0] / n) for m in ms}
    acc = {m: 0.0 for m in ms}
    n_s = steps // 4
    var = 0.0
    for _ in range(n_s):
        lbm.step(4)
        u = lbm.velocity(half_force=False)
        var += float(np.mean(u * u))
        for m in ms:
            acc[m] += float(np.abs(np.sum(u[1] * ph[m])) ** 2)
    kTV = lbm.kT * n * n
    return {
        "n": n,
        "k": [2 * np.pi * m / n for m in ms],
        "mode_power_over_kTV": [acc[m] / n_s / kTV for m in ms],
        "realspace_var": var / n_s,
        "kT": lbm.kT,
        "realspace_over_kT": (var / n_s) / lbm.kT,
        "n_samples": n_s,
    }


TASKS = {
    "controls": task_controls,
    "spectrum": task_spectrum,
}


def build_task_list(seeds: int, steps: int) -> list[tuple[str, tuple]]:
    jobs: list[tuple[str, tuple]] = [("controls", ()), ("spectrum", ())]
    for s in range(seeds):
        jobs.append(("gk", (s, 16, BASE["nu"], BASE["kT"], steps, 1)))
    for s in range(max(2, seeds // 8)):
        jobs.append(("brownian", (100 + s, 16, steps + 2000)))
    # sensitivity over tau (same seeds -> paired comparison)
    for tau in (0.7, 0.9):
        nu = (tau - 0.5) / 3.0
        for s in range(max(2, seeds // 8)):
            jobs.append(("gk", (200 + s, 16, nu, BASE["kT"], steps, 1)))
    # sensitivity over box size
    for n in (12, 20):
        for s in range(max(2, seeds // 8)):
            jobs.append(("gk", (300 + s, n, BASE["nu"], BASE["kT"], steps, 1)))
    return jobs


def run_job(job: tuple[str, tuple]) -> tuple[str, dict]:
    kind, args = job
    if kind == "controls":
        return "controls", TASKS["controls"]()
    if kind == "spectrum":
        return "spectrum", TASKS["spectrum"](*args)
    if kind == "gk":
        seed, n, nu, kT, steps, k_mode = args
        return f"gk_s{seed}_n{n}_nu{nu:.5f}_m{k_mode}", task_gk(*args)
    if kind == "brownian":
        return f"brownian_s{args[0]}_n{args[1]}", task_brownian(*args)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(result: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    v = result["verdict"]

    # 1. Stokes-Einstein: MSD + fit
    br = result["brownian"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, r in enumerate(br):
        t = np.array(r["msd_t"])
        m = np.array(r["msd_v"])
        if i == 0:
            ax.plot(t, m, ".", ms=2, color="tab:blue", label="measured MSD")
            lo, hi = r["fit_window"]
            ax.plot(
                t[(t >= lo) & (t <= hi)],
                6 * r["D"] * t[(t >= lo) & (t <= hi)],
                "-",
                color="tab:red",
                label="fit slope 6D, D=%.3e" % r["D"],
            )
            ax.plot(
                t,
                6 * r["D_se"] * t,
                "--",
                color="k",
                label="Stokes-Einstein 6D_SE t, D_SE=%.3e" % r["D_se"],
            )
    ax.set_xlabel("t [lattice steps]")
    ax.set_ylabel(r"$\langle \Delta r^2\rangle$ [lattice$^2$]")
    ax.set_title(
        "J1 Stokes-Einstein (a=%.1f, eta=%.3f, kT=%.0e)"
        % (br[0]["radius"], br[0]["eta"], br[0]["kT"])
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "j1_stokes_einstein_msd.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # 2. Green-Kubo: eta per seed + mean
    gk = result["gk_runs"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    eta = np.array([r["eta"] for r in gk])
    eta_raw = np.array([r["eta_raw"] for r in gk])
    ax.plot(
        np.arange(eta.size) + 1,
        eta_raw,
        "o",
        ms=4,
        mfc="none",
        label=r"raw lattice moment ($c_0=1$)",
    )
    ax.plot(np.arange(eta.size) + 1, eta, "s", ms=4, label=r"with $c_0$ correction")
    ax.axhline(gk[0]["nu"], color="k", ls="--", label=r"input $\eta=\nu\rho$")
    ax.axhline(
        eta.mean(),
        color="tab:green",
        ls=":",
        label=r"mean %.4f $\pm$ %.4f" % (eta.mean(), v["J2_green_kubo"]["eta_gk_stderr"]),
    )
    ax.set_xlabel("seed")
    ax.set_ylabel(r"$\eta_{GK}$")
    ax.set_title("J2 Green-Kubo viscosity per realisation (k=2$\\pi$/L)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "j2_green_kubo_per_seed.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # 3. equipartition: mode power spectrum (2D) + per-seed ratios
    sp = result["spectrum"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    a1.semilogx(sp["k"], sp["mode_power_over_kTV"], "o-")
    a1.axhline(1.0, color="k", ls="--", label="kT/rho (equipartition)")
    a1.set_xlabel("k")
    a1.set_ylabel(r"$\langle|u_\perp(k)|^2\rangle/(k_BTV)$")
    a1.set_title("mode resolved equipartition (2D, L=%d)" % sp["n"])
    a1.legend(fontsize=7)
    eq = np.array([r["eq_ratio"] for r in gk])
    a2.hist(eq, bins=8, color="tab:blue", alpha=0.8)
    a2.axvline(1.0, color="k", ls="--")
    a2.axvline(eq.mean(), color="tab:red", label="mean %.3f" % eq.mean())
    a2.set_xlabel(r"$\langle|u_\perp(k)|^2\rangle/(k_BTV)$")
    a2.set_ylabel("realisations")
    a2.set_title("J3 equipartition, 3D L=16")
    a2.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "j3_equipartition.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # 4. negative control: noise amplitude vs sqrt(kT)
    ctl = result["controls"]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    kT = np.array(ctl["sqrtT"]["kT"])
    amp = np.array(ctl["sqrtT"]["noise_rms"])
    ax.loglog(kT, amp, "o", label="measured noise RMS")
    ax.loglog(
        kT,
        amp[0] * np.sqrt(kT / kT[0]),
        "-",
        label="slope 1/2 fit (%.3f)" % ctl["sqrtT"]["log_log_slope"],
    )
    ax.set_xlabel("kT")
    ax.set_ylabel("RMS of injected force")
    ax.set_title("J5 negative control: noise amplitude $\\propto\\sqrt{k_BT}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "j5_sqrt_kT.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # 5. viscosity vs k (box size) extrapolation
    res = result["resolution"]
    if len(res) >= 2:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        kk = np.array([2 * np.pi / r["n"] for r in res])
        ee = np.array([r["eta_mean"] for r in res])
        ax.plot(kk**2, ee, "o-", label=r"$\eta_{GK}(k)$")
        try:
            sl, ic = np.polyfit(kk**2, ee, 1)
            ax.plot(
                [0, kk.max() ** 2 * 1.05],
                [ic, ic + sl * kk.max() ** 2 * 1.05],
                "--",
                label=r"k$\to$0 extrapolation: %.4f" % ic,
            )
        except Exception:  # pragma: no cover
            pass
        ax.axhline(res[0].get("nu", 0.1), color="k", ls=":", label="input $\\eta$")
        ax.set_xlabel(r"$k^2$")
        ax.set_ylabel(r"$\eta_{GK}$")
        ax.set_title("J7 discrete sensitivity: eta vs box size")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIG / "j7_viscosity_vs_k.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p.name)
    return made


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def stage_tasks(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = build_task_list(args.seeds, args.steps)
    todo = []
    for job in jobs:
        kind, jargs = job
        if kind in ("controls", "spectrum"):
            key = kind
        elif kind == "gk":
            seed, n, nu, kT, steps, k_mode = jargs
            key = f"gk_s{seed}_n{n}_nu{nu:.5f}_m{k_mode}"
        else:
            key = f"brownian_s{jargs[0]}_n{jargs[1]}"
        if not (CACHE / f"{key}.json").exists():
            todo.append((key, job))
    print(f"[tasks] {len(jobs)} total, {len(todo)} to run, workers={args.workers}")
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, job): key for key, job in todo}
        for k, fut in enumerate(futs):
            key = futs[fut]
            try:
                _, data = fut.result()
            except Exception as exc:  # pragma: no cover
                print(f"  !! {key} failed: {exc}")
                continue
            (CACHE / f"{key}.json").write_text(json.dumps(data, default=float))
            print(f"  {key} done ({time.time() - t0:.0f}s elapsed)")
    print(f"[tasks] finished in {time.time() - t0:.0f}s")


def _load(pattern: str) -> list[dict]:
    out = []
    for p in sorted(CACHE.glob(pattern)):
        out.append(json.loads(p.read_text()))
    return out


def stage_aggregate(args: argparse.Namespace) -> int:
    gk_runs = sorted(_load("gk_*.json"), key=lambda r: (r["n"], r["nu"], r["seed"]))
    brown = sorted(_load("brownian_*.json"), key=lambda r: r["seed"])
    controls = json.loads((CACHE / "controls.json").read_text())
    spectrum = json.loads((CACHE / "spectrum.json").read_text())
    if not gk_runs or not brown:
        raise SystemExit("task cache is incomplete; run --stage tasks first")

    base = [r for r in gk_runs if r["n"] == 16 and abs(r["nu"] - BASE["nu"]) < 1e-12]
    eta = np.array([r["eta"] for r in base])
    eta_raw = np.array([r["eta_raw"] for r in base])
    eq = np.array([r["eq_ratio"] for r in base])
    c0 = np.array([r["c0"] for r in base])
    eta_mean = float(eta.mean())
    eta_se = float(eta.std(ddof=1) / np.sqrt(eta.size))
    eq_mean = float(eq.mean())
    eq_se = float(eq.std(ddof=1) / np.sqrt(eq.size))

    D = np.array([r["D"] for r in brown])
    D_se = np.array([r["D_se"] for r in brown])
    D_mean = float(D.mean())
    D_stderr = float(D.std(ddof=1) / np.sqrt(D.size))
    D_se_val = float(D_se.mean())

    # sensitivities (paired by seed, per configuration mean)
    def group(pred):
        return [r for r in gk_runs if pred(r)]

    tau_levels = {}
    for tau, nu in ((0.7, (0.7 - 0.5) / 3.0), (0.8, BASE["nu"]), (0.9, (0.9 - 0.5) / 3.0)):
        rows = group(lambda r, nu=nu: r["n"] == 16 and abs(r["nu"] - nu) < 1e-9)
        if rows:
            tau_levels[tau] = {
                "nu": nu,
                "eta_norm": float(np.mean([r["eta"] / r["nu"] for r in rows])),
                "eta_mean": float(np.mean([r["eta"] for r in rows])),
                "n_seeds": len(rows),
            }
    res_levels = {}
    for n in (12, 16, 20):
        rows = group(lambda r, n=n: r["n"] == n and abs(r["nu"] - BASE["nu"]) < 1e-12)
        if rows:
            res_levels[n] = {
                "k": 2 * np.pi / n,
                "eta_mean": float(np.mean([r["eta"] for r in rows])),
                "eta_norm": float(np.mean([r["eta"] / BASE["nu"] for r in rows])),
                "eq_mean": float(np.mean([r["eq_ratio"] for r in rows])),
                "n_seeds": len(rows),
            }
    spreading_tau = (
        max(v["eta_norm"] for v in tau_levels.values())
        / min(v["eta_norm"] for v in tau_levels.values())
        - 1.0
    )
    eta_res = np.array([v["eta_norm"] for v in res_levels.values()])
    spreading_res = float(eta_res.max() / eta_res.min() - 1.0) if eta_res.size > 1 else float("nan")

    # k -> 0 extrapolation of eta(k)/eta_input
    if len(res_levels) >= 2:
        k2 = np.array([v["k"] ** 2 for v in res_levels.values()])
        en = np.array([v["eta_mean"] / (v["k"] * 0 + BASE["nu"]) for v in res_levels.values()])
        sl, ic = np.polyfit(k2, en, 1)
        eta_extrap = float(ic)
    else:
        eta_extrap = float("nan")

    verdict = {
        "J1_stokes_einstein": {
            "criterion": "|D_meas - kT/(6 pi eta a)|/D_SE <= 10%",
            "pass": abs(D_mean - D_se_val) / D_se_val <= 0.10,
            "D_measured": D_mean,
            "D_measured_stderr": D_stderr,
            "D_stokes_einstein": D_se_val,
            "rel_error": (D_mean - D_se_val) / D_se_val,
            "n_realisations": int(D.size),
            "radius": brown[0]["radius"],
            "gamma": brown[0]["gamma"],
            "kT": brown[0]["kT"],
            "eta": brown[0]["eta"],
            "n_steps": brown[0]["n_steps"],
        },
        "J2_green_kubo": {
            "criterion": "|eta_GK - eta_input|/eta_input <= 10%",
            "pass": abs(eta_mean - BASE["nu"]) / BASE["nu"] <= 0.10,
            "eta_input": BASE["nu"],
            "eta_gk_mean": eta_mean,
            "eta_gk_stderr": eta_se,
            "rel_error": (eta_mean - BASE["nu"]) / BASE["nu"],
            "eta_gk_raw_mean": float(eta_raw.mean()),
            "eta_gk_raw_rel_error": float((eta_raw.mean() - BASE["nu"]) / BASE["nu"]),
            "lattice_response_c0_mean": float(c0.mean()),
            "c0_scatter": float(c0.std(ddof=1)),
            "eta_extrapolated_k0": eta_extrap,
            "n_realisations": int(eta.size),
            "per_seed": eta.tolist(),
        },
        "J3_equipartition": {
            "criterion": "|<|u_perp(k)|^2>/(kT V) - 1| <= 10% (mode resolved)",
            "pass": abs(eq_mean - 1.0) <= 0.10,
            "eq_ratio_mean": eq_mean,
            "eq_ratio_stderr": eq_se,
            "rel_error": eq_mean - 1.0,
            "n_realisations": int(eq.size),
            "realspace_var_over_kT": spectrum["realspace_over_kT"],
            "realspace_note": (
                "real-space <u^2>/kT is ~6x below the continuum value: the lattice "
                "momentum-flux divergence has the symbol sin(k) which suppresses the "
                "short-wavelength (UV) modes, so only the hydrodynamic modes carry "
                "kT/rho each. The mode-resolved statement is the physical one."
            ),
            "spectrum": spectrum,
        },
        "J4_zero_temperature_bitwise": {
            "criterion": "kT=0 march bitwise identical to plain BGK (max|diff| = 0)",
            "pass": controls["kT_zero_bitwise_identical"]
            and controls["kT_zero_max_abs_diff"] == 0.0,
            "max_abs_diff": controls["kT_zero_max_abs_diff"],
            "bitwise_identical": controls["kT_zero_bitwise_identical"],
        },
        "J5_negative_controls": {
            "criterion": "no noise == standard LBM (bitwise); dlog(noise)/dlog(kT) = 0.5",
            "pass": controls["noise_off_bitwise_identical"]
            and abs(controls["sqrtT"]["log_log_slope"] - 0.5) <= 0.02,
            "noise_off_bitwise_identical": controls["noise_off_bitwise_identical"],
            "noise_off_max_abs_diff": controls["noise_off_max_abs_diff"],
            "sqrtT": controls["sqrtT"],
            "fdt_covariance_2d": controls["fdt_covariance_2d"],
            "fdt_covariance_3d": controls["fdt_covariance_3d"],
        },
        "J6_statistical_error": {
            "criterion": "relative standard error of reported means <= 5%",
            "pass": max(eq_se, eta_se / BASE["nu"], D_stderr / abs(D_mean)) <= 0.05,
            "eta_gk_rel_se": eta_se / BASE["nu"],
            "eq_ratio_rel_se": eq_se,
            "D_rel_se": D_stderr / abs(D_mean),
            "n_realisations_gk": int(eta.size),
            "n_realisations_D": int(D.size),
        },
        "J7_discrete_sensitivity": {
            "criterion": "spread of eta/nu over tau and L <= 5% (normalised)",
            "pass": bool(spreading_tau <= 0.05 and spreading_res <= 0.05),
            "tau_levels": tau_levels,
            "resolution_levels": res_levels,
            "spread_over_tau": spreading_tau,
            "spread_over_L": spreading_res,
        },
    }

    all_pass = all(bool(v["pass"]) for v in verdict.values())
    result = {
        "campaign_node": "p254w1-n06-lbm-fluctuating-hydro",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/fluctuating_lbm.py",
        "method": (
            "Landau-Lifshitz random stress with the exact traceless FDT covariance, "
            "injected as a force through a Guo-forced BGK collision (second order "
            "accurate, nu = c_s^2 (tau - 1/2) preserved)"
        ),
        "lattice_units": "dx = dt = rho0 = 1, c_s^2 = 1/3, k_B = 1",
        "config": {
            "base": BASE,
            "n_box_gk": 16,
            "n_seeds_gk": int(len(base)),
            "n_steps_gk": int(base[0]["k"] * 0 + base[0]["n_samples"]),
            "brownian": {
                "n": 16,
                "nu": 0.05,
                "radius": 1.5,
                "kT": 1e-3,
                "n_steps": brown[0]["n_steps"],
            },
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "gk_runs": base,
        "brownian": brown,
        "controls": controls,
        "spectrum": spectrum,
        "resolution": [{"n": n, **v} for n, v in sorted(res_levels.items())],
        "tau_sensitivity": tau_levels,
    }
    figures = make_figures(result)
    result["figures"] = figures
    (OUT / "result.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"[aggregate] all_pass={all_pass}")
    for k, v in verdict.items():
        print(f"  {k}: pass={v['pass']}")
    print(f"  wrote {OUT / 'result.json'} and {len(figures)} figures")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tasks", "aggregate", "all"), default="all")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    if args.stage in ("tasks", "all"):
        stage_tasks(args)
    if args.stage in ("aggregate", "all"):
        return stage_aggregate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
