#!/usr/bin/env python
"""Driver for the D3Q27 collision-operator verification face (mission p246w2-n01).

Produces, per task, a JSON artefact under ``artifacts/cumulant_verification/``:

============  ==========================================================
``a1a2``      A1 equilibrium moment identities, A2 transform round-trip and
              collision conservation (all operators)
``gali``      A3/A7 Galilean-defect **definition matrix**: D1 (single
              collision), D2 (stress-mode amplitude in a stream), D3 (decay
              rate in a stream) plus amplitude-independence and
              collision-free controls
``stability`` A5/A4 high-Re Taylor–Green τ sweep, ``(Ma, τ, operator)`` table,
              with an online realizability (``λ_min``) monitor
``poiseuille`` A6 force-driven channel profile vs the parabolic solution
``realizability`` A4 long run with a λ_min / ρ_min time series
``figures``   PNG figures from the collected JSON

Usage
-----
    python experiments/cumulant_d3q27_verification.py --task gali
    python experiments/cumulant_d3q27_verification.py --task stability --shard 0/8
    python experiments/cumulant_d3q27_verification.py --task figures

Deterministic: no random seeds other than the fixed ones inside the module, and
every number in the output JSON is recomputed by the command recorded in
``meta.command``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cumulant_verification import (  # noqa: E402
    _decay_rate,  # noqa: E402
    collision_conservation_error,
    covariance_spectrum,
    fit_power_law_slope,
    kinetic_energy,
    make_operator,
    moment_identity_errors,
    poiseuille_profile,
    shear_wave_amplitude_series,
    single_collision_stress_defect,
    stability_run,
    taylor_green_setup,
    transform_roundtrip_error,
)

OUT = ROOT / "artifacts" / "cumulant_verification"
FIG = ROOT / "figures"
ALL_OPS = ("bgk", "kbc", "cumulant", "cumulant_single_rate")


def _meta(task: str) -> dict:
    return {
        "task": task,
        "command": " ".join([sys.executable, *sys.argv]),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
    }


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(f"[write] {path}")
    return path


# ---------------------------------------------------------------------------
# A1 / A2
# ---------------------------------------------------------------------------
def task_a1a2() -> dict:
    payload = {
        "meta": _meta("a1a2"),
        "A1_moment_identities": moment_identity_errors(),
        "A2_transform_roundtrip": transform_roundtrip_error(),
        "A2_collision_conservation": {
            op: collision_conservation_error(op, 0.55) for op in ALL_OPS
        },
    }
    a1 = payload["A1_moment_identities"]
    a2 = payload["A2_transform_roundtrip"]
    cons = payload["A2_collision_conservation"]
    payload["verdict"] = {
        "A1_pass": max(
            a1["mass_rel_err"], a1["momentum_rel_err"], a1["second_moment_rel_err"]
        )
        < 1e-12,
        "A2_transform_pass": a2["cumulant_roundtrip_rel_err"] < 1e-13,
        "A2_conservation_pass": max(
            max(v["mass_rel_err"], v["momentum_rel_err"]) for v in cons.values()
        )
        < 1e-14,
    }
    return _write("a1a2.json", payload)


# ---------------------------------------------------------------------------
# A3 / A7 — defect definition matrix
# ---------------------------------------------------------------------------
def task_gali(
    *,
    n: int = 32,
    nsteps: int = 600,
    amp: float = 2.0e-2,
    amp_control: float = 1.0e-3,
    tau: float = 0.55,
    machs: tuple[float, ...] = (0.025, 0.05, 0.1, 0.2),
    ops: tuple[str, ...] = ("bgk", "kbc", "cumulant", "cumulant_single_rate", "no_collision"),
    window: int = 300,
) -> dict:
    """D1/D2/D3 defect definitions for every operator + two controls.

    * ``D1`` single-collision stress defect (operator-only, no streaming)
    * ``D2`` shear-stress Fourier-amplitude defect in a uniform stream
    * ``D3`` shear-wave decay-rate defect in a uniform stream (transport level)
    * control ``amp``  — D3 must be amplitude independent (linear response)
    * control ``no_collision`` — how much of D2 survives with streaming only
    """
    results: dict[str, dict] = {}
    for op in ops:
        t0 = time.perf_counter()
        d1 = single_collision_stress_defect(op, tau, mach=0.1, amp=amp, n=n)
        ref = shear_wave_amplitude_series(op, tau, mach=0.0, amp=amp, n=n, nsteps=nsteps)
        ref_series = list(ref["amplitudes"])
        lam0 = _decay_rate(ref_series[window:])
        lam0_early = _decay_rate(ref_series[:window])
        d2: list[float] = []
        d3: list[float] = []
        decay: list[list[float]] = []
        for ma in machs:
            run = shear_wave_amplitude_series(
                op, tau, mach=ma, amp=amp, n=n, nsteps=nsteps
            )
            series = list(run["amplitudes"])
            m = min(len(series), len(ref_series))
            d2.append(
                max(abs(series[i] - ref_series[i]) for i in range(window, m))
                / ref_series[window]
            )
            lam = _decay_rate(series[window:])
            decay.append([lam, _decay_rate(series[:window])])
            d3.append(abs(lam / lam0 - 1.0))
        # amplitude-independence control (D3 at low amplitude, max Ma only)
        run_low = shear_wave_amplitude_series(
            op, tau, mach=machs[-1], amp=amp_control, n=n, nsteps=nsteps
        )
        ref_low = shear_wave_amplitude_series(
            op, tau, mach=0.0, amp=amp_control, n=n, nsteps=nsteps
        )
        d3_low = abs(
            _decay_rate(list(run_low["amplitudes"])[window:])
            / _decay_rate(list(ref_low["amplitudes"])[window:])
            - 1.0
        )
        results[op] = {
            "D1_single_collision_defect_ma0.1": d1,
            "D2_amplitude_defect": d2,
            "D3_decay_rate_defect": d3,
            "D2_slope": fit_power_law_slope(list(machs), d2),
            "D3_slope": fit_power_law_slope(list(machs), d3),
            "decay_rate_rest": lam0,
            "decay_rate_rest_early_window": lam0_early,
            "decay_rates_late_early": decay,
            "D3_amplitude_control": d3_low,
            "D3_amplitude_control_ratio": d3_low / d3[-1] if d3[-1] else float("nan"),
            "elapsed_s": time.perf_counter() - t0,
        }
        print(
            f"[gali] {op:22s} D1={d1:.2e} D2({machs[-1]})={d2[-1]:.3e} "
            f"D3({machs[-1]})={d3[-1]:.3e} slopeD2={results[op]['D2_slope']:.2f}",
            flush=True,
        )

    payload = {
        "meta": _meta("gali"),
        "settings": {
            "tau": tau,
            "n": n,
            "nsteps": nsteps,
            "amp": amp,
            "amp_control": amp_control,
            "machs": list(machs),
            "window": window,
            "theory_decay_rate": (1.0 / 3.0) * (tau - 0.5) * (2.0 * math.pi / n) ** 2,
        },
        "results": results,
    }
    # A3 / A7 verdicts
    cum = results["cumulant"] if "cumulant" in results else None
    bgk = results["bgk"]
    neg = results.get("cumulant_single_rate")
    payload["verdict"] = {
        "A3_cumulant_defect_at_ma0.1": (
            cum["D2_amplitude_defect"][list(machs).index(0.1)] if cum and 0.1 in machs else None
        ),
        "A3_cumulant_machine_precision_pass": (
            bool(cum["D2_amplitude_defect"][list(machs).index(0.1)] <= 1e-10)
            if cum and 0.1 in machs
            else None
        ),
        "A3_bgk_measurable_pass": bool(bgk["D2_amplitude_defect"][0] > 1e-10),
        "A3_bgk_slope_in_2.0pm0.3": bool(abs(bgk["D2_slope"] - 2.0) <= 0.3),
        "A7_negative_control_slope_in_2.0pm0.3": (
            bool(abs(neg["D2_slope"] - 2.0) <= 0.3) if neg else None
        ),
        "operator_spread_at_ma0.1": {
            op: results[op]["D2_amplitude_defect"][list(machs).index(0.1)]
            for op in ("bgk", "cumulant", "cumulant_single_rate")
            if op in results and 0.1 in machs
        },
    }
    return _write("gali.json", payload)


# ---------------------------------------------------------------------------
# A5 / A4 — stability domain
# ---------------------------------------------------------------------------
def _stability_configs(u0: float = 0.05) -> list[dict]:
    """Declared (operator, τ) grid; τ−½ log-spaced to cover Re 19 … 4.7e3."""
    dtaus = (5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4)
    taus = [0.5 + d for d in dtaus]
    cfgs: list[dict] = []
    for op in ("bgk", "cumulant", "cumulant_single_rate"):
        for tau in taus:
            cfgs.append({"operator": op, "tau": tau, "u0": u0, "nsteps": 1500})
    # KBC is ~7× the cost per step: cross-check on the four stiffest τ only.
    for tau in taus[3:]:
        cfgs.append({"operator": "kbc", "tau": tau, "u0": u0, "nsteps": 1500})
    # Lower-Mach second pass to separate the Ma and Re axes: 2.4e-4 is chosen so
    # that (u0=0.02, tau-0.5=2.4e-4) sits at the *same* Re as (u0=0.05, 1e-3)
    # with Ma smaller by 2.5x.
    for op in ("bgk", "cumulant"):
        for dtau in (1e-3, 2.4e-4):
            cfgs.append({"operator": op, "tau": 0.5 + dtau, "u0": 0.02, "nsteps": 1500})
    return cfgs


def task_stability(
    *,
    shard: tuple[int, int] | None = None,
    only: tuple[str, ...] | None = None,
    n: int = 32,
    check_every: int = 20,
    kbc_max_iter: int = 24,
) -> dict:
    cfgs = _stability_configs()
    if only:
        cfgs = [c for c in cfgs if c["operator"] in only]
    if shard is not None:
        i, m = shard
        cfgs = [c for k, c in enumerate(cfgs) if k % m == i]
    rows = []
    for cfg in cfgs:
        op = cfg["operator"]
        op_kwargs = {"kbc_max_iter": kbc_max_iter} if op == "kbc" else None
        try:
            run = stability_run(
                op,
                cfg["tau"],
                n=n,
                u0=cfg["u0"],
                nsteps=cfg["nsteps"],
                check_every=check_every,
                track_realizability=True,
                op_kwargs=op_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - one bad config must not lose a shard
            print(f"[stab] ERROR {cfg}: {exc!r}", flush=True)
            rows.append({**cfg, "error": repr(exc)})
            continue
        row = run.as_dict()
        row["d_tau"] = cfg["tau"] - 0.5
        rows.append(row)
        print(
            f"[stab] {op:22s} tau-0.5={cfg['tau'] - 0.5:.1e} Re={run.re:.1f} "
            f"steps={run.steps_completed}/{run.steps_requested} "
            f"div={run.diverged}@{run.fail_step} lam_min={run.lambda_min:.3e} "
            f"({run.elapsed_s:.0f}s)",
            flush=True,
        )
    payload = {
        "meta": _meta("stability"),
        "settings": {
            "n": n,
            "check_every": check_every,
            "u0_default": 0.05,
            "kbc_max_iter": kbc_max_iter,
            "shard": list(shard) if shard else None,
            "re_definition": "u0 * 2*pi / nu, nu = cs2*(tau-0.5)",
        },
        "rows": rows,
    }
    name = "stability.json" if shard is None else f"stability_shard_{shard[0]}of{shard[1]}.json"
    if only:
        name = name.replace(".json", "_" + "-".join(only) + ".json")
    return _write(name, payload)


# ---------------------------------------------------------------------------
# A6 — Poiseuille
# ---------------------------------------------------------------------------
def task_poiseuille(
    *, ny: int = 32, nsteps: int = 40000, tau: float = 0.8, only: tuple[str, ...] | None = None
) -> dict:
    """A6 part 1.  ``tau = 0.8`` (ν = 0.1) keeps the diffusive transient
    ``H²/ν`` inside the step budget: at ``tau = 0.55`` the channel needs
    ``H²/ν ≈ 15 000`` steps for ``H = 16`` and an unconverged run reports a fake
    8 % viscosity error (measured; see the report's pitfalls section)."""
    rows = {}
    for op in (only or ALL_OPS):
        r = poiseuille_profile(op, tau, ny=ny, nsteps=nsteps, sample_every=200)
        rows[op] = r
        print(
            f"[pois] {op:22s} nu_err={r['nu_rel_err']:.2e} prof_err={r['rel_err']:.2e} "
            f"converged={r['converged']} steps={r['steps']}",
            flush=True,
        )
    payload = {
        "meta": _meta("poiseuille"),
        "settings": {"ny": ny, "nsteps": nsteps, "tau": tau, "force": 2.0e-6},
        "rows": rows,
        "verdict": {
            "A6_poiseuille_viscosity_1pct_all_operators": all(
                v["nu_rel_err"] < 0.01 for v in rows.values()
            ),
            "A6_poiseuille_profile_1pct_all_operators": all(
                v["rel_err"] < 0.01 for v in rows.values()
            ),
            "all_converged": all(v["converged"] for v in rows.values()),
        },
    }
    return _write(
        "poiseuille.json" if not only else "poiseuille_" + "-".join(only) + ".json", payload
    )


# ---------------------------------------------------------------------------
# A4 — long-run realizability
# ---------------------------------------------------------------------------
def task_realizability(
    *,
    n: int = 16,
    nsteps: int = 20000,
    d_tau: float = 2e-3,
    u0: float = 0.05,
    every: int = 100,
    only: tuple[str, ...] | None = None,
) -> dict:
    rows = {}
    for op in (only or ("bgk", "cumulant", "cumulant_single_rate")):
        f = taylor_green_setup(n, u0=u0)
        opf = make_operator(op, 0.5 + d_tau)
        series = []
        lam_min, rho_min, f_min = float("inf"), float("inf"), float("inf")
        ke0 = kinetic_energy(f)
        diverged_at = -1
        for step in range(1, nsteps + 1):
            from tensorlbm.d3q27 import stream27_roll

            f = stream27_roll(opf(f))
            if not torch.isfinite(f).all():
                diverged_at = step
                break
            if step % every == 0:
                cov = covariance_spectrum(f)
                lam_min = min(lam_min, cov["lambda_min"])
                rho_min = min(rho_min, cov["rho_min"])
                f_min = min(f_min, cov["f_min"])
                if step % (every * 20) == 0:
                    series.append(
                        [step, cov["lambda_min"], cov["rho_min"], kinetic_energy(f)]
                    )
                if cov["lambda_min"] < -1e-12 or cov["rho_min"] <= 0:
                    pass  # recorded through the min values
        rows[op] = {
            "operator": op,
            "d_tau": d_tau,
            "re": float(u0 * 2 * math.pi / ((1 / 3) * d_tau)),
            "steps_completed": step if diverged_at < 0 else diverged_at,
            "diverged_at": diverged_at,
            "lambda_min": lam_min,
            "rho_min": rho_min,
            "f_min": f_min,
            "kelvin_initial": ke0,
            "series": series,
        }
        print(
            f"[real] {op:22s} steps={rows[op]['steps_completed']} lam_min={lam_min:.3e} "
            f"rho_min={rho_min:.3e} div={diverged_at}",
            flush=True,
        )
    payload = {
        "meta": _meta("realizability"),
        "settings": {"n": n, "nsteps": nsteps, "d_tau": d_tau, "u0": u0, "every": every},
        "rows": rows,
        "verdict": {
            "A4_psd_all_operators": all(v["lambda_min"] >= -1e-12 for v in rows.values()),
            "A4_positive_density": all(v["rho_min"] > 0 for v in rows.values()),
            "note": "1e5-step target reduced to the step count actually run; registered in the report",
        },
    }
    return _write(
        "realizability.json" if not only else "realizability_" + "-".join(only) + ".json", payload
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def task_figures() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    gali_path = OUT / "gali.json"
    stab_path = OUT / "stability.json"
    pois_path = OUT / "poiseuille.json"
    real_path = OUT / "realizability.json"

    def merged(base: Path, pattern: str) -> dict | None:
        """Merge a base artefact with per-operator/ per-shard siblings."""
        rows: dict | list = None
        settings: dict = {}
        srcs = ([base] if base.exists() else []) + sorted(OUT.glob(pattern))
        if not srcs:
            return None
        for p in srcs:
            d = json.loads(p.read_text())
            if "rows" in d and isinstance(d["rows"], dict):
                rows = dict(rows or {})
                rows.update(d["rows"])
            elif "rows" in d:
                rows = list(rows or [])
                have = {(r.get("operator"), r.get("tau"), r.get("mach")) for r in rows}
                for r in d["rows"]:
                    if (r.get("operator"), r.get("tau"), r.get("mach")) not in have:
                        rows.append(r)
            settings.update({k: v for k, v in d.get("settings", {}).items() if k not in settings})
        return {"rows": rows, "settings": settings}

    stab_any = merged(stab_path, "stability_*.json")
    if stab_any:
        stab_path = OUT / "_stab_merged.json"
        stab_path.write_text(json.dumps(stab_any))
    pois_any = merged(pois_path, "poiseuille_*.json")
    if pois_any:
        pois_path = OUT / "_pois_merged.json"
        pois_path.write_text(json.dumps(pois_any))
    real_any = merged(real_path, "realizability_*.json")
    if real_any:
        real_path = OUT / "_real_merged.json"
        real_path.write_text(json.dumps(real_any))

    if gali_path.exists():
        g = json.loads(gali_path.read_text())
        machs = g["settings"]["machs"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for op, r in g["results"].items():
            style = "-o" if op != "no_collision" else "--s"
            axes[0].loglog(machs, r["D2_amplitude_defect"], style, label=f"{op}")
            axes[1].loglog(machs, r["D3_decay_rate_defect"], style, label=f"{op}")
        for ax, title, ylab in (
            (axes[0], "D2: shear-stress amplitude defect", r"$\max_t|\hat A_{Ma}-\hat A_0|/\hat A_0$"),
            (axes[1], "D3: decay-rate (viscosity) defect", r"$|\lambda_{Ma}/\lambda_0-1|$"),
        ):
            ax.set_xlabel("Ma = U/c_s")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            ma_ref = [machs[0], machs[-1]]
            scale = 1e-2 / machs[-1] ** 2
            ax.plot(ma_ref, [scale * m**2 for m in ma_ref], "k:", lw=1,
                    label=r"$\propto Ma^2$")
        fig.suptitle("Galilean-invariance defect of an advected shear wave (D3Q27, τ=0.55)")
        fig.tight_layout()
        p = FIG / "cumulant_gali_defect.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p.relative_to(ROOT)))

    if stab_path.exists():
        s = json.loads(stab_path.read_text())
        rows = s["rows"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ops = sorted({r["operator"] for r in rows})
        marker = {"bgk": "o", "kbc": "s", "cumulant": "^", "cumulant_single_rate": "v"}
        for op in ops:
            sel = sorted([r for r in rows if r["operator"] == op], key=lambda r: r["re"])
            x = [r["re"] for r in sel]
            y = [r["steps_completed"] / r["steps_requested"] for r in sel]
            ax.semilogx(x, y, marker.get(op, "o") + "-", label=op)
        ax.set_xlabel("Re = U·2π/ν  (lattice units, 32³ TG)")
        ax.set_ylabel("fraction of requested steps survived")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Stability domain: Taylor–Green, τ sweep → Re sweep")
        fig.tight_layout()
        p = FIG / "cumulant_stability_domain.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p.relative_to(ROOT)))

    if pois_path.exists():
        pz = json.loads(pois_path.read_text())
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        any_row = next(iter(pz["rows"].values()))
        ax.plot(any_row["analytic"], label="analytic", color="k", lw=2)
        for op, r in pz["rows"].items():
            ax.plot(r["profile"], "--", label=f"{op} (err {r['rel_err']*100:.2f}%)")
        ax.set_xlabel("y (lattice nodes, walls at −½ and ny−½)")
        ax.set_ylabel("u_x")
        ax.legend(fontsize=8)
        ax.set_title("Poiseuille profile vs parabola (τ=0.55)")
        fig.tight_layout()
        p = FIG / "cumulant_poiseuille.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p.relative_to(ROOT)))

    if real_path.exists():
        rj = json.loads(real_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for op, r in rj["rows"].items():
            if not r["series"]:
                continue
            st = [row[0] for row in r["series"]]
            axes[0].semilogy(st, [max(row[1], 1e-18) for row in r["series"]], "-", label=op)
            axes[1].semilogy(st, [row[3] for row in r["series"]], "-", label=op)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel(r"$\lambda_{min}(\kappa_{ij})$")
        axes[0].set_title("realizability margin")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("mean kinetic energy")
        axes[1].set_title("kinetic energy decay")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIG / "cumulant_realizability.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p.relative_to(ROOT)))

    return _write("figures.json", {"meta": _meta("figures"), "figures": made})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        required=True,
        choices=["a1a2", "gali", "stability", "poiseuille", "realizability", "figures"],
    )
    ap.add_argument("--shard", default=None, help="i/n partition for the stability sweep")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma separated operator filter")
    args = ap.parse_args()

    only = tuple(args.only.split(",")) if args.only else None
    if args.threads:
        torch.set_num_threads(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))

    if args.task == "a1a2":
        task_a1a2()
    elif args.task == "gali":
        kw = {}
        if args.n:
            kw["n"] = args.n
        if args.nsteps:
            kw["nsteps"] = args.nsteps
        task_gali(**kw)
    elif args.task == "stability":
        shard = None
        if args.shard:
            i, m = args.shard.split("/")
            shard = (int(i), int(m))
        kw = {"shard": shard, "only": only}
        if args.n:
            kw["n"] = args.n
        task_stability(**kw)
    elif args.task == "poiseuille":
        kw = {"only": only}
        if args.nsteps:
            kw["nsteps"] = args.nsteps
        task_poiseuille(**kw)
    elif args.task == "realizability":
        kw = {"only": only}
        if args.n:
            kw["n"] = args.n
        if args.nsteps:
            kw["nsteps"] = args.nsteps
        task_realizability(**kw)
    elif args.task == "figures":
        task_figures()


if __name__ == "__main__":
    main()