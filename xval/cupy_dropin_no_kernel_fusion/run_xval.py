#!/usr/bin/env python3
"""XVAL entry point: TensorLBM SOY free-energy LBM vs ``ebinan92/Fingering_dynamics``.

Anchor: ``XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION``.

What is cross-validated
-----------------------
The OSS project runs its D2Q9 free-energy LBM on GPU by *copying* the NumPy
script and swapping ``import numpy as np`` for ``import cupy as cp`` -- the
"cupy drop-in, no kernel fusion" practice.  This runner measures whether that
practice transfers to TensorLBM: the same algorithm is executed through two
array backends (NumPy and PyTorch eager) of one implementation, and the results
are compared against (a) the *executed* OSS reference code and (b) the analytic
Young-Laplace law ``dp = sigma/R``.

Arms
----
``tensorlbm_torch``  TensorLBM implementation, PyTorch eager tensors (drop-in GPU-ready)
``tensorlbm_numpy``  TensorLBM implementation, NumPy arrays (array-API swap only)
``oss_exec``         the OSS project's own ``class Compute`` + ``stream``, executed
                     read-only from the OSS checkout (see ``oss_adapter.py``)

Acceptance (predeclared in ``predeclared.json``, owner standard): relative error
of the measured Laplace jump vs ``sigma/R`` must be ``< 15 %``; the two TensorLBM
backends must additionally agree to ``< 1e-6`` relative (drop-in equivalence).

Usage
-----
    python xval/cupy_dropin_no_kernel_fusion/run_xval.py                # full study
    python xval/cupy_dropin_no_kernel_fusion/run_xval.py --quick        # smoke (~1 min)
    python xval/cupy_dropin_no_kernel_fusion/run_xval.py --no-oss       # skip OSS arm
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import platform as _pf
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from tensorlbm.multiphase_soy_dropin import (  # noqa: E402
    SOYConfig,
    SOYFreeEnergyD2Q9,
    laplace_error_pct,
)

THRESHOLD_PCT = 15.0      # owner precision standard
AGREEMENT_PCT = 1e-4      # two TensorLBM backends must agree to 1e-4 % (1e-6 rel)


# ---------------------------------------------------------------------------
def run_backend_arm(backend: str, cfg_kwargs: dict, steps: int, probe_every: int = 0,
                    device: str = "cpu") -> dict:
    """Run one TensorLBM arm (backend selected by ``cfg_kwargs['backend']``)."""
    cfg = SOYConfig(**{**cfg_kwargs, "backend": backend, "device": device})
    solver = SOYFreeEnergyD2Q9(cfg)
    t0 = time.perf_counter()
    solver.init_droplet()
    history = []
    remaining = steps
    while remaining > 0:
        chunk = min(probe_every, remaining) if probe_every else remaining
        solver.run(chunk)
        remaining -= chunk
        if probe_every:
            m = solver.measure()
            history.append({"step": solver.steps, "dp": m["dp"], "u_max": m["u_max"],
                            "err_pct": laplace_error_pct(m["dp"], cfg.sigma, cfg.radius)})
    out = solver.measure()
    wall = time.perf_counter() - t0
    out.update({
        "arm": backend,
        "history": history,
        "wall_time_s": wall,
        "ops_per_step": solver.ops_per_step(),
        "config": cfg.as_dict(),
        "err_pct": laplace_error_pct(out["dp"], cfg.sigma, cfg.radius),
        "pass": abs(laplace_error_pct(out["dp"], cfg.sigma, cfg.radius)) < THRESHOLD_PCT,
    })
    # field snapshots for the report (down-sampled, JSON-safe)
    out["fields"] = {
        "psi": solver.b.to_numpy(solver.psi).round(6).tolist(),
        "p": solver.b.to_numpy(solver.p).round(12).tolist(),
        "grid": [cfg.ny, cfg.nx],
    }
    return out


def run_oss_arm(cfg_kwargs: dict, steps: int, probe_every: int = 0) -> dict:
    from oss_adapter import OSSReference

    ref = OSSReference(**{k: v for k, v in cfg_kwargs.items()
                          if k in {"nx", "ny", "radius", "sigma", "xi", "rho0",
                                   "eta_n", "viscosity_ratio", "tau_g", "gamma"}})
    prov = ref.provenance()
    if not ref.available:
        return {"arm": "oss_exec", "available": False, "reason": ref.reason,
                "provenance": prov, "pass": None}
    t0 = time.perf_counter()
    ref.setup()
    out = ref.run_and_measure(steps, probe_every=probe_every)
    wall = time.perf_counter() - t0
    for h in out.get("history", []):
        h["err_pct"] = laplace_error_pct(h["dp"], cfg_kwargs["sigma"], cfg_kwargs["radius"])
    out.update({
        "arm": "oss_exec",
        "available": True,
        "provenance": prov,
        "wall_time_s": wall,
        "err_pct": laplace_error_pct(out["dp"], cfg_kwargs["sigma"], cfg_kwargs["radius"]),
        "pass": abs(laplace_error_pct(out["dp"], cfg_kwargs["sigma"], cfg_kwargs["radius"]))
        < THRESHOLD_PCT,
    })
    return out


def history_of(arm: dict) -> list:
    return arm.get("history", [])


# ---------------------------------------------------------------------------
def figures(results: dict, figdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    made = []
    main = results["main"]
    ref = main["reference_dp"]

    # 1) dp(t) convergence with the analytic line
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, arm in main["arms"].items():
        hist = history_of(arm)
        if not hist:
            continue
        ax.plot([h["step"] for h in hist], [h["dp"] for h in hist], marker="o",
                ms=3, label=f"{name} (final {100 * arm['err_pct']:+.2f}%)")
    ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"analytic $\sigma/R$")
    ax.axhspan(ref * (1 - 0.15), ref * (1 + 0.15), color="gray", alpha=0.15,
               label="±15% acceptance band")
    ax.set_xlabel("lattice step")
    ax.set_ylabel(r"$\Delta p$ (lattice units)")
    ax.set_title(r"Laplace pressure convergence ($R=15$, $\sigma=0.01$, $\xi=3$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "fig1_laplace_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # 2) field snapshots (order parameter + pressure) of the torch arm
    arm = results["main"]["arms"].get("tensorlbm_torch") or next(
        iter(results["main"]["arms"].values()))
    if "fields" in arm:
        psi = np.asarray(arm["fields"]["psi"])
        pr = np.asarray(arm["fields"]["p"])
        fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
        im0 = axs[0].imshow(psi, cmap="RdBu_r", origin="lower")
        axs[0].set_title(r"order parameter $\psi$ (t = %d)" % arm["steps"])
        fig.colorbar(im0, ax=axs[0], shrink=0.85)
        im1 = axs[1].imshow(pr, cmap="viridis", origin="lower")
        axs[1].set_title("pressure $p=\\rho/3+\\psi\\mu$")
        fig.colorbar(im1, ax=axs[1], shrink=0.85)
        for a in axs:
            a.set_xlabel("x"); a.set_ylabel("y")
        fig.suptitle("TensorLBM SOY free-energy field, static droplet (torch backend)")
        fig.tight_layout()
        p = figdir / "fig2_fields.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 3) error vs droplet radius (diffuse-interface convergence)
    conv = results.get("convergence") or []
    if conv:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        R = [c["radius"] for c in conv]
        e = [c["err_pct"] for c in conv]
        ax.plot(R, e, "o-", label="TensorLBM (numpy backend)")
        ax.plot(R, [c["err_pct_oss"] for c in conv], "s--", label="OSS exec")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(-THRESHOLD_PCT, color="r", ls=":", label="acceptance -15%")
        ax.set_xlabel("initial droplet radius R (lattice units)")
        ax.set_ylabel(r"error vs $\sigma/R$ (%)")
        ax.set_title(r"Diffuse-interface error vs $R$ ($\xi=3$ fixed)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "fig3_error_vs_radius.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 4) sensitivity sweeps
    sens = results.get("sensitivity") or []
    if sens:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        numeric = {}
        for s in sens:
            if isinstance(s["value"], (int, float)):
                numeric.setdefault(s["param"], []).append(s)
        for param in sorted(numeric):
            rows = sorted(numeric[param], key=lambda r: r["value"])
            ax.plot([r["value"] for r in rows], [r["err_pct"] for r in rows],
                    "o-", label=param)
        ax.axhline(THRESHOLD_PCT, color="r", ls=":", label="+15%")
        ax.axhline(-THRESHOLD_PCT, color="r", ls=":")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("parameter value (or index for categorical sweeps)")
        ax.set_ylabel(r"error vs $\sigma/R$ (%)")
        ax.set_title("Parameter sensitivity of the Laplace jump")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "fig4_sensitivity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 5) cost of "no kernel fusion": ops/step and wall time per step
    cost = results.get("cost") or []
    if cost:
        names = [c["arm"] for c in cost]
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
        axs[0].bar(names, [c["ms_per_step"] for c in cost], color=["#4C72B0", "#DD8452"])
        axs[0].set_ylabel("wall time per step (ms)")
        axs[0].set_title("no-fusion cost: array-API overhead")
        axs[1].bar(names, [c["ops_per_step"] for c in cost], color=["#4C72B0", "#DD8452"])
        axs[1].set_ylabel("array-op calls per step")
        axs[1].set_title("array ops per time step (unfused)")
        for a in axs:
            a.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = figdir / "fig5_cost.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    return made


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000, help="time steps of the main case")
    ap.add_argument("--probe-every", type=int, default=100, help="dp sampling interval")
    ap.add_argument("--out-dir", default=str(HERE / "artifacts"))
    ap.add_argument("--fig-dir", default=str(HERE / "figures"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--quick", action="store_true", help="short smoke run")
    ap.add_argument("--no-oss", action="store_true", help="skip the OSS reference arm")
    ap.add_argument("--no-torch", action="store_true", help="skip the torch arm")
    args = ap.parse_args(argv)

    steps = 200 if args.quick else args.steps
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predeclared = json.loads((HERE / "predeclared.json").read_text(encoding="utf-8"))
    base = dict(nx=60, ny=60, radius=15.0, sigma=0.01, xi=3.0, rho0=1.0,
                tau_g=1.0 / (3.0 - math.sqrt(3.0)), gamma=0.1,
                eta_n=1.0 / 6.0, viscosity_ratio=1.0, pressure_form="oss",
                init_profile="equilibrium")

    results: dict = {
        "anchor": "XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "platform": platform.platform(), "machine": _pf.machine()},
        "predeclared": predeclared,
        "config_base": base,
        "steps": steps,
    }
    try:
        import torch
        results["host"]["torch"] = torch.__version__
        results["host"]["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:                                    # pragma: no cover
        results["host"]["torch"] = None

    # ---- main case: three arms on identical physics ----
    t_all = time.perf_counter()
    arms: dict[str, dict] = {}
    arms["tensorlbm_numpy"] = run_backend_arm("numpy", base, steps, args.probe_every)
    if not args.no_torch:
        arms["tensorlbm_torch"] = run_backend_arm("torch", base, steps, args.probe_every)
    if not args.no_oss:
        arms["oss_exec"] = run_oss_arm(base, steps, args.probe_every)

    ref_dp = base["sigma"] / base["radius"]
    main_block = {
        "reference_dp": ref_dp,
        "reference": "Young-Laplace dp = sigma/R (analytic, sigma = model sigma from kappa = 0.75 sigma xi)",
        "arms": arms,
    }
    # drop-in equivalence between the two TensorLBM backends
    if "tensorlbm_torch" in arms:
        a, b = arms["tensorlbm_numpy"], arms["tensorlbm_torch"]
        rel = abs(a["dp"] - b["dp"]) / abs(a["dp"])
        main_block["backend_agreement"] = {
            "dp_numpy": a["dp"], "dp_torch": b["dp"],
            "rel_diff": rel, "rel_diff_pct": 100 * rel,
            "pass": 100 * rel < AGREEMENT_PCT * 100,
            "tolerance_rel": 1e-6,
        }
    if "oss_exec" in arms and arms["oss_exec"].get("available"):
        a = arms["tensorlbm_numpy"]
        rel = abs(a["dp"] - arms["oss_exec"]["dp"]) / abs(arms["oss_exec"]["dp"])
        main_block["oss_agreement"] = {"dp_oss": arms["oss_exec"]["dp"],
                                       "dp_tensorlbm": a["dp"],
                                       "rel_diff": rel, "rel_diff_pct": 100 * rel}
    results["main"] = main_block

    # ---- convergence vs droplet radius (diffuse-interface error) ----
    if not args.quick:
        conv = []
        for R in predeclared["convergence"]["radii"]:
            kw = {**base, "radius": float(R)}
            arm = run_backend_arm("numpy", kw, steps, probe_every=0)
            row = {"radius": float(R), "dp": arm["dp"], "err_pct": arm["err_pct"],
                   "u_max": arm["u_max"], "R_over_xi": float(R) / base["xi"],
                   "steps": steps}
            if not args.no_oss:
                oss = run_oss_arm(kw, steps, probe_every=0)
                row["dp_oss"] = oss.get("dp")
                row["err_pct_oss"] = oss.get("err_pct")
            conv.append(row)
            print(f"[conv] R={R}: err={arm['err_pct']:+.3f}% "
                  f"(oss {row.get('err_pct_oss')})", flush=True)
        results["convergence"] = conv

    # ---- parameter sensitivity (>=2 parameters, predeclared) ----
    sens = []
    for sweep in predeclared["sensitivity"]:
        param, values = sweep["param"], sweep["values"]
        for v in values:
            kw = {**base, param: v}
            arm = run_backend_arm("numpy", kw, steps, probe_every=0)
            sens.append({"param": param, "value": v, "dp": arm["dp"],
                         "err_pct": arm["err_pct"], "u_max": arm["u_max"],
                         "sigma_model": kw["sigma"], "steps": steps})
            print(f"[sens] {param}={v}: err={arm['err_pct']:+.3f}%", flush=True)
    results["sensitivity"] = sens

    # ---- cost of the no-fusion practice ----
    cost = []
    for name in [k for k in ("tensorlbm_numpy", "tensorlbm_torch") if k in arms]:
        arm = arms[name]
        ms = 1000.0 * arm["wall_time_s"] / max(arm["steps"], 1)
        cost.append({"arm": name, "ms_per_step": ms,
                     "ops_per_step": arm["ops_per_step"],
                     "steps": arm["steps"]})
    results["cost"] = cost
    results["total_wall_time_s"] = time.perf_counter() - t_all

    # ---- acceptance summary ----
    checks = {
        "laplace_tensorlbm_numpy": {"err_pct": arms["tensorlbm_numpy"]["err_pct"],
                                    "tol_pct": THRESHOLD_PCT,
                                    "pass": abs(arms["tensorlbm_numpy"]["err_pct"]) < THRESHOLD_PCT},
    }
    if "tensorlbm_torch" in arms:
        checks["laplace_tensorlbm_torch"] = {
            "err_pct": arms["tensorlbm_torch"]["err_pct"], "tol_pct": THRESHOLD_PCT,
            "pass": abs(arms["tensorlbm_torch"]["err_pct"]) < THRESHOLD_PCT}
    if "oss_exec" in arms:
        checks["laplace_oss_exec"] = {
            "err_pct": arms["oss_exec"].get("err_pct"), "tol_pct": THRESHOLD_PCT,
            "pass": arms["oss_exec"].get("pass")}
    if "backend_agreement" in main_block:
        checks["backend_dropin_equivalence"] = main_block["backend_agreement"]
    results["checks"] = checks
    results["all_pass"] = all(bool(c.get("pass")) for c in checks.values())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / (f"run_{args.tag}_{stamp}.json" if args.tag != "main"
                           else "run_main.json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # dp(t) CSV of the main case
    csv_path = out_dir / "dp_history.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("arm,step,dp,err_pct,u_max\n")
        for name, arm in arms.items():
            for h in history_of(arm):
                fh.write(f"{name},{h['step']},{h['dp']:.12e},{h['err_pct']:+.6f},{h['u_max']:.6e}\n")

    figs = figures(results, fig_dir)
    results["figures"] = figs
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "checks": results["checks"], "all_pass": results["all_pass"],
                      "total_wall_time_s": results["total_wall_time_s"]}, indent=2))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())