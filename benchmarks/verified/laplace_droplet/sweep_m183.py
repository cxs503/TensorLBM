#!/usr/bin/env python
"""Independent single-radius driver for B23 laplace_droplet (mission m183).

Purpose
-------
Mission 25/183 "SDAA 验证案例扫荡" needs an *independently driven* re-run of
the verified case ``benchmarks/verified/laplace_droplet`` that
  * does not touch the case's business code (``run.py`` / ``result.json`` are
    read-only for us),
  * always leaves an artifact on disk (incremental flush + wall-clock budget +
    NaN guard), so a slow/killed job still produces evidence instead of the
    "no finish / no artifact" failure seen in mission 25,
  * runs one (dim, R) job per process so radii can be swept in parallel on a
    many-core CPU box (no GPU here: ``torch.cuda.is_available() == False``).

Physics is *not* re-implemented: the Shan-Chen collision, streaming and the
EOS/measurement helpers are imported from the case module itself
(``run.py``: ``init_rho``, ``eos_pressure``, ``fit_laplace``, ``_radius_field``)
and from ``tensorlbm`` (``collide_sc_single_component`` / ``stream`` /
``collide_sc_single_component_3d`` / ``stream3d``), exactly like the case does.
Only the time-stepping *driver* is ours (budget, flush, extra diagnostics):

  extra diagnostics added here (beyond the case driver)
  -----------------------------------------------------
  1. measurement-band sensitivity: the same final field is measured with the
     canonical bands (inside 0.5 R_eq, outside 1.5 R_eq) and with tightened
     bands (0.6 / 1.4) -> quantifies how much sigma depends on the pressure
     stencil (an uncertainty estimate the case does not report);
  2. radial density profile rho(r) export (interface width / coexistence
     check, used for the report figures);
  3. per-sample history export (Delta p(t), R_eq(t) convergence traces);
  4. field snapshot export (2D full field / 3D centre slice) for the report
     cloud maps;
  5. self-check: our band-parameterised measurement must reproduce the case's
     own ``measure()`` bit-for-bit at the canonical bands (guards against
     silent criterion drift).

Run: python sweep_m183.py --dim 2 --R 15 --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load_case_module():
    """Import the case's own run.py under a private module name (no edits)."""
    spec = importlib.util.spec_from_file_location("laplace_bench_case", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


CASE = _load_case_module()
sys.path.insert(0, str(HERE.parent))  # <repo>/benchmarks
from compile_route import normalize_compile_mode, route_step  # noqa: E402


def measure_bands(dim: int, f: torch.Tensor, R_guess: float, rr: torch.Tensor,
                  in_frac: float = 0.5, out_frac: float = 1.5) -> dict:
    """Band-parameterised copy of the case's ``measure()`` (canonical: 0.5/1.5).

    Identical iteration / threshold / equivalent-radius logic; only the two band
    fractions are free parameters so the stencil sensitivity can be probed.
    """
    if dim == 2:
        rho, ux, uy = CASE.macroscopic(f)
        umag = torch.sqrt(ux**2 + uy**2)
    else:
        rho, ux, uy, uz = CASE.macroscopic3d(f)
        umag = torch.sqrt(ux**2 + uy**2 + uz**2)
    p = CASE.eos_pressure(rho)
    r_eq = R_guess
    for _ in range(3):
        inside = rr <= r_eq * in_frac
        outside = rr >= r_eq * out_frac
        rho_in = float(rho[inside].mean().item())
        rho_out = float(rho[outside].mean().item())
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        r_eq_new = (math.sqrt(n_liq / math.pi) if dim == 2
                    else (3.0 * n_liq / (4.0 * math.pi)) ** (1.0 / 3.0))
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    inside = rr <= r_eq * in_frac
    outside = rr >= r_eq * out_frac
    return {
        "p_in": float(p[inside].mean().item()),
        "p_out": float(p[outside].mean().item()),
        "dp": float(p[inside].mean().item() - p[outside].mean().item()),
        "rho_in": float(rho[inside].mean().item()),
        "rho_out": float(rho[outside].mean().item()),
        "R_eq": r_eq,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def radial_profile(dim: int, f: torch.Tensor, rr: torch.Tensor, dr: float = 0.5) -> dict:
    """Shell-averaged rho(r) (interface width / bulk density diagnostics)."""
    rho = f.sum(dim=0)
    r_max = float(rr.max().item())
    edges = np.arange(0.0, r_max + dr, dr)
    idx = torch.clamp((rr / dr).long(), max=len(edges) - 2)
    n_bins = len(edges) - 1
    sums = torch.zeros(n_bins, dtype=torch.float64)
    counts = torch.zeros(n_bins, dtype=torch.float64)
    sums.index_add_(0, idx.reshape(-1), rho.reshape(-1).double())
    counts.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.float64))
    cnt = counts.clamp(min=1)
    return {
        "r": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
        "rho": (sums / cnt).tolist(),
        "n": counts.tolist(),
        "dr": dr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 3], required=True)
    ap.add_argument("--R", type=float, required=True)
    ap.add_argument("--domain-factor", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--min-steps", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--conv-dp", type=float, default=2e-4)
    ap.add_argument("--conv-r", type=float, default=2e-3)
    ap.add_argument("--budget-s", type=float, default=1500.0,
                    help="wall-clock budget for the time loop (partial result is still written)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--compile-mode", default="eager",
                    choices=["eager", "default", "max-autotune-no-cudagraphs"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    mode = normalize_compile_mode(args.compile_mode)
    dim, R = args.dim, args.R
    L = int(round(args.domain_factor * R))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    rho0 = CASE.init_rho(dim, L, R, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = (CASE.equilibrium(rho0, zero, zero) if dim == 2
         else CASE.equilibrium3d(rho0, zero, zero, zero))
    rr = CASE._radius_field(dim, L, device)

    def _step(state):
        if dim == 2:
            return CASE.stream(CASE.collide_sc_single_component(
                state, G=CASE.G_LIB, tau=CASE.TAU, psi_fn=CASE.psi_exp))
        return CASE.stream3d(CASE.collide_sc_single_component_3d(
            state, G=CASE.G_LIB, tau=CASE.TAU, psi_fn=CASE.psi_exp))

    step_fn = route_step(_step, mode, name=f"laplace_droplet[{dim}d]R{int(R)}")

    meta = {
        "case": "benchmarks/verified/laplace_droplet",
        "dim": dim, "R_init": R, "L": L, "domain_factor": args.domain_factor,
        "G_lib": CASE.G_LIB, "G_eff": CASE.G_EFF, "tau": CASE.TAU,
        "rho_l_init": CASE.RHO_L, "rho_v_init": CASE.RHO_V, "W_int": CASE.W_INT,
        "compile_mode_requested": args.compile_mode,
        "max_steps": args.max_steps, "min_steps": args.min_steps,
        "sample_interval": args.sample_interval,
        "conv_dp": args.conv_dp, "conv_r": args.conv_r,
        "budget_s": args.budget_s, "threads": args.threads,
        "device": "cpu", "torch": torch.__version__,
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    hist: list[dict] = []
    stage = "init"
    t0 = time.perf_counter()
    step = 0
    converged = False
    selfcheck_err = None
    try:
        for step in range(1, args.max_steps + 1):
            stage = "step"
            f = step_fn(f)
            if step % args.sample_interval == 0:
                stage = "measure"
                rho_cur = f.sum(dim=0)
                if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                    raise RuntimeError(f"NaN/negative rho at step {step}")
                m = measure_bands(dim, f, R, rr)
                if selfcheck_err is None:  # cross-check vs the case's own measure()
                    ref = CASE.measure(dim, f, R, rr, device)
                    selfcheck_err = max(abs(m[k] - ref[k]) / max(abs(ref[k]), 1e-12)
                                        for k in ("dp", "R_eq", "p_in", "p_out"))
                m["step"] = step
                m["t_s"] = time.perf_counter() - t0
                hist.append(m)
                stage = "flush"
                (out / "history.json").write_text(
                    json.dumps({"meta": meta, "history": hist}, indent=1) + "\n",
                    encoding="utf-8")
                if len(hist) >= 2:
                    d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                    d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                    if d_dp < args.conv_dp and d_r < args.conv_r and step >= args.min_steps:
                        converged = True
                        stage = "converged"
                        break
            if time.perf_counter() - t0 > args.budget_s:
                stage = "budget_exhausted"
                break
    except Exception as exc:  # noqa: BLE001 - record, never lose the job
        stage = f"error:{type(exc).__name__}"
        (out / "error.json").write_text(
            json.dumps({"stage": stage, "msg": str(exc), "step": step,
                        "meta": meta}, indent=2) + "\n", encoding="utf-8")

    dt = time.perf_counter() - t0
    tail = hist[-3:] if hist else []
    final = {
        "step": int(step), "stage": stage, "converged": bool(converged),
        "elapsed_s": dt, "ms_per_step": (dt / max(step, 1)) * 1000.0,
        "n_samples": len(hist),
    }
    if tail:
        for k in ("p_in", "p_out", "dp", "rho_in", "rho_out", "R_eq", "max_u"):
            final[k] = float(np.mean([s[k] for s in tail]))
        final["dp_std"] = float(np.std([s["dp"] for s in tail]))
        final["R_eq_std"] = float(np.std([s["R_eq"] for s in tail]))
        final["mass_drift"] = abs(float(tail[-1]["mass"]) - mass0) / mass0
        # band sensitivity on the final field
        band_a = measure_bands(dim, f, R, rr, 0.5, 1.5)
        band_b = measure_bands(dim, f, R, rr, 0.6, 1.4)
        final["band_check"] = {
            "dp_05_15": band_a["dp"], "dp_06_14": band_b["dp"],
            "dp_rel_diff_pct": abs(band_b["dp"] - band_a["dp"]) / abs(band_a["dp"]) * 100.0,
            "R_eq_05_15": band_a["R_eq"], "R_eq_06_14": band_b["R_eq"],
            "R_eq_rel_diff_pct": abs(band_b["R_eq"] - band_a["R_eq"]) / band_a["R_eq"] * 100.0,
        }
        sigma_i = final["dp"] * final["R_eq"] if dim == 2 else final["dp"] * final["R_eq"] / 2.0
        final["sigma_i_this_run"] = sigma_i
        prof = radial_profile(dim, f, rr)
        (out / "profile.json").write_text(json.dumps(prof) + "\n", encoding="utf-8")
        if dim == 2:
            snap = f.sum(dim=0).cpu().numpy()
        else:
            snap = f.sum(dim=0)[L // 2].cpu().numpy()  # centre x-y slice at z=L/2
        np.save(out / "snapshot.npy", snap)
    final["selfcheck_vs_case_measure_relerr"] = selfcheck_err
    row = {
        "R_init": R, "R_eq": final.get("R_eq"), "p_in": final.get("p_in"),
        "p_out": final.get("p_out"), "dp": final.get("dp"),
        "rho_in": final.get("rho_in"), "rho_out": final.get("rho_out"),
        "max_u": final.get("max_u"), "step": final["step"],
        "mass_drift": final.get("mass_drift"), "converged": final["converged"],
    }
    result = {"meta": meta, "final": final, "row": row,
              "history": [{k: s[k] for k in ("step", "dp", "R_eq", "rho_in", "rho_out",
                                             "p_in", "p_out", "max_u", "t_s")}
                          for s in hist]}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[m183] dim={dim} R={R} L={L} stage={stage} converged={converged} "
          f"steps={step} dt={dt:.1f}s dp={final.get('dp')} R_eq={final.get('R_eq')} "
          f"sigma_i={final.get('sigma_i_this_run')}")


if __name__ == "__main__":
    main()