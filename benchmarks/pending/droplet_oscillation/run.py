#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B24 2D droplet oscillation benchmark — Rayleigh frequency verification (SCMP).

STATUS 2026-08-19: PENDING (not verified). Observed m=2 frequency is 6-8%
below the Rayleigh rho_l formula on R=20/30/40 (criterion 3%) because the
SC94 model has an intrinsic damping rate gamma ~= 0.72/R^2 (~4.5x the bulk
viscosity estimate, tau-independent), which lowers the observable frequency
by ~gamma^2/(2 omega^2). Restoring the undamped eigenfrequency
omega_0 = sqrt(omega_d^2 + gamma^2) (gamma measured from the same fit)
matches theory to ~1-2%. Full diagnosis: benchmarks/pending/droplet_oscillation/README.md.

Model: SC94 psi_exp (psi = 1 - exp(-rho)), physical coupling G_eff = -5.0,
       tau = 1.0, float32. Library is called with G = +5.0 (backward-gather
       convention: G>0 attractive) — the verified laplace_droplet config.

Physics: 2D liquid droplet (initial mean radius R0) in vapour, initial m=2
shape perturbation  R(theta) = R0*(1 + eps*cos(2*theta)),  eps = 0.05.
The droplet performs free shape oscillations. Linear theory (Rayleigh 1880,
2D, inviscid, both fluids):  omega^2 = m*(m^2-1)*sigma/((rho_l+rho_v)*R^3),
m = 2  ->  omega^2 = 6*sigma/((rho_l+rho_v)*R^3).  The task-specified
rho-only form  omega^2 = 6*sigma/(rho_l*R^3)  (vapour neglected) is also
reported as the primary criterion; the rho_l+rho_v form is the exact 2D
linear result (derivation: pressure balance of r^m / r^-m potentials).

sigma = sigma_eff measured in the verified laplace_droplet benchmark
(2D: 0.056112); rho_l = 1.957, rho_v = 0.1596 (discrete coexistence);
R = measured equilibrium radius R_eq (mid-interface, self-consistent).

Signal: mass-weighted quadrupole Q(t) = <x^2-y^2>_rho (liquid core) is
proportional to the m=2 deformation amplitude a(t) ~ eps*cos(omega*t).
Two estimators: (1) FFT peak of Q(t); (2) damped-sinusoid least-squares
fit  Q = A*exp(-g*t)*cos(w*t+p)+c  (grid search in w,g + linear solve),
which also yields the damping rate g.

Pass criteria (real simulation only, no extrapolation):
  * |omega_sim/omega_theory - 1| <= 3% on EVERY grid (R=20 and R=30);
  * convergence: error on the finer grid (R=30) <= error on the coarser (R=20).

Usage: python run.py [--radii 20,30] [--steps 30000] [--sample 50]
                     [--eps 0.05] [--out DIR] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np
import torch

torch.set_num_threads(32)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0
G_LIB = 5.0   # library argument (sign-flipped convention, G>0 attractive)
G_EFF = -5.0  # physical standard-convention SC94 coupling
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596  # discrete coexistence (measured, laplace_droplet)
SIGMA_EFF = 0.056112          # measured 2D surface tension (laplace_droplet)
W_INT = 4.0


def init_ellipse_rho(
    L: int, R0: float, eps: float, device: torch.device
) -> torch.Tensor:
    """Elliptically perturbed droplet: R(theta) = R0*(1 + eps*cos(2*theta)).

    tanh interface profile (width W_INT=4), liquid inside, vapour outside,
    periodic domain L x L.  Perturbed-interface initial field is NOT provided
    by the common modules (gap documented in /tmp/droplet_osc_gap.md).
    """
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx = xx - L / 2.0
    dy = yy - L / 2.0
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
    r_surf = R0 * (1.0 + eps * torch.cos(2.0 * theta))
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh((r_surf - r) / W_INT))
    return rho.clamp(min=1e-3)


def _interface_extent(rho: torch.Tensor, mid: float, L: int, axis: int) -> float:
    """Sub-grid interface position (mid-level crossing) along a centre line."""
    if axis == 0:  # along x at y = L/2
        line = rho[L // 2, :]
    else:          # along y at x = L/2
        line = rho[:, L // 2]
    cross = (line[:-1] - mid) * (line[1:] - mid) < 0
    idx = torch.nonzero(cross).flatten().tolist()
    if len(idx) < 2:
        return float("nan")
    xint = []
    for i in idx:
        r0 = float(line[i])
        r1 = float(line[i + 1])
        if abs(r1 - r0) < 1e-12:
            xint.append(float(i))
        else:
            xint.append(i + (mid - r0) / (r1 - r0))
    lo = min(xint)
    hi = max(xint)
    return float(hi - lo) / 2.0


def measure_osc(
    f: torch.Tensor, R_guess: float, xx: torch.Tensor, yy: torch.Tensor,
    rr: torch.Tensor, L: int,
) -> dict:
    """Self-consistent liquid-core measurement -> quadrupole + interface radii."""
    rho, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux * ux + uy * uy)
    r_eq = R_guess
    rho_in = rho_out = float("nan")
    mid = 0.5 * (RHO_L + RHO_V)
    for _ in range(3):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
        rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        r_eq_new = math.sqrt(n_liq / math.pi)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    mask = rho > mid
    rho_liq = rho[mask]
    if rho_liq.numel() == 0:
        return {"error": "no liquid core found"}
    m_liq = float(rho_liq.sum().item())
    q = float((rho_liq * (xx[mask] ** 2 - yy[mask] ** 2)).sum().item()) / m_liq
    rx = _interface_extent(rho, mid, L, 0)
    ry = _interface_extent(rho, mid, L, 1)
    return {
        "Q": q,
        "R_eq": r_eq,
        "R_x": rx,
        "R_y": ry,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def fft_peak(y: np.ndarray, dt: float) -> dict:
    """Dominant spectral peak of a real time series (units of 1/step)."""
    y = y - y.mean()
    n = len(y)
    X = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(X)
    kmin = 3
    k = kmin + int(np.argmax(mag[kmin:]))
    if 1 <= k < len(mag) - 1:
        denom = mag[k - 1] - 2.0 * mag[k] + mag[k + 1]
        delta = 0.5 * (mag[k - 1] - mag[k + 1]) / denom if abs(denom) > 1e-30 else 0.0
    else:
        delta = 0.0
    f_peak = freqs[k] + delta * freqs[1]
    return {
        "f_peak": float(f_peak),
        "omega_sim": float(2.0 * math.pi * f_peak),
        "peak_bin": int(k),
        "n_points": int(n),
    }


def damped_sine_fit(t: np.ndarray, y: np.ndarray, w0: float) -> dict:
    """Fit y = A*exp(-g*t)*cos(w*t+p)+c by grid search (w,g) + linear solve."""
    t = t - t[0]
    yc = y - y.mean()
    best = None
    for w in np.linspace(0.5 * w0, 1.6 * w0, 111):
        for g in np.linspace(0.0, 4e-3, 41):
            e = np.exp(-g * t)
            A = np.column_stack([e * np.cos(w * t), e * np.sin(w * t), np.ones_like(t)])
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            rss = float(np.sum((A @ coef - y) ** 2))
            if best is None or rss < best[0]:
                best = (rss, w, g, coef, res)
    rss, w, g, coef, res = best
    # refine w,g by local parabolic minimisation of rss over the grid neighbours
    return {
        "omega_fit": float(w),
        "gamma_fit": float(g),
        "amp_fit": float(math.hypot(coef[0], coef[1])),
        "phase_fit": float(math.atan2(-coef[1], coef[0])),
        "offset_fit": float(coef[2]),
        "rss": float(rss),
    }


def run_case(
    R0: float,
    L: int,
    eps: float,
    device: torch.device,
    max_steps: int,
    sample_interval: int,
) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    rho0 = init_ellipse_rho(L, R0, eps, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx - L / 2.0
    yy = yy - L / 2.0
    rr = torch.sqrt(xx * xx + yy * yy)

    hist: list[dict] = []
    for step in range(1, max_steps + 1):
        f = collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp)
        f = stream(f)
        if step % sample_interval == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"R={R0}: NaN/negative rho at step {step}")
            m = measure_osc(f, R0, xx, yy, rr, L)
            m["step"] = step
            hist.append(m)
    dt_s = time.perf_counter() - t0

    # ---- analysis ----
    # Signal-dominated window: skip acoustic/relaxation transient (~10 samples),
    # keep ~4 theoretical periods (damping gamma~0.7/R^2 kills the signal later;
    # late windows are noise-dominated and must NOT be used).
    hist_np = {k: np.array([h[k] for h in hist]) for k in ("Q", "R_eq", "R_x", "R_y")}
    steps = np.array([h["step"] for h in hist], dtype=float)
    dt = float(sample_interval)
    R_eq_mean = float(hist_np["R_eq"][-100:].mean())
    w_theo_rho = math.sqrt(6.0 * SIGMA_EFF / (RHO_L * R_eq_mean**3))
    T = 2.0 * math.pi / w_theo_rho
    nwin = int(4.0 * T / dt)
    start = 10
    end = min(len(hist), start + nwin)
    R_eq_mean = float(hist_np["R_eq"][start:].mean())

    omega_theory_rhov = math.sqrt(6.0 * SIGMA_EFF / ((RHO_L + RHO_V) * R_eq_mean**3))
    omega_theory_rho = math.sqrt(6.0 * SIGMA_EFF / (RHO_L * R_eq_mean**3))

    fft_Q = fft_peak(hist_np["Q"][start:end], dt)
    fit_Q = damped_sine_fit(steps[start:end], hist_np["Q"][start:end],
                            omega_theory_rho)
    omega0 = math.sqrt(fit_Q["omega_fit"] ** 2 + fit_Q["gamma_fit"] ** 2)

    nu = (TAU - 0.5) / 3.0
    gamma_est = nu / R_eq_mean**2
    final = {
        "R_init": R0,
        "L": L,
        "eps": eps,
        "max_steps": int(max_steps),
        "sample_interval": int(sample_interval),
        "n_samples": len(hist),
        "fit_window_steps": [int(steps[start]), int(steps[end - 1])],
        "R_eq_mean": R_eq_mean,
        "R_eq_std": float(hist_np["R_eq"][start:].std()),
        "rho_in_mean": float(np.mean([h["rho_in"] for h in hist[start:end]])),
        "rho_out_mean": float(np.mean([h["rho_out"] for h in hist[start:end]])),
        "max_u_mean": float(np.mean([h["max_u"] for h in hist[start:end]])),
        "mass_drift": abs(float(hist[-1]["mass"]) - mass0) / mass0,
        "dt_s": dt_s,
        # theories (R = R_eq_mean)
        "omega_theory_rho_l": omega_theory_rho,
        "omega_theory_rho_lv": omega_theory_rhov,
        "period_theory_rho_l": 2.0 * math.pi / omega_theory_rho,
        "period_theory_rho_lv": 2.0 * math.pi / omega_theory_rhov,
        # estimators (observed damped frequency, damping rate, undamped eigenfrequency)
        "omega_sim_fft": fft_Q["omega_sim"],
        "omega_sim_fit": fit_Q["omega_fit"],
        "gamma_fit": fit_Q["gamma_fit"],
        "omega_0_fit": omega0,
        "gamma_est_nu_R2": gamma_est,
        "fft_peak_bin": fft_Q["peak_bin"],
        # errors: observed (task criterion) and undamped-restored, both theory forms
        "err_pct_fit_vs_rho_l": (fit_Q["omega_fit"] - omega_theory_rho) / omega_theory_rho * 100.0,
        "err_pct_fft_vs_rho_l": (fft_Q["omega_sim"] - omega_theory_rho) / omega_theory_rho * 100.0,
        "err_pct_fit_vs_rho_lv": (fit_Q["omega_fit"] - omega_theory_rhov) / omega_theory_rhov * 100.0,
        "err_pct_omega0_vs_rho_l": (omega0 - omega_theory_rho) / omega_theory_rho * 100.0,
        "err_pct_omega0_vs_rho_lv": (omega0 - omega_theory_rhov) / omega_theory_rhov * 100.0,
    }
    return final, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="20,30")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda unavailable, fallback cpu")
        device = torch.device("cpu")

    results: dict = {
        "benchmark": "droplet_oscillation",
        "model": (
            "SCMP SC94 psi_exp (psi=1-exp(-rho)), physical G_eff=-5.0, tau=1.0; "
            "library called with G=+5.0 (backward-gather convention); "
            "identical to verified laplace_droplet"
        ),
        "physics": (
            "2D droplet, m=2 shape oscillation, R(theta)=R0*(1+eps*cos(2*theta)), "
            "eps=0.05; omega^2 = m(m^2-1)*sigma/(rho*R^3), m=2 -> 6*sigma/(rho*R^3); "
            "rho = rho_l (task form, vapour neglected) or rho_l+rho_v (exact 2D)"
        ),
        "constants": {
            "sigma_eff": SIGMA_EFF,
            "sigma_eff_source": "laplace_droplet verified 2D measurement",
            "rho_l": RHO_L,
            "rho_v": RHO_V,
            "tau": TAU,
        },
        "criteria": {
            "pass": "|err|<=3% on every grid AND err(R=30)<=err(R=20) (err vs rho_l form)",
        },
        "cases": {},
    }
    radii = [float(x) for x in args.radii.split(",")]
    rows = []
    for R0 in radii:
        L = int(4 * R0)
        print(f"\n===== R={R0:.0f}  L={L}  steps={args.steps}  device={device} =====", flush=True)
        try:
            final, hist = run_case(R0, L, args.eps, device, args.steps, args.sample)
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            rows.append({"R_init": R0, "error": str(e)})
            continue
        rows.append(final)
        print(
            f"  R_eq={final['R_eq_mean']:.3f}  T_theory(rho_l)={final['period_theory_rho_l']:.0f}  "
            f"win={final['fit_window_steps']}\n"
            f"  omega_obs(fit)={final['omega_sim_fit']:.6e}  err_rho_l={final['err_pct_fit_vs_rho_l']:+.3f}%  "
            f"err_rho_lv={final['err_pct_fit_vs_rho_lv']:+.3f}%  gamma={final['gamma_fit']:.2e}\n"
            f"  omega_0(restored)={final['omega_0_fit']:.6e}  err_rho_l={final['err_pct_omega0_vs_rho_l']:+.3f}%  "
            f"mass_drift={final['mass_drift']:.2e}  ({final['dt_s']:.0f}s)"
        )
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(
                out / f"hist_R{int(R0)}.npz",
                step=np.array([h["step"] for h in hist]),
                Q=np.array([h["Q"] for h in hist]),
                R_eq=np.array([h["R_eq"] for h in hist]),
                R_x=np.array([h["R_x"] for h in hist]),
                R_y=np.array([h["R_y"] for h in hist]),
            )
            (out / f"hist_R{int(R0)}.json").write_text(
                json.dumps(
                    [{k: (round(v, 8) if isinstance(v, float) else v) for k, v in h.items()}
                     for h in hist],
                    indent=1,
                ),
                encoding="utf-8",
            )

    good = [r for r in rows if "error" not in r]
    for r in good:
        results["cases"][f"R{int(r['R_init'])}"] = r
    if len(good) == len(radii) and len(good) >= 2:
        # PRIMARY criterion: observed (damped) frequency vs Rayleigh rho_l form
        errs = {r["R_init"]: abs(r["err_pct_fit_vs_rho_l"]) for r in good}
        all_pass = all(e <= 3.0 for e in errs.values())
        r_fine = max(radii)
        r_coarse = min(radii)
        conv = errs[r_fine] <= errs[r_coarse]
        results["pass"] = bool(all_pass and conv)
        results["convergence"] = {
            "err_pct_observed_vs_rho_l_by_R": {str(int(k)): v for k, v in errs.items()},
            "finer_grid_err_le_coarser": bool(conv),
        }
        # diagnostic: undamped eigenfrequency restored via omega_0=sqrt(w_d^2+g^2)
        results["diagnostics_undamped"] = {
            str(int(r["R_init"])): {
                "gamma_fit": r["gamma_fit"],
                "omega_0_fit": r["omega_0_fit"],
                "err_pct_omega0_vs_rho_l": r["err_pct_omega0_vs_rho_l"],
            }
            for r in good
        }
    else:
        results["pass"] = False
    print(f"\n  PASS={results['pass']}  {results.get('convergence', {})}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Saved -> {out / 'result.json'}")


if __name__ == "__main__":
    main()
