"""W2-D benchmark: D2Q9 plane acoustic wave — linear dispersion & attenuation.

First acoustics-class benchmark for TensorLBM.  Library kernels only
(tensorlbm.solver.collide_bgk / stream, tensorlbm.d2q9.equilibrium /
macroscopic); periodic domain; fp64 CPU; no extrapolation, no artificial
correction, no sponge, no forcing.

Pre-registered theory (NOTES.md section 1, registered before any run):
  continuum :  omega = c_s k,  amplitude decay delta = nu k^2,
               c_s = 1/sqrt(3), nu = c_s^2 (tau - 1/2)
  discrete  :  exact root s(k, tau) of the linearized collide-then-stream
               operator (9x9 per-wavenumber eigenproblem, cross-checked
               against the analytic dispersion determinant)

Cases:
  main       : N in {16, 32, 64}, tau = 1.0, eps = 1e-3  (convergence)
  tau sweep  : N = 32, tau in {0.6, 0.8, 1.2}            (secondary)
  linearity  : N = 32, eps in {5e-4, 1e-2}               (secondary)
  cavity     : square L in {32, 64}, BB walls            (secondary, honest)

Output: result.json + console tables.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

from tensorlbm.d2q9 import OPPOSITE, C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

DEVICE = torch.device("cpu")
DTYPE = torch.float64
CS = 1.0 / math.sqrt(3.0)  # lattice sound speed, LU/step
TAU_MAIN = 1.0
EPS_MAIN = 1.0e-3
SQ3 = math.sqrt(3.0)

# ----------------------------------------------------------------------------
# theory references (exact discrete eigenvalue of the linear update)
# ----------------------------------------------------------------------------
_W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_CX9 = C[:, 0].numpy().astype(float)


def eig_acoustic(k: float, tau: float) -> complex:
    """Right-going acoustic eigenvalue s = log(lambda) of the linear
    collide-then-stream update for wavenumber k (axis-aligned).

    lambda F_q = e^{i k c_qx} [ (1-r) F_q + r w_q (R + c_qx U / c_s^2) ],
    R = sum_p F_p, U = sum_p c_px F_p, r = 1/tau, c_s^2 = 1/3.
    """
    r_ = 1.0 / tau
    I9 = np.eye(9)
    A = np.zeros((9, 9), dtype=complex)
    for q in range(9):
        A[q, :] = np.exp(1j * k * _CX9[q]) * (
            (1.0 - r_) * I9[q] + r_ * _W9[q] * (np.ones(9) + 3.0 * _CX9[q] * _CX9)
        )
    lam = np.linalg.eigvals(A)
    j = int(np.argmin(np.abs(np.log(lam) - 1j * k * CS)))
    return complex(np.log(lam[j]))


def refs(k: float, tau: float) -> dict:
    """Continuum + discrete-theory references for one (k, tau)."""
    s_star = eig_acoustic(k, tau)
    nu = CS * CS * (tau - 0.5)
    return {
        "k": k,
        "tau": tau,
        "nu": nu,
        "c_cont": CS,
        "delta_cont": nu * k * k,
        "s_discrete": [s_star.real, s_star.imag],
        "c_discrete": s_star.imag / k,
        "delta_discrete": -s_star.real,
    }


def fit_decay_robust(
    zp: np.ndarray, n_arr: np.ndarray, i0: int, i1: int, omega_hat: float
) -> tuple[float, float, float, float]:
    """Attenuation from |zp|^2 = A e^{-2 delta n} (1 + eta cos(2 w n + phi)).

    The equilibrium initial condition seeds a small counter-rotating (left-
    going) acoustic wave with relative amplitude b/a ~ nu k/(2 c_s) (velocity
    eigenvector imaginary-part mismatch); it beats |zp| at 2 omega and biases
    a plain log-amplitude LSQ slope (measured bias +0.033%, diag_ripple.py).
    This estimator is linear in (A, B, C) given delta and absorbs any 2*omega
    content exactly.  Returns (delta, eta, r2, delta_naive).
    """
    n = n_arr[i0 : i1 + 1]
    y = np.abs(zp[i0 : i1 + 1]) ** 2
    cosb = np.cos(2.0 * omega_hat * n)
    sinb = np.sin(2.0 * omega_hat * n)
    M = np.stack([np.ones_like(n), cosb, sinb], axis=1)

    def sse(delta: float) -> tuple[float, np.ndarray]:
        w = y * np.exp(2.0 * delta * n)
        c, *_ = np.linalg.lstsq(M, w, rcond=None)
        r = w - M @ c
        return float(r @ r), c

    d0 = -float(np.polyfit(n, 0.5 * np.log(y), 1)[0])  # naive start
    grid = np.geomspace(max(0.2 * d0, 1e-12), 5.0 * d0, 240)
    vals = [sse(d)[0] for d in grid]
    j = int(np.argmin(vals))
    lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    for _ in range(80):
        m1 = lo + 0.38197 * (hi - lo)
        m2 = hi - 0.38197 * (hi - lo)
        if sse(m1)[0] < sse(m2)[0]:
            hi = m2
        else:
            lo = m1
    d_star = 0.5 * (lo + hi)
    _, c = sse(d_star)
    eta = float(math.hypot(c[1], c[2]) / c[0])
    pred = np.exp(-2.0 * d_star * n) * (M @ c)
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst
    return d_star, eta, r2, d0


def self_test_estimator() -> None:
    """Synthetic check: recover delta/omega of a e^{sn} + b e^{s*n} + floor."""
    rng = np.random.default_rng(0)
    s = -2.604e-2 + 1j * 0.113337  # ~ N=16, tau=1
    b_rel, phi_b = 0.058, 1.1  # measured contamination
    n = np.arange(1200, dtype=float)
    zp = np.exp(s * n) + b_rel * np.exp(np.conj(s) * n + 1j * phi_b)
    zp *= np.exp(1j * 0.3)
    zp = zp.astype(complex)
    ph = np.unwrap(np.angle(zp))
    omega_hat = -float(np.polyfit(n, ph, 1)[0])
    d, eta, r2, d0 = fit_decay_robust(zp, n, 0, len(n) - 1, omega_hat)
    err_r = -s.real
    print(f"[self-test] true delta={err_r:.6e} omega={s.imag:.6f}")
    print(
        f"[self-test] naive slope delta={d0:.6e} (err {100 * (d0 / err_r - 1):+.4f}%)"
        f"  robust delta={d:.6e} (err {100 * (d / err_r - 1):+.6f}%)  eta/2={eta / 2:.4f}"
        f" (b/a true {b_rel})  R2={r2:.8f}"
    )
    assert abs(d / err_r - 1) < 1e-7, "robust estimator failed synthetic test"
    assert abs(eta / 2 - b_rel) / b_rel < 0.01


# ----------------------------------------------------------------------------
# plane-wave simulation (library kernels, periodic, fp64)
# ----------------------------------------------------------------------------
def plane_wave_run(N: int, tau: float, eps: float, ny: int = 8, n_periods: float = 16.0) -> dict:
    """Run one plane-wave case and measure the fundamental mode."""
    nx = N
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, device=DEVICE, dtype=DTYPE)
    phase = k * x
    rho0 = 1.0 + eps * torch.sin(phase)
    ux0 = CS * eps * torch.sin(phase)  # isothermal u' = c_s rho'/rho0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(
        rho0.unsqueeze(0).expand(ny, nx).contiguous(),
        ux0.unsqueeze(0).expand(ny, nx).contiguous(),
        uy0.unsqueeze(0).expand(ny, nx).contiguous(),
    )

    period = SQ3 * nx  # acoustic period in steps
    nsteps = int(math.ceil(n_periods * period))
    e_minus = np.exp(-1j * k * np.arange(nx))
    e_plus = np.conj(e_minus)

    zp = np.empty(nsteps + 1, dtype=complex)
    zm = np.empty(nsteps + 1, dtype=complex)
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        rp = (rho[0].numpy()).astype(np.float64) - 1.0  # y-invariant
        zp[n] = rp @ e_minus / nx
        zm[n] = rp @ e_plus / nx
        if n == nsteps:
            break
        f = collide_bgk(f, tau)
        f = stream(f)
    wall = time.time() - t0

    i0 = int(math.ceil(0.5 * period))
    i1 = min(int(math.floor(15.5 * period)), nsteps)
    n_arr = np.arange(nsteps + 1, dtype=np.float64)
    sl = slice(i0, i1 + 1)

    ph = np.unwrap(np.angle(zp[sl]))
    slope_p, icept_p = np.polyfit(n_arr[sl], ph, 1)
    predp = np.polyval([slope_p, icept_p], n_arr[sl])
    ss_resp = float(np.sum((ph - predp) ** 2))
    ss_totp = float(np.sum((ph - ph.mean()) ** 2))
    r2_ph = 1.0 - ss_resp / ss_totp
    omega_meas = -slope_p  # arg zp rotates at -omega

    delta_meas, eta, r2_amp, delta_naive = fit_decay_robust(zp, n_arr, i0, i1, omega_meas)
    c_meas = omega_meas / k

    rho_end = macroscopic(f)[0]
    out = {
        "N": N,
        "ny": ny,
        "k": k,
        "tau": tau,
        "eps": eps,
        "nsteps": nsteps,
        "period_steps": period,
        "fit_window": [i0, i1],
        "wall_s": wall,
        "delta_meas": delta_meas,
        "delta_meas_naive": delta_naive,
        "omega_meas": omega_meas,
        "c_meas": c_meas,
        "r2_amp": r2_amp,
        "r2_phase": r2_ph,
        "counterwave_ratio_eta_over_2": eta / 2.0,
        "max_abs_rho_minus_1_end": float((rho_end - 1).abs().max()),
        "rho_min_end": float(rho_end.min()),
        "rho_max_end": float(rho_end.max()),
        "finite": bool(torch.isfinite(f).all()),
    }
    out.update(refs(k, tau))
    out["c_err_cont_pct"] = 100.0 * (c_meas / CS - 1.0)
    out["delta_err_cont_pct"] = 100.0 * (delta_meas / out["delta_cont"] - 1.0)
    out["c_err_disc_pct"] = 100.0 * (c_meas / out["c_discrete"] - 1.0)
    out["delta_err_disc_pct"] = 100.0 * (delta_meas / out["delta_discrete"] - 1.0)
    return out


# ----------------------------------------------------------------------------
# secondary: square cavity standing wave, second eigenmode (1,0), BB walls
# ----------------------------------------------------------------------------
def cavity_run(L: int, tau: float, eps: float = 1.0e-3, n_periods: float = 14.0) -> dict:
    """Standing wave in a square cavity with half-way BB walls.

    Wall pattern = repo-validated pre-streaming reflection (verified/
    poiseuille_2d / couette_2d / womersley): wall rows' post-collision state
    is replaced by the reflection of their PRE-collision state via the
    library OPPOSITE table, then the library periodic stream.
    """
    ny = nx = L
    leff = L - 2  # wall planes at 0.5 / L-1.5
    k = math.pi / leff
    xg = torch.arange(nx, device=DEVICE, dtype=DTYPE)
    mode = torch.cos(k * (xg - 0.5))
    rho0 = torch.ones((ny, nx), dtype=DTYPE)
    rho0[1 : ny - 1, 1 : nx - 1] = 1.0 + eps * mode[1 : nx - 1]
    u0 = torch.zeros((ny, nx), dtype=DTYPE)
    f = equilibrium(rho0, u0.clone(), u0.clone())

    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    opp = OPPOSITE.to(DEVICE)

    ref = refs(k, tau)
    omega_th = ref["c_discrete"] * k
    period = 2.0 * math.pi / omega_th
    nsteps = int(math.ceil(n_periods * period))
    tq = max(1, int(round(period / 4.0)))

    zz = np.empty(nsteps + 1)
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        rp = (rho[1 : ny - 1, 1 : nx - 1] - 1.0).mean(dim=0).numpy()
        zz[n] = 2.0 * float(rp @ mode[1 : nx - 1].numpy()) / leff
        if n == nsteps:
            break
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(solid.unsqueeze(0), f_pre[opp], f)
        f = stream(f)
    wall = time.time() - t0

    i0 = int(math.ceil(1.0 * period))
    i1 = nsteps - tq
    n_arr = np.arange(nsteps + 1, dtype=np.float64)
    # quadrature-pair envelope: A^2 = Z(n)^2 + Z(n+T/4)^2 -> slope -2 delta
    a2 = zz[i0 : i1 + 1] ** 2 + zz[i0 + tq : i1 + 1 + tq] ** 2
    slope, icept = np.polyfit(n_arr[i0 : i1 + 1], np.log(a2), 1)
    delta_meas = -slope / 2.0

    # measured frequency: count zero crossings of Z over the window
    zwin = zz[i0 : i1 + 1]
    ncross = int(np.sum(np.sign(zwin[:-1]) * np.sign(zwin[1:]) < 0))
    omega_meas = math.pi * ncross / (n_arr[i1] - n_arr[i0])

    out = {
        "L": L,
        "L_eff": leff,
        "k": k,
        "tau": tau,
        "eps": eps,
        "nsteps": nsteps,
        "period_steps": period,
        "wall_s": wall,
        "delta_meas": delta_meas,
        "omega_meas": omega_meas,
        "omega_discrete": omega_th,
        "delta_bulk_discrete": ref["delta_discrete"],
        "delta_bulk_cont": ref["delta_cont"],
        "delta_err_vs_bulk_pct": 100.0 * (delta_meas / ref["delta_discrete"] - 1.0),
        "omega_err_vs_discrete_pct": 100.0 * (omega_meas / omega_th - 1.0),
        "finite": bool(torch.isfinite(f).all()),
    }
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    torch.set_num_threads(1)
    self_test_estimator()
    res: dict = {
        "meta": {
            "task": "W2-D D2Q9 plane acoustic wave dispersion & attenuation",
            "worktree": "/nfs/wangxi/worktrees/bm_w2",
            "git": "c0b84d96d1",
            "library": [
                "tensorlbm.solver.collide_bgk",
                "tensorlbm.solver.stream",
                "tensorlbm.d2q9.equilibrium",
                "tensorlbm.d2q9.macroscopic",
            ],
            "dtype": "float64",
            "device": "cpu",
            "torch_threads": 1,
        }
    }

    print("=" * 78)
    print("MAIN: convergence N in {16,32,64}, tau=1.0 (nu=1/6 fixed), eps=1e-3")
    print("=" * 78)
    main_cases = []
    for N in (16, 32, 64):
        out = plane_wave_run(N, TAU_MAIN, EPS_MAIN)
        main_cases.append(out)
        print(f"N={N:>3} k={out['k']:.6f} steps={out['nsteps']}")
        print(
            f"   c_meas={out['c_meas']:.9f}  vs cont {CS:.9f}: "
            f"{out['c_err_cont_pct']:+.4f}%   vs disc {out['c_discrete']:.9f}: "
            f"{out['c_err_disc_pct']:+.5f}%"
        )
        print(
            f"   delta_meas={out['delta_meas']:.6e} (naive "
            f"{out['delta_meas_naive']:.6e}) vs cont "
            f"{out['delta_cont']:.6e}: {out['delta_err_cont_pct']:+.4f}%"
            f"   vs disc {out['delta_discrete']:.6e}: "
            f"{out['delta_err_disc_pct']:+.5f}%"
        )
        print(
            f"   R2 amp={out['r2_amp']:.8f} R2 ph={out['r2_phase']:.8f}"
            f"  counter-wave b/a={out['counterwave_ratio_eta_over_2']:.4f}"
            f"  finite={out['finite']}"
        )
    ec = [abs(c["c_err_cont_pct"]) for c in main_cases]
    ed = [abs(c["delta_err_cont_pct"]) for c in main_cases]
    mono_c = all(ec[i + 1] < ec[i] for i in range(len(ec) - 1))
    mono_d = all(ed[i + 1] < ed[i] for i in range(len(ed) - 1))
    verdict = {
        "c_err_pct_max": max(ec),
        "delta_err_pct_max": max(ed),
        "c_monotone": mono_c,
        "delta_monotone": mono_d,
        "pass_c": max(ec) <= 3.0 and mono_c,
        "pass_delta": max(ed) <= 3.0 and mono_d,
    }
    print(
        f"VERDICT main: |c err| max {max(ec):.4f}% (mono {mono_c}) -> "
        f"{'PASS' if verdict['pass_c'] else 'FAIL'}; "
        f"|delta err| max {max(ed):.4f}% (mono {mono_d}) -> "
        f"{'PASS' if verdict['pass_delta'] else 'FAIL'}"
    )
    res["convergence"] = main_cases
    res["verdict_main"] = verdict

    print("=" * 78)
    print("TAU SWEEP: N=32, tau in {0.6, 0.8, 1.0, 1.2}")
    print("=" * 78)
    sweep = [plane_wave_run(32, tv, EPS_MAIN) for tv in (0.6, 0.8, 1.0, 1.2)]
    for out in sweep:
        ratio = out["delta_meas"] / (out["tau"] - 0.5)
        print(
            f"tau={out['tau']:.2f} nu={out['nu']:.5f} "
            f"delta_meas={out['delta_meas']:.6e} "
            f"err cont={out['delta_err_cont_pct']:+.4f}% "
            f"err disc={out['delta_err_disc_pct']:+.5f}% "
            f"delta/(tau-1/2)={ratio:.6e}"
        )
    res["tau_sweep"] = sweep

    print("=" * 78)
    print("LINEARITY: N=32, tau=1.0, eps in {5e-4, 1e-3, 1e-2}")
    print("=" * 78)
    lin = [plane_wave_run(32, TAU_MAIN, ev) for ev in (5.0e-4, 1.0e-3, 1.0e-2)]
    for out in lin:
        print(
            f"eps={out['eps']:.0e} c_meas={out['c_meas']:.9f} "
            f"({out['c_err_cont_pct']:+.4f}%) "
            f"delta_meas={out['delta_meas']:.6e} "
            f"({out['delta_err_cont_pct']:+.4f}%)"
        )
    res["linearity"] = lin

    print("=" * 78)
    print("SECONDARY: square cavity standing wave (BB walls), L in {32, 64}")
    print("=" * 78)
    cav = [cavity_run(Lv, TAU_MAIN) for Lv in (32, 64)]
    for out in cav:
        print(
            f"L={out['L']} k={out['k']:.6f} "
            f"omega_meas={out['omega_meas']:.6f} vs disc "
            f"{out['omega_discrete']:.6f}: "
            f"{out['omega_err_vs_discrete_pct']:+.3f}%"
        )
        print(
            f"   delta_meas={out['delta_meas']:.6e} vs bulk "
            f"{out['delta_bulk_discrete']:.6e}: "
            f"{out['delta_err_vs_bulk_pct']:+.2f}% (BB wall loss included)"
        )
    res["cavity"] = cav

    with open("result.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote result.json")


if __name__ == "__main__":
    main()
