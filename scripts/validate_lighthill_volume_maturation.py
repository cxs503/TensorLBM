"""Validation driver for the Lighthill volume-quadrupole maturation (P170W2B-N03).

Produces ``artifacts/m9740/lighthill_volume_maturation_results.json`` plus the
figures used in ``docs/lighthill_volume/MATURATION_REPORT.md``.

Criteria implemented (numbers are written into the JSON, never hand-copied):

A. Operator anchor  : 2-D volume-quadrupole integral vs exact (non-compact)
                      far-field closed form -> <= 2 % (task (b)).
B. Retarded time    : delayed-time 3-D FW-H volume term vs the exact
                      frequency-domain operator -> <= 2 % (task (a)).
C. Vortex pair      : peak at 2*f_rot, phase-locked |cos 2theta| four-lobe,
                      amplitude ~ M^3 in Mach-normalised form (task (c)).
D. Surface/volume   : Kirchhoff (FWH permeable surface) vs volume integral at
                      r/lambda >= 2 -> |dSPL| <= 3 dB (task (d)).
E. Compact limit    : exact finite-size vs point-quadrupole closed form and
                      closed-form acoustic power vs numerical intensity
                      integral (task (e)).

Run:  python scripts/validate_lighthill_volume_maturation.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.lighthill_volume import (  # noqa: E402
    analytical_gaussian_quadrupole_farfield,
    co_rotating_vortex_pair,
    compact_quadrupole_power,
    compute_lighthill_tensor,
    gaussian_quadrupole_farfield_3d,
    gaussian_quadrupole_farfield_exact,
    kirchhoff_surface_2d,
    rotating_quadrupole_analysis,
    volume_quadrupole_frequency_domain,
    volume_quadrupole_time_domain,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C0 = 1.0 / math.sqrt(3.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "m9740")
os.makedirs(ART, exist_ok=True)
FIG: List[Dict[str, str]] = []
RESULTS: Dict[str, object] = {}


def _save(fig, name: str, caption: str) -> None:
    path = os.path.join(ART, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    FIG.append({"name": name, "path": path, "caption": caption})
    print(f"  figure -> {path}")


# ---------------------------------------------------------------------------
# A. Operator anchor: 2-D volume quadrupole vs exact far-field closed form
# ---------------------------------------------------------------------------

def _gaussian_source_2d(omega: float, sigma: float, extent_sigma: float = 5.0,
                        dx_over_sigma: float = 0.25, quad: str = "xx-yy"):
    half = extent_sigma * sigma
    dx = dx_over_sigma * sigma
    n = int(round(2 * half / dx)) + 1
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    g = np.exp(-(xx ** 2 + yy ** 2) / sigma ** 2)
    A = np.array([[1.0, 0.0], [0.0, -1.0]]) if quad == "xx-yy" else np.array([[0.0, 1.0], [1.0, 0.0]])
    T = g[..., None, None] * A
    src = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    w = np.full(xx.size, dx * dx)
    return dict(T=T, src=src, w=w, dx=dx, n=n, xx=xx, yy=yy, g=g)


def criterion_a() -> Dict[str, object]:
    print("[A] operator anchor: 2-D volume integral vs exact far-field closed form")
    omega = 0.15
    k = omega / C0
    lam = 2 * math.pi / k
    out: Dict[str, object] = {"omega": omega, "rows": [], "dx_rows": [], "kr_rows": []}
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    def _num_err(src_dict, obs, sigma):
        """Relative error, restricted to the lobes (the 45-degree nulls of a
        quadrupole make pointwise relative errors ill-conditioned there).
        Returns (max pointwise error on the lobes, L2 error normalised by the
        pattern maximum over all observers)."""
        p_num = volume_quadrupole_frequency_domain(
            src_dict["T"].reshape(-1, 2, 2), src_dict["src"], src_dict["w"], obs,
            omega, C0, ndim=2, batch_size=12)
        p_ana = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
        ref = np.abs(p_ana)
        lobe = ref > 0.2 * ref.max()
        pointwise = np.abs(np.abs(p_num[lobe]) - ref[lobe]) / ref[lobe]
        l2 = np.sqrt(np.mean((np.abs(p_num) - ref) ** 2)) / ref.max()
        return pointwise, l2

    # (A1) sigma/lambda sweep at fixed far-field radius
    for s_lam in (0.02, 0.05, 0.10, 0.20):
        sigma = s_lam * lam
        g2 = _gaussian_source_2d(omega, sigma)
        r = 30 * lam
        obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        rel, l2 = _num_err(g2, obs, sigma)
        out["rows"].append({
            "sigma_over_lambda": s_lam,
            "k_sigma": k * sigma,
            "r_over_lambda": 30.0,
            "kr": k * r,
            "max_rel_err_lobes": float(rel.max()),
            "l2_rel_err": float(l2),
            "nsrc": int(g2["src"].shape[0]),
        })
        print(f"   sigma/lam={s_lam:.2f} kr={k*r:.0f}: lobe max rel err {rel.max():.2e}, "
              f"L2 {l2:.2e}")

    # (A2) grid convergence at fixed sigma/lambda
    sigma = 0.10 * lam
    r = 30 * lam
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    for dxos in (1.0, 0.5, 0.25, 0.125):
        g2 = _gaussian_source_2d(omega, sigma, dx_over_sigma=dxos)
        rel, l2 = _num_err(g2, obs, sigma)
        out["dx_rows"].append({"dx_over_sigma": dxos, "nsrc": int(g2["src"].shape[0]),
                               "max_rel_err_lobes": float(rel.max()), "l2_rel_err": float(l2)})
        print(f"   dx/sigma={dxos:5.3f}: lobe max rel err {rel.max():.2e}")

    # (A3) far-field truncation of the closed form: error vs kr
    g2 = _gaussian_source_2d(omega, sigma, dx_over_sigma=0.25)
    for kr_target in (25.0, 50.0, 100.0, 200.0, 400.0):
        r = kr_target / k
        obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        rel, l2 = _num_err(g2, obs, sigma)
        out["kr_rows"].append({"kr": kr_target, "r_over_lambda": r / lam,
                               "max_rel_err_lobes": float(rel.max()), "l2_rel_err": float(l2)})
        print(f"   kr={kr_target:5.0f}: lobe max rel err {rel.max():.2e}")

    out["max_rel_err"] = float(max(r["max_rel_err_lobes"] for r in out["rows"]))
    out["max_l2_rel_err"] = float(max(r["l2_rel_err"] for r in out["rows"]))
    out["threshold"] = 0.02
    out["verdict"] = "PASS" if out["max_rel_err"] <= 0.02 else "FAIL"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.loglog([r["sigma_over_lambda"] for r in out["rows"]],
              [r["max_rel_err_lobes"] for r in out["rows"]], "o-",
              label="max rel. error on lobes (kr=188)")
    ax.axhline(0.02, color="r", ls="--", label="acceptance 2 %")
    ax.set_xlabel(r"$\sigma/\lambda$")
    ax.set_ylabel("relative error")
    ax.set_title("(A1) non-compact source sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax = axes[1]
    ax.loglog([r["kr"] for r in out["kr_rows"]], [r["l2_rel_err"] for r in out["kr_rows"]],
              "s-", label="L2 error over all observers")
    kr = np.array([r["kr"] for r in out["kr_rows"]], dtype=float)
    ax.loglog(kr, 3.0 / kr, "k:", label=r"$O(1/kr)$ reference")
    ax.axhline(0.02, color="r", ls="--")
    ax.set_xlabel(r"$kr$")
    ax.set_ylabel("max relative error")
    ax.set_title("(A3) far-field truncation of the closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "A_operator_anchor.png",
          "A: volume-quadrupole operator vs exact far-field closed form "
          "(sigma/lambda sweep and kr truncation).")
    return out


# ---------------------------------------------------------------------------
# B. Retarded-time (delayed-time) 3-D FW-H volume term
# ---------------------------------------------------------------------------

def _td_setup(A3: np.ndarray, sigma: float = 0.25, dx_over_sigma: float = 2.5,
              n: int = 11, omega: float = 0.8, spp: int = 64, rol: float = 25.0):
    lam = 2 * math.pi * C0 / omega
    per = 2 * math.pi / omega
    dx = sigma / dx_over_sigma
    x = (np.arange(n) - n // 2) * dx
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
    TT = g[..., None, None] * A3
    org = (-(n // 2) * dx,) * 3
    r = rol * lam
    th = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3])
    obs = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=-1)
    dt = per / spp
    n_t = int((rol + 2.5) * spp)
    T = np.empty((n_t, n, n, n, 3, 3), dtype=np.float32)
    for it in range(n_t):
        T[it] = (TT * math.cos(omega * it * dt)).astype(np.float32)
    return dict(T=T, dx=dx, org=org, obs=obs, dt=dt, omega=omega, sigma=sigma,
                lam=lam, r=r, spp=spp, xx=xx, yy=yy, zz=zz, n=n, TT=TT)


def _fit_amp_phase(signal: np.ndarray, t: np.ndarray, omega: float):
    A = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coef, *_ = np.linalg.lstsq(A, signal, rcond=None)
    return float(np.hypot(*coef)), float(math.atan2(-coef[1], coef[0]))


def criterion_b() -> Dict[str, object]:
    print("[B] retarded-time (delayed-time) 3-D FW-H volume term")
    out: Dict[str, object] = {"rows": [], "time_series": {}}
    A3 = np.diag([1.0, -1.0, 0.0])
    for near in (False, True):
        for spp in (64, 96):
            s = _td_setup(A3, spp=spp)
            t0 = time.time()
            p_td, t_td = volume_quadrupole_time_domain(
                s["T"], s["dx"], s["obs"], C0, s["dt"], grid_origin=s["org"],
                include_near_field=near)
            elapsed = time.time() - t0
            start = int(np.argmax(t_td > (s["r"] + 2.0) / C0))
            sel = slice(start, start + spp)
            tt = t_td[sel]
            p_fx = volume_quadrupole_frequency_domain(
                s["TT"].reshape(-1, 3, 3),
                np.stack([s["xx"].ravel(), s["yy"].ravel(), s["zz"].ravel()], axis=-1),
                np.full(s["n"] ** 3, s["dx"] ** 3), s["obs"], s["omega"], C0,
                ndim=3, batch_size=6)
            p_ana = gaussian_quadrupole_farfield_3d(s["obs"], s["omega"], C0, 1.0, s["sigma"])
            e_fx, e_ana, ph_fx = [], [], []
            for i in range(s["obs"].shape[0]):
                amp, ph = _fit_amp_phase(p_td[i, sel], tt, s["omega"])
                e_fx.append(abs(amp - abs(p_fx[i])) / abs(p_fx[i]))
                e_ana.append(abs(amp - abs(p_ana[i])) / abs(p_ana[i]))
                ph_fx.append(abs(np.angle(np.exp(1j * (ph - np.angle(p_fx[i]))))))
            row = {
                "include_near_field": near,
                "samples_per_period": spp,
                "dt": s["dt"],
                "n_t": int(s["T"].shape[0]),
                "kr": float(s["omega"] / C0 * s["r"]),
                "max_rel_err_vs_freq_exact": float(max(e_fx)),
                "max_rel_err_vs_analytic_farfield": float(max(e_ana)),
                "max_phase_err_rad": float(max(ph_fx)),
                "runtime_s": float(elapsed),
            }
            out["rows"].append(row)
            print(f"   near_field={near} spp={spp}: err vs exact {max(e_fx):.2e}, "
                  f"vs analytic {max(e_ana):.2e}, phase {max(ph_fx):.2e} rad")
            if near and spp == 96:
                amp, ph = _fit_amp_phase(p_td[0, sel], tt, s["omega"])
                out["time_series"] = {
                    "t": tt.tolist(),
                    "p_td": p_td[0, sel].tolist(),
                    "amp_td": float(amp),
                    "phase_td": float(ph),
                    "amp_analytic": float(abs(p_ana[0])),
                    "phase_analytic": float(np.angle(p_ana[0])),
                    "omega": s["omega"],
                }
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].plot(tt, p_td[0, sel], "o", ms=3, label="retarded-time integral")
                axes[0].plot(tt, amp * np.cos(s["omega"] * tt + ph), "-",
                             label="least-squares harmonic fit")
                axes[0].set_xlabel("observer time (lattice units)")
                axes[0].set_ylabel("p' (lattice units)")
                axes[0].set_title("(B1) delayed-time signal at r/lambda=25")
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                ax = axes[1]
                for near_flag, marker in ((False, "s"), (True, "o")):
                    rows = [r for r in out["rows"] if r["include_near_field"] is near_flag]
                    ax.loglog([r["samples_per_period"] for r in rows],
                              [max(r["max_rel_err_vs_freq_exact"], 1e-6) for r in rows],
                              marker + "-",
                              label=("3-term kernel" if near_flag else "far-field term only"))
                ax.axhline(0.02, color="r", ls="--", label="acceptance 2 %")
                ax.set_xlabel("samples per acoustic period")
                ax.set_ylabel("max rel. error vs exact operator")
                ax.set_title("(B2) time-resolution convergence")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
                _save(fig, "B_retarded_time.png",
                      "B: retarded-time (delayed-time) FW-H volume term validated against the "
                      "exact frequency-domain operator; 3-term vs far-field-only kernel.")
    e = min(r["max_rel_err_vs_freq_exact"] for r in out["rows"] if r["samples_per_period"] == 96)
    out["best_rel_err"] = float(e)
    out["threshold"] = 0.02
    out["verdict"] = "PASS" if e <= 0.02 else "FAIL"
    return out


# ---------------------------------------------------------------------------
# C. Physics anchor: co-rotating vortex pair
# ---------------------------------------------------------------------------

def _pair_T(xs, ys, gamma, half_sep, core, t):
    ux, uy = co_rotating_vortex_pair(xs, ys, gamma, half_sep, core, t)
    T = np.empty((xs.size, 2, 2))
    T[:, 0, 0] = ux * ux
    T[:, 0, 1] = T[:, 1, 0] = ux * uy
    T[:, 1, 1] = uy * uy
    return T


def vortex_pair_farfield(M: float, *, half_sep: float = 6.0, core: float = 2.5,
                         mask_radius: float = 16.0, dx: float = 0.5,
                         n_obs: int = 64, n_t: int = 96, r_over_lambda: float = 10.0,
                         harmonics=(1, 2, 3, 4)):
    """Far-field complex amplitude of the rotating-pair quadrupole, 2-D."""
    gamma = M * C0 * 4.0 * math.pi * half_sep
    omega_rot = gamma / (4.0 * math.pi * half_sep ** 2)
    omega2 = 2.0 * omega_rot
    k = omega2 / C0
    lam = 2.0 * math.pi / k
    r = r_over_lambda * lam

    L = mask_radius + 0.5
    x = np.arange(-L, L + 1e-12, dx)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mask = (xx ** 2 + yy ** 2) <= mask_radius ** 2
    xs, ys = xx[mask], yy[mask]
    src = np.stack([xs, ys], axis=-1)
    w = np.full(xs.size, dx * dx)

    th = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)

    period = math.pi / omega_rot          # 2-fold symmetric pattern period
    dt = period / n_t
    T_hat = np.zeros((xs.size, 2, 2), dtype=complex)
    for it in range(n_t):
        t = it * dt
        T_hat += _pair_T(xs, ys, gamma, half_sep, core, t) * np.exp(-1j * omega2 * t) * dt
    T_hat *= 2.0 / period                 # Re(T_hat e^{i omega t}) = T amplitude

    p_hat = volume_quadrupole_frequency_domain(
        T_hat, src, w, obs, omega2, C0, ndim=2, batch_size=16)
    analysis = rotating_quadrupole_analysis(p_hat, th, omega2)

    harm = {}
    per_full = 2 * math.pi / omega_rot
    nts = 4 * n_t
    dts = per_full / nts
    cache = [_pair_T(xs, ys, gamma, half_sep, core, it * dts) for it in range(nts)]
    for n in harmonics:
        w_n = n * omega_rot
        T_hat_n = np.zeros((xs.size, 2, 2), dtype=complex)
        for it in range(nts):
            T_hat_n += cache[it] * np.exp(-1j * w_n * it * dts) * dts
        T_hat_n *= 2.0 / per_full
        p_n = volume_quadrupole_frequency_domain(
            T_hat_n, src, w, obs, w_n, C0, ndim=2, batch_size=16)
        harm[n] = float(np.abs(p_n).mean())
    return dict(p_hat=p_hat, th=th, r=r, lam=lam, omega_rot=omega_rot, omega2=omega2,
                analysis=analysis, harmonic_amp=harm, nsrc=int(xs.size), M=M)


def criterion_c() -> Dict[str, object]:
    print("[C] physics anchor: co-rotating vortex pair")
    out: Dict[str, object] = {"runs": [], "scaling": {}, "convergence": []}

    machs = (0.03, 0.05, 0.075, 0.10)
    amps = []
    for M in machs:
        res = vortex_pair_farfield(M)
        a = res["analysis"]
        amps.append(a["amplitude"])
        harm = res["harmonic_amp"]
        h2 = harm[2]
        peak_n = max(harm, key=lambda n: harm[n])
        out["runs"].append({
            "mach": M,
            "amplitude": a["amplitude"],
            "mach_normalised": a["amplitude"] / (C0 ** 2 * M),
            "dominant_mode": a["dominant_mode"],
            "winding": a["winding"],
            "rms_isotropy": a["isotropy"],
            "pattern_corr_cos2": a["pattern_corr_cos2"],
            "pattern_corr_sin2": a["pattern_corr_sin2"],
            "peak_harmonic_n": int(peak_n),
            "peak_freq_over_2f_rot": float(peak_n / 2.0),
            "harmonic_ratio": {f"{n}Omega_over_2Omega": harm[n] / h2 for n in sorted(harm)},
            "nsrc": res["nsrc"],
            "r_over_lambda": res["r"] / res["lam"],
        })
        print(f"   M={M}: |p|={a['amplitude']:.3e} norm={a['amplitude']/(C0**2*M):.3e} "
              f"winding={a['winding']:+.3f} "
              f"corr={max(a['pattern_corr_cos2'], a['pattern_corr_sin2']):.4f} "
              f"iso={a['isotropy']:.4f} peak_n={peak_n}")

    logM = np.log(np.array(machs))
    raw = np.polyfit(logM, np.log(np.array(amps)), 1)[0]
    norm = np.polyfit(logM, np.log(np.array(amps) / (C0 ** 2 * np.array(machs))), 1)[0]
    out["scaling"] = {
        "raw_exponent": float(raw),
        "mach_normalised_exponent": float(norm),
        "raw_target": 4.0,
        "mach_normalised_target": 3.0,
        "raw_rel_err": float(abs(raw - 4.0) / 4.0),
        "mach_normalised_rel_err": float(abs(norm - 3.0) / 3.0),
    }
    print(f"   scaling: raw p ~ M^{raw:.3f} (theory 4 at fixed r/lambda), "
          f"Mach-normalised ~ M^{norm:.3f} (theory 3)")

    base = None
    for dm, rr in ((1.0, 12.0), (0.5, 16.0), (0.5, 20.0), (0.25, 16.0)):
        res = vortex_pair_farfield(0.05, mask_radius=rr, dx=dm, n_t=48, n_obs=48)
        ampv = res["analysis"]["amplitude"]
        if base is None:
            base = ampv
        out["convergence"].append({
            "dx": dm, "mask_radius": rr, "amplitude": ampv,
            "rel_change_vs_first": float((ampv - base) / base),
            "nsrc": res["nsrc"],
        })
        print(f"   dx={dm} mask={rr}: |p|={ampv:.4e} (rel change {100*(ampv-base)/base:+.3f} %)")

    out["min_pattern_corr"] = float(min(
        max(r["pattern_corr_cos2"], r["pattern_corr_sin2"]) for r in out["runs"]))
    out["max_rms_isotropy"] = float(max(r["rms_isotropy"] for r in out["runs"]))
    out["min_winding_abs"] = float(min(abs(r["winding"]) for r in out["runs"]))
    out["max_peak_freq_dev"] = float(max(abs(r["peak_freq_over_2f_rot"] - 1.0)
                                         for r in out["runs"]))
    out["max_fundamental_ratio"] = float(max(r["harmonic_ratio"]["1Omega_over_2Omega"]
                                             for r in out["runs"]))
    out["max_3omega_ratio"] = float(max(r["harmonic_ratio"]["3Omega_over_2Omega"]
                                        for r in out["runs"]))
    out["thresholds"] = {
        "peak_freq_dev_vs_2f_rot": 0.02,
        "pattern_corr_vs_abs_cos2theta": 0.95,
        "winding_abs": 0.05,
        "rms_isotropy": 0.02,
        "mach_normalised_exponent_rel_err": 0.10,
        "raw_exponent_rel_err": 0.10,
    }
    out["checks"] = {
        "frequency_law": out["max_peak_freq_dev"] <= out["thresholds"]["peak_freq_dev_vs_2f_rot"],
        "four_lobe_directivity": out["min_pattern_corr"] >= out["thresholds"]["pattern_corr_vs_abs_cos2theta"],
        "winding_number_2": abs(out["min_winding_abs"] - 2.0) <= out["thresholds"]["winding_abs"],
        "rms_isotropic": out["max_rms_isotropy"] <= out["thresholds"]["rms_isotropy"],
        "amplitude_M3_mach_normalised": (out["scaling"]["mach_normalised_rel_err"]
                                         <= out["thresholds"]["mach_normalised_exponent_rel_err"]),
        "amplitude_M4_raw_at_fixed_r_over_lambda": (out["scaling"]["raw_rel_err"]
                                                    <= out["thresholds"]["raw_exponent_rel_err"]),
    }
    out["verdict"] = "PASS" if all(out["checks"].values()) else "FAIL"
    print("   checks:", out["checks"])

    # ---- figures -------------------------------------------------------
    M = 0.05
    gamma = M * C0 * 4.0 * math.pi * 6.0
    x = np.arange(-20, 20 + 1e-12, 0.25)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    ux, uy = co_rotating_vortex_pair(xx, yy, gamma, 6.0, 2.5, 0.0)
    omega_z = np.gradient(uy, x, axis=1) - np.gradient(ux, x, axis=0)
    T = compute_lighthill_tensor(np.ones_like(xx), np.stack([ux, uy], axis=0), c0=C0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im0 = axes[0].pcolormesh(xx, yy, omega_z, shading="auto", cmap="RdBu_r")
    axes[0].set_title(r"(C1a) vorticity $\omega_z$ of the rotating pair (M=0.05)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(xx, yy, T[..., 0, 0] * 1e6, shading="auto", cmap="inferno")
    axes[1].set_title(r"(C1b) Lighthill source $T_{xx}\times 10^{6}$")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
    _save(fig, "C1_source_field.png",
          "C1: prescribed co-rotating vortex-pair source field at t=0 (vorticity and "
          "Lighthill T_xx, 2-D cloud maps).")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), subplot_kw={"projection": "polar"})
    for ax, M in zip(axes, (0.05, 0.10)):
        res = vortex_pair_farfield(M, n_obs=96, n_t=96)
        an = res["analysis"]
        th = res["th"]
        pat = np.abs(an["pattern"])
        ax.plot(th, pat / pat.max(), "-", label="phase-locked $|p'(\\theta,t_0)|$")
        ax.plot(th, np.abs(np.cos(2 * th)), "--", label=r"$|\cos 2\theta|$")
        ax.set_title("M=%s: corr=%.4f" % (M, max(an["pattern_corr_cos2"],
                                                   an["pattern_corr_sin2"])), fontsize=9)
        ax.legend(loc="lower left", fontsize=6)
    _save(fig, "C2_directivity.png",
          "C2: phase-locked far-field directivity of the rotating quadrupole vs the ideal "
          "four-lobe |cos 2theta| (polar plots, M=0.05 and 0.10).")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.loglog(machs, amps, "o-", label="numerical (r/lambda=10)")
    ax.loglog(machs, amps[1] * (np.array(machs) / machs[1]) ** 4.0, "k:", label=r"$M^4$")
    ax.set_xlabel("acoustic Mach number M")
    ax.set_ylabel(r"$|\hat p|$")
    ax.set_title("(C3a) raw amplitude vs M")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax = axes[1]
    norm_amps = np.array(amps) / (C0 ** 2 * np.array(machs))
    ax.loglog(machs, norm_amps, "s-", label="numerical")
    ax.loglog(machs, norm_amps[1] * (np.array(machs) / machs[1]) ** 3.0, "k:", label=r"$M^3$")
    ax.set_xlabel("acoustic Mach number M")
    ax.set_ylabel(r"$|\hat p|/(\rho_0 c_0^2 M)$")
    ax.set_title("(C3b) Mach-normalised amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "C3_amplitude_law.png",
          "C3: amplitude scaling of the vortex-pair quadrupole measured at fixed r/lambda "
          "(raw M^4, Mach-normalised M^3).")
    return out


# ---------------------------------------------------------------------------
# D. Kirchhoff surface integral vs volume integral
# ---------------------------------------------------------------------------

def criterion_d() -> Dict[str, object]:
    print("[D] Kirchhoff (permeable surface) vs volume integral")
    omega = 0.3
    k = omega / C0
    lam = 2 * math.pi / k
    sigma = lam / 20.0
    g2 = _gaussian_source_2d(omega, sigma, extent_sigma=4.0, dx_over_sigma=0.25)
    T = g2["T"].reshape(-1, 2, 2)
    src, w = g2["src"], g2["w"]

    def volume(obs):
        return volume_quadrupole_frequency_domain(T, src, w, np.atleast_2d(obs), omega, C0,
                                                  ndim=2, batch_size=16)

    r_surface = 1.0 * lam
    out: Dict[str, object] = {"surface_rows": [], "r_surface_over_lambda": 1.0}
    for n_surf in (128, 256, 512):
        ths = np.linspace(0, 2 * np.pi, n_surf, endpoint=False)
        S = np.stack([r_surface * np.cos(ths), r_surface * np.sin(ths)], axis=-1)
        nrm = S / r_surface
        dr = lam / 400.0
        p_s = volume(S)
        dp_dn = (volume(S + dr * nrm) - volume(S - dr * nrm)) / (2 * dr)
        ds = 2 * np.pi * r_surface / n_surf
        angles = np.array([0.0, 0.3, 1.0, 1.3])
        row = {"n_surface": n_surf, "rows": []}
        for rol in (2.0, 3.0, 5.0, 10.0, 20.0):
            r = rol * lam
            obs = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=-1)
            p_vol = volume(obs)
            p_kir = kirchhoff_surface_2d(p_s, dp_dn, S, nrm, np.full(n_surf, ds), obs, omega, C0)
            d_spl = 20 * np.log10(np.abs(p_kir) / np.abs(p_vol))
            row["rows"].append({
                "r_over_lambda": rol,
                "max_abs_dSPL": float(np.abs(d_spl).max()),
                "mean_abs_dSPL": float(np.abs(d_spl).mean()),
                "amp_ratio_mean": float(np.mean(np.abs(p_kir) / np.abs(p_vol))),
            })
        out["surface_rows"].append(row)
        print(f"   n_surface={n_surf}: max |dSPL| over r/lambda>=2 = "
              f"{max(rr['max_abs_dSPL'] for rr in row['rows']):.3f} dB")

    best = min(max(rr["max_abs_dSPL"] for rr in r["rows"]) for r in out["surface_rows"])
    out["max_abs_dSPL_r_ge_2"] = float(best)
    out["threshold_dB"] = 3.0
    out["verdict"] = "PASS" if best <= 3.0 else "FAIL"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    row = out["surface_rows"][-1]
    axes[0].plot([r["r_over_lambda"] for r in row["rows"]],
                 [r["max_abs_dSPL"] for r in row["rows"]], "o-")
    axes[0].axhline(3.0, color="r", ls="--", label="acceptance 3 dB")
    axes[0].set_xlabel(r"$r/\lambda$")
    axes[0].set_ylabel(r"max $|\Delta$SPL$|$ (dB)")
    axes[0].set_title("(D1) Kirchhoff vs volume (n_surf=512)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].semilogy([r["n_surface"] for r in out["surface_rows"]],
                     [max(rr["max_abs_dSPL"] for rr in r["rows"]) for r in out["surface_rows"]],
                     "s-")
    axes[1].set_xlabel("surface samples")
    axes[1].set_ylabel(r"max $|\Delta$SPL$|$ (dB)")
    axes[1].set_title("(D2) surface-quadrature convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    _save(fig, "D_kirchhoff_vs_volume.png",
          "D: Kirchhoff permeable-surface integral vs direct volume integral for the compact "
          "quadrupole source (r/lambda >= 2).")
    return out


# ---------------------------------------------------------------------------
# E. Compact limit / closed-form cross-check
# ---------------------------------------------------------------------------

def criterion_e() -> Dict[str, object]:
    print("[E] compact limit and closed-form power cross-check")
    omega = 0.3
    k = omega / C0
    lam = 2 * math.pi / k
    out: Dict[str, object] = {"rows": [], "power": {}}
    ang = np.deg2rad(np.array([0.0, 15.0, 30.0, 60.0, 75.0, 90.0]))
    obs = 500.0 * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    for s_lam in (0.01, 0.02, 0.05):
        sigma = s_lam * lam
        p_exact = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
        p_point = analytical_gaussian_quadrupole_farfield(obs, omega, C0, 1.0, sigma,
                                                           (0.0, 0.0), "xx-yy")
        dev = np.abs(np.abs(p_exact) - np.abs(p_point)) / np.abs(p_point)
        out["rows"].append({
            "sigma_over_lambda": s_lam,
            "finite_size_factor": float(math.exp(-(k * sigma) ** 2 / 4.0)),
            "theory_amplitude_defect": float(1.0 - math.exp(-(k * sigma) ** 2 / 4.0)),
            "max_rel_dev_exact_vs_point": float(dev.max()),
        })
        print(f"   sigma/lam={s_lam}: exact vs point-quadrupole dev {dev.max():.4f} "
              f"(1-exp(-k^2 sigma^2/4) = {1-math.exp(-(k*sigma)**2/4):.4f})")

    sigma = 0.005 * lam  # finite-size factor exp(-k^2 sigma^2/4) = 1 - 2e-5
    Q = np.array([[1.0, 0.0], [0.0, -1.0]]) * math.pi * sigma ** 2
    r = 4000.0
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    obs2 = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    p2 = gaussian_quadrupole_farfield_exact(obs2, omega, C0, 1.0, sigma)
    P_num_2d = float(np.sum(np.abs(p2) ** 2 / (2 * C0) * r * (2 * np.pi / th.size)))
    P_cf_2d = compact_quadrupole_power(Q, omega, C0, 1.0, 2)
    out["power"]["2d"] = {"closed_form": P_cf_2d, "numerical_intensity": P_num_2d,
                          "rel_dev": float(abs(P_num_2d - P_cf_2d) / P_cf_2d)}
    print(f"   2-D power: closed form {P_cf_2d:.4e} vs numerical {P_num_2d:.4e} "
          f"(dev {abs(P_num_2d-P_cf_2d)/P_cf_2d:.5f})")

    omega3 = 0.8
    k3 = omega3 / C0
    sigma3 = 0.005 * (2 * math.pi / k3)
    Q3 = np.diag([1.0, -1.0, 0.0]) * math.pi ** 1.5 * sigma3 ** 3
    r3 = 4000.0
    n_theta, n_phi = 128, 64
    th_g = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / (2 * n_theta)
    ph_g = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TG, PG = np.meshgrid(th_g, ph_g, indexing="ij")
    obs3 = np.stack([r3 * np.sin(TG) * np.cos(PG), r3 * np.sin(TG) * np.sin(PG),
                     r3 * np.cos(TG)], axis=-1).reshape(-1, 3)
    p3 = gaussian_quadrupole_farfield_3d(obs3, omega3, C0, 1.0, sigma3)
    dOmega = np.sin(TG) * (np.pi / n_theta) * (2 * np.pi / n_phi)  # solid angle
    P_num_3d = float(np.sum(np.abs(p3) ** 2 / (2 * C0) * r3 ** 2 * dOmega.ravel()))
    P_cf_3d = compact_quadrupole_power(Q3, omega3, C0, 1.0, 3)
    out["power"]["3d"] = {"closed_form": P_cf_3d, "numerical_intensity": P_num_3d,
                          "rel_dev": float(abs(P_num_3d - P_cf_3d) / P_cf_3d)}
    print(f"   3-D power: closed form {P_cf_3d:.4e} vs numerical {P_num_3d:.4e} "
          f"(dev {abs(P_num_3d-P_cf_3d)/P_cf_3d:.5f})")

    out["max_rel_dev"] = float(max(r["max_rel_dev_exact_vs_point"] for r in out["rows"]
                                   if r["sigma_over_lambda"] <= 0.02))
    out["power_max_rel_dev"] = float(max(out["power"]["2d"]["rel_dev"],
                                         out["power"]["3d"]["rel_dev"]))
    out["threshold"] = 0.02
    out["verdict"] = ("PASS" if out["max_rel_dev"] <= 0.02
                      and out["power_max_rel_dev"] <= 0.02 else "FAIL")

    fig, ax = plt.subplots(figsize=(5.6, 4))
    sl = [r["sigma_over_lambda"] for r in out["rows"]]
    ax.loglog(sl, [r["max_rel_dev_exact_vs_point"] for r in out["rows"]], "o-",
              label="exact vs point-quadrupole")
    ax.loglog(sl, [max(r["theory_amplitude_defect"], 1e-6) for r in out["rows"]], "k:",
              label=r"$1-e^{-k^2\sigma^2/4}$")
    ax.axhline(0.02, color="r", ls="--")
    ax.set_xlabel(r"$\sigma/\lambda$")
    ax.set_ylabel("relative deviation")
    ax.set_title("(E) compact limit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "E_compact_limit.png",
          "E: compact-limit cross-check - exact finite-size far field vs the point-quadrupole "
          "closed form, and the analytical finite-size factor.")
    return out


def main() -> None:
    t_start = time.time()
    RESULTS["mission"] = "P170W2B-N03"
    RESULTS["node"] = "TensorLBM FWH chain: Lighthill volume quadrupole maturation"
    RESULTS["c0"] = C0
    RESULTS["python"] = sys.version.split()[0]
    RESULTS["numpy"] = np.__version__
    RESULTS["platform"] = platform.platform()
    try:
        RESULTS["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        RESULTS["git_commit"] = None
    RESULTS["branch"] = os.environ.get("AGENT_BRANCH", "agent/m9740")

    RESULTS["A_operator_anchor"] = criterion_a()
    RESULTS["B_retarded_time"] = criterion_b()
    RESULTS["C_vortex_pair"] = criterion_c()
    RESULTS["D_surface_vs_volume"] = criterion_d()
    RESULTS["E_compact_limit"] = criterion_e()

    RESULTS["elapsed_s"] = time.time() - t_start
    RESULTS["figures"] = FIG
    verdicts = {k: v["verdict"] for k, v in RESULTS.items()
                if isinstance(v, dict) and "verdict" in v}
    RESULTS["verdicts"] = verdicts
    RESULTS["all_pass"] = all(v == "PASS" for v in verdicts.values())

    path = os.path.join(ART, "lighthill_volume_maturation_results.json")
    with open(path, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nresults -> {path}")
    print("verdicts:", verdicts, "all_pass:", RESULTS["all_pass"],
          f"(elapsed {RESULTS['elapsed_s']:.1f} s)")


if __name__ == "__main__":
    main()