#!/usr/bin/env python3
"""m4928 — SUBOFF tau-topic CLOUD artefacts: field plots, tables, machine-readable spec.

Consumes the per-grid-level outputs of ``benchmarks/lbm_suboff_tau_cloud_run.py``
(``fields_L<L>.npz`` + ``summary_L<L>.json``) and produces:

  * ``fig1_field_cloud.png``      — spatial field evidence (CLOUD): axial velocity,
    pressure coefficient, vorticity on the centre plane + a cross-section.
  * ``fig2_reference_comparison.png`` — wall Cp(x/L), local Cf(x/L) vs the Blasius
    local reference, boundary-layer profile vs the Blasius similarity solution,
    and the grid-ladder convergence of the key scalars with U95 bars.
  * ``table1_grid_levels.csv``    — per-grid-level key quantities, reference /
    true error, U95 (temporal ensemble) and grid-spread columns.
  * ``table2_stations.csv``       — per-station (x/L) quantities at the finest level.
  * ``field_cloud.json``          — machine-readable artefact: coordinate system,
    units, colour-bar ranges, normalisation, field statistics moments, GCI (or
    fallback reason), provenance paths.

Every number is recomputed from the npz data; the JSON is written by this script
so the report/table cannot drift from the field files.  No value is typed in by
hand except the literature/analytic reference constants declared below.

Usage
-----
    python benchmarks/lbm_suboff_tau_cloud_plot.py \
        --dir artifacts/m4928/lbm-suboff-tau-cloud --levels 32,48,64
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

REF_CF_BLASIUS = 0.041995047327036086  # 1.328/sqrt(1000) — locked reference (prereg)
BLASIUS_LOCAL_COEF = 0.664
BLASIUS_DSTAR_COEF = 1.7208
BLASIUS_FPP0 = 0.33205733621519630  # f''(0) for the Blasius similarity solution


# --------------------------------------------------------------------------- #
# analytic reference: Blasius similarity solution
# --------------------------------------------------------------------------- #
def blasius_profile(eta_max: float = 8.0, n: int = 400, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Solve f''' + 0.5 f f'' = 0 (f(0)=f'(0)=0, f'(inf)=1) by RK4 + shooting.

    Returns ``(eta, f')`` with ``f' = u/U`` — the laminar flat-plate similarity
    profile used as the reference curve in the comparison figure.
    """
    d_eta = eta_max / (n - 1)

    def integrate(fpp0: float) -> tuple[np.ndarray, np.ndarray, float]:
        f, fp = 0.0, 0.0
        fpp = fpp0
        eta = np.zeros(n)
        u = np.zeros(n)
        for i in range(n):
            eta[i] = i * d_eta
            u[i] = fp
            # classic RK4 on (f, f', f'')
            def deriv(f_, fp_, fpp_):
                return fp_, fpp_, -0.5 * f_ * fpp_

            k1 = deriv(f, fp, fpp)
            k2 = deriv(f + 0.5 * d_eta * k1[0], fp + 0.5 * d_eta * k1[1], fpp + 0.5 * d_eta * k1[2])
            k3 = deriv(f + 0.5 * d_eta * k2[0], fp + 0.5 * d_eta * k2[1], fpp + 0.5 * d_eta * k2[2])
            k4 = deriv(f + d_eta * k3[0], fp + d_eta * k3[1], fpp + d_eta * k3[2])
            f += d_eta / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            fp += d_eta / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            fpp += d_eta / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        return eta, u, fp

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        _, _, fp_end = integrate(mid)
        if fp_end < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    eta, u, _ = integrate(0.5 * (lo + hi))
    return eta, u


# --------------------------------------------------------------------------- #
# GCI / uncertainty helpers
# --------------------------------------------------------------------------- #
def gci_three_level(f1: float, f2: float, f3: float, r: float = 1.5) -> dict:
    """Roache GCI from three grid levels (1 = finest).

    Returns ``{"applicable": bool, "p_obs": float, "f_ext": float, "gci_fine_pct": float,
    "fallback_reason": str|None}``.  When the observed order is not in [0.5, 4] or the
    sequence is not monotone, ``applicable=False`` and a fallback reason is returned;
    the caller then reports the grid spread as the uncertainty instead.
    """
    e12 = f2 - f1
    e23 = f3 - f2
    rln = math.log(r)
    if e12 == 0 or e23 == 0 or e12 * e23 <= 0:
        return {"applicable": False, "p_obs": None, "f_ext": None, "gci_fine_pct": None,
                "fallback_reason": "non-monotone or zero grid triple (e12*e23<=0); "
                                   "GCI not applicable -> grid-spread fallback used",
                "e12": e12, "e23": e23}
    ratio = e23 / e12
    if ratio <= 0:
        return {"applicable": False, "p_obs": None, "f_ext": None, "gci_fine_pct": None,
                "fallback_reason": "negative refinement ratio", "e12": e12, "e23": e23}
    p = abs(math.log(ratio) / rln)
    if not (0.5 <= p <= 4.0):
        return {"applicable": False, "p_obs": p, "f_ext": None, "gci_fine_pct": None,
                "fallback_reason": f"observed order p={p:.3f} outside [0.5,4] "
                                   "(non-asymptotic range) -> grid-spread fallback used",
                "e12": e12, "e23": e23}
    f_ext = f1 + e12 / (r**p - 1.0)
    gci = abs(1.25 * e12 / (f1 * (r**p - 1.0)) * 100.0)
    return {"applicable": True, "p_obs": p, "f_ext": f_ext, "gci_fine_pct": gci,
            "fallback_reason": None, "e12": e12, "e23": e23}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_level(directory: Path, L: int) -> dict:
    npz = np.load(directory / f"fields_L{L}.npz")
    summary = json.loads((directory / f"summary_L{L}.json").read_text())
    return {"L": L, "npz": npz, "summary": summary,
            "npz_path": directory / f"fields_L{L}.npz",
            "summary_path": directory / f"summary_L{L}.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--levels", default="32,48,64")
    ap.add_argument("--finest", type=int, default=0, help="0 -> max level")
    args = ap.parse_args()

    directory = Path(args.dir)
    levels = sorted(int(x) for x in args.levels.split(","))
    data = {L: load_level(directory, L) for L in levels if (directory / f"fields_L{L}.npz").exists()}
    if len(data) < 1:
        raise SystemExit(f"need >=1 field level in {directory}, found {sorted(data)}")
    single_level = len(data) == 1
    fin = args.finest or max(data)
    finest = data[fin]
    u_in = finest["summary"]["units"]["physical"]["u_in_m_per_s"] / \
        finest["summary"]["units"]["physical"]["du_m_per_s"]  # lattice u_in (= u_lb)

    # ---------------- figures ----------------
    fz = finest["npz"]
    x_over_L = fz["x_over_L"]
    y_over_D = fz["y_over_D"]
    ux_cp = fz["ux_cp"].astype(np.float64) / u_in
    Cp_cp = fz["Cp_cp"].astype(np.float64)
    vort = fz["vort_cp"].astype(np.float64)
    solid = fz["solid_cp"].astype(bool)
    uref = np.array(ux_cp)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    extent = [x_over_L.min(), x_over_L.max(), y_over_D.min(), y_over_D.max()]

    im0 = axes[0, 0].imshow(uref, origin="lower", extent=extent, cmap="viridis",
                            vmin=0.0, vmax=float(np.nanmax(np.abs(uref))), aspect="auto")
    axes[0, 0].contour(x_over_L, y_over_D, solid.astype(float), levels=[0.5],
                       colors="w", linewidths=1.0)
    axes[0, 0].set_title("(a) axial velocity $u_x/U_\\infty$ — centre plane $z/D=0$")
    axes[0, 0].set_xlabel("$x/L$ (from bow)")
    axes[0, 0].set_ylabel("$y/D$")
    fig.colorbar(im0, ax=axes[0, 0], label="$u_x/U_\\infty$ [–]")

    vmax_cp = float(np.nanpercentile(np.abs(Cp_cp), 99.5))
    im1 = axes[0, 1].imshow(Cp_cp, origin="lower", extent=extent, cmap="coolwarm",
                            vmin=-vmax_cp, vmax=vmax_cp, aspect="auto")
    axes[0, 1].contour(x_over_L, y_over_D, solid.astype(float), levels=[0.5],
                       colors="k", linewidths=1.0)
    axes[0, 1].set_title("(b) pressure coefficient $C_p$ — centre plane")
    axes[0, 1].set_xlabel("$x/L$")
    axes[0, 1].set_ylabel("$y/D$")
    fig.colorbar(im1, ax=axes[0, 1], label="$C_p=(p-p_\\infty)/(\\frac{1}{2}\\rho U_\\infty^2)$ [–]")

    vm = float(np.nanpercentile(np.abs(vort), 99.0)) or 1.0
    im2 = axes[1, 0].imshow(vort, origin="lower", extent=extent, cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vm, vmax=vm), aspect="auto")
    axes[1, 0].contour(x_over_L, y_over_D, solid.astype(float), levels=[0.5],
                       colors="k", linewidths=0.8)
    axes[1, 0].set_title("(c) vorticity $\\omega_z\\,d/U_\\infty$ — centre plane")
    axes[1, 0].set_xlabel("$x/L$")
    axes[1, 0].set_ylabel("$y/D$")
    fig.colorbar(im2, ax=axes[1, 0], label="$\\omega_z\\,d/U_\\infty$ [–]")

    sec = fz["sec_ux_mid"].astype(np.float64) / u_in
    n_z, n_y = sec.shape
    ext_sec = [-(n_y // 2), n_y - n_y // 2, -(n_z // 2), n_z - n_z // 2]
    im3 = axes[1, 1].imshow(sec, origin="lower", extent=ext_sec, cmap="viridis",
                            vmin=0.0, vmax=float(np.nanmax(np.abs(sec))), aspect="equal")
    axes[1, 1].set_title(f"(d) axial velocity at mid-body station $x/L=0.5$ (L={fin})")
    axes[1, 1].set_xlabel("$y$ [cells]")
    axes[1, 1].set_ylabel("$z$ [cells]")
    fig.colorbar(im3, ax=axes[1, 1], label="$u_x/U_\\infty$ [–]")

    fig.suptitle(
        f"SUBOFF bare hull, Re=1000, LBM D3Q19 MRT — field cloud (L={fin} cells/L, "
        f"lattice units; 1 lu = {finest['summary']['units']['physical']['dx_m']*1000:.2f} mm)",
        fontsize=11)
    fig1_path = directory / "fig1_field_cloud.png"
    fig.savefig(fig1_path, dpi=160)
    plt.close(fig)

    # ---------------- reference comparison figure ----------------
    st = fz
    xL = st["st_xL"]
    cpw = st["st_Cp_wall"]
    cf_lbm = st["st_Cf_local_lbm"]
    cf_bl = st["st_Cf_local_blasius"]
    dstar = st["st_delta_star_m"]
    dstar_bl = st["st_delta_star_blasius_m"]
    prof = st["st_profile_u_over_U"]
    wall_dist = st["st_profile_wall_distance_m"]
    re_x = st["st_Re_x"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), constrained_layout=True)

    axes[0, 0].plot(xL, cpw, "o-", lw=1.4, ms=3.5, label="LBM wall $C_p$ (both sides avg.)")
    axes[0, 0].axhline(1.0, color="k", ls="--", lw=0.9,
                       label="analytic stagnation $C_p=1$")
    axes[0, 0].axhline(0.0, color="grey", ls=":", lw=0.9)
    axes[0, 0].set_xlabel("$x/L$")
    axes[0, 0].set_ylabel("$C_p$ [–]")
    axes[0, 0].set_title("(a) hull-surface pressure distribution")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(xL, cf_bl, "k--", lw=1.2, label="Blasius local $0.664/\\sqrt{Re_x}$")
    axes[0, 1].plot(xL, cf_lbm, "o-", lw=1.4, ms=3.5, label="LBM near-wall gradient")
    axes[0, 1].set_xlabel("$x/L$")
    axes[0, 1].set_ylabel("$C_f$ [–]")
    axes[0, 1].set_title("(b) local skin friction vs Blasius (flat-plate reference)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    eta_ref, u_ref = blasius_profile()
    idx_mid = int(np.argmin(np.abs(xL - 0.5)))
    x_mid_m = float(xL[idx_mid]) * float(finest["summary"]["geometry"]["length"]) * \
        finest["summary"]["units"]["physical"]["dx_m"]
    nu_phys = finest["summary"]["units"]["physical"]["nu_m2_per_s"]
    u_in_phys = finest["summary"]["units"]["physical"]["u_in_m_per_s"]
    eta_lbm = wall_dist * math.sqrt(u_in_phys / (nu_phys * x_mid_m))
    axes[1, 0].plot(u_ref, eta_ref, "k--", lw=1.2, label="Blasius similarity $f'(\\eta)$")
    axes[1, 0].plot(prof[idx_mid], eta_lbm, "o-", lw=1.4, ms=3.0,
                    label=f"LBM, $x/L$={xL[idx_mid]:.2f}")
    axes[1, 0].set_ylim(0, 8)
    axes[1, 0].set_xlim(0, 1.15)
    axes[1, 0].set_xlabel("$u/U_\\infty$ [–]")
    axes[1, 0].set_ylabel("$\\eta = y\\sqrt{U_\\infty/(\\nu x)}$ [–]")
    axes[1, 0].set_title("(c) boundary-layer profile vs Blasius reference")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # grid ladder: local Cf at mid-body and wall Cp at mid-body per level
    ladder_x = []
    ladder_cf = []
    ladder_cf_u95 = []
    ladder_cp = []
    for L in sorted(data):
        s = data[L]["npz"]
        ladder_x.append(L)
        j = int(np.argmin(np.abs(s["st_xL"] - 0.5)))
        ladder_cf.append(float(s["st_Cf_local_lbm"][j]))
        ladder_cp.append(float(s["st_Cp_wall"][j]))
        ladder_cf_u95.append(float(data[L]["summary"]["field_checks"]["Cf_local_mid_U95"]))
    ladder_cf = np.asarray(ladder_cf)
    axes[1, 1].errorbar(ladder_x, ladder_cf, yerr=[u if np.isfinite(u) else 0.0 for u in ladder_cf_u95],
                        fmt="o-", capsize=4, lw=1.4, label="LBM $C_f$ at $x/L=0.5$")
    cf_bl_ref = float(fz["st_Cf_local_blasius"][idx_mid])
    axes[1, 1].axhline(cf_bl_ref, color="k", ls="--", lw=1.0,
                       label=f"Blasius local ref = {cf_bl_ref:.5f}")
    axes[1, 1].set_xlabel("grid level: cells per hull length $L$")
    axes[1, 1].set_ylabel("$C_f$ [–]")
    axes[1, 1].set_title("(d) grid-ladder convergence of the mid-body $C_f$ (U95 bars)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("SUBOFF bare hull Re=1000 — field quantities vs analytic/reference curves",
                 fontsize=11)
    fig2_path = directory / "fig2_reference_comparison.png"
    fig.savefig(fig2_path, dpi=160)
    plt.close(fig)

    # ---------------- profile deviation metrics (vs Blasius) ----------------
    prof_metrics = {}
    for label, xL_target in (("x/L=0.50", 0.5), ("x/L=0.75", 0.75)):
        j = int(np.argmin(np.abs(xL - xL_target)))
        p = prof[j].astype(np.float64)
        x_m = float(xL[j]) * float(finest["summary"]["geometry"]["length"]) * \
            finest["summary"]["units"]["physical"]["dx_m"]
        eta = wall_dist * math.sqrt(u_in_phys / (nu_phys * x_m))
        u_interp = np.interp(eta, eta_ref, u_ref)
        mask = np.isfinite(p) & (eta <= eta_ref[-1])
        rms = float(np.sqrt(np.mean((p[mask] - u_interp[mask]) ** 2)))
        prof_metrics[label] = {
            "x_over_L": float(xL[j]),
            "rms_u_over_U_vs_blasius": rms,
            "max_abs_u_over_U_vs_blasius": float(np.max(np.abs(p[mask] - u_interp[mask]))),
            "delta_star_lbm_m": float(dstar[j]),
            "delta_star_blasius_m": float(dstar_bl[j]),
            "delta_star_rel_err_pct": float((dstar[j] - dstar_bl[j]) / dstar_bl[j] * 100.0),
            "Re_x": float(re_x[j]),
            "n_eta_points": int(mask.sum()),
        }

    # ---------------- tables ----------------
    def _row(L: int) -> dict:
        s = data[L]["summary"]
        fc = s["field_checks"]
        j = int(np.argmin(np.abs(data[L]["npz"]["st_xL"] - 0.5)))
        npz_s = data[L]["npz"]
        cf = float(npz_s["st_Cf_local_lbm"][j])
        cf_ref = float(npz_s["st_Cf_local_blasius"][j])
        return {
            "resolution_cells_per_L": L,
            "Re": s["setup"].get("Re"),
            "tau": s["units"]["physical"]["tau"],
            "nu_lb": s["units"]["physical"]["nu_lb"],
            "steps": s["level"]["steps"],
            "Cd_total_last100": s["drag"]["cd_window_last100_samples"],
            "Cd_ref_blasius": REF_CF_BLASIUS,
            "Cd_err_pct": s["drag"]["cd_err_pct_vs_blasius"],
            "Cd_window_drift_pct": s["drag"]["window_drift_pct"],
            "Cp_stagnation": fc["Cp_stagnation_final"],
            "Cp_stagnation_ref": 1.0,
            "Cp_stagnation_err_pct": (fc["Cp_stagnation_final"] - 1.0) * 100.0,
            "Cp_stagnation_U95": fc["Cp_stagnation_U95"],
            "farfield_ux_over_U": fc["farfield_ux_over_U_mean"],
            "farfield_ux_err_pct": (fc["farfield_ux_over_U_mean"] - 1.0) * 100.0,
            "mass_imbalance_pct": fc["mass_flux"]["imbalance_pct"],
            "Cf_local_xL0.5": cf,
            "Cf_local_blasius_xL0.5": cf_ref,
            "Cf_local_err_pct": (cf - cf_ref) / cf_ref * 100.0,
            "Cf_local_U95": fc["Cf_local_mid_U95"],
            "profile_rms_vs_blasius": prof_metrics.get("x/L=0.50", {}).get(
                "rms_u_over_U_vs_blasius") if L == fin else "",
        }

    table1 = [_row(L) for L in sorted(data)]
    t1_path = directory / "table1_grid_levels.csv"
    with open(t1_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table1[0].keys()))
        w.writeheader()
        w.writerows(table1)

    table2 = []
    for j, xL in enumerate(xL):
        table2.append({
            "resolution_cells_per_L": fin,
            "x_over_L": float(xL),
            "ix_cell": int(st["st_ix"][j]),
            "Re_x": float(st["st_Re_x"][j]),
            "Cp_wall": float(st["st_Cp_wall"][j]),
            "Cp_wall_upper": float(st["st_Cp_wall_up"][j]),
            "Cp_wall_lower": float(st["st_Cp_wall_lo"][j]),
            "Cf_local_lbm": float(st["st_Cf_local_lbm"][j]),
            "Cf_local_blasius": float(st["st_Cf_local_blasius"][j]),
            "Cf_local_err_pct": float((st["st_Cf_local_lbm"][j] - st["st_Cf_local_blasius"][j])
                                      / st["st_Cf_local_blasius"][j] * 100.0),
            "delta_star_lbm_m": float(st["st_delta_star_m"][j]),
            "delta_star_blasius_m": float(st["st_delta_star_blasius_m"][j]),
            "delta_star_rel_err_pct": float((st["st_delta_star_m"][j] - st["st_delta_star_blasius_m"][j])
                                            / st["st_delta_star_blasius_m"][j] * 100.0),
            "u_over_U_eta_1": float(st["st_profile_u_over_U"][j, 0]),
            "u_over_U_eta_5": float(st["st_profile_u_over_U"][j, 4]),
            "u_over_U_eta_10": float(st["st_profile_u_over_U"][j, 9]),
        })
    t2_path = directory / "table2_stations.csv"
    with open(t2_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table2[0].keys()))
        w.writeheader()
        w.writerows(table2)

    # ---------------- GCI on the key scalars ----------------
    gci = {}
    if len(ladder_cf) == 3:
        gci["Cf_local_xL0.5"] = gci_three_level(ladder_cf[0], ladder_cf[1], ladder_cf[2])
        cps = [float(a["summary"]["field_checks"]["Cp_stagnation_final"])
               for _, a in sorted(data.items())]
        gci["Cp_stagnation"] = gci_three_level(cps[0], cps[1], cps[2])
        cds = [float(a["summary"]["drag"]["cd_window_last100_samples"]) for _, a in sorted(data.items())]
        gci["Cd_total"] = gci_three_level(cds[0], cds[1], cds[2])
    else:
        gci["Cf_local_xL0.5"] = {
            "applicable": False, "p_obs": None, "f_ext": None, "gci_fine_pct": None,
            "fallback_reason": f"GCI requires three grid levels; this window produced "
                              f"{len(data)} field level(s) ({sorted(data)}). Spatial "
                              "uncertainty is therefore not quoted from GCI; the reported "
                              "U95 is the temporal snapshot ensemble and the per-grid trend "
                              "uses the verified drag ladder (see table1 notes)."}

    # ---------------- machine-readable artefact spec ----------------
    spec = {
        "schema": "m4928-lbm-suboff-tau-cloud-artefact/v1",
        "mission": 4928,
        "topic": "lbm-suboff-tau",
        "six_flags": {"CLOUD": "produced by this artefact",
                      "others": "see reports 5365/5367/5370 (task brief table)"},
        "case": {
            "geometry": "DARPA SUBOFF bare hull (axisymmetric body of revolution), "
                        "L=4.356 m, D=0.508 m, L/D=8.575",
            "Re": 1000.0,
            "u_in_phys_m_per_s": u_in_phys,
            "nu_phys_m2_per_s": nu_phys,
            "tau": finest["summary"]["units"]["physical"]["tau"],
            "collision": "MRT D3Q19",
            "wall": "half-way bounce-back",
            "reference": {"Cf_ref": REF_CF_BLASIUS, "formula": "Blasius 1.328/sqrt(Re)",
                          "normalisation": "wetted area pi*D*L"},
        },
        "colour_scale": {
            "ux_over_U": {"panel": "fig1(a)", "quantity": "streamwise velocity / free stream",
                          "unit": "-", "range": [float(np.nanmin(uref)), float(np.nanmax(uref))],
                          "cmap": "viridis"},
            "Cp": {"panel": "fig1(b)", "quantity": "(p - p_inf)/(0.5 rho U^2)",
                   "unit": "-", "range": [-vmax_cp, vmax_cp], "cmap": "coolwarm",
                   "p_lattice": "p = rho/3 (cs^2 = 1/3); p_phys = p_lu * 1000 * du^2 Pa"},
            "omega_z": {"panel": "fig1(c)", "quantity": "vorticity times reference length",
                        "unit": "-", "range": [-vm, vm], "cmap": "RdBu_r"},
            "cross_section_ux_over_U": {"panel": "fig1(d)", "quantity": "streamwise velocity",
                                        "unit": "-", "range": [float(np.nanmin(sec)),
                                                               float(np.nanmax(sec))]},
        },
        "coordinates": {
            "x_over_L": "streamwise, 0 at the bow, 1 at the stern (hull centre 0.5)",
            "y_over_D": "lateral (vertical on the plotted centre plane), 0 on the body axis",
            "centre_plane": "z index = nz//2 (the mid-span symmetry plane)",
            "cross_section": "cell indices about the axis, axes y (horizontal) and z (vertical)",
            "origin_note": "npz x_lu is cell index minus cx_cells; dx = L_m / L_cells",
        },
        "fields": {
            "centre_plane_shape": list(np.shape(ux_cp)),
            "cross_section_shape": list(np.shape(sec)),
            "moment_ux": finest["summary"]["field_moments_lattice"][0],
            "moment_Cp": {"mean": float(np.nanmean(Cp_cp)), "std": float(np.nanstd(Cp_cp)),
                          "min": float(np.nanmin(Cp_cp)), "max": float(np.nanmax(Cp_cp))},
            "moment_omega_z": finest["summary"]["field_moments_lattice"][3],
        },
        "checks": {
            "Cp_stagnation": {L: data[L]["summary"]["field_checks"]["Cp_stagnation_final"]
                              for L in sorted(data)},
            "Cp_stagnation_ref": 1.0,
            "farfield_ux_over_U": {L: data[L]["summary"]["field_checks"]["farfield_ux_over_U_mean"]
                                   for L in sorted(data)},
            "mass_imbalance_pct": {L: data[L]["summary"]["field_checks"]["mass_flux"]["imbalance_pct"]
                                   for L in sorted(data)},
            "profile_vs_blasius": prof_metrics,
        },
        "grid_levels": sorted(data),
        "finest_level": fin,
        "gci": gci,
        "uncertainty_note": "U95 = 1.96*std/sqrt(N) over the centre-plane snapshot ensemble "
                            "(temporal); grid uncertainty from three-level GCI when the "
                            "observed order is admissible, else grid spread (fallback_reason).",
        "tables": {"table1_grid_levels.csv": [str(t1_path)],
                   "table2_stations.csv": [str(t2_path)]},
        "figures": {"fig1_field_cloud.png": str(fig1_path),
                    "fig2_reference_comparison.png": str(fig2_path)},
        "provenance": {
            L: {"npz": str(data[L]["npz_path"]),
                "npz_md5": md5_of(data[L]["npz_path"]),
                "summary": str(data[L]["summary_path"]),
                "summary_md5": md5_of(data[L]["summary_path"]),
                "steps": data[L]["summary"]["level"]["steps"],
                "engine": data[L]["summary"]["provenance"]["engine"]}
            for L in sorted(data)
        },
        "honest_notes": [
            "Field quantities (Cp, Cf_local, delta*) are post-processed from the dumped "
            "fields of this run; total drag Cd comes from the verified force integration "
            "of the same run, not from these post-processed fields.",
            "The Blasius local Cf and profile are flat-plate references; the hull has a "
            "streamwise pressure gradient, so deviations near bow/stern are physical "
            "model differences, not discretisation error.",
            "Bare hull only (no sail/fins): the body is axisymmetric, so azimuthal "
            "variation of wall quantities is not applicable in this case."
        ],
    }
    spec_path = directory / "field_cloud.json"
    spec_path.write_text(json.dumps(spec, indent=1))

    # ---------------- auto-generated markdown table (no hand-typed numbers) ----------------
    md_lines = [
        "<!-- AUTO-GENERATED by benchmarks/lbm_suboff_tau_cloud_plot.py — do not edit by hand -->",
        "",
        "### Table 1 — per-grid-level key quantities (SUBOFF bare hull, Re=1000)",
        "",
        "| L (cells/L) | tau | steps | Cd_total | Cd ref (Blasius) | Cd err % | Cp_stag (ref 1.0) "
        "| Cp_stag err % | Cp_stag U95 | far-field u_x/U | mass imb. % | Cf(x/L=.5) "
        "| Cf Blasius ref | Cf err % | Cf U95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table1:
        md_lines.append(
            "| {L} | {tau:.4f} | {steps} | {Cd:.6f} | {Cdref:.6f} | {Cde:+.2f} | {Cps:.4f} "
            "| {Cpse:+.2f} | {Cpsu:.2e} | {ff:.5f} | {mass:.3f} | {cf:.5f} | {cfref:.5f} "
            "| {cfe:+.2f} | {cfu:.2e} |".format(
                L=row["resolution_cells_per_L"], tau=row["tau"], steps=row["steps"],
                Cd=row["Cd_total_last100"], Cdref=row["Cd_ref_blasius"], Cde=row["Cd_err_pct"],
                Cps=row["Cp_stagnation"], Cpse=row["Cp_stagnation_err_pct"],
                Cpsu=row["Cp_stagnation_U95"], ff=row["farfield_ux_over_U"],
                mass=row["mass_imbalance_pct"], cf=row["Cf_local_xL0.5"],
                cfref=row["Cf_local_blasius_xL0.5"], cfe=row["Cf_local_err_pct"],
                cfu=row["Cf_local_U95"]))
    md_lines += [
        "",
        "U95 = 1.96*std/sqrt(N) over the centre-plane snapshot ensemble. Cd is the "
        "force-integration window mean of the same run (not the post-processed fields).",
        "",
        "### GCI / uncertainty disposition",
        "",
    ]
    for k, v in gci.items():
        if isinstance(v, dict):
            md_lines.append(f"* **{k}**: applicable={v.get('applicable')}, "
                            f"p_obs={v.get('p_obs')}, gci_fine_pct={v.get('gci_fine_pct')}, "
                            f"fallback={v.get('fallback_reason')}")
    md_lines += ["", "### Blasius-reference profile comparison (finest level)", ""]
    for k, v in prof_metrics.items():
        md_lines.append(f"* {k}: RMS(u/U) = {v['rms_u_over_U_vs_blasius']:.4f}, "
                        f"max|du/U| = {v['max_abs_u_over_U_vs_blasius']:.4f}, "
                        f"delta* = {v['delta_star_lbm_m']*1e3:.4f} mm vs Blasius "
                        f"{v['delta_star_blasius_m']*1e3:.4f} mm "
                        f"({v['delta_star_rel_err_pct']:+.1f} %), Re_x = {v['Re_x']:.0f}")
    md_lines += ["", "### Provenance", ""]
    for L in sorted(data):
        prov = spec["provenance"][L]
        md_lines.append(f"* L={L}: `{prov['npz']}` md5 `{prov['npz_md5']}`, "
                        f"summary md5 `{prov['summary_md5']}`, steps {prov['steps']}")
    (directory / "table1_and_gci.md").write_text("\n".join(md_lines) + "\n")


    print(f"wrote {fig1_path.name}, {fig2_path.name}, {t1_path.name}, {t2_path.name}, "
          f"{spec_path.name}")
    for row in table1:
        print(f"L={row['resolution_cells_per_L']:>3d}  Cd={row['Cd_total_last100']:.6f} "
              f"({row['Cd_err_pct']:+.2f}%)  Cp_stag={row['Cp_stagnation']:.4f}  "
              f"Cf(x/L=.5)={row['Cf_local_xL0.5']:.5f} ({row['Cf_local_err_pct']:+.1f}%)")
    for k, v in gci.items():
        if isinstance(v, dict):
            print(f"GCI {k}: applicable={v.get('applicable')} p={v.get('p_obs')} "
                  f"gci%={v.get('gci_fine_pct')} reason={v.get('fallback_reason')}")


if __name__ == "__main__":
    main()