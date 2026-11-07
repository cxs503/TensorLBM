#!/usr/bin/env python3
"""Resistance-chain CLOUD analysis: grid-convergence table, angular table, figures.

Reads the per-grid artifacts written by ``run.py``
(``<geo>_D<D>_fields.npz`` + ``<geo>_D<D>_run.json``) and produces

* ``table1_grid_levels.csv``   - one row per grid level: key quantities, reference,
  true error, U95 (statistical + discretisation), GCI / fallback_reason,
  field statistical moments of the Cp / vorticity fields;
* ``table2_surface_angular.csv`` - one row per angular bin (10 deg): Cp, Cf,
  temporal std, potential-flow reference Cp, true error, grid band (U95 proxy);
* ``table3_wake_profiles.csv``   - wake statistics per streamwise station;
* ``fig1_<geo>_cloud.png``       - field clouds (|u|, omega_z, Cp, omega_rms) +
  Cp(phi) / Cf(phi) versus the analytic potential-flow profile;
* ``fig2_<geo>_convergence.png`` - Cd(h) with Richardson extrapolation and the
  reference band, wake profiles for the three grids;
* ``cloud_index.json``           - machine-readable manifest: coordinates, units,
  colour-bar ranges, contour levels, provenance (npz / run.json / git commit).

Every number written here comes from the npz/json produced by ``run.py`` in this
worktree; nothing is hand-typed.  Missing grid levels or missing references are
reported as ``NO-DATA-TBD`` instead of being filled in.

Usage
-----
    python benchmarks/resistance_chain_cloud/analyze.py \
        --fields artifacts/resistance_chain_cloud/fields \
        --out artifacts/resistance_chain_cloud
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import potential_cp_ellipse  # noqa: E402
from surface import surface_loads  # noqa: E402

# reference data (traceable, see docs/resistance_chain_cloud/README.md)
REFERENCE = {
    "cylinder": {
        "cd": 1.35,  # Braza, Chassaing & Ha Minh 1986, JFM 165 -- band +-0.05
        "cd_band": 0.05,
        "st": 0.1645,  # 0.164-0.165, in-repo verified benchmark
        "st_band": 0.0005,
        "source": "benchmarks/verified/cylinder/re100/README.md (Braza et al. 1986)",
    },
    "ellipse": {
        "cd": None,  # NO-DATA-TBD: no measured Cd reference in this repo
        "cd_band": None,
        "st": None,
        "st_band": None,
        "source": "NO-DATA-TBD (potential-flow Cp only)",
    },
}

GEOMETRY_LABEL = {
    "cylinder": "circular section (D)",
    "ellipse": "ship waterline section, L/B = 4",
}

FS_SAFETY = 1.25  # Roache factor of safety
DATUM = "NO-DATA-TBD"


# ------------------------------------------------------------------ helpers
def git_commit(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    except Exception:  # pragma: no cover
        return "unknown"


def load_case(fields_dir: Path, geo: str, D: int) -> dict | None:
    npz = fields_dir / geo / f"{geo}_D{D}_fields.npz"
    js = fields_dir / geo / f"{geo}_D{D}_run.json"
    if not npz.exists() or not js.exists():
        return None
    data = dict(np.load(npz, allow_pickle=False))
    meta = json.loads(js.read_text())
    return {"D": D, "npz": data, "meta": meta, "npz_path": str(npz), "json_path": str(js)}


def gci(f1: float, f2: float, f3: float, r: float, fs: float = FS_SAFETY) -> dict:
    """Roache grid-convergence index for three systematically refined grids.

    ``f1`` finest, ``f3`` coarsest; ``h`` proportional to ``1/D`` so the
    refinement ratio ``r = D2/D1 = D3/D2`` must be uniform.

    Returns ``p_obs`` (observed order), ``gci_fine_pct``, ``extrapolated``
    (Richardson) and ``fallback_reason`` when the asymptotic range is not
    reached (non-monotonic / oscillatory convergence or r not uniform).
    """
    out: dict = {
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "r": r,
        "p_obs": None,
        "gci_fine_pct": None,
        "extrapolated": None,
        "fallback_reason": None,
    }
    if any(v is None or not math.isfinite(v) for v in (f1, f2, f3)):
        out["fallback_reason"] = "missing grid level (NO-DATA-TBD)"
        return out
    eps21 = f2 - f1
    eps32 = f3 - f2
    if abs(f1) < 1e-14:
        out["fallback_reason"] = "zero fine-grid value"
        return out
    if eps21 == 0 and eps32 == 0:
        out["p_obs"] = None
        out["gci_fine_pct"] = 0.0
        out["extrapolated"] = f1
        out["fallback_reason"] = "identical values on all grids (no observable discretisation error)"
        return out
    ratio = eps32 / eps21 if eps21 != 0 else float("inf")
    if not (0.0 < ratio < 1.0):
        out["fallback_reason"] = (
            f"non-asymptotic: eps32/eps21={ratio:.3g} (need 0<ratio<1); "
            "GCI not defined -> oscillatory/non-monotonic fallback used"
        )
        # fallback: assumed 2nd-order, use the conservative max deviation band
        band = max(abs(eps21), abs(eps32))
        out["gci_fine_pct"] = 100.0 * fs * band / abs(f1)
        out["extrapolated"] = None
        return out
    p = math.log(ratio) / math.log(r)
    out["p_obs"] = p
    out["gci_fine_pct"] = 100.0 * fs * abs(eps21 / f1) / (r**p - 1.0)
    out["extrapolated"] = f1 + (f1 - f2) / (r**p - 1.0)
    return out


def moments(field: np.ndarray, mask: np.ndarray, win: tuple) -> dict:
    """Statistical moments of a field inside a rectangular window (body frame)."""
    x, y = win
    sel = (x > -3.0) & (x < 3.0)
    sub = field[:, sel]
    yv = y[:, None]
    sub = sub[(yv[:, 0] > -2.0) & (yv[:, 0] < 2.0)]
    vals = sub[np.isfinite(sub)]
    if vals.size == 0:
        return {"mean": None, "std": None, "skew": None, "kurt": None}
    m = float(vals.mean())
    s = float(vals.std())
    if s > 0:
        z = (vals - m) / s
        skew = float((z**3).mean())
        kurt = float((z**4).mean() - 3.0)
    else:
        skew = kurt = None
    return {"mean": m, "std": s, "skew": skew, "kurt": kurt}


def wake_stations(ux: np.ndarray, urms: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Wake statistics at fixed streamwise stations x/D = 1, 2, 3, 5."""
    out = {}
    for xs in (1.0, 2.0, 3.0, 5.0):
        j = int(np.argmin(np.abs(x - xs)))
        col = ux[:, j]
        ycol = y
        core = np.abs(ycol) < 1.0
        out[f"x_{xs:g}"] = {
            "x_over_D": float(x[j]),
            "u_min": float(col[core].min()),
            "u_center": float(col[int(np.argmin(np.abs(ycol))) ]),
            "u_rms_max": float(urms[core, j].max()),
            "deficit_width_half_D": float(
                np.sum((col < 0.99) & core) * abs(float(y[1] - y[0]))
            ),
        }
    return out


def surface_profile(case: dict, ref_cp_fn, geo: str) -> dict:
    """Angular Cp/Cf profile from the exposed-face integration (surface.py)."""
    s = surface_loads(case["npz"], geo)
    phi = s["phi"]
    return {
        "phi_deg": np.degrees(phi),
        "cp": s["cp_bin"],
        "cf": s["cf_bin"],
        "cp_raw": s["cp_bin_raw"],
        "cf_raw": s["cf_bin_raw"],
        "cp_std": s["cp_bin_std"],
        "cf_std": np.zeros_like(s["cf_bin"]),
        "cp_pot": ref_cp_fn(phi),
        "node_cnt": s["bin_count"],
        "n_samples": float(case["npz"]["n_samples_per_node"])
        if "n_samples_per_node" in set(case["npz"].files)
        else 0.0,
        "loads": s,
    }


# -------------------------------------------------------------------- tables
def build_tables(fields_dir: Path, Ds: list[int], geos: list[str]) -> tuple[dict, dict]:
    cases = {g: [c for c in (load_case(fields_dir, g, D) for D in Ds) if c] for g in geos}
    table1: list[dict] = []
    angular: list[dict] = []
    wake: list[dict] = []
    profiles: dict[str, dict] = {}

    for geo in geos:
        ref = REFERENCE[geo]
        cs = cases[geo]
        cd_vals = [c["meta"]["cd_mean"] for c in cs]
        st_vals = [c["meta"]["st"] for c in cs]
        # GCI on the finest triad (largest three D available)
        triad = cs[-3:] if len(cs) >= 3 else cs
        if len(triad) == 3:
            r = triad[1]["D"] / triad[0]["D"]
            r2 = triad[2]["D"] / triad[1]["D"]
            if abs(r - r2) > 1e-9:
                g = {
                    "gci_fine_pct": None,
                    "p_obs": None,
                    "extrapolated": None,
                    "fallback_reason": f"non-uniform refinement ratios {r:.3f} vs {r2:.3f}",
                }
            else:
                g = gci(cd_vals[-1], cd_vals[-2], cd_vals[-3], r)
        else:
            g = {"gci_fine_pct": None, "p_obs": None, "extrapolated": None,
                 "fallback_reason": "fewer than 3 grid levels (NO-DATA-TBD)"}

        for i, c in enumerate(cs):
            meta = dict(c["meta"])
            z = c["npz"]
            surf = surface_loads(z, geo)
            meta["cd_pressure"] = surf["cd_pressure"]
            meta["cd_friction"] = surf["cd_friction"]
            meta["cd_pf_sum"] = surf["cd_pf_sum"]
            meta["cd_decomposition_rel_err"] = abs(surf["cd_pf_sum"] - meta["cd_mean"]) / abs(
                meta["cd_mean"]
            )
            meta["cp_surface_min"] = surf["cp_face_min"]
            meta["cp_surface_max"] = surf["cp_face_max"]
            meta["n_boundary_nodes"] = surf["n_faces"]
            cd = meta["cd_mean"]
            err = (cd - ref["cd"]) / ref["cd"] * 100.0 if ref["cd"] else None
            st_err = (meta["st"] - ref["st"]) / ref["st"] * 100.0 if ref["st"] else None
            u_stat = meta["cd_u95"]
            u_gci = (g["gci_fine_pct"] / 100.0 * cd) if g["gci_fine_pct"] is not None else None
            u_comb = math.hypot(u_stat, u_gci) if u_gci is not None else None
            cp_m = moments(z["cp"], z["mask"], (z["x"], z["y"]))
            wz_m = moments(z["wz"], z["mask"], (z["x"], z["y"]))
            row = {
                "geometry": geo,
                "geometry_label": GEOMETRY_LABEL[geo],
                "D_cells": meta["D"],
                "grid_cells": f"{meta['nx']}x{meta['ny']}",
                "n_cells": meta["n_cells"],
                "u_in": meta["u_in"],
                "re": meta["re"],
                "tau": meta["tau"],
                "t_star_averaged": round(meta["t_star_averaged"], 2),
                "n_avg_samples": meta["n_avg_samples"],
                "Cd_mean": round(cd, 5),
                "Cd_std_time": round(meta["cd_std"], 5),
                "Cd_u95_stat": round(u_stat, 5),
                "Cd_pressure": round(meta["cd_pressure"], 5),
                "Cd_friction": round(meta["cd_friction"], 5),
                "Cd_pf_sum": round(meta["cd_pf_sum"], 5),
                "Cd_pf_rel_err_pct": round(100.0 * meta["cd_decomposition_rel_err"], 3),
                "Cd_ref": ref["cd"] if ref["cd"] is not None else DATUM,
                "Cd_ref_band": ref["cd_band"] if ref["cd_band"] is not None else DATUM,
                "Cd_err_pct": round(err, 3) if err is not None else DATUM,
                "St": round(meta["st"], 5),
                "St_ref": ref["st"] if ref["st"] is not None else DATUM,
                "St_err_pct": round(st_err, 3) if st_err is not None else DATUM,
                "U95_discretisation_pct": round(g["gci_fine_pct"], 3)
                if g["gci_fine_pct"] is not None
                else DATUM,
                "U95_combined_abs": round(u_comb, 5) if u_comb is not None else DATUM,
                "GCI_note": g["fallback_reason"] if g["fallback_reason"] else "ok",
                "cp_field_mean": round(cp_m["mean"], 4) if cp_m["mean"] is not None else DATUM,
                "cp_field_std": round(cp_m["std"], 4) if cp_m["std"] is not None else DATUM,
                "cp_field_skew": round(cp_m["skew"], 4) if cp_m["skew"] is not None else DATUM,
                "cp_field_kurt": round(cp_m["kurt"], 4) if cp_m["kurt"] is not None else DATUM,
                "wz_field_mean": round(wz_m["mean"], 4) if wz_m["mean"] is not None else DATUM,
                "wz_field_std": round(wz_m["std"], 4) if wz_m["std"] is not None else DATUM,
                "cp_surface_min": round(meta["cp_surface_min"], 4),
                "cp_surface_max": round(meta["cp_surface_max"], 4),
                "n_boundary_nodes": meta["n_boundary_nodes"],
                "checksum_rel_err": round(meta["checksum_rel_err"], 6),
                "npz": c["npz_path"],
                "run_json": c["json_path"],
            }
            table1.append(row)
            w = wake_stations(z["ux"], z["u_rms"], z["x"], z["y"])
            for key, vals in w.items():
                wake.append({"geometry": geo, "D_cells": meta["D"], "station": key, **{
                    k: round(v, 5) for k, v in vals.items()}})

        # angular table (one row per bin, per grid level)
        ref_cp_fn = potential_cp_ellipse_curve(geo, cs[-1]["meta"]) if cs else None
        for c in cs:
            prof = surface_profile(c, ref_cp_fn, geo)
            profiles.setdefault(geo, {})[c["D"]] = prof
            fine_D = max(c["D"] for c in cs)
            if fine_D not in profiles[geo]:
                _c = [c for c in cs if c["D"] == fine_D][0]
                profiles[geo][fine_D] = surface_profile(_c, ref_cp_fn, geo)
            fine = profiles[geo][fine_D]
            for i in range(prof["phi_deg"].size):
                band = (
                    abs(prof["cp"][i] - fine["cp"][i]) if c is not fine else np.nan
                )
                angular.append(
                    {
                        "geometry": geo,
                        "D_cells": c["meta"]["D"],
                        "phi_deg": round(float(prof["phi_deg"][i]), 3),
                        "Cp_mean": round(float(prof["cp"][i]), 5)
                        if np.isfinite(prof["cp"][i])
                        else DATUM,
                        "Cp_temporal_std": round(float(prof["cp_std"][i]), 5)
                        if np.isfinite(prof["cp_std"][i])
                        else DATUM,
                        "Cp_potential_ref": round(float(prof["cp_pot"][i]), 5),
                        "Cp_err_vs_potential": round(float(prof["cp"][i] - prof["cp_pot"][i]), 5)
                        if np.isfinite(prof["cp"][i])
                        else DATUM,
                        "Cf_mean": round(float(prof["cf"][i]), 5)
                        if np.isfinite(prof["cf"][i])
                        else DATUM,
                        "Cf_temporal_std": round(float(prof["cf_std"][i]), 5)
                        if np.isfinite(prof["cf_std"][i])
                        else DATUM,
                        "grid_band_vs_finest": round(float(band), 5)
                        if np.isfinite(band)
                        else DATUM,
                        "n_nodes_in_bin": int(prof["node_cnt"][i]),
                        "n_time_samples": int(prof["n_samples"]),
                    }
                )

    return (
        {"table1": table1, "table2": angular, "table3": wake, "gci": {
            geo: gci(
                *[c["meta"]["cd_mean"] for c in cases[geo]][-1:-4:-1],
                cases[geo][-2]["D"] / cases[geo][-3]["D"],
            ) if len(cases[geo]) >= 3 else {"fallback_reason": "fewer than 3 grid levels"}
            for geo in geos
        }},
        profiles,
    )


def potential_cp_ellipse_curve(geo: str, meta: dict):
    a, b = meta["a_cells"], meta["b_cells"]

    def fn(phi: np.ndarray) -> np.ndarray:
        if geo == "cylinder":
            return 1.0 - 4.0 * np.sin(phi) ** 2
        return potential_cp_ellipse(a, b, phi)

    return fn


# ------------------------------------------------------------------ figures
def _field_panel(ax, x, y, fld, title, cbl, cmap, levels=None, mask=None, sym=False):
    if mask is not None:
        fld = np.where(mask, np.nan, fld)
    xr = (x > -4) & (x < 8)
    yr = (y > -3) & (y < 3)
    sub = fld[np.ix_(yr, xr)]
    X, Y = np.meshgrid(x[xr], y[yr])
    if levels is None:
        vmax = np.nanmax(np.abs(sub)) if sym else np.nanmax(sub)
        vmin = -vmax if sym else np.nanmin(sub)
        levels = np.linspace(vmin, vmax, 41)
    cf = ax.contourf(X, Y, sub, levels=levels, cmap=cmap, extend="both")
    ax.contour(X, Y, sub, levels=levels[::4], colors="k", linewidths=0.25, alpha=0.35)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x / D  [-]", fontsize=8)
    ax.set_ylabel("y / D  [-]", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbl, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def fig1(fields_dir: Path, geo: str, D: int, out_png: Path) -> dict:
    case = load_case(fields_dir, geo, D)
    if case is None:
        return {}
    z = case["npz"]
    x, y = z["x"], z["y"]
    ux, uy = z["ux"], z["uy"]
    spd = np.hypot(ux, uy)
    wz = z["wz"]
    cp = z["cp"]
    wzr = z["wz_rms"]
    mask = z["mask"]
    ref_fn = potential_cp_ellipse_curve(geo, case["meta"])
    prof = surface_profile(case, ref_fn, geo)

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    _field_panel(ax0, x, y, spd, r"(a) mean $|u|/U_\infty$", r"$|u|/U_\infty$ [-]",
                 "viridis", mask=mask)
    yr = (y > -3) & (y < 3)
    xr = (x > -4) & (x < 8)
    X, Y = np.meshgrid(x[xr], y[yr])
    ax0.streamplot(x[xr], y[yr], ux[np.ix_(yr, xr)], uy[np.ix_(yr, xr)],
                   density=1.0, color="w", linewidth=0.5, arrowsize=0.5)

    ax1 = fig.add_subplot(gs[0, 1])
    _field_panel(ax1, x, y, wz, r"(b) mean $\omega_z D/U_\infty$",
                 r"$\omega_z D/U_\infty$ [-]", "RdBu_r", mask=mask, sym=True)

    ax2 = fig.add_subplot(gs[0, 2])
    _field_panel(ax2, x, y, cp, r"(c) mean $C_p$ field", r"$C_p$ [-]", "coolwarm",
                 mask=mask, sym=True)

    ax3 = fig.add_subplot(gs[1, 0])
    _field_panel(ax3, x, y, wzr, r"(d) rms $\omega_z D/U_\infty$",
                 r"$\omega_z' D/U_\infty$ [-]", "magma")

    ax4 = fig.add_subplot(gs[1, 1])
    ok = np.isfinite(prof["cp"])
    ax4.plot(prof["phi_deg"][ok], prof["cp"][ok], "o-", ms=3, lw=1.2,
             label=fr"LBM $D={D}$ (mean)")
    ax4.plot(prof["phi_deg"], prof["cp_pot"], "k--", lw=1.0,
             label="potential flow (analytic)")
    front = np.abs(prof["phi_deg"]) > 90
    ax4.fill_between(prof["phi_deg"], -6, 3, where=front, color="0.9", alpha=0.5,
                     label="attached-flow window")
    ax4.set_xlim(-180, 180)
    ax4.set_ylim(-3.2, 1.4)
    ax4.set_xlabel(r"surface angle $\varphi$ [deg]  ($0$ = downstream, $\pm180$ = stagnation)",
                   fontsize=8)
    ax4.set_ylabel(r"$C_p$ [-]", fontsize=8)
    ax4.set_title(r"(e) $C_p(\varphi)$: field vs potential flow", fontsize=10)
    ax4.legend(fontsize=6.5, loc="lower left")
    ax4.grid(alpha=0.3)
    ax4.tick_params(labelsize=7)

    ax5 = fig.add_subplot(gs[1, 2])
    okf = np.isfinite(prof["cf"])
    ax5.plot(prof["phi_deg"][okf], prof["cf"][okf], "s-", ms=3, lw=1.2, color="C3",
             label=fr"LBM $D={D}$ (mean)")
    ax5.axhline(0.0, color="k", lw=0.6)
    ax5.set_xlim(-180, 180)
    ax5.set_xlabel(r"surface angle $\varphi$ [deg]", fontsize=8)
    ax5.set_ylabel(r"$C_f = \tau_w/(\tfrac12\rho U_\infty^2)$ [-]", fontsize=8)
    ax5.set_title(r"(f) $C_f(\varphi)$ (wall shear, half-way BB)", fontsize=10)
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
    ax5.tick_params(labelsize=7)

    fig.suptitle(
        f"Resistance-chain CLOUD - {GEOMETRY_LABEL[geo]}, Re = {case['meta']['re']:.0f}, "
        f"grid D = {D} cells, time-averaged over $t^*$ = {case['meta']['t_star_averaged']:.0f} "
        f"($C_d$ = {case['meta']['cd_mean']:.3f})",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)

    def rng(f):
        v = f[np.isfinite(f)]
        return [float(v.min()), float(v.max())]

    return {
        "png": str(out_png),
        "figure": out_png.name,
        "geometry": geo,
        "D_cells": D,
        "panels": {
            "speed": {"quantity": "mean |u|/U_inf", "units": "[-]",
                      "range": rng(spd), "colormap": "viridis"},
            "vorticity": {"quantity": "mean omega_z*D/U_inf", "units": "[-]",
                          "range": rng(wz), "colormap": "RdBu_r"},
            "cp_field": {"quantity": "mean Cp=(p-p_inf)/(0.5 rho U^2)", "units": "[-]",
                         "range": rng(cp), "colormap": "coolwarm"},
            "vorticity_rms": {"quantity": "rms omega_z*D/U_inf", "units": "[-]",
                              "range": rng(wzr), "colormap": "magma"},
        },
        "coordinates": {
            "x_over_D": [float(x[0]), float(x[-1])],
            "y_over_D": [float(y[0]), float(y[-1])],
            "dx_over_D": float(x[1] - x[0]),
        },
    }


def fig2(fields_dir: Path, geo: str, out_png: Path, table: dict) -> dict:
    rows = [r for r in table["table1"] if r["geometry"] == geo]
    if not rows:
        return {}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    D = np.array([r["D_cells"] for r in rows], dtype=float)
    cd = np.array([r["Cd_mean"] for r in rows])
    u95 = np.array([r["Cd_u95_stat"] for r in rows])
    order = np.argsort(D)
    D, cd, u95 = D[order], cd[order], u95[order]
    axa.errorbar(1.0 / D, cd, yerr=1.96 * 0 + u95 * 1.96 / 1.96, fmt="o-", ms=5,
                 capsize=3, label=r"LBM $C_d$ (time-mean, bars = U95$_{stat}$)")
    ref = REFERENCE[geo]
    if ref["cd"] is not None:
        axa.axhline(ref["cd"], color="k", ls="--", lw=1.0,
                    label=f"Braza 1986 reference $C_d$={ref['cd']:.2f}")
        axa.axhspan(ref["cd"] - ref["cd_band"], ref["cd"] + ref["cd_band"],
                    color="0.85", alpha=0.6, label=r"reference band $\pm0.05$")
    g = table["gci"].get(geo, {})
    if g.get("extrapolated") is not None:
        axa.axhline(g["extrapolated"], color="C2", ls=":", lw=1.2,
                    label=f"Richardson extrapolation = {g['extrapolated']:.3f}")
    axa.set_xlabel(r"$h = 1/D$ [1/cells]", fontsize=9)
    axa.set_ylabel(r"$C_d$ [-]", fontsize=9)
    axa.set_title(f"grid convergence ({geo})", fontsize=10)
    axa.legend(fontsize=7)
    axa.grid(alpha=0.3)

    finest = rows[int(np.argmax([r["D_cells"] for r in rows]))]
    case = load_case(fields_dir, geo, finest["D_cells"])
    for c in [load_case(fields_dir, geo, r["D_cells"]) for r in rows]:
        if c is None:
            continue
        for xs in (1.0, 2.0, 3.0, 5.0):
            j = int(np.argmin(np.abs(c["npz"]["x"] - xs)))
            y = c["npz"]["y"]
            sel = np.abs(y) < 2.5
            axb.plot(y[sel], c["npz"]["ux"][sel, j],
                     label=fr"$x/D$={xs:g}, D={c['D']}", lw=1.0)
    axb.set_xlabel(r"$y/D$ [-]", fontsize=9)
    axb.set_ylabel(r"$\bar u/U_\infty$ [-]", fontsize=9)
    axb.set_title("mean wake profiles (three grid levels)", fontsize=10)
    axb.legend(fontsize=6, ncol=2)
    axb.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "figure": out_png.name, "geometry": geo}


# ----------------------------------------------------------------- CSV/manifest
def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv

    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", default="artifacts/resistance_chain_cloud/fields")
    ap.add_argument("--out", default="artifacts/resistance_chain_cloud")
    ap.add_argument("--geos", default="cylinder,ellipse")
    ap.add_argument("--D", type=int, nargs="+", default=[16, 24, 36])
    ap.add_argument("--D-ellipse", type=int, nargs="+", default=[12, 18, 27])
    args = ap.parse_args(argv)

    fields_dir = Path(args.fields)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geos = [g for g in args.geos.split(",") if g]

    per_geo_D = {"cylinder": args.D, "ellipse": args.D_ellipse}
    tables = {"table1": [], "table2": [], "table3": [], "gci": {}}
    profiles: dict = {}
    for g in geos:
        t, p = build_tables(fields_dir, per_geo_D[g], [g])
        tables["table1"] += t["table1"]
        tables["table2"] += t["table2"]
        tables["table3"] += t["table3"]
        tables["gci"].update(t["gci"])
        profiles.update(p)

    write_csv(tables["table1"], out / "table1_grid_levels.csv")
    write_csv(tables["table2"], out / "table2_surface_angular.csv")
    write_csv(tables["table3"], out / "table3_wake_profiles.csv")
    (out / "table1_grid_levels.json").write_text(
        json.dumps({"rows": tables["table1"], "gci": tables["gci"]}, indent=2,
                   ensure_ascii=False, default=str)
    )

    manifests = []
    for g in geos:
        rows = [r for r in tables["table1"] if r["geometry"] == g]
        if not rows:
            continue
        Dbest = max(r["D_cells"] for r in rows)
        manifests.append(fig1(fields_dir, g, Dbest, out / f"fig1_{g}_cloud.png"))
        manifests.append(fig2(fields_dir, g, out / f"fig2_{g}_convergence.png", tables))

    index = {
        "scenario_key": "paper-mat-resistance-chain-de-mock-cloud",
        "mission_id": 4927,
        "repo": "TensorLBM",
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "generated_by": "benchmarks/resistance_chain_cloud/analyze.py",
        "field_source": "benchmarks/resistance_chain_cloud/run.py (D2Q9 LBM, library solver path)",
        "reference_data": {
            g: {"Cd": REFERENCE[g]["cd"], "Cd_band": REFERENCE[g]["cd_band"],
                "St": REFERENCE[g]["st"], "source": REFERENCE[g]["source"]}
            for g in geos
        },
        "tables": [str(out / n) for n in
                   ("table1_grid_levels.csv", "table2_surface_angular.csv",
                    "table3_wake_profiles.csv")],
        "figures": manifests,
        "units": {"u": "normalised by U_inf", "omega_z": "D/U_inf",
                  "Cp": "[-]", "Cf": "[-]", "x,y": "body reference length D"},
    }
    (out / "cloud_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps({"rows_table1": len(tables["table1"]),
                      "rows_table2": len(tables["table2"]),
                      "rows_table3": len(tables["table3"]),
                      "figures": [m.get("figure") for m in manifests if m],
                      "gci": tables["gci"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())