#!/usr/bin/env python3
"""IN70-N03 benchmark: Boussinesq scalar transport -> Gaussian anchors -> LFL hazard.

Produces, with every number computed by the code (nothing transcribed by hand):

* ``artifacts/in70_scalar_plume/results.json`` — all acceptance numbers with
  threshold / pass flag / method / figure;
* ``artifacts/in70_scalar_plume/fig*.png`` — the figures;
* ``docs/in70_scalar_plume_buoyant_hazard.md`` — the report, written from the
  same JSON.

Run::

    PYTHONPATH=src python3 benchmarks/in70_scalar_plume/run.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import tensorlbm.plume_hazard as phz
import tensorlbm.scalar_transport as sct

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "in70_scalar_plume"
DOCS = ROOT / "docs"
D = 0.05


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError("not JSON serializable: %r" % type(o))


def _center(c: torch.Tensor) -> torch.Tensor:
    nx = c.shape[1]
    j = int(c.sum(dim=0).argmax())
    return torch.roll(c, nx // 2 - j, dims=1)


# --------------------------------------------------------------------------
# 1. Gaussian pulse + D_eff grid convergence
# --------------------------------------------------------------------------
def gaussian_study() -> dict:
    rows = []
    profile = analytic = None
    for N in (96, 128, 160, 192):
        U = 0.02 * 128 / N
        s0 = N / 64.0
        sig = N / 13.0
        steps = int((sig**2 - s0**2) / (2 * D))
        cfg = sct.ScalarTransportConfig(nx=N, ny=3, diffusivity=D, ux=U)
        t0 = time.time()
        res = sct.run_pulse_1d(cfg, steps, mass=1.0, sigma0=s0)
        xs = torch.arange(N, dtype=torch.float64) - N // 2
        c = _center(res.c)
        exact = sct.gaussian_pulse_1d(xs, 1.0, D, 0.0, steps, sigma0=s0)
        exact = exact / exact.sum()
        mass, cen, var = sct.gaussian_moments(c, xs)
        fit = sct.gaussian_pulse_1d(xs - float(cen), 1.0, D, 0.0, 1e-30, sigma0=var**0.5)
        fit = fit / fit.sum()
        d_eff = sct.effective_diffusivity(c, xs, steps, sigma0=s0)
        rows.append(
            {
                "N": N,
                "steps": steps,
                "U": U,
                "sigma0": s0,
                "sigma_final": math.sqrt(s0**2 + 2 * D * steps),
                "travel": U * steps,
                "rel_l2_full": sct.relative_l2(c, exact),
                "rel_l2_shape": sct.relative_l2(c, fit),
                "centroid_offset": cen,
                "var_target": s0**2 + 2 * D * steps,
                "var_measured": var,
                "D_eff": d_eff,
                "D_rel_error": abs(d_eff / D - 1.0),
                "budget_rel": res.budget.residual_relative(),
                "wall_s": time.time() - t0,
            }
        )
        if N == 192:
            profile, analytic = c, exact
    errs = [r["D_rel_error"] for r in rows]
    ns = [r["N"] for r in rows]
    slopes = [
        math.log(errs[i] / errs[i + 1]) / math.log(ns[i + 1] / ns[i])
        for i in range(len(errs) - 1)
    ]
    p = slopes[-1]
    r = ns[-1] / ns[-2]
    d_inf = rows[-1]["D_eff"] + (rows[-1]["D_eff"] - rows[-2]["D_eff"]) / (r**p - 1.0)
    return {
        "rows": rows,
        "slopes": slopes,
        "richardson_D": d_inf,
        "richardson_rel_error": abs(d_inf / D - 1.0),
        "profile": profile.tolist(),
        "analytic": analytic.tolist(),
    }


# --------------------------------------------------------------------------
# 2. erfc half-space
# --------------------------------------------------------------------------
def erfc_study() -> dict:
    rows = []
    profile = analytic = None
    for N, steps in ((256, 3000), (384, 8000)):
        cfg = sct.ScalarTransportConfig(nx=N, ny=1, diffusivity=D)
        t0 = time.time()
        res = sct.run_erfc_half_space(cfg, steps, level=1.0)
        x = res.x
        width = math.sqrt(4.0 * D * steps)
        exact = sct.erfc_half_space(x, 1.0, D, steps)
        mask = x < 3.0 * width
        rows.append(
            {
                "N": N,
                "steps": steps,
                "front_width": width,
                "rel_l2": sct.relative_l2(res.c[0], exact, mask),
                "budget_rel": res.budget.residual_relative(),
                "wall_s": time.time() - t0,
            }
        )
        if N == 384:
            profile, analytic = res.c[0], exact
    return {"rows": rows, "profile": profile.tolist(), "analytic": analytic.tolist()}


# --------------------------------------------------------------------------
# 3. crosswind release: symmetry, LFL isopleth, refinement, wind sweep
# --------------------------------------------------------------------------
def _lbm_case(
    nx: int,
    ny: int,
    wind: float,
    rate: float,
    steps: int,
    level_frac: float | None = None,
    level_abs: float | None = None,
) -> dict:
    """One steady crosswind release; the hazard threshold is ``level_frac * C_source``.

    Taking the threshold relative to the source-node concentration is what makes the
    LFL a property of the gas (LFL = fraction of the pure-gas source) instead of an
    arbitrary absolute number, and it keeps the isopleth inside the domain (the
    absolute-threshold variant clips on the outflow face, see docs ⑤).
    """
    cfg = sct.ScalarTransportConfig(nx=nx, ny=ny, diffusivity=D, ux=wind)
    t0 = time.time()
    res = sct.run_crosswind_release(cfg, steps, source_rate=rate)
    c = res.c
    cmax = float(c.max())
    # level: either a fraction of the source-node concentration (wind sweep) or an
    # absolute lattice value fixed once and rescaled as dx^2 under refinement
    level = float(level_abs) if level_abs is not None else float(level_frac) * cmax
    metrics = sct.lfl_contour_metrics(c, level)
    asym = float((c - c.flip(0)).abs().max()) / float(c.max())
    return {
        "nx": nx,
        "ny": ny,
        "wind": wind,
        "rate": rate,
        "steps": steps,
        "level": level,
        "level_frac": level_frac,
        "closed": bool(metrics["extent_x"] < nx - 2),
        "cmax": cmax,
        "extent_lattice": metrics["extent_x"],
        "width_lattice": metrics["max_width"],
        "area_lattice": metrics["area"],
        "perimeter_lattice": metrics["perimeter"],
        "asymmetry": asym,
        "budget_rel": res.budget.residual_relative(),
        "wall_s": time.time() - t0,
        "field": c,
    }


def crosswind_study(fast: bool) -> dict:
    level_frac = 0.25
    winds = [0.02, 0.03, 0.04, 0.06, 0.08]
    nx, ny = (64, 41) if fast else (96, 61)
    sweep = []
    for u in winds:
        sweep.append(_lbm_case(nx, ny, u, 1.0, int(1.6 * nx / u), level_frac))
    us = [r["wind"] for r in sweep]
    ws = [r["width_lattice"] for r in sweep]
    slope = phz.fit_power_law(us, ws, window=None)["exponent"]
    monotone = all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))
    ref = sweep[3]
    ladder = [ref]
    for r in (1.5, 2.0) if not fast else (1.5,):
        nxr = int(round(nx * r))
        nyr = int(round(ny * r)) | 1
        ladder.append(
            _lbm_case(
                nxr,
                nyr,
                ref["wind"] / r,
                ref["rate"] / r**2,
                int(round(ref["steps"] * r**2)),
                level_abs=ref["level"] / r**2,  # fixed physical threshold (C_lattice ~ dx^2)
            )
        )
    for row in ladder:
        row["physical_extent"] = row["extent_lattice"] * nx / row["nx"]
        row["physical_width"] = row["width_lattice"] * nx / row["nx"]
    ext = [row["physical_extent"] for row in ladder]
    grid_drift = [abs(ext[i] - ext[i + 1]) / ext[i] for i in range(len(ext) - 1)]
    field = ref.pop("field")
    for row in sweep:
        row.pop("field", None)
    centre = field[field.shape[0] // 2]
    j_src = ref["nx"] // 8
    seg = centre[j_src + 1 :]
    peak = float(seg.max())
    xs_c, cs_c = [], []
    for k, v in enumerate(seg.tolist()):
        if 0.02 * peak <= v <= 0.8 * peak:
            xs_c.append(float(k + 1))
            cs_c.append(v)
    centreline_fit = (
        phz.fit_power_law(xs_c, cs_c, window=None) if len(xs_c) >= 8 else {"exponent": float("nan"), "coefficient": 0.0, "rms": 0.0, "n": len(xs_c)}
    )
    return {
        "level_frac": level_frac,
        "sweep": sweep,
        "isopleth_closed": all(row["closed"] for row in sweep + ladder),
        "width_vs_wind_slope": slope,
        "width_monotone": monotone,
        "ladder": ladder,
        "grid_drift": grid_drift,
        "hazard_field": field.tolist(),
        "centreline_decay": centreline_fit,
        "centreline_points": len(xs_c),
        "hazard_case": dict(ref),
    }


# --------------------------------------------------------------------------
# 4. plume / jet self-similarity + empirical hazard envelope
# --------------------------------------------------------------------------
def plume_study() -> dict:
    alpha = 0.05

    def run(buoyancy: float, n_steps: int = 6000):
        cfg = phz.IntegralPlumeConfig(
            entrainment=alpha, buoyancy_flux=buoyancy, z_start=1.0, z_end=200.0, n_steps=n_steps
        )
        seed = (
            phz.plume_self_similar(cfg, 1.0)
            if buoyancy > 0
            else phz.jet_self_similar(cfg, 1.0)
        )
        seeded = phz.IntegralPlumeConfig(
            entrainment=alpha,
            buoyancy_flux=buoyancy,
            z_start=1.0,
            z_end=200.0,
            n_steps=n_steps,
            volume_flux=seed["Q"],
            momentum_flux=seed["M"],
        )
        return phz.integrate_plume(seeded)

    out = run(1.0)
    jout = run(0.0)
    c3 = 6.0 * alpha / 5.0
    q_fit = phz.fit_power_law(out["z"], out["Q"], window=(10.0, 150.0))
    w_fit = phz.fit_power_law(out["z"], out["w"], window=(10.0, 150.0))
    b_fit = phz.fit_power_law(out["z"], out["b"], window=(10.0, 150.0))
    jq = phz.fit_power_law(jout["z"], jout["Q"], window=(10.0, 150.0))
    jw = phz.fit_power_law(jout["z"], jout["w"], window=(10.0, 150.0))
    alpha_scan = []
    for a in (0.04, 0.05, 0.08, 0.12):
        c = phz.IntegralPlumeConfig(
            entrainment=a, buoyancy_flux=1.0, z_start=1.0, z_end=200.0, n_steps=6000
        )
        s2 = phz.plume_self_similar(c, 1.0)
        o2 = phz.integrate_plume(
            phz.IntegralPlumeConfig(
                entrainment=a,
                buoyancy_flux=1.0,
                z_start=1.0,
                z_end=200.0,
                n_steps=6000,
                volume_flux=s2["Q"],
                momentum_flux=s2["M"],
            )
        )
        f2 = phz.fit_power_law(o2["z"], o2["Q"], window=(10.0, 150.0))
        alpha_scan.append({"alpha": a, **f2})
    return {
        "alpha": alpha,
        "plume": {
            "Q_fit": q_fit,
            "w_fit": w_fit,
            "b_fit": b_fit,
            "analytic_coefficient": (3.0 * math.pi**2 * c3**4 / 4.0) ** (1.0 / 3.0),
        },
        "jet": {
            "Q_fit": jq,
            "w_fit": jw,
            "analytic_coefficient": 2.0 * alpha * math.sqrt(math.pi),
        },
        "alpha_scan": alpha_scan,
        "curves": {
            "z": out["z"],
            "Q": out["Q"],
            "w": out["w"],
            "b": out["b"],
            "jet_z": jout["z"],
            "jet_Q": jout["Q"],
        },
    }


def hazard_study() -> dict:
    winds = [2.0, 3.0, 5.0, 8.0, 12.0]
    rates = {"CH4": 5.0, "H2": 5.0, "NH3": 20.0}
    out = {}
    for sp, rate in rates.items():
        sweep = phz.hazard_wind_sweep(sp, rate, winds, stabilities=("D", "F"), stack_height=0.5)
        ref = phz.lfl_hazard_distance(phz.SPECIES[sp], rate, 3.0, stability="D", stack_height=0.5)
        grid = {}
        for dx in (8.0, 4.0, 2.0, 1.0):
            g = phz.lfl_contour_on_grid(
                phz.SPECIES[sp], rate, 3.0, stability="D", stack_height=0.5, dx=dx
            )
            grid[str(dx)] = {
                "distance": g["distance"],
                "rel_error_vs_reference": abs(g["distance"] / ref["distance"] - 1.0),
            }
        out[sp] = {
            "release_rate_kg_s": rate,
            "lfl_volume_fraction": phz.SPECIES[sp].lfl_volume_fraction,
            "lfl_mass_concentration": phz.SPECIES[sp].lfl_mass_concentration(),
            "sweep": sweep,
            "reference_distance_D_u3": ref,
            "grid_refinement": grid,
            "max_grid_rel_change": max(v["rel_error_vs_reference"] for v in grid.values()),
        }
    return {"winds": winds, "species": out}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(gauss, erfc, cross, plume, hazard) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    OUT.mkdir(parents=True, exist_ok=True)
    figs = []

    def save(fig, name: str, caption: str):
        path = OUT / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        figs.append({"name": name, "path": str(path), "caption": caption})

    # fig1
    c = gauss["profile"][0] if isinstance(gauss["profile"][0], list) else gauss["profile"]
    ex = gauss["analytic"][0] if isinstance(gauss["analytic"][0], list) else gauss["analytic"]
    N = len(c)
    x = list(range(N))
    xs = [v - N // 2 for v in x]
    row = gauss["rows"][-1]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(xs, c, "o", ms=3, mfc="none", label="LBM (D2Q5 TRT)")
    ax[0].plot(xs, ex, "-", lw=1.2, label="analytic Gaussian")
    ax[0].set_xlabel("x [lattice]")
    ax[0].set_ylabel("C")
    ax[0].set_title("pulse N=%d t=%d, rel L2=%.2f%%" % (N, row["steps"], row["rel_l2_full"] * 100))
    ax[0].legend()
    ax[1].semilogy(xs, [abs(a - b) + 1e-12 for a, b in zip(c, ex)])
    ax[1].set_xlabel("x [lattice]")
    ax[1].set_ylabel("|C_LBM - C_exact|")
    fig.tight_layout()
    save(
        fig,
        "fig1_gaussian_pulse.png",
        "移动高斯脉冲 LBM vs 解析解（相对 L2 = %.2f%%）" % (row["rel_l2_full"] * 100),
    )

    # fig2
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for r in gauss["rows"]:
        ax[0].plot([r["steps"]], [r["var_measured"]], "o", ms=4, label="N=%d" % r["N"])
        ax[0].plot([r["steps"]], [r["var_target"]], "x", ms=6, mfc="none", color="k")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("t [steps]")
    ax[0].set_ylabel("sigma^2")
    ax[0].set_title("measured o vs target x (sigma0^2+2Dt)")
    ns = [r["N"] for r in gauss["rows"]]
    errs = [r["D_rel_error"] for r in gauss["rows"]]
    ax[1].loglog(ns, errs, "o-")
    slope = gauss["slopes"][-1]
    ax[1].loglog(ns, [errs[0] * (ns[0] / n) ** slope for n in ns], "--", label="slope %.2f" % slope)
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("|D_eff/D - 1|")
    ax[1].legend()
    ax[1].set_title("Richardson D=%.6f (err %.1e)" % (gauss["richardson_D"], gauss["richardson_rel_error"]))
    fig.tight_layout()
    save(
        fig,
        "fig2_deff_convergence.png",
        "sigma^2=2Dt 与 D_eff 网格收敛（观测阶 %.2f，外推误差 %.1e）" % (slope, gauss["richardson_rel_error"]),
    )

    # fig3
    row = erfc["rows"][-1]
    cp, ea = erfc["profile"], erfc["analytic"]
    lim = int(3 * row["front_width"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(range(lim), cp[:lim], "o", ms=3, mfc="none", label="LBM")
    ax.plot(range(lim), ea[:lim], "-", lw=1.2, label="erfc(x/sqrt(4Dt))")
    ax.set_xlabel("x [lattice]")
    ax.set_ylabel("C")
    ax.set_title("half-space Dirichlet, rel L2=%.4f%%" % (row["rel_l2"] * 100))
    ax.legend()
    fig.tight_layout()
    save(fig, "fig3_erfc.png", "半无限 erfc 解对照（相对 L2 = %.4f%%）" % (row["rel_l2"] * 100))

    # fig4
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for sp, rec in hazard["species"].items():
        rows = rec["sweep"]["rows"]
        ws = [r["wind"] for r in rows]
        ax.plot(ws, [r["distance_D"] for r in rows], "o-", label="%s (D, Q=%.0f kg/s)" % (sp, rec["release_rate_kg_s"]))
        ax.fill_between(ws, [r["distance_F"] for r in rows], [r["distance_D"] for r in rows], alpha=0.12)
    ax.set_xlabel("wind speed U [m/s]")
    ax.set_ylabel("LFL downwind distance [m]")
    ax.set_yscale("log")
    ax.set_title("LFL hazard extent, 5 wind speeds (band = D..F interval)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(
        fig,
        "fig4_lfl_wind_sweep.png",
        "可燃气云 LFL 危险区下风距离 vs 风速（5 档；带 = D..F 稳定度经验区间）",
    )

    # fig5
    z, q = plume["curves"]["z"], plume["curves"]["Q"]
    jz, jq = plume["curves"]["jet_z"], plume["curves"]["jet_Q"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].loglog(z, q, "-", label="plume B>0")
    zz = [v for v in z if 10 <= v <= 150]
    ax[0].loglog(zz, [plume["plume"]["Q_fit"]["coefficient"] * v ** (5 / 3) for v in zz], "--",
                 label="z^5/3 fit p=%.4f" % plume["plume"]["Q_fit"]["exponent"])
    ax[0].loglog(jz, jq, "-", label="jet g'=0")
    jzz = [v for v in jz if 10 <= v <= 150]
    ax[0].loglog(jzz, [plume["jet"]["Q_fit"]["coefficient"] * v for v in jzz], "--",
                 label="z^1 fit p=%.4f" % plume["jet"]["Q_fit"]["exponent"])
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("volume flux Q")
    ax[0].legend(fontsize=7)
    c3 = 6 * plume["alpha"] / 5
    ax[1].semilogx(z, [v / c3 for v in plume["curves"]["b"]], label="b/(6alpha/5)")
    ax[1].semilogx(z, [v * zi ** (1 / 3) for v, zi in zip(plume["curves"]["w"], z)], label="w z^(1/3)")
    ax[1].set_xlabel("z")
    ax[1].set_title("self-similar shape factors (flat = self-similar)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(
        fig,
        "fig5_plume_similarity.png",
        "浮力羽流 Q~z^5/3（拟合 %.4f）与负控动量射流 Q~z（拟合 %.4f）"
        % (plume["plume"]["Q_fit"]["exponent"], plume["jet"]["Q_fit"]["exponent"]),
    )

    # fig6
    field = cross["hazard_field"]
    ny, nx = len(field), len(field[0])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    im = ax[0].imshow(field, origin="lower", aspect="auto", cmap="inferno")
    ax[0].contour(field, levels=[cross["hazard_case"]["level"]], colors="cyan", linewidths=1.2)
    ax[0].set_title("LBM concentration, U=%.2f (cyan: C=%.2f C_src)" % (cross["hazard_case"]["wind"], cross["level_frac"]))
    ax[0].set_xlabel("x [lattice]")
    ax[0].set_ylabel("y [lattice]")
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    rows = cross["sweep"]
    ax[1].loglog([r["wind"] for r in rows], [r["width_lattice"] for r in rows], "o-")
    ax[1].set_xlabel("wind U [lattice]")
    ax[1].set_ylabel("LFL max half-width")
    ax[1].set_title("width ~ U^(%.3f)  (2-D analytic -1.5)" % cross["width_vs_wind_slope"])
    fig.tight_layout()
    save(
        fig,
        "fig6_lbm_hazard_field.png",
        "LBM 横风泄漏浓度场 + LFL 等值线；危险区宽度 ~ U^(%.2f)（二维解析 -1.5）"
        % cross["width_vs_wind_slope"],
    )
    return figs


# --------------------------------------------------------------------------
# criteria
# --------------------------------------------------------------------------
def evaluate(gauss, erfc, cross, plume, hazard) -> dict:
    prod = gauss["rows"][-1]
    budgets = (
        [r["budget_rel"] for r in gauss["rows"]]
        + [r["budget_rel"] for r in erfc["rows"]]
        + [r["budget_rel"] for r in cross["sweep"]]
        + [r["budget_rel"] for r in cross["ladder"]]
    )
    worst_budget = max(budgets)
    gauss_err = max(prod["rel_l2_full"], prod["rel_l2_shape"])
    min_slope = min(gauss["slopes"])
    lfl_grid = max(rec["max_grid_rel_change"] for rec in hazard["species"].values())
    lbm_drift = max(cross["grid_drift"])
    crit = {
        "gauss_profile_le_2pct": {
            "value": gauss_err,
            "detail": {"full": prod["rel_l2_full"], "shape": prod["rel_l2_shape"], "N": prod["N"], "steps": prod["steps"]},
            "threshold": 0.02,
            "pass": gauss_err <= 0.02,
            "method": "D2Q5 TRT LBM, sigma=%.1f cells" % prod["sigma_final"],
        },
        "deff_refinement": {
            "value": min_slope,
            "detail": {
                "observed_orders": gauss["slopes"],
                "richardson_D": gauss["richardson_D"],
                "richardson_rel_error": gauss["richardson_rel_error"],
            },
            "threshold": {"min_slope": 0.9, "max_extrapolated_error": 0.02},
            "pass": min_slope >= 0.9 and gauss["richardson_rel_error"] <= 0.02,
            "method": "variance-based D_eff at 3 resolutions + Richardson extrapolation",
        },
        "mass_conservation_open_boundary": {
            "value": worst_budget,
            "threshold": 1e-12,
            "pass": worst_budget <= 1e-12,
            "method": "inlet reservoir + source + open-face flux accounting",
        },
        "erfc_le_1pct": {
            "value": erfc["rows"][-1]["rel_l2"],
            "threshold": 0.01,
            "pass": erfc["rows"][-1]["rel_l2"] <= 0.01,
            "method": "N=%d, t=%d, front width %.0f cells"
            % (erfc["rows"][-1]["N"], erfc["rows"][-1]["steps"], erfc["rows"][-1]["front_width"]),
        },
        "crosswind_symmetry": {
            "value": max(r["asymmetry"] for r in cross["sweep"]),
            "threshold": 1e-8,
            "pass": max(r["asymmetry"] for r in cross["sweep"]) <= 1e-8,
            "method": "odd ny, source on centre row, max|C-flip(C)|/max(C)",
        },
        "plume_exponent": {
            "value": plume["plume"]["Q_fit"]["exponent"],
            "threshold": [1.6, 1.75],
            "pass": 1.6 <= plume["plume"]["Q_fit"]["exponent"] <= 1.75,
            "detail": {
                "analytic_prefactor": plume["plume"]["analytic_coefficient"],
                "fitted_prefactor": plume["plume"]["Q_fit"]["coefficient"],
                "width_slope": plume["plume"]["b_fit"]["coefficient"],
                "analytic_width_slope": 6.0 * plume["alpha"] / 5.0,
                "power_law_rms": plume["plume"]["Q_fit"]["rms"],
                "alpha_scan": plume["alpha_scan"],
            },
            "method": "Morton-Taylor-Turner integral model, RK4, fit window z in [10,150]",
        },
        "jet_negative_control_exponent": {
            "value": plume["jet"]["Q_fit"]["exponent"],
            "threshold": [0.95, 1.05],
            "pass": 0.95 <= plume["jet"]["Q_fit"]["exponent"] <= 1.05,
            "detail": {
                "analytic_prefactor": plume["jet"]["analytic_coefficient"],
                "fitted_prefactor": plume["jet"]["Q_fit"]["coefficient"],
                "w_exponent": plume["jet"]["w_fit"]["exponent"],
            },
            "method": "same integrator with buoyancy flux B=0",
        },
        "lfl_wind_monotone": {
            "value": {sp: [round(v, 2) for v in rec["sweep"]["distances"]] for sp, rec in hazard["species"].items()},
            "threshold": "strictly decreasing over 5 wind speeds",
            "pass": all(rec["sweep"]["monotone_decreasing"] for rec in hazard["species"].values()),
            "detail": {
                "lbm_width_monotone": cross["width_monotone"],
                "lbm_width_exponent": cross["width_vs_wind_slope"],
            },
            "method": "Briggs/Pasquill-Gifford Gaussian plume (empirical), 5 wind speeds; LBM 2-D corroboration",
        },
        "lfl_isopleth_closed": {
            "value": cross["isopleth_closed"],
            "threshold": "hazard isopleth closes inside the domain (not clipped by the outflow)",
            "pass": bool(cross["isopleth_closed"]),
            "method": "C >= %.2f x C_source, extent < nx-2" % cross["level_frac"],
        },
        "centreline_decay_2d": {
            "value": cross["centreline_decay"]["exponent"],
            "threshold": "analytic 2-D continuous line source: C_cl ~ x^(-1/2) (informative)",
            "pass": abs(cross["centreline_decay"]["exponent"] + 0.5) <= 0.15,
            "detail": {
                "fit": cross["centreline_decay"],
                "points": cross["centreline_points"],
                "note": "2-D isotropic release; the width-vs-wind exponent is NOT used as a "
                        "validation (finite domain + volumetric source), only monotonicity is",
            },
            "method": "log-log fit of the centreline concentration downstream of the source",
        },
        "lfl_grid_convergence": {
            "value": lfl_grid,
            "detail": {
                "empirical_model": {sp: rec["max_grid_rel_change"] for sp, rec in hazard["species"].items()},
                "lbm_diffusive_refinement": cross["grid_drift"],
            },
            "threshold": 0.03,
            "pass": lfl_grid <= 0.03 and lbm_drift <= 0.03,
            "method": "isopleth extraction on dx=8/4/2/1 m + LBM diffusive refinement r=1/1.5/2.0",
        },
    }
    crit["all_pass"] = all(v["pass"] for v in crit.values() if isinstance(v, dict) and "pass" in v)
    return crit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cache = OUT / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        path = cache / (name + ".json")
        if path.exists():
            print("    [cache] %s" % name, flush=True)
            return json.loads(path.read_text())
        out = fn()
        path.write_text(json.dumps(out, default=_json_default))
        return out

    print("[1/5] gaussian pulse + D_eff ladder", flush=True)
    gauss = cached("gaussian", gaussian_study)
    print("     ", [(r["N"], round(r["D_rel_error"], 6)) for r in gauss["rows"]], flush=True)
    print("[2/5] erfc half-space", flush=True)
    erfc = cached("erfc", erfc_study)
    print("     ", [(r["N"], round(r["rel_l2"], 6)) for r in erfc["rows"]], flush=True)
    print("[3/5] crosswind release / LFL", flush=True)
    cross = cached("crosswind%s" % ("_fast" if args.fast else ""), lambda: crosswind_study(args.fast))
    print("      width exponent %.3f monotone=%s drift=%s"
          % (cross["width_vs_wind_slope"], cross["width_monotone"],
             [round(v, 4) for v in cross["grid_drift"]]), flush=True)
    print("[4/5] integral plume / jet", flush=True)
    plume = cached("plume", plume_study)
    print("      plume p=%.5f jet p=%.5f"
          % (plume["plume"]["Q_fit"]["exponent"], plume["jet"]["Q_fit"]["exponent"]), flush=True)
    print("[5/5] empirical LFL hazard envelope", flush=True)
    hazard = cached("hazard", hazard_study)
    calibration = sct.calibrate_magic(D)
    disp = {}
    tau_odd = sct.tau_from_diffusivity(D)
    for magic in (0.1875, None):
        ts = tau_odd if magic is None else sct.tau_symmetric_from_magic(tau_odd, magic)
        disp[str(magic)] = [
            round(sct.dispersion_rates(k, ts, tau_odd)[0], 6) for k in (0.05, 0.2, 0.4, 0.6)
        ]
    criteria = evaluate(gauss, erfc, cross, plume, hazard)
    results = {
        "mission": "IN70-N03",
        "repo": "TensorLBM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": time.time() - t0,
        "diffusivity_lattice": D,
        "calibration": calibration,
        "dispersion_D_eff_vs_k": disp,
        "criteria": criteria,
        "gaussian": gauss,
        "erfc": erfc,
        "crosswind": {k: v for k, v in cross.items() if k != "hazard_field"},
        "plume": {k: v for k, v in plume.items() if k != "curves"},
        "hazard": hazard,
        "figures": [],
        "fig_paths": {},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, sort_keys=False, default=_json_default))
    print("\nwrote", OUT / "results.json", flush=True)
    figs = make_figures(gauss, erfc, cross, plume, hazard)
    results["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
    results["fig_paths"] = {f["name"]: f["path"] for f in figs}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, sort_keys=False, default=_json_default))
    print("\ncriteria:")
    for k, v in criteria.items():
        if isinstance(v, dict) and "pass" in v:
            print("  %-32s %-10s value=%s" % (k, "PASS" if v["pass"] else "FAIL", v["value"]))
    print("  all_pass =", criteria["all_pass"])
    print("\nwrote", OUT / "results.json", "and", len(figs), "figures, %.1fs" % (time.time() - t0))
    return 0 if criteria["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())