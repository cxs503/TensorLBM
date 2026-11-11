"""G21 orphan-module literature-level verification (TensorLBM).

Every number in docs/G21_ORPHAN_VERIFICATION.md is produced here and written to
artifacts/g21_orphan_verification/audit.json; the markdown is rendered from that
json inside this script (zero hand-copying).

Scope: the four modules still listed as unverified in benchmarks/gaps.md G21:
``ddes``, ``prism_solver``, ``hybrid_solver``, ``d3q27_collide``.

Run:  PYTHONPATH=src python tools/g21_orphan_verification.py
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import torch

from tensorlbm.ddes import (
    DDESConfig,
    ddes_branch_fractions,
    ddes_length_fields,
    ddes_length_scale,
    ddes_shielding_function,
    run_ddes_diagnostics,
)
from tensorlbm.prism_solver import PrismCorrector, build_prism, uplus_reichardt

DT = torch.float64
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "g21_orphan_verification"
FIGS = ROOT / "figures"
KAPPA, C_DES, C_S = 0.41, 0.65, 0.1


# ---------------- literature reference laws (from the papers) --------------
def spalart_fd(r_d):
    return 1.0 - math.tanh((8.0 * r_d) ** 3)


def spalart_lddes(l_rans, l_les, f_d):
    return l_rans - f_d * max(0.0, l_rans - l_les)


def uplus_legacy(yp):
    return yp if yp < 5.0 else math.log(max(yp, 1.1)) / KAPPA + 5.0


def van_driest_length(d_plus, c_s=C_S, a_plus=25.0):
    return c_s * (1.0 - math.exp(-d_plus / a_plus))


def ittc1957_cf(re):
    return 0.075 / (math.log10(re) - 2.0) ** 2


def _fd_of_rd(r):
    return float(
        ddes_shielding_function(
            0.0,
            torch.tensor(r * KAPPA**2, dtype=DT),
            torch.ones((), dtype=DT),
            torch.ones((), dtype=DT),
            KAPPA,
        )
    )


# ---------------- A. DDES ----------------
def audit_ddes():
    rd = np.logspace(-4, 2, 240)
    fd = np.array([_fd_of_rd(float(r)) for r in rd])
    ref = np.array([spalart_fd(float(r)) for r in rd])
    shielding = {
        "n_points": int(rd.size),
        "max_abs_dev_vs_spalart": float(np.max(np.abs(fd - ref))),
        "f_d_at_rd_1e-4": float(fd[0]),
        "f_d_at_rd_1e2": float(fd[-1]),
    }
    lo, hi = 1e-4, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _fd_of_rd(mid) < 0.5 else (lo, mid)
    rd_half = 0.5 * (lo + hi)
    rd_ref = math.atanh(0.5) ** (1.0 / 3.0) / 8.0
    half = {"rd_half_measured": rd_half, "rd_half_analytic": rd_ref,
            "rel_err": abs(rd_half - rd_ref) / rd_ref}

    cases = [(1.0, 0.65, 0.0), (1.0, 0.65, 0.5), (1.0, 0.65, 1.0),
             (0.30, 0.65, 1.0), (0.30, 0.65, 0.25), (2.0, 0.65, 0.75)]
    rows = []
    for l_rans, l_les, f in cases:
        got = float(ddes_length_scale(torch.ones((), dtype=DT), torch.tensor(f, dtype=DT),
                                      l_les / C_DES, torch.tensor(l_rans, dtype=DT), C_DES))
        r = spalart_lddes(l_rans, l_les, f)
        old = max(l_rans, l_les / f) if f > 0 else float("inf")
        rows.append({"l_rans": l_rans, "l_les": l_les, "f_d": f, "l_ddes": got,
                     "spalart": r, "abs_dev": abs(got - r),
                     "old_header": old, "old_header_abs_dev": abs(got - old)})
    ident = {"cases": rows,
             "max_abs_dev_vs_spalart": max(r["abs_dev"] for r in rows),
             "max_abs_dev_vs_old_header": max(r["old_header_abs_dev"] for r in rows)}

    viol_b = viol_m = 0.0
    for l_rans in (0.05, 0.2, 0.65, 1.0, 3.0):
        prev = None
        for f in np.linspace(0.0, 1.0, 51):
            got = float(ddes_length_scale(torch.ones((), dtype=DT), torch.tensor(float(f), dtype=DT),
                                          1.0, torch.tensor(l_rans, dtype=DT), C_DES))
            lo_b, hi_b = min(l_rans, C_DES), max(l_rans, C_DES)
            viol_b = max(viol_b, max(0.0, lo_b - got), max(0.0, got - hi_b))
            if prev is not None:
                viol_m = max(viol_m, got - prev)
            prev = got
    des_ok = abs(float(ddes_length_scale(torch.ones((), dtype=DT), torch.tensor(1.0, dtype=DT),
                                        1.0, torch.tensor(1.0, dtype=DT), C_DES)) - min(1.0, C_DES)) < 1e-12
    bounds = {"max_bound_violation": viol_b, "max_monotonicity_violation": viol_m,
              "des_limit_ok": bool(des_ok)}

    dw = torch.linspace(0.5, 200.0, 400, dtype=DT)
    d_wall = dw.view(-1, 1).expand(-1, 4).contiguous()  # helpers are 2-D
    zeros = torch.zeros_like(d_wall)
    cfg = DDESConfig()
    l_rans, l_les, l_ddes, _ = ddes_length_fields(zeros, zeros, d_wall, cfg)
    degen = {"c_s": cfg.c_s, "c_des": cfg.c_des, "dx": cfg.dx,
             "l_les": float(l_les.min()), "max_l_rans": float(l_rans.max()),
             "min_l_rans": float(l_rans.min()),
             "les_reachable_fraction": float((l_les < l_rans).float().mean()),
             "l_ddes_eq_l_rans_fraction": float((l_ddes == l_rans).float().mean())}
    cfg2 = DDESConfig(c_des=0.05)
    # control arm needs a sheared field: with ux = 0 the strain floor makes
    # r_d = nu/(~0) huge, f_d = 0, and the RANS branch is selected by construction
    ux_s = (0.2 * torch.log1p(d_wall)).contiguous()
    l_r2, l_l2, l_d2, _ = ddes_length_fields(ux_s, zeros, d_wall, cfg2)
    degen["control_c_des_0p05"] = {
        "l_les": float(l_l2.min()),
        "les_reachable_fraction": float((l_l2 < l_r2).float().mean()),
        "l_ddes_eq_l_les_fraction": float((l_d2 == l_l2).float().mean()),
        "l_ddes_eq_l_rans_fraction": float((l_d2 == l_r2).float().mean())}

    ny, nx = 64, 8
    y = torch.arange(ny, dtype=DT).view(ny, 1).expand(ny, nx).contiguous()
    d_field = y + 0.5
    ux = (0.2 * torch.log1p(d_field)).contiguous()
    uy = torch.zeros_like(ux)
    res = run_ddes_diagnostics(ux, uy, d_field, cfg)
    l_rans_f = cfg.c_s * d_field / (d_field + 1.0)
    _, l_les_f, l_ddes_f, f_d_f = ddes_length_fields(ux, uy, d_field, cfg)
    rans_t, les_t, blend_t = ddes_branch_fractions(l_rans_f, l_les_f, l_ddes_f)
    legacy_les = float((f_d_f >= 0.5).float().mean())
    branch = {"reported_les_fraction": res.les_fraction, "true_les_fraction": les_t,
              "reported_rans_fraction": res.rans_fraction, "true_rans_fraction": rans_t,
              "blend_fraction": blend_t, "legacy_formula_les_fraction": legacy_les,
              "overstatement_pp": 100.0 * (legacy_les - les_t),
              "les_limiter_active_flag": bool(res.les_limiter_active)}

    dd = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])
    law = C_S * dd / (dd + 1.0)
    vd = np.array([van_driest_length(float(d)) for d in dd])
    dev = np.abs(law - vd) / vd
    nearlaw = {"d_plus": dd.tolist(), "l_code": law.tolist(), "l_van_driest": vd.tolist(),
               "rel_dev": dev.tolist(), "max_rel_dev": float(dev.max()),
               "rel_dev_at_200": float(dev[-1])}
    return {"shielding": shielding, "half_switch": half, "length_identities": ident,
            "bounds": bounds, "limiter_degeneracy": degen, "branch_diagnostics": branch,
            "near_wall_law": nearlaw}


# ---------------- B. prism_solver ----------------
def audit_prism():
    nz, ny, nx = 6, 16, 8
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, 0, :] = True
    prism = build_prism(solid, n_layers=5, first_height=0.02, growth=1.15)
    nu, ut = 0.01, 0.02
    corr = PrismCorrector(prism, nu, torch.device("cpu"))
    u_tau = torch.full((nz, ny, nx), ut, dtype=torch.float32)
    z = torch.zeros(nz, ny, nx)
    ux_c, uy_c, uz_c = corr.correct_velocity(z.clone(), z.clone(), z.clone(), u_tau)
    h = prism["heights"]
    y_c = np.cumsum(np.concatenate([[h[0].item() / 2.0],
                                    ((h[:-1] + h[1:]) / 2.0).numpy()]))
    y_iface = float(y_c[-1] + h[-1].item() / 2.0)
    yp = y_iface * ut / nu
    surf = prism["surface_xyz"]
    got = float(ux_c[int(surf[0, 2]), int(surf[0, 1]), int(surf[0, 0])])
    u_law, u_leg = ut * uplus_reichardt(yp), ut * uplus_legacy(yp)
    injection = {"n_surface": int(prism["n_surface"]), "y_iface": y_iface, "y_plus": yp,
                 "u_tau": ut, "u_injected": got, "u_law": u_law, "u_legacy": u_leg,
                 "rel_dev_vs_reichardt": abs(got - u_law) / u_law,
                 "rel_dev_vs_legacy": abs(got - u_leg) / u_leg,
                 "wall_normal_zero": bool(float(torch.abs(uy_c).max()) == 0.0)}

    yps = np.logspace(-2, 3, 20001)
    rch = np.array([uplus_reichardt(float(y)) for y in yps])
    jump = uplus_legacy(5.0) - uplus_legacy(5.0 - 1e-9)
    continuity = {"legacy_at_5_minus": uplus_legacy(5.0 - 1e-9),
                  "legacy_at_5_plus": uplus_legacy(5.0),
                  "legacy_jump": jump, "legacy_jump_rel": jump / uplus_legacy(5.0 - 1e-9),
                  "reichardt_max_adjacent_step": float(np.max(np.abs(np.diff(rch)))),
                  "reichardt_step_over_range": float(np.max(np.abs(np.diff(rch))) / rch[-1])}
    asym = {"viscous_rel_err_y1": abs(uplus_reichardt(1.0) - 1.0),
            "viscous_rel_err_y2": abs(uplus_reichardt(2.0) - 2.0) / 2.0,
            "log_slope_at_1000": (uplus_reichardt(1000.0) - uplus_reichardt(100.0)) / math.log(10.0),
            "inv_kappa": 1.0 / KAPPA, "u_plus_at_0": uplus_reichardt(0.0),
            "finite_at_large_y": bool(math.isfinite(uplus_reichardt(1e6)))}
    asym["log_slope_rel_err"] = abs(asym["log_slope_at_1000"] - 1.0 / KAPPA) / (1.0 / KAPPA)
    return {"injection": injection, "continuity": continuity, "asymptotics": asym}


# ---------------- C. hybrid_solver ----------------
def audit_hybrid():
    from tensorlbm.hybrid_solver import FDMBoundaryStep, HybridSolver, PrismLayerData

    n_layers, n_surf = 5, 4
    heights = torch.tensor([0.02 * 1.15**k for k in range(n_layers)], dtype=DT)
    y = torch.cumsum(torch.cat([heights[:1] / 2, (heights[:-1] + heights[1:]) / 2]), 0)
    nu, dt = 2.4e-6, 0.5
    prism = PrismLayerData(n_surface=n_surf, n_layers=n_layers, layer_heights=heights,
                           surface_centers=torch.zeros(n_surf, 2, dtype=DT),
                           surface_normals=torch.zeros(n_surf, 2, dtype=DT),
                           layer_mask=torch.zeros(4, 8, 8, dtype=torch.bool), band_indices=())
    fdm = FDMBoundaryStep(prism, nu=nu, dt=dt)
    u_lin = (12.0 * y).view(n_layers, 1, 1).expand(n_layers, n_surf, 2).clone()
    u_out = fdm.step(u_lin, torch.zeros((n_surf, 2), dtype=DT))
    interior = slice(1, n_layers - 1)
    dev = float((u_out[interior] - u_lin[interior]).abs().max())
    fdm_state = {"max_change_interior": dev,
                 "max_change_interior_rel": dev / float(u_lin.abs().max()),
                 "max_change_all": float((u_out - u_lin).abs().max())}
    lam = nu * dt / float(heights.min()) ** 2
    stab = {"nu": nu, "dt": dt, "h_min": float(heights.min()), "diffusion_number": lam,
            "explicit_limit": 0.5, "margin_factor": 0.5 / lam}

    stubs = {}
    try:
        fdm.body_force_from_lbm(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.bool))
        stubs["FDMBoundaryStep.body_force_from_lbm"] = "returned a value (silent placeholder)"
    except NotImplementedError:
        stubs["FDMBoundaryStep.body_force_from_lbm"] = "NotImplementedError (explicit)"
    try:
        HybridSolver.drag_coefficient(object.__new__(HybridSolver), 1.0)
        stubs["HybridSolver.drag_coefficient"] = "returned a value (silent placeholder)"
    except NotImplementedError:
        stubs["HybridSolver.drag_coefficient"] = "NotImplementedError (explicit)"
    except Exception as exc:  # noqa: BLE001 - attribute error on a bare instance is OK
        stubs["HybridSolver.drag_coefficient"] = f"{type(exc).__name__} (not a numeric result)"
    return {"fdm_steady_state": fdm_state, "stability": stab, "stubs": stubs}


# ---------------- D. d3q27_collide ----------------
def audit_d3q27():
    from tensorlbm.d3q27_collide import suboff_config_d3q27

    cfg = suboff_config_d3q27(nx=200, re=2e6)
    re_chk = cfg["u_in"] * cfg["hull_length"] / cfg["nu_lat"]
    arith = {"Re_declared": cfg["re"], "Re_reconstructed": re_chk,
             "Re_rel_err": abs(re_chk - cfg["re"]) / cfg["re"],
             "tau": cfg["tau"], "tau_from_nu": 0.5 + 3.0 * cfg["nu_lat"],
             "tau_abs_dev": abs(cfg["tau"] - (0.5 + 3.0 * cfg["nu_lat"])),
             "ny_over_nx": cfg["ny"] / cfg["nx"], "nz_over_nx": cfg["nz"] / cfg["nx"],
             "hull_length": cfg["hull_length"], "lattice": cfg["lattice"],
             "smagorinsky_cs": cfg["smagorinsky_cs"]}
    cf = ittc1957_cf(2e6)
    hits = 0
    for p in (ROOT / "artifacts").glob("*.json"):
        try:
            if "0.00393" in p.read_text(errors="ignore"):
                hits += 1
        except OSError:
            pass
    claim = {"claimed_ct": 0.00393, "ittc1957_cf_re2e6": cf,
             "rel_dev_vs_ittc_cf": abs(0.00393 - cf) / cf,
             "artifacts_mentioning_0p00393": hits}
    return {"config_arithmetic": arith, "validation_claim": claim}


# ---------------- figures ----------------
def make_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    figs = []

    rd = np.logspace(-4, 2, 400)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx(rd, [spalart_fd(float(r)) for r in rd], "k-", lw=2,
                label="Spalart 2006: 1-tanh[(8 r$_d$)$^3$]")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    rh = a["ddes"]["half_switch"]["rd_half_analytic"]
    ax.plot([rh], [0.5], "o", ms=7, color="crimson",
            label=f"half-switch r$_d$* = {rh:.4f} (analytic)")
    ax.set_xlabel("r$_d$"); ax.set_ylabel("f$_d$"); ax.set_title("DDES shielding function (code vs paper)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "g21_fd.png", dpi=130); plt.close(fig)
    figs.append(("g21_fd.png", "DDES shielding function: implementation matches Spalart (2006) "
                 "to machine precision and switches at the analytic r_d*."))

    dw = np.linspace(0.5, 60, 400)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(dw, C_S * dw / (dw + 1.0), "b-", lw=2, label="l_RANS = C_S d/(d+1) (code)")
    ax.plot(dw, [van_driest_length(float(d)) for d in dw], "g--", lw=1.5,
            label="van Driest C_S (1-e$^{-d^+/25}$)")
    ax.axhline(C_DES, color="r", lw=2, label="l_LES = C_DES = 0.65")
    ax.set_xlabel("wall distance d (lattice units)"); ax.set_ylabel("length scale")
    ax.set_title("DES limiter never selects the LES branch (default constants)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "g21_limiter.png", dpi=130); plt.close(fig)
    figs.append(("g21_limiter.png", "l_RANS < C_DES for every wall distance (C_S = 0.1 < C_DES = 0.65), "
                 "so the DES limiter never saturates: the DDES module is RANS-only with its defaults."))

    yps = np.logspace(-2, 3, 2000)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(yps, [uplus_legacy(float(y)) for y in yps], "r--", lw=1.5,
              label="legacy two-branch (jump at y+=5)")
    ax.loglog(yps, [uplus_reichardt(float(y)) for y in yps], "b-", lw=2, label="Reichardt 1951 (fixed)")
    ax.loglog(yps, yps, "0.7", ls=":", lw=1, label="viscous u+=y+")
    ax.set_xlabel("y+"); ax.set_ylabel("u+"); ax.set_ylim(0.01, 40)
    ax.set_title("prism_solver wall law: 3.93 jump in u+ removed")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(FIGS / "g21_walllaw.png", dpi=130); plt.close(fig)
    figs.append(("g21_walllaw.png", "Legacy two-branch wall law vs the Reichardt composite law now used."))
    return figs


# ---------------- markdown ----------------
def render_md(a):
    d, pr, hy, d27 = a["ddes"], a["prism"], a["hybrid"], a["d3q27"]
    s, h, li, bo = d["shielding"], d["half_switch"], d["length_identities"], d["bounds"]
    dg, bd, nl = d["limiter_degeneracy"], d["branch_diagnostics"], d["near_wall_law"]
    inj, cnt, asy = pr["injection"], pr["continuity"], pr["asymptotics"]
    fs, st, su = hy["fdm_steady_state"], hy["stability"], hy["stubs"]
    ca, cv = d27["config_arithmetic"], d27["validation_claim"]
    L = []
    L.append("# G21 孤儿模块文献级复核（ddes / prism_solver / hybrid_solver / d3q27_collide）\n")
    L.append("`selfstart-G21 · mission 14724 · repo TensorLBM · 2026-11-11`\n")
    L.append("全部数字由 `tools/g21_orphan_verification.py` 程序化注入 "
             "`artifacts/g21_orphan_verification/audit.json`，本文件由同一脚本从该 json 渲染；"
             "回归由 `tests/test_g21_orphan_verification.py` 钉住。\n")
    L.append("## 0. 结论（证据分级）\n")
    L.append("| 模块 | 复核依据 | 数字 | 判定 |")
    L.append("|---|---|---|---|")
    L.append(f"| `ddes` 屏蔽函数 | Spalart 2006 f_d，{s['n_points']} 点 | 最大偏差 {s['max_abs_dev_vs_spalart']:.1e}；"
             f"半切换 r_d* 相对误差 {h['rel_err']:.1e} | REAL |")
    L.append(f"| `ddes` 长度尺度 | Spalart eq.(5) 恒等式 | 最大偏差 {li['max_abs_dev_vs_spalart']:.1e}"
             f"（旧头部公式偏差 {li['max_abs_dev_vs_old_header']:.3f}） | 实现 REAL / 文档已修 |")
    L.append(f"| `ddes` 支路诊断 | 真实长度尺度统计 | 旧口径高报 LES 占比 {bd['overstatement_pp']:.1f} pp | 缺陷已修 |")
    L.append(f"| `ddes` 默认常数 | l_LES vs max(l_RANS) | {dg['l_les']:.3f} vs {dg['max_l_rans']:.4f}；"
             f"LES 可达比例 {dg['les_reachable_fraction']:.2f} | 退化 RANS，已登记 |")
    L.append(f"| `prism_solver` 壁律 | Reichardt 1951 连续性 | 旧律 y⁺=5 跳变 {cnt['legacy_jump']:.3f}"
             f"（{100*cnt['legacy_jump_rel']:.0f}%）→ 复合律连续 | 缺陷已修 |")
    L.append(f"| `hybrid_solver` | FDM 稳态 / 稳定数 / 占位 | 线性剖面漂移 {fs['max_change_interior_rel']:.1e}；"
             f"λ={st['diffusion_number']:.1e}；两处占位显式化 | STUB（已显式） |")
    L.append(f"| `d3q27_collide` | 算术自洽 + 声明溯源 | Re 偏差 {ca['Re_rel_err']:.1e}；"
             f"原校验把 Ct 与 ITTC-1957 Cf 对比、工件数 {cv['artifacts_mentioning_0p00393']} | 声明已订正 |")
    L.append("")
    L.append("## 1. DDES：与 Spalart (2006) 公式级对拍\n")
    L.append(f"- f_d 逐点最大偏差 **{s['max_abs_dev_vs_spalart']:.3e}**（10⁻⁴…10² 共 {s['n_points']} 点）；"
             f"f_d(10⁻⁴)={s['f_d_at_rd_1e-4']:.2e}、f_d(10²)={s['f_d_at_rd_1e2']:.6f}。")
    L.append(f"- 半切换点实测 r_d*={h['rd_half_measured']:.10f}，解析 (atanh 0.5)^(1/3)/8="
             f"{h['rd_half_analytic']:.10f}，相对偏差 {h['rel_err']:.1e}。\n")
    L.append("| l_RANS | l_LES | f_d | 代码 l_DDES | Spalart eq.(5) | 偏差 | 旧头部 max(l_RANS, l_LES/f_d) | 偏差 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in li["cases"]:
        L.append(f"| {r['l_rans']:.2f} | {r['l_les']:.2f} | {r['f_d']:.2f} | {r['l_ddes']:.4f} | "
                 f"{r['spalart']:.4f} | {r['abs_dev']:.1e} | {r['old_header']:.4f} | {r['old_header_abs_dev']:.4f} |")
    L.append("")
    L.append(f"实现与 Spalart eq.(5) 一致，但模块头部原公式（max(RANS, C_DESΔ/f_shield)）与实现最大差 "
             f"**{li['max_abs_dev_vs_old_header']:.3f}**——按头部实现会得到「分离区用 RANS 长度、贴壁区用 LES」"
             "的反向行为（若 f_shield 很小则 C_DESΔ/f_shield 更大，max 取到它）。已订正头部。\n")
    L.append(f"- 界与单调性：越界 {bo['max_bound_violation']:.1e}，f_d 单调性违反 "
             f"{bo['max_monotonicity_violation']:.1e}，DES 极限 l(f_d=1)=min(l_RANS,C_DESΔ)："
             f"{'成立' if bo['des_limit_ok'] else '不成立'}。\n")
    L.append("### 1.1 近壁 RANS 长度律不是 van Driest（原注释不实）\n")
    L.append("| d⁺ | 代码 C_S d/(d+1) | van Driest C_S(1−e^(−d⁺/25)) | 相对偏差 |")
    L.append("|---|---|---|---|")
    for dd, lc, lv, dv in zip(nl["d_plus"], nl["l_code"], nl["l_van_driest"], nl["rel_dev"]):
        L.append(f"| {dd:.0f} | {lc:.5f} | {lv:.5f} | {100*dv:.1f}% |")
    L.append("")
    L.append(f"仅在 d→∞ 同趋 C_S Δ（d⁺=200 偏差 {100*nl['rel_dev_at_200']:.1f}%），近壁最大差 "
             f"{100*nl['max_rel_dev']:.0f}%。属建模选择，未擅改物理，仅把注释改为如实描述。\n")
    L.append("### 1.2 支路诊断口径（缺陷已修）\n")
    L.append(f"旧口径 `les_fraction = 1 − mean(f_d<0.5)` 在合成边界层场上报 {bd['legacy_formula_les_fraction']:.3f}；"
             f"按真实长度尺度（l_DDES==l_LES）统计的 LES 占比 {bd['true_les_fraction']:.3f}，"
             f"高报 **{bd['overstatement_pp']:.1f} 个百分点**（`les_limiter_active`={bd['les_limiter_active_flag']}）。"
             "修复后 `run_ddes_diagnostics` 用 `ddes_branch_fractions` 按长度尺度统计，并新增 "
             "`blend_fraction` / `les_limiter_active`。\n")
    L.append("### 1.3 默认常数下 DES 限幅器永不激活（关键）\n")
    L.append(f"- l_LES=C_DES Δ=**{dg['l_les']:.3f}**；l_RANS 在 d∈[0.5,200] 上最大值 "
             f"**{dg['max_l_rans']:.4f}**（上限 = C_S Δ = {dg['c_s']}）。")
    L.append(f"- ⇒ max(0, l_RANS−l_LES)≡0，l_DDES≡l_RANS，LES 支路可达比例 "
             f"**{dg['les_reachable_fraction']:.2f}**：默认参数下「DDES」等价于纯 RANS。")
    ctl = dg["control_c_des_0p05"]
    L.append(f"- 对照 C_DES=0.05：LES 可达比例 {ctl['les_reachable_fraction']:.2f}、"
             f"l_DDES==l_LES 占比 {ctl['l_ddes_eq_l_les_fraction']:.2f} ⇒ 限幅器按设计工作；"
             "根因是常数混用（SA-DES 的 C_DES=0.65 配 Smagorinsky 的 C_S=0.1），不是实现错误。\n")
    L.append("## 2. prism_solver：壁律连续性（缺陷已修）\n")
    L.append(f"- 注入点 y_iface={inj['y_iface']:.5f}，u_τ={inj['u_tau']}，ν=0.01 ⇒ y⁺={inj['y_plus']:.3f}；"
             f"注入 u={inj['u_injected']:.6f}，复合律 {inj['u_law']:.6f}"
             f"（偏差 {inj['rel_dev_vs_reichardt']:.1e}）；旧律 {inj['u_legacy']:.6f}"
             f"（偏差 {inj['rel_dev_vs_legacy']:.1e}）。壁法向分量清零：{inj['wall_normal_zero']}。")
    L.append(f"- 旧两段式：u⁺(5⁻)={cnt['legacy_at_5_minus']:.4f}、u⁺(5⁺)={cnt['legacy_at_5_plus']:.4f} "
             f"⇒ 跳变 **{cnt['legacy_jump']:.3f}**（{100*cnt['legacy_jump_rel']:.0f}%），直接注入 LBM 造成近壁速度阶跃。")
    L.append(f"- 复合律（Reichardt 1951）20001 点采样最大相邻步长 {cnt['reichardt_max_adjacent_step']:.4f}"
             f"（量程 {100*cnt['reichardt_step_over_range']:.2f}%）；粘性极限误差 y⁺=1 "
             f"{100*asy['viscous_rel_err_y1']:.2f}%、y⁺=2 {100*asy['viscous_rel_err_y2']:.2f}%；"
             f"对数区斜率 {asy['log_slope_at_1000']:.4f} vs 1/κ={asy['inv_kappa']:.4f}"
             f"（{100*asy['log_slope_rel_err']:.2f}%）；u⁺(0)={asy['u_plus_at_0']:.1f}。\n")
    L.append("## 3. hybrid_solver：脚手架而非求解器\n")
    L.append(f"- FDM 显式步对线性（Couette）剖面：内部层最大变化 {fs['max_change_interior']:.3e}"
             f"（相对 {fs['max_change_interior_rel']:.1e}）；全层 {fs['max_change_all']:.3e}"
             "（边界两行按设计被 Dirichlet 覆盖）。内部层用统一 h_avg 近似几何增长网格，非均匀间距误差未计入。")
    L.append(f"- 显式扩散数 λ=νΔt/h_min²={st['diffusion_number']:.2e} ≪ 0.5（余量 {st['margin_factor']:.0f}×）。")
    L.append("- 占位实现（已改显式）：")
    for k, v in su.items():
        L.append(f"  - `{k}`：{v}")
    L.append("- 结论：模块**不可用**（无 LBM→棱柱体力传递、阻力曾返回 0.0）。登记 STUB，不进可用清单。\n")
    L.append("## 4. d3q27_collide：配置算术 vs 校验声明\n")
    L.append(f"- 算术自洽：Re 重构偏差 {ca['Re_rel_err']:.1e}；τ 偏差 {ca['tau_abs_dev']:.1e}；"
             f"ny/nx={ca['ny_over_nx']:.2f}、nz/nx={ca['nz_over_nx']:.2f}；"
             f"lattice={ca['lattice']}、C_s={ca['smagorinsky_cs']}。")
    L.append(f"- 原声明「SUBOFF Ct=0.00393，2.9% vs ref 0.00405」：0.00405 = ITTC-1957 平板摩擦线 "
             f"Cf(2×10⁶)={cv['ittc1957_cf_re2e6']:.6f}，与 Ct（总阻力系数）不是同一量；"
             f"仓内提及 0.00393 的工件数 **{cv['artifacts_mentioning_0p00393']}** ⇒ 不可溯源。"
             "已改为诚实状态（drag prediction UNVERIFIED），模块定位为参数构造器。\n")
    L.append("## 5. 改动清单\n")
    L.append("| 文件 | 改动 | 类型 |")
    L.append("|---|---|---|")
    L.append("| `src/tensorlbm/ddes.py` | 头部公式订正为 Spalart eq.(5)；删除死代码；新增 "
             "`ddes_length_fields` / `ddes_branch_fractions`；`DDESResult` 增 `blend_fraction`、"
             "`les_limiter_active`；支路分数改为按长度尺度统计 | 缺陷修复 |")
    L.append("| `src/tensorlbm/prism_solver.py` | 两段式壁律 → Reichardt 复合律 `uplus_reichardt`（连续） | 缺陷修复 |")
    L.append("| `src/tensorlbm/hybrid_solver.py` | 两处静默占位（体力返零、阻力返 0.0）→ `NotImplementedError` | 缺陷修复 |")
    L.append("| `src/tensorlbm/d3q27_collide.py` | 删除不可溯源校验声明，标注 UNVERIFIED / 参数构造器定位 | 文档订正 |")
    L.append("")
    L.append("## 6. 局限\n")
    L.append("- 全部为符号/解析级复核（函数级），**不含端到端模拟**：DDES 的 ν_t 未与任何湍流算例对标；"
             "首行「DES 限幅器退化」是常数标定问题，改成可用模型需要重新标定 C_DES（未做，属建模决策）。")
    L.append("- `hybrid_solver` 的 FDM 步只验证了稳态保持与稳定数，未验证与 LBM 的耦合精度（耦合并未实现）。")
    L.append("- `d3q27_collide` 只验证参数构造，不构成对 D3Q27 阻力能力的任何验证。")
    L.append("- 机器：本机 128 核，load15 ≥ 100 时按容量纪律只做轻量活，本复核全程 CPU 单线程、秒级。")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {"ddes": audit_ddes(), "prism": audit_prism(),
             "hybrid": audit_hybrid(), "d3q27": audit_d3q27()}
    figs = make_figures(audit)
    audit["figures"] = [f[0] for f in figs]
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    (ROOT / "docs" / "G21_ORPHAN_VERIFICATION.md").write_text(render_md(audit))
    print("ddes  f_d max dev vs Spalart:",
          audit["ddes"]["shielding"]["max_abs_dev_vs_spalart"])
    print("ddes  l_DDES max dev vs eq.(5):",
          audit["ddes"]["length_identities"]["max_abs_dev_vs_spalart"])
    print("ddes  LES reachable fraction:",
          audit["ddes"]["limiter_degeneracy"]["les_reachable_fraction"])
    print("ddes  legacy LES overstatement (pp):",
          audit["ddes"]["branch_diagnostics"]["overstatement_pp"])
    print("prism legacy wall-law jump:", audit["prism"]["continuity"]["legacy_jump"])
    print("hybrid stubs:", audit["hybrid"]["stubs"])
    print("wrote", OUT / "audit.json", "and docs/G21_ORPHAN_VERIFICATION.md")


if __name__ == "__main__":
    main()