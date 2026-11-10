"""p247w1-N10 / C6：H = 32/64/128 网格族上的 b(sigma) 曲线与收敛阶。

产出：
  artifacts/p247w1_n10_c6_grid.json   —— 判据数字（零手抄，报告从这里注入）
  figures/p247w1_n10_*.png            —— 图（b(sigma) 曲线 / 偏置 c(H) / 剖面残差）

用法：  PYTHONPATH=src python scripts/p247w1_n10_c6_grid.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorlbm.boundary.partial_slip_second_order import (
    channel_poiseuille_2nd,
    probe_parabola_residual,
    slip_law_theory,
    spurious_slip_coefficient,
)
from tensorlbm.boundary.partial_slip_wall import (
    analytic_plane_profile,
    invert_slip_length,
    nu_from_tau,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
TAU = 1.1          # nu = 0.2：格点粘度大、收敛快（偏置闭式与 tau 无关，已在 tests 里扫过）
G = 1e-5
SIGMAS = (0.0, 0.05, 0.2, 0.5)
HS = (32, 64, 128)
# 成本限制（诚实申报）：H=128 只跑 C1/C2 的两个 σ（0.05=小滑移区、0.2=中滑移区），
# 其余 σ 只到 H=64；H=128 的每一次定常解都受最慢扩散模态支配（~2e5 步）。
H128_SIGMAS = (0.05, 0.2)


def ny_of(H: int) -> int:
    return 2 * H + 2


def main() -> dict:
    ART.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    nu = nu_from_tau(TAU)
    rows = []
    profiles = {}
    import time
    for H in HS:
        ny = ny_of(H)
        steps = int(6.0 * H * H / nu) + 40000
        for sigma in SIGMAS:
            if H == 128 and sigma not in H128_SIGMAS:
                continue
            for corr in (True, False):
                t0 = time.time()
                r = channel_poiseuille_2nd(
                    sigma, ny=ny, tau=TAU, g=G, steps=steps, tol=1e-11, correction=corr
                )
                inv = invert_slip_length(r)
                bt = slip_law_theory(sigma, nu)
                row = {
                    "H": H,
                    "ny": ny,
                    "sigma": sigma,
                    "correction": bool(corr),
                    "tau": TAU,
                    "nu": nu,
                    "g": G,
                    "b": float(inv["b"]),
                    "b_theory": float(bt),
                    "c_offset_in_b_times_H": float((inv["b"] - bt) * H),
                    "rel_err": float((inv["b"] - bt) / bt) if bt > 0 else None,
                    "curvature_ratio": float(inv["curvature_ratio"]),
                    "converged": bool(r.converged),
                    "steps": int(r.steps),
                    "residual": float(r.residual),
                    "wallclock_s": float(time.time() - t0),
                }
                if corr and sigma == 0.0:
                    prof = probe_parabola_residual(r)
                    row["parabola_std_offset"] = prof["std_offset"]
                    row["parabola_offset_over_g"] = prof["offset_over_g"]
                    row["kappa_closed_form"] = spurious_slip_coefficient(TAU)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if sigma == 0.0:
                    profiles[(H, sigma, corr)] = (np.asarray(r.coord), np.asarray(r.u))

    # ---- 判据汇总（零手抄：全部从 rows 里算）
    corrected = [r for r in rows if r["correction"]]
    baseline = [r for r in rows if not r["correction"]]
    err_c = [abs(r["rel_err"]) for r in corrected if r["rel_err"] is not None]
    # sigma -> 0 的振幅偏置 c：|b| 在 sigma=0.05 上的值（b_theory 已 < 0.16）
    c_rows = [r for r in corrected if r["sigma"] == 0.05]
    c_base = [r for r in baseline if r["sigma"] == 0.05]
    sigma0 = [r for r in corrected if r["sigma"] == 0.0]

    # 收敛阶：基线偏差 * H 应为常值（O(1/H) 偏置），修正后偏差应为舍入级
    order = []
    for sigma in (0.05, 0.2):
        eb = [r for r in baseline if r["sigma"] == sigma]
        ec = [r for r in corrected if r["sigma"] == sigma]
        eb.sort(key=lambda r: r["H"])
        ec.sort(key=lambda r: r["H"])
        for a, b, c1, c2 in zip(eb[:-1], eb[1:], ec[:-1], ec[1:]):
            order.append(
                {
                    "sigma": sigma,
                    "H_pair": [a["H"], b["H"]],
                    "baseline_err_ratio": abs(a["rel_err"]) / abs(b["rel_err"]),
                    "corrected_err_ratio": (
                        abs(c1["rel_err"]) / abs(c2["rel_err"])
                        if c2["rel_err"] not in (None, 0.0)
                        else None
                    ),
                    "baseline_c_times_H": [a["c_offset_in_b_times_H"], b["c_offset_in_b_times_H"]],
                    "corrected_c_times_H": [c1["c_offset_in_b_times_H"], c2["c_offset_in_b_times_H"]],
                }
            )

    summary = {
        "max_rel_err_corrected": float(max(err_c)),
        "max_abs_b_sigma0_corrected": float(max(abs(r["b"]) for r in sigma0)),
        "max_rel_err_baseline": float(
            max(abs(r["rel_err"]) for r in baseline if r["rel_err"] is not None)
        ),
        "c_times_H_baseline_sigma0p05": float(np.mean([r["c_offset_in_b_times_H"] for r in c_base])),
        "c_times_H_corrected_sigma0p05": float(np.mean([r["c_offset_in_b_times_H"] for r in c_rows])),
        "sigma0p05_min_H": float(min(r["H"] for r in c_rows)),
        "C1_threshold_0p01_over_H": float(0.01 / min(r["H"] for r in c_rows)),
        "max_abs_bias_corrected": float(
            max(abs(r["b"] - r["b_theory"]) for r in corrected)
        ),
        "max_abs_bias_times_H_corrected": float(
            max(abs(r["b"] - r["b_theory"]) * r["H"] for r in corrected)
        ),
        "max_abs_bias_times_H_baseline": float(
            max(abs(r["b"] - r["b_theory"]) * r["H"] for r in baseline)
        ),
        "C1_pass": bool(all(abs(r["b"] - r["b_theory"]) * r["H"] < 0.01 for r in corrected)),
        "C2_pass": bool(all(abs(r["rel_err"]) < 0.03 for r in corrected if r["sigma"] > 0)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "curvature_ratio_range": [float(min(r["curvature_ratio"] for r in rows)),
                                  float(max(r["curvature_ratio"] for r in rows))],
        "order": order,
    }

    out = {
        "node": "p247w1-N10",
        "source": "m7963-c1",
        "tau": TAU,
        "nu": nu,
        "g": G,
        "sigmas": list(SIGMAS),
        "H_grid": list(HS),
        "kappa_closed_form": spurious_slip_coefficient(TAU),
        "spurious_slip_velocity": float(G * spurious_slip_coefficient(TAU)),
        "rows": rows,
        "summary": summary,
    }
    (ART / "p247w1_n10_c6_grid.json").write_text(json.dumps(out, indent=2))
    _figures(out, profiles, nu)
    return out


def _figures(out: dict, profiles: dict, nu: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    # 图 1：b(sigma) 曲线（H=64）：基线 / 二阶重构 / 理论
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    H = 64
    th = [slip_law_theory(s, nu) for s in out["sigmas"] if s < 1]
    ss = [s for s in out["sigmas"] if s < 1]
    ax.plot(ss, th, "k-", lw=1.2, label=r"theory $b=3\nu\sigma/(1-\sigma)$")
    for corr, style, lab in ((False, "o--", "baseline (mixed BB/specular)"),
                             (True, "s-", "2nd-order reconstruction")):
        rs = [r for r in rows if r["H"] == H and r["correction"] is corr and r["sigma"] < 1]
        rs.sort(key=lambda r: r["sigma"])
        ax.plot([r["sigma"] for r in rs], [r["b"] for r in rs], style, ms=4, label=lab)
    ax.set_xlabel(r"$\sigma$ (specular fraction)")
    ax.set_ylabel(r"$b$  [lattice units]")
    ax.set_title(f"slip length vs $\\sigma$  (H={H}, $\\tau$={out['tau']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p247w1_n10_b_sigma.png", dpi=140)
    plt.close(fig)

    # 图 2：偏置 c*H 随 H（基线常数 = O(1/H) 偏置；修正后塌到 0）
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for corr, style, lab in ((False, "o--", "baseline"), (True, "s-", "reconstructed")):
        rs = sorted([r for r in rows if r["correction"] is corr and r["sigma"] == 0.05],
                    key=lambda r: r["H"])
        ax.semilogy([r["H"] for r in rs],
                    [max(abs(r["c_offset_in_b_times_H"]), 1e-12) for r in rs],
                    style, label=lab)
    ax.axhline(0.01, color="r", ls=":", label=r"$C1$ threshold $0.01$")
    ax.set_xlabel("H"); ax.set_ylabel(r"$|b-b_{theory}|\,\cdot H$")
    ax.set_title("O(1/H) amplitude bias: before vs after")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "p247w1_n10_bias_vs_H.png", dpi=140)
    plt.close(fig)

    # 图 3：剖面残差（σ=0）：基线常值偏移，重构后到舍入
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for corr, style, lab in ((False, "o--", "baseline"), (True, "s-", "reconstructed")):
        rs = [r for r in rows if r["H"] == 64 and r["sigma"] == 0.0 and r["correction"] is corr]
        if not rs:
            continue
        key = (64, 0.0, corr)
        if key in profiles:
            y, u = profiles[key]
            ana = analytic_plane_profile(y, 64.0, nu, out["g"], 0.0)
            ax.plot(y, (u - ana) / out["g"], style, ms=3,
                    label=f"{lab} (H=64, $\\sigma$=0)")
    ax.set_xlabel("y [lattice]"); ax.set_ylabel(r"$(u-u_{ana})/g$")
    ax.set_title("bulk residual: constant shift vs zero")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p247w1_n10_profile_residual.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["summary"], indent=2))