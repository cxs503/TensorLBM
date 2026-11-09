"""Build figures + paper-grade report payload for the phase-change node."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.phase_change.lee_model import PhaseChangeParameters, ThermalLBE2D
from tensorlbm.phase_change.stefan_anchors import (
    anchor_radius,
    exact_stefan_coefficient,
    heat_flux_estimate_coefficient,
    similarity_temperature_profile,
)

ROOT = Path("artifacts/phase_change_lee")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ALPHA = 0.25
CAMP = json.loads((ROOT / "campaign.json").read_text())
DT_PLANAR = CAMP["meta"]["dt_planar"]
DT_BUBBLE = CAMP["meta"]["dt_bubble"]


def series(name: str):
    s = CAMP["series"][name]
    return (np.array(s["time"], float), np.array(s["radius"], float),
            np.array(s["interface_temperature"], float))


def fig_planar() -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for dT in DT_PLANAR:
        t, R, _ = series(f"planar_dT{dT:g}_n1024")
        ax[0].plot(t, R - R[0], "-", label=f"LBM dT={dT:g} (Ja={dT:g})")
        tt = np.linspace(1.0, t[-1], 200)
        ax[0].plot(tt, anchor_radius(tt, dT, ALPHA, "planar"), "k:", lw=0.9)
        ax[0].plot(tt, exact_stefan_coefficient(dT, "planar") * np.sqrt(ALPHA * tt),
                   color="0.5", ls="--", lw=0.9)
    ax[0].set_xlabel("t [lattice steps]")
    ax[0].set_ylabel(r"$\Delta X$ [lattice units]")
    ax[0].set_title("1-D planar Stefan: LBM vs anchors")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    # log-log scaling
    for dT in DT_PLANAR:
        t, R, _ = series(f"planar_dT{dT:g}_n1024")
        ok = t > 0
        ax[1].loglog(t[ok], (R - R[0])[ok], "o", ms=2, label=f"dT={dT:g}")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$\Delta X$")
    ax[1].set_title(r"$\Delta X \propto t^{1/2}$ (slope 1/2 reference)")
    ref = np.array([1e2, 2e4])
    ax[1].loglog(ref, 0.02 * np.sqrt(ref), "k--", lw=1.0, label=r"$t^{1/2}$")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG / "fig1_planar_stefan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_bubble() -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for dT in DT_BUBBLE:
        n = max([g for g in CAMP["meta"]["grids_bubble"]
                 if f"bubble_dT{dT:g}_n{g}_s0" in CAMP["series"]])
        t, R, _ = series(f"bubble_dT{dT:g}_n{n}_s0")
        ax.plot(t, R - R[0], "-", marker=".", ms=2,
                label=f"LBM n={n} dT={dT:g}")
        tt = np.linspace(1.0, t[-1], 200)
        try:
            c = exact_stefan_coefficient(dT, "cylindrical")
            ax.plot(tt, c * np.sqrt(ALPHA * tt), "--", lw=0.8, color="0.5")
        except ValueError:
            pass
    ax.set_xlabel("t [lattice steps]")
    ax.set_ylabel(r"$\Delta R$ [lattice units]")
    ax.set_title("2-D single bubble: growth vs exact cylindrical self-similar\n"
                 "(grey dashed: exact; see table for deviations)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig2_bubble_growth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_coefficients() -> Path:
    ja = np.logspace(-2.2, -0.55, 60)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ja, [heat_flux_estimate_coefficient(j) for j in ja], "k--",
              label=r"campaign anchor $2Ja/\sqrt{\pi}$")
    for geom, style in (("planar", "-"), ("cylindrical", "-."), ("spherical", ":")):
        vals = []
        for j in ja:
            try:
                vals.append(exact_stefan_coefficient(j, geom))
            except ValueError:
                vals.append(np.nan)
        ax.loglog(ja, vals, style, lw=1.3, label=f"exact {geom}")
    meas = CAMP["cases"]
    for dT in DT_PLANAR:
        c = meas[f"planar_dT{dT:g}_n1024"]["final_coefficient"]
        ax.loglog(dT, c, "o", mfc="none", color="C0", ms=8)
    for dT in DT_BUBBLE:
        n = max([g for g in CAMP["meta"]["grids_bubble"]
                 if f"bubble_dT{dT:g}_n{g}_s0" in meas])
        c = meas[f"bubble_dT{dT:g}_n{n}_s0"]["final_coefficient"]
        ax.loglog(dT, c, "s", mfc="none", color="C1", ms=8)
    ax.loglog([], [], "o", mfc="none", color="C0", label="LBM planar (n=1024)")
    ax.loglog([], [], "s", mfc="none", color="C1", label="LBM 2-D bubble")
    ax.set_xlabel("Ja = c_p dT / h_fg")
    ax.set_ylabel(r"$C$ in $\Delta R = C\sqrt{\alpha t}$")
    ax.set_title("Self-similar coefficients: anchor vs exact vs LBM")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG / "fig3_coefficients.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fields() -> Path:
    """Temperature field + radial profile of a 2-D bubble run."""
    p = PhaseChangeParameters(alpha=ALPHA)
    sim = ThermalLBE2D(nx=128, ny=128, params=p, T_far=1.2, radius0=4.0, seed=0)
    for _ in range(900):
        sim.step()
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0))
    im = ax[0].imshow(sim.T - p.T_sat, cmap="inferno", origin="lower")
    ax[0].set_title(f"T - T_sat after 900 steps (R={sim.R:.2f})")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    r = np.arange(int(sim.R) + 1, 40)
    ax[1].plot(r - sim.R, sim.T[sim.ny // 2, int(sim.cx) + 1: int(sim.cx) + 1 + r.size]
               - p.T_sat, "o-", ms=3, label="LBM (dT=0.2)")
    eta = (r - sim.R) / (2.0 * math.sqrt(ALPHA * 900 * 0.35))
    ax[1].plot(r - sim.R, 0.2 * similarity_temperature_profile(eta, 0.2, "cylindrical"),
               "k--", lw=1.0, label="cylindrical similarity")
    ax[1].set_xlabel(r"$r - R$ [lattice units]")
    ax[1].set_ylabel(r"$T - T_{sat}$")
    ax[1].set_yscale("symlog", linthresh=1e-4)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("liquid-side radial profile")
    fig.tight_layout()
    out = FIG / "fig4_field_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def table() -> str:
    rows = ["| case | Ja | C_LBM | C_exact | rel.err | anchor err | exponent |",
            "|---|---|---|---|---|---|---|"]
    for dT in DT_PLANAR:
        c = CAMP["cases"][f"planar_dT{dT:g}_n1024"]
        rows.append(
            f"| planar dT={dT:g} (n=1024) | {dT:g} | {c['final_coefficient']:.5f} | "
            f"{exact_stefan_coefficient(dT,'planar'):.5f} | "
            f"{abs(c['final_coefficient']/exact_stefan_coefficient(dT,'planar')-1)*100:.2f}% | "
            f"{c['anchor_error']['max_rel_error']*100:.2f}% | {c['exponent']:.3f} |")
    for dT in DT_BUBBLE:
        n = max([g for g in CAMP["meta"]["grids_bubble"]
                 if f"bubble_dT{dT:g}_n{g}_s0" in CAMP["cases"]])
        c = CAMP["cases"][f"bubble_dT{dT:g}_n{n}_s0"]
        ce = exact_stefan_coefficient(dT, "cylindrical")
        rows.append(
            f"| bubble dT={dT:g} (n={n}) | {dT:g} | {c['final_coefficient']:.5f} | "
            f"{ce:.5f} | {abs(c['final_coefficient']/ce-1)*100:.2f}% | "
            f"{c['anchor_error']['max_rel_error']*100:.2f}% | {c['exponent']:.3f} |")
    return "\n".join(rows)


def main() -> None:
    f1, f2, f3, f4 = fig_planar(), fig_bubble(), fig_coefficients(), fig_fields()
    crit = CAMP["criteria"]
    crit_rows = ["| criterion | value | threshold | verdict |", "|---|---|---|---|"]
    for k, v in crit.items():
        val = v.get("value")
        thr = v.get("threshold", "—")
        pass_ = v.get("pass")
        verdict = "PASS" if pass_ else ("FAIL" if pass_ is False else "n/a")
        vtxt = f"{val:.3e}" if isinstance(val, (int, float)) else str(val)
        ttxt = f"{thr:.3g}" if isinstance(thr, (int, float)) else str(thr)
        crit_rows.append(f"| {k} | {vtxt} | {ttxt} | {verdict} |")
    md = f"""## Results

### Self-similar coefficients

{table()}

### Acceptance criteria (machine evaluated, `artifacts/phase_change_lee/campaign.json`)

{chr(10).join(crit_rows)}

C4 is flagged `independent_check=false`: the front-tracking closure obtains
`dR/dt` from the interface flux, so `h_fg dR = int(q dt)` is an identity; the
independent sensible-vs-latent enthalpy check is *not* available in this
implementation (diagnosed spurious ~50x energy exchange of the crude vapour
Dirichlet reset, see limitations).
"""
    (ROOT / "results.md").write_text(md)
    print(md)

    def b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode()

    report = {
        "title": "TensorLBM 相变验证面：Lee 模型骨架 + 单气泡/平面 Stefan 生长标度律与界面饱和一致性",
        "summary": (
            "在 TensorLBM 新建 phase_change 能力面：D2Q9/D1Q3 热 LBM + 前沿跟踪 Stefan 通量闭合 +"
            "有限速率 Lee 质量/能量源；建立三套几何（平面/柱/球）精确自相似锚与判据 C1–C7。"
            "平面一维：C=ΔX/√(αt) 与精确 Neumann 解误差 1.1–1.4%，与战役锚 2Ja√(αt/π) 误差 ≤2.4%，"
            "指数 0.509；网格 512/1024/2048 系数变化 0.17%；三种子离散 0.32%；负控 ΔR=0。"
            "二维单气泡：ΔR~t^0.67–0.76（非 0.5），系数较柱几何精确解低 1.2–26.9%；"
            "界面饱和判据在二维气泡上 honest FAIL。并给出解析结论：战役锚公式仅对平面小 Ja 成立。"
        ),
        "scenario_key": "lbm_phase_change_verification",
        "mission_id": 7466,
        "tags": ["TensorLBM", "相变", "Lee模型", "Stefan问题", "单气泡生长", "验证面"],
        "markdown": md,
        "figures": [
            {"name": "fig1_planar_stefan", "caption": "1-D 平面 Stefan：ΔX(t) 与战役锚/精确 Neumann 对比；右图双对数验证 t^0.5 标度",
             "png_base64": b64(f1)},
            {"name": "fig2_bubble_growth", "caption": "2-D 单气泡 ΔR(t)（四档过热度，灰虚线为柱几何精确自相似解）",
             "png_base64": b64(f2)},
            {"name": "fig3_coefficients", "caption": "自相似系数 C(Ja)：战役锚、平面/柱/球精确解与 LBM 实测点",
             "png_base64": b64(f3)},
            {"name": "fig4_field_profile", "caption": "二维气泡温度场云图（900 步）与液侧径向温度剖面 vs 柱自相似剖面",
             "png_base64": b64(f4)},
        ],
    }
    Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
    print("report payload written:", Path("/tmp/report.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()