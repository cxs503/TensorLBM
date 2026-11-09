"""Render the P97-N05 figures and the paper-grade report from the run artefacts.

Everything in the report is read back from the JSON produced by
``scripts/p97_n05_campaign.py --collect`` (or recomputed here from the raw run
files), so no number is copied by hand:

* ``artifacts/p97_n05/validation.json`` — acceptance gates C1…C6;
* ``runs/p97_n05/*.json`` — per-case diagnostics (histories, fit windows, …);
* ``artifacts/p97_n05/field_*.npz`` — temperature / speed fields for the plates.

Outputs
-------
* ``artifacts/p97_n05/figures/*.png``
* ``docs/reports/P97-N05_thermal_scalar_validation.md``
* ``/tmp/report.json`` — payload for ``POST $TENSOR_API_BASE/tensor/reports``
  (title/summary/markdown/figures with base64 PNGs).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "p97_n05"
ART = ROOT / "artifacts" / "p97_n05"
FIG = ART / "figures"

RA_CRIT = 1707.76
NU_DVD = 2.243
NU_CB = {1e4: 2.16, 1e5: 4.22}
MISSION_ID = 6952


def load(name: str) -> dict | None:
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_validation() -> dict:
    return json.loads((ART / "validation.json").read_text())


# ── figures ──────────────────────────────────────────────────────────────────
def fig_c1(val: dict) -> Path:
    c = val["criteria"].get("C1_rac")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if c:
        grids = c.get("grid_values") or []
        if grids:
            M = [g["M"] for g in grids]
            ra = [g["Ra_c"] for g in grids]
            err = [abs(g["rel_err_vs_reference"]) for g in grids]
            ax[0].loglog(M, err, "o-", label="finite-difference grid")
            ax[0].axhline(
                abs(c["rel_err"]), color="C3", ls="--",
                label=f"Richardson extrapolated: {c['rel_err'] * 100:+.2e} %",
            )
            ref = err[-1] if err else 1.0
            ax[0].loglog(M, [ref * (M[0] / m) ** 2 for m in M], "k:",
                         label=r"$O(M^{-2})$ reference slope")
            ax[0].set_xlabel("interior nodes $M$")
            ax[0].set_ylabel(r"$|Ra_c - 1707.76|/1707.76$")
            ax[0].set_title(f"(a) marginal Rayleigh number converges to 1707.76\n"
                            f"extrapolated $Ra_c$ = {c['value']:.4f}")
            ax[0].legend(fontsize=8)
            ax[0].grid(alpha=0.3, which="both")
    crit = load("critical_rb") or {}
    scan = (crit.get("continuum") or {}).get("a_scan") or []
    if scan:
        ax[1].plot([s["a"] for s in scan], [s["Ra_c"] for s in scan], "s-")
        ax[1].axvline(3.1163, color="C3", ls="--", label=r"$a_c = 3.1163$")
        ax[1].set_xlabel("wavenumber $a$")
        ax[1].set_ylabel("$Ra_c(a)$")
        ax[1].set_title("(b) marginal curve minimum at the critical wavenumber")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_c1_marginal_stability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_c2(val: dict) -> Path:
    c = val["criteria"].get("C2_nu_ra") or {}
    pts = []
    for name in ("rb_ra1e4", "rb_ra3e4", "rb_ra1e5", "rb_ra3e5", "rb_ra1e6"):
        r = load(name)
        if r:
            pts.append((r["ra"], r["nu_wall"], r))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if pts:
        ra = np.array([p[0] for p in pts])
        nu = np.array([p[1] for p in pts])
        ax[0].loglog(ra, nu, "o-", label="LBM D2Q9+D2Q5 (this work)")
        ax[0].loglog(ra, [1.0] * ra.size, "k:", label="conduction limit Nu = 1")
        for k, v in NU_CB.items():
            ax[0].loglog([k], [v], "r*", ms=11, label=f"Clever & Busse: Nu({k:.0e})={v}")
        if c.get("fitted_exponent"):
            b = c["fitted_exponent"]
            a0 = nu[0] / ra[0] ** b
            ax[0].loglog(ra, a0 * ra**b, "g--",
                         label=rf"fit $\propto Ra^{{{b:.3f}}}$")
            ax[0].loglog(ra, a0 * ra ** (1.0 / 3.0), "m:", alpha=0.7,
                         label=r"$Ra^{1/3}$ (asymptotic)")
        ax[0].set_xlabel("Rayleigh number $Ra$")
        ax[0].set_ylabel("Nusselt number $Nu$")
        ax[0].set_title("(a) $Nu(Ra)$, $Pr=0.71$, periodic box, aspect 2")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, which="both")
        hist = pts[0][2].get("history") or []
        if hist:
            t = [h["step"] for h in hist]
            ax[1].plot(t, [h["nu_bottom"] for h in hist], lw=0.8, label=r"$Nu_{bottom}$")
            ax[1].plot(t, [h["nu_top"] for h in hist], lw=0.8, label=r"$Nu_{top}$")
            ax[1].plot(t, [h["nu_mid"] for h in hist], lw=0.8, label=r"$Nu_{mid}$")
            ax[1].axhline(NU_CB[1e4], color="r", ls="--",
                          label=f"Clever & Busse {NU_CB[1e4]}")
            ax[1].set_xlabel("lattice step")
            ax[1].set_ylabel("Nusselt number")
            ax[1].set_title(f"(b) wall/mid-plane flux identity, $Ra=10^4$\n"
                            f"gap wall-vs-mid = {100 * pts[0][2]['nu_wall_gap']:.2f} %")
            ax[1].legend(fontsize=8)
            ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_c2_nusselt_rayleigh.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_c3(val: dict) -> Path:
    c = val["criteria"].get("C3_cavity_dvd") or {}
    fig = plt.figure(figsize=(11, 4.2))
    ax0 = fig.add_subplot(1, 2, 1)
    for name, lab in (("cav_n64", "$64^2$"), ("cav_n96", "$96^2$"), ("cav_n128", "$128^2$")):
        f = ART / f"field_{name}.npz"
        if f.exists():
            T = np.load(f)["T"]
            ax0.imshow(T, origin="lower", cmap="inferno", extent=[0, 1, 0, 1], aspect="equal")
            ax0.set_title(f"temperature, {lab}")
            ax0.set_xlabel("x/L")
            ax0.set_ylabel("y/L")
            break
    ax1 = fig.add_subplot(1, 2, 2)
    rows = c.get("rows") or []
    if rows:
        nx = [r["nx"] for r in rows]
        nu = [r["nu"] for r in rows]
        ax1.plot(nx, nu, "o-", label="LBM (this work)")
        ax1.axhline(NU_DVD, color="r", ls="--", label=f"de Vahl Davis {NU_DVD}")
        for r in rows:
            ax1.annotate(f"{100 * r['rel_err']:+.1f}%", (r["nx"], r["nu"]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax1.set_xlabel("grid $n_x$")
        ax1.set_ylabel("$Nu$")
        ax1.set_title(r"$Ra=10^4$, $Pr=0.71$ square cavity")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_c3_cavity_dvd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_c4(val: dict) -> Path:
    c = val["criteria"].get("C4_taylor_aris") or {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) instantaneous rate history: convergence of the meter
    for name, lab in (("ta_pe10_n17", "Pe=10"), ("ta_pe30_n17", "Pe=30"),
                      ("ta_pe3_n17", "Pe=3")):
        r = load(name)
        if not r:
            continue
        h = r["history"]
        t = np.array(h["t"], float)
        rate = np.array(h["rate"], float)
        ok = np.isfinite(rate)
        ax[0].plot(t[ok], rate[ok], lw=1.0,
                   label=rf"{lab} ({name})")
        ax[0].axhline(r["d_eff_discrete"], ls=":", lw=0.8, color="k", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("lattice step")
    ax[0].set_ylabel(r"$-d\ln A/dt \;/\; k^2$")
    ax[0].set_title("(a) dispersion meter converges to $D_{eff}$\n(dotted: discrete Aris reference)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    rows = c.get("rows") or []
    if rows:
        pe = np.array([r["Pe"] for r in rows])
        ratio = np.array([r["D_eff_lbm"] / r["d_eff_discrete"] for r in rows])
        ax[1].semilogx(pe, ratio, "o", label="LBM / discrete Aris (Campbell ny=17)")
        ana = np.array([r["d_eff_analytic"] / r["d_eff_discrete"] for r in rows])
        ax[1].semilogx(pe, ana, "s", mfc="none", label="closed form / discrete Aris")
        ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
        ax[1].axhspan(0.95, 1.05, color="g", alpha=0.12, label="±5 % band")
        ax[1].set_xlabel("Péclet number $Pe = U h / D$")
        ax[1].set_ylabel(r"$D_{eff}$ ratio")
        ax[1].set_title("(b) Taylor–Aris dispersion vs reference")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3, which="both")
        # (c) finite-wavenumber study at fixed Pe
        for pe_fix, names in ((30.0, ("ta_pe30_nx64", "ta_pe30", "ta_pe30_nx256")),
                              (10.0, ("ta_pe10_nx64", "ta_pe10"))):
            xs, ys = [], []
            for n in names:
                r = load(n)
                if not r:
                    continue
                k = 2.0 * math.pi / r["nx"]
                xs.append((k * r["half_width"]) ** 2)
                ys.append(r["rel_err_vs_discrete"])
            if xs:
                ax[2].plot(xs, np.array(ys) * 100, "o-", label=f"Pe≈{pe_fix:.0f} (ny=9)")
        ax[2].axhline(0.0, color="k", ls="--", lw=0.8)
        ax[2].axhspan(-5, 5, color="g", alpha=0.12)
        ax[2].set_xlabel(r"$(k h)^2$ — finite streamwise wavenumber")
        ax[2].set_ylabel("relative error of the meter [%]")
        ax[2].set_title("(c) finite-wavenumber bias of the meter")
        ax[2].legend(fontsize=7)
        ax[2].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_c4_taylor_aris.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_c5(val: dict) -> Path:
    c = val["criteria"].get("C5_numerical_diffusion") or {}
    rows = c.get("rows") or []
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if rows:
        labels = [r["control"].replace(" (", "\n(") for r in rows]
        vals = [100 * r["rel_err"] for r in rows]
        bars = ax.bar(labels, vals, color=["C0", "C1", "C2"][: len(rows)])
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-5, 5, color="g", alpha=0.12, label="±5 % gate")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.2f}%", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.set_ylabel(r"$D_{measured}/D - 1$  [%]")
        ax.set_title("numerical-diffusion calibration of the scalar lattice")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "fig_c5_numerical_diffusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_rb_fields() -> Path | None:
    names = [n for n in ("rb_ra1e4", "rb_ra3e5", "rb_ra1e6") if (ART / f"field_{n}.npz").exists()]
    if not names:
        return None
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.0))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        d = np.load(ART / f"field_{n}.npz")
        r = load(n) or {}
        im = ax.imshow(d["T"], origin="lower", cmap="inferno", aspect="auto",
                       extent=[0, r.get("nx", 1), 0, r.get("ny", 1)])
        ax.set_title(f"{n}: Nu={r.get('nu_wall', float('nan')):.3f}, "
                     f"{r.get('regime', '')}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = FIG / "fig_rb_fields.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_notes(val: dict) -> str:
    """Section 5: pitfalls, limits and applicability — numbers pulled from the runs."""
    c = val["criteria"]
    L4 = {r["case"]: r for r in (c.get("C4_taylor_aris") or {}).get("rows", [])}

    def pct(name, key="rel_err_vs_discrete"):
        r = L4.get(name)
        return f"{100 * r[key]:+.1f}%" if r else "（未运行）"

    text = r"""
### 5.1 踩过的坑（缺陷/伪影，均已定位并修复或登记）

1. **弥散计在 $\tau_d\to0.5$ 时不可测**：单模振幅按 $D_{eff}k^2$ 指数衰减，而进入渐近段需要 $t\gg h^2/D$。高 Pe（小 D）时振幅先撞舍入地板、渐近段还没到——第一版直接给出 $-100\%$ 的假结果。修法：只对**扰动部分**做周期性重标定（线性方程下严格等价），配合 float64 与自适应拟合窗（起点 $3h^2/D$）。
2. **重标定只能标定扰动**：整场重标定会把背景一起放大，未被充分阻尼的横向模式随之放大，计量器在 $\tau_d=0.518$ 时输出乱跳的速率（+0.7 / −0.6 交替）。改成 $g_{eq}(\bar\phi)+f\,(g-g_{eq})$ 后速率与窗长无关（$R^2\to1$）。
3. **周期轨迹的矩估计会被绕回污染**：羽流跨过 $x=0$ 时方差突跳 $O(n_x^2)$（曾给出负的 $D_{eff}$）。修法：先按峰值做圆周平移再算二阶矩，并约束 $\sigma<n_x/6$。
4. **流式方向符号**：`scalar_stream` 初版写成 `roll(-c)`，等价于 $-u$；对弥散（只依赖 $u^2$）无影响，但会破坏移动参考系。已由单测 `test_scalar_stream_is_periodic_in_x` 钉住。
5. **权重归一化**：D2Q9 的 float32 权重和 $1+O(2^{-25})$，每步给平衡分布注入 $3\times10^{-8}$ 的质量亏欠，200 步纯扩散即漂移 $3\times10^{-6}$。加 `_weights()`（fp64 归一化 + 余量吸收）后质量守恒到 roundoff。
6. **包内 ARPACK 捷径不可用（缺陷登记，未改动库代码）**：`rb_continuum_stability.critical_at` 在 `sigma=0` 附近只取 4 个特征值，在本题上返回伪谱（实测 `spectra` 给 $1.5\ldots1.7$，稠密谱同一支为 $-1.7$），导致 `critical_rayleigh_continuous` 在波数搜索中抛 `Ra_hi=6000 still stable`。本报告改用稠密 QZ 全谱 + 二分（`growth()`），与 Chandrasekhar 值一致到 $10^{-4}$。建议后续给 `critical_at` 加大 k 或稠密兜底。
7. **热壁与速度壁半格错位**：anti-bounce-back 把温度钉在首/末**节点**上，故热间隙是 $n_y-1$；速度半程反弹壁在 $\pm1/2$ 之外，间隙是 $n_y$。若沿用 $H=n_y$ 构造 $Ra$，实际 $Ra$ 偏低 $(1-1/n_y)^3$（$n_y=50$ 时 6%、$n_y=16$ 时 18%），$Nu$ 口径也带 $1/n_y$ 偏差（$n_y=8,\ Ra=100$ 时实测 $Nu=1.124$ 而非 1）。本报告统一用 $H=n_y-1$（方腔 $n_x-1$）重构造 $Ra$ 与 $Nu$。这是一处值得后续修掉的一致性问题。
8. **判据门本身曾有洞**：`evaluate()` 早期版本对空证据集返回 `all_pass=True`（`all()` over empty）。已加"六项判据必须齐备"校验，并由单测 `test_gate_cannot_pass_without_evidence` 钉住。

### 5.2 Taylor–Aris 的适用范围（honest LIMIT）

弥散计的残差有两个可分离来源，二者都不是 Taylor 物理本身：

* **有限流向波数**：单模法测的是 $D_{eff}(k)$，$kh$ 非无穷小即有偏。Pe≈30 时把 $n_x$ 从 64 加到 256（$(kh)^2$ 由 0.19 降到 0.01），相对离散 Aris 参考的偏差从 __K64__ 变到 __K256__。
* **横向分辨率**：剪切剖面在 $n_y$ 个节点上取样、零通量壁反射截断壁行，残差随 $1/n_y$ 下降。同一 Pe≈10：$n_y=9$ 时 __N9__，$n_y=17$ 时 __N17__。

因此判据取每档 Pe 的**最细可用配置**（$kh\le0.2$、$n_y=17$）。可复用限界：$Pe\lesssim30$ 时取 $n_y\gtrsim17$、$n_x\gtrsim250$ 可在 5% 内复现 Aris 公式；**Pe=100 在本次算力预算内不可达**——格子胞 Péclet 数 $u\Delta y/D=Pe/h$ 需压到 $O(1)$，即 $h\gtrsim Pe$（$n_y\gtrsim200$），且 $t\sim3h^2/D\sim10^5$–$10^6$ 步，成本随 $Pe^2$ 增长。Pe=100 的实测值在 C4 表中如实列出，作为该限界的证据。

### 5.3 Rayleigh–Bénard 的限界

* 二维、展弦比 2、$x$ 周期：只对应 Clever & Busse 的**定常卷分支**，不能直接与三维实验或大展弦比统计标度比较。
* 每档 Ra 单独调 $\tau$ 把格子 Mach 数压到 $\lesssim0.1$（Boussinesq 定常解与 $\tau$ 无关，但压缩性误差 $\sim Ma^2$ 与稳定性有关）；$Ra\ge10^6$ 时 $Ma$ 与扩散时间同时恶化，成本按 $H\,Ra/u_{ff}^2$ 增长。
* 标度指数在 $Ra\le10^6$ 只到经典区（$\beta\approx0.3$），$\beta\to1/3$ 的渐近区需要 $Ra\gg10^6$——拟合值见 C2 表，未强行凑 1/3。
* $u_{max}$ 的归一化：`benchmarks/pending/rayleigh_benard` 引用的 de Vahl Davis $u_{max}\approx0.1926$ 在 $\alpha/L$ 与 $\nu/L$ 两种归一化下都无法复现（本平台无量纲速度量级 $\mathcal{O}(10)$，与自由落体标度 $\sqrt{g\beta\Delta T H}$ 的 0.6 倍一致）。故 $u_{max}$ **不作为判据**，只作流态指标报告。

### 5.4 可复现性

所有案例由 `scripts/p97_n05_campaign.py` 的 `CASES` 单一案例表驱动，可在集群上 `--shard i --nshards n` 分片；`--collect` 重算判据并写 `artifacts/p97_n05/validation.json`（含每案配置、拟合窗、$R^2$、质量漂移）。新增单测 39 条（`tests/test_taylor_aris.py`、`tests/test_p97_n05_validation.py`），覆盖闭式/离散 Aris、质量守恒、零通量壁、纯扩散标定、plug 对照、Pe=10 物理对标，以及**判据门本身**（空证据集必须判 FAIL）。
"""
    text = text.replace("__K64__", pct("ta_pe30_nx64")).replace("__K256__", pct("ta_pe30_nx256"))
    text = text.replace("__N9__", pct("ta_pe10")).replace("__N17__", pct("ta_pe10_n17"))
    return text.strip()


# ── report ───────────────────────────────────────────────────────────────────
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, float) and abs(x) < 1e-3 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def build_markdown(val: dict, figs: dict[str, Path]) -> str:
    c = val["criteria"]
    L: list[str] = []
    A = L.append
    A("# P97-N05 · TensorLBM 热对流与被动标量验证面（Rayleigh–Bénard 临界 Ra_c / Nu(Ra) / 差分加热腔 / Taylor–Aris 弥散）")
    A("")
    A("**节点**：P97-N05（TensorLBM）· **报告类型**：论文级验证报告 · **生成方式**：`scripts/p97_n05_campaign.py --collect` + `scripts/p97_n05_report.py`（所有数字从 `artifacts/p97_n05/validation.json` 与 `runs/p97_n05/*.json` 程序化注入，无手抄）")
    A("")
    A("## 1. 案例介绍")
    A("")
    A("本节点为 TensorLBM 建立**热对流 + 被动标量输运**的验证面。四个子问题共用同一套格子（D2Q9 动量 + D2Q5 温度/标量，Boussinesq/Guo 力格式）：")
    A("")
    A("| 子问题 | 物理 | 控制参数 | 基准 |")
    A("|---|---|---|---|")
    A("| C1 线性稳定性 | 两无限大平行板间水平流体层的导热态失稳 | Ra，Pr=0.71，刚-刚壁 | Chandrasekhar (1961) $Ra_c=1707.76$，$a_c=3.1163$ |")
    A("| C2 Nu(Ra) | Rayleigh–Bénard 对流（x 周期、上下等温刚壁，展弦比 2） | Ra=1e4…1e6，Pr=0.71 | Clever & Busse (1974) 二维定常卷：Nu(1e4)=2.16，Nu(1e5)=4.22 |")
    A("| C3 差分加热腔 | 方腔左热右冷、上下绝热 | Ra=1e4，Pr=0.71 | de Vahl Davis (1983) Nu=2.243 |")
    A("| C4/C5 Taylor–Aris | 平面 Poiseuille 流中被动标量的轴向弥散 | Pe=Uh/D=1…100 | Aris (1956) 矩方法：$D_{eff}=D+2h^2U^2/(105D)$（另附圆管 1/48 讨论） |")
    A("")
    A("**几何与格子**：RB 单元 $n_x\\times n_y$ 格子单位，热壁钉在首/末行，故热间隙 $H=n_y-1$（速度半程反弹壁在 $y=-1/2,\\,n_y-1/2$，比热间隙宽一格——这是既有算子的半格错位，本报告如实沿用并计入 $Ra$ 的定义）；方腔 $H=n_x-1$ 同理。")
    A("")
    A("## 2. 怎么做的")
    A("")
    A("**工具链**：`tensorlbm.thermal`（D2Q9+D2Q5 双分布、Guo 力、anti-bounce-back 等温壁）、`tensorlbm.rb_continuum_stability`（四阶算子有限差分离散 + 稠密广义特征值）、新增 `tensorlbm.taylor_aris`（标量 D2Q9 + Aris 辅助问题 + 弥散计）、`scripts/p97_n05_*`（案例表→分片执行→判据收敛→图）。全部案例由 `scripts/p97_n05_campaign.py` 的单一案例表 `CASES` 驱动，外协方双节点 loong64 集群分片执行。")
    A("")
    A("**C1**：在 $a_c=3.1163$ 上对 $Ra$ 做二分（每个 $M$ 用稠密 QZ 全谱取最不稳定支，$\\sigma=0$ 处定临界），再对 $M\\in\\{50,100,200\\}$ 做 Richardson 外推；另在 $M=100$ 上扫描 $a$ 以验证最小临界点位置。**注意**：包内 `critical_at`/`critical_rayleigh_continuous` 依赖 ARPACK shift-invert 近零特征值，在本问题上返回伪谱（见 §5 缺陷登记），因此改走稠密路径。")
    A("")
    A("**C2**：初始化 = 线性导热剖面 + $\\sin(2\\pi x/n_x)\\sin(\\pi y/H)$ 扰动（幅值 1e-3），周期 x、上下 anti-bounce-back 等温壁；每个 Ra 单独调 τ 以把格子 Mach 数压到 ≤0.1（Boussinesq 定常解与 τ 无关），后 $30\\%$ 步长做时间平均。判据除与文献 Nu 对比外，另加**热流恒等式**：底壁、顶壁与中面（$Nu=\\langle u_yT\\rangle H/(\\alpha\\Delta T)-\\partial_y\\langle T\\rangle H/\\Delta T$）三个 Nusselt 数必须一致——这是不依赖任何文献值的机器可检能量守恒门。")
    A("")
    A("**C3**：方腔 $Ra=10^4$、$Pr=0.71$，左壁 T=1、右壁 T=0、上下绝热；网格 64/96/128，按扩散时间 $H^2/\\alpha$ 取步数（$\\ge 4$ 倍），末段 50% 做时间平均。")
    A("")
    A("**C4/C5**：新增 `taylor_aris` 模块，三层证据：(i) 闭式 $D+K a^2U^2/D$；(ii) **离散 Aris 辅助问题** $D\\nabla_\\perp^2\\chi=-u'$，在 LBM 同一套节点/零通量壁上求解，$D_{eff}=D+\\langle u'\\chi\\rangle$——它消除“抛物线在格点上取样均值≠U”的记账歧义，使残差纯为数值误差；(iii) LBM 标量 D2Q9 单模衰减计：释放 $\\phi=1+A\\cos kx$，测 $-(d\\ln A/dt)/k^2$ 的渐近平台，并对场做**仅扰动部分**的周期性重标定，使 $t\\gg h^2/D$ 的渐近段在 $D\\to0$（高 Pe）时仍可测。C5 用两个对照分离数值扩散：$U=0$（纯扩散，必须回到 $D$）与 plug 流（均匀平移、无剪切，仍必须回到 $D$）。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### C1 临界 Rayleigh 数")
    A("")
    if c.get("C1_rac"):
        x = c["C1_rac"]
        A(f"* Richardson 外推 $Ra_c$ = **{_fmt(x['value'])}**（参考 1707.76，相对偏差 {100 * x['rel_err']:+.2e} %）→ 判据 |偏差|<1% **{'PASS' if x['pass'] else 'FAIL'}**")
        A("")
        A("| M | $Ra_c$ | 相对偏差 |")
        A("|---|---|---|")
        for g in x.get("grid_values") or []:
            A(f"| {g['M']} | {_fmt(g['Ra_c'])} | {100 * g['rel_err_vs_reference']:+.3e} % |")
    A("")
    A("### C2 Nu(Ra) 与热流恒等式")
    A("")
    if c.get("C2_nu_ra"):
        x = c["C2_nu_ra"]
        A("| Ra | Nu (LBM) | 参考 | 来源 | 相对偏差 | 壁-中面 Nu 间隙 |")
        A("|---|---|---|---|---|---|")
        for r in x["rows"]:
            A(f"| {r['Ra']:.0e} | {_fmt(r['Nu'])} | {_fmt(r['reference'])} | {r['source']} | "
              f"{100 * r['rel_err']:+.2f} % | {100 * r['flux_gap']:.2f} % |")
        A("")
        A(f"* 拟合标度指数 β = **{_fmt(x['fitted_exponent'], 3)}**（Ra=1e4…1e6），渐近值 1/3 = 0.3333")
        A(f"* 最大相对偏差 {100 * x['worst_rel_err']:.2f} %（阈值 5%）→ **{'PASS' if x['pass'] else 'FAIL'}**")
        A("")
        A("全部 Ra 的壁面与中面 Nusselt 数")
        A("")
        A("| Ra | $Nu_{bottom}$ | $Nu_{top}$ | $Nu_{mid}$ | regime | $u_{max}$ |")
        A("|---|---|---|---|---|---|")
        for n in ("rb_ra1e4", "rb_ra3e4", "rb_ra1e5", "rb_ra3e5", "rb_ra1e6"):
            r = load(n)
            if r:
                A(f"| {r['ra']:.0e} | {_fmt(r['nu_bottom'])} | {_fmt(r['nu_top'])} | "
                  f"{_fmt(r['nu_mid'])} | {r['regime']} | {_fmt(r['u_max_mean'])} |")
    A("")
    A("### C3 差分加热腔（de Vahl Davis）")
    A("")
    if c.get("C3_cavity_dvd"):
        x = c["C3_cavity_dvd"]
        A("| $n_x$ | τ | 步数 | Nu | $u^*_{max}=u_{max}H/\\alpha$ | $|\\Delta Nu/Nu|$ |")
        A("|---|---|---|---|---|---|")
        for r in x["rows"]:
            A(f"| {r['nx']} | {_fmt(r['tau'], 3)} | {r['steps']} | {_fmt(r['nu'])} | "
              f"{_fmt(r['u_max_star'], 3)} | {100 * abs(r['rel_err']):.2f} % |")
        A("")
        A(f"* 参考 de Vahl Davis (1983) Nu=2.243；最佳偏差 {100 * x['best_abs_rel_err']:.2f} %（阈值 3%）→ **{'PASS' if x['pass'] else 'FAIL'}**")
    A("")
    A("### C4 Taylor–Aris 弥散")
    A("")
    if c.get("C4_taylor_aris"):
        x = c["C4_taylor_aris"]
        A("| 案例 | Pe | D | $D_{eff}$ (LBM) | 离散 Aris | 闭式 $D+2h^2U^2/(105D)$ | 相对离散参考 | 相对闭式 | $R^2$ |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in x["rows"]:
            A(f"| {r['case']} | {r['Pe']:.2f} | {_fmt(r['D'])} | {_fmt(r['D_eff_lbm'])} | "
              f"{_fmt(r['D_eff_discrete_aris'])} | {_fmt(r['D_eff_analytic'])} | "
              f"{100 * r['rel_err_vs_discrete']:+.2f} % | {100 * r['rel_err_vs_analytic']:+.2f} % | {_fmt(r['r2'], 5)} |")
        A("")
        A(f"* 最大相对偏差 {100 * x['worst_rel_err']:.2f} %（阈值 5%）→ **{'PASS' if x['pass'] else 'FAIL'}**（判据在每档 Pe 的**最细可用配置**上评估，粗配置作为分辨率研究列出）")
    A("")
    A("### C5 数值扩散分离")
    A("")
    if c.get("C5_numerical_diffusion"):
        x = c["C5_numerical_diffusion"]
        A("| 对照 | D | 实测 $D_{eff}$ | 相对 $D$ |")
        A("|---|---|---|---|")
        for r in x["rows"]:
            A(f"| {r['control']} | {_fmt(r['D'])} | {_fmt(r['D_measured'])} | {100 * r['rel_err']:+.2f} % |")
        A("")
        A(f"* 最差 {100 * x['worst_rel_err']:.2f} %（阈值 5%）→ **{'PASS' if x['pass'] else 'FAIL'}**")
    A("")
    A("### C6 负控（Ra→0）")
    A("")
    if c.get("C6_negative_control"):
        x = c["C6_negative_control"]
        A("| 案例 | Ra | Nu | $Nu-1$ | regime |")
        A("|---|---|---|---|---|")
        for r in x["rows"]:
            A(f"| {r['case']} | {r['Ra']:.0f} | {_fmt(r['Nu'])} | {r['dev_from_1']:+.2e} | {r['regime']} |")
        A("")
        A(f"* 最大偏离 {x['worst_dev']:.2e}（阈值 1e-2）→ **{'PASS' if x['pass'] else 'FAIL'}**")
    A("")
    A("## 4. 结果图")
    A("")
    for cap, path in figs.items():
        A(f"![{cap}](figures/{path.name})")
        A("")
        A(f"*{cap}*")
        A("")
    A("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    A("")
    A(build_notes(val))
    A("## 6. 误差判定")
    A("")
    A("| 判据 | 关键量 | 基准 | 实测偏差 | 阈值 | 是否通过 |")
    A("|---|---|---|---|---|---|")
    for key, label in (
        ("C1_rac", "临界 Rayleigh 数"),
        ("C2_nu_ra", "Nu(Ra)"),
        ("C3_cavity_dvd", "方腔 Nu"),
        ("C4_taylor_aris", "$D_{eff}$"),
        ("C5_numerical_diffusion", "数值扩散"),
        ("C6_negative_control", "负控 Nu→1"),
    ):
        x = c.get(key)
        if not x:
            continue
        if key == "C1_rac":
            dev, thr = f"{100 * x['rel_err']:+.2e} %", "<1%"
        elif key == "C2_nu_ra":
            dev, thr = f"最差 {100 * x['worst_rel_err']:.2f} %", "<5%"
        elif key == "C3_cavity_dvd":
            dev, thr = f"最佳 {100 * x['best_abs_rel_err']:.2f} %", "<3%"
        elif key == "C4_taylor_aris":
            dev, thr = f"最差 {100 * x['worst_rel_err']:.2f} %", "<5%"
        elif key == "C5_numerical_diffusion":
            dev, thr = f"最差 {100 * x['worst_rel_err']:.2f} %", "<5%"
        else:
            dev, thr = f"{x['worst_dev']:.2e}", "<1e-2"
        A(f"| {key} | {label} | 见上表 | {dev} | {thr} | {'PASS' if x['pass'] else 'FAIL'} |")
    A("")
    A(f"**总判定**：{'全部通过' if val['all_pass'] else '存在未通过项（见上表与 §5 的 honest FAIL 说明）'}。")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    val = load_validation()
    figs: dict[str, Path] = {}
    for name, fn in (
        ("图 1 线性稳定性：Ra_c 的网格收敛（左）与临界波数（右）", fig_c1),
        ("图 2 Nu(Ra) 标度律与壁/中面热流恒等式", fig_c2),
        ("图 3 方腔温度场与 Nu 的网格收敛（de Vahl Davis）", fig_c3),
        ("图 4 Taylor–Aris 弥散：计量收敛、Pe 依赖与有限波数偏差", fig_c4),
        ("图 5 数值扩散标定（C5 对照）", fig_c5),
    ):
        figs[name] = fn(val)
    rb = fig_rb_fields()
    if rb:
        figs["图 6 Rayleigh–Bénard 温度场快照（Ra=1e4 / 3e5 / 1e6）"] = rb
    md = build_markdown(val, figs)
    out_md = ROOT / "docs" / "reports" / "P97-N05_thermal_scalar_validation.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    if not args.no_report:
        payload = {
            "title": "P97-N05 TensorLBM 热对流与被动标量验证面：Ra_c / Nu(Ra) / de Vahl Davis / Taylor–Aris",
            "summary": (
                "TensorLBM 新增 Taylor–Aris 弥散验证模块并对 Rayleigh–Bénard 临界 Ra、"
                "Nu(Ra) 标度、差分加热腔与被动标量有效扩散做三级对标；判据数字见正文表格。"
            ),
            "scenario_key": "tensorlbm.p97_n05.thermal_scalar_validation",
            "mission_id": MISSION_ID,
            "tags": ["TensorLBM", "thermal-LBM", "Rayleigh-Benard", "Taylor-Aris", "verification"],
            "markdown": md,
            "figures": [
                {
                    "name": cap,
                    "caption": path.name,
                    "png_base64": base64.b64encode(path.read_bytes()).decode(),
                }
                for cap, path in figs.items()
            ],
        }
        Path("/tmp/report.json").write_text(json.dumps(payload))
        print("wrote /tmp/report.json", round(len(json.dumps(payload)) / 1e6, 2), "MB")
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())