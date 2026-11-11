#!/usr/bin/env python
"""REPRO-LBM-GUO2002 —— 汇总/外推/画图（论文级产物）。

读取 ``runs/repro_guo2002_dhc/ra*_ny*_ma*.json``（+ 同名 ``.npz`` 末态场），

1. 逐 Ra 对 de Vahl Davis (1983) 表列平均 Nu 求误差（主判据：≤5%）；
2. 对网格阶梯做 **h→0 外推**（Richardson / Roache GCI：三点观测阶 p、
   外推值 Nu_ext、细网格 GCI），判定"细网格直接值"与"外推值"两个口径；
3. 出图：等温线/流线云图（逐 Ra）、Nu–Ra 对表曲线、Nu–h 收敛与外推；
4. 写 ``summary.json``（含逐项判据结论，程序化注入，零手抄）。

用法：
    python scripts/repro_guo2002_analysis.py --rundir runs/repro_guo2002_dhc \
        --figdir figures/repro_guo2002 --out runs/repro_guo2002_dhc/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

TOL_PCT = 5.0          # 主判据（任务给定）
REF = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519}
RA_LABEL = {1e3: "1e3", 1e4: "1e4", 1e5: "1e5"}


def load(rundir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(rundir / "ra*_ny*_ma*.json"))):
        d = json.loads(Path(f).read_text())
        d["_tag"] = Path(f).stem
        d["_npz"] = str(Path(f).with_suffix(".npz"))
        rows.append(d)
    return rows


def richardson(ys: list[float], hs: list[float]) -> dict:
    """Richardson 外推：Nu(h) = Nu0 + C·h^p，三个最细网格。

    返回观测阶 p（Logan 三点式）、外推值 Nu0、细网格 GCI（Roache，F_s=1.25）。
    """
    ys = list(ys)
    hs = list(hs)
    if len(ys) < 3:
        return {"ok": False, "reason": "need >=3 grids"}
    y3, y2, y1 = ys[0], ys[1], ys[2]      # 细、中、粗
    h3, h2, h1 = hs[0], hs[1], hs[2]
    r21 = h2 / h3
    r32 = h1 / h2
    e21 = y2 - y3
    e32 = y1 - y2
    out: dict = {"ok": True, "r21": r21, "r32": r32, "e21": e21, "e32": e32}
    if abs(e21) < 1e-14 or abs(e32) < 1e-14 or e21 / e32 <= 0:
        out.update({"ok": False, "reason": "non-monotone or zero differences"})
        return out
    if abs(r21 - r32) > 1e-6:
        out.update({"ok": False, "reason": "non-uniform grid ratio"})
        return out
    p = math.log(abs(e32 / e21)) / math.log(r21)
    denom = r21 ** p - 1.0
    # Richardson：y0 ≈ y_fine + (y_fine − y_medium)/(r^p − 1)
    ext = y3 + (y3 - y2) / denom
    gci_fine = 1.25 * abs(e21 / y3) / denom * 100.0
    out.update({"p_observed": p, "nu_extrapolated": ext, "gci_fine_pct": gci_fine})
    return out


def analyse(rows: list[dict]) -> dict:
    by_ra: dict[float, list[dict]] = {}
    for r in rows:
        by_ra.setdefault(float(r["ra"]), []).append(r)
    result: dict = {"tolerance_pct": TOL_PCT, "by_ra": {}}
    for ra, rs in sorted(by_ra.items()):
        rs.sort(key=lambda d: d["ny"])          # 粗 → 细
        direct = [{
            "ny": d["ny"], "nu": d["nu"], "nu_energy": d.get("nu_energy"),
            "err_pct": d["err_pct"], "bias_pct": d["bias_pct"],
            "steps": d["steps"], "turnovers": d.get("turnovers"),
            "ma": d["ma"], "tau_f": d["tau_f"], "ma_max_target": d["ma_max_target"],
            "wall_s": d.get("wall_s"), "tag": d["_tag"],
            "nu_std": d.get("nu_std"), "nu_imbalance": d.get("nu_imbalance"),
            "ra_effective_by_gap": d.get("ra_effective_by_gap"),
            "source": d.get("source", "this_work"),
            "dtype": d.get("dtype"),
        } for d in rs]
        # 主口径：ma_max 目标值相同的一组（同一马赫数轴）取最细三个网格外推
        ma_groups: dict[float, list[dict]] = {}
        for d in rs:
            ma_groups.setdefault(float(d["ma_max_target"]), []).append(d)
        extrap: dict[str, dict] = {}
        for ma, g in ma_groups.items():
            g = sorted(g, key=lambda d: d["ny"])
            if len(g) >= 3:
                ex = richardson([d["nu"] for d in g[:3]], [1.0 / d["ny"] for d in g[:3]])
                if ex.get("ok"):
                    ex["err_pct"] = 100.0 * abs(ex["nu_extrapolated"] - REF[ra]) / REF[ra]
                ex["grids"] = [d["ny"] for d in g[:3]]
                extrap[f"ma{ma:g}"] = ex
        finest = rs[-1]
        entry = {
            "ra": ra, "nu_ref": REF[ra],
            "direct": direct,
            "finest_ny": finest["ny"], "finest_nu": finest["nu"],
            "finest_err_pct": finest["err_pct"],
            "finest_pass_5pct": bool(finest["err_pct"] <= TOL_PCT),
            "extrapolation": extrap,
        }
        if extrap:
            key = f"ma{float(finest['ma_max_target']):g}"
            ex = extrap.get(key) or list(extrap.values())[0]
            entry["extrapolated_nu"] = ex.get("nu_extrapolated")
            entry["extrapolated_err_pct"] = ex.get("err_pct")
            entry["extrapolated_pass_5pct"] = bool(
                ex.get("err_pct") is not None and ex["err_pct"] <= TOL_PCT)
        result["by_ra"][RA_LABEL.get(ra, f"{ra:g}")] = entry
    # 总体判定
    verdicts = []
    for k, e in result["by_ra"].items():
        v = "PASS" if (e["finest_pass_5pct"] or e.get("extrapolated_pass_5pct")) else "FAIL"
        verdicts.append({"ra": k, "verdict": v, "finest_err_pct": e["finest_err_pct"],
                         "extrapolated_err_pct": e.get("extrapolated_err_pct")})
    result["verdicts"] = verdicts
    result["overall"] = "PASS" if all(v["verdict"] == "PASS" for v in verdicts) else \
        ("PARTIAL" if any(v["verdict"] == "PASS" for v in verdicts) else "FAIL")
    return result


# ── 图件 ────────────────────────────────────────────────────────────────────


def plot_isoterms(rows: list[dict], outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    fin = {}
    for r in rows:
        if not Path(r["_npz"]).exists():
            continue
        ra = float(r["ra"])
        if ra in fin and fin[ra]["ny"] >= r["ny"]:
            continue
        fin[ra] = r
    if not fin:
        return made
    ras = sorted(fin)
    fig, axes = plt.subplots(1, len(ras), figsize=(4.4 * len(ras), 4.2))
    axes = np.atleast_1d(axes)
    for ax, ra in zip(axes, ras):
        d = np.load(fin[ra]["_npz"])
        T = d["T_field"]
        u, v = d["u_field"], d["v_field"]
        ny, nx = T.shape
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        im = ax.contourf(x, y, T, levels=20, cmap="inferno")
        cs = ax.contour(x, y, T, levels=np.linspace(0.1, 0.9, 9), colors="w", linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=5, fmt="%.1f")
        ax.streamplot(x, y, u, v, color="c", density=1.0, linewidth=0.5, arrowsize=0.5)
        ax.set_title(f"Ra={RA_LABEL.get(ra, f'{ra:g}')}, Ny={fin[ra]['ny']}\n"
                     f"Nu={fin[ra]['nu']:.4f} (ref {REF[ra]:.3f}, "
                     f"err {fin[ra]['err_pct']:.2f}%)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        fig.colorbar(im, ax=ax, shrink=0.8, label="T* (−)")
    fig.suptitle("REPRO-LBM-GUO2002: differentially heated square cavity "
                 "(Pr=0.71) — isotherms + streamlines", fontsize=11)
    fig.tight_layout()
    p = outdir / "fig1_isoterms_dhc.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def plot_nu_ra(res: dict, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ras = [float(k) for k in res["by_ra"]]
    ref = [REF[r] for r in ras]
    ax.plot(ras, ref, "k--o", label="de Vahl Davis (1983) ref", ms=7)
    for ra in ras:
        e = res["by_ra"][RA_LABEL.get(ra, f"{ra:g}")]
        nys = [d["ny"] for d in e["direct"]]
        nus = [d["nu"] for d in e["direct"]]
        ax.plot([ra] * len(nys), nus, "v", color="tab:blue", alpha=0.55, ms=8,
                label="LBM direct (grid ladder)" if ra == ras[0] else None)
        if e.get("extrapolated_nu"):
            ax.plot([ra], [e["extrapolated_nu"]], "D", color="tab:red", ms=9,
                    label="LBM Richardson h→0" if ra == ras[0] else None)
            for d, n in zip(e["direct"], nys):
                if n == e["extrapolation"][list(e["extrapolation"])[0]].get("grids", [None])[0]:
                    pass
    ax.set_xscale("log")
    ax.set_xlabel("Ra")
    ax.set_ylabel("mean Nusselt number on hot wall")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("REPRO-LBM-GUO2002: Nu vs Ra (dhc, Pr=0.71)")
    fig.tight_layout()
    p = outdir / "fig2_nu_vs_ra.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [str(p)]


def plot_convergence(res: dict, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(res["by_ra"]), figsize=(4.6 * len(res["by_ra"]), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (k, e) in zip(axes, res["by_ra"].items()):
        ra = float(k)
        hs = [1.0 / d["ny"] for d in e["direct"]]
        nus = [d["nu"] for d in e["direct"]]
        ax.plot(hs, nus, "o-", label="LBM")
        ax.axhline(REF[ra], color="k", ls="--", lw=1, label="DVD 1983")
        if e.get("extrapolated_nu"):
            ax.axhline(e["extrapolated_nu"], color="tab:red", ls=":", lw=1.2,
                       label=f"h→0 ext = {e['extrapolated_nu']:.4f}")
            ax.errorbar([0.0], [e["extrapolated_nu"]], fmt="D", color="tab:red")
        ax.set_xlabel("h = 1/Ny")
        ax.set_ylabel("Nu")
        ax.set_title(f"Ra={k}: finest err {e['finest_err_pct']:.2f}%", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("REPRO-LBM-GUO2002: grid convergence of mean Nu", fontsize=11)
    fig.tight_layout()
    p = outdir / "fig3_grid_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [str(p)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/repro_guo2002_dhc")
    ap.add_argument("--figdir", default="figures/repro_guo2002")
    ap.add_argument("--out", default="runs/repro_guo2002_dhc/summary.json")
    args = ap.parse_args(argv)
    rundir = Path(args.rundir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    rows = load(rundir)
    if not rows:
        raise SystemExit(f"no result json under {rundir}")
    res = analyse(rows)
    figs: list[str] = []
    try:
        figs += plot_isoterms(rows, figdir)
        figs += plot_nu_ra(res, figdir)
        figs += plot_convergence(res, figdir)
    except Exception as exc:  # 图件失败不掩盖数值结论
        res["plot_error"] = repr(exc)
    res["figures"] = figs
    res["n_runs"] = len(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("overall", "verdicts", "tolerance_pct")},
                     indent=1, ensure_ascii=False))
    print("figures:", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())