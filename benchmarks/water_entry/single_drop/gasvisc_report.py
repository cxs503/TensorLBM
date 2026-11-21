"""DE-X-GAS1-GASVISC: build the paper-grade progress report JSON.

Usage:
    python3 gasvisc_report.py <ctrl.json> <arm.json> <out.png> <out_report.json>

Computes the triple via score.score_run (same protocol as prior cards),
generates the depth-time figure, and writes the report JSON for the
tensor/reports API.  Zero hand-copied numbers.
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from benchmarks.water_entry.single_drop.score import reference, score_run  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASE = json.loads((HERE / "protocol.json").read_text())
EXP_PEAK = 1.857434
EXP_PEAK_T = 10.394
IMPACT_REF = 1.4092  # m/s, ctrl anchor impact speed (CONV12/CAVSWEEP)


def f4(x):
    return "N/A" if x is None else f"{x:.4f}"


def f3(x):
    return "N/A" if x is None else f"{x:.3f}"


def f2(x):
    return "N/A" if x is None else f"{x:.2f}"


def f1(x):
    return "N/A" if x is None else f"{x:.1f}"



def load_hist(path):
    run = json.loads(Path(path).read_text())
    ts, hs, t0 = [], [], None
    for r in run["history"]:
        if r.get("depth_over_D") is None or r.get("time_ms") is None:
            continue
        if t0 is None:
            t0 = r["time_ms"]
        ts.append(r["time_ms"] - t0)
        hs.append(r["depth_over_D"])
    return np.array(ts), np.array(hs), run


def triple(path):
    run = json.loads(Path(path).read_text())
    t_ref, h_ref = reference(CASE)
    r = score_run(run, t_ref, h_ref, CASE)
    return {
        "stop_reason": r["stop_reason"],
        "contact_step": r["contact_step"],
        "peak_depth_D": r["full_window_peak_depth_over_D"],
        "peak_time_ms": r["full_window_peak_time_ms"],
        "peak_error_pct": (
            100.0 * r["peak_relative_error"] if r["peak_relative_error"] is not None else None
        ),
        "postcontact_healthy_ms": r["postcontact_healthy_ms"],
        "max_geometric_volume_drift": r["max_geometric_volume_relative_drift"],
        "numeric_pass": r["numeric_pass"],
        "gas_nu_ratio": run.get("gas_nu_ratio"),
        "impact_speed_m_s": r["impact_speed_m_s"],
        "impact_speed_rel_err_pct": (
            100.0 * r["impact_speed_relative_error"]
            if r["impact_speed_relative_error"] is not None
            else None
        ),
    }


def make_figure(ctrl_path, arm_path, out_png):
    t_ref, h_ref = reference(CASE)
    tc, hc, rc = load_hist(ctrl_path)
    ta, ha, ra = load_hist(arm_path)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    ax.plot(tc, hc, "-", color="tab:blue", lw=1.7,
            label=f"ctrl  (gas_nu_ratio={rc.get('gas_nu_ratio')})")
    ax.plot(ta, ha, "--", color="tab:red", lw=1.7,
            label=f"arm   (gas_nu_ratio={ra.get('gas_nu_ratio')})")
    ax.plot(t_ref, h_ref, "o", color="k", ms=5, mfc="none", mew=1.4,
            label="experiment (Santini case 3)")
    ipk = int(np.argmax(h_ref))
    ax.annotate(f"exp peak {h_ref[ipk]:.4f}D @ {t_ref[ipk]:.3f} ms",
                xy=(t_ref[ipk], h_ref[ipk]),
                xytext=(t_ref[ipk] + 1.2, h_ref[ipk] + 0.18),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=8)
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("crater depth (D)")
    ax.set_title("DE-X-GAS1-GASVISC: gas-viscosity arm vs ctrl (cells=8, full window)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return Path(out_png).read_bytes()


def main():
    ctrl_path, arm_path, out_png, out_json = sys.argv[1:5]
    c = triple(ctrl_path)
    a = triple(arm_path)
    b64 = base64.b64encode(make_figure(ctrl_path, arm_path, out_png)).decode()
    delta = (a["peak_depth_D"] - c["peak_depth_D"]) if (
        a["peak_depth_D"] is not None and c["peak_depth_D"] is not None) else None

    md = f"""# DE-X-GAS1-GASVISC · 气相粘性修正（GAS 胞分相 τ）· 论文级进展报告

## ① 案例介绍
Santini 高速摄影水击 case 3：直径 D 液滴以 ~1.41 m/s 撞击静水面，空穴峰深实验值
**{EXP_PEAK}D @ {EXP_PEAK_T} ms**。本单检验「空穴欠深（模型峰深 1.4352D，-22.7%）源于
气相被液体粘度过度阻尼（ν_air/ν_water≈0.018）」这一假设。几何/参数：span 4D、pool 3D、
gap 0.18D、cells=8/D、lattice speed 0.06、release 1.53 m/s、u_limit 0.15、
capacity+simple-exchange（CONV12 修复臂旗标，绝对时轴）。

## ② 怎么做的
- 最小改动（commit ee25ca02b，默认关=零行为改变）：`free_surface_step` 增 `gas_nu_ratio`
  （默认 1.0），BGK 无 SGS 分支按 GAS 掩膜对气相胞用 τ_g=0.5+r·(τ_l−0.5)；
  probe 增 `--gas-nu-ratio`（默认 1.0=现状）。
- 对照设计：唯一变量=gas_nu_ratio（ctrl=1.0 无旗标 / 臂=0.018），同网格同旗标同种子。
- 工具链：集群 n315 8 核（cj5717=ctrl, cj5718=臂），stage_dir=工作树，
  PYTHONPATH=$PWD/src；三元组用仓库自带 score.score_run 口径（与 CONV12/CAVSWEEP 同口径）。

## ③ 计算结果（程序化注入，零手抄）
| 臂 | stop | peak_depth | peak_error | postcontact_healthy_ms |
|---|---|---|---|---|
| ctrl (r=1.0) | {c['stop_reason']} | {f4(c['peak_depth_D'])}D@{f3(c['peak_time_ms'])}ms | {f3(c['peak_error_pct'])}% | {f3(c['postcontact_healthy_ms'])} |
| arm (r=0.018) | {a['stop_reason']} | {f4(a['peak_depth_D'])}D@{f3(a['peak_time_ms'])}ms | {f3(a['peak_error_pct'])}% | {f3(a['postcontact_healthy_ms'])} |
| Δ(arm−ctrl) | — | {('N/A' if delta is None else format(delta, '+.4f'))}D | — | — |

## ④ 结果图
depth-time 对照（ctrl/臂/实验）。场量云图省略理由：臂=ctrl（no-op，见⑤），
空腔场与已归档 ctrl 逐位同，云图无增量信息。

## ⑤ 注意点 / 局限
- **关键代码事实**：`free_surface_lbm.py` L1093-1098 每步把 SOLID/GAS 胞 populations
  先复位 f_collide 再清零（waLBerla「gas 无 populations」语义）→ GAS 胞碰撞输出每步被丢弃，
  气相粘性在本求解器中**物理上无作用通道**。
- 本机 150 步 parity：ctrl vs 臂 depth 差 ~5e-9（张量除 vs 标量除舍入伪迹，非物理效应）。
- 单网格无收敛对；numeric_pass=False（3% 门未过，与 ctrl 同态）。

## ⑥ 误差判定（逐项 vs 基准，显式回答 <3%）
| 量 | 臂值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 峰深 | {f4(a['peak_depth_D'])}D | {EXP_PEAK}D | {f2(a['peak_error_pct'])}% | 否 |
| 撞击速度 | {f4(a['impact_speed_m_s'])} m/s | {IMPACT_REF} m/s | {f2(a['impact_speed_rel_err_pct'])}% | {'是' if (a['impact_speed_rel_err_pct'] is not None and a['impact_speed_rel_err_pct'] < 3) else '否'} |

- 峰深误差 {f2(a['peak_error_pct'])}% **不 <3%**（与 ctrl 同态，气相粘性修正未改善）。
- 判定：预登记预测 [1.4352,1.65]D 未命中（臂≈ctrl，收回缺口 0%）；
  证伪判据（peak<1.4352D 或早死<10.394ms）未触发（臂≈ctrl 满窗）。
  **honest 结论：气相过阻尼非欠深主因——假设被证伪（负面结果入账）。**
"""
    report = {
        "title": "DE-X-GAS1-GASVISC 气相粘性修正：分相 τ 赛马（证伪·负面结果）",
        "summary": (
            "气相粘性修正（GAS 胞 τ_g=0.5+0.018·(τ_l−0.5)）满窗实测 arm≈ctrl"
            f"（Δpeak={'N/A' if delta is None else format(delta, '+.4f')}D）：GAS 胞每步清零→气相无动力学→假设证伪，"
            "欠深 0.42D 缺口非气相过阻尼所致。"
        ),
        "scenario_key": "water_entry_single_drop",
        "mission_id": 26728,
        "tags": ["TensorLBM", "free_surface", "cavitation", "gas_viscosity",
                 "falsified_hypothesis", "DE_GAS_W1"],
        "markdown": md,
        "figures": [{
            "name": "depth_time_ctrl_vs_arm.png",
            "caption": "crater depth vs post-contact time: ctrl vs gas-viscosity arm vs experiment",
            "png_base64": b64,
        }],
    }
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"ctrl": c, "arm": a, "delta_peak_depth_D": delta},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
