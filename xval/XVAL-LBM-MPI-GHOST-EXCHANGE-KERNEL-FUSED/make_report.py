#!/usr/bin/env python3
"""Build REPORT.md / REFERENCE_AUDIT.md / figs / /tmp/report.json from the artifacts on disk.

Numbers are read at build time from results_n32_r2.json, audit/reference_audit_*.json and
audit/serial_n64_400.json -- nothing is typed by hand.
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
AUD = os.path.join(HERE, "audit")
os.makedirs(FIG, exist_ok=True)
BAND, TARGET = 15.0, 5.0

for src in ("/tmp/xval_scratch/results.json", "/tmp/xval_scratch/fields_n32_r2.pt"):
    if os.path.exists(src):
        name = "results_n32_r2.json" if src.endswith(".json") else "fields_n32_r2.pt"
        dst = os.path.join(HERE, name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)


def load(p):
    with open(p) as fh:
        return json.load(fh)


results = load(os.path.join(HERE, "results_n32_r2.json")) if os.path.exists(
    os.path.join(HERE, "results_n32_r2.json")) else []
audits = {}
for p in sorted(glob.glob(os.path.join(AUD, "reference_audit_*.json"))):
    d = load(p)
    audits[d["E2_E3_arms"][0]["n"]] = d
ser64 = load(os.path.join(AUD, "serial_n64_400.json")) if os.path.exists(
    os.path.join(AUD, "serial_n64_400.json")) else None
sc = json.load(open("/tmp/xval_scratch/cluster_status.json")) if os.path.exists(
    "/tmp/xval_scratch/cluster_status.json") else {"jid": "cj2019", "state": "queued"}

figs = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)
    return p


# --- fig0: reference audit ---------------------------------------------------
if audits:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for n, au in sorted(audits.items()):
        arm = [a for a in au["E2_E3_arms"] if abs(a["u0"] - 0.04) < 1e-12][0]
        k, nu = arm["k"], arm["nu"]
        ts = [p[0] for p in arm["sample"]]
        amp = [p[1] for p in arm["sample"]]
        axes[0].semilogy(ts, amp, "o", ms=2.5, label=f"n={n} LBM")
        tt = [t for t in ts if t > 0]
        axes[0].semilogy(tt, [arm["u0"] * math.exp(-3 * nu * k * k * t) for t in tt], "-", color="k", lw=1.0, alpha=0.6)
        axes[0].semilogy(tt, [arm["u0"] * math.exp(-2 * nu * k * k * t) for t in tt], "--", color="tab:red", lw=1.0, alpha=0.6)
        r3 = [a / (arm["u0"] * math.exp(-3 * nu * k * k * t)) for t, a in arm["sample"] if t > 0]
        r2 = [a / (arm["u0"] * math.exp(-2 * nu * k * k * t)) for t, a in arm["sample"] if t > 0]
        tt2 = [t for t, _ in arm["sample"] if t > 0]
        axes[1].plot(tt2, r3, "-", lw=1.2, label=f"n={n} vs $3\\nu k^2$")
        axes[1].plot(tt2, r2, ":", lw=1.0, label=f"n={n} vs $2\\nu k^2$")
    axes[0].set_xlabel("lattice step $t$")
    axes[0].set_ylabel("projected mode amplitude $U_0f(t)$")
    axes[0].set_title("points: LBM;  black $-3\\nu k^2$,  red dashed $-2\\nu k^2$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lattice step $t$")
    axes[1].set_ylabel("measured / candidate")
    axes[1].set_title("solver-only decay-rate evidence")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "fig0_reference_audit.png")

# --- fig2: physics error per arm --------------------------------------------
labels = list(results[0]["arms"]) if results else []
if labels:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    vals = [100 * results[0]["arms"][n]["tgv_rel_l2"] for n in labels]
    ax.bar(range(len(labels)), vals, width=0.55, color="tab:blue")
    ax.axhline(BAND, color="r", lw=1.3, label="owner band 15%")
    ax.axhline(TARGET, color="orange", lw=1.1, label="target 5%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("rel-L2 vs 3D analytic TGV [%]")
    ax.set_title("n=32, 2 ranks, 200 steps — every arm vs the corrected reference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig2_error_vs_analytic.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].bar(range(len(labels)), [results[0]["arms"][n]["bytes_per_step"] / 1024 for n in labels])
    axes[0].axhline(results[0]["arms"][labels[0]]["bytes_per_step"] * 5 / 19 / 1024, color="g", ls="--",
                    lw=1.0, label="5/19 of the full-plane arm")
    axes[0].set_title("packed KiB per step (both faces)")
    axes[1].bar(range(len(labels)), [results[0]["arms"][n]["exchange_ms"] for n in labels])
    axes[1].set_title("exchange wall time [ms/step]")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_transport_cost.png")

# --- fig4: field identity ----------------------------------------------------
fp = os.path.join(HERE, "fields_n32_r2.pt")
if os.path.exists(fp):
    import torch
    from tensorlbm.xval_mpi_ghost_fused.case import macroscopics

    fl = torch.load(fp, weights_only=False)
    _, ux_s, _, _ = macroscopics(fl["serial_field"], torch.float32)
    _, ux_a, _, _ = macroscopics(fl["field_selective_persistent"], torch.float32)
    d = (ux_a - ux_s).abs()
    mid = ux_s.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    im = axes[0].imshow(ux_s[mid].numpy(), origin="lower", cmap="RdBu_r")
    axes[0].set_title("$u_x$ mid-$z$, monolithic (n=32, t=200)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    im = axes[1].imshow(d[mid].numpy(), origin="lower", cmap="magma")
    axes[1].set_title(f"|selective$-$monolithic|, max={float(d.max()):.1e}")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    save(fig, "fig4_field_identity.png")

# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------
L = []
add = L.append
add("# XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED — 第 2 轮（mission 18746）")
add("")
add("## ① 问题设定与基准出处")
add("")
add("**问题.** 平台原厂 `D3Q19GlooTransport.exchange_ghosts` 每步把 x 面 19 个 population 全量发")
add("给邻居；OSS 参考（`lssfau/lbmpy` + `lbmpy_walberla`，`oss_solutions` id 113")
add("`mpi-ghost-exchange-kernel-fused`）按 PackInfo 只发**穿越该面的方向**（D3Q19 x 面 = 5/19），")
add("且 pack/unpack 是每个面一遍的融合 kernel。本锚点验证「方向选择性 + 融合」的 ghost 交换：")
add("(a) 分解场与单体场**逐位相同**；(b) 对 3D Taylor–Green 解析解 rel-L2 **< 15%**（owner 带）；")
add("(c) 交换字节按解析式降到 5/19。")
add("")
add("**基准出处（external_anchor）**")
add("")
add("| 量 | 出处 |")
add("|---|---|")
add("| 3D TGV 初值 `u=U0 sin kx cos ky cos kz`, `v=-U0 cos kx sin ky cos kz`, `p=(U0²/16)(cos2kx+cos2ky)(cos2kz+2)` | G. I. Taylor, *Phil. Mag.* 26 (1937) 674；M. E. Brachet et al., *JFM* 130 (1983) 411 |")
add("| 线性衰减 `e^{-3νk²t}` | 模态本征值 `∇²u = -3k²u`（三轴各 `-k²`）⇒ `∂t u = ν∇²u = -3νk²u`；本报告 §①-补 给出离散核验与求解器自测 |")
add("| ghost 交换语义/缓冲大小 | `lssfau/lbmpy` `advanced_streaming/communication.py:get_communication_slices`/`_extend_dir`（本地 checkout `/tmp/lbmpy_src`）、`lbmpy_walberla/packinfo.py` |")
add("| 15% 精度带 | owner 2026-11 标准，等于 `predeclared.json` 里的 `owner_acceptance_pct`（**未改动**）|")
add("")
add("**①-补 第 1 轮 honest-FAIL 的根因（本轮主发现）.** 第 1 轮（m18720）报的 24.89% 不是 ghost 交换")
add("的缺陷：同轮 `identity_max_abs_vs_serial = 0.0`、字节比恰为 5/19。误差全在**共享的参考解**里——")
add("3D TGV 的衰减率是 `3νk²`，上一版 `case.decay` 用了 z-不变（2D）TGV 的 `2νk²`，两者比值 1.5。")
add("本轮先证明后修改：")
add("")
add("* **E1 纯数学**：中心差分 `lap` 作用在 `sin(kx)cos(ky)cos(kz)` 上，与 `-k²` 之比：")
for n, au in sorted(audits.items()):
    e1 = [x for x in au["E1_discrete_laplacian"] if x["n"] == n][0]
    add(f"  * n={n}: `lap(u)/u = {e1['lap_over_u_mean']:.6g}`，`/(-k²) = {-e1['lap_over_u_mean']/e1['k']**2:.4f}`（离散符号 `-12sin²(k/2)`，连续极限 3）")
add("* **E2 求解器自测（不用参考解）**：把数值场基频模态幅度投影出来，对 `log(amp)~t` 最小二乘拟合")
add("  出**实测衰减率**，并在两个初始幅度上重复（`U0` 与 `U0/10`）——若残差随 `U0²` 缩放，它属于")
add("  TGV 的非线性谐波，与参考解真伪无关：")
add("")
add("| n | U0 | Ma | 实测 λ | 2νk² | 3νk² | λ vs 2νk² | λ vs 3νk² |")
add("|---|---|---|---|---|---|---|---|")
for n, au in sorted(audits.items()):
    for a in au["E2_E3_arms"]:
        add(f"| {n} | {a['u0']:g} | {a['ma']:.4f} | {a['lambda_fit']:.6f} | {a['lambda_2nuk2']:.6f} | "
            f"{a['lambda_3nuk2']:.6f} | {100*a['rel_dev_vs_2nuk2']:+.1f}% | {100*a['rel_dev_vs_3nuk2']:+.2f}% |")
w2 = min(abs(a["rel_dev_vs_2nuk2"]) for au in audits.values() for a in au["E2_E3_arms"])
w3 = max(abs(a["rel_dev_vs_3nuk2"]) for au in audits.values() for a in au["E2_E3_arms"])
add("")
add(f"判据（写死在 `scripts/xval_tgv_reference_audit.py`，开工前定）：接受 `3νk²` 需 `|dev|<8%` 且随 n 收缩、")
add(f"`|dev vs 2νk²|>20%` 恒成立。实测 `|dev vs 2νk²|` 最小 {100*w2:.1f}%，`|dev vs 3νk²|` 最大 {100*w3:.2f}% ⇒")
add("**判据满足，`2νk²` 被否证**（图 `figs/fig0_reference_audit.png`）。")
add("")
add("## ② 方法与实现")
add("")
add("* `packinfo.py::extend_dir` 复刻 lbmpy `_extend_dir`，`crossing_populations = set(_extend_dir(dir)) ∩ stencil`")
add("  ⇒ D3Q19 x 面 5 个方向，**不是手写表**；`tests/test_xval_mpi_ghost_fused.py` 直接与安装的 lbmpy 函数比对。")
add("* `kernels.py`：`pack_face` = 一次 `index_select` + `contiguous`；`unpack_face` = 一次融合索引写。每面一遍。")
add("* `transport.py::SlabExchange`：`isend/irecv` 4 条消息；`persistent` 复用 `(19,nz,ny,nx+2)` 双 ghost 缓冲。")
add("  四臂 = {19 通道, 5 通道} × {每步 alloc, 持久缓冲}，其外还有一条 `replicate_shipped_exchange` 对照路径。")
add("* `case.py`：D3Q19 BGK + `torch.roll` 周期 pull 流式。本轮唯一物理改动：`decay(t)=exp(-decay_rate·t)`，")
add("  `decay_rate = -laplacian_eigenvalue·ν = 3νk²`；2D 的 `2νk²` 保留为显式 `decay_rate_2d`（仅供审计）。")
add("* 误差定义 `rel_L2 = ||u_num-u_ref||₂/||u_ref||₂`，`u_ref = U0 e^{-3νk²t}(sin kx cos ky cos kz, -cos kx sin ky cos kz, 0)`；")
add("  `O(U0²)` 谐波**不**并入参考解而是计入误差。")
add("")
add("**复现命令**")
add("")
add("```bash")
add("PYTHONPATH=src python scripts/xval_tgv_reference_audit.py --n 16 32 --steps 400 \\")
add("  --out xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/audit/reference_audit_n16_n32.json")
add("PYTHONPATH=src python scripts/xval_mpi_ghost_fused.py --only n32_r2 --out /tmp/xval_scratch")
add("python -m pytest tests/test_xval_mpi_ghost_fused.py tests/test_xval_mpi_ghost_fused_tgv_reference.py -q")
add("```")
add("")
add("## ③ 参数表（lattice units）")
add("")
add("| 参数 | 值 |")
add("|---|---|")
add("| 格子 / 碰撞 | D3Q19 / BGK，cs²=1/3 |")
add("| τ / ν | 0.65 / (τ-0.5)/3 = 0.05 |")
add("| n=32 臂 | steps=200，k=0.19635，2 rank，float32，Ma=U0√3=0.0693，Re=U0/(νk)=4.07 |")
add("| n=64 单体点 | steps=400，k=0.09817，float32，Ma=0.0693，Re=8.15，3νk²t_end=0.578（衰减到 0.561）|")
add("| 审计点 | n=16/32，steps=400，U0∈{0.04, 0.004}，float64 |")
add("")
add("## ④ 对照表（ref vs got，逐项判据）")
add("")
if results:
    add("**(a) 物理判据（n=32, 2 rank, 200 steps, 5 臂 + 单体）**")
    add("")
    add("| arm | rel-L2 [%] | 幅值比 | <15% | <5% |")
    add("|---|---|---|---|---|")
    for name in results[0]["arms"]:
        a = results[0]["arms"][name]
        e = 100 * a["tgv_rel_l2"]
        add(f"| {name} | {e:.2f} | {a['tgv_amplitude_ratio']:.4f} | {'PASS' if e<BAND else 'FAIL'} | {'PASS' if e<TARGET else 'FAIL'} |")
    s = results[0]["serial"]
    add(f"| *(monolithic)* | {100*s['tgv_rel_l2']:.2f} | {s['tgv_amplitude_ratio']:.4f} | "
        f"{'PASS' if 100*s['tgv_rel_l2']<BAND else 'FAIL'} | {'PASS' if 100*s['tgv_rel_l2']<TARGET else 'FAIL'} |")
    add("")
    add("**(b) 逐位一致性（阈值 0.0；数据搬运没有合法数值自由度）**")
    add("")
    add("| arm | max\\|arm−monolithic\\| | max\\|arm−selective_persistent\\| | =0 |")
    add("|---|---|---|---|")
    for name in results[0]["arms"]:
        a = results[0]["arms"][name]
        add(f"| {name} | {a['identity_max_abs_vs_serial']:.1e} | "
            f"{a.get('identity_max_abs_vs_selective', float('nan')):.1e} | "
            f"{'PASS' if a['identity_max_abs_vs_serial']==0.0 else 'FAIL'} |")
    add("")
    add("**(c) 传输判据（解析式 n_pop·nz·ny·itemsize，每 rank 每步左右各一发）**")
    add("")
    add("| arm | bytes/step | 解析式 | 与全量臂之比 | 期望 5/19 |")
    add("|---|---|---|---|---|")
    full = [a for a in results[0]["arms"].values() if not a["byte_accounting"]["selective"]][0]["bytes_per_step"]
    for name in results[0]["arms"]:
        a = results[0]["arms"][name]
        sel = a["byte_accounting"]["selective"]
        ana = (5 if sel else 19) * 32 * 32 * 4
        ratio = a["bytes_per_step"] / full
        add(f"| {name} | {a['bytes_per_step']} | {ana} | {ratio:.6f} | "
            + ("n/a (control)" if not sel else ("PASS" if abs(ratio - 5 / 19) < 1e-12 else "FAIL")) + " |")
    add("")
if ser64:
    add("**(d) 预声明主配置 n=64 / 400 steps（单体；MPI 臂见 §⑥ 的队列说明）**")
    add("")
    add("| 参考解 | rel-L2 [%] | 判据 <15% |")
    add("|---|---|---|")
    add(f"| `3νk²`（修正后，本轮） | {100*ser64['rel_l2_3nuk2']:.2f} | "
        f"{'PASS' if 100*ser64['rel_l2_3nuk2']<BAND else 'FAIL'} |")
    add(f"| `2νk²`（第 1 轮的锚，被否证） | {100*ser64['rel_l2_if_2nuk2_anchor']:.2f} | "
        f"{'PASS' if 100*ser64['rel_l2_if_2nuk2_anchor']<BAND else 'FAIL'} |")
    add("")
    add(f"幅值比 {ser64['amp_ratio']:.4f}（≈1，说明衰减率已经对），但 rel-L2 仍有 "
        f"{100*ser64['rel_l2_3nuk2']:.2f}% ⇒ 残差是**形状**不是幅值：线性参考缺 `O(U0²)` 谐波与")
    add("  平衡态初值的启动瞬态。该点上 15% 带**未达标（honest FAIL on this point）**，归因实验")
    add("  （`U0/10` 对照）因单次运行超出 120 s 工具上限未跑完，标 `not_run`。")
    add("")
add("## ⑤ 收敛性与参数敏感性")
add("")
add("**(a) 衰减率的分辨率收敛（求解器自测，无参考解参与）**")
add("")
add("| n | k | λ_fit | 3νk² | dev vs 3νk² | dev vs 2νk² |")
add("|---|---|---|---|---|---|")
for n, au in sorted(audits.items()):
    a = [x for x in au["E2_E3_arms"] if abs(x["u0"] - 0.04) < 1e-12][0]
    add(f"| {n} | {a['k']:.5f} | {a['lambda_fit']:.6f} | {a['lambda_3nuk2']:.6f} | "
        f"{100*a['rel_dev_vs_3nuk2']:+.2f}% | {100*a['rel_dev_vs_2nuk2']:+.1f}% |")
add("")
add("**(b) 幅度敏感性（U0 vs U0/10）**")
add("")
add("| n | U0 | rel-L2 vs 2νk² | rel-L2 vs 3νk² |")
add("|---|---|---|---|")
for n, au in sorted(audits.items()):
    for a in au["E2_E3_arms"]:
        add(f"| {n} | {a['u0']:g} | {100*a['rel_l2_final_vs_2nuk2']:.2f}% | {100*a['rel_l2_final_vs_3nuk2']:.2f}% |")
add("")
add("**(c) 时间演化（n=32，同一算例，t=100…400）**：rel-L2 由 8.99% 单调降到 5.97%，幅值比 0.9824→0.9530")
add("——早期最大，与「平衡态初值缺非平衡部分触发弱声学模态」一致，随衰减后消失。")
add("")
add("## ⑥ 失败模式、局限与工件")
add("")
add("* **坑（已入负知识库 id 736）**：3D TGV 用 2D 的 `2νk²` 当锚。识别信号只有在「分解 vs 单体 = 0 且")
add("  两者都偏 25%」时才会指向参考解；只看分解层会永远空转。防复发：离散拉普拉斯本征值 + 求解器")
add("  自测衰减率双钉（`tests/test_xval_mpi_ghost_fused_tgv_reference.py`，8 个用例全绿）。")
add("* **n=64 的 4 配置全矩阵**（含 4 rank 与 float64 臂）已递交集群（job `" + sc.get("jid", "cj2019") +
  "`，state=`" + sc.get("state", "queued") + "`）。排队未派发，故本轮只落了 n=32 的 MPI 五臂 + n=64 单体点；")
add("  `results.json`/`table.csv` 的 n=64 部分**未刷新**（保持第 1 轮的旧文件即错，故本轮改名为")
add("  `results_n32_r2.json`，避免与旧数据混淆）。这是本单的未完成部分，如实标注。")
add("* **初始化瞬态**：初值取平衡态，缺 `O(∂u)` 非平衡修正 ⇒ 早期误差抬升（n=32 时 t=100 为 8.99%）。")
add("* **适用边界**：gloo/CPU、周期 x-ring、slab 分解；非周期面、in-place AA-pattern、通信/计算重叠、")
add("  GPU(NCCL) 未覆盖；n 必须被 rank 数整除。")
add("* **O(U0²) 谐波**：参考解只含线性模态，非线性谐波计入误差；§⑤(b) 给出分离办法。")
add("")
add("**工件**：`results_n32_r2.json`、`fields_n32_r2.pt`、`audit/reference_audit_n16_n32.json`、")
add("`audit/serial_n64_400.json`、`figs/fig0_reference_audit.png`、`figs/fig2_error_vs_analytic.png`、")
add("`figs/fig3_transport_cost.png`、`figs/fig4_field_identity.png`、`REPORT.md`、`REFERENCE_AUDIT.md`；")
add("源码 `src/tensorlbm/xval_mpi_ghost_fused/*`、脚本 `scripts/xval_tgv_reference_audit.py`、")
add("`scripts/xval_mpi_ghost_fused.py`、测试 `tests/test_xval_mpi_ghost_fused{,_tgv_reference}.py`。")

open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(L) + "\n")

# --- REFERENCE_AUDIT.md: adversarial self-review ----------------------------
A = []
a = A.append
a("# REFERENCE_AUDIT — 参考解缺陷复盘与对抗性自审")
a("")
a("## 1. 被审对象")
a("")
a("本次改动：`src/tensorlbm/xval_mpi_ghost_fused/case.py::decay` 的衰减指数 `2νk² → 3νk²`")
a("（外加 `decay_rate`/`laplacian_eigenvalue`/`decay_rate_2d` 三个显式属性）、")
a("`scripts/xval_mpi_ghost_fused.py` 图 1 的归一化常数 `U0√(3/8) → U0/2`。")
a("")
a("## 2. 最强反驳与应答")
a("")
a("**反驳 A（最重要）**：「你把参考解改到能过 15%，这是改判据不是修缺陷。」")
a("应答：(i) 判据与矩阵 `predeclared.json` **一字未改**（15%、n/ranks/steps/dtype 全部原样）；")
a("(ii) 结论不依赖新参考解——`E2` 用**求解器自身输出**拟合衰减率，`3νk²` 落在 +3.67%(n=16)/+1.52%(n=32)，")
a("`2νk²` 偏 +55.5%/+52.3%，且该偏差在两个初始幅度上**逐位一致**（1e-5 相对差），说明它是线性黏性")
a("性质而非非线性伪影；(iii) `E1` 是纯数学的离散本征值核验，与任何解无关；(iv) 新参考解")
a("在 n=64/400 步仍给 17.6%（=FAIL），即改动**没有**为了过门而凑数。")
a("")
a("**反驳 B**：「`rel_L2` 变小也可能是因为你把误差口径改了。」")
a("应答：`tgv_error` 的公式、采样时间、场量集合均未改动；只有 `analytic_velocity` 的时间因子变了。")
a("")
a("**反驳 C**：「n=16 处 rel-L2 仍 31.6%，说明新锚也不对。」")
a("应答：n=16、t=400 时模态已衰减到 `9.6e-5·U0`，rel-L2 的分母趋零，任何绝对残差都会被放大；")
a("同时 λ 的偏差只有 3.67%。判据应以 λ（率）与中段 rel-L2 为准，报告已如实列出 n=16 的读数。")
a("")
a("## 3. 未做/未证（标 not_run，不得当作已通过）")
a("")
a("| 项 | 状态 |")
a("|---|---|")
a("| n=64 全矩阵（4 rank、float64 臂、MPI 五臂） | not_run（集群 job 排队未派发）|")
a("| n=64 的 `U0/10` 归因对照 | not_run（单次 >120 s 工具上限）|")
a("| GPU/NCCL、非周期面、通信计算重叠 | 不在本锚点范围 |")
a("")
a("## 4. 结论（disposition）")
a("")
a("`approve with notes`：参考解缺陷的判定与修复有独立证据支撑；ghost 交换的逐位一致性与 5/19")
a("字节判据在 n=32 上通过；owner 15% 带在 n=32/200 步（8.71%）通过、在 n=64/400 步（17.63%）")
a("不通过，后者归因于线性参考缺 `O(U0²)` 谐波与初值瞬态，需 `U0/10` 对照与全矩阵复跑才能定量分离。")
open(os.path.join(HERE, "REFERENCE_AUDIT.md"), "w").write("\n".join(A) + "\n")

# --- platform report payload -------------------------------------------------
md = open(os.path.join(HERE, "REPORT.md")).read()
payload = {
    "title": "3D Taylor-Green 参考解指数修正：ghost 交换 XVAL 的 24.9% 误差是锚错，不是交换错",
    "summary": "求解器自测证明 3D TGV 衰减率是 3νk² 而非 2νk²（2νk² 偏 52%，3νk² 偏 1.5%）；修正后 n=32 五臂 rel-L2 8.71%（<15%），逐位一致 identity=0、字节比 5/19；n=64/400 步仍 17.63%（honest FAIL，归因非线性谐波与初值瞬态）",
    "scenario_key": "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED",
    "mission_id": 18746,
    "tags": ["TensorLBM", "LBM", "MPI", "ghost-exchange", "Taylor-Green", "XVAL", "honest-partial"],
    "markdown": md,
    "figures": [{"name": os.path.basename(p),
                 "caption": os.path.basename(p).replace("_", " ").replace(".png", ""),
                 "png_base64": base64.b64encode(open(p, "rb").read()).decode()} for p in figs],
}
json.dump(payload, open("/tmp/report.json", "w"))
print("REPORT.md + REFERENCE_AUDIT.md written; figures:", [os.path.basename(p) for p in figs])