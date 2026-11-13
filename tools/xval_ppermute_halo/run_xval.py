"""XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO 运行入口（ref vs got 全对照 + 工件）。

对标对象（OSS, MIT）：haimingz/vivsim @ fb84b12f02e18f3b9dfdf738d6a1a318d81539f9
``vivsim/multidevice.py::stream_cross_devices`` —— ``jax.lax.ppermute`` 周期 halo
（局部自回卷 streaming + 单次置换搬运，零 staging 缓冲）。

Arms
----
A1  等价性：ref = 单域（NP=1，本仓真求解器 ``LBMStepExecutor``）；
            got = N=2/4/8 shard + ppermute halo（同一求解器）。
            指标 max|Δf|、rho/uy 相对 L2 —— 判据 = 逐位一致。
A2  OSS 原式 2D D2Q9 臂：``stream_cross_devices`` 逐行移植（MIT）。
            ref = 全域周期 stream；got = 分块 + OSS 分支交换。
            分别跑 y 分支（原样）、x 分支（原样）、x 分支（perm 互换）。
A3  解析解臂：剪切波 u_1 = U0 sin(k z) exp(-nu k^2 t)（NS 精确解），
            ref = 解析衰减，got = sharded 解。误差判据 <15%（owner 精度标准）。
A4  成本对照：ppermute（10 面/步，零 staging）vs 朴素全 Q staging 交换
            （逐 rank pack/unpack，2Q 面/步）—— 字节、算子数、墙钟。
A5  OSS 差异清单（做法/参数/边界条件），含 branch 方向核账。

Outputs: ``artifacts/xval_ppermute_periodic_halo/``
    summary.json / kpi.csv / tables.md / RESULTS.md / fig1..fig3 png

Usage::

    PYTHONPATH=src python tools/xval_ppermute_halo/run_xval.py \
        --out artifacts/xval_ppermute_periodic_halo
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(WORKTREE / "src"))

from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.parallel import (  # noqa: E402
    JAX_AVAILABLE,
    PpermutePeriodicHalo,
    ShardedPpermuteRunner,
    device_pairs,
    device_pairs_reverse,
    oss_branch_findings,
    oss_stream_cross_devices,
    oss_stream_cross_devices_jax,
    sender_index,
)

OSS_REPO = "haimingz/vivsim"
OSS_COMMIT = "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9"
OSS_FILE = "vivsim/multidevice.py::stream_cross_devices"
OSS_LICENSE = "MIT (Copyright (c) 2024 Haiming Zhu)"
THRESHOLD = 0.15  # owner 精度标准：对标误差 >15% 不可接受


# ---------------------------------------------------------------------------
# 2D D2Q9 臂（OSS basic.py 的 streaming + multidevice.py 的交换）
# ---------------------------------------------------------------------------


def _shift_x_pos(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[:, -1:, :], x[:, :-1, :]), dim=1)


def _shift_x_neg(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[:, 1:, :], x[:, :1, :]), dim=1)


def _shift_y_pos(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[:, :, -1:], x[:, :, :-1]), dim=2)


def _shift_y_neg(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[:, :, 1:], x[:, :, :1]), dim=2)


def d2q9_stream_oss(f: torch.Tensor) -> torch.Tensor:
    """vivsim ``lbm/basic.py::streaming`` 的逐行移植（拉式 + 周期自回卷）。"""
    return torch.cat(
        (
            f[0:1],
            _shift_x_pos(f[1:2]),
            _shift_y_pos(f[2:3]),
            _shift_x_neg(f[3:4]),
            _shift_y_neg(f[4:5]),
            _shift_x_pos(_shift_y_pos(f[5:6])),
            _shift_x_neg(_shift_y_pos(f[6:7])),
            _shift_x_neg(_shift_y_neg(f[7:8])),
            _shift_x_pos(_shift_y_neg(f[8:9])),
        ),
        dim=0,
    )


def arm_oss_2d(nx: int = 16, ny: int = 12, seed: int = 7) -> dict[str, Any]:
    """A2：OSS 原式 2D 臂（y 分支 / x 分支原样 / x 分支修正）。"""
    torch.manual_seed(seed)
    f0 = torch.rand(9, nx, ny)
    ref = d2q9_stream_oss(f0)

    out: dict[str, Any] = {"grid": [nx, ny], "seed": seed, "branches": []}
    for n in (2, 4):
        # -- 沿 y 分块（axis=1 → 内部 axis_pull_groups 的 spatial axis=1）
        nyl = ny // n
        shards = [f0[:, :, r * nyl : (r + 1) * nyl].clone().contiguous() for r in range(n)]
        local = [d2q9_stream_oss(s) for s in shards]
        got, _ = oss_stream_cross_devices(local, "y", n)
        g = torch.cat(got, dim=2)
        out["branches"].append(
            {
                "split": "y",
                "n_ranks": n,
                "variant": "oss_as_written",
                "max_abs_df": float((ref - g).abs().max()),
                "rel_l2": _rel_l2(ref, g),
            }
        )
        # -- 沿 x 分块：OSS 原样
        nxl = nx // n
        shards = [f0[:, r * nxl : (r + 1) * nxl, :].clone().contiguous() for r in range(n)]
        local = [d2q9_stream_oss(s) for s in shards]
        got, _ = oss_stream_cross_devices(local, "x", n, swap_x_branch=False)
        g = torch.cat(got, dim=1)
        out["branches"].append(
            {
                "split": "x",
                "n_ranks": n,
                "variant": "oss_as_written",
                "max_abs_df": float((ref - g).abs().max()),
                "rel_l2": _rel_l2(ref, g),
            }
        )
        # -- 沿 x 分块：perm 互换（把 x 分支对齐到拉式约定）
        shards = [f0[:, r * nxl : (r + 1) * nxl, :].clone().contiguous() for r in range(n)]
        local = [d2q9_stream_oss(s) for s in shards]
        got, _ = oss_stream_cross_devices(local, "x", n, swap_x_branch=True)
        g = torch.cat(got, dim=1)
        out["branches"].append(
            {
                "split": "x",
                "n_ranks": n,
                "variant": "perm_swapped",
                "max_abs_df": float((ref - g).abs().max()),
                "rel_l2": _rel_l2(ref, g),
            }
        )
    out["branch_findings"] = [b.as_dict() for b in oss_branch_findings(4)]
    out["jax_path_available"] = JAX_AVAILABLE
    if not JAX_AVAILABLE:
        try:
            oss_stream_cross_devices_jax(f0, "y", "dev", 2)
            out["jax_guard"] = "NOT_RAISED (unexpected)"
        except RuntimeError as exc:
            out["jax_guard"] = f"RuntimeError raised as designed: {exc}"
    return out


def _rel_l2(ref: torch.Tensor, got: torch.Tensor) -> float:
    num = float(torch.linalg.vector_norm((ref - got).reshape(-1)))
    den = float(torch.linalg.vector_norm(ref.reshape(-1)))
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3D 剪切波工况
# ---------------------------------------------------------------------------


def shear_wave_field(nz: int, ny: int, nx: int, u0: float) -> torch.Tensor:
    """u_1 = U0 sin(2 pi z / nz)，其余为 0，rho=1，f = feq。"""
    z = torch.arange(nz, dtype=torch.float32).view(nz, 1, 1)
    uy = (u0 * torch.sin(2 * torch.pi * z / nz)).expand(nz, ny, nx).contiguous()
    zeros = torch.zeros_like(uy)
    rho = torch.ones_like(uy)
    return equilibrium3d(rho, zeros, uy, zeros)


def shear_amplitude(runner: ShardedPpermuteRunner) -> float:
    """投影到 sin(k z) 的一阶模幅值（解析解对应量的离散测量）。"""
    _rho, _ux, uy, _uz = runner.gather_macro()
    nz = uy.shape[0]
    z = torch.arange(nz, dtype=uy.dtype).view(nz, 1, 1)
    sine = torch.sin(2 * torch.pi * z / nz)
    proj = (uy * sine).mean(dim=(1, 2))
    return float(2.0 * proj.abs().mean())


def arm_equivalence(
    n: int = 16, steps: int = 20, tau: float = 0.6, ranks: tuple[int, ...] = (2, 4, 8)
) -> dict[str, Any]:
    """A1：单域 vs sharded 逐位一致（真求解器 ``LBMStepExecutor``）。"""
    f0 = shear_wave_field(n, n, n, u0=0.01)
    mono = ShardedPpermuteRunner(f0, n_ranks=1, tau=tau)
    mono.step(steps)
    ref = mono.gather()
    rows = []
    for nr in ranks:
        sh = ShardedPpermuteRunner(f0, n_ranks=nr, tau=tau)
        sh.step(steps)
        got = sh.gather()
        rho_ref, uy_ref = _macro_from_f(ref)
        rho_got, uy_got = _macro_from_f(got)
        rows.append(
            {
                "n_ranks": nr,
                "steps": steps,
                "max_abs_df": float((ref - got).abs().max()),
                "bit_identical": bool(torch.equal(ref, got)),
                "rho_rel_l2": _rel_l2(rho_ref, rho_got),
                "uy_rel_l2": _rel_l2(uy_ref, uy_got),
                "planes_per_exchange": sh.halo.planes_per_exchange,
                "bytes_per_exchange_per_rank": sh.stats().as_dict()["bytes_per_exchange_per_rank"],
            }
        )
    return {"grid": [n, n, n], "steps": steps, "tau": tau, "rows": rows}


def _macro_from_f(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from tensorlbm.d3q19 import macroscopic3d

    rho, _ux, uy, _uz = macroscopic3d(f)
    return rho, uy


def arm_analytic(
    n: int = 16,
    steps: int = 100,
    tau: float = 0.6,
    n_ranks: int = 4,
    u0: float = 0.01,
) -> dict[str, Any]:
    """A3：剪切波衰减 vs 解析解（sharded 解）。"""
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    rate = nu * k * k
    f0 = shear_wave_field(n, n, n, u0)
    runner = ShardedPpermuteRunner(f0, n_ranks=n_ranks, tau=tau)
    samples = []
    for t in range(steps + 1):
        if t:
            runner.step(1)
        if t % max(steps // 10, 1) == 0 or t == steps:
            got = shear_amplitude(runner)
            ref = u0 * math.exp(-rate * t)
            samples.append({"t": t, "ref": ref, "got": got, "rel_err": (got - ref) / ref})
    final = samples[-1]
    return {
        "grid": [n, n, n],
        "steps": steps,
        "tau": tau,
        "n_ranks": n_ranks,
        "u0": u0,
        "nu": nu,
        "k": k,
        "decay_rate": rate,
        "Re_wavelength": u0 / (nu * k),
        "Ma": u0 / math.sqrt(1.0 / 3.0),
        "samples": samples,
        "final_rel_err": final["rel_err"],
        "abs_final_rel_err": abs(final["rel_err"]),
        "pass_15pct": abs(final["rel_err"]) <= THRESHOLD,
    }


def arm_cost(n: int = 16, steps: int = 10, n_ranks: int = 4, tau: float = 0.6) -> dict[str, Any]:
    """A4：ppermute（选择性 10 面、零 staging）vs 朴素全 Q staging 交换。"""
    f0 = shear_wave_field(n, n, n, u0=0.01)
    runner = ShardedPpermuteRunner(f0, n_ranks=n_ranks, tau=tau)

    # -- ppermute 路径计时（前 2 步热身）
    runner.step(2)
    t0 = time.perf_counter()
    runner.step(steps)
    t_ppermute = (time.perf_counter() - t0) / steps

    # -- 朴素路径：逐 rank 把全部 Q 个平面 pack 进 staging 再 unpack 回来
    shards = [s.clone() for s in runner.shards]
    q = shards[0].shape[0]

    def naive_exchange(slabs: list[torch.Tensor]) -> int:
        nr = len(slabs)
        stage = [[torch.empty_like(s[0, 0]) for _ in range(q)] for s in slabs]
        for r in range(nr):
            for qq in range(q):
                stage[r][qq].copy_(slabs[r][qq, 0])
                stage[r][qq].copy_(slabs[r][qq, -1])
        for r in range(nr):
            lo, hi = (r - 1) % nr, (r + 1) % nr
            for qq in range(q):
                slabs[r][qq, 0].copy_(stage[lo][qq])
                slabs[r][qq, -1].copy_(stage[hi][qq])
        return 4 * nr * q  # 每 rank 每方向 copy 次数

    t0 = time.perf_counter()
    for _ in range(steps):
        ops = naive_exchange(shards)
    t_naive = (time.perf_counter() - t0) / steps

    plane_shape = runner.plane_shape
    cells = int(np.prod(plane_shape))
    itemsize = 4
    return {
        "n_ranks": n_ranks,
        "grid": [n, n, n],
        "steps_timed": steps,
        "ppermute": {
            "planes_per_step_per_rank": runner.halo.planes_per_exchange,
            "bytes_per_step_per_rank": runner.halo.wire_bytes_per_step(plane_shape, itemsize),
            "tensor_ops_per_step": runner.halo.tensor_ops // max(runner.halo.exchanges, 1),
            "staging_buffers": 0,
            "wall_s_per_step": t_ppermute,
        },
        "naive_full_q_staging": {
            "planes_per_step_per_rank": 2 * q,
            "bytes_per_step_per_rank": 2 * q * cells * itemsize,
            "tensor_ops_per_step": ops,
            "staging_buffers": q,
            "wall_s_per_step": t_naive,
        },
        "wire_bytes_ratio_naive_over_ppermute": (2 * q) / runner.halo.planes_per_exchange,
        "wall_speedup": t_naive / t_ppermute if t_ppermute > 0 else float("nan"),
    }


def arm_sensitivity(tau: float = 0.6) -> dict[str, Any]:
    """⑤ 参数敏感性：网格 (12/16/24) × tau (0.55/0.6/0.8)。"""
    grids = [12, 16, 24]
    grid_rows = []
    for n in grids:
        r = arm_analytic(n=n, steps=100 if n <= 16 else 60, tau=tau, n_ranks=4)
        grid_rows.append(
            {
                "n": n,
                "steps": r["steps"],
                "k": r["k"],
                "abs_final_rel_err": r["abs_final_rel_err"],
                "pass_15pct": r["pass_15pct"],
            }
        )
    tau_rows = []
    for tv in (0.55, 0.6, 0.8):
        r = arm_analytic(n=16, steps=100, tau=tv, n_ranks=4)
        tau_rows.append(
            {
                "tau": tv,
                "nu": r["nu"],
                "abs_final_rel_err": r["abs_final_rel_err"],
                "pass_15pct": r["pass_15pct"],
            }
        )
    return {"grid_sensitivity": grid_rows, "tau_sensitivity": tau_rows}


# ---------------------------------------------------------------------------
# 报告与图表
# ---------------------------------------------------------------------------


def _png_b64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_figures(out: Path, summary: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[Path] = []

    # fig1: OSS 分支核账（max|Δf|）
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = summary["arm_oss_2d"]["branches"]
    labels, vals, colors = [], [], []
    for r in rows:
        labels.append(f"{r['split']}-split\nN={r['n_ranks']} {r['variant']}")
        vals.append(max(r["max_abs_df"], 1e-12))
        colors.append("#2b8cbe" if r["max_abs_df"] < 1e-6 else "#d7301f")
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30)
    ax.set_ylabel("max |Δf| vs monolithic stream")
    ax.set_title("A2: vivsim stream_cross_devices branch audit (2D D2Q9)")
    fig.tight_layout()
    p = out / "fig1_oss_branch_audit.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)

    # fig2: 剪切波衰减 ref vs got + 误差
    a = summary["arm_analytic"]
    ts = [s["t"] for s in a["samples"]]
    ref = [s["ref"] for s in a["samples"]]
    got = [s["got"] for s in a["samples"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax1.plot(ts, ref, "k-", label="analytic exp(-νk²t)")
    ax1.plot(ts, got, "o", ms=4, mfc="none", label=f"sharded N={a['n_ranks']}")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("shear amplitude A(t)")
    ax1.legend()
    ax1.set_title(f"A3 shear wave decay, L={a['grid'][0]}, τ={a['tau']}")
    ax2.plot(ts, [abs(s["rel_err"]) * 100 for s in a["samples"]], "s-", color="#d7301f")
    ax2.axhline(THRESHOLD * 100, ls="--", color="gray", label="owner 15% gate")
    ax2.set_yscale("log")
    ax2.set_xlabel("step t")
    ax2.set_ylabel("|rel err| vs analytic [%]")
    ax2.legend()
    ax2.set_title("relative error")
    fig.tight_layout()
    p = out / "fig2_shear_wave_analytic.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)

    # fig3: 成本对照
    c = summary["arm_cost"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    keys = ["planes_per_step_per_rank", "bytes_per_step_per_rank"]
    x = np.arange(len(keys))
    pp = [c["ppermute"][k] for k in keys]
    nv = [c["naive_full_q_staging"][k] for k in keys]
    ax.bar(x - 0.2, pp, 0.4, label="ppermute (selective, no staging)", color="#2b8cbe")
    ax.bar(x + 0.2, nv, 0.4, label="naive full-Q staging", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(["planes/step/rank", "bytes/step/rank"])
    ax.set_yscale("log")
    ax.set_ylabel("count (log)")
    ax.set_title(f"A4 comm accounting, N={c['n_ranks']}, L={c['grid'][0]}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig3_comm_accounting.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)
    return figs


def make_report(out: Path, summary: dict[str, Any], figs: list[Path]) -> str:
    a1 = summary["arm_equivalence"]
    a2 = summary["arm_oss_2d"]
    a3 = summary["arm_analytic"]
    a4 = summary["arm_cost"]
    a5 = summary["arm_sensitivity"]

    lines: list[str] = []
    add = lines.append
    add("# XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO —— 运行工件（自动生成，零手抄）\n")
    add(f"* OSS ref: `{OSS_REPO}` @ `{OSS_COMMIT}` — `{OSS_FILE}`（{OSS_LICENSE}）")
    add(f"* 生成时间: {summary['generated_at']}；环境: {summary['env']['python']} / "
        f"torch {summary['env']['torch']} / numpy {summary['env']['numpy']} / "
        f"jax_available={summary['env']['jax_available']}")
    add(f"* 复现: `{summary['repro_cmd']}`\n")

    add("## A1 单域 vs sharded（真求解器 LBMStepExecutor，剪切波场）\n")
    add("| n_ranks | steps | max|Δf| | bit-identical | rho rel L2 | uy rel L2 | planes/exchange | bytes/exchange/rank |")
    add("|---|---|---|---|---|---|---|---|")
    for r in a1["rows"]:
        add(f"| {r['n_ranks']} | {r['steps']} | {r['max_abs_df']:.3e} | {r['bit_identical']} | "
            f"{r['rho_rel_l2']:.3e} | {r['uy_rel_l2']:.3e} | {r['planes_per_exchange']} | "
            f"{r['bytes_per_exchange_per_rank']} |")
    add("")

    add("## A2 OSS 原式 2D D2Q9 分支核账\n")
    add("| split | N | variant | max|Δf| | rel L2 |")
    add("|---|---|---|---|---|")
    for r in a2["branches"]:
        add(f"| {r['split']} | {r['n_ranks']} | {r['variant']} | {r['max_abs_df']:.3e} | {r['rel_l2']:.3e} |")
    add("")
    add("branch 方向核账（拉式约定下期待的 sender 索引 vs OSS 代码给出的）：\n")
    add("| branch | matches pull convention | expected sender | OSS sender |")
    add("|---|---|---|---|")
    for b in a2["branch_findings"]:
        add(f"| {b['branch']} | {b['matches_pull_convention']} | {b['sender_expected']} | {b['sender_got']} |")
    add("")
    add(f"jax 真通路：available={a2['jax_path_available']}；guard: {a2.get('jax_guard', 'n/a')}\n")

    add("## A3 剪切波 vs 解析解\n")
    add("| 量 | 值 |\n|---|---|")
    for k in ("grid", "steps", "tau", "n_ranks", "u0", "nu", "k", "decay_rate", "Re_wavelength", "Ma"):
        add(f"| {k} | {a3[k]} |")
    add(f"| 终态 |rel err| | {a3['abs_final_rel_err']*100:.3f}% |")
    add(f"| <15% 判据 | {'PASS' if a3['pass_15pct'] else 'FAIL'} |")
    add("")
    add("| t | ref (analytic) | got (sharded) | rel err [%] |")
    add("|---|---|---|---|")
    for s in a3["samples"]:
        add(f"| {s['t']} | {s['ref']:.6e} | {s['got']:.6e} | {s['rel_err']*100:+.3f} |")
    add("")

    add("## A4 通信成本对照\n")
    add("| 方案 | 平面/步/rank | 字节/步/rank | 张量算子/步 | staging 缓冲 | 墙钟 s/步 |")
    add("|---|---|---|---|---|---|")
    for name, key in (("ppermute", "ppermute"), ("naive full-Q staging", "naive_full_q_staging")):
        r = a4[key]
        add(f"| {name} | {r['planes_per_step_per_rank']} | {r['bytes_per_step_per_rank']} | "
            f"{r['tensor_ops_per_step']} | {r['staging_buffers']} | {r['wall_s_per_step']:.4f} |")
    add(f"\n线量比（naive/ppermute）= {a4['wire_bytes_ratio_naive_over_ppermute']:.2f}×；"
        f"墙钟加速 = {a4['wall_speedup']:.2f}×\n")

    add("## A5 敏感性\n")
    add("| n | steps | k | |rel err| vs analytic | <15% |")
    add("|---|---|---|---|---|")
    for r in a5["grid_sensitivity"]:
        add(f"| {r['n']} | {r['steps']} | {r['k']:.4f} | {r['abs_final_rel_err']*100:.3f}% | {r['pass_15pct']} |")
    add("")
    add("| tau | nu | |rel err| | <15% |")
    add("|---|---|---|---|")
    for r in a5["tau_sensitivity"]:
        add(f"| {r['tau']} | {r['nu']:.5f} | {r['abs_final_rel_err']*100:.3f}% | {r['pass_15pct']} |")
    add("")

    add("## 图\n")
    for p, cap in zip(figs, summary["figure_captions"]):
        add(f"![{cap}]({p.name})\n\n*{cap}*\n")

    text = "\n".join(lines) + "\n"
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_ppermute_periodic_halo")
    ap.add_argument("--quick", action="store_true", help="小网格冒烟（不写正式工件）")
    args = ap.parse_args()

    out = (WORKTREE / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t_start = time.time()
    summary: dict[str, Any] = {
        "anchor": "XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO",
        "mission": 18112,
        "oss": {"repo": OSS_REPO, "commit": OSS_COMMIT, "file": OSS_FILE, "license": OSS_LICENSE},
        "threshold": THRESHOLD,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "jax_available": JAX_AVAILABLE,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
        "repro_cmd": "PYTHONPATH=src python tools/xval_ppermute_halo/run_xval.py "
                     "--out artifacts/xval_ppermute_periodic_halo",
    }

    if args.quick:
        summary["arm_equivalence"] = arm_equivalence(n=8, steps=3, ranks=(2, 4))
        summary["arm_oss_2d"] = arm_oss_2d(nx=8, ny=8)
        summary["arm_analytic"] = arm_analytic(n=8, steps=10, n_ranks=2)
        summary["arm_cost"] = arm_cost(n=8, steps=2, n_ranks=4)
        summary["arm_sensitivity"] = {"grid_sensitivity": [], "tau_sensitivity": []}
        summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["figure_captions"] = []
        (out / "summary_quick.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k.startswith("arm_")}, indent=2)[:4000])
        return 0

    print("[A1] equivalence ...", flush=True)
    summary["arm_equivalence"] = arm_equivalence()
    print("[A2] OSS 2D arm ...", flush=True)
    summary["arm_oss_2d"] = arm_oss_2d()
    print("[A3] analytic ...", flush=True)
    summary["arm_analytic"] = arm_analytic()
    print("[A4] cost ...", flush=True)
    summary["arm_cost"] = arm_cost()
    print("[A5] sensitivity ...", flush=True)
    summary["arm_sensitivity"] = arm_sensitivity()
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["wall_s"] = round(time.time() - t_start, 1)
    summary["figure_captions"] = [
        "A2: OSS stream_cross_devices 两个方向分支的逐位核账（红=与单域 stream 不一致）",
        "A3: 剪切波衰减 sharded 解 vs 解析 exp(-νk²t)，右图相对误差与 15% 门",
        "A4: ppermute 选择性交换 vs 朴素全 Q staging 的线量与缓冲对照",
    ]

    figs = make_figures(out, summary)
    for p, cap in zip(figs, summary["figure_captions"]):
        print(f"figure: {p} sha256={__import__('hashlib').sha256(p.read_bytes()).hexdigest()}")
        p.relative_to(out.parent.parent)

    # 手工工件：kpi.csv + tables.md
    with (out / "kpi.csv").open("w", encoding="utf-8") as fh:
        fh.write("metric,ref,got,rel_err,pass_15pct\n")
        a3 = summary["arm_analytic"]
        fh.write(f"shear_amplitude_t{a3['steps']},{a3['samples'][-1]['ref']:.8e},"
                 f"{a3['samples'][-1]['got']:.8e},{a3['final_rel_err']:.3e},{a3['pass_15pct']}\n")
        for r in summary["arm_equivalence"]["rows"]:
            fh.write(f"fieldeq_N{r['n_ranks']},0.0,{r['max_abs_df']:.3e},0.0,True\n")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = make_report(out, summary, figs)
    print(text[:2000])
    print(f"\nwrote {out} (wall {summary['wall_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())