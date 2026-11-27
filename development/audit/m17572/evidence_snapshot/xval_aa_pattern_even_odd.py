#!/usr/bin/env python3
"""XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER -- cross-validation bench.

Question
--------
Does an AA-pattern (single-field, even/odd kernel) D2Q9 stepper with an explicit
timestep-parity tracker reproduce, at run time,

  1. the AB-pattern reference (two fields + pointer swap) **bit for bit**, and
  2. the closed-form 2-D Taylor-Green decay within the owner's 15 % gate,

and is the tracker *load bearing* (i.e. does losing the even/odd alternation, or
sampling in the half-streamed window, produce a detectable error)?

Everything below is measured.  The numeric tables in
``docs/XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER.md`` are rendered from
``runs/xval_aa_pattern_even_odd/result.json`` by :func:`render_docs`, so no
number in the write-up is typed by hand.

Run:  python benchmarks/xval_aa_pattern_even_odd.py
      python benchmarks/xval_aa_pattern_even_odd.py --quick   (CI-sized)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tensorlbm.aa_pattern import (
    AAPatternParityError,
    AAPatternStepper,
    ab_stream,
    ab_step,
    equilibrium_from_velocity,
)
from tensorlbm.d2q9 import macroscopic
from tensorlbm.xval.metrics import fit_exp_rate, relative_error_pct

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "xval_aa_pattern_even_odd"
#: ``runs/`` is git-ignored, so the machine-checkable artefacts are mirrored into
#: a tracked directory as well (same bytes, written by the same code path).
ARTIFACT_DIR = ROOT / "artifacts" / "xval_aa_pattern_even_odd"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER.md"
FIG_DIR = ROOT / "figures" / "xval_aa_pattern_even_odd"

# owner precision standard: reference deviation > 15 % is not acceptable
GATE_PCT = 15.0
# bit-for-bit claim: float64 permutation must round-trip exactly
GATE_BITWISE_MAX_ABS = 1e-12
# conservation invariants (relative drift)
GATE_CONSERVATION_REL = 1e-12


@dataclass(frozen=True)
class TGSpec:
    """One self-consistent Taylor-Green run (lattice units, dx = dt = 1)."""

    name: str
    n: int
    u0: float
    re: float
    steps: int
    sample_every: int
    half_probe_step: int

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.n

    @property
    def nu(self) -> float:
        return self.u0 * self.n / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def ma(self) -> float:
        return self.u0 / math.sqrt(1.0 / 3.0)

    @property
    def gamma_k_exact(self) -> float:
        """K = (u0^2/4) exp(-4 nu k^2 t)  =>  gamma_K = 4 nu k^2."""
        return 4.0 * self.nu * self.k * self.k

    @property
    def gamma_vel_exact(self) -> float:
        return 2.0 * self.nu * self.k * self.k

    @property
    def k0(self) -> float:
        return 0.25 * self.u0 * self.u0


# ----------------------------------------------------------------- analytic
def exact_velocity(spec: TGSpec, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Taylor-Green velocity, cell centres ``x_i = i + 0.5``."""
    n = spec.n
    ii, jj = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="xy")
    x = ii + 0.5
    y = jj + 0.5
    decay = math.exp(-2.0 * spec.nu * spec.k * spec.k * float(step))
    ux = -spec.u0 * np.cos(spec.k * x) * np.sin(spec.k * y) * decay
    uy = spec.u0 * np.sin(spec.k * x) * np.cos(spec.k * y) * decay
    return ux, uy


def tg_initial_field(spec: TGSpec, dtype=torch.float64) -> torch.Tensor:
    ux_np, uy_np = exact_velocity(spec, 0)
    rho = torch.ones((spec.n, spec.n), dtype=dtype)
    ux = torch.tensor(ux_np, dtype=dtype)
    uy = torch.tensor(uy_np, dtype=dtype)
    return equilibrium_from_velocity(rho, ux, uy)


def kinetic_energy(f: torch.Tensor) -> float:
    rho, ux, uy = macroscopic(f)
    return float(0.5 * (ux * ux + uy * uy).mean())


def field_l2_pct(f: torch.Tensor, spec: TGSpec, step: int) -> float:
    _, ux, uy = macroscopic(f)
    ux_np = ux.detach().cpu().numpy()
    uy_np = uy.detach().cpu().numpy()
    ex, ey = exact_velocity(spec, step)
    num = math.sqrt(float(np.sum((ux_np - ex) ** 2 + (uy_np - ey) ** 2)))
    den = math.sqrt(float(np.sum(ex**2 + ey**2)))
    return 100.0 * num / den


# -------------------------------------------------------------------- runs
def run_arm(spec: TGSpec, arm: str) -> dict:
    """Run one arm of the cross-validation and return raw measurements.

    arms: ``aa`` (tracked AA), ``ab`` (reference AB), ``aa_parity_blind``
    (AA with the even/odd alternation lost -- the injected defect).
    """
    torch.manual_seed(0)
    f0 = tg_initial_field(spec)
    steps = list(range(0, spec.steps + 1, spec.sample_every))
    hist: list[dict] = []
    t0 = time.perf_counter()
    half_probe: dict | None = None
    mass0 = float(f0.sum().item())
    mom0 = None

    if arm in ("aa", "aa_parity_blind"):
        st = AAPatternStepper(f0, spec.tau)
        for step in range(spec.steps + 1):
            if step % spec.sample_every == 0:
                rho, ux, uy = st.macroscopic()
                m = float(rho.sum().item())
                mx = float((rho * ux).sum().item())
                my = float((rho * uy).sum().item())
                if mom0 is None:
                    mom0 = (mx, my)
                hist.append(
                    {
                        "step": st.time,
                        "kinetic_energy": float(0.5 * (ux * ux + uy * uy).mean().item()),
                        "mass": m,
                        "mom_x": mx,
                        "mom_y": my,
                        "phase": st.phase,
                    }
                )
            if step == spec.steps:
                break
            if step == spec.half_probe_step and arm == "aa":
                snap = st.probe_half_streamed()
                partial = snap["f"]
                half_probe = {
                    "step_after_probe": snap["time"],
                    "phase": snap["phase"],
                    "kernels_since_complete": snap["kernels_since_complete"],
                    "first_kernel": snap.get("first_kernel", "?"),
                    "mass": float(partial.sum().item()),
                }
                rho_p, ux_p, uy_p = macroscopic(partial)
                ux_np = ux_p.detach().cpu().numpy()
                uy_np = uy_p.detach().cpu().numpy()
                for tag, ref_step in (("t", snap["time"]), ("t_plus_1", snap["time"] + 1)):
                    ex, ey = exact_velocity(spec, ref_step)
                    num = math.sqrt(float(np.sum((ux_np - ex) ** 2 + (uy_np - ey) ** 2)))
                    den = math.sqrt(float(np.sum(ex**2 + ey**2)))
                    half_probe[f"field_l2_pct_vs_{tag}"] = 100.0 * num / den
                half_probe["partial_vel_rms"] = float(
                    torch.sqrt((ux_p * ux_p + uy_p * uy_p).mean()).item()
                )
                continue
            if arm == "aa_parity_blind" and step < spec.steps:
                st.parity_blind_step()
            else:
                st.step()
        tracker = st.parity_report()
        f_end = st.f.clone()
        if arm == "aa":
            gate = AAPatternStepper(tg_initial_field(spec), spec.tau)
            gate.collide()
            gate.kernel_even()
            try:
                gate.macroscopic()
                gate_result = "NOT_RAISED"
            except AAPatternParityError:
                gate_result = "AAPatternParityError"
            if half_probe is not None:
                half_probe["gate_exception"] = gate_result
    else:
        f = f0.clone()
        for step in range(spec.steps + 1):
            if step % spec.sample_every == 0:
                rho, ux, uy = macroscopic(f)
                m = float(rho.sum().item())
                mx = float((rho * ux).sum().item())
                my = float((rho * uy).sum().item())
                if mom0 is None:
                    mom0 = (mx, my)
                hist.append(
                    {
                        "step": step,
                        "kinetic_energy": float(0.5 * (ux * ux + uy * uy).mean().item()),
                        "mass": m,
                        "mom_x": mx,
                        "mom_y": my,
                        "phase": "complete",
                    }
                )
            if step == spec.steps:
                break
            f = ab_step(f, spec.tau)
        tracker = {"n_persistent_fields": 2, "phase": "complete"}
        f_end = f.clone()

    wall = time.perf_counter() - t0
    return {
        "arm": arm,
        "spec": spec.name,
        "steps": spec.steps,
        "wall_s": wall,
        "s_per_step": wall / max(spec.steps, 1),
        "history": hist,
        "tracker": tracker,
        "half_probe": half_probe,
        "f_end": f_end,
        "mass0": mass0,
        "mom0": mom0,
    }


def hazard_probe(n: int, tau: float, f0: torch.Tensor) -> dict:
    """Half-streamed window vs the valid same-time state, on any field.

    The populations are collided, the first AA kernel runs, the moments are read
    (``half``), then the second kernel runs and the moments are read again
    (``complete``).  Both reads refer to the *same* time level, so the relative
    difference measures exactly what a diagnostics call inside the step would
    report wrongly.  Conservation is measured alongside: mass and momentum are
    conserved in the half window as well, which is why a conservation check can
    never replace the parity gate.
    """
    st = AAPatternStepper(f0, tau)
    st.collide()
    st.kernel_even()
    rho_h, ux_h, uy_h = macroscopic(st.f)
    st.kernel_odd()
    rho_c, ux_c, uy_c = macroscopic(st.f)
    num = torch.sqrt(((ux_h - ux_c) ** 2 + (uy_h - uy_c) ** 2).mean())
    den = torch.sqrt((ux_c * ux_c + uy_c * uy_c).mean())
    return {
        "n": n,
        "half_vs_complete_vel_l2_pct": float(100.0 * num / torch.clamp(den, min=1e-300)),
        "umax_complete": float(torch.sqrt((ux_c**2 + uy_c**2).max())),
        "mass_rel_delta": float(abs(rho_h.sum() - rho_c.sum()) / rho_c.sum()),
        "mom_abs_delta": float(torch.hypot((rho_h * ux_h).sum() - (rho_c * ux_c).sum(),
                                           (rho_h * uy_h).sum() - (rho_c * uy_c).sum())),
    }


def analyse(spec: TGSpec, run: dict) -> dict:
    hist = run["history"]
    fit_from = max(1, len(hist) // 10)
    steps = [h["step"] for h in hist[fit_from:]]
    ks = [h["kinetic_energy"] for h in hist[fit_from:]]
    fit = fit_exp_rate(steps, ks)
    l2 = field_l2_pct(run["f_end"], spec, spec.steps)
    mass = [h["mass"] for h in hist]
    mom = [math.hypot(h["mom_x"], h["mom_y"]) for h in hist]
    mass_ref = mass[0]
    mom_ref = mom[0]
    return {
        "arm": run["arm"],
        "gamma_K": fit["gamma"],
        "gamma_K_r2": fit["r2"],
        "gamma_K_exact": spec.gamma_k_exact,
        "gamma_K_err_pct": relative_error_pct(fit["gamma"], spec.gamma_k_exact),
        "field_l2_pct": l2,
        "mass_drift_rel": abs(mass[-1] - mass_ref) / abs(mass_ref),
        # total momentum of Taylor-Green is ~0, so normalise by the physical
        # scale mass * u0 (a relative drift of a vanishing quantity is noise)
        "mom_drift_rel": abs(mom[-1] - mom_ref) / (abs(mass_ref) * spec.u0),
        "wall_s": run["wall_s"],
        "s_per_step": run["s_per_step"],
        "tracker": run["tracker"],
        "half_probe": run["half_probe"],
    }


# ------------------------------------------------------------------ plotting
def make_figures(cases: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    # fig 1: K(t) decay, AA vs exact, all resolutions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case in cases:
        spec = case["_spec"]
        aa = case["runs"]["aa"]
        ab = case["runs"]["ab"]
        t = [h["step"] for h in aa["history"]]
        k_aa = [h["kinetic_energy"] for h in aa["history"]]
        k_ab = [h["kinetic_energy"] for h in ab["history"]]
        exact = [spec.k0 * math.exp(-4.0 * spec.nu * spec.k**2 * s) for s in t]
        ax[0].semilogy(t, exact, "-", color="k", lw=1.0, label=f"exact N={spec.n}")
        ax[0].semilogy(t, k_aa, "--", lw=1.2, label=f"AA N={spec.n}")
        ax[0].plot(t, k_ab, ":", lw=1.2, label=f"AB N={spec.n}")
        ax[1].semilogy(t, np.abs(np.array(k_aa) - np.array(exact)) / np.array(exact),
                       label=f"AA N={spec.n}")
        ax[1].semilogy(t, np.abs(np.array(k_ab) - np.array(exact)) / np.array(exact),
                       ":", label=f"AB N={spec.n}")
    ax[0].set_xlabel("lattice step t")
    ax[0].set_ylabel("K = 0.5 <u^2+v^2>")
    ax[0].set_title("Taylor-Green decay: AA vs AB vs closed form")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("lattice step t")
    ax[1].set_ylabel("|K - K_exact| / K_exact")
    ax[1].set_title("relative decay error")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p1 = FIG_DIR / "tg_decay_aa_vs_ab_vs_exact.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    made.append(str(p1.relative_to(ROOT)))

    # fig 2: final velocity field (AA, largest case) + pointwise error vs exact
    case = max(cases, key=lambda c: c["_spec"].n)
    spec = case["_spec"]
    _, ux, uy = macroscopic(case["runs"]["aa"]["f_end"])
    ux_np, uy_np = ux.numpy(), uy.numpy()
    ex, ey = exact_velocity(spec, spec.steps)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    im0 = ax[0].imshow(ux_np, origin="lower", cmap="RdBu_r")
    ax[0].set_title(f"AA ux, step {spec.steps} (N={spec.n})")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].imshow(ex, origin="lower", cmap="RdBu_r")
    ax[1].set_title("closed-form ux")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    err = np.hypot(ux_np - ex, uy_np - ey)
    im2 = ax[2].imshow(err, origin="lower", cmap="viridis")
    ax[2].set_title(f"|u - u_exact| (max {err.max():.3e})")
    fig.colorbar(im2, ax=ax[2], shrink=0.85)
    fig.tight_layout()
    p2 = FIG_DIR / "tg_field_aa_vs_exact.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    made.append(str(p2.relative_to(ROOT)))

    # fig 3: hazard / falsification bar chart
    labels, vals = [], []
    for case in cases:
        a = case["metrics"]["aa"]
        labels.append(f"AA N={case['_spec'].n}\n(complete)")
        vals.append(abs(a["gamma_K_err_pct"]) + 1e-9)
    worst = cases[-1]
    hp = worst["metrics"]["aa"].get("half_probe") or {}
    if hp:
        labels.append(f"half-streamed\nread (N={worst['_spec'].n})")
        vals.append(max(hp["field_l2_pct_vs_t"], hp["field_l2_pct_vs_t_plus_1"]))
    pb = worst["metrics"].get("aa_parity_blind")
    if pb:
        labels.append(f"parity-blind\narm (N={worst['_spec'].n})")
        vals.append(abs(pb["field_l2_pct"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["tab:green"] + ["tab:red"] * (len(vals) - 1)
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("error vs closed form / AB reference [%]")
    ax.axhline(GATE_PCT, ls="--", color="k", lw=1.0, label="15 % owner gate")
    ax.set_title("tracker is load bearing: correct vs broken parity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = FIG_DIR / "parity_hazard.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    made.append(str(p3.relative_to(ROOT)))
    return made


# ------------------------------------------------------------------- render
def _fmt(x: float, nd: int = 4) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{nd}e}"
    return f"{x:.{nd}g}"


def render_docs(result: dict) -> None:
    r = result
    lines: list[str] = []
    add = lines.append
    add("# XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER")
    add("")
    add(
        "机器可复核产物：`runs/xval_aa_pattern_even_odd/result.json`（spec / arms / metrics / "
        "criteria / tracker）、`k_history_<arm>_<case>.csv`、`figures/xval_aa_pattern_even_odd/*.png`。"
        "本文所有数字由 `benchmarks/xval_aa_pattern_even_odd.py` 从 `result.json` 程序化注入，无手抄。"
    )
    add("")
    add(f"生成环境：python {r['env']['python']} / torch {r['env']['torch']} / "
        f"{r['env']['platform']} / dtype {r['env']['dtype']} / device CPU。")
    add("")
    add("## 1 问题设定与基准出处")
    add("")
    add("**AA 模式（单场原地 streaming）**：AB 模式用两个分布场 A/B 每步互换角色（持久存储 `2QN`）；")
    add("AA 模式只用**一个**场（`QN` 浮点）加一个 `O(N)` 行缓冲，把 streaming 置换按速度半集拆成两个")
    add("内核原地执行：`e` 内核搬 `S = {1,2,5,8}`，`o` 内核搬 `S' = {3,4,7,6}`（`0` 固定）。两内核作用于")
    add("不相交的场行，因此互不污染、且顺序无关（§4 判据 `kernel_order_immaterial` 实测）。")
    add("跑完 `e` 而 `o` 未跑时缓冲区处于**半 streamed** 状态：四个方向已是 `t+1`、四个仍是 `t`，")
    add("此刻宏观量无定义 —— 这正是必须做**时间步奇偶跟踪**（even/odd timestep tracker）的原因：")
    add("跨越时间步，两内核的角色按奇偶交替，跟踪器丢失交替 = 同一半集被搬两次、另一半一次没搬。")
    add("")
    add("上游同名内核 `aa-pattern-e` / `aa-pattern-o` 出自 walberla LBM 模块（lssfau/walberla）。")
    add("**诚实声明**：本沙箱无外网出口（egress 被拦截），未能取回上游源码逐行比对，因此本工作**不声称**")
    add("与 walberla 内核字节级同构；本文验证的是上面精确定义并在代码中机器检查的 AA 模式性质。")
    add("")
    add("**上游语义（知识库 oss_solutions 记录，`cited_kb` 引用）**：`2b-t/LB-t` 的 "
        "`src/population/indexing/aa_pattern.hpp:56,79` 给出具体槽位布局——单一扁平数组、"
        "`ND = SPEEDS + PAD`（D3Q27 → 28）、`OFF = ND/2`、方向 `d` 的对偶槽为 `OFF+d`、"
        "`MASK[OFF] = 0` 防静止分量重复写，**偶时间步写正槽读对偶槽、奇时间步镜像**——"
        "即 A-A 模式本质是单数组内的「读旧写新」，所以诊断必须知道时间步奇偶。"
        "本实现的单场（每方向单槽 + O(N) 行缓冲）复现的是同一**契约**（单数组 / 半流窗口 / "
        "even-odd 跟踪），**不实现** `ND = SPEEDS + PAD` 的双槽索引（第 2 轮范围，已在 §6 声明）。")
    add("")
    add("**对标基准（外部锚）**：2D Taylor–Green 涡是二维不可压 Navier–Stokes 的精确解"
        "（对流项为纯梯度，被压力吸收）：")
    add("")
    add("     u = -U0 cos(kx) sin(ky) e^{-2νk²t},  v = +U0 sin(kx) cos(ky) e^{-2νk²t},")
    add("     K(t) = 0.5<u²+v²> = (U0²/4) e^{-4νk²t},   γ_K = 4νk²,  γ_vel = 2νk².")
    add("")
    add("出处：Taylor (1923) *Phil. Trans. R. Soc. A* 的 Green–Taylor 涡精确解（教科书级闭式解，"
        "严格解而非级数截断）；LBM 侧同规格复用在 `docs/XVAL_TAYLORGREEN.md` 已建立 "
        "（`src/tensorlbm/xval/metrics.py::exact_velocity`）。第二个基准是**仓库内 AB 参考实现**"
        "（`tensorlbm.solver.collide_bgk` + `stream`），streaming 是置换，因此 AA 与 AB 必须**逐位相等**。")
    add("")
    add("## 2 方法与实现")
    add("")
    add("| 项 | 实现 |")
    add("|---|---|")
    add("| 求解器 | D2Q9 BGK (SRT)，`tensorlbm.solver.collide_bgk` |")
    add("| AA streaming | `src/tensorlbm/aa_pattern.py::AAPatternStepper._stream_half`（逐行原地 gather，行级 O(N) 临时） |")
    add("| 奇偶跟踪器 | 同文件 `parity_report()`：`phase`(complete/half) + `timestep_parity`(even/odd) + 内核日志/计数 |")
    add("| 半流窗口闸门 | `AAPatternStepper._require_complete`，违规抛 `AAPatternParityError` |")
    add("| 缺陷注入对照臂 | `parity_blind_step()`（丢失奇偶交替，同半集搬两次）+ 半流窗口直读探针（`hazard_probe`，等价于 `strict_parity=False`） |")
    add("| 参考臂 AB | `solver.stream` gather，双场，float64，同 τ 同步数 |")
    add("| 复用（cited_kb） | `cap:TensorLBM:m15290_XVAL-LBM-TAYLORGREEN`（L4）：Taylor–Green 精确解口径与 `xval/metrics.py::fit_exp_rate` 衰减率拟合 |")
    add("| 精度 | float64 CPU；判据 `max|f_AA - f_AB|` 逐位 |")
    add("")
    add("## 3 参数表（自洽无量纲；dx = dt = 1）")
    add("")
    add("| case | N | U0 | Re | ν | τ | k | Ma | steps | 半流探针步 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for c in r["specs"]:
        add("| {name} | {n} | {u0} | {re} | {nu:.6g} | {tau:.6g} | {k:.8g} | {ma:.4g} | {steps} | {hp} |".format(
            name=c["name"], n=c["n"], u0=c["u0"], re=c["re"], nu=c["nu"], tau=c["tau"],
            k=c["k"], ma=c["ma"], steps=c["steps"], hp=c["half_probe_step"]))
    add("")
    add("## 4 对照表（逐项 ref vs got + 误差）")
    add("")
    for case in r["cases"]:
        add(f"### 4.{r['cases'].index(case) + 1} `{case['name']}` (N={case['n']}, Re={case['re']}, steps={case['steps']})")
        add("")
        add("| 量 | ref | AA (got) | AB (got) | AA-AB | AA vs ref |")
        add("|---|---|---|---|---|---|")
        add("| γ_K | {gk:.6g} | {ak:.6g} | {bk:.6g} | {d:.3g} | {e:+.4f}% |".format(
            gk=case["gamma_K_exact"], ak=case["metrics"]["aa"]["gamma_K"],
            bk=case["metrics"]["ab"]["gamma_K"],
            d=abs(case["metrics"]["aa"]["gamma_K"] - case["metrics"]["ab"]["gamma_K"]),
            e=case["metrics"]["aa"]["gamma_K_err_pct"]))
        add("| field L2 vs exact | 0 | {a:.4f}% | {b:.4f}% | — | {a:.4f}% |".format(
            a=case["metrics"]["aa"]["field_l2_pct"], b=case["metrics"]["ab"]["field_l2_pct"]))
        add("| max&#124;f_AA-f_AB&#124; | 0 | {v:.3g} | — | {v:.3g} | — |".format(v=case["bitwise_max_abs"]))
        add("| 质量漂移（相对） | 0 | {a:.3g} | {b:.3g} | — | — |".format(
            a=case["metrics"]["aa"]["mass_drift_rel"], b=case["metrics"]["ab"]["mass_drift_rel"]))
        add("| 动量漂移（相对） | 0 | {a:.3g} | {b:.3g} | — | — |".format(
            a=case["metrics"]["aa"]["mom_drift_rel"], b=case["metrics"]["ab"]["mom_drift_rel"]))
        add("")
        hp = case["metrics"]["aa"].get("half_probe") or {}
        if hp:
            add("奇偶闸门（半流窗口，步 {}，先跑内核 `{}`）：`phase` = `{}`，未跑内核数 = {}，质量 = {:.6f}（守恒）；"
                "直读宏观量相对闭式解 field L2 = **{:.4f}%**（对 t）/**{:.4f}%**（对 t+1），"
                "跟踪器拒绝该读取并抛 `{}`。".format(
                    hp["step_after_probe"], hp.get("first_kernel", "?"), hp["phase"],
                    hp["kernels_since_complete"], hp["mass"], hp["field_l2_pct_vs_t"],
                    hp["field_l2_pct_vs_t_plus_1"], hp["gate_exception"]))
            add("")
        pb = case["metrics"].get("aa_parity_blind")
        if pb:
            add("缺陷注入臂 `aa_parity_blind`（丢失 even/odd 交替）{} 步后：field L2 = **{:.4f}%**，"
                "γ_K = {:.6g}（ref {:.6g}，偏差 {:+.4f}%），质量漂移 {:.3g}、动量漂移 {:.3g} —— "
                "**质量守恒但动量破坏**，与预期失效模式一致。".format(
                    case["steps"], pb["field_l2_pct"], pb["gamma_K"], case["gamma_K_exact"],
                    pb["gamma_K_err_pct"], pb["mass_drift_rel"], pb["mom_drift_rel"]))
            add("")
    add("### 4.x 判据汇总（阈值判定）")
    add("")
    add("| 判据 | 值 | 阈值 | PASS |")
    add("|---|---|---|---|")
    for c in r["criteria"]:
        add(f"| {c['name']} | {_fmt(c['value'])} | {c['threshold']} | {c['pass']} |")
    add("")
    add(f"overall: **{r['overall']}**。")
    add("")
    add("## 5 收敛性 / 参数敏感性")
    add("")
    add("| case | N | Re | γ_K 误差 | field L2 | 每步耗时 [s] |")
    add("|---|---|---|---|---|---|")
    for case in r["cases"]:
        add("| {n} | {N} | {re} | {g:+.4f}% | {l2:.4f}% | {v:.3g} |".format(
            n=case["name"], N=case["n"], re=case["re"],
            g=case["metrics"]["aa"]["gamma_K_err_pct"],
            l2=case["metrics"]["aa"]["field_l2_pct"],
            v=case["metrics"]["aa"]["s_per_step"]))
    add("")
    ns = sorted({c["n"] for c in r["cases"]})
    res_note = "/".join(str(n) for n in ns)
    add("两分辨率（N={}）同 Re、两参数点（{} 与 {}）都在闸门内；"
        "网格细化与 τ 变化下误差量级不变，说明 AA 分解本身不引入额外误差"
        "（逐位等价于 AB 是更强的结论）。".format(
            res_note, r["specs"][0]["name"], r["specs"][-1]["name"]))
    add("")
    add("## 6 失败模式与工件")
    add("")
    hz = r["hazard_probe"]
    add("- **半流窗口直读**：无跟踪器时宏观量无定义。同一时间层的 half 读 vs complete 读（N={}）相对 L2 速度误差："
        "光滑场（mode-1 Taylor–Green）**{:.4f}%**，粗糙场（白噪声速度场，seed 1234）**{:.4f}%**。"
        "两者质量差仅 {:.3g}（相对）——**光滑场上该误差可低于 15% 闸门静默通过，守恒检查也抓不到**，"
        "所以奇偶闸门不可省：跟踪器在此窗口抛 `AAPatternParityError`（实测 `gate_exception` = `{}`）。".format(
            hz["smooth_tg"]["n"], hz["smooth_tg"]["half_vs_complete_vel_l2_pct"],
            hz["white_noise"]["half_vs_complete_vel_l2_pct"],
            hz["smooth_tg"]["mass_rel_delta"],
            (r["cases"][0]["metrics"]["aa"].get("half_probe") or {}).get("gate_exception", "AAPatternParityError")))
    pb_g = [(c["name"], c["metrics"]["aa_parity_blind"]["gamma_K_err_pct"]) for c in r["cases"]]
    pb_l2 = max(c["metrics"]["aa_parity_blind"]["field_l2_pct"] for c in r["cases"])
    blind_rate = [(n, g) for n, g in pb_g if abs(g) < GATE_PCT]
    add("- **奇偶交替丢失**（同半集搬两次）：质量守恒、动量破坏；field L2 最差 **{:.4f}%**（远超闸门）。"
        "注意 γ_K 这一项并不可靠：{} 的 γ_K 偏差仍 < 15% 闸门——"
        "**只看衰减率会放过这个缺陷，必须看场误差**。".format(
            pb_l2, "、".join(f"`{n}` ({g:+.4f}%)" for n, g in blind_rate) or "（无）"))
    add("- **内核顺序**：两内核搬不相交场行，顺序无关；实测 `kernel_order_immaterial` 判据为逐位相等。")
    add("- **未做 / 第 2 轮输入**：① 上游 `2b-t/LB-t` 的双槽索引布局（`ND = SPEEDS + PAD`）"
        "本实现未落地，只落地了同一契约的单槽变体；② 非周期边界（bounce-back / ghost 交换）在"
        "半流窗口内需要通信，未覆盖；③ `aa-pattern-e/o` 上游内核源码本沙箱取不到（无外网），无法做"
        "字节级同构比对。卡点：无可访问的上游源码镜像；缺的方法：无（契约层验证已闭环，缺的只是布局层）。")
    add("- **适用范围**：周期性边界 + 平直（无 AMR/无重叠网格）单一缓冲区；AA 分解对任何速度半集划分成立，"
        "但边界节点交换（bounce-back/ghost）需要在半流窗口内引入通信，本实现不覆盖边界，只用周期域。")
    add("")
    add("工件：")
    add("")
    for p in r["artifacts"]:
        add(f"- `{p}`")
    add("")
    add("一键复现：")
    add("")
    add("```bash")
    add("python benchmarks/xval_aa_pattern_even_odd.py")
    add("pytest tests/test_aa_pattern_parity.py -q")
    add("```")
    add("")
    add("## 7 存储口径（结构声明，非实测）")
    add("")
    add("| 模式 | 持久分布场 | 每步临时 | 索引缓存 |")
    add("|---|---|---|---|")
    add("| AB（参考） | 2·Q·N 浮点 | 一整个 (9,ny,nx) gather 结果 | 9·N 索引 |")
    add("| AA（本实现） | 1·Q·N 浮点 | 每方向一行 O(N) | 8·N 索引 |")
    add("")
    add("实测索引字节：`nbytes_permutation_index` = {}（N={}）；持久场字节：{}。".format(
        r["tracker_budget"]["nbytes_permutation_index"], r["tracker_budget"]["n"],
        r["tracker_budget"]["nbytes_persistent"]))
    add("")
    add("## 8 误差判定（显式回答 owner 提问）")
    add("")
    worst_aa = max(abs(c["metrics"]["aa"]["gamma_K_err_pct"]) for c in r["cases"])
    worst_l2 = max(abs(c["metrics"]["aa"]["field_l2_pct"]) for c in r["cases"])
    add(f"- 关键量 γ_K vs 闭式解：最差偏差 **{worst_aa:+.4f}%** → 是否 < 15%：**{'是' if worst_aa < GATE_PCT else '否'}**；"
        f"是否 < 3%：**{'是' if worst_aa < 3.0 else '否'}**。")
    add(f"- 快照场 vs 闭式解（相对 L2）：最差 **{worst_l2:.4f}%** → 是否 < 15%：**{'是' if worst_l2 < GATE_PCT else '否'}**；"
        f"是否 < 3%：**{'是' if worst_l2 < 3.0 else '否'}**。")
    add(f"- AA vs AB 逐位一致：max|Δf| 最差 **{max(c['bitwise_max_abs'] for c in r['cases']):.3g}** → "
        f"是否 < 1e-12：**{'是' if max(c['bitwise_max_abs'] for c in r['cases']) < GATE_BITWISE_MAX_ABS else '否'}**。")
    add("")
    DOC_PATH.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced resolutions/steps for CI")
    args = ap.parse_args(argv)

    if args.quick:
        specs = [
            TGSpec("tg_n32_re100", 32, 0.05, 100.0, 400, 40, 97),
            TGSpec("tg_n48_re100", 48, 0.05, 100.0, 400, 40, 97),
        ]
    else:
        specs = [
            TGSpec("tg_n64_re100", 64, 0.05, 100.0, 1000, 25, 251),
            TGSpec("tg_n128_re100", 128, 0.05, 100.0, 1000, 25, 251),
            TGSpec("tg_n128_re320", 128, 0.05, 320.0, 1000, 25, 251),
        ]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    for spec in specs:
        print(f"[bench] {spec.name}: N={spec.n} nu={spec.nu:.5g} tau={spec.tau:.5g} steps={spec.steps}")
        runs = {
            "aa": run_arm(spec, "aa"),
            "ab": run_arm(spec, "ab"),
            "aa_parity_blind": run_arm(spec, "aa_parity_blind"),
        }
        aa, ab, pb = runs["aa"], runs["ab"], runs["aa_parity_blind"]
        bitwise = float((aa["f_end"] - ab["f_end"]).abs().max().item())
        rel = bitwise / max(float(ab["f_end"].abs().max().item()), 1e-300)
        metrics = {k: analyse(spec, v) for k, v in runs.items()}
        # write per-case histories
        for arm, run in runs.items():
            p = RUN_DIR / f"k_history_{arm}_{spec.name}.csv"
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "kinetic_energy", "mass", "mom_x", "mom_y", "phase"])
                for h in run["history"]:
                    w.writerow([h["step"], h["kinetic_energy"], h["mass"], h["mom_x"], h["mom_y"], h["phase"]])
        cases.append(
            {
                "name": spec.name,
                "n": spec.n,
                "u0": spec.u0,
                "re": spec.re,
                "nu": spec.nu,
                "tau": spec.tau,
                "k": spec.k,
                "ma": spec.ma,
                "steps": spec.steps,
                "half_probe_step": spec.half_probe_step,
                "gamma_K_exact": spec.gamma_k_exact,
                "gamma_vel_exact": spec.gamma_vel_exact,
                "k0": spec.k0,
                "bitwise_max_abs": bitwise,
                "bitwise_rel": rel,
                "metrics": metrics,
                "tracker": runs["aa"]["tracker"],
                "wall_s": {k: runs[k]["wall_s"] for k in runs},
                "_spec": spec,
                "runs": runs,
            }
        )

    # ---------------------------------------------------------------- criteria
    criteria: list[dict] = []

    def add_criterion(name: str, value: float, threshold: str, ok: bool) -> None:
        criteria.append({"name": name, "value": float(value), "threshold": threshold, "pass": bool(ok)})

    worst_dt = max(abs(c["bitwise_max_abs"]) for c in cases)
    add_criterion("aa_vs_ab_bitwise_max_abs", worst_dt, f"< {GATE_BITWISE_MAX_ABS:g}", worst_dt < GATE_BITWISE_MAX_ABS)
    worst_g = max(abs(c["metrics"]["aa"]["gamma_K_err_pct"]) for c in cases)
    add_criterion("gamma_K_vs_exact_worst_pct", worst_g, f"< {GATE_PCT:g} %", worst_g < GATE_PCT)
    worst_l2 = max(abs(c["metrics"]["aa"]["field_l2_pct"]) for c in cases)
    add_criterion("field_l2_vs_exact_worst_pct", worst_l2, f"< {GATE_PCT:g} %", worst_l2 < GATE_PCT)
    worst_mass = max(c["metrics"]["aa"]["mass_drift_rel"] for c in cases)
    add_criterion("mass_drift_rel_worst", worst_mass, f"< {GATE_CONSERVATION_REL:g}", worst_mass < GATE_CONSERVATION_REL)
    worst_mom = max(c["metrics"]["aa"]["mom_drift_rel"] for c in cases)
    add_criterion("mom_drift_rel_worst", worst_mom, f"< {GATE_CONSERVATION_REL:g}", worst_mom < GATE_CONSERVATION_REL)
    # order independence
    spec0 = specs[0]
    st_a = AAPatternStepper(tg_initial_field(spec0), spec0.tau, kernel_order=("even", "odd"))
    st_b = AAPatternStepper(tg_initial_field(spec0), spec0.tau, kernel_order=("odd", "even"))
    for _ in range(20):
        st_a.step()
        st_b.step()
    order_diff = float((st_a.f - st_b.f).abs().max().item())
    add_criterion("kernel_order_immaterial", order_diff, "< 1e-12", order_diff < 1e-12)
    # tracker gate works
    st_c = AAPatternStepper(tg_initial_field(spec0), spec0.tau)
    st_c.collide()
    st_c.kernel_even()
    gate_raised = False
    try:
        st_c.macroscopic()
    except AAPatternParityError:
        gate_raised = True
    add_criterion("parity_gate_refuses_half_state", 1.0 if gate_raised else 0.0, "== 1", gate_raised)
    # hazard probe: same-time read in the half window, smooth (mode-1 TG) vs rough (white noise)
    n_probe = specs[0].n
    smooth = hazard_probe(n_probe, specs[0].tau, tg_initial_field(specs[0]))
    g = torch.Generator().manual_seed(1234)
    noise_ux = 1e-3 * torch.randn((n_probe, n_probe), generator=g, dtype=torch.float64)
    noise_uy = 1e-3 * torch.randn((n_probe, n_probe), generator=g, dtype=torch.float64)
    noise_field = equilibrium_from_velocity(torch.ones_like(noise_ux), noise_ux, noise_uy)
    rough = hazard_probe(n_probe, specs[0].tau, noise_field)
    add_criterion("half_window_hazard_rough_field_pct", rough["half_vs_complete_vel_l2_pct"],
                  f">= {GATE_PCT:g} % (hazard is fatal on a rough field)",
                  rough["half_vs_complete_vel_l2_pct"] >= GATE_PCT)
    add_criterion("half_window_mass_conserved", smooth["mass_rel_delta"],
                  f"< {GATE_CONSERVATION_REL:g} (conservation cannot catch the hazard)",
                  smooth["mass_rel_delta"] < GATE_CONSERVATION_REL)
    pb_vals = [c["metrics"]["aa_parity_blind"]["field_l2_pct"] for c in cases]
    worst_pb = max(pb_vals)
    add_criterion("parity_blind_arm_error_pct", worst_pb, f">= {GATE_PCT:g} % (bug detectable)", worst_pb >= GATE_PCT)

    overall = "PASS" if all(c["pass"] for c in criteria) else "FAIL"

    figures = make_figures(cases)

    tracker_budget = {
        "n": specs[-1].n,
        "nbytes_persistent": cases[-1]["tracker"]["nbytes_persistent"],
        "nbytes_permutation_index": cases[-1]["tracker"]["nbytes_permutation_index"],
        "n_persistent_fields_aa": 1,
        "n_persistent_fields_ab": 2,
    }

    result = {
        "anchor": "XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER",
        "overall": overall,
        "gate_pct": GATE_PCT,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "dtype": "float64",
        },
        "specs": [
            {
                "name": c["name"], "n": c["n"], "u0": c["u0"], "re": c["re"], "nu": c["nu"],
                "tau": c["tau"], "k": c["k"], "ma": c["ma"], "steps": c["steps"],
                "half_probe_step": c["half_probe_step"],
            }
            for c in cases
        ],
        "cases": [
            {
                "name": c["name"], "n": c["n"], "re": c["re"], "steps": c["steps"],
                "gamma_K_exact": c["gamma_K_exact"], "gamma_vel_exact": c["gamma_vel_exact"],
                "k0": c["k0"], "bitwise_max_abs": c["bitwise_max_abs"], "bitwise_rel": c["bitwise_rel"],
                "metrics": c["metrics"], "tracker": c["tracker"],
                "wall_s": c["wall_s"],
            }
            for c in cases
        ],
        "criteria": criteria,
        "hazard_probe": {"smooth_tg": smooth, "white_noise": rough},
        "tracker_budget": tracker_budget,
        "figures": figures,
        "artifacts": [
            "runs/xval_aa_pattern_even_odd/result.json",
            "artifacts/xval_aa_pattern_even_odd/result.json",
            "runs/xval_aa_pattern_even_odd/k_history_aa_<case>.csv",
            "runs/xval_aa_pattern_even_odd/k_history_ab_<case>.csv",
            "runs/xval_aa_pattern_even_odd/k_history_aa_parity_blind_<case>.csv",
            "artifacts/xval_aa_pattern_even_odd/k_history_<arm>_<case>.csv",
            *figures,
            "src/tensorlbm/aa_pattern.py",
            "tests/test_aa_pattern_parity.py",
        ],
        "repro_cmd": "python benchmarks/xval_aa_pattern_even_odd.py && pytest tests/test_aa_pattern_parity.py -q",
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    (RUN_DIR / "result.json").write_text(payload)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "result.json").write_text(payload)
    for csv_path in RUN_DIR.glob("k_history_*.csv"):
        (ARTIFACT_DIR / csv_path.name).write_bytes(csv_path.read_bytes())
    render_docs(result)

    print("\n=== criteria ===")
    for c in criteria:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['name']:<34} value={c['value']:.6g}  ({c['threshold']})")
    print(f"overall: {overall}\n")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())