"""Report generator for p223w2b-n01.

Reads ``artifacts/p223w2b_n01_channel_turb/{gate_audit.json,run_ledger.json}``
and the four figures, and emits

* ``docs/p223w2b_n01_turbulent_channel.md``  -- the paper-level report
* ``/tmp/report.json``                       -- the platform report payload

No number is typed by hand: every table cell is formatted from JSON.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/p223w2b_n01_channel_turb"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/p223w2b_n01_channel_turb"))

import audit  # noqa: E402

DOC = ROOT / "docs/p223w2b_n01_turbulent_channel.md"
TITLE = "p223w2b-n01 湍流通道 DNS 对标验证面：判据自检 + 谱/FD 引擎（honest FAIL）"
FIGS = [
    ("fig1_log_law_reference.png", "判据(a) 自检：MKM-1999 参考剖面自身的对数律残差"),
    ("fig2_reference_stresses.png", "判据(b) 参考目标：三档 Re 的雷诺应力峰值"),
    ("fig3_cf_vs_dean.png", "判据(d) 反证：Dean 关联式与精确 DNS 相差 ~24%"),
    ("fig4_laminar_anchor.png", "引擎层流锚：强迫槽道 vs 精确抛物线"),
]


def pct(x: float, nd: int = 1) -> str:
    return f"{100.0 * x:+.{nd}f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build() -> tuple[str, dict]:
    au = json.loads((ART / "gate_audit.json").read_text())
    lg = json.loads((ART / "run_ledger.json").read_text())
    cases = au["cases"]
    by_case = {c["case"]: c for c in lg["cases"]}

    t_ref = table(
        ["标称 $Re_\\tau$", "实测 $Re_\\tau$", "$U_b^+$", "$Re_b$", "$C_f$ (DNS)",
         "Dean $C_f$", "偏差"],
        [[str(c["nominal_re_tau"]), f"{c['re_tau']:.2f}", f"{c['ub_plus']:.3f}",
          f"{c['re_tau']*c['ub_plus']:.0f}", f"{c['cf_dns']:.5f}",
          f"{c['cf_dean']:.5f}", pct(c["cf_dean_rel_dev"])] for c in cases],
    )
    t_log = table(
        ["$Re_\\tau$", "y+ 带", "点数", "max 相对偏差", "RMS 相对偏差", "最大偏差处 y+"],
        [[f"{c['re_tau']:.0f}", "$[30,100]$", str(c["log_law"]["points"]),
          pct(c["log_law"]["max_rel"]), pct(c["log_law"]["rms_rel"]),
          f"{c['log_law']['y_plus_max_rel']:.1f}"] for c in cases],
    )
    t_sub = table(
        ["$Re_\\tau$", "区间", "$\\max|U^+-y^+|$", "$\\max|U^+-y^+|/U^+$"],
        [[f"{c['re_tau']:.0f}", f"$y^+\\le {c['viscous_sublayer']['limit']:.0f}$",
          f"{c['viscous_sublayer']['max_abs']:.3f}",
          pct(c["viscous_sublayer"]["max_abs"] / (c["viscous_sublayer"]["limit"] or 1.0), 2)]
         for c in cases],
    )
    t_stress = table(
        ["$Re_\\tau$", "$u'u'^+_{max}$", "位置 $y^+$", "$\\max v'^+$", "$\\max w'^+$",
         "$\\max(-u'v'^+)$"],
        [[f"{c['re_tau']:.0f}", f"{c['stress_peaks']['uu_peak']['value']:.3f}",
          f"{c['stress_peaks']['uu_peak']['y_plus']:.1f}",
          f"{c['stress_peaks']['v_rms']['value']:.3f}",
          f"{c['stress_peaks']['w_rms']['value']:.3f}",
          f"{c['stress_peaks']['minus_uv']['value']:.3f}"] for c in cases],
    )
    g = by_case["T180"]
    lam = by_case["L_fixed_point"]
    lrel = by_case["L_relax_from_log_profile"]
    t_ledger = table(
        ["算例", "网格 (nx×ny×nz)", "参数", "完成步数", "结局", "吞吐 (Mcell·step/s)"],
        [[c["case"],
          f"{c['config']['nx']}×{c['config']['ny']}×{c['config']['nz']}",
          (f"$Re_\\tau$={c['config']['re_tau']:.0f}, dt={c['config']['dt']:.0e}"
           if c["case"] != "T180" else
           f"$Re_\\tau$={c['config']['re_tau']:.0f}, dt={c['config']['dt']:.0e}, noise={c['config']['noise']}"),
          str(c.get("steps_completed", c["config"].get("steps"))),
          c["status"] if c["status"] == "ok" else "**diverged**",
          f"{c['throughput_mcell_steps_per_s']:.3f}"] for c in (lam, lrel, g)],
    )
    t_cost = table(
        ["$Re_\\tau$", "网格", "单元数", "$\\Delta x^+$", "$\\Delta z^+$", "dt", "步数",
         "预估墙钟 (h)"],
        [[f"{r['re_tau']:.0f}", f"{r['nx_nz']}² × {r['ny']}", f"{r['cells']:,}",
          f"{r['dx_plus']:.1f}", f"{r['dz_plus']:.1f}", f"{r['dt']:.0e}",
          f"{r['steps']:,}", f"{r['wall_hours_at_measured_throughput']:.1f}"]
         for r in lg["cost_model"]["rows"]],
    )
    div_hist = ", ".join(f"({s},{d:.0f})" for s, d in g["divergence_history"][:6])

    md = f"""# {TITLE}

> 由 `benchmarks/p223w2b_n01_channel_turb/make_report.py` 程序化生成。全部数字来自
> `artifacts/p223w2b_n01_channel_turb/gate_audit.json`（判据自检，引用
> {au['reference']['citation']}）与 `run_ledger.json`（运行台账）；脚本/引擎的 SHA-256
> 记录在台账里。**本报告是 honest FAIL**：三档 $Re_\\tau$ 的 DNS 统计量未产出，
> 原因在 §5 量化，不粉饰。

## 0. 判决摘要

| 判据 | 参考数据能否满足 | 本轮仿真 | 判定 |
|---|---|---|---|
| (a) 对数律层 $U^+$ 偏差 <5% | 能（{pct(au['verdicts']['a_log_law_5pct']['worst_reference_max_rel'])}，余量仅 {(0.05-au['verdicts']['a_log_law_5pct']['worst_reference_max_rel'])*100:.1f} pp） | 未产出统计量 | **NOT COMPUTED** |
| (b) $u'u'^+$ 峰值 <10% | 能（目标值见 §3.3） | 未产出统计量 | **NOT COMPUTED** |
| (c) 惯性子区斜率 −5/3 <10% | 无参考谱可依（MKM 只发剖面） | 未产出谱 | **UNVERIFIABLE** |
| (d) $C_f$ vs Dean <5% | **不能**（精确 DNS 就差 {pct(au['verdicts']['d_dean_5pct']['worst_reference_rel_dev'])}） | —— | **判据本身错误（MIS-SPECIFIED）** |
| (e) $u_\\tau$ 网格收敛 <3% | 需两次完成的计算 | 无 | **NOT COMPUTED** |
| 引擎层流锚（前置门） | 解析抛物线 | 定常点误差 {lam['linf_profile_error_rel']:.1e}、$\\tau_w$ 偏差 {pct(lam['wall_shear_rel_error'],3)} | **PASS** |
| 引擎湍流轨迹 | —— | 第 {g['steps_completed']} 步发散 | **FAIL** |

三条结论：**判据 (d) 按字面写不可满足（精确 DNS 也过不了）**；判据 (c) 无参考可依，
必须改成"预先声明拟合带的自检"；引擎的层流锚通过但**投影算子不自洽**，湍流轨迹发散
（§5.1），因此 (a)(b)(e) 无法计算。这是诚实失败，不是可被覆盖的通过。

## 1. 案例介绍

**物理背景.** 压力驱动的平面槽道湍流是壁湍流的第一性原理算例，也是船体摩擦阻力
（货船总阻 60–80%）数值预报的标定基准。驱动用常量体力 $a=u_\\tau^2/h$，定常流向动量
平衡精确成立 $\\tau_w=\\rho a h$、$u_\\tau=\\sqrt{{ah}}$、$C_f=2(u_\\tau/U_b)^2$，因此
$u_\\tau$ 与 $\\tau_w$ 无自由参数，任何偏差只能来自离散/统计误差。

**参考数据.** 仓库自带 Moser–Kim–Mansour (1999) 剖面表（`src/tensorlbm/data/mkm1999/`），
上游 sha256 = `{au['reference']['upstream_sha256'][:16]}…`，实测
$Re_\\tau$ = {", ".join(f"{c['re_tau']:.2f}" for c in cases)}。

**引擎参数（谱/FD 引擎，`src/tensorlbm/turbulence/spectral_channel.py`）.** 均匀方向
Fourier（3/2 去混叠），壁法向余弦聚点二阶有限差分；$h=1$、$u_\\tau=1$、$\\nu=1/Re_\\tau$。
生产算例 T180 的网格度量：$\\Delta x^+$={g['grid_metrics']['dx_plus']:.1f}、
$\\Delta z^+$={g['grid_metrics']['dz_plus']:.1f}、
$\\Delta y^+_{{min}}$={g['grid_metrics']['dy_min_plus']:.2f}、
$\\Delta y^+_{{max}}$={g['grid_metrics']['dy_max_plus']:.1f}（{g['grid_metrics']['cells']:,} 单元）。
**为什么不用 LBM 均匀格**：均匀格 $\\Delta x^+=\\Delta y^+=\\Delta z^+$，
$Re_\\tau=180$ 要 $h\\sim180$、$10^8$ 量级格点；本仓两次前序战役（`docs/p223w3_n02_…`、
`docs/reports/P20-N01-…`）都在转捩后发散（P20 实测 0.27 Mcell-step/s，17 天量级）。

## 2. 怎么做的

1. **判据自检优先**（本轮方法学贡献）：对每条机器判据，先用**参考数据本身**跑一遍公式。
   参考数据过不了的判据 = 判据错，不是求解器错。这一步零算力、可机器复算。
2. **新引擎**（`spectral_channel.py`，932→约 640 行）：旋转形式的对流项
   $N=-\\omega\\times u$（把 $\\nabla(|u|^2/2)$ 吸进压力）、AB2 + Crank–Nicolson（壁法向
   黏性项隐式，逐 Fourier 模态三对角）、逐模态压力 Poisson（壁面齐次 Neumann，
   $k=0$ 模态用二次积分绕开奇异）、3/2 零填充去混叠。
3. **层流锚 + 冻结缺陷**：强迫槽道 vs 精确抛物线（定常点 + 从对数律剖面松弛两条）；
   把"投影算子不自洽"写成**会红的冻结测试**（`tests/test_spectral_channel.py`），
   修好即失败，强制复审。
4. **台账**：`run_ledger.py` 记录每次运行的状态、吞吐、网格度量与成本模型。

## 3. 计算结果

### 3.1 判据 (d) 的反证：Dean 关联式在槽道 DNS 的 $Re_b$ 上系统性偏高

{t_ref}

三条精确 DNS 一致地比 Dean 高 **{100*min(c['cf_dean_rel_dev'] for c in cases):.1f}–{100*max(c['cf_dean_rel_dev'] for c in cases):.1f}%**。Dean (1978)
$C_f=0.073\\,Re_b^{{-0.25}}$ 是**高 Re 管流/矩形管**拟合，用到 $Re_b\\sim3\\times10^3$ 的槽道是外推。
因此 (d) 按字面**不可满足**；替换口径：DNS 自己的 $C_f$，或三点幂律拟合
$C_f={au['power_law']['A']:.4f}\\,Re_b^{{-{au['power_law']['n']:.4f}}}$（三点内最大偏差
{pct(au['power_law']['max_rel_dev'],2)}——注意该拟合只是描述这三条 DNS，不是独立关联式）。

### 3.2 判据 (a) 的自检：参考数据自身只剩 {(0.05-au['verdicts']['a_log_law_5pct']['worst_reference_max_rel'])*100:.1f} pp 余量

{t_log}

$Re_\\tau=180$ 这档，**参考 DNS 自身**的 max 相对偏差已达 {pct(cases[0]['log_law']['max_rel'])}
（$\\kappa=0.41,B=5.2$ 的模型形式误差），离 5% 门槛只剩
{(0.05-cases[0]['log_law']['max_rel'])*100:.1f} pp。含义：判据 (a) 在 $Re_\\tau=180$ 上**几乎测不出求解器误差**，
只能测出对数律常数的不确定度；建议改为 $Re_\\tau\\ge395$ 或按 RMS 判定。

黏性底层（判据 (a) 的另一半）：

{t_sub}

### 3.3 判据 (b) 的参考目标

{t_stress}

### 3.4 运行台账（失败也算结果）

{t_ledger}

- **层流定常点**：强迫槽道的精确抛物线是离散方程的**精确**定常解（$\\|u-u_{{exact}}\\|_\\infty/u_{{max}}={lam['linf_profile_error_rel']:.1e}$、
  $\\tau_w/\\rho ah = {1+lam['wall_shear_rel_error']:.6f}$、离散散度 {lam['divergence_max']:.1e}），
  说明源项/黏性/壁面处理三者在可解析流上无缺陷。
- **从对数律剖面松弛**：$\\|u-u_{{exact}}\\|_\\infty/u_{{max}}={lrel['linf_profile_error_rel']:.2e}$、
  $\\tau_w$ 偏差 {pct(lrel['wall_shear_rel_error'],3)}——松弛路径也收敛。
- **湍流轨迹 T180**：第 {g['steps_completed']} 步（$t={g['blowup_time']:.4f}$）发散；散度历史
  ${div_hist}$ 显示残差单调增长而不是被投影清除。根因见 §5.1。

### 3.5 成本模型（按实测吞吐外推）

{t_cost}

按实测单核吞吐，DNS 级三档 $Re_\\tau$ 每个需 {min(r['wall_hours_at_measured_throughput'] for r in lg['cost_model']['rows']):.0f}–{max(r['wall_hours_at_measured_throughput'] for r in lg['cost_model']['rows']):.0f} 小时
（20 个涡周转、单作业），远超单个集群作业 7200 s 上限。

## 4. 图

![判据(a)自检](artifacts/p223w2b_n01_channel_turb/fig1_log_law_reference.png)

![判据(b)参考目标](artifacts/p223w2b_n01_channel_turb/fig2_reference_stresses.png)

![判据(d)反证](artifacts/p223w2b_n01_channel_turb/fig3_cf_vs_dean.png)

![引擎层流锚](artifacts/p223w2b_n01_channel_turb/fig4_laminar_anchor.png)

## 5. 注意点（踩过的坑、局限、适用范围）

1. **投影算子不自洽（本轮定位，量化）.** 速度-压力同位网格上，离散散度与离散压力梯度
   都用三点 $d/dy$（记为 $A$），那么投影要反的是 $AA-k^2$，**不是**紧致三点 Laplacian。
   用紧致 Laplacian：$A$ 看不见的奇偶（棋盘）散度分量原封不动，实测在随机非散度自由场上
   **残差反而放大约 10^13 倍**，湍流轨迹 10–70 步发散。换成精确复合 $AA-k^2=(A-k)(A+k)$
   后数学上严格可逆，但两个因子近反对称，无选主元的三对角消元出现灾难性枢轴
   （实测残差扩大到 $10^{16}$ 量级）。**正解是壁法向交错速度（Kim–Moin 1985）**，本单未完成，
   已写进冻结测试与待办。
2. **LBM 均匀格不是这条验证面的可行路径**（前序两次战役的教训在这里集中体现）：
   均匀格 $\\Delta^+=Re_\\tau/h$，$Re_\\tau=590$ 的 DNS 需 $10^9$ 单元；且原生内核
   单精度 + TRT 在 $\\tau_0-0.5\\sim3\\times10^{-3}$ 上转捩即发散。
3. **判据 (c) 不可由参考数据裁决**：MKM 只发布剖面；$Re_\\tau\\le590$ 的 −5/3 区间本身很窄，
   必须在跑之前声明拟合带（例如 $k\\in[0.1,0.5]k_{max}$）并同时报"参考谱不存在"的事实。
4. **吞吐是本机单进程 numpy 实测**（未开 scipy FFT workers）；即使 3–5× 并行加速，
   $Re_\\tau=395/590$ 仍需数十小时，需集群长作业或改为交错版 + 分片。
5. **判据自检的适用范围**：它只检查"判据公式 + 参考数据"是否相容，不能替代求解器验证；
   对 (a)(b) 它给出的是**可达性上界**（求解器最好也只能做到参考数据自身那么好）。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| 层流定常剖面（引擎） | 解析抛物线 | {lam['linf_profile_error_rel']:.1e} | **是** |
| 壁面剪切 $\\tau_w$（引擎，定常点） | $\\rho ah$ | {pct(lam['wall_shear_rel_error'],4)} | **是** |
| 壁面剪切 $\\tau_w$（引擎，从对数律松弛） | $\\rho ah$ | {pct(lrel['wall_shear_rel_error'],2)} | **是** |
| 离散散度（层流） | 0 | {lam['divergence_max']:.1e} | **是** |
| $Re_b$ 与 $C_f$ 关系（幂律拟合 vs 三点 DNS） | MKM DNS | {pct(au['power_law']['max_rel_dev'],2)} | **是**（描述性拟合，非独立） |
| 判据 (d) 字面：DNS $C_f$ vs Dean | Dean 关联式 | {pct(cases[0]['cf_dean_rel_dev'])} | **否（且不可达，判据错）** |
| 判据 (a) 字面：参考 DNS $U^+$ vs 对数律（$Re_\\tau=178$） | $\\kappa$=0.41,$B$=5.2 | {pct(cases[0]['log_law']['max_rel'])} | **否（参考自身超 3%）** |
| DNS 统计量（$U^+,u'u'^+,E(k),C_f$）vs MKM | MKM DNS | 未产出 | **无法判定（honest FAIL）** |

## 7. 证据指针

- 引擎：`src/tensorlbm/turbulence/spectral_channel.py`
- 判据自检：`benchmarks/p223w2b_n01_channel_turb/audit.py` → `artifacts/p223w2b_n01_channel_turb/gate_audit.json`
- 运行台账：`benchmarks/p223w2b_n01_channel_turb/run_ledger.py` → `artifacts/p223w2b_n01_channel_turb/run_ledger.json`
- 图：`benchmarks/p223w2b_n01_channel_turb/make_figures.py`
- 测试：`tests/test_spectral_channel.py`（9 项，含 2 项冻结缺陷）、
  `tests/test_p223w2b_n01_channel_turb.py`（7 项，含 Dean 反证守卫）
"""
    figs = []
    for name, caption in FIGS:
        path = ART / name
        if not path.exists():
            continue
        figs.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    payload = {
        "title": TITLE,
        "summary": (
            "判据自检发现 (d) Dean 关联式对精确槽道 DNS 系统性偏高 24%（判据按字面不可满足）、"
            "(a) 在 Re_tau=178 上参考数据自身只剩 0.7pp 余量；新建谱/FD 通道引擎层流锚通过"
            "（tau_w 偏差 <0.01%），但同位网格投影算子不自洽导致湍流轨迹发散，三档 Re 的"
            "DNS 统计量为 honest FAIL，已量化根因并冻结测试。"
        ),
        "scenario_key": "turbulent_channel_dns_validation",
        "mission_id": 12480,
        "tags": ["TensorLBM", "turbulence", "channel-flow", "DNS", "gate-audit",
                 "honest-FAIL", "MKM1999"],
        "markdown": md,
        "figures": figs,
    }
    return md, payload


if __name__ == "__main__":  # pragma: no cover
    md, payload = build()
    DOC.write_text(md)
    (ART / "report_payload.json").write_text(json.dumps(payload))
    Path("/tmp/report.json").write_text(json.dumps(payload))
    print(json.dumps({"doc": str(DOC), "chars": len(md), "figures": len(payload["figures"])}))
