#!/usr/bin/env python3
"""Build /tmp/report.json for mission 6319 (zig-zag verification)."""
import base64, io, json, pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/verified/maneuvering_zigzag"
res = json.loads((BENCH / "result.json").read_text())

figs = []
total = 0
for name, caption in (
    ("fig1_zigzag_histories.png", "图1 10/10 与 20/20 zig-zag 的首向/舵角历程：解析分段模态解（实线）与事件驱动 RK4（dt=0.05s，虚线）重合，标注换舵时刻与超调角"),
    ("fig2_overshoot_convergence.png", "图2 第一超调角误差 vs 时间步（双对数）：Euler 斜率 1（观测阶 0.998），RK4 斜率 4（观测阶 4.02）"),
    ("fig3_imo_gate.png", "图3 IMO MSC.137(76) 超调判据 vs 首摇阻尼降级系数：名义船 PASS，N′r×0.3 起 FAIL（R/L<1，非物理船，仅证明判据有鉴别力）"),
):
    im = Image.open(BENCH / "figures" / name).convert("RGB")
    im.thumbnail((880, 880))
    buf = io.BytesIO()
    im.convert("P", palette=Image.ADAPTIVE, colors=64).save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    total += len(b64)
    figs.append({"name": name, "caption": caption, "png_base64": b64})

r = res["routes"]
conv = res["convergence"]
inv = res["invariance"]
fam = res["imo"]["family"]
v = res["verdict"]

def n(x, k=6):
    return f"{x:.{k}f}"

rows = []
for key, block in r.items():
    for row in block["rows"]:
        rows.append(f"| {key} | {row['quantity']} | {n(row['sim'],9)} | {n(row['analytic'],9)} | {row['rel_err']:.2e} |")
route_tbl = "\n".join(rows)

conv_rows = []
for key, c in conv.items():
    conv_rows.append(
        f"| {key} | Euler {c['semi_implicit_euler']['dt']} | {c['semi_implicit_euler']['observed_order']:.3f} | {c['semi_implicit_euler']['rel_err'][-1]:.2e} |"
        f" RK4 {c['rk4']['dt']} | {c['rk4']['observed_order']:.3f} | {c['rk4']['rel_err'][-1]:.2e} |"
    )
conv_tbl = "\n".join(conv_rows)

fam_rows = "\n".join(
    f"| {f['N_r_scale']:.1f} | {n(f['first_overshoot_deg'],3)} | {n(f['second_overshoot_deg'],3)} | {n(f['R_over_L'],2)} | "
    f"{'是' if f['directionally_stable'] else '否'} | {'PASS' if f['imo_pass'] else 'FAIL'} |"
    for f in fam
)

markdown = f"""## ① 案例介绍

**物理背景**：回转试验只回答"船稳到哪儿"，zig-zag（超调）试验回答"反向操舵后什么时候才停得住"——
舵角 δ 下达后首向偏差达到 ψ_zz 即反向操舵，首向会继续越过 ψ_zz，越过量就是**超调角**；
10°/10° 与 20°/20° 是 IMO 操纵性衡准（Res. MSC.137(76) annex 2.3）的强制指标。
本任务补的是 TensorLBM `tensorlbm.maneuvering` 的**闭环操舵能力**：此前该模块只有开环固定舵角的
稳态回转（`simulate_turn`），全仓 grep `zigzag|overshoot` 零命中。

**几何/参数表**（KVLCC2 级大型油轮实尺度，与已验证的 `maneuvering_steady_turn` 同一套参数）

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| L_pp | {res['ship']['L_m']} m | Cb | {res['ship']['Cb']} |
| B | {res['ship']['B_m']} m | ρ | 1000 kg/m³ |
| T | {res['ship']['T_m']} m | U | {res['ship']['U_mps']} m/s |
| L/V | {res['ship']['L_over_V_s']:.2f} s | 导数集 | `NOMINAL_TANKER_PRIME`（代表性线性集，非 PMM 反演） |

L/V ≥ 10 s ⇒ 10/10 判据走 20° 分支（代码中显式分支）。模型：线性横荡–首摇
`M q̇ = A q + b(δ) + f_nl(v)`，`ψ̇ = r`，`R = U/r`，`K = r/δ`（Abkowitz 1964 / Nomoto 1957 / Fossen 2011）。

## ② 怎么做的

新增（`src/tensorlbm/maneuvering.py`，`__all__` 已导出）：

* `simulate_zigzag` —— **事件驱动**积分：换舵时刻（ψ 达 ±ψ_zz）与首向极值时刻（r 变号）都在**步内**
  用分数二分定位（1e-13），因此换舵时间不带 O(dt) 量化误差；ψ 作为第 3 个 RK4 状态用完整四段求积。
* `analytic_zigzag` —— **精确分段模态解**：每段舵角恒定 ⇒ LTI，q(t)、ψ(t) 为双指数闭式，换舵/极值
  时刻是闭式的根（扫描+二分到 1e-14 s）；模型带 cross-flow 项时直接 `ValueError` 拒绝，不静默丢项。
* `ZigZagResult` / `imo_msc137_zigzag_limits` / `check_imo_msc137_zigzag`（判据实现）。

**判据先于动手**：两路线相对差 <1e-4；Euler 观测阶 ∈[0.8,1.3]；RK4 观测阶 ≳3.5 且最细 dt 相对误差
<1e-5；线性尺度不变性 <1e-9；IMO 判据对名义船 PASS、对降级船 FAIL。
**工具链**：`benchmarks/verified/maneuvering_zigzag/run.py`（程序化写 result.json + 3 图）、
`tests/test_maneuvering_zigzag.py`（13 条 fast 测试，35 s）。

## ③ 计算结果（全部由 run.py 程序化写入 result.json 后注入，零手抄）

**两条独立路线对照**（参考 dt=0.05 s，RK4）

| 试验 | 量 | 数值路线 | 解析路线 | 相对差 |
|---|---|---|---|---|
{route_tbl}

**时间步收敛（第一超调角 vs 解析值）**

| 试验 | Euler dt 序列 [s] | 观测阶 | 最细相对误差 | RK4 dt 序列 [s] | 观测阶 | 最细相对误差 |
|---|---|---|---|---|---|---|
{conv_tbl}

**线性模型性质自检**：20/20 超调角 = 2×10/10 相对差 {inv['max_rel_err']:.1e}；左/右舵镜像相对差
{inv['mirror_overshoot_rel_err']:.1e}；带 cross-flow（Y_vv=|Y_v|/U, N_vv=|N_v|/U）时比值降到
{res['crossflow']['ratio_20_20_over_10_10_first']:.4f}（线性应为 2.0）——真实船正是靠这种非线性破缺。

**IMO MSC.137(76) 判据（20/20，首/二超调限 25°/40°）**

| N′_r × scale | 首超调 [°] | 二超调 [°] | R/L | 方向稳定 | IMO |
|---|---|---|---|---|---|
{fam_rows}

## ④ 结果图

共 3 张（图 1 场量历程曲线 + 两路线叠合；图 2 收敛阶；图 3 判据鉴别力），均由 run.py 中
matplotlib 程序化生成：{figs[0]['caption']}；{figs[1]['caption']}；{figs[2]['caption']}。

## ⑤ 注意点

1. **ψ 求积阶数会悄悄把 RK4 降成 3 阶**（本次实踩）：最初中点首摇率取自预测态（O(dt²)）、末段误用
   积分后的平均态而非 k3 段值 → 观测阶稳定在 3.02，而 q 本身仍是 4 阶。把 ψ 当第 3 个 RK4 状态、
   四段全部取**段值**后回到 4.02。诊断手法：单独 `_step_sway_yaw` 定舵角积分到已知 T，与
   `_segment_solution` 闭式比误差比值是否 = 2^p。
2. **换舵/极值必须步内定位**，否则换舵时间 O(dt) 量化使超调角收敛退化为 1 阶，两路线无法裁决到 1e-12。
3. **线性模型 10/10 与 20/20 尺度不变**（ψ(t)=δ·g(t)）：这两档不是两个独立证据，而是同一解的等比
   放大；真正的第二独立证据是 cross-flow 案例与 Euler/RK4 两种离散。
4. **绝对精度 WITHELD**：导数集是代表性线性集（非 PMM/CFD 反演），无舵速限制、无桨舵尾流、无横摇
   耦合、无自由面。文献 KVLCC2 的 20/20 首超调约 10–15°，本线性集给 {n(res['crossflow']['overshoot_10_10_deg'][0],2)}°
   (10/10) / {n(res['crossflow']['overshoot_20_20_deg'][0],2)}° (20/20，带 cross-flow)，差距来自导数集与非线性缺失。
5. **判据适用范围**：MSC.137(76) 只规定 10/10、20/20；其它组合返回 `applicable: False` 而非自造阈值；
   10/10 的限值依赖 L/V，未给 L/V 直接拒绝出结论。
6. **降级船族不是真船**：N′_r×0.5 已给 R/L=0.2、×0.4 起方向不稳定，仅用于证明判据有鉴别力。

## ⑥ 误差判定（逐项显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3% ? |
|---|---|---|---|
| 第一/第二超调角（10/10、20/20） | `analytic_zigzag` 闭式解 | {max(x['rel_err'] for b in r.values() for x in b['rows']):.1e} 相对 | **是** |
| 换舵时刻 / 首向极值时刻（4 个/试验） | 同上 | ≤4e-10 相对 | **是** |
| 线性尺度不变性 20/20 = 2×10/10 | 解析恒等式 | {inv['max_rel_err']:.1e} | **是** |
| 左/右舵镜像 | 对称性 | {inv['mirror_overshoot_rel_err']:.1e} | **是** |
| Euler 观测阶 | 理论 1 | {conv['20/20']['semi_implicit_euler']['observed_order']:.3f} | **是** |
| RK4 观测阶 | 理论 4 | {conv['20/20']['rk4']['observed_order']:.3f} | **是** |
| 绝对超调角 vs 实船 | 无可用基准（导数集非实船反演） | — | **无法判定，如实声明 WITHELD** |

总判定：`VERDICT: {'PASS' if v['pass'] else 'FAIL'}`（route {v['route_max_rel_err']:.1e}，
Euler p={v['euler_observed_order'][0]:.3f}，RK4 最细 {v['rk4_finest_rel_err'][0]:.1e}，
不变性 {v['invariance_max_rel_err']:.1e}，IMO 判据有鉴别力 {v['imo_gate_discriminates']}）。
"""

report = {
    "title": "操纵性闭环验证：TensorLBM zig-zag 超调角双路线裁决（1.2e-12）+ IMO MSC.137(76) 判据",
    "summary": (
        "为 tensorlbm.maneuvering 补上此前缺失的闭环操舵能力（simulate_zigzag 事件驱动积分 + "
        "analytic_zigzag 精确分段模态解 + ZigZagResult/IMO 判据），并用两条独立路线互相裁决："
        f"6 量×2 试验最大相对差 {v['route_max_rel_err']:.1e}，Euler 观测阶 {v['euler_observed_order'][0]:.3f}、"
        f"RK4 {v['rk4_finest_rel_err'][0]:.1e}@{conv['20/20']['rk4']['dt'][-1]}s（阶 {conv['20/20']['rk4']['observed_order']:.2f}），"
        "线性尺度不变性与左右舵镜像精确成立，cross-flow 案例破缺比值 1.92（线性应为 2.0）；"
        "IMO MSC.137(76) 判据对名义船 PASS、对降级船 FAIL（有鉴别力）。"
        "绝对精度 WITHELD（导数集为代表性线性集）已如实披露。"
    ),
    "scenario_key": "maneuvering_zigzag_overshoot_verification",
    "mission_id": 6319,
    "tags": ["maneuvering", "zigzag", "overshoot", "IMO MSC.137(76)", "verification", "TensorLBM", "closed-loop helm"],
    "markdown": markdown,
    "figures": figs,
}

out = pathlib.Path("/tmp/report.json")
out.write_text(json.dumps(report))
print("payload bytes:", out.stat().st_size, "base64 total:", total)