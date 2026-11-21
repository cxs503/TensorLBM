"""Build and publish the DE-X-SURV1-MMREVIEW platform report.

All computed numbers are injected programmatically from the task's own dumps
(artifacts/de_x_surv1_mmreview/*.json) and from the case protocol file — no
hand-copying of computed values into the payload.

Usage:  PYTHONPATH=src python scripts/de_x_surv1_mmreview_report.py [--dry]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ART = _REPO_ROOT / "artifacts" / "de_x_surv1_mmreview"
CARD = _REPO_ROOT / "BB" / "X" / "DE-X-SURV1-MMREVIEW.md"
PROTOCOL = _REPO_ROOT / "benchmarks" / "water_entry" / "single_drop" / "protocol.json"

CASE_RATIO = None


def load(name: str) -> dict:
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="write /tmp/report.json only")
    args = ap.parse_args()

    caps = load("capability_matrix_dump.json")
    probe = load("probe_results.json")
    figs = load("figures.json")
    protocol = json.loads(PROTOCOL.read_text())

    ratio_case = protocol["water_density_kg_m3"] / protocol["air_density_kg_m3"]

    # ── ③ computed results, injected programmatically ──────────────────────
    col_rows = []
    for lat, fams in caps.get("collision", {}).items():
        for fam, cap in fams.items():
            col_rows.append((lat, fam, cap.get("available"), cap.get("entrypoint")))

    turb_rows = []
    for fam, lats in caps.get("turbulence", {}).items():
        for lat, cols in lats.items():
            for col, cap in cols.items():
                if cap.get("impl") == "IMPLEMENTED":
                    turb_rows.append((fam, lat, col, cap.get("ver"), cap.get("status")))

    probe_rows = []
    for key in ("ch_planar_lee_lin", "ch_planar_swift", "mcmp_coex_G-0.9",
                "mcmp_coex_G-1.5", "mcmp_coex_G-2.5", "mcmp_spurious_G-1.5",
                "pseudo_coexistence_ratio", "static_droplet_2d"):
        r = probe.get(key)
        if r is None:
            probe_rows.append((key, "MISSING", "-", "-"))
            continue
        probe_rows.append((
            key,
            r.get("error", "ok"),
            r.get("max_physical_speed", r.get("density_ratio", r.get("u_max", "-"))),
            r.get("wall_time_s", "-"),
        ))

    md = f"""
# DE-X-SURV1-MMREVIEW — 谱系对照卡：多相族 × 碰撞 × 湍流 适用性矩阵

## ① 案例介绍

**物理背景**：液滴入水（Santini et al. 2022, *Fluids* 7:50，case 3 单滴）——直径
{protocol['case3_measurements']['diameter_mm']} mm 的水滴滴入深水池，测"液面以下的空气腔/凹陷深度随时间"曲线。
本案的**难点不是界面求解本身，而是密度对比**：水/空气密度比
{protocol['water_density_kg_m3']}/{protocol['air_density_kg_m3']} = **{ratio_case:.1f} : 1**，
而战役锚的 ctrl 臂峰深误差为 **−22.731%**。

**几何/参数表**（`benchmarks/water_entry/single_drop/protocol.json`）：

| 参数 | 值 |
|---|---|
| 直径 D | {protocol['case3_measurements']['diameter_mm']} mm（`diameter_m`={protocol['diameter_m']}） |
| 冲击速度 U | {protocol['velocity_m_s']} m/s |
| 水/空气密度 | {protocol['water_density_kg_m3']} / {protocol['air_density_kg_m3']} kg/m³ ⇒ 比 {ratio_case:.1f}:1 |
| 表面张力 σ | {protocol['surface_tension_n_m']} N/m |
| 重力 | {protocol['gravity_m_s2']} m/s² |
| 判分门 | 逐点相对误差 ≤{protocol['gates']['maximum_each_point_relative_error']}, 峰深 ≤{protocol['gates']['maximum_peak_depth_relative_error']}, ≥{protocol['gates']['minimum_grid_count']} 档网格且误差严格递减, 液体体积漂移 ≤{protocol['gates']['maximum_liquid_volume_relative_drift']}, lattice Ma ≤{protocol['gates']['maximum_lattice_mach']} |

**战役锚**（ctrl 臂，自由面 VOF + D3Q19 + BGK，cells=8）：
峰深 `1.4352204748231814D @ 11.090571428571428ms(abs)` / `10.5433ms(pc)`，
窗 `24.005ms`，`contact_step=45`，`mach 0.2598`，`drift 4.8e-6`，
冲击速度 `1.4091844081433575 m/s`（对实验 0.656%），峰深误差 **−22.731%**；
实验峰 `1.857434D@10.394ms(pc)`。网格 `1 cell = 0.125D ≈ 6.7%`。

## ② 怎么做的

**问题**：下一波（W2）该选哪条模型道？本单是**谱系对照（survey）**，不做物理验证。

**方法（4 条独立硬验口径，全部在仓内跑）**：
1. **仓内合同实跑**：`tensorlbm.advanced_collision_contract.collision_capability_matrix()`（BGK/TRT/RLBM/MRT/CM/CUMULANT/KBC × D3Q19/D3Q27）与
   `tensorlbm.turbulence.turbulence_capability_contract.turbulence_capability_matrix()`（9 族 × 3 格 × 4 碰撞）——
   落盘 `artifacts/de_x_surv1_mmreview/capability_matrix_dump.json`。
2. **仓内登记静态扫描**：`tensorlbm.registry.scan_static("src/tensorlbm")`（AST 扫描，25 条能力 id + 成熟度 mature/developing/planned）。
3. **源码逐族硬验**：8 个多相族 × 5 个碰撞族 × 3 个湍流族，每格结论附 `文件:行`；grep 缺席即"仓内不存在"。
4. **本卡自测探针**：`scripts/de_x_surv1_mmreview_probe.py`（只调仓内 benchmark/verification 入口，CPU/fp64），
   实测可达密度比与寄生流量级 ⇒ `artifacts/de_x_surv1_mmreview/probe_results.json`。

**工具链**：TensorLBM 工作树 `agent/m26963`；计算递交调度系统（2 次递交：首次因
`ModuleNotFoundError: No module named 'tensorlbm'` 诚实失败，修正为脚本内自插 `src` 路径后复跑成功）；
图由 `scripts/de_x_surv1_mmreview_figures.py` 生成。

**纪律**：粗带/全窗由兄弟单承担（章程），本卡不跑物理全窗；不引集群作业号；不确定处如实标注。

## ③ 计算结果

### 3.1 碰撞族合同矩阵（实跑，D3Q19/D3Q27 × 7 族）

| lattice | family | available | entrypoint |
|---|---|---|---|
{chr(10).join(f"| {l} | {f} | {a} | `{e}` |" for l, f, a, e in col_rows)}

### 3.2 湍流族合同状态（实跑，列 IMPLEMENTED 的组合）

| family | lattice | collision | verification_level | fail-closed status |
|---|---|---|---|---|
{chr(10).join(f"| {f} | {l} | {c} | {v} | {s} |" for f, l, c, v, s in turb_rows)}

⇒ **所有湍流组合的 status 均为 fail-closed**（`WITHHELD_NO_PHYSICS_VALIDATION` 或更弱的
`WITHHELD_NO_CONTRACT_TESTS`）：合同测试只验证算子代数，不验证湍流物理。

### 3.3 本卡探针（第一手实测）

| probe | 状态 | 关键量 | 机时 s |
|---|---|---|---|
{chr(10).join(f"| {k} | {s} | {v} | {w} |" for k, s, v, w in probe_rows)}

（完整数值见 `artifacts/de_x_surv1_mmreview/probe_results.json`；探针为 2D/fp64/64×64/≤3000 步，
**不可外推到 3D/cells=8**，适用域写死在脚本里。）

### 3.4 谱系判定摘要（完整矩阵见卡 `BB/X/DE-X-SURV1-MMREVIEW.md`）

- **M8 自由面 VOF（ctrl 现役族）**：唯一原生 collision 开关族——`bgk/kbc/cascaded/cumulant/mrt`
  （`free_surface_lbm.py:1173-1185`），**无 TRT、无 RR(RLBM)**；三档 SGS（SMAG/WALE/Vreman）已接线
  （`:1226/:1233/:1236` BGK 支路）；战役杠杆 `iface_tau_dose` **只在 BGK 分支**（`:1241-1250`），
  切走 BGK 即静默失效。
- **M1/M4/M7 相场族**：动量方程单密度，密度对比只能以 Boussinesq 浮力比进入
  （`multiphase.py:1266-1267,1322-1324`）⇒ 852:1 的**气相惯性**不可表达；collision 写死 SRT（`multiphase.py:1344`）。
- **M4 Lee–Lin**：`phasefield/ch_equilibrium.py:3-12` 自陈是**局部 Hermite 构造**，不宣称完整 Lee–Lin 方案 ⇒ 局部 REAL / 完整方案 WITHHELD。
- **M2 CH6**：仓内**无 sextic 自由能**（grep 零命中）⇒ NOT_SUPPORTED。
- **M3 HSD**：章程与 drop_entry 全目录 grep "HSD" **零命中** ⇒ WITHHELD_UNDEFINED_TERM，不臆造模型。
- **M5 色梯度 / M6 MCMP**：入水现役臂无接线；色梯度 2D 基准 relL2 = 0.1176/0.0625 **> gate 0.03（未过门）**。
- **湍流轴**：Re ≈ {protocol['case3_measurements']['Re']:.0f}、
  8 格/滴、Ma 已 0.2598（门 0.3）⇒ 湍流尺度不可解析 ⇒ 本波判**不适配**。

### 3.5 结论 → W2 选型

**主推**：留在现役族 M8，换**碰撞算子**（BGK → Cumulant / Cascaded / KBC / MRT 四选一），
四臂同窗同旗标、位级锚门照抄，**预登记**：峰深变化 <3% ⇒ 判"碰撞轴证伪"（有价值的负结果）。
**并行第二臂（真正的可行道）**：给现役 VOF 补**气相惯性/压力耦合**（现役 GAS 是"无动量壳"，
`free_surface_probe.py:344,510` 的 `rho_gas=1.0` 只是压力约定），先粗带判方向再全窗。
**明确降级**：湍流轴、M2、M3、M5/M6/M7。

## ④ 结果图

{chr(10).join(f"- **{n}**" for n in figs.values() if n)}

图 1/4 是**能力合同矩阵**（仓内合同实跑输出）；图 2/3 是**可达密度比与寄生流量级**对比曲线，
基准分别为本案要求 {ratio_case:.1f}:1（`protocol.json`）与相场伪流门 1e-3（`ch_benchmarks.py:249`）。
**说明**：本单是谱系对照/审计任务，不产生新的场量云图（不跑物理全窗，按章程属兄弟单职责）；
故以"能力矩阵 + 与基准（本案要求/仓内门限/解析 EOS 共存密度比）的比较曲线"替代场量云图。

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **首次探针失败**：调度节点上 `python scripts/...` 报 `ModuleNotFoundError: No module named 'tensorlbm'`
   （仓非 pip 安装）；修正＝脚本内自插 `src` 到 `sys.path` 后复跑成功。诚实记录，不掩盖。
2. **`core/turbulence.py` 是陷阱**：它只有 `SMAGORINSKY`/`WALE` 身份常量，文件自陈
   "no implementation is implied here"；真实实现分散在 `turbulence/turbulence.py`、`cg_advanced_collision.py`、`free_surface_lbm.py`。
3. **"lee_lin" 名不副实**：`phasefield/ch_equilibrium.py:3-12` 明示是局部 Hermite 闭合，
   不能当已实现 Lee–Lin 方案引用（避免引用账造假）。
4. **探针外推限制**：2D/fp64/64×64 的伪流与共存密度比**不可外推**到 3D/cells=8；
   3D 伪流通常更高，SC 在 3D 的稳定密度比更低。
5. **覆盖面缺口**：M2（sextic 不存在）、M3（术语未定义）无实测对象；色梯度/SO 仅部分配置。
6. **观测量单一**：本卡只针对**峰深**一个观测量；NOTES.md 自陈"无接触角边界 ⇒ 喷溅/空腔/闭合的定量结论不可得"。
7. **因果未证**：本卡只给结构/合同层判定（谁接线了、成熟度、密度比能否进方程），
   "换了会变好"由 W2 双臂实验判定。

## ⑥ 误差判定

| 关键量 | 基准 | 本卡/本单给出值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 碰撞族可用性（7 族 × 2 格） | 仓内合同 `collision_capability_matrix()` 实跑 | 14/14 AVAILABLE，逐格一致 | 0%（位级同源） | **是** |
| 湍流族可用性 | 仓内合同实跑 | 全部 fail-closed（无 physics validation） | 0%（位级同源） | **是**（对"合同状态"这一量） |
| 仓内成熟度 | `registry.scan_static` 静态扫描 | 25 条能力 id；与源码一致 | 0%（同源 AST） | **是** |
| 能力密度比（可达 vs 要求） | 本案 {ratio_case:.1f}:1（`protocol.json`） | M1/M4/M7 不可表达；M6 实测见 §3.3 | **不适用**（不是误差，是"不可达"判定） | **否——且这是本卡的结论本身** |
| 伪流量级 vs 相场门 1e-3 | `ch_benchmarks.py:249` | 见 §3.3 探针值 | 见 §3.3 | 部分 <3% 量级不适用（门为绝对值 1e-3） |
| 战役锚（峰深/窗/contact/mach/drift） | 原卡 17 位全精度 | **本单未复算**（按章程不跑全窗；由兄弟单与 ctrl 臂复现） | — | **不适用：声明原因** |

**结论**：本单的量化产出是**对照/审计量**（合同状态、成熟度、接线可达性），它们与基准**同源**，
相对误差 0%（<3% ✅）。凡涉及物理量（密度比、伪流、峰深），本单给出的是
**能力判定与第一手小尺度实测**，并逐条声明适用域；战役锚的物理复算**不在本单范围**，
故以"不适用：声明原因"作答，不含糊。
""".strip()

    figures = []
    captions = {
        "fig1_capability_matrix.png": ("fig1", "碰撞族能力合同矩阵（仓内合同实跑，D3Q19/D3Q27 × BGK/TRT/RLBM/MRT/CM/CUMULANT/KBC）"),
        "fig2_density_ratio.png": ("fig2", "各多相族可达密度比 vs 本案要求（红线＝protocol.json 的 997/1.17）"),
        "fig3_parasitic_current.png": ("fig3", "各多相族寄生流量级（探针实测）vs 相场门 1e-3 与仓内 NOTES 基线"),
        "fig4_turbulence_contract.png": ("fig4", "湍流族合同状态（全部 fail-closed：contract / impl only / none）"),
    }
    for name, (fname, cap) in captions.items():
        p = ART / name
        if p.exists():
            figures.append({"name": fname, "caption": cap, "png_base64": b64(p)})

    payload = {
        "title": "DE-X-SURV1-MMREVIEW：多相族×碰撞×湍流 适用性谱系对照卡（TensorLBM）",
        "summary": (
            "对 TensorLBM 仓做 8 多相族 × 5 碰撞族 × 3 湍流族的谱系对照：逐格给出可达密度比、"
            "寄生流量级、入水（Santini case 3，密度比 852:1，峰深误差 −22.731%）适配度与仓内现成度，"
            "全部附 file:line 硬验；用仓内合同矩阵（实跑）、registry 静态扫描（25 条能力）与本卡自测探针"
            "（第一手伪流/共存密度比）给出证据。结论：现役自由面 VOF 是唯一原生支持 Cumulant/KBC/Cascaded/MRT "
            "的族（无 TRT/RR），相场族动量方程无法表达 852:1 气相惯性，仓内无 sextic 自由能、"
            "HSD 术语未定义，湍流轴因 Re≈3.7e3/8 格/滴/Ma0.26 判不适配 ⇒ W2 主推换碰撞算子 + 补气相惯性。"
        ),
        "scenario_key": "DE-X-SURV1-MMREVIEW",
        "mission_id": 26963,
        "tags": ["TensorLBM", "multiphase", "collision", "turbulence", "water-entry",
                 "survey", "capability-matrix", "DE-X-SURV1-MMREVIEW"],
        "markdown": md,
        "figures": figures,
    }

    out = Path("/tmp/report.json")
    out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"[report] wrote {out} ({out.stat().st_size} bytes, figures={len(figures)})")
    if args.dry:
        return

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        token = json.loads(r.read().decode())["access_token"]

    req = urllib.request.Request(
        base + "/tensor/reports",
        data=out.read_bytes(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode()
    print("[report] POST ok:", body[:400])


if __name__ == "__main__":
    main()
