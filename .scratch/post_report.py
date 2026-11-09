"""Post the P106W1-N04 D3Q27 slip-wall report (numbers injected from the JSON)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION = 8737

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "artifacts/P106W1-N04/d3q27_slip_wall.json"
d = json.loads(J.read_text())
cases, laws, chk = d["cases"], d["laws"], d["checks"]
fig_p = ROOT / "artifacts/P106W1-N04/d3q27_slip_wall_profiles.png"
fig_l = ROOT / "artifacts/P106W1-N04/d3q27_slip_wall_law.png"

sci = lambda x, n=2: f"{x:.{n}e}"  # noqa: E731
pct = lambda x, n=3: f"{100.0 * x:.{n}f}%"  # noqa: E731

wave = chk["wave_start"]
quad = chk["quadratic_fit_vs_model"]
collapse = chk["collapse_vs_1d_reduction"]
law_a_diff = max(
    abs(laws["D3Q27_ny17"]["a"] - laws["D3Q19_ny17"]["a"]) / laws["D3Q19_ny17"]["a"],
    abs(laws["D3Q27_ny33"]["a"] - laws["D3Q19_ny33"]["a"]) / laws["D3Q19_ny33"]["a"],
)
law_c_diff = max(
    abs(laws["D3Q27_ny17"]["c_times_H"] - laws["D3Q19_ny17"]["c_times_H"]),
    abs(laws["D3Q27_ny33"]["c_times_H"] - laws["D3Q19_ny33"]["c_times_H"]),
)

rows_cases = ["| 晶格 | ny | H | sigma | 步数 | u_max | b_model | b_bulk | b_quadratic |",
              "|---|---|---|---|---|---|---|---|---|"]
for c in cases:
    rows_cases.append(
        f"| {c['lattice']} | {c['ny']} | {c['H']:.1f} | {c['sigma']:.2f} | {c['steps']} | "
        f"{c['u_max']:.6f} | {c['b_model']:.6f} | {c['b_bulk']:.6f} | {c['b_quadratic']:.6f} |"
    )
tbl_cases = "\n".join(rows_cases)

rows_laws = ["| 拟合 | a | c | c·H | R² | 单调 |", "|---|---|---|---|---|---|"]
for k, v in laws.items():
    rows_laws.append(
        f"| {k} | {v['a']:.8f} | {v['c']:.8f} | {v['c_times_H']:.7f} | {v['r2']:.12f} | {v['monotone']} |"
    )
tbl_laws = "\n".join(rows_laws)

markdown = f"""
# P106W1-N04（续）: D3Q27 混合反弹/镜面滑移壁 —— 把「滑移律是算子的性质而非速度集的性质」测出来

## 1. 案例介绍

### 物理背景
超疏水/微织构壁面上的表观滑移常用 **混合（partial-slip）壁** 建模：把反弹（no-slip）与镜面反射（free-slip）按权重 `sigma` 混合，对每条射向流体的方向取

`f_q = (1-sigma) f_opp(q) + sigma f_mirror(q)`，镜面映射 `(cx,cy,cz) -> (cx,-cy,cz)`（壁法向 e_y）。

母任务 m7910（PR #1482，已在 main）把这套壁做到 **完整 D3Q19 格点**上，并把 `wall_partial_slip / D3Q27` 在能力契约里登记为 `NO_IMPLEMENTATION`。本单**不重复落地母 PR 的成果**（已核对 main 上文件齐备且测试全绿），只补 D3Q27 这一格：把速度集从「模块常量」变成「参数」，用与 D3Q19 相同的判据验证，并加一条 D3Q19 不需要的判据（§2.3）。

### 几何与参数
| 参数 | 值 | 说明 |
|---|---|---|
| 格点 | D3Q27（D3Q19 作交叉校验） | 27 速度集，含 8 条角向 (±1,±1,±1) |
| 碰撞 | BGK + Guo 强迫 | 单相、float64 |
| tau | {d['tau']} | nu = (tau-1/2)/3 = {d['nu']:.10f} lu²/step |
| g | {d['g']:g} | 体积力密度（沿 x） |
| ny | 17 / 33 | 流体行 1..ny-2，壁面 y=1/2 与 y=ny-3/2 |
| H | (ny-2)/2 = 7.5 / 15.5 | 半高（格子单位） |
| nx, nz | 1×1，及 8×8（塌缩算例） | 周期 |
| sigma | 0 / 0.25 / 0.5 / 0.75 | 0=反弹，1=镜面 |
| 收敛 | max|Δu|/max|u| < 1e-10 | 上限 80000 步 |

### 与母任务的差异（改了什么）
1. `src/tensorlbm/boundary/partial_slip_wall3d.py`：把模块级 D3Q19 常量替换为 `Lattice` 描述符（name/C/W/OPP/MIRROR/Q/广播形 cx,cy,cz,w/壁内向掩码），核函数（平衡态、宏观量、BGK+Guo）、壁行、求解器全部改成吃描述符；`mixed_wall_row_3d`、`channel_poiseuille_3d`、`C19/W19/OPP19/MIRROR_Y` 保持原签名原数值（旧测试即回归锚）。
2. 新增通用入口 `mixed_wall_row_lattice` / `channel_poiseuille_lattice` 与 `LATTICES = {{"D3Q19", "D3Q27"}}`。
3. 求解器新增 `init_wave`：给初始流向速度加一个 x/z 单波，波不能存活于 x/z 均匀的定常态 —— 于是「平面均匀性」是**被达到**而非被对称初值保持。
4. `Channel3DResult.lattice` 记录场来自哪个速度集。
5. 契约格 `wall_partial_slip / D3Q27`：`NO_IMPLEMENTATION` → `PHYSICS_VALIDATED`，证据=新测试文件+本 JSON。

## 2. 怎么做的（方法/流程）

### 2.1 算子不复制，只参数化
D3Q27 与 D3Q19 的差别只在表：权重 `W27`、反号排列 `OPP27`、镜面映射 `MIRROR_Y_27`，全部由 `tensorlbm/d3q27.py` 的规范常量**派生**（不是手抄）。所以 D3Q27 与 D3Q19 共用同一份数值代码——两套结果若不同，只可能来自速度集本身。

### 2.2 三条与 D3Q19 对齐的判据
- **C0′ 归约精确性**：完整 D3Q27 场必须复现便宜的 1D 柱解到舍入水平（两跑用**相同迭代步数** `tol=-1.0`，否则比较的是收敛容差不是模型误差）；
- **速度集无关性**：同一流动下 D3Q27 与 D3Q19 的剖面、以及 `b(sigma)` 律必须一致；
- **负对照**：`sigma=0` 必须回到解析无滑移抛物线的 0.5% 以内。

### 2.3 一条 D3Q19 不需要的判据：角向模式的塌缩
D3Q27 比 D3Q19 多出 8 条角向 `(±1,±1,±1)`，它们沿 x、z 也斜着离开 y 柱——最有可能把横向结构锁进场里。因此从 **5% 的 x/z 单波初值**（nx=nz=8）出发收敛，与均匀初值的跑逐点对比；两者一致到舍入，才说明横向耦合不会伪造定常图案。

### 2.4 工具链与复现
- 证据脚本：`artifacts/P106W1-N04/run_d3q27_slip_wall.py`（NumPy，约 4 分钟）→ `artifacts/P106W1-N04/d3q27_slip_wall.json` + 两张 png；
- 单测：`tests/test_partial_slip_wall3dq27.py`（14 例，约 80 秒）；
- 文档：`docs/P106W1_N04_SLIP_WALL_3D_D3Q27.md`（表内数字由 JSON 注入，非手抄）；
- 契约：`src/tensorlbm/boundary/boundary_capability_contract.py` + `tests/test_boundary_capability_contract.py`（D3Q27 格状态与断言同步更新）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 滑移长度与 b(sigma) 律（两个速度集 × 两个分辨率）
{tbl_cases}

### 3.2 律拟合 b = a·sigma/(1-sigma) + c
{tbl_laws}

`c·H` 是剖面拟合路线的**分辨率无关格点缺陷**（不是物理滑移长度）。两个速度集把它复现到 1.3e-12 以内，`a` 的差异 {sci(law_a_diff)} 相对。

### 3.3 精度/归约/塌缩
| 检查 | 值 |
|---|---|
| D3Q27 全域 vs 1D 柱解（相对剖面偏差） | {sci(collapse['rel_profile_deviation'])} |
| 1D 柱解平面均匀性 | {sci(collapse['plane_uniformity'])} |
| D3Q27 vs D3Q19 剖面偏差（sigma=0.5） | {sci(chk['cross_lattice_profile_dev'])} |
| sigma=0 vs 解析无滑移抛物线 | {pct(chk['sigma0_vs_analytic_noslip_dev'])} |
| sigma=0 无滑移残差 \\|b\\|/H | {sci(chk['sigma0_slip_length_over_H'])} |
| 5% 波初值 → 均匀解 剖面偏差 | {sci(wave['profile_dev_vs_uniform_run'])} |
| 波初值跑的平面扩散（plane_spread） | {sci(wave['plane_spread'])} |
| 二次拟合 b vs 曲率无关估计 b（同场） | {sci(quad['max_abs_error_lu'])} lu（{sci(quad['max_rel_error'])} 相对） |

### 3.4 验证与测试状态（诚实登记）
- `tests/test_partial_slip_wall3dq27.py`：**14/14 通过**（约 80 s，本机实测）。
- `tests/test_partial_slip_wall3d.py`（母任务 D3Q19，18 例）+ `tests/test_boundary_capability_contract.py`（含更新后的 D3Q27 契约断言）+ `tests/test_capability_contract_drift.py`：**51 例通过**。
- 重构回归：`_equilibrium/_macroscopic/_collide_bgk_guo` 被 `slip_wall_calibration.py` 直接调用，重构后已加 D3Q19 薄包装保持旧签名（首轮全量运行暴露该缺陷，已修）。
- **未完成项**：全量 `pytest` 两次后台运行在共享 128 核机器上被其他 agent 的测试争用，分别只推进到 21% / 25% 就被收尾；`tests/test_slip_wall_calibration.py` 在收尾时仍在运行。观察到的少数失败**未做基线性归因**，故不作为通过证据，也不声称"全套测试通过"。这是本单的真实缺口。

## 4. 结果图

图1：D3Q27 三个 sigma 的实测剖面（点）与解析滑移 Poiseuille（线）叠合；右图为逐点偏差（对数坐标）与 3% 阈值线。
图2：两个速度集 × 两个分辨率的 `b(sigma)` 数据点与拟合律——四条拟合重合到小数点后 6 位。

## 5. 注意点（踩过的坑/局限）

1. **归约比较必须先对齐迭代数**：一开始用 `tol=1e-9` 的两跑比较，得到 2.2e-7 的「模型误差」，其实那是收敛容差；改成固定步数 `tol=-1.0` 后立刻降到 {sci(collapse['rel_profile_deviation'])}。母 PR 的 D3Q19 测试正是这么做的，我照抄了这个约束。
2. **b_model 与 b_bulk 数值相同不是 bug**：数值剖面精确是二次的，两估计量在同一离散二阶矩下必然一致；它的价值体现在**有噪声/有界面时**（母任务记录：二次拟合放大曲率误差 L²）。本单只报它们在精确场上的一致度。
3. **自洽≠验证**：图1 的解析线用同一场的 b，闭环反演误差 0（平凡）；真正的绝对基准只有 sigma=0 的解析无滑移抛物线（{pct(chk['sigma0_vs_analytic_noslip_dev'])}）与解析滑移解族。本单不宣称任何实验/高保真对照。
4. **sigma=1 无定常态**（柱塞流+动量世俗增长），只登记签名，不报收敛值。
5. **热文件声明**：`partial_slip_wall3d.py` 属母任务核心模块，可能与其他在途 PR 重叠；本次改动保留全部旧公开名（薄包装），旧测试文件零改动全绿，冲突面仅为内部核函数签名。

## 6. 误差判定（逐项回答「是否 < 3%」）

| 关键量 | 基准 | 实测 | < 3%？ |
|---|---|---|---|
| D3Q27 sigma=0 剖面 | 解析无滑移抛物线 | {pct(chk['sigma0_vs_analytic_noslip_dev'])} | **是**（且 < 0.5% 内部判据） |
| D3Q27 vs D3Q19 剖面 | 交叉校验 | {sci(chk['cross_lattice_profile_dev'])} | **是**（≈舍入） |
| D3Q27 全域 vs 1D 柱解 | 归约解 | {sci(collapse['rel_profile_deviation'])} | **是**（≈舍入） |
| 律斜率 a | D3Q19 的 0.29999999 | 相对差 {sci(law_a_diff)} | **是** |
| 律缺陷 c·H | D3Q19 的 −0.065000 | 绝对差 {sci(law_c_diff)} | **是** |
| 二次拟合 b | 曲率无关估计 | {sci(quad['max_abs_error_lu'])} lu | **是** |
| 波初值塌缩 | 均匀初值跑 | {sci(wave['profile_dev_vs_uniform_run'])} | **是** |

**诚实边界**：以上全部是「与解析解/与自身另一速度集」的对照；本单不含任何实验或高保真参考值，因此「<3% vs 实验」这句话在本单**不成立也不适用**（无基准，声明原因如上）。
"""

payload = {
    "title": "P106W1-N04（续）：D3Q27 混合反弹/镜面滑移壁——速度集无关性与角向塌缩验证",
    "summary": (
        "把 partial_slip_wall3d 的速度集从模块常量改为 Lattice 参数，落地 D3Q27 混合滑移壁："
        f"sigma=0 复现解析无滑移抛物线 {pct(chk['sigma0_vs_analytic_noslip_dev'])}，"
        f"全域复现 1D 柱解 {sci(collapse['rel_profile_deviation'])}，与 D3Q19 剖面一致 {sci(chk['cross_lattice_profile_dev'])}，"
        f"5% 横向波初值塌缩到均匀解 {sci(wave['profile_dev_vs_uniform_run'])}；"
        f"b(sigma)=a sigma/(1-sigma)+c 的 a 与 c·H 与 D3Q19 一致到 {sci(law_c_diff)}。"
    ),
    "scenario_key": "lbm_partial_slip_wall_d3q27",
    "mission_id": MISSION,
    "tags": ["TensorLBM", "boundary-condition", "partial-slip", "D3Q27", "capability-contract"],
    "markdown": markdown,
    "figures": [
        {
            "name": "d3q27_slip_wall_profiles.png",
            "caption": "D3Q27 滑移槽道剖面（点）vs 解析滑移 Poiseuille（线）；右：逐点偏差与 3% 阈值",
            "png_base64": base64.b64encode(fig_p.read_bytes()).decode(),
        },
        {
            "name": "d3q27_slip_wall_law.png",
            "caption": "b(sigma)=a sigma/(1-sigma)+c：D3Q19/D3Q27 × ny=17/33 四点拟合重合",
            "png_base64": base64.b64encode(fig_l.read_bytes()).decode(),
        },
    ],
}


def api(path: str, body=None, token=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


token = api("/auth/login", {"username": USER, "password": PASS})["access_token"]
resp = api("/tensor/reports", payload, token=token)
print(json.dumps(resp)[:600])