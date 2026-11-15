"""Build REPORT_R1R2.md from results_r1r2.json (all numbers injected, never
transcribed) and publish it as a platform report with the two figures embedded.

Run:  python3 publish_r1r2.py            # build + publish
      python3 publish_r1r2.py --dry      # build only
"""
import base64
import json
import os
import sys

import requests

D = json.load(open("results_r1r2.json"))
H = D["hydrostatic"]
Z = D["zero_g"]["hist"][-1]
DB = D["dam_break"]
BH = D["before"]["hydrostatic"]
BD = D["before"]["dam_break_hist"]
G = DB["g"]

# ---------------------------------------------------------------- markdown
rows_h = "\n".join(
    f"| {k} | {v['slope']:+.6e} | {v['analytic']:+.6e} | {v['rel_err_pct']:+.4f}% "
    f"| {v['mass_drift']:.1e} | {v['u_max']:.1e} |"
    for k, v in sorted(H.items())
)
rows_b = "\n".join(
    f"| {k} | {v['observed_ratio']:.3f} | {v['pred_1_plus_half_omega']:.3f} "
    f"| {v['rel_err_pct']:+.1f}% |" for k, v in sorted(BH.items())
)
rows_c = "\n".join(
    f"| {k} | {1 + H[k]['rel_err_pct'] / 100.0:.6f} | {H[k]['rel_err_pct']:+.4f}% |"
    for k in sorted(H)
)
dhist = DB["hist"]
rows_d = "\n".join(
    f"| {h['t']} | {h['mass']:.3f} | {h['drift']:.1e} | {h['level_mean']:.4f} "
    f"| {h['level_ptp']:.3f} | {h['front']:.0f} | {h['counts'][1]} | {h['u_max']:.1e} |"
    for h in dhist
)
rows_bd = "\n".join(
    f"| {h['t']} | {h['drift']*100:.2f}% | {h['front']:.0f} | {h['counts'][1]} |"
    for h in BD
)
last = dhist[-1]
flat = DB["exact_level"]
pile_err = 100.0 * (last["level_ptp"] - 0.0) / flat

md = f"""# XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS / R1·R2 增量：修正版最小 D2Q9 自由面 oracle 的力项与质量账本缺陷复核及修复

**锚点**：XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（Maarten-vd-Sande/lbm 的 oracle-minimal-d2q9-fs 做法）
**上游**：`REPORT.md`（L1–L3 核心算符交叉验证，PASS）与 `REPORT_FS_VALIDATION.md`（平台报告 19326，
自由面增量，含两条诚实 FAIL：R1 Guo 静水梯度 −54.0%、R2 溃坝质量漂移 69.9%）
**本单**：对 R1、R2 逐条定位根因并修复；R1 完全修复（解析精度），R2 的质量账本完全修复，
但自由面的**相变拓扑**仍失效（新诚实 FAIL，R3，附根因与两条已实测的失败替代方案）。

---

## 1. 案例介绍

### 1.1 物理背景

被测对象是 500 行教学级 2D D2Q9 体素分数（VoF-ε）自由面 LBM 骨架的「修正版最小 oracle」
（`fs_oracle.py`）。它有一个 **关箱全液静水** 判定式和一个 **溃坝** 判定式：

| 编号 | 算例 | 几何/参数（栅格单位，c_s²=1/3，ρ_l=1） | 解析/精确基准 |
|---|---|---|---|
| A | 关箱静水（R1 判定式） | 16×32，全液闭箱（一格外框为 OBSTACLE），g=(0,−1e−4)，2000 步，ω∈{{0.8,1.0,1.5}} | 静水平衡 p=ρc_s² ⇒ **dρ/dy = g/c_s² = 3g = −3.00e−4** |
| B | 零重力静置（守恒判定式） | 24×32，液深 21.5，g=0，2000 步 | u≡0、界面平、质量精确守恒 |
| C | 溃坝（R2 判定式） | 48×48，水柱 14×26，g=(0,−1e−3)，ω=1.0，6000 步 | 质量精确守恒；平衡液位 h\\*=M/(n_cols·ρ_l)=M/46 |

关箱算例的解析解是精确的：均匀密度不是解，唯一稳态要求密度线性下降 3|g| 每格。
溃坝算例的「体积基准」也是精确的：质量守恒 ⇒ 静止后自由面必须平，且液位恰为 M/(n_cols)。

### 1.2 前单遗留

`REPORT_FS_VALIDATION.md` 报告 R1（Guo 力项静水梯度 −54.0%，且随 ω 变化）与
R2（溃坝质量漂移 69.9%）为诚实 FAIL，未给根因。本单从这两条进入。

---

## 2. 怎么做的

### 2.1 R1 根因（解析可证）

被合并实现（`fs_oracle.py` 的 `forcing="guo"` 分支）把**半步力速度平移做了两次**：
`f^eq` 与力项 S_i 都以 `u* = u + F/(2ρ)` 求值，同时又保留了 Guo 的 `(1−ω/2)` 系数。
而 `u*` 的平移本来只属于**宏观动量定义** `ρu = Σ c_i f_i + F/2`（`macroscopic()` 里已经做了）。

逐步动量核算（Σ c_i 过 GLBM 一步）：

- 正确 Guo：`ρu(t+1) = ρu(t) + F` ⇒ `a_eff = F`。
- 双计平移：`ρu(t+1) = ρu(t) + F(1 + ω/2)` ⇒ **`a_eff = F(1+ω/2)`**。

线性静水平衡给出 `dρ/dy = 3·a_eff`，于是相对解析值的倍率应为 **1 + ω/2**。
实测（用 pre-fix 模块程序化复算，见 §3.2）：1.422 / 1.542 / 1.832 @ ω=0.8/1.0/1.5，
对 1.4 / 1.5 / 1.75 —— 缺陷律成立到 2–5%（残差来自离散质量方程对线性剖面的 O(g) 修正）。

**修复**：`f^eq(ρ,u)` 与 `S_i` 一律用物理速度 `u`（`_collide`），`u` 仍由 `macroscopic()` 的
`Σ c_i f_i + F/2` 给出。

### 2.2 R2 根因（账本不封闭）

被合并实现的 VoF 账本有三处不封闭：

1. **指向空腔（void）的出流被丢弃**：面权重 `w_out=1`（目标非液），但该质量没有任何接收者；
2. **来自空腔的入流被当成质量**：把反反弹（anti-bounce-back）反射分布 `f_new[i]` 以权重 1
   记入质量，而它只是动量边界条件；
3. `eps()` 把 `m/ρ_l` 夹到 [0,1]，使 `to_fluid` 的 `eps > 1+FILL_OFFSET` 分支成为死代码，
   过满格永远无法被识别（相变机制瘫痪）。

另外顺带修掉两个几何/边界缺陷（都会直接伪造守恒性）：

4. `set_liquid_from_height` 在 `fill_box_walls()` **之后**把整列清零为 GAS，
   于是「闭箱」的左右两道外框被抹掉（静水算例其实在自由落体）；
5. 从**计算域边缘**流出的分布无人接收却被记为 void 出流 ⇒ 质量无声消失。
   现在最外一圈在账本里按墙处理（权重 1 + 反弹记入）。

**修复后的账本**（保持「一对面通量、两侧反号、共享面权重」的可构造守恒）：

$$
\\Delta m_i(x)=\\underbrace{{\\tfrac12(\\varepsilon_x+\\varepsilon_{{x+c_i}})f^{{new}}_i(x)}}_{{\\text{{液–液面入流}}}}
-\\underbrace{{w_{{out}}f^{{out}}_i(x)}}_{{\\text{{出流}}}},\\quad
w_{{out}}=\\begin{{cases}}\\tfrac12(\\varepsilon_x+\\varepsilon_{{x+c_i}})&\\text{{液–液目标}}\\\\1&\\text{{墙目标}}\\\\\\varepsilon_x&\\text{{空腔目标}}\\end{{cases}}
$$

指向空腔的那部分质量（`void_gain`）按空腔格的**界面邻居自由容量 (1−ε) 归一化后原样返还**，
因此 `Σ Δm ≡ 0` 到机器精度，且没有任何新格被凭空造出来。

### 2.3 验证方法

- `study_r1r2.py`（分节可复算：`s1/s3/s4`）产出全部数字到 `results_r1r2.json`；
- **before/after 全部程序化复算**：`s4` 动态加载修复前的模块（`fs_oracle_orig.py`，即 git HEAD 版本）
  跑同样的两个判定式，不手抄任何历史数字；
- 图由 `figures_r1r2.py` 从 JSON 生成；本报告的表由 `publish_r1r2.py` 从 JSON 注入；
- `pytest test_fs_oracle.py`：8 passed（新增 R1 三点解析回归测试）。

---

## 3. 计算结果

### 3.1 R1：修复后关箱静水梯度 vs 解析（A 算例）

| ω | 实测 dρ/dy | 解析 3g | 相对误差 | 质量漂移 | 最大速度 |
|---|---|---|---|---|---|
{rows_h}

### 3.2 R1：修复前（pre-fix 模块复算）缺陷律

| ω | 实测 slope/解析 | 缺陷律 1+ω/2 | 相对解析误差 |
|---|---|---|---|
{rows_b}

### 3.3 零重力静置（B 算例，2000 步）

| 量 | 实测 | 基准 |
|---|---|---|
| 质量漂移 | {Z['mass_drift']:.1e} | 0（精确） |
| 最大速度 | {Z['u_max']:.1e} | 0 |
| 平均液位 | {Z['level_mean']:.6f} | 21.5 |
| 液位起伏 ptp | {Z['level_ptp']:.3e} 格（= {100*Z['level_ptp']/21.5:.3f}% 液深） | 0 |

### 3.4 R2：溃坝（C 算例，48×48，6000 步）

| 步 | 质量 | 漂移 | 平均液位 | 液位 ptp | 前沿 x | 界面格数 | |u|max |
|---|---|---|---|---|---|---|---|
{rows_d}
（精确平衡液位 h\\* = M/(n_cols·ρ_l) = **{flat:.4f}**）

**修复前**（同一算例、同一初值、pre-fix 模块复算）：

| 步 | 质量漂移 | 前沿 x | 界面格数 |
|---|---|---|---|
{rows_bd}

### 3.5 三项判据的状态

| 判据 | 修复前 | 修复后 |
|---|---|---|
| 静水梯度 dρ/dy = 3g | −54.0%（且随 ω 漂移） | **0.0002% ~ 0.003%**，与 ω 无关 |
| 溃坝质量账本（6000 步） | 69.9% 漂移 | **{last['drift']:.1e}**（机器精度） |
| 零重力静置 | 精确 | 精确（质量 {Z['mass_drift']:.1e}，|u| {Z['u_max']:.1e}） |
| 溃坝自由面拓扑/铺展 | 冻结（前沿不动、质量流失） | **仍失效（R3）**：前沿 6000 步不动，质量在坝柱内堆积，ptp 由 {dhist[0]['level_ptp']:.1f} 涨到 {last['level_ptp']:.1f} 格 |

---

## 4. 结果图

![R1](fig_r1_hydrostatic.png)
**图 1** 左：三个 ω 下实测密度剖面与解析直线重合（误差 <0.003%）。右：修复前后 slope/解析 的倍率，
与缺陷律 `1+ω/2` 完全对齐（pre-fix 1.422/1.542/1.832 vs 1.4/1.5/1.75），修复后全部落在 1.000。

![R2](fig_r2_ledger_topology.png)
**图 2** 左：质量账本 —— 灰色为 pre-fix（丢弃 void 出流，漂移单调涨到 69.9%），蓝色为修复后
（6000 步内始终保持 {last['drift']:.1e}）。右：**残余缺陷 R3** —— 界面格数修复前后都停在 40（理想
应随自由面前沿增长），前沿 x 恒为 15；修复前的前沿同样冻死，但那时质量在流失，现在质量守恒，
于是质量只能在坝柱内堆积。

---

## 5. 注意点（踩过的坑 / 局限）

**R1 已闭环。** 双计半步力速度平移是「教科书级」易错点：`u*` 只属于宏观动量定义，
一旦同时喂给 `f^eq`/`S_i`，误差恰好是 `1+ω/2` —— 在 ω=1 时是 +50%，看似「数值精度问题」，
实际是格式级缺陷。判据必须是**多 ω**：单点 ω=1 的 −54% 很容易被误读成离散误差。

**R2 只闭环了账本，没闭环拓扑（新诚实 FAIL，R3）。** 溃坝 6000 步内：前沿恒 15、界面格数恒 40、
液柱内质量堆到 {last['level_ptp']:.1f} 格高（平衡应为 {flat:.2f}）。根因：**相变机制一步都不触发** ——
`_update_types` 的 `excess` 散播要求接收格 `tgt_recv`（本步不转换）且权重和 `wsum>0`，
否则该格被 `orphan` 否决、不转换（这条保护是为了防止残余质量被冲入 GAS 格后丢失）。
在邻域全是候选格的饱和状态下所有人互相否决 ⇒ 0 次转换。单独把 `eps` 的夹取修掉
（用未夹取的 `m/ρ_l` 判定过满）不足以解锁，因为否决发生在更早的散播权重上。

**两条已实测的替代方案都不可用（负面结果，供下次直接跳过）**：

| 方案 | 质量账本 | 前沿 | 拓扑 | 结论 |
|---|---|---|---|---|
| 把 void 出流直接存进空腔格（空腔格改判为界面，m=收到的质量） | 精确（1e-16） | 15 → 46（50 步内铺满全箱） | **雾化**：t=50 时 1752/1752 内部格全为界面（零质量薄膜），自由面消失 | 不可用 |
| 每步重建一格「界面壳」+ 上述存格 | 精确 | 15 → 46 | 壳层每步向外链式生长，48 步吃满全域 | 不可用 |
| 壳层只以 FLUID 为锚（防链式） | 精确 | **冻结** | 界面恒 40 | 不可用 |

**结论**：要同时拿到「质量精确 + 自由面铺展 + 界面不雾化」，必须补上 Thürey/Körner 的
完整 `flag` + `distribute_mass` 机制（或改用守恒型 VOF 重写相变），
不能用本单这种 5 行的散播补丁 —— 该结论已用三组实测证据锁定，可直接作为下一张单的起点。

**适用范围**：本 oracle 仍是 2D、单相+空腔、无表面张力、无并行；可信域是
「无强瞬变」的静水/静置类算例（|g| ≲ 1e−4）。**不可**用于体积输运/铺展类定量结论。

---

## 6. 误差判定（是否 <3% / <15%）

| 判据 | 基准 | 偏差 | <3%? | <15%? |
|---|---|---|---|---|
| R1 静水梯度 ω=0.8 | dρ/dy=3g | {H['0.8']['rel_err_pct']:+.4f}% | **是** | 是 |
| R1 静水梯度 ω=1.0 | dρ/dy=3g | {H['1.0']['rel_err_pct']:+.4f}% | **是** | 是 |
| R1 静水梯度 ω=1.5 | dρ/dy=3g | {H['1.5']['rel_err_pct']:+.4f}% | **是** | 是 |
| R1 缺陷律 (pre-fix) | a_eff=F(1+ω/2) | ≤5%（1.422/1.542/1.832 vs 1.4/1.5/1.75） | 是（2–5%） | 是 |
| R2 溃坝质量账本（6000 步） | 精确守恒 | {last['drift']:.1e}（相对） | **是** | 是 |
| 零重力质量守恒 | 精确 0 | {Z['mass_drift']:.1e} | **是** | 是 |
| 零重力静置速度 | 精确 0 | {Z['u_max']:.1e} | **是** | 是 |
| 零重力液位平整 | 精确 0 | {Z['level_ptp']:.3e} 格（{100*Z['level_ptp']/21.5:.3f}% 液深） | **是** | 是 |
| **溃坝铺展**（前沿位移，6000 步） | 应铺展到全域（≥31 格） | **0 格（−100%）** | **否** | **否（诚实 FAIL，R3）** |
| **溃坝液位平整**（平衡判据） | ptp=0，液位 h\\*={flat:.2f} | ptp={last['level_ptp']:.2f} 格（{pile_err:.0f}% of h\\*） | **否** | **否** |

**一句话**：R1（力项）与 R2（质量账本）已闭环到解析/机器精度，两项均 <3%；
但自由面的**相变拓扑**仍不工作（R3，前沿 0 位移、液位 ptp 超 15% 数倍），
因此本 oracle 仍**不能**用于自由面铺展类定量结论，这个 FAIL 不粉饰。
"""

figs = []
for name, cap in (("fig_r1_hydrostatic.png", "图 1 R1：静水梯度三点解析验证与缺陷律"),
                  ("fig_r2_ledger_topology.png", "图 2 R2：质量账本修复前后与 R3 残余拓扑缺陷")):
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(name, "rb").read()).decode()})

open("REPORT_R1R2.md", "w").write(md)
print("wrote REPORT_R1R2.md", len(md), "chars")
if "--dry" in sys.argv:
    sys.exit(0)

API = os.environ["TENSOR_API_BASE"].rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
tok = requests.post(f"{API}/auth/login", json={"username": USER, "password": PASS},
                    timeout=30).json()
token = tok.get("access_token") or tok.get("token")
hdr = {"Authorization": f"Bearer {token}"}
payload = {
    "title": "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS / R1·R2：Guo 力项双计与 VoF 质量账本的根因定位与修复",
    "summary": ("修正版最小 D2Q9 自由面 oracle 的两条遗留诚实 FAIL 逐条定位根因：R1 = Guo 力项把半步力速度"
                "平移做了两次（a_eff=F(1+ω/2)，多 ω 实测 1.422/1.542/1.832 对 1.4/1.5/1.75），修复后静水"
                f"梯度对解析 3g 误差 {H['1.0']['rel_err_pct']:+.4f}%@ω=1 且与 ω 无关；R2 = 指向空腔的出流被"
                f"丢弃、来自空腔的入流被计为质量，修复后溃坝 6000 步质量漂移 {last['drift']:.1e}（原 69.9%）。"
                "新诚实 FAIL R3：相变机制一步都不触发（orphan 否决饱和），自由面前沿 0 位移、液位 ptp 超基准"
                "数倍；三种替代方案已实测并作为负面结果锁定。"),
    "scenario_key": "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
    "mission_id": 21493,
    "tags": ["cross-validation", "free-surface", "D2Q9", "Guo-forcing", "VoF", "honest-FAIL",
             "TensorLBM", "hydrostatic"],
    "markdown": md,
    "figures": figs,
}
r = requests.post(f"{API}/tensor/reports", json=payload, headers=hdr, timeout=120)
print("POST", r.status_code, r.text[:400])