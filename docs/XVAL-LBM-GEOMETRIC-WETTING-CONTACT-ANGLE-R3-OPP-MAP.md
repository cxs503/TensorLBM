# XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE — R3：反弹映射缺陷（R1/R2 端到端证据失效）与密封流送算子

锚点 `XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`｜对标对象 `PorousMediaSimulation/openLBMPM`（`RKCG2D` 几何润湿）
| owner 精度标准 误差 ≤15% | 域 TensorLBM（color-gradient 两相 + 几何润湿）
| 前序：R1 `docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE.md`（honest-FAIL）、
R2 `docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE-R2-CONVENTION.md`（约定镜修复 + 解析帽验收）

**一句话结论（诚实）**：R2 §四 归因为「远壁相凝聚」的伪团，根因不是润湿边界条件、也不是相成核，
而是**反弹映射表写错**：R1/R2 的 sessile-drop 驱动把 D2Q9 的竖直两向 `(0,±1)` 映射到了**自己身上**
（`OPP=(0,3,2,1,4,7,8,5,6)`，正确应为 `solver.OPPOSITE_2D=[0,3,4,1,2,7,8,5,6]`），壁面因此**根本不是反弹壁**：
润湿相从液滴漏出、被周期边界送到远壁重新凝聚，液滴面积 246→177（−28%）、θ_cap 59°→100°。
修正后同一算例 θ_cap 误差 **+2.61%**、θ_circle **+0.79%**、伪团面积 **0**。
θ=60/90/120 三臂端到端全部 ≤2.7%，**首次通过 owner ≤15% 判据**（且均 <3%）。
附带交付：密封流送算子 `tensorlbm.stream_boxed`（非周期、半步反弹、质量严格守恒），消除闭合域两壁经周期缝互通的**结构性**耦合。

## 一、怎么发现的（方法论）

R2 的结论是「端到端受远壁伪团限制，无法验收」。本轮不改物理、只做**组件普查**：
在 t=400 的 φ 场上做 8-连通标号（`blob_mechanism.py::_blob_stats`，比 R2 的 `measure_wall_drop` 多记 x 质心与到壁行距）：

| 连通域 | 面积 | 质心 (x,y) | 行域 | φ_max | 身份 |
|---|---|---|---|---|---|
| 1 | 9 | (35.6, 1.7) | 1–3 | 0.74 | 接触线残留（左） |
| 2 | **177** | (48.0, 5.4) | 1–10 | 1.00 | 液滴本体 |
| 3 | 9 | (60.4, 1.7) | 1–3 | 0.76 | 接触线残留（右） |
| 4 | 37 | (48.3, 78.3) | 77–79 | 0.91 | 远壁团（与液滴同 x） |
| 5 | 7 | (35.3, 78.6) | 78–79 | 0.71 | 远壁团（与左残留同 x） |
| 6 | 6 | (60.5, 78.7) | 78–79 | 0.71 | 远壁团（与右残留同 x） |

两条判据把「相凝聚」假说否掉了：
1. 远壁三团与近壁三结构 **x 坐标一一对应**（48.3/35.3/60.5 vs 48.0/35.6/60.4）⇒ 是**近壁结构的投影**，不是新核化；
2. 全臂(θ=60/90/120)、含 BC 关闭臂都出现同类团（R2 表：70/84/84/18）⇒ 与润湿角无单调关系。

投影 + 漏相的组合只能来自**边界算子的连接关系**，于是查反弹映射表 —— 见 §二。

## 二、缺陷（可机器判定，无需物理判断）

`src/tensorlbm/cg_wetting_geometric.py::run_sessile_drop_contact_angle` 与 R1/R2 的三个基准脚本
（`sweep_convention.py`、`probe_wetting_direction.py`）都硬编码

```python
OPP = (0, 3, 2, 1, 4, 7, 8, 5, 6)        # 错
```

而仓内标准是 `solver.OPPOSITE_2D = [0, 3, 4, 1, 2, 7, 8, 5, 6]`（`C` 域序：1=(1,0), 2=(0,1), 3=(−1,0), 4=(0,−1)）。
对错表只有第 2、4 项不同：错表把 `(0,+1)` 和 `(0,−1)` 映到自己。
判据 `c[OPP[k]] == −c[k]` 在错表上对 k=2,4 失败 —— 这是**逻辑错误**，不是约定分歧：

* 竖直两向不反射 ⇒ 壁面法向动量不被翻转 ⇒ 该壁既非无滑移也非自由滑移；
* 对角四向照常镜像（5↔7, 6↔8）⇒ 壁面同时「翻转 y」又「不翻转 y」，自相矛盾；
* 竖直向由壁向流体的通量因此**不守恒**，相从液滴经壁漏出。

`tests/test_stream_boxed.py::test_legacy_map_is_not_a_bounce_back` 把这一条钉死为回归测试。

## 三、修正与端到端结果（80×96 / R=20 / β=0.7 / τ=1.0 / A=0.04 / ξ=2）

修法：驱动与脚本改用 `solver.OPPOSITE_2D`（一处变量，无物理改动）。
`theta_cap` = 球冠恒等式 `2 atan(2H/W)`，`theta_circle` = 液滴弧 Kasa 拟合；
`floor` = **t=0 的解析帽测量值**（同一估计器在精确球冠上的离散地板，非求解误差）。

| θ_cmd | 步数 | θ_cap 尾均 | 误差 | θ_circle 尾均 | 误差 | t=0 floor | 伪团面积 | ≤15% | ≤3% |
|---|---|---|---|---|---|---|---|---|---|
| 60° | 400 | 61.59° | **+2.65%** | 60.47° | **+0.79%** | 59.22° (−1.30%) | 0 | ✓ | ✓ |
| 90° | 300 | 91.11° | **+1.23%** | 90.44° | **+0.49%** | 91.03° (+1.14%) | 0 | ✓ | ✓ |
| 120° | 300 | 120.05° | **+0.04%** | 121.16° | **+0.97%** | 119.85° (−0.13%) | 0 | ✓ | ✓ |

同算例的**错表对照**（θ=60，t=400，同一初值/步数/参数）：

| 映射表 | θ_cap | 误差 | 液滴面积 | 伪团面积 | 远壁带 φ 质量 |
|---|---|---|---|---|---|
| 错表（R1/R2） | 100.23° | +67.05% | 177（−28%） | 68 | 24.36 |
| 正表（本单） | 61.57° | +2.61% | 244（−1%） | 0 | 0.00 |

结论：
1. R2 §四「远壁相凝聚」与 §五「端到端无法验收」**由错表造成**，不是润湿边界的物理局限；
2. R2 §五 的「120°/150° 臂对约定不敏感 ⇒ 假通过」判断也随之失效——修正后 120° 臂 θ_cap 与 floor 只差 0.2°，
   说明该臂此前"不动"是漏相把液滴冻结，而非估计器退化；
3. 修正后各臂测量误差与 t=0 floor 同量级（≤2.7%），**几何离散地板（ξ=2、水平集穿越规则）**才是剩余误差主项。

## 四、附带交付：`tensorlbm.stream_boxed`（密封流送）

即便映射表修正后 80×96/400 步不再出现伪团，`solver.stream` 仍是**双向全周期** gather：
贴缝的固壁行会从**对侧**取数，再被反弹反射回本地贴壁流体行 —— 闭合域两壁存在**结构性互通**。
本单新增：

* `stream_boxed(f, solid)`：流送一步 + 半步反弹（上游为固体或出域即 `f[OPP(k)]`），固体格置零；
  与已合并的 numpy `amr_composite2d.stream_walls` **逐位一致**（测试断言 max|Δ|=0）；
* 质量严格守恒（400 步漂移 <1e-9）；
* `seam_leak_fraction(f, solid)`：量化「经周期缝进入贴壁层的对侧通量占比」（本算例初值 0.57）；
* 语义变更如实登记：本算子用**当前**碰撞后分布做反射，`solver.stream`+固体反射的数据滞后一步；两者不可在同一算例混用。

`tests/test_stream_boxed.py` 8 项全绿（含端到端 sessile-drop 验收项，`-k not sessile` 为 7 项）：

```
PYTHONPATH=src python -m pytest tests/test_stream_boxed.py -q      # 8 passed
```

## 五、注意点 / 局限

1. **不覆盖 R1/R2 的解析帽证据**：R2 §三（`angle(ĝ_out,n_s)` 与 `θ_arg` 精确相等、residual 3.1°–4.6°）与约定修复
   （`contact_angle_convention="physical"`）**仍然成立且被本单复用**；本单否掉的是 R2 的**端到端归因**，不是约定修复。
2. 本轮端到端为 80×96/R=20/300–400 步的**局部分辨率**结果；128×160/R=24/4000 步的收敛矩阵已递交集群
   （`cj3328`/`cj3329`/`cj3330`，θ=60/90/120 × {periodic, boxed}），**取回后需按本表同一判据复核**；
   未取回前本表只声明「80×96 分辨率下通过」，不外推到网格无关。
3. τ=1.0 单点：τ 变化（0.7/2.0）与 β∈{0.3,0.7} 的敏感性仍未做，本单不声明其结论。
4. `stream_boxed` 只在两壁位于网格首/末行且**整行为固体**的闭合域上是"正确用法"；部分固体行需自行处理边界行。
5. 伪团判据：`blob_area` 为 8-连通非贴壁正相区总面积；本单新增 x 质心与到壁行距，R 之前的 `blob_centroid_y`
   在多团情形下是加权混合量，**不应**单独引用。
6. 换算到物理量：本单全为格子单位，没有对标实验数据；对标基准是**解析球冠几何**（`theta = 2 atan(2H/W)` 与
   t=0 精确帽的一致性），文献对标仍需三维/实验臂。

## 六、误差判定（逐项回答「是否 <3%」）

| 量 | 基准 | 偏差 | <3%？ |
|---|---|---|---|
| θ=60° θ_cap | 解析帽 60° | +2.65% | 是 |
| θ=60° θ_circle | 解析帽 60° | +0.79% | 是 |
| θ=90° θ_cap / θ_circle | 90° | +1.23% / +0.49% | 是 |
| θ=120° θ_cap / θ_circle | 120° | +0.04% / +0.97% | 是 |
| 估计器离散地板（t=0 精确帽） | θ_cmd | −1.30% / +1.14% / −0.13% | 是（同量级，说明剩余误差主要为几何离散） |
| `stream_boxed` vs `stream_walls` | numpy 已合并实现 | max\|Δ\|=0 | 是（逐位相等） |
| 流体质量守恒（400 步） | 精确守恒 | <1e-9 | 是 |

owner 精度标准 ≤15%：**端到端三臂全部通过**；<3%：三臂全部通过（网格无关性未证，见 §五.2）。

## 七、工件与复现

| 工件 | 路径 |
|---|---|
| 修复（驱动） | `src/tensorlbm/cg_wetting_geometric.py::run_sessile_drop_contact_angle`（改用 `solver.OPPOSITE_2D`） |
| 新算子 | `src/tensorlbm/stream_boxed.py`（`stream_boxed`、`seam_leak_fraction`） |
| 机制探测 | `benchmarks/xval_openlbmpm_wetting/blob_mechanism.py`（8 个算例臂：base/tall/wide/topwall/nowall/beta/boxed…） |
| 图 | `benchmarks/xval_openlbmpm_wetting/make_figure_r3.py` → `runs/xval_blob_mechanism_r3/figures/r3_bounceback_map_evidence.png` |
| 数据 | `runs/xval_blob_mechanism_r3/phi_{legacy,canonical}_opp_t400.npy`、`runs/xval_blob_mechanism_r3_local/*.json` |
| 单测 | `tests/test_stream_boxed.py`（8 passed） |

```bash
# 端到端（θ=60/90/120，正表 vs 错表对照）
PYTHONPATH=src python benchmarks/xval_openlbmpm_wetting/blob_mechanism.py \
    --cases base,boxed --theta 60 --steps 400 --sample-every 200 --out runs/xval_blob_mechanism_r3
PYTHONPATH=src python -m pytest tests/test_stream_boxed.py -q
```

**给下次自找活的线索**：`cj3328/3329/3330` 收敛矩阵取回后更新 §三（网格无关性）；
R2 的 β/τ 敏感性扫描仍未做；`cg_wetting2d.py`（Akai2018 解析式 ghost 分支）与几何旋转分支的
**同题双实现对标**至今没有在同一算例上做过（R1 只做了核级等价），可作为独立验证条目。