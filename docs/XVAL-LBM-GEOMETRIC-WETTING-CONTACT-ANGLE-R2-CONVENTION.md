# XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE — R2：接触角约定缺陷（镜像）与解析帽验证

锚点 `XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`｜对标对象 `PorousMediaSimulation/openLBMPM`（`RKCG2D`）
| 判据 owner 精度标准 ≤15% | 域 TensorLBM（color-gradient 两相 + 几何润湿）
| 前序：`docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE.md`（R1，端到端 honest-FAIL）

**一句话结论（诚实）**：R1 的两条"通过"证据都是**自指**的（对参考核的字面转写等价、以及用旋转后梯度反读角——
量的正是核自己设定的量），因此看不到约定问题。本单用**独立几何真值**（平衡态液帽 = 球冠，其外法向闭式已知）验证，
发现 **移植实现把物理接触角当成了参考实现的内部角**，实际执行的是 `θ_realized = 180° − θ_phys`（镜像）；
`θ=90°` 是唯一不动点，所以单臂 90° 对标永远查不出来。已修（`contact_angle_convention="physical"` 为默认，
`"reference"` 保留以精确复现 R1），解析帽残差由 `|180−2θ|`（20.4°…121.6°）降到 **3.1°…4.6°（离散地板）**。
端到端液帽仍受"远壁相凝聚"伪影影响，只有 250 步短时（形状尚未偏离初值）时 60° 臂由 +25.68% 改善到 +11.98%，
此数字**不作为验收通过**，见 §五。

## 一、为什么 R1 会漏掉（方法论缺陷）

| R1 的检查 | 检查的是什么 | 能否发现约定错 |
|---|---|---|
| 660 例 `reference_rotation_scalar` vs 向量化核 | **保真度**（移植是否忠实参考） | ✗ 参考本身若如此，逐节点等价反而把错一起搬过来 |
| 解析界面 22 例反读角 0.000% | `acos(ĝ·n_s)`，即核**刚设定**的量 | ✗ 同义反复 |
| 求解器液帽 60/90/120 | 端到端外形 | 部分可以，但 R1 的估计器被壁面伪影污染（见 §三），且 90° 臂对称 |

正确做法是找一个**不被核设定的几何真值**：平衡态液帽的形状是球冠，其外法向在接触线处可由几何解析给出。

## 二、约定缺陷的推导（本单核心）

记壁面在 `y=y_wall`、固体在下（`n_s=(0,−1)`，由流体指向固体），液帽接触角 θ 在液滴内测。
左接触线处界面切向（向滴内抬起）`t=(cosθ, sinθ)`，滴内部位于 `t` 右侧，故外法向

```
ĝ = (−sinθ, cosθ)   ⇒   ĝ·n_s = −cosθ   ⇒   angle(ĝ, n_s) = 180° − θ      (★)
```

而移植核强制 `angle(ĝ, n_s) = θ_arg`（`cg_wetting_geometric.rotate_wall_color_gradient`，
写回 `g := −|g| n_chosen`，故 `ĝ_out = n_chosen`）。两者只在 θ=90° 重合 ⇒ **实际接触角为镜像 180°−θ**。
这也解释了参考实现为何"只支持 θ<90°、内部传 180°−θ"：它的 `theta` 参数是**另一相**的角度，R1 把这个内部角
当成了物理接触角直接暴露给宿主求解器。

`θ=0°/180°` 互换、`θ=90°` 不动，正好与 R1 的臂表一致（90° 臂 ±9.68% "看起来正常"）。

## 三、解析帽验证（独立真值，可机器复核）

构造精确球冠：圆心 `(x_c, y_wall − R cosθ)`、半径 R，`φ = tanh(d/ξ)`；它在壁上相交于 `x_c ± R sinθ`，
外法向 `n = (r − r_c)/|r − r_c|`。取壁邻两层、接触线 ±3 网格内的样本（`|φ|<0.9`、`|g|>1e-3`），
比较核输出 `ĝ_out` 与该点解析外法向的夹角。

| θ_phys | 样本 | `angle(ĝ_out,n_s)`（核精确设定值） | 物理要求 (★) | 残差 reference | 残差 physical | 镜像预测 \|180−2θ\| |
|---|---|---|---|---|---|---|
| 30° | 28 | 30.00° | 150° | 121.58° | **4.32°** | 120° |
| 60° | 22 | 60.00° | 120° | 62.53° | **3.15°** | 60° |
| 90° | 22 | 90.00° | 90° | 3.07° | **3.07°** | 0° |
| 120° | 24 | 120.00° | 60° | 57.38° | **3.13°** | 60° |
| 150° | 28 | 150.00° | 30° | 118.76° | **4.59°** | 120° |

两点结论：
1. `angle(ĝ_out,n_s)` 与 `θ_arg` **精确**相等（1e-6 内），证明核按设计执行（不是实现 bug，是**约定/语义** bug）；
2. 残差在 reference 下逐臂命中 `|180−2θ|`（±3° 内），在 physical 下落到 3.1°–4.6° 的**离散地板**（有限差分 + ξ=2 漫散界面）。
   90° 两列相同 ⇒ R1 的单 90° 臂在原理上无鉴别力。

实现与入口
* `src/tensorlbm/cg_wetting_convention.py`（新）：`physical_to_reference_angle`、`cap_phase_field`、
  `cap_outward_normal_field`、`cap_convention_error`（解析帽验证，返回 `physical_ok`）。
* `src/tensorlbm/color_gradient2d.py`：`color_gradient_two_tau_step` 新增 `contact_angle_convention`
  （默认 `"physical"`，内部对 `wall_contact_angle` 与逐面 map 都做 `180−θ` 映射；`"reference"` = R1 逐位复现）。
* `src/tensorlbm/cg_wetting_geometric.py`：`rotate_wall_color_gradient` 的 `theta_deg` 文档改为"参考内部角 = 180−物理角"。

## 四、测量缺陷（R1 honest-FAIL 的直接原因之一）

宿主求解器会在**远壁**凝聚出一小团润湿相（R1 §五已见"向上壁迁移相位形成伪团"，但未定量）。R1 的
全局圆拟合/面积反演把这团与液滴混在一起：顶点高度被判成上壁（`H=78.19`）、Kasa 拟合退化成直线（180°）。
本单新增 `benchmarks/xval_openlbmpm_wetting/wall_drop_measure.py`：按 8-连通取**贴壁连通域**为液滴，
其余正相区**显式登记**为 `blob_area`（量化而不隐藏），并给出两个独立估计（球冠恒等式 `θ=2 atan(2H/W)`、
液滴弧 Kasa 拟合）。

| 臂（80×96, R=20, 800 步） | 液滴面积 | 伪团面积 | 伪团质心 y | 说明 |
|---|---|---|---|---|
| θ=60 wetting | 181 | 70 | 75.8 | 液滴面积 ≈245（θ=60 帽） ⇒ 丢的正是伪团量 |
| θ=90 wetting | 546 | 84 | 76.4 | 628−546 ≈ 84 |
| θ=120 wetting | 920 | 84 | 76.5 | — |
| θ=90 无 BC 对照 | 604 | 18 | 77.7 | BC 关闭时伪团仍存在（更小） |

`φ` 体积与总质量**逐位守恒**（`mass=7680.000000`、`Σφ` 漂移 <1e-4），所以这不是质量泄漏，而是**相迁移**：
润湿相从液滴转移到远壁伪团。这一点是 R1 "外形全臂失效"的真正原因，也是本单端到端仍不能验收的原因。

## 五、端到端（诚实状态）

250 步短时矩阵（`runs/xval_wetting_sweep_local/`，80×96/R=20/β=0.7，上边界周期性）：

| θ | convention | θ_cap 尾均 | 误差 | ≤15% | 伪团 | H 终 | \|u\|max |
|---|---|---|---|---|---|---|---|
| 60° | reference | 75.41° | +25.68% | ✗ | 21 | 9.83 | 3.1e-2 |
| 60° | physical | 67.19° | **+11.98%** | ✓ | 44 | 9.75 | 1.2e-2 |
| 90° | 两者相同 | 98.18° | +9.09% | ✓ | 43 | 19.72 | 2.0e-2 |
| 120° | reference | 126.81° | +5.68% | ✓ | 49 | 29.70 | 3.9e-2 |
| 120° | physical | 132.52° | +10.44% | ✓ | 31 | 29.72 | 2.0e-2 |

* 60° 臂方向正确（reference 把液滴往 H(120°)=30 推、顶点高度单调上行；physical 保持 H≈9.75≈H_cap(60°)=10.0）。
  但**短时形状仍接近初值**，因此这些百分数只能说明"方向修正"，**不能作为 ≤15% 的验收证据**——验收证据是 §三。
* 120°/150° 臂对约定**不敏感**（`dθ/dθ_arg` 在浅/深帽两端退化：面积不变、H 由初值冻结），
  它们 R1 "通过"属于假通过。
* 未完成：长时（≥2e4 步）+ 高分辨率（ny≥128, R≥24）+ β=0.3 的收敛矩阵已在集群递交 `cj3251`
  （6 分片 × 4 配置），排队 16 min 未开始执行，本单时限内未取回工件 ⇒ 如实登记为**未运行**，不写成通过。

## 六、注意点 / 局限

1. **不可用 R1 的两条证据做验收**：等价性测试捕获不了参考自身的约定；反读角是同义反复。
2. 修的是**约定/语义**，不是核的数值实现——核仍与参考逐节点等价（R1 660 例证据仍成立）。
3. 端到端仍受远壁相凝聚伪影限制；候选机理（未证）：再着色 β 与漫散尾部导致的相成核/远壁润湿层，
   需要 β 敏感性 + `A`/`τ` 扫描来确认，`cj3251` 设计里已包含 β∈{0.7,0.3}。
4. 球冠恒等式估计器对**体积守恒**敏感：伪团偷相会让 W 收缩、把 θ_cap 抬高（同向偏差），
   所以本单把 `blob_area` 与 `volume` 一并登记，凡是"角度变好"都必须同时看伪团是否变大。
5. 适用范围：0<θ<180；`theta_by_direction`（逐面覆盖）同样走 `180−θ` 映射（有单测）。

## 七、工件与复现

| 工件 | 路径 |
|---|---|
| 约定模块（新） | `src/tensorlbm/cg_wetting_convention.py` |
| 求解器接入 | `src/tensorlbm/color_gradient2d.py`（`contact_angle_convention`，默认 physical） |
| 解析帽验证入口 | `benchmarks/xval_openlbmpm_wetting/analytic_cap_convention.py` |
| 贴壁连通域测量（新） | `benchmarks/xval_openlbmpm_wetting/wall_drop_measure.py` |
| 液帽探针 / 扫描 | `benchmarks/xval_openlbmpm_wetting/probe_wetting_direction.py`、`sweep_convention.py` |
| 单测（新） | `tests/test_wetting_contact_angle_convention.py`（16 项全绿） |
| 数据/图 | `runs/xval_wetting_convention/cap_convention.json`、`runs/xval_wetting_convention/figures/convention_evidence.png`、`runs/xval_wetting_sweep_local/sweep_shard0of1.json` |

**复现**

```bash
python benchmarks/xval_openlbmpm_wetting/analytic_cap_convention.py --out cap_convention.json
python benchmarks/xval_openlbmpm_wetting/sweep_convention.py --thetas 60,90,120 --conventions physical,reference \
       --steps 250 --ny 80 --nx 96 --radius 20 --top open --out runs/xval_wetting_sweep_local
PYTHONPATH=src python -m pytest tests/test_wetting_contact_angle_convention.py -q      # 16 passed
```

**未合并/下一步**：`cj3251`（长时高分辨率矩阵）取回后按 §三同一判据更新端到端表；远壁相凝聚的机理确认需
β∈{0.1,0.3,0.7} × A∈{0.02,0.04} × τ∈{1.0,2.0} 扫描，并按"角度 + 伪团面积 + 体积"三量联合判定。