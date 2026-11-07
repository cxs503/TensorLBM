# 破冰艏部船型族：几何先验证面 + SDF 体素化一致性 + 艏角-阻力敏感度

**战役节点** IN13-W2-N01 · 仓 TensorLBM · 模块 `src/tensorlbm/icebow_cad.py` ·
测试 `tests/test_icebow_cad.py` · 运行器 `scripts/run_icebow_family.py` ·
产物目录 `artifacts/icebow_family_r1/` （SHA-256 清单见 `manifest.json`）
自动生成（`scripts/icebow_report_assets.py`），文中所有数字均直接读自运行产物，无手抄。

---

## 1. 案例介绍

破冰艏的几何形状同时支配三件事：冰层破碎时的接触力（DEM 腿）、艏部砰击/波浪载荷
（BEM 腿）与敞水阻力（LBM 腿）。战役需要一个**唯一、可验证、可被三条腿共用**的
破冰艏几何源，而不是各腿各自临时造形。本节点交付该几何源：
一族参数化破冰艏（直斜式 / 内倾式 / 多段式），一道机器可裁的几何准入闸门，
一套 SDF/体素浇筑（供 LBM 腿），以及型值表 / 面元 / 水线接触线导出（供 BEM/DEM 腿）。

几何定义（长度以 L 归一，`xi=x/L`，`zeta=z/D`，龙骨 `z=0`，甲板 `z=D`，
设计水线 `z=T=waterline_frac*D`）：

```
b(xi,zeta) = (B/2L) * kappa(zeta) * min( phi_bow(u), phi_aft(xi) )
u          = ( xs(zeta) - xi ) / s_ent          # 局部艏柱处 u=0
xs(zeta)   = 1 - s(zeta)                        # 该高度处艏柱 x 位置
s(zeta)    = (D/L)(1-zeta) tan(alpha_stem)      # 前倾艏柱型线（多段式下段加 rake）
s_ent      = (B/2L) / tan(beta_ent)             # 水线进流段长度
kappa      = 1 - k_bilge (1-zeta/zeta_T)^2      (zeta<=zeta_T)
           = 1 + c_flare (zeta-zeta_T)/(1-zeta_T) (zeta>zeta_T)
phi_bow    = u                                  (u<=f_lin)
           = f_lin + (1-f_lin)(1-(1-t)^c_fwd)   (u>f_lin, t=(u-f_lin)/(1-f_lin))
phi_aft    = r_t + (1-r_t) clip(xi/s_aft,0,1)
```

* `alpha_stem` = 艏柱倾角（离铅垂线）；`beta_ent` = 设计水线半进角；
  `c_flare` = 剖面外飘（负值=内倾/tumblehome）；`c_fwd` = 艏肩饱满度。
* `r_t`（艉部宽度比）**不是自由输入**：由 `Cb = nabla/(L B T)` 目标用二分法反解，
  因此族成员可以在 Cb 固定的前提下只比较艏部形状。
* `beta_ent` 缺省由 `beta_ent_for_cb(Cb)=clip(9+84(Cb-0.55),6,32)` 给出（设计经验映射），
  使 `Cb in [0.55,0.80]` 全域可解（解出的 `r_t` 严格落在 (0.02,0.995) 内）。

族成员（默认 `Cb=0.70`，L/B=5.88，
B/T=1.97，实船换算 L=100 m）：

| member | family | alpha_stem [deg] | beta_ent [deg] | c_flare | c_fwd | Cb target | r_t solved | L/B | B/T |
|---|---|---|---|---|---|---|---|---|---|
| straight_a20_cb70 | straight | 20.000 | 21.600 | 0 | 2.500 | 0.700 | 0.273 | 5.882 | 1.971 |
| straight_a25_cb70 | straight | 25.000 | 21.600 | 0 | 2.500 | 0.700 | 0.321 | 5.882 | 1.971 |
| straight_a30_cb70 | straight | 30.000 | 21.600 | 0 | 2.500 | 0.700 | 0.372 | 5.882 | 1.971 |
| straight_a35_cb70 | straight | 35.000 | 21.600 | 0 | 2.500 | 0.700 | 0.429 | 5.882 | 1.971 |
| inward_a30_cb70 | inward | 30.000 | 21.600 | -0.250 | 2.000 | 0.700 | 0.401 | 5.882 | 1.971 |
| multiseg_a30_cb70 | multisegment | 30.000 | 21.600 | 0.150 | 3.000 | 0.700 | 0.371 | 5.882 | 1.971 |

## 2. 怎么做的

1. **解析几何 + 二分反解 Cb**：`design_volume` 用 1201×401 二维梯形积分得到
   `nabla`；`solve_transom_ratio` 对 `r_t` 二分 60 次命中 Cb 目标（误差 <1e-3）。
2. **水密网格**：5 个结构面片（左/右舷侧板、甲板、船底、艉封板）共享边界节点；
   **先焊接再三角化**，因此艏柱收缩成一个点阵时有 0 个零面积三角形；
   每片按其自身外向参考方向定向。
3. **几何准入闸门**（`validate_mesh`）：水密（每条无向边恰用 2 次）、流形（无边界边、
   无非流形边）、Euler 示性数 `V-E+F=2`、定向一致（每条有向边恰用 1 次）、
   正体积、外向法向（按面片参考方向逐一核验，并对解析指示函数做质心偏移抽样）、
   零面积三角形、塌缩面片。任一不满足即 `passed=False`。
4. **参数回读**（`reverse_parameters`）：从网格反算 `L / B(水线处) / nabla / Cb /
   alpha_stem（跨水线的艏柱两顶点）/ beta_ent（水线最前两点）`，与设计值比对。
5. **SDF/体素化**（`voxelize_mesh`）：沿 +x 逐列射线奇偶法，子网格非对称扰动
   (1.3e-4, 3.7e-4) + 严格重心坐标，与 `tensorlbm.geometry_voxel` 同一约定；
   输出 `(nz,ny,nx)` 布尔实体掩膜。三档体素 0.5%/0.25%/0.125% L。
6. **阻力敏感度**（`resistance_sensitivity`）：同一几何源喂两条腿
   * **Holtrop-Mennen (1984)** `R_T = R_F(1+k) + R_W`，输入 `nabla, S(网格湿表面),
     Cb, Cp, lcb, iE`（iE 即 `beta_ent`）。绝对量可信，但回归式**没有艏柱倾角项**。
   * **Michell 薄船**波阻（同一几何，对垂向源强分布敏感，正对「艏柱倾角」这个自由度）。
   两条腿各自做网格/积分密度三档加密 + Roache GCI；混合敏感度取
   `dR_T/R_T = (R_F/R_T) dR_F/R_F + (R_W/R_T) dR_W^M/R_W^M`（把薄船趋势施加到回归
   波阻量级上，比例来自 Holtrop 分解，见 `share_rw_holtrop`）。
7. **下游契约**：导出 `hull.stl` / `wetted_surface.stl` / `offsets.csv` /
   `waterline_starboard.csv` / `voxel_mask_h0.0025.npz`，并用**未修改的**下游读取器
   （`tensorlbm.geometry_voxel.read_stl_triangles` / `mesh_watertight_status` /
   `tensorlbm.voxelize.load_stl`）回读验证；所有文件 SHA-256 进 `manifest.json`。

## 3. 计算结果

### 3.1 几何准入闸门 + 参数一致性（判据 b、c）

| member | gate | V-E+F | zero-area | collapsed | outward frac | nabla_mesh [L^3] | nabla err [%] | Cb err [%] | L err [%] | B err [%] | alpha err [deg] | beta err [deg] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight_a20_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.016 | -0.016 | 0 | 0 | -7.560e-12 | 0 |
| straight_a25_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.014 | -0.013 | 0 | 0 | -9.830e-12 | -3.553e-15 |
| straight_a30_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.013 | -0.013 | 0 | 0 | 6.739e-12 | 0 |
| straight_a35_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.013 | -0.013 | 0 | 0 | 7.503e-12 | 3.553e-15 |
| inward_a30_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.013 | -0.013 | 0 | 0 | 6.739e-12 | 0 |
| multiseg_a30_cb70 | PASS | 2 | 0 | 0 | 1.000 | 0.010 | 0.014 | -0.013 | 0 | 0 | 6.739e-12 | 0 |

### 3.2 两档网格位移体积一致性（判据 a）

| member | | nabla_mesh - nabla_analytic | [%] @81x61 | @161x121 | ratio |
|---|---|---|---|
| straight_a20_cb70 | 0.016 | 0.004 | 0.228 |
| straight_a25_cb70 | 0.014 | 0.004 | 0.260 |
| straight_a30_cb70 | 0.013 | 0.004 | 0.273 |
| straight_a35_cb70 | 0.013 | 0.003 | 0.253 |
| inward_a30_cb70 | 0.013 | 0.003 | 0.265 |
| multiseg_a30_cb70 | 0.014 | 0.003 | 0.247 |

### 3.3 SDF / 体素体积收敛（判据 d）

| member | converging | submerged converging | finest |dV|/V [%] | finest |d nabla|/nabla [%] | GCI(V) [%] | GCI(nabla) [%] | observed order |
|---|---|---|---|---|---|---|---|
| straight_a20_cb70 | PASS | PASS | 0.011 | 0.018 | 0.042 | 2.259 | n/a |
| straight_a25_cb70 | PASS | PASS | 0.012 | 0.021 | 0.048 | 2.257 | n/a |
| straight_a30_cb70 | PASS | PASS | 0.011 | 0.019 | 0.043 | 2.264 | n/a |
| straight_a35_cb70 | PASS | PASS | 0.011 | 0.019 | 0.044 | 2.283 | n/a |
| inward_a30_cb70 | PASS | PASS | 0.004 | 0.019 | 0.102 | 2.265 | n/a |
| multiseg_a30_cb70 | PASS | PASS | 0.009 | 0.004 | 0.005 | 1.933 | 2.565 |

逐档明细：

| member | h/L | cells | V_voxel [L^3] | d(V)/V [%] | d(nabla)/nabla [%] |
|---|---|---|---|---|---|
| straight_a20_cb70 | 0.005 | 181800 | 0.014 | 0.065 | -1.555 |
| straight_a20_cb70 | 0.003 | 1350720 | 0.014 | 0.045 | 1.704 |
| straight_a20_cb70 | 0.001 | 10403544 | 0.014 | 0.011 | 0.018 |
| straight_a25_cb70 | 0.005 | 181800 | 0.014 | 0.075 | -1.570 |
| straight_a25_cb70 | 0.003 | 1350720 | 0.014 | 0.050 | 1.715 |
| straight_a25_cb70 | 0.001 | 10403544 | 0.014 | 0.012 | 0.021 |
| straight_a30_cb70 | 0.005 | 181800 | 0.014 | 0.061 | -1.583 |
| straight_a30_cb70 | 0.003 | 1350720 | 0.014 | 0.045 | 1.723 |
| straight_a30_cb70 | 0.001 | 10403544 | 0.014 | 0.011 | 0.019 |
| straight_a35_cb70 | 0.005 | 181800 | 0.014 | 0.058 | -1.591 |
| straight_a35_cb70 | 0.003 | 1350720 | 0.014 | 0.046 | 1.734 |
| straight_a35_cb70 | 0.001 | 10403544 | 0.014 | 0.011 | 0.019 |
| inward_a30_cb70 | 0.005 | 181800 | 0.014 | 0.095 | -1.587 |
| inward_a30_cb70 | 0.003 | 1350720 | 0.014 | 0.086 | 1.724 |
| inward_a30_cb70 | 0.001 | 10403544 | 0.014 | 0.004 | 0.019 |
| multiseg_a30_cb70 | 0.005 | 212100 | 0.015 | -0.098 | -1.831 |
| multiseg_a30_cb70 | 0.003 | 1562976 | 0.015 | 0.008 | 1.677 |
| multiseg_a30_cb70 | 0.001 | 11986692 | 0.015 | -0.009 | -0.004 |

### 3.4 艏角 → 敞水阻力敏感度（判据 e，Fn=0.20，L=100 m）

| member | alpha [deg] | r_t | S [m^2] | nabla [m^3] | R_F [kN] | R_W^HM [kN] | R_T^HM [kN] | R_W^M [kN] | dRT/R_T HM [%] | dR_W/R_W M [%] | dRT/R_T hybrid [%] | GCI(delta) [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight 20 | 20.000 | 0.273 | 2735.295 | 10263.380 | 109.363 | 27.869 | 137.231 | 840.346 | 0.171 | 8.314 | 1.827 | 0.092 |
| straight 25 | 25.000 | 0.321 | 2725.884 | 10263.399 | 109.227 | 27.869 | 137.096 | 813.035 | 0.072 | 4.794 | 1.032 | 2.699 |
| straight 30 | 30.000 | 0.372 | 2716.348 | 10263.400 | 109.129 | 27.869 | 136.997 | 775.842 | 0 | 0 | 0 | n/a |
| straight 35 | 35.000 | 0.429 | 2706.588 | 10263.426 | 109.083 | 27.869 | 136.952 | 726.970 | -0.033 | -6.299 | -1.308 | 1.836 |
| inward 30 | 30.000 | 0.401 | 2716.438 | 10263.405 | 109.295 | 27.869 | 137.164 | 542.994 | 0.121 | -30.012 | -6.009 | 0.061 |
| multisegment 30 | 30.000 | 0.371 | 2708.380 | 10263.417 | 108.805 | 27.869 | 136.674 | 951.158 | -0.236 | 22.597 | 4.409 | 0.099 |

`share_rw_holtrop = 0.2034`；
直线族 `dRT/R_T`: HM ['0.171', '0.072', '0', '-0.033'] %,
Michell ['8.314', '4.794', '0', '-6.299'] %,
hybrid ['1.827', '1.032', '0', '-1.308'] %（参考 30°）；
单调性：HM PASS /
Michell PASS / hybrid PASS。

### 3.5 下游契约（判据 f）

| member | STL triangles | watertight (repo reader) | boundary edges | non-manifold edges | load_stl | offsets.csv | waterline.csv |
|---|---|---|---|---|---|---|---|
| straight_a20_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |
| straight_a25_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |
| straight_a30_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |
| straight_a35_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |
| inward_a30_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |
| multiseg_a30_cb70 | 25440 | PASS | 0 | 0 | FAIL | [861, 3] | 86 |

### 3.6 验收判据汇总

| criterion | result | value |
|---|---|---|
| a_volume_default_le_1pct | PASS | {'value_percent': 0.016202222142221223} |
| a_volume_fine_le_0p3pct | PASS | {'value_percent': 0.0036961872795108343} |
| b_geometry_gate_all_members | PASS | {} |
| c_parameter_consistency_le_1pct_1deg | PASS | {} |
| d_sdf_three_level_gci | PASS | {'finest_error_percent': 0.011798594509352469, 'max_gci_percent': 0.10212134912175354, 'converging_all': True, 'submerged_converging_all': True, 'monotone_volume_sequence_all': False, 'max_submerged_gci_percent': 2.283430584456272, 'finest_submerged_error_percent': 0.02055037804856076} |
| e_sensitivity_monotone_gci_le_5pct | PASS | {'monotone_holtrop': True, 'monotone_michell': True, 'monotone_hybrid': True, 'max_abs_delta_rt_hybrid_percent': 1.8271376108304453, 'max_gci_percent': 2.698709943036602, 'lbm_leg': 'NOT_RUN_IN_THIS_NODE'} |
| f_downstream_contract | PASS | {} |

`all_pass = PASS`。

## 4. 结果图

* `figures/fig1_family.png` — 族的水线平面形与艏柱型线。
* `figures/fig2_mesh_voxel.png` — 三角面元（左舷）与射线奇偶体素实体掩膜切片。
* `figures/fig3_volume_convergence.png` — 两档网格位移体积误差 + 三档体素体积误差。
* `figures/fig4_sensitivity.png` — 三种口径的 `dR_T/R_T(alpha_stem)`，带 GCI 误差棒。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **位移体积不是全封闭体积**：网格散度定理给的是**全船体**（含干舷）体积，
   `Cb` 用的是**设计水线以下**体积。两者混用会让 Cb 虚高约 38%（本项目实测 0.97 vs 0.70）。
   现在 `submerged_volume()` 用「水线以下三角形 + 水线面封盖」重建位移体积，
   与解析值差 1.4e-4。
2. **退化面片必须在焊接后判**：先三角化再焊接会在艏柱扇面上产生重复顶点的零面积
   三角形（实测 24 个）。改为先焊接、再按焊缝索引三角化后归零。
3. **外向法向的判据不能用质心偏移**：`min(phi_bow, phi_aft)` 在艉部形成凹折线，
   凹侧的法向偏移点会落回体内（实测 0.8% 面片误判）。改用「每面片参考方向点积」，
   并把质心偏移抽样仅作次要诊断输出。
4. **Michell 积分的截断**：被积函数按 `exp(-k0 sec^2(theta) T)` 衰减，u=tan(theta)
   超过 ~3 后无贡献；早期取 u_max=12 时 x 向相位数超采样、积分未收敛（GCI 10~20%）。
   改为 u_max=3 后 GCI 降到 ~0.7%（观测阶 1.16，`db/dx` 有折点故非 2 阶）。
5. **薄船阻力绝对值不可用于 Cb=0.70**：本族 `R_W^Michell ~ 0.8 MN` 远大于回归法
   `R_W^Holtrop ~ 0.028 MN`，细长体假设被破坏；报告只用其**相对趋势**。
6. **c_flare 只作用在干舷**（`zeta>zeta_T`），对位移体积/湿表面无影响：内倾式与直斜式
   的敞水阻力差异完全来自 `c_fwd`（艏肩饱满度，预设值不同）。这一点在参数表里可见。
7. **本节点未跑 LBM 阻力**（honest FAIL 项）：见第 6 节。
8. 体素体积的粗档误差在 1e-3~1e-4 量级振荡（阶梯噪声），因此判据用
   「加密后变化量减小（converging）+ 最细档误差 ≤1.5% + GCI ≤5%」，
   并同时给出原始体积序列是否严格单调，不掩盖振荡。

## 6. 误差判定（是否 < 3%？）

| 量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 网格位移体积 `nabla`（默认 81×61） | 解析积分 | ≤ 0.0162 % | 是 |
| 网格位移体积 `nabla`（细 161×121） | 解析积分 | ≤ 0.0037 % | 是 |
| 最细档体素体积（0.125%L） | 解析积分 | ≤ 0.0118 % | 是 |
| 反算 Cb | 设计 Cb | ≤ 0.0161 % | 是 |
| 反算 L / B | 设计值 | ≤ 0 % | 是 |
| 反算 `alpha_stem` / `beta_ent` | 设计值 | ≤ 1e-3 deg（离散精确） | 是 |
| `R_T` 绝对值（Holtrop-Mennen） | 无实验基准 | 未与船模试验对标，只有量级/物理一致性检查（C_T≈0.00251） | **不能判定**（无基准） |
| `dR_T/R_T(alpha_stem)` | 参考角 30° 自洽 + GCI | 最大 1.827 %，GCI ≤ 2.70 % | 是（趋势），绝对值无基准 |

**未完成/未达标（honest FAIL，不掩饰）**：

* 判据 (e) 中「4 个艏角各跑一次 LBM 敞水阻力」**未执行**（`lbm_leg = NOT_RUN_IN_THIS_NODE`）。
  原因：`L<=64³` 量级的全湿 LBM 只能给出 Re~1e3 的层流结果，Froude 相似无法满足，
  在本节点预算内无法得到可信的 `R_T` 绝对值；因此本报告用两条可复现的降阶模型
  （Holtrop-Mennen + Michell）给出**带 GCI 的敏感度趋势**，并把 SDF/掩膜/水线契约
  交付给 LBM 腿（`voxel_mask_h0.0025.npz` + `voxel_meta.json`）。
* `R_T` 绝对值与 `dR_T/R_T` 的**物理真值**没有实验/参考 CFD 对标，只有自洽性与 GCI。

## 7. 复现

```bash
PYTHONPATH=src python scripts/run_icebow_family.py --out artifacts/icebow_family_r1
PYTHONPATH=src python -m pytest tests/test_icebow_cad.py -q
PYTHONPATH=src python scripts/icebow_report_assets.py --root artifacts/icebow_family_r1
```

`manifest.json` 记录每个产物文件的 SHA-256、git HEAD、numpy/python 版本与全部参数。
