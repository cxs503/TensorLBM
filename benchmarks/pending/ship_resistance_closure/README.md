# ship_resistance_closure — 船型模型尺度 C_T 闭合（⛔ 未达标：thin-ship 腿对满型船不可用、LBM 腿缺失）

**状态**：`pending` / `promotion: blocked`（诚实 FAIL，不晋升 `verified/`）。
本案例不新增已验证案例，也不重复 `verified/` 中任何既有案例；它把
「KCS / DTMB 5415 / Series 60 × Fn=0.2/0.3/0.4 的 C_T/C_F/C_W 与实验对比」这条
战役要求**量化成三个带证据级的技术腿 + 一台机器判据**，并把不可达的部分如实定性。

## 1. 物理问题与口径

模型尺度裸船体总阻力

```
C_T(Fn) = (1 + k) C_F(Re) + C_W(Fn),   C = R / (0.5 rho U^2 S_wet)
```

| 船型 | 模型长度 | 主要参数 | 几何来源 | 锁定的 Cb |
|---|---|---|---|---|
| KCS | 7.28 m | L/B=7.143, B/T=2.981 | `ship_cad._kcs_half_beam`（参数化近似） | 0.6505（Tokyo-2015） |
| DTMB 5415 | 5.46 m | L/B=7.435, B/T=3.106 | `benchmarks/pending/dtmb5415_resistance/geometry/dtmb5415_underwater.stl`（首手曲面） | 0.507（裸壳，IIHR 口径） |
| Series 60 | 6.096 m（Todd 20 ft 模型） | L/B=6.5, B/T=2.74 | 满度族 `(B/2)(1-xi^2)^p zeta^q`，`p` 二分标定到 Cb | 0.60 / 0.70 / 0.80 |
| Wigley | 6.096 m | L/B=10, L/T=16 | `ship_cad._wigley_half_beam` | 4/9（闭式） |

参考值在 `refs_lock.json` **先锁定后跑**，其 sha256 写入 `results.json.reference_lock`。
`ct_ref = null` 的条目（Series 60 Cb=0.70/0.80）表示库内无首手参考 → `WITHHELD`，
不允许凭空补一个数。

## 2. 方法：三条腿，三级证据，互不晋升

| 腿 | 实现 | 证据级 | 能否当作求解器结果 |
|---|---|---|---|
| GEOMETRY | 半宽图 `b(x,z)` → `nabla, Cb, Cm, Cp, Cwp, S_wet`；对锁定 Cb 失败即 `geometry_rejected` | GEOMETRY | 是（几何审计） |
| VISCOUS | ITTC-1957 `C_F` + Holtrop-Mennen (1984) 形状因子 | CORRELATION | **否**（相关式，不是模拟） |
| WAVE | Michell thin-ship 积分（在船体自身 `db/dx` 上做 Havelock 型 3 重数值积分，含 4×/2×/1× 加密与 GCI） | THEORY | 是（一阶理论，带适用性门） |

**薄船适用性门**（在实测几何上判定，不按船型名称判定）：`L/B >= 6`、
`Cb <= 0.60`、`B/L <= 0.16`、`T/L <= 0.10`。门外的船型**不输出 C_W/C_T**，
只留 `diagnostic_cw` 供归因。

**LBM 腿缺失（诚实声明）**：本仓可达网格下，壁面模型交付的是
**分辨率 Re** 的摩擦系数（`benchmarks/pending/suboff_highre_resistance`：
`Ct_fric/ITTC = 2.49–2.53`，y+ 1878–4660），自由面/壁函数组合又在准入处 fail-closed，
所以「D3Q27+VOSET+LES 直接算模型尺度 C_T」在本仓当前是**不可达**而非「精度稍差」。
把相关式结果冒充 LBM 结果即为未获证据的晋升，本案例拒绝这样做。

## 3. 计算结果（全部由 `run.py` 程序化写入 `results.json`）

| case | Cb（网格 / 锁定） | Cwp | S_wet [m²] | 薄船门 | C_T (Fn=0.2/0.3/0.4) | 参考 Fn | C_T 参考 | 偏差 |
|---|---|---|---|---|---|---|---|---|
| kcs | 0.6455 / 0.6505 | 0.791 | 8.904 | rejected | — | 0.26 | — | withheld |
| dtmb5415 | 0.4993 / 0.507 | 0.761 | 4.423 | admitted | 0.00444 / — / 0.00775 | 0.28 | 0.0042 | **+26.5%** |
| series60_cb0.600 | 0.6000 / 0.60 | 0.615 | 7.423 | admitted | 0.00459 / 0.00658 / 0.01219 | 0.30 | 0.0076 | **−13.4%** |
| series60_cb0.700 | 0.7000 / 0.70 | 0.718 | 7.983 | rejected | — | 0.30 | — | withheld |
| series60_cb0.800 | 0.8000 / 0.80 | 0.820 | 8.622 | rejected | — | 0.30 | — | withheld |
| wigley | 0.4444 / 0.4444 | 0.665 | 5.529 | admitted | 0.00449 / 0.00549 / 0.00593 | — | — | 仅量级自检 |

波阻腿的加密与 GCI（Fn=参考值）：dtmb5415 `C_W=1.836e-3`（GCI **4.24%**）、
Series 60 Cb=0.60 `C_W=3.202e-3`（GCI 1.45%）、Wigley `C_W=1.641e-3`（GCI 3.31%）。
阻力分解自洽残差：8 个样本 `max |C_T − [(1+k)C_F + C_W]| / C_T = 0.0`（浮点精确）。

## 4. 机器判据（`results.json.criteria`）

| 判据 | 状态 | 数字 |
|---|---|---|
| (a) KCS Fn=0.26 偏差 <8% | **withheld** | 无可用波阻腿：参数化 KCS 的 Cb=0.6455、Cwp=0.791（实船 0.667）双双重门 |
| (b) DTMB 5415 Fn=0.28 偏差 <10% | **fail** | +26.5%（同面积口径）/ +44.2%（换算到参考 S 口径） |
| (c) Series 60 Cb=0.70 Fn=0.3 偏差 <10% | **withheld** | 无首手参考 + Cb=0.70 门拒 |
| (d) 网格收敛 GCI <8% | **pass** | 1.45% / 3.31% / 4.24% |
| (e) 分解自洽残差 <1e-6 | **pass** | 0.0（8 样本） |
| (f) C_T(Fn) 单调递增 | **partial** | 完整阶梯 2/2 严格单调（Series 60 Cb=0.6、Wigley）；DTMB 5415 Fn=0.30 因最粗级非单调而拒发，不给假点 |
| (g) pending → verified | **fail** | (a)-(c) 未达标，不得晋升 |
| (h) pytest ≥15 全绿 | **pass** | `tests/test_ship_resistance_closure.py` 24 passed |

## 5. 注意点 / 局限

* **几何保真度是主因**：参数化 KCS 的水线面系数 0.791 对实船 0.667（+19%），
  使 thin-ship 的源强 `db/dx` 系统性过大——其诊断 `C_W=6.86e-3` 是实测残差
  `C_R = C_T − (1+k)C_F ≈ 0.3e-3` 的约 20 倍。这不是可调偏差，而是几何不可用。
* **thin-ship 一阶理论的系统高估**：5415 在 Fn=0.28（波峰区）高估 +26.5%；
  Series 60 Cb=0.6 反而 −13.4%，说明误差不单调、不能靠单一校正因子吸收。
* **参考面积口径**：5415 参考值按 IIHR `S/L^2=0.1302`；本案例 STL 三角面
  积分给 `S=4.423 m²`（`S/L^2=0.1483`，含声纳导流罩与面板化），换算后偏差 +44.2%。
  两种口径都给出，不挑选有利者。
* **台阶/离散误差**：半宽图来自 STL 重心网格采样的 `max|y|`；`Cb` 随采样分辨率
  在 0.4993（289×145）与 0.5028（145×97）之间移动 ±0.7%，属采样误差而非物理。
* **波阻积分的 Fn 下限**：轴向相位 `k0 sec(theta) dx` 超过 π/8 时模块拒绝出数
  （`quadrature_unresolved`），因此 Fn≲0.1 不在可用域；这不是"算得慢"，是混叠。
* **形状因子假设**：Holtrop 的 `lcb=0%`、`Cstern=0` 是显式假设（这些船型无锁定 LCB），
  `results.json` 记录 ±2%L / Cstern±10 的敏感性区间。

## 6. 晋级路径

1. 几何：为 KCS 与 Series 60 家族换**首手型值**（Tokyo-2015 offsets / Todd 系列型值表），
   把 `Cwp` 误差压到 <2%，重跑本 runner（几何门会自动重判）。
2. 波阻：thin-ship 换成 **Rankine 面元或 ITTC-1978 三因次外推**，并在 Wigley 与
   Series 60 上与实测 `C_W` 对标；GCI 门保持 <8%。
3. 粘性/总力：只有在自由面 + 壁面模型的准入矩阵打通、且壁链能交付**模型 Re** 摩擦
   （而非分辨率 Re）之后，才允许把 LBM 结果放进 `C_T`；在此之前 (b)(c) 只能 by construction 失败。
4. 全部达成后由归档 PR 把本目录移入 `verified/ship_resistance_closure/`。

## PROVENANCE（值 — 文件 — 键，机器可核验）

| 值 | 文件 | 键 |
|---|---|---|
| Cb(kcs)=0.6455, Cwp=0.791 | `results.json` | `cases.kcs.geometry.cb_grid`, `.cwp_grid` |
| Cb(5415)=0.4993, S=4.423 | `results.json` | `cases.dtmb5415.geometry.cb_grid`, `.wetted_area_m2` |
| C_T(5415,Fn=0.28)=5.314e-3, 偏差 +26.53% | `results.json` | `cases.dtmb5415.reference_row.total.ct`, `.comparison.error_pct` |
| C_T(S60 Cb0.6,Fn=0.30)=6.581e-3, 偏差 −13.41% | `results.json` | `cases.series60_cb0.600.reference_row.total.ct`, `.comparison.error_pct` |
| GCI 波阻腿 | `results.json` | `criteria.d_grid_convergence_gci_below_8.gci_pct` |
| 分解残差 | `results.json` | `criteria.e_decomposition_residual_below_1e-6.max_residual_relative` |
| KCS 诊断 C_W=6.861e-3 | `results.json` | `cases.kcs.wave_at_reference_fn.diagnostic_cw` |
| 参考锁 sha256 | `results.json` | `reference_lock.sha256` |

复现：

```bash
PYTHONPATH=src python benchmarks/pending/ship_resistance_closure/run.py     # ~100 s, CPU
PYTHONPATH=src python -m pytest tests/test_ship_resistance_closure.py -q     # 24 passed
```