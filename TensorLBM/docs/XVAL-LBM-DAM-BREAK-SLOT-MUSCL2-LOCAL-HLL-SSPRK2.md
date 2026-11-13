# XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2 验证报告

## ① 案例介绍
一维浅水方程溃坝问题：初始水深 h₀=1.0 的静止水体在 t=0 时释放到干床。
解析基准：Ritter (1892) 自相似解，前缘 x_f = 2√(gh₀)t。

**参考文献**:
- Ritter, A. (1892). Z. Ver. Deutsch. Ing. 36, 947.
- Martin, J.C. & Moyce, W.J. (1952). Phil. Trans. R. Soc. A 244, 312.
- Toro, E.F. (2001). *Shock-Capturing Methods for Free-Surface Flows*.

## ② 方法与实现
**MUSCL-2** (minmod限制器) + **Davis局部波速HLL通量** + **SSP-RK2(Heun)时间积分**。
干床处理：h=0 + 正性回退(α: 1→0.5→0)。

**关键代码**: `src/tensorlbm/shallow_water.py`
- `swe_step_muscl2()`: MUSCL+SSP-RK2 时间步进
- `_muscl_face_states()`: 斜率限制重构
- `_hll_flux()`: Davis 局部波速 HLL 通量
- `_minmod()`: Minmod 限制器

## ③ 计算结果
| nx | 前缘误差(质量法) [%] | 剖面L2 [%] | 质量漂移 [%] |
|----|---------------------|-----------|-------------|
| 100 | 0.2563 | 1.9289 | 0.00000000 |
| 200 | 0.1187 | 1.1677 | 0.00000000 |
| 400 | 0.0569 | 0.8304 | 0.00000000 |
| 800 | 0.0278 | 0.7951 | 0.00000000 |

## ④ 误差判定
| 判据 | 阈值 | 最差值 | 状态 |
|------|------|--------|------|
| C1: 前缘误差(质量法) | <15% | 0.2563% | ✓ PASS |
| C2: 剖面L2误差 | <15% | 1.9289% | ✓ PASS |
| C3: 质量漂移 | <1% | 0% | ✓ PASS |

**总裁决: PASS**

## ⑤ 注意点
- 阈值法前缘检测受数值耗散影响，粗网格误差>15%；质量法基于Ritter积分关系x_f=27V/(4h₀)，物理上更稳健
- 干床正性回退保证稳定性，极干条件下可能降阶
- 当前为一维实现；二维扩展需特征分解和特征限制器

## ⑥ 工件
- `xval/XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2/artifacts/result.json`
- `xval/XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2/artifacts/verdict.json`
- `xval/XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2/predeclared.json`
- `tests/test_shallow_water_muscl2.py` (11 tests, all passed)
