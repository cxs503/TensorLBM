# XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC — 正则化边界（非平衡应力重构）对标报告

## 1 问题与基准出处
对标 `joconnor22/LIFE` `GridClass::regularisedBC`（`src/Grid.cpp:386-466`；调用点 :321/:333/:351/:368 = eWall/eVelocity/eFreeSlip/ePressure）。OSS 做法：边界节点用 Zou/He 闭包定密度（速度型）或法向速度（压力型）→ 未知方向反对称补齐 `f_q=f_q^eq+(f_opp-f_opp^eq)` → 以该节点非平衡应力 `S_ab=Σ c_a c_b(f-f^eq)` 做二阶 Hermite 重写 `f_q=f_q^eq+w_q/(2c_s^4)Q_q:S`。判据：owner 15%，拉伸 3%。

## 2 方法与实现
* 生产实现 `src/tensorlbm/regularised_bc.py`（NumPy，D2Q9/D3Q19）：`regularised_face` / `complete_unknowns` / `project_second_order` / `apply_regularised_wall`，入口 `python -m tensorlbm.regularised_bc`。
* 主求解器调用：`src/tensorlbm/channel_kernel_numpy.py::run_channel(wall_bc="regularised")`（D3Q19，opt-in，默认仍 bounce_back）。
* 对标驱动 `xval/.../run_xval.py`（identity/steady/oscillating 三臂），工件 `artifacts/*.json`，本报告由 `make_report.py` 从工件程序化注入。
* 独立校验：`_oss_with_wall` 是 OSS 例程的逐行标量转写，与生产向量化实现对比。

## 3 参数表
| 量 | 值 |
|---|---|
| ν | 0.1（τ=0.8） |
| Ma=U/c_s | 0.02/√(1/3)≈0.035 |
| Couette U_wall | 0.02 |
| Poiseuille u_max | 0.02（F=8νu_max/H²） |
| ny | 16（稳态）、48（振荡壁） |
| nx（x 周期） | 2 |
| 收敛判据 | max|Δu|/max|u|<1e-10（每 200 步） |
| 振荡壁 | U0=0.02，T=200/100 步，6 周期，末周期 DFT |
| 精度 | float64 单进程 CPU |

## 4 对照表（ref vs got）
### 4.1 闭包恒等性（synthetic 边界态 × 4 个法向）
| 指标 | 参考 | got | 判据 |
|---|---|---|---|
| 非平衡应力保真 | 重构前 S（精确） | 5.873e-16 | <1e-13 ✓ |
| 非平衡三阶矩 ghost | 0 | 1.735e-17 | <1e-13 ✓ |
| 与 OSS 标量转写一致 | LIFE Grid.cpp 逐行转写 | 1.388e-17 | <1e-12 ✓ |

### 4.2 稳态 Couette / Poiseuille（解析解参考，ny=16）
| 算例 | 闭包 | L2 相对误差 | <15% | <3% | 拟合壁面位置 | 收敛步数 |
|---|---|---|---|---|---|---|
| couette | regularised | 4.007e-09 % | ✓ | ✓ | (0.000000, 15.000000) | 5400 |
| couette | bounce_back | 6.319e-09 % | ✓ | ✓ | (0.500000, 14.500000) | 4600 |
| poiseuille | regularised | 1.129e+00 % | ✓ | ✓ | (0.032069, 14.967931) | 5400 |
| poiseuille | bounce_back | 6.427e-01 % | ✓ | ✓ | (0.516448, 14.483552) | 4800 |

### 4.3 振荡壁 Stokes 第二问题（ny=48，末周期 DFT，解析 Stokes 层参考）
| 闭包 | T(步) | δ | H/δ | 复振幅 L2 | <15% | 幅值误差 | 相位误差(掩膜) | 逐周期漂移 |
|---|---|---|---|---|---|---|---|---|
| regularised | 200 | 2.523 | 18.6 | 2.874e+00 % | ✓ | 4.413e-01 % | 3.021e-02 rad | 4.0e-05 |
| regularised | 100 | 1.784 | 26.3 | 5.624e+00 % | ✓ | 1.031e+00 % | 5.770e-02 rad | 5.3e-05 |
| bounce_back | 200 | 2.523 | 18.2 | 4.653e+00 % | ✓ | 4.792e-01 % | 4.831e-02 rad | 3.3e-05 |
| bounce_back | 100 | 1.784 | 25.8 | 9.181e+00 % | ✓ | 1.177e+00 % | 9.363e-02 rad | 4.0e-05 |

## 5 收敛性 / 参数敏感性
* 振荡壁 regularised：T=200→100（δ 2.52→1.78，H/δ 18.6→26.3）复振幅 L2 2.874e+00 %→5.624e+00 %，末周期漂移 ≤5.3e-05（已是稳态周期解）。
* 振荡壁 bounce_back：T=200→100（δ 2.52→1.78，H/δ 18.2→25.8）复振幅 L2 4.653e+00 %→9.181e+00 %，末周期漂移 ≤4.0e-05（已是稳态周期解）。
* 稳态 Couette：两闭包都在机器精度内复现线性解（reg 4.007e-09 %，BB 6.319e-09 %），拟合壁面分别在 0 与 0.5（正则化闭包本身即壁面；半程反弹在半个链接外），说明稳态算例不区分闭包，区分闭包的是非定常项（§4.3）。
* 分辨率敏感性（振荡壁 T=100 时 δ=1.78 仅约 2.7 个格子）误差随 δ/Δx 减小而增大，符合边界闭包在薄 Stokes 层上的分辨要求。

## 6 误差判定（逐项回答是否 <3%）
* steady couette/regularised: 4.007e-09 % → <3% ✓
* steady couette/bounce_back: 6.319e-09 % → <3% ✓
* steady poiseuille/regularised: 1.129e+00 % → <3% ✓
* steady poiseuille/bounce_back: 6.427e-01 % → <3% ✓
* oscillating/regularised/T200: 2.874e+00 % → <3% ✓
* oscillating/regularised/T100: 5.624e+00 % → <15% ✓（未达 3%）
* oscillating/bounce_back/T200: 4.653e+00 % → <15% ✓（未达 3%）
* oscillating/bounce_back/T100: 9.181e+00 % → <15% ✓（未达 3%）
* identity 三项：5.9e-16 / 1.7e-17 / 1.4e-17 → 恒等性判据（1e-12~1e-13）✓（远严于 3%）

## 7 失败模式、局限与工件
* 局限：LIFE 本体未执行（C++/Boost/LAPACK FSI；ChannelFlow 例子无对应参数工况），OSS 语义由逐行标量转写交叉校验，属对标限制。
* 未移植：角点（双法向）分支与 CENTRAL_MOMENTS 分支（OSS_DIFF.md 列明）。
* 失败模式（实测）：把半程反弹的反射种群取成碰撞后状态会让壁节点形成自反馈回路并数值发散（ny=16,U=0.05,τ=0.8，约 1e2 步 rho 溢出），控制臂固定用到达态（碰撞前）反射，代码注释登记。
* 工件：`artifacts/{identity,steady,oscillating,summary}.json` + 本报告 + OSS_DIFF.md + PREREG.md。

## 8 复现命令
```bash
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval.py --arm all
PYTHONPATH=src python -m pytest tests/test_regularised_bc.py -q
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/make_report.py
```
