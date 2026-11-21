# TensorLBM 已验证 Benchmark 使用教程（35 案）

每个案例含：Benchmark 介绍、计算条件设置、软件使用步骤、计算结果（速度场 / 压力场等）、与文献或解析解的比较（阻力、表面压力系数等）。判据数字一律取自入库 `result.json` 机器档案；演示图为缩短时长的可视化档（`docs/benchmarks/demos/`），定量结论以存档扫描为准。

| 案例 | 分类 | 网页 | Word | PDF | 入库位置 |
|---|---|---|---|---|---|
| [顶盖驱动方腔流 Re=100（lid-driven cavity）（cavity_re100）](cavity_re100.md) | 方腔流（腔体族） | [md](cavity_re100.md) | [docx](files/cavity_re100.docx) | [pdf](files/cavity_re100.pdf) | `benchmarks/verified/cavity/re100` |
| [顶盖驱动方腔流 Re=400（lid-driven cavity）（cavity_re400）](cavity_re400.md) | 方腔流（腔体族） | [md](cavity_re400.md) | [docx](files/cavity_re400.docx) | [pdf](files/cavity_re400.pdf) | `benchmarks/verified/cavity/re400` |
| [顶盖驱动方腔流 Re=1000（lid-driven cavity，RLBM 碰撞）（cavity_re1000）](cavity_re1000.md) | 方腔流（腔体族） | [md](cavity_re1000.md) | [docx](files/cavity_re1000.docx) | [pdf](files/cavity_re1000.pdf) | `benchmarks/verified/cavity/re1000` |
| [3D 方腔流 Re=400（展向周期，D3Q19 MRT）（cavity_3d）](cavity_3d.md) | 方腔流（腔体族） | [md](cavity_3d.md) | [docx](files/cavity_3d.docx) | [pdf](files/cavity_3d.pdf) | `benchmarks/verified/cavity/3d` |
| [全 3D 立方方腔流 Re=1000（5 壁 + 移动顶盖，D3Q19 TRT）（cavity_3d_full）](cavity_3d_full.md) | 方腔流（腔体族） | [md](cavity_3d_full.md) | [docx](files/cavity_3d_full.docx) | [pdf](files/cavity_3d_full.pdf) | `benchmarks/verified/cavity_3d_full` |
| [Schäfer–Turek 2D-1 圆柱绕流 Re=20：稳态层流基准（cylinder_re20_st）](cylinder_re20_st.md) | 圆柱绕流（外流族） | [md](cylinder_re20_st.md) | [docx](files/cylinder_re20_st.docx) | [pdf](files/cylinder_re20_st.pdf) | `benchmarks/verified/cylinder/re20_st` |
| [自由流圆柱绕流 Re=100：Kármán 涡街（Braza 1986）（cylinder_re100）](cylinder_re100.md) | 圆柱绕流（外流族） | [md](cylinder_re100.md) | [docx](files/cylinder_re100.docx) | [pdf](files/cylinder_re100.pdf) | `benchmarks/verified/cylinder/re100` |
| [自由流圆柱绕流 Re=200：Kármán 涡街（Braza 1986）（cylinder_re200）](cylinder_re200.md) | 圆柱绕流（外流族） | [md](cylinder_re200.md) | [docx](files/cylinder_re200.docx) | [pdf](files/cylinder_re200.pdf) | `benchmarks/verified/cylinder/re200` |
| [3D 展向周期圆柱绕流 Re=40：稳态尾流 Cd（表面 MEM 口径）（cylinder_3d）](cylinder_3d.md) | 圆柱绕流（外流族） | [md](cylinder_3d.md) | [docx](files/cylinder_3d.docx) | [pdf](files/cylinder_3d.pdf) | `benchmarks/verified/cylinder_3d` |
| [自由流方柱绕流 Re=100：涡脱落对 2D 数值簇验证（square_cylinder）](square_cylinder.md) | 方柱绕流（外流族） | [md](square_cylinder.md) | [docx](files/square_cylinder.docx) | [pdf](files/square_cylinder.pdf) | `benchmarks/verified/square_cylinder` |
| [DARPA SUBOFF 裸艇体 Re=1000：总阻力 Blasius 验证（suboff_re1000）](suboff_re1000.md) | 潜体绕流（外流族） | [md](suboff_re1000.md) | [docx](files/suboff_re1000.docx) | [pdf](files/suboff_re1000.pdf) | `benchmarks/verified/suboff_re1000` |
| [等温 Sod 激波管（D2Q9 可压缩 Riemann 问题）（sod_shock_tube）](sod_shock_tube.md) | 可压缩流 | [md](sod_shock_tube.md) | [docx](files/sod_shock_tube.docx) | [pdf](files/sod_shock_tube.pdf) | `benchmarks/verified/sod_shock_tube` |
| [Kovasznay 二维稳态层流（解析 Navier–Stokes 解）（kovasznay_2d）](kovasznay_2d.md) | 可压缩流 | [md](kovasznay_2d.md) | [docx](files/kovasznay_2d.docx) | [pdf](files/kovasznay_2d.pdf) | `benchmarks/verified/kovasznay_2d` |
| [二维 Poiseuille 流（压力差驱动）— 解析抛物线剖面验证（poiseuille_2d）](poiseuille_2d.md) | 内流解析解 | [md](poiseuille_2d.md) | [docx](files/poiseuille_2d.docx) | [pdf](files/poiseuille_2d.pdf) | `benchmarks/verified/poiseuille_2d` |
| [二维体积力驱动平面 Poiseuille 流（D2Q9 BGK）— 解析抛物线验证 + Guo 强迫源项零和修复（poiseuille_2d_bodyforce）](poiseuille_2d_bodyforce.md) | 内流解析解 | [md](poiseuille_2d_bodyforce.md) | — | — | `benchmarks/verified/poiseuille_2d_bodyforce` |
| [圆管 Poiseuille 流（3D D3Q19）— 水力半径口径解析抛物线验证（poiseuille_3d_pipe）](poiseuille_3d_pipe.md) | 内流解析解 | [md](poiseuille_3d_pipe.md) | [docx](files/poiseuille_3d_pipe.docx) | [pdf](files/poiseuille_3d_pipe.pdf) | `benchmarks/verified/poiseuille_3d_pipe` |
| [方形与 2:1 矩形管 Poiseuille 流（3D D3Q19）— 双重奇正弦级数与 Shah-London 不变量验证（poiseuille_3d_duct）](poiseuille_3d_duct.md) | 内流解析解 | [md](poiseuille_3d_duct.md) | [docx](files/poiseuille_3d_duct.docx) | [pdf](files/poiseuille_3d_duct.pdf) | `benchmarks/verified/poiseuille_3d_duct`、`benchmarks/verified/poiseuille_3d_duct/duct_ar05` |
| [二维 Couette 流（平板剪切流）— 解析线性剖面验证（couette_2d）](couette_2d.md) | 内流解析解 | [md](couette_2d.md) | [docx](files/couette_2d.docx) | [pdf](files/couette_2d.pdf) | `benchmarks/verified/couette_2d` |
| [二维 Couette–Poiseuille 混合驱动通道流：上壁滑动 + 体积力的解析叠加验证（couette_poiseuille）](couette_poiseuille.md) | 内流解析解 | [md](couette_poiseuille.md) | — | — | `benchmarks/verified/couette_poiseuille` |
| [Taylor-Couette 环形流（内柱旋转）— Couette 族解析剖面与转矩验证（taylor_couette）](taylor_couette.md) | 内流解析解 | [md](taylor_couette.md) | [docx](files/taylor_couette.docx) | [pdf](files/taylor_couette.pdf) | `benchmarks/verified/taylor_couette` |
| [Stokes 第一问题：平板突然起动的边界层增长（erfc 解）（stokes_first）](stokes_first.md) | 非定常解析 | [md](stokes_first.md) | [docx](files/stokes_first.docx) | [pdf](files/stokes_first.pdf) | `benchmarks/verified/stokes_first_problem` |
| [Stokes 第二问题：振荡平板的受迫振荡边界层（stokes_second）](stokes_second.md) | 非定常解析 | [md](stokes_second.md) | [docx](files/stokes_second.docx) | [pdf](files/stokes_second.pdf) | `benchmarks/verified/stokes_second_problem` |
| [Womersley 振荡流：周期脉动压力梯度驱动的通道流（womersley）](womersley.md) | 非定常解析 | [md](womersley.md) | [docx](files/womersley.docx) | [pdf](files/womersley.pdf) | `benchmarks/verified/womersley` |
| [启动 Poiseuille 流：突加恒定压力梯度的通道瞬态（startup_poiseuille）](startup_poiseuille.md) | 非定常解析 | [md](startup_poiseuille.md) | [docx](files/startup_poiseuille.docx) | [pdf](files/startup_poiseuille.pdf) | `benchmarks/verified/startup_poiseuille` |
| [衰减剪切波：单 Fourier 模的粘性耗散（shear_wave_decay）](shear_wave_decay.md) | 非定常解析 | [md](shear_wave_decay.md) | [docx](files/shear_wave_decay.docx) | [pdf](files/shear_wave_decay.pdf) | `benchmarks/verified/shear_wave_decay` |
| [二维 Taylor–Green 涡：经典衰减基准（taylor_green_2d）](taylor_green_2d.md) | 非定常解析 | [md](taylor_green_2d.md) | [docx](files/taylor_green_2d.docx) | [pdf](files/taylor_green_2d.pdf) | `benchmarks/verified/taylor_green_2d` |
| [三维 Taylor–Green 涡：近解析衰减与涡拉伸（taylor_green_3d）](taylor_green_3d.md) | 非定常解析 | [md](taylor_green_3d.md) | [docx](files/taylor_green_3d.docx) | [pdf](files/taylor_green_3d.pdf) | `benchmarks/verified/taylor_green_3d` |
| [液滴毛细振荡：m=2 模态频率 vs Rayleigh 解析（droplet_oscillation）](droplet_oscillation.md) | 多相/热/声/多孔 | [md](droplet_oscillation.md) | [docx](files/droplet_oscillation.docx) | [pdf](files/droplet_oscillation.pdf) | `benchmarks/verified/droplet_oscillation` |
| [静态液滴 Young–Laplace 定律（表面张力标定）（laplace_droplet）](laplace_droplet.md) | 多相/热/声/多孔 | [md](laplace_droplet.md) | [docx](files/laplace_droplet.docx) | [pdf](files/laplace_droplet.pdf) | `benchmarks/verified/laplace_droplet` |
| [PR 状态方程高密度比伪势液滴 Young–Laplace：σ 全自测与界面加密收敛（laplace_pr_eos）](laplace_pr_eos.md) | 多相/热/声/多孔 | [md](laplace_pr_eos.md) | [docx](files/laplace_pr_eos.docx) | [pdf](files/laplace_pr_eos.pdf) | `benchmarks/verified/laplace_pr_eos` |
| [CAC 相场两相液滴 Young–Laplace：Route-B σ 标定与 R/密度比迁移（laplace_cac_phasefield）](laplace_cac_phasefield.md) | 多相/热/声/多孔 | [md](laplace_cac_phasefield.md) | [docx](files/laplace_cac_phasefield.docx) | [pdf](files/laplace_cac_phasefield.pdf) | `benchmarks/verified/laplace_cac_phasefield` |
| [方腔自然对流（de Vahl Davis 基准）（thermal_cavity）](thermal_cavity.md) | 多相/热/声/多孔 | [md](thermal_cavity.md) | [docx](files/thermal_cavity.docx) | [pdf](files/thermal_cavity.pdf) | `benchmarks/verified/thermal_cavity` |
| [方腔自然对流 Ra=10⁴ 重算与验证教程（cavity_natural_convection）](cavity_natural_convection.md) | 多相/热/声/多孔 | [md](cavity_natural_convection.md) | [docx](files/cavity_natural_convection.docx) | [pdf](files/cavity_natural_convection.pdf) | `benchmarks/verified/cavity_natural_convection` |
| [平面声波：线性色散与衰减（acoustics）](acoustics.md) | 多相/热/声/多孔 | [md](acoustics.md) | [docx](files/acoustics.docx) | [pdf](files/acoustics.pdf) | `benchmarks/verified/acoustics` |
| [周期圆柱方阵渗透率（Sangani–Acrivos 1982）（permeability）](permeability.md) | 多相/热/声/多孔 | [md](permeability.md) | [docx](files/permeability.docx) | [pdf](files/permeability.pdf) | `benchmarks/verified/permeability` |
