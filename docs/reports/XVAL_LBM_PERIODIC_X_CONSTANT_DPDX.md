# XVAL：本仓 vs joconnor22/LIFE `single-phase-periodic-x-plus-constant-dpdx-driving`

锚点 `XVAL-LBM-SINGLE-PHASE-PERIODIC-X-PLUS-CONSTANT-DP`｜轮次 1｜OSS 版本 v1.0.3（2020-06-10）｜GPL-3.0：只转写算法，不复制代码。

## ① 问题与 ref 出处
OSS 该 tag 的做法=**x 周期 + 常 dp/dx 体积力驱动的单相通道**。从源码查得（零手抄）：
* 网格/权重 `src/Grid.cpp:1248-1249`；时间步 `lbmKernel` `Grid.cpp:37-98`
* push 流式、x/y 全周期取模 `Grid.cpp:229,240`
* BGK 碰撞 + 力 `Grid.cpp:243`：`f=(f_n + ω(feq−f_n) + (1−ω/2)S)`
* Guo 二阶源项 `Grid.cpp:274-285`；宏观速度半步修正 `Grid.cpp:297-298`
* 常 dp/dx 体积力**只在初始化时算一次**、不随 ρ 更新 `Grid.cpp:1042-1043`；单位换算 `Grid.cpp:1257-1260`
* x 周期 = 左右墙设 `eFluid`（`Grid.cpp:930-949`，`inc/params.h:64` 注释「set to eFluid for periodic」）
* 上下墙 = `eWall` → regularised BC（Latt 2008），`Grid.cpp:302-330`

**关键事实（差异清单第 0 条）**：LIFE 仓库**没有任何算例真正跑过这条驱动路径**——7 个 `examples/*/params.h` 全部 `dpdx=0`、`gravityX/Y=0`。所以 OSS 提供的是*机制*而非*数值参考*；数值锚点必须是解析解。

## ② 方法与实现
* `xval/xval_lbm_periodic_dpdx/life_ref.py`：OSS 做法**独立转写**（保留 OSS 的格子编号顺序，与 `tensorlbm.d2q9` 不同），逐阶段对齐 `lbmKernel` 四阶段。
* `src/tensorlbm/periodic_channel_dpdx.py`：本仓新增 D2Q9 周期通道模块，两种壁面闭合（halfway bounce-back / regularised）+ 两种力的注入（split-Guo / Guo 半步冻结）+ dpdx→格子力换算器。
* 数值锚点：Exact plane Poiseuille `u(y)=G(y−y_w)(H−(y−y_w))/(2ν)`，壁法向坐标 `s=(y−y_w)/H`。

## ③ 参数表（无量纲）
| 量 | 值 |
|---|---|
| D2Q9，τ | 0.5375（=LIFE ChannelFlow 值） |
| ν = (τ−1/2)/3 | 0.0125 |
| 目标 u_max | 0.05 → Re_H = u_max H/ν = 128 (H=32) |
| 力 G = 8νu_max/H² | H=8: 7.8e-4；H=16: 1.95e-4 |
| Ma = u_max/c_s | 0.087 |
| ρ0 | 1.0，格距/时距 1 |
| x | 周期（nx=2，流态 x 不变） |

## ④ 对照表（ref vs got）
| 臂 | H | rel-L2 vs 解析 | u_max err | 收敛步 | <15%? |
|---|---|---|---|---|---|
| LIFE-REF | 8 | 2.554e-02 | 1.993e-02 | 9200 | ✓ |
| LIFE-REF | 16 | 6.608e-03 | 4.984e-03 | 33600 | ✓ |

| 臂 | 结果 | 判据 |
|---|---|---|
| GOT-PORT-REG（本仓模块，regularised 移植） | 3000 步质量漂移 **-1.196e-02** | ✗ 不守恒 → **移植臂未通过**（honest-FAIL） |
| GOT-PORT-BB（本仓模块，halfway BB + split-Guo） | 质量漂移 **-9.853e-03** | ✗ 同上 |
| CONTROL（去掉 y 壁，全周期） | 质量漂移 **-5.503e-01** | ✓ 度量有齿（能区分对错） |

## ⑤ 收敛性 / 参数敏感性
* 网格：rel-L2 2.554e-02(H=8) → 6.608e-03(H=16)，实测阶 **1.95**（理论 2 阶）✓
* 力幅（Ma）：scale 0.25/4.0 两档 rel-L2 均为 2.554e-02（该网格阶在解析截断主导，Ma 依赖被淹没，诚实记为**未分辨**）
* 松弛 τ：0.5375 → 9.074e-03；τ=0.8 → 9.074e-03

## ⑥ 失败模式与工件
* **失败模式 1（已证）**：本仓新增模块的两个臂在 3000 步内出现 ~1% 质量漂移并最终发散 → 移植臂 **honest-FAIL**（fail_stage=port-conservation；failure_reason=流式/壁面闭合未守恒，尚未定位）。未把它登记为能力。
* **失败模式 2（已证）**：LIFE 的 dpdx 单位换算 `force_xy=dpdx·Dx·Dt²/Dm, Dm=Drho·Dx³` 与动量标定值差 **1/Dx**（比值 50，1/Dx=50），且该路径无算例覆盖 → 移植时**必须**显式选择换算约定（本仓模块提供 `standard` / `life_compat` 两个）。
* 工件：`xval/xval_lbm_periodic_dpdx/out/xval_life_periodic_dpdx.json`、`fig_xval_periodic_dpdx.png`；入口 `PYTHONPATH=src python3 xval/xval_lbm_periodic_dpdx/run_xval.py`。

## 差异清单（本仓 vs OSS）
| 项 | OSS (LIFE) | 本仓 |
|---|---|---|
| 周期性 | push 流式 x/y 全取模，左右墙 `eFluid` | pull 流式 + `np.roll` x 周期；y 由壁面闭合决定 |
| 壁面闭合 | 节点式 regularised BC（Latt 2008），壁在节点中心，H=ny−1 | 生产约定 halfway bounce-back，壁在链中点，H=ny−2（差 1 格几何偏移） |
| 力注入 | (1−ω/2) 全 Guo + 宏观速度 +F/2，力**冻结**在 ρ0 初值 | split-Guo：碰撞后注入、权重 1、速度无半步修正（`channel_kernel_numpy` 生产约定） |
| 单位换算 | `dpdx·Dx·Dt²/Dm`（=标准值/Dx） | 提供 standard 与 life_compat 两种，默认 standard |
| 校验面 | 逐字节 RefData，但 dpdx/gravity 路径**零覆盖** | 以解析 Poiseuille + 网格阶 + 控制臂为锚 |
| 语言/精度 | C++ OMP，double | NumPy，float64 |

## 误差判定（显式回答）
* LIFE-REF 转写 vs 解析解：H=8 **2.55%**，H=16 **0.66%** → **<15% ✓**；但**不 <3%**（H=16 为 0.66%，属 O(dx²) 未收敛区，须 H≈64 以上才可达 3%，本轮计算预算未覆盖，如实记录）。
* 本仓移植臂：**无有效数值，honest-FAIL**。
