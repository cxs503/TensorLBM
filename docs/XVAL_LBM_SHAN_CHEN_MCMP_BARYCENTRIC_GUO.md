# XVAL — TensorLBM Shan-Chen MCMP (barycentric velocity + Guo forcing) vs `lssfau/lbmpy`

Anchor: **XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO** · 第 1 轮 · 开源对标 · repo TensorLBM · mission 18325
结论：**PASS**，主臂相对 L2 误差 0.001068 ≪ 15 %

## ① 问题与参考出处

两个组分 Shan-Chen (MCMP) 是船舶气层减阻 / 空化 / 气液掺混模拟的底座模型。本仓 MCMP 路径
`tensorlbm.multiphase.collide_sc_two_component` 的默认约定是“速度平移 + 线性伪势”，与成熟开源实现的
“重心(barycentric)速度 + Guo 二阶受力 + 指数伪势”是两套不同离散化。本单用 OSS 自带的**数值参考解**
做交叉对标：我们的重心-Guo 路径能否复现成熟实现。

参考解（零手抄，程序化抓取）：`lssfau/lbmpy` @ `0afc9d52896e`，文件
`tests/shan_chen/test_shan_chen_two_component.py`，源文本 sha256 `7e88e89825a34a56...`，参考数组 sha256
`3a2b6127f160d8dc...`。该测试断言 1000 步后 64x64 周期盒子两个组分密度剖面（参考值本身来自
`lbm-principles-practice` 第 9 章独立 C++ 实现）。抓取器 `xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/fetch_oss_ref.py`
走 GitHub contents API + base64 后**解析**（不誊写）。

## ② 方法与实现

* 对比口径：两组分密度剖面逐点比，主判据 = 相对 L2 范数（跨组分取最差）。
* 主臂（本仓路径）：`collide_sc_two_component(..., u_eq="mixture", force_scheme="guo", psi=psi_exp)`
  —— 平衡态用重心速度、Guo 2002 显式源项 `(1-1/2tau)`、伪势 `psi(rho)=rho0*(1-exp(-rho/rho0))`。
* **本单最小代码改动**：`sc_two_component_force` / `collide_sc_two_component` 新增可选 `psi=` 参数
  （默认 `None` = 历史线性 psi，逐位不变）——复现 OSS 所必需的唯一能力缺口。
* 符号映射（自行推导）：OSS `F_a = -g_ab psi_a * sum_i w_i c_i psi_b(x+c_i)`；本仓
  `F_a = -G_12 psi_a * sum_i w_i c_i psi_b(x-c_i)`；周期对称模板下两者相等当且仅当 `G_12 = -g_ab`，故 `G_12 = -6`。
* 单因子消融：速度耦合（重心/自身/lbmpy 文档式）、受力（Guo vs tau-平移）、力离散（标准模板 vs 精确差分 EDM）、伪势（指数 vs 线性）。

## ③ 参数表（无量纲，lattice units）

| 量 | 值 | 来源 |
|---|---|---|
| 网格 / 边界 | 64x64，周期 | OSS `N`，`periodicity=True` |
| 步数 | 1000 | OSS `for i in range(1000)` |
| 松弛率 omega_a, omega_b | 1.0, 1.0 | OSS 常量 |
| 耦合 g_aa, g_ab=g_ba, g_bb | 0.0, 6.0, 0.0 | OSS 常量 |
| 本仓映射 G_12 | -6.0 | 符号推导（见 ②） |
| 伪势 rho0 | 1.0 | OSS `rho0` |
| 初值 rho_a / rho_b | 0.1->0.9 / 0.9->0.1 阶跃（界面在 x=32 与周期边界） | OSS `dh.fill` |
| 初速度 | 0 | OSS |
| 自发 u_max | 0.285 | 测得（无外加驱动，Re 非该算例控制量） |
| 密度下界 rho_min | 0.04745 > 0 | 测得 |

## ④ 对照表（ref vs got，误差打勾叉）

| 量 | OSS ref | TensorLBM | 判定 |
|---|---|---|---|
| 相对 L2（rho_a / rho_b） | 0 | 0.001068 / 0.001068 | PASS（≪15%） |
| 最大绝对误差 / 相对 Linf | - | 0.00115 / 0.001133 | PASS |
| 体相平台均值 | 0.5135 | 0.5135（相对 0.0001584） | PASS |
| 界面 10-90 厚度 | 3.084 | 3.081 | PASS |
| 界面过 0.5 位置（两处） | ['32', '63'] | ['32', '63'] | PASS |

臂级消融（一次只改一个因子）：

| 臂 | relL2(rho_a) | relL2(rho_b) | max abs err | 体相平台相对误差 | 发散步 | <15% |
|---|---|---|---|---|---|---|
| `lib_barycentric_guo_exp` | 0.001068 | 0.001068 | 0.00115 | 0.0001584 | - | PASS |
| `lib_self_guo_exp` | 0.05007 | 0.05007 | 0.07155 | 0.006649 | - | PASS |
| `lib_barycentric_shift_exp` | 0.3991 | 0.3991 | 0.7408 | 0.005861 | - | FAIL |
| `lib_barycentric_guo_edm` | nan | nan | nan | nan | 130 | FAIL |
| `lib_barycentric_guo_linear` | nan | nan | nan | nan | 30 | FAIL |
| `oss_literal_barycentric` | 0.03194 | 0.03194 | 0.07399 | 0.01075 | - | PASS |
| `oss_lbmpy_doc_shift` | nan | nan | nan | nan | 50 | FAIL |
| `oss_self_corrected` | 0.05007 | 0.05007 | 0.07155 | 0.006649 | - | PASS |

## ⑤ 收敛性 / 参数敏感性（四点）

1. **时间对齐**（步数扫描，主臂）：误差在 OSS 步数处取极小；500 步时已达 0.03294，说明对比对瞬态时刻极敏感，不是靠平滑偶然对上。

| steps | relL2(rho_a) | max abs err |
|---|---|---|
| 500 | 0.03294 | 0.04433 |
| 900 | 0.00674 | 0.009808 |
| 1000 | 0.001068 | 0.00115 |
| 1100 | 0.00448 | 0.006814 |
| 1500 | 0.02131 | 0.03145 |

2. **分辨率**（N=128 块平均回 64 点）：0.1011，仍在 15% 内但比 N=64 大两个数量级 —— 该参考解是**特定分辨率的解快照**，对标通过 != 物理量网格收敛。

| N | relL2(rho_a) | max abs err |
|---|---|---|
| 64 | 0.001068 | 0.00115 |
| 128 | 0.1011 | 0.2141 |

3. **松弛时间 tau**（参考绑死 tau=1）：tau=1 处最小，两侧单调变差（0.0261 / 0.01706）—— 对比对离散参数敏感。

| tau | relL2(rho_a) | max abs err |
|---|---|---|
| 0.80 | 0.0261 | 0.03739 |
| 1.00 | 0.001068 | 0.00115 |
| 1.20 | 0.01706 | 0.02536 |

4. **算术精度** float32 vs float64：0.001064 vs 0.001068 —— 残余误差**不是**浮点噪声（改精度不改结论），而是速度耦合约定之差。

| dtype | relL2(rho_a) | max abs err |
|---|---|---|
| float32 | 0.001064 | 0.001145 |
| float64 | 0.001068 | 0.00115 |

## ⑥ 失败模式与工件

| 现象 | 触发 | 证据（results.json 臂名） |
|---|---|---|
| 残差 0.001068 -> 0.03194 | 重心速度**同时**塞进 Guo 源项（OSS 代码字面读法） | `oss_literal_barycentric` |
| 50 步内发散（rho_min=0.000251） | 平衡态重心+F/2rho 平移、源项用重心速度（lbmpy 文档式读法） | `oss_lbmpy_doc_shift` |
| 30 步内发散，rho 跑到 -1.579e+07 | 线性伪势 psi=rho 配 g=6：psi 无上界，强耦合直接崩 | `lib_barycentric_guo_linear` |
| 130 步发散 | EDM（精确差分）受力 + Guo + 重心速度，本耦合强度不稳定 | `lib_barycentric_guo_edm` |
| relL2 = 0.3991（超差） | 历史 tau-平移受力（非 Guo） | `lib_barycentric_shift_exp` |

工件：`artifacts/xval_lbm_shan_chen_mcmp/`（`oss_ref_*.json` 带 provenance、`results.json`、`predeclared.json`、`friction.json`、`fig1..fig5*.png`）。

## 本仓 vs OSS 差异清单（做法 / 参数 / 边界条件）

| 方面 | lssfau/lbmpy (OSS) | TensorLBM | 影响（本算例实测） |
|---|---|---|---|
| 伪势 | 指数式 psi=rho0*(1-exp(-rho/rho0)) | 默认线性 psi=rho；新增 `psi=` 可传指数式 | 线性 psi 在 g=6 发散（决定性差异） |
| 平衡态速度 | 重心速度 u_bary | `u_eq="mixture"` = 重心；`"self"` = 组分速度 | self 式 -> 0.05007 |
| 受力离散 | Guo（(1-omega/2) 乘子，源项取输入速度） | `force_scheme="guo"`（平衡态与源项都用 u_s+F_s/(2 rho_s)）；另有 tau-平移、EDM | Guo 0.001068 vs tau-平移 0.3991 |
| 力模板 | sum_i w_i c_i psi(x+c_i) | sum_i w_i c_i psi(x-c_i)，周期下等价 | 需 G_12 = -g_ab（符号映射） |
| 组分自作用 | g_aa=g_bb=0 | 该接口只有交叉耦合 G_12 | 无差异（本算例自作用为零） |
| 时间步组织 | 力核 -> 碰撞核 -> stream 核（多个 kernel） | 单次 collide 内合成后 stream | 顺序相同（force -> collide -> stream） |
| 精度 | 单精度编译核 | 默认 float32，可按需 float64 | 0.001064 vs 0.001068（无实质差异） |
| 边界条件 | 仅周期 | 周期 + 可选固体掩膜 / 中性润湿壁（`wall_mode`）/ EDM 力 | 本算例未启用 |

**移植取舍**：①只移植“可选伪势”这一必要能力（`psi=` 默认关闭、历史结果逐位不变），不照搬 OSS
“重心速度同时进源项”的写法 —— 同等耦合下它把误差放大 30 倍；②保留本仓“组分速度 + F/(2 rho) 进 Guo 源项”
的约定，因为它恰好最接近 OSS 参考解（0.001068）且在本耦合强度下稳定；③EDM / tau-平移两条既有分支
在本算例不适用（发散 / 0.3991），不进默认路径，保留为可选并标注适用范围。

## 复现与环境

```
python xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/fetch_oss_ref.py
OMP_NUM_THREADS=1 PYTHONPATH=src python xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/run_xval.py
python xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/make_figures.py
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_xval_shan_chen_mcmp_barycentric_guo.py -q
```
python 3.11.6 / torch 2.7.1 / numpy 2.4.6 / tensorlbm 0.3.0 / 单线程 CPU（Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38）。
全程确定性（无 RNG），总机时约 6 min。

## 是否 < 3%（逐项误差判定）

| 关键量 | 基准 | 误差 | <3% |
|---|---|---|---|
| rho_a 剖面（相对 L2） | OSS 参考解 | 0.001068 | 是 |
| rho_b 剖面（相对 L2） | OSS 参考解 | 0.001068 | 是 |
| 最大绝对误差（相对 Linf） | OSS 参考解 | 0.001133 | 是 |
| 体相平台均值 | OSS 参考解 | 0.0001584 | 是 |
| 界面 10-90 厚度 | OSS 参考解 | 0.001035 | 是 |

结论：五项关键量相对误差全部 < 3%（最大项 0.001133），远超 owner 的 15% 门槛；本仓重心-Guo 路径可视为与该 OSS
实现等价（残差仅为其速度耦合约定之差，量级 0.1%）。**未覆盖**：G / rho0 / omega / EOS 的完整参数空间未扫；
N=128 与 N=64 自解差 10%，即本对标只证明**实现等价**，不证明网格收敛。

生态摩擦（详见 `artifacts/xval_lbm_shan_chen_mcmp/friction.json`）：knowledge_base 搜索引擎 8021 拒连（engine_offline）；
bootstrap 里写的三个 search 路径 404（真实路径是 /tensor/knowledge/search、/tensor/negative-knowledge/search）；
可编辑安装指向另一个 worktree（/root/tp-agent/work/m17918），跑本仓代码必须显式 PYTHONPATH=src；
raw.githubusercontent.com 不可达（走 api.github.com 取源）；torch 默认线程在 128 核高负载机上把 10 步从 0.10 s 拖到 1.4 s（全部单线程）。
