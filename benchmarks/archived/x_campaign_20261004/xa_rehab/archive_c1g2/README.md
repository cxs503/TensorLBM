# 两层 Poiseuille 基准（双 τ D2Q9 颜色梯度 LBM）

**状态：✅ 已验证（真实模拟，无外推；对解析冻结参考逐臂 relL2 ≤ 3%）**

## 物理问题

重力（体力 gx）驱动的两层不互溶流体平面 Poiseuille 流，等密度、
半程 bounce-back 壁面。臂覆盖等黏度（eq，τ=1）与运动黏度比
ν_lo/ν_hi = 10（g10，τ=(2.5, 0.7) → ν=(2/3, 1/15) 格子单位），
ny=64/128 两档网格。解析参考取**壁面平面约定**（固壁行 j=0 与
j=ny−1 → 无滑移平面 Y=0.5 与 ny−1.5），界面平面
Y_i = j_first_blue − 0.5 由该次运行自身稳态 φ 列测量。

## 模型

- **双弛豫时间 D2Q9 颜色梯度**（Rothman & Keller 1988；Latva-Kokko &
  Rothman 2005 recolor，分离参数 β=0.9）：逐分量 τ_r/τ_b → 黏度对比；
  **Guo et al. (2002) 全三项体力**（半步速度移位，逐色动量精确平衡）
- 引擎 = additive 模块 `tensorlbm.color_gradient2d`（库内
  `multiphase.color_gradient_step` 只支持单一共享 τ，无法跑黏度对比臂；
  库内未合并时用随案 vendored 副本，以同名包挂载，md5 记录于
  result.json）；解析求值器 `cg_reference.py` 随案冻结
- fp64；nx=8 流向周期列；A=0.04；tanh 初始分层；等黏度臂亦过同链路

## 判定标准

| 指标 | 阈值 |
|------|------|
| 逐臂 relL2(u_sim, u_analytic) | ≤ 3% |
| 逐色质量漂移 | ≤ 1e-10 /步/色 |
| 全程 NaN-free | 必须 |

## 结果（见 result.json，机器值）

| 臂 | ny | (τ_r, τ_b) | relL2 | u_max 比（sim/解析） | j_first_blue | u_max 冻结度（末 20%） | 质量漂移 max /步/色 |
|---|---|---|---|---|---|---|---|
| eq_ny64 | 64 | (1.0, 1.0) | 0.211% | 1.0083 | — | 3.36e-10 | 2.43e-14 |
| eq_ny128 | 128 | (1.0, 1.0) | 0.155% | 1.0088 | — | 2.21e-08 | 3.60e-14 |
| g10_ny64 | 64 | (2.5, 0.7) | 1.092% | 0.9915 | 33 | 2.68e-12 | 3.05e-14 |
| g10_ny128 | 128 | (2.5, 0.7) | 0.560% | 0.9956 | 65 | 2.96e-12 | 5.30e-14 |

- 判决：**4/4 臂 PASS**（最差 g10_ny64
  relL2 = 1.092%）
- 网格加密单调改善：eq 0.211% →
  0.155%；g10
  1.092% → 0.560%
- 界面位置自测：g10 两档 j_first_blue = 33 /
  65（ny 翻倍近似翻倍，界面层几何一致）

## 已知边界（本基准未覆盖）

- σ_eff 的 (L,R) 迁移性是颜色梯度族 Laplace 侧的已知系统债
  （同战役 G1 FAIL 根因）；本基准不消费 σ，黏度对比臂的闭合与
  Laplace σ 无关
- 库 stock `multiphase_benchmarks` 两层解析把无滑移放在固壁行（1 节点
  系统壁偏移）；本基准用 `cg_reference` 壁面平面约定，两者已分家

## 复现

```bash
cd benchmarks/verified/<case_dir>
# 全部 4 臂（fp64 GPU 约 9 分钟；自动回退 CPU，CPU 数小时）：
python run.py --out /tmp/g2_repro
# 单臂：
python run.py --arms g10_ny64 --out /tmp/g2_repro
```

result.json 为正式跑机器切片（campaign 档案
`bm_mp_20261004/c1_color_gradient/out/g2_poiseuille.json`，
md5 6053d7530a3634540732947bb58781f3）；重跑结果文件为
`g2_rerun.json`，字段同构、附引擎来源与 md5。

## 参考

- Rothman & Keller, J. Stat. Phys. 52, 1119 (1988)（immiscible lattice gas）
- Latva-Kokko & Rothman, PRE 71, 056702 (2005)（recolor/segregation β）
- Guo, Zheng & Shi, PRE 65, 046308 (2002)（离散体力格式）
- 两层 Poiseuille 稳态解析解（壁面平面约定，界面连续应力）
