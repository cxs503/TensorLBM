# B6: DARPA SUBOFF bare hull — Re=1000 阻力 benchmark

## 物理问题
SUBOFF 裸艇体（DARPA SUBOFF bare hull，L=4.356 m，D=0.508 m，L/D=8.57）零攻角均匀来流。
Re = u·L/ν = 1000（层流）。测量总阻力系数 Ct（摩擦+压差）。

## 参考值
- **主参考（本库 Re=1000 基准族口径）**: Blasius 层流平板摩擦系数
  `Cf = 1.328/√Re = 0.0420`（Re=1000），湿面积归一化 S=π·D·L，
  dpS = 0.5·u²·π·D·L。历史同口径结果: Cd_tot=0.0436（+3.8% CUDA / +3.6% SDAA，L=80, 5000 步）。
- **口径说明**: benchmarks/TODO.md 中 B6 的 "Ct=0.004 (实验)" 实为 AFF-8 全尺寸
  Re=2e6 总阻力系数（见 src/tensorlbm/suboff_reference_data.py），不适用于 Re=1000。
  本 benchmark 统一使用 Blasius 0.0420 口径（与全部历史 Re=1000 运行一致）。

## 配置（共性模块入口: GeneralSimEngine / PARAMETRIC_SUBOFF）
- 几何: `suboff_length=4.356 m, suboff_radius=0.254 m`（掩码: suboff_cad.build_suboff_mask, bare_hull）
- 物理: Re=1000（u=1e-3 m/s, ν=4.356e-6 m²/s, L_ref=4.356 m）→ u_lb=0.05, τ=0.512
- 网格: 分辨率 = 每艇长格数 L_cells=80/160；域 6L（流向 1L 上游 + 4L 下游），侧向 2L+D
- 碰撞: MRT（Re<1000 自动 MRT；Re=1000 显式 MRT，无 Smagorinsky 人工粘性）
- 壁面: half-way bounce-back（Re<10000 自动 BB）
- 求解: lbm_step_correct + far_field_bc_3d（x- 入流自由流、x+ 零梯度、侧向自由流）+ 质量修正 200 步
- 力: drag_pressure_integration（**extrap='none'**，p0='near_wall'）+ drag_friction_integration（standard）
- 归一化: 湿面积 πDL（与历史族一致；引擎默认迎风面积 πR² 已在后处理重标定，见 G1）
- 步数: ≥20000；力采样间隔 10 步；末窗 1000 样本平均

## 结果
（见 result.json）

## 误差分析
（见 result.json / 下文）

## 运行命令
```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
/home/wxsc/anaconda3/envs/ftw-env/bin/python benchmarks/pending/suboff_re1000/run.py \
  --resolution 80 --steps 20000 --device cuda:2 --collision mrt
```

## 状态
- [ ] 误差 ≤1%（真实模拟、无外推）→ 已保存 benchmarks/verified/suboff_re1000/
- [ ] 未达标 → 详细分析 + 下一步（见 result.json / /tmp/suboff_gap.md）
