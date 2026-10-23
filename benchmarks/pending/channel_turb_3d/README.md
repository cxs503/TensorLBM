# 3D 槽道湍流 channel_turb（终裁）

**状态：⏸️ TERMINAL（根因：计算域尺寸不足 + 统计样本不充分）**

## 物理问题

3D 周期槽道流，恒定体积力驱动，Re_τ=180。目标：u+ 剖面 vs DNS（Moser, Kim & Mansour 1999, Re_τ=178.12）在 y+∈[30,100] 窗口 RMS 误差 ≤3%。

**与 2D 槽道的本质区别**：3D 有涡拉伸机制，转捩可自持（2D 无线性转捩路径，Re_max≈4200 < 5772）。

## 共性模块

- 库原语：`tensorlbm.wall_resolved_channel3d` 的 `WallResolvedChannel3DConfig` + `run_wall_resolved_channel3d`
- 格子：D3Q19，碰撞 = Natural KBC（`collision_d3q19_advanced.natural_kbc`）
- 边界：上下壁 post-streaming bounce-back（`boundaries3d.bounce_back_cells_3d`），流向/展向周期
- 驱动：Guo 体型体力 a_x = 2·u_τ²/H（`wall_model.guo_body_force_d3q19`）
- 初始化：谱无散度扰动（`spectral_solenoidal`），模拟转捩触发

## 计算结果（三档网格，GPU5 实测）

| 网格 | Lx/h | Lz/h | dy+ | 涡旋数 | u_τ 误差 | RMS y+∈[30,100] | 单调 |
|------|------|------|-----|--------|----------|-----------------|------|
| L1: 64×32×32 | 4.27 | 2.13 | 12.0 | 3.00 | -20.55% | **20.43%** | - |
| L2: 128×48×64 | 5.57 | 2.78 | 7.83 | 1.96 | -14.26% | **11.67%** | ✓ |
| L3: 256×64×128 | 8.26 | 4.13 | 5.81 | 1.45 | -7.64% | **5.33%** | ✓ |

**DNS 参考域**：Lx/h=12.6, Lz/h=4.2（Moser et al. 1999）

**关键发现**：
1. ✓ 3D 槽道成功转捩（Reynolds 应力 <uv>+ 非零，L3 峰值 ~-0.5）
2. ✓ 三档严格单调收敛（20.43% → 11.67% → 5.33%）
3. ✗ 最佳结果 5.33% > 3% 门限
4. ✗ u_τ 系统性偏低（-7.64%），壁面剪切应力不足

## 根因诊断

### 1. 计算域尺寸不足（主因）
- L3 域 Lx/h=8.26 仅为 DNS 的 65%（12.6），Lz/h=4.13 接近但略小
- 湍流大尺度结构（超大尺度运动 VLSM）需要 Lx/h ≥ 12 才能充分发展
- 小域导致流向相关长度被截断，雷诺应力偏低，u_τ 系统性偏低

### 2. 统计样本不充分
- L3 仅 1.45 个涡旋周转时间（eddy turnover），远低于湍流统计所需的 2-3 个
- 平均窗 30000 步（warmup 20000 后），采样间隔 10，共 3000 样本
- 统计噪声导致 u+ 剖面波动，RMS 误差偏高

### 3. 壁面模型限制
- Post-streaming bounce-back 为一阶精度，壁面剪切应力偏低
- 无壁面偏移校正（wall_offset），有效壁面位置不确定
- 高分辨率壁面模型（Spalding/壁面函数）未启用

### 4. 格子分辨率
- L3 dy+=5.81，近壁分辨率不足（DNS 通常 dy+≈1 在近壁）
- 无壁面加密，对数律区域 y+∈[30,100] 仅 ~12 个网格点

## 四门判决

| 门 | 标准 | 结果 | 证据 |
|----|------|------|------|
| G1 共性模块 | `from tensorlbm` 库路径 | ✅ PASS | `run_probe.py` 调用 `wall_resolved_channel3d` |
| G2 直接观测量 | RMS ≤3% | ❌ FAIL | 最佳 5.33%（L3），超差 2.33% |
| G3 单调收敛 | ≥2 档严格单调 | ✅ PASS | 20.43% → 11.67% → 5.33% |
| G4 参考先锁 | prereg append-only | ✅ PASS | `prereg.json` sha256 冻结 |

**判决：TERMINAL**（G2 失败，最佳结果 5.33% > 3%）

## 晋级路径（需 owner 决策）

### 路径 A：增大计算域 + 延长统计（推荐）
- L4: 400×64×132（Lx/h=12.9, Lz/h=4.26，匹配 DNS）
- 步数 100000（warmup 30000 + 平均 70000），涡旋数 ~3.5
- 预算：~8 GPU·h（GPU5）
- 预期：RMS 降至 2-3%，u_τ 误差 <3%

### 路径 B：壁面模型升级
- 启用 Spalding 壁面模型（`spalding_wall_model`）
- 壁面偏移校正（wall_offset=0.5）
- 预算：~2 GPU·h（调参 + 重跑 L3）
- 预期：u_τ 误差改善 2-3%，但域尺寸限制仍在

### 路径 C：近壁网格加密
- 双曲正切拉伸（tanh stretching），近壁 dy+≈1
- 预算：~3 GPU·h（网格生成 + 重跑）
- 预期：对数律区域分辨率提升，但计算量增加

### 路径 D：维持 TERMINAL
- 记录负结果，3D 槽道转捩成功但精度不足
- 预算：0 GPU·h
- 价值：证明 3D 格式无系统性偏差，域尺寸是主因

## 复现

```bash
cd benchmarks/pending/channel_turb_3d/runs_3d
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run_probe.py --level L1
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run_probe.py --level L2
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run_probe.py --level L3
/nfs/wangxi/venvs/tensorlbm/bin/python postprocess.py
```

原始数据：`runs_3d/{L1,L2,L3}_result.json`；DNS 参考：`dns_ref/mkm180.csv`。

<!-- PROVENANCE
[{"v": 20.43434670057914, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[0].rms_y30_100_pct"},
 {"v": 11.665111463802477, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[1].rms_y30_100_pct"},
 {"v": 5.330965065942242, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[2].rms_y30_100_pct"},
 {"v": -20.55, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[0].u_tau_err_pct"},
 {"v": -14.26, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[1].u_tau_err_pct"},
 {"v": -7.64, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[2].u_tau_err_pct"},
 {"v": "4228e65a3c90b2704a06ec69668ef4115e838434ab564d863cc6f521ddbd8ccb", "f": "benchmarks/pending/channel_turb_3d/runs_3d/prereg.json", "k": "sha256"}]
-->
