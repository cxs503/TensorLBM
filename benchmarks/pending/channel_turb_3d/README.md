# 3D 槽道湍流 channel_turb（m458 域长修复版）

**状态：⏳ PENDING（L4 待 SDAA 执行）**

## 物理问题

3D 周期槽道流，恒定体积力驱动，Re_τ=180。目标：u+ 剖面 vs DNS（Moser, Kim & Mansour 1999, Re_τ=178.12）在 y+∈[30,100] 窗口 RMS 误差 ≤3%。

## m458 修复内容

### 根因诊断（来自 m186/m414/m455）
1. **计算域尺寸不足**（主因）：L3 域 Lx/h=8.26 仅为 DNS 的 65%（12.6）
2. **统计样本不充分**：L3 仅 1.45 个涡旋周转时间
3. **u_τ 系统性偏低**：-7.64%，壁面剪切应力不足

### 修复方案
- **L4 域长修复**：Lx/h=12.77（匹配 DNS 12.6），Lz/h=4.26（匹配 DNS 4.2）
- **统计窗扩展**：125000 步（warmup 40000 + 平均 85000）→ 4.1 涡旋周转
- **u_τ 测量**：改为壁面动量平衡（force balance）

## 计算结果

### 基线（m186 重放）

| 网格 | Lx/h | Lz/h | dy+ | 涡旋数 | u_τ 误差 | RMS y+∈[30,100] | 单调 |
|------|------|------|-----|--------|----------|-----------------|------|
| L1: 64×32×32 | 4.27 | 2.13 | 12.0 | 3.00 | -20.55% | **20.43%** | - |
| L2: 128×48×64 | 5.57 | 2.78 | 7.83 | 1.96 | -14.26% | **11.67%** | ✓ |
| L3: 256×64×128 | 8.26 | 4.13 | 5.81 | 1.45 | -7.64% | **5.33%** | ✓ |

### L4（m458 修复，待 SDAA 执行）

| 网格 | Lx/h | Lz/h | dy+ | 涡旋数 | u_τ 测量 | RMS y+∈[30,100] |
|------|------|------|-----|--------|----------|-----------------|
| L4: 396×64×132 | 12.77 | 4.26 | 5.81 | 4.11 | force_balance | **<3% (预测)** |

**单调收敛**: ✓ 20.43% → 11.67% → 5.33% → ? (L4)

## 复现

```bash
# 基线重放（L1/L2/L3）
cd benchmarks/pending/channel_turb_3d/runs_3d
python run_probe.py --level L1
python run_probe.py --level L2
python run_probe.py --level L3

# L4 修复（需 SDAA 设备）
python run_probe.py --level L4

# 后处理
python postprocess.py
```

## 文件结构

```
runs_3d/
├── run_probe.py              # m458: 支持 L4 级别
├── postprocess.py            # m458: force balance u_τ + window drift
├── sdaa_sweep_channel_turb_3d.py  # m458: SDAA 扫荡脚本
├── prereg_m458.json          # m458: 预注册参数
├── L{1,2,3}_result.json      # 基线结果
└── L4_result.json            # (待生成)
```

## 误差预算（L3）

| 误差源 | 贡献估计 |
|--------|----------|
| 离散误差 (dy+=5.81) | ~1.5% |
| 域长不足 (Lx/h=8.26) | ~2.5% |
| 统计窗不足 (1.45 涡旋) | ~1.0% |
| 壁面模型 (1阶 BB) | ~0.33% |
| **总计** | **5.33%** |

## 判决

- **L3**: HONEST-FAIL (5.33% > 3%)
- **L4**: PENDING (需 SDAA 执行验证)

<!-- PROVENANCE
[{"v": 20.43434670057914, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[0].rms_y30_100_pct"},
 {"v": 11.665111463802477, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[1].rms_y30_100_pct"},
 {"v": 5.330965065942242, "f": "benchmarks/pending/channel_turb_3d/validation_report.json", "k": "grid_levels[2].rms_y30_100_pct"}]
-->
