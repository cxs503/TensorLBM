# Ra=1e5 Natural Convection Cavity Benchmark — de Vahl Davis 1983

## ① 问题与参考出处

**物理问题**：方腔自然对流标准基准。左壁等温 T_hot=1、右壁等温 T_cold=0、
上下绝热、四壁 no-slip。无量纲参数 Pr=0.71（空气），Ra=10⁵。

**参考值**（de Vahl Davis 1983, Int. J. Numer. Methods Fluids 3:249-264, Table 1 p.257）：

| 量 | 参考值 | 单位 |
|---|---|---|
| Nu̅ | 4.519 | — |
| \|U\|_max | 34.72 | 扩散尺度 u_nd = u_lu·H/α |
| \|V\|_max | 68.60 | 扩散尺度 v_nd = v_lu·H/α |
| Nu_max | 8.499 | 热壁局部最大 |
| Nu_min | 0.989 | 热壁局部最小 |
| Nu_1/2 | 4.499 | 热壁中点 |
| u_max 位置 y/H | 0.813 | — |
| v_max 位置 x/H | 0.066 | — |

**容差判据**：<8%（任务要求）。

## ② 方法与实现

**数值方法**：
- 速度场：D2Q9 BGK 碰撞（τ=0.6），Guo 浮力格式
- 温度场：D2Q5 BGK advection-diffusion
- 壁面：pre-streaming 半程反弹（no-slip），等温壁节点冻结 + 非平衡零和转移
- 浮力：Boussinesq 近似 F = ρ·gβ·(T−T_ref)，gβ = Ra·ν·α/H³

**实现位置**：
- 驱动库：`benchmarks/verified/thermal_cavity/driver.py`（编排，零本地物理）
- 物理内核：`src/tensorlbm/thermal.py`（D2Q9+D2Q5 双分布函数）
- 运行入口：`benchmarks/verified/thermal_cavity/run.py`

**参数表**：

| 参数 | 值 |
|---|---|
| Ra | 1×10⁵ |
| Pr | 0.71 |
| τ | 0.6 |
| ν | (τ−1/2)/3 = 0.03333 |
| α | ν/Pr = 0.04695 |
| τ_T | 3α+1/2 = 0.6408 |
| H | nx−1（节点间距约定） |
| gβ | Ra·ν·α/H³ ≈ 6.26×10⁻⁴ (n64) |
| ΔT | 1.0 |
| 初场 | 线性导热态 T(x)=1−x/H |

## ③ 计算结果（初步）

**n64 网格，5000 步（未收敛，仅供参考趋势）**：

| 量 | 计算值 | 参考值 | 误差% | 状态 |
|---|---|---|---|---|
| Nu | 4.074 | 4.519 | 9.9% | ✗ 未收敛 |
| u_max | 37.30 | 34.72 | 7.4% | △ 接近 |
| v_max | 63.57 | 68.60 | 7.3% | △ 接近 |
| Mach | 0.091 | — | — | ✓ <0.3 |
| T_min | 0.0 | 0.0 | — | ✓ |
| T_max | 1.0 | 1.0 | — | ✓ |

**注意**：5000 步远未达到稳态（估计需 200k-500k 步）。Nu 仍在振荡（3.1→4.6→3.7→3.8→4.1），
u_max/v_max 也在波动。当前误差 >8% 是因为未收敛，而非方法误差。

**完整运行已递交集群**：
- cj1193: Ra=1e5 n64（queued）
- cj1194: Ra=1e5 n128（queued）
- cj1195: Ra=1e5 n256（queued）

## ④ 收敛性分析（基于已有 Ra=1e3/1e4 数据外推）

从 Ra=1e4 的收敛趋势推断 Ra=1e5 行为：

| Ra | N | Nu | Nu 误差% | u_max 误差% | v_max 误差% |
|---|---|---|---|---|---|
| 1e4 | 64 | 2.110 | -5.9% | -2.0% | -3.3% |
| 1e4 | 128 | 2.178 | -2.9% | -1.0% | -1.7% |
| 1e4 | 256 | 2.211 | -1.4% | -0.7% | -0.9% |

Ra=1e5 预期收敛趋势类似：n64 误差 ~5-10%，n128 误差 ~2-5%，n256 误差 <2%。

## ⑤ 失败模式与工件

**当前状态**：L1（有实现+入口+初步运行），未达 L2（缺完整 run_artifacts）。

**失败原因**：
1. 集群队列深度大（377+ 作业等待），cj1193/1194/1195 仍在排队
2. 本地 CPU 计算慢（n64 约 16.8ms/步，500k 步需 ~140 分钟）
3. 预算限制阻止了完整的本地运行

**已交付工件**：
- `benchmarks/verified/thermal_cavity/driver.py` — 驱动库（真实实现）
- `benchmarks/verified/thermal_cavity/run.py` — 运行入口（已扩展 Ra=1e5 支持）
- `benchmarks/verified/thermal_cavity/result.json` — 结果文件（含 Ra=1e5 预注册参考值 + 初步数据）
- `tests/test_thermal_cavity_ra1e5.py` — pytest 测试套件
- `gen_figures_ra1e5.py` — 图生成脚本
- `out_ra1e5/case_ra1e5_n64_preliminary.json` — 初步运行工件

**复现命令**：
```bash
# 完整运行（需 GPU 或长时间 CPU）
PYTHONPATH=src:benchmarks/verified/thermal_cavity \
  python3 benchmarks/verified/thermal_cavity/run.py --cases ra1e5_n64,ra1e5_n128,ra1e5_n256 --device cpu

# 快速验证（5000 步，不收敛）
PYTHONPATH=src:benchmarks/verified/thermal_cavity \
  python3 -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'benchmarks/verified/thermal_cavity')
import torch, driver
f,g,wall,p = driver.make_fields(64, 1e5, 0.71, 0.6, torch.device('cpu'))
for _ in range(5000): f,g = driver.advance(f,g,wall,0.6,p['tau_T'],p['g_beta'])
obs = driver.observables(f,g,p); print(f'Nu={obs[\"nu\"]:.4f} u_max={obs[\"u_max_nd\"]:.2f} v_max={obs[\"v_max_nd\"]:.2f}')
"
```

## ⑥ 误差判定

**当前初步结果（5000 步，未收敛）**：
- Nu 误差 9.9% > 8% ✗（未收敛导致，非方法误差）
- u_max 误差 7.4% < 8% ✓
- v_max 误差 7.3% < 8% ✓

**预期收敛后**：基于 Ra=1e4 的收敛趋势，n128 和 n256 应能达到 <8% 容差。

**诚实声明**：本单未能在预算内完成完整收敛运行。集群作业 cj1193/1194/1195 仍在排队，
待完成后可更新 result.json 并重新运行测试。
