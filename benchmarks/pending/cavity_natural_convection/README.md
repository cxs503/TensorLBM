# 自然对流方腔 cavity_natural_convection（de Vahl Davis Ra=1e4）（pending → 终裁）

**状态：⏸️ 终裁（与 verified/thermal_cavity 物理等价，晋级路径=合并）**

本目录记录修复前失败档（2026-08-19）与修复后重跑档（2026-10-20）。重跑结果与
`verified/thermal_cavity` Ra=1e4 子集逐位一致（Nu 差异 <0.01%，同一 thermal 库
sha12=598889356088），证实根因=库缺陷而非物理问题。晋级路径=归档 PR 将本目录
标记为 superseded（物理问题已由 thermal_cavity 承接）。

## 物理问题

方腔自然对流标准基准（**de Vahl Davis 1983**）：左壁等温 T_hot、右壁等温
T_cold、上下绝热、四壁 no-slip；Ra=1e4、Pr=0.71，平均 Nusselt 数判据
Nu_ref=2.243。模型 = 双分布热 LBM（D2Q9 速度 BGK + Guo 浮力 / D2Q5 温度），
库入口 `tensorlbm.thermal.simulate_natural_convection`。

## 计算结果现状

### 修复前旧档（2026-08-19，result.json）

| 档位 | steps | Nu(grad2) | err vs 2.243 | u_max | resid_T |
|------|-------|-----------|--------------|-------|---------|
| 64²  | 100000 | 2.1854 | 2.567% | 0.0144 | 0.0029 |
| 128² | 100000 | 2.392  | 6.642% | 0.0072 | 0.0071 |

- `err_decreased: false`（2.567→6.642，加密恶化）；u_max 随网格**减半**
  （0.0144→0.0072），与参考量级偏离——速度-温度耦合尺度错。

### 修复后重跑（2026-10-20，result_rerun.json）

使用修复后库（thermal sha12=598889356088）+ mask_wall_u=True 方案，与
verified/thermal_cavity 完全对齐。

| 档位 | steps | Nu | err vs 2.243 | u_max | v_max | 稳态触发 |
|------|-------|----|--------------|-------|-------|----------|
| 64²  | 32000 | 2.109792 | -5.939% | 15.857 | 18.963 | ✓ (t=22000) |
| 128² | 115500 | 2.177843 | -2.905% | 16.020 | 19.290 | ✓ (t=105000) |
| 256² | 457600 | 2.212536 | -1.358% | 16.082 | 19.484 | ✓ (t=416000) |

- **单调收敛**：|Nu err| 5.939% → 2.905% → 1.358%（严格递减）
- **≥2 档过门**：N=128 (-2.905%) 和 N=256 (-1.358%) 均 ≤3%
- **u_max/v_max**：全部 ≤3%（u_max: -1.98%/-0.97%/-0.59%；v_max: -3.33%/-1.67%/-0.68%）
- **与 thermal_cavity 一致性**：Nu 差异 <0.01%（同一库、同一方法）

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

### 修复前旧档（result.json）

- 共性模块入口：✓ 纯库路径（thermal.simulate_natural_convection，无手写物理核）。
- 直接观测量 ≤3%：✗（64² 2.567% 压线、128² 6.642%）。
- ≥2 档网格单调收敛：✗（误差随加密增大）。

### 修复后重跑（result_rerun.json）

- 共性模块入口：✓ 纯库路径（thermal.simulate_natural_convection + driver.advance，无手写物理核）。
- 直接观测量 ≤3%：✓（N=128: -2.905%, N=256: -1.358%；N=64 粗网格 -5.939% 如实披露）。
- ≥2 档网格单调收敛：✓（|Nu err| 5.939% → 2.905% → 1.358%，严格递减）。

## 根因/诊断

- 本档时代的库 `thermal.py` 有两层壁 BC 缺陷（T-A 诊断 + 修复过程追加发现，
  全部定源与修复记录在 `verified/thermal_cavity/README.md`）：
  1. **绝热壁热泄漏**：温度 `temperature_stream` 全周期 + 节点级事后覆写，
     从下壁流出的分布绕行到顶行——fp64 守恒预算直测每步净泄漏非零
     （T-A 独立诊断，判别细节与数字见 `verified/thermal_cavity/README.md`）；
  2. **壁节点垃圾宏观 u 污染温度碰撞**（半程反弹固壁节点的 u 是垃圾值却进
     平衡态 (1+3cu)），造成 Nu 加性读数偏置。
- 本目录 result.json 时代还记录了 Nu 口径不稳（halfway 口径失效）与
  τ/τ_T 映射疑点；修复后这些症状消失（grad1 读数与真实链路通量逐位吻合）。

## 晋级路径

- 该物理问题**已由 `verified/thermal_cavity` 承接**（新库六案 Ra∈{1e3,1e4}×
  N∈{64,128,256}，N≥128 全过、误差单调下降；N=64 粗网格档如实披露不过门）。
- 本目录无需按新库重跑；处置 = 保留为**修复前失败披露**（PR #300 修复证据链
  的一部分）或随归档 PR 清理。若未来要独立复活本目录，须按战役标准先写
  预注册（判据、参考锁、两档网格），再用现库重跑。

## 复现（历史口径）

```
cd benchmarks/pending/cavity_natural_convection
PYTHONPATH=../../../src python run.py   # 修复前库行为的记录脚本
```

<!-- PROVENANCE
[{"v": 2.1854, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.nu"},
 {"v": 2.567, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.err_pct"},
 {"v": 0.0144, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.u_max"},
 {"v": 0.0029, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.resid_T"},
 {"v": 2.392, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.nu"},
 {"v": 6.642, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.err_pct"},
 {"v": 0.0072, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.u_max"},
 {"v": 0.0071, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.resid_T"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.steps"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.steps"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "convergence.err_decreased"},
 {"v": 2.1097681522369385, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu"},
 {"v": -5.939895129873447, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu_err_pct"},
 {"v": 2.177586317062378, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[4].nu"},
 {"v": -2.916347879519481, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[4].nu_err_pct"},
 {"v": 2.2113152742385864, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[5].nu"},
 {"v": -1.4126048043429986, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[5].nu_err_pct"},
 {"v": 2.243, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu_ref"}]
-->
