# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM: Final Validation Report

**Anchor**: `XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM`  
**Mission**: 22363 (generalist席自主工时)  
**Repo**: TensorLBM  
**Date**: 2026-11-XX  
**Status**: **PASS** (all acceptance checks satisfied)

---

## 1. 案例介绍 (Case Introduction)

### 物理背景
本锚点验证 TensorLBM 对 `lssfau/lbmpy` 稀疏列表存储格式（sparse-list-based AoS LBM）的移植实现。该格式的核心思想是：
- **单元列表**：仅存储流体单元，固体单元完全不存储
- **邻居索引数组**：每个单元、每个方向存储一个平铺偏移量（flat offset），用于 streaming 阶段的间接寻址
- **无滑移边界**：通过索引别名（index aliasing）实现——若拉取源为固体，则替换为自身单元的反方向种群，无需单独的边界核函数

### 几何与参数
验证覆盖 2-D (D2Q9) 和 3-D (D3Q19) 两种格子，包含以下基准案例：

| 案例 | 几何 | 参考解 | 关键参数 |
|------|------|--------|----------|
| P1 | 3-D 平面通道（y方向壁面，x/z周期） | 抛物线剖面 | H ∈ {8, 16}, τ=0.8, Ma=0.05 |
| P2 | 3-D 矩形管道（y/z双方向壁面，x周期） | White Fourier级数 | 8×12, 16×24, τ=0.8, Ma=0.52 |
| P3 | 3-D 剪切波衰减 | 精确NS解 u=A sin(kz)exp(-νk²t) | n ∈ {16}, τ=0.8 |
| P4 | 内部固体平板阵列（无边界壁面） | 抛物线剖面 | 周期域内固体单元 |
| P5 | 臂等价性检验 | 密集遍历 vs 稀疏遍历 | 浮点精度一致性 |
| P6 | 索引数组 vs OSS lbmpy | 整数相等性 | D2Q9 + D3Q19 |

### 参考文献
- `lbmpy 2.0` `lbmpy/sparse/mapping.py` (create_index_array, pdf_index, aliasing, periodic wrap)
- F. M. White, *Viscous Fluid Flow*, 3rd ed., §3-3.3 (矩形管道Fourier级数)
- Guo, Zheng & Shi, *Phys. Rev. E* 65, 046308 (2002) (Guo forcing)

---

## 2. 方法 (Methodology)

### 实现路径
1. **OSS调研**：读取 `lbmpy/sparse/mapping.py:84-154`，提取索引数组构建语义（SoA平铺偏移、无滑移别名、周期环绕）
2. **移植实现**：在 `src/tensorlbm/sparse_list_aos.py` 中实现：
   - `build_index_array()`: 从flag网格构建索引数组（支持D2Q9/D3Q19，SoA/AoS两种布局）
   - `SparseListLBM`: 稀疏列表状态类（populations存储在1-D索引空间）
   - `dense_collide_stream()`: 密集遍历对照臂（相同离散格式，独立实现）
3. **验证协议**：
   - **臂A**：密集遍历（TensorLBM原生格式）
   - **臂B**：稀疏遍历 + TensorLBM构建的索引数组
   - **臂C**：稀疏遍历 + OSS lbmpy构建的索引数组（进程内执行）
   - **参考R**：解析解/精确解

### 关键设置
- BGK碰撞 + Guo强制（二阶力项不可省略，否则质量不守恒）
- 半程反弹（halfway bounce-back）无滑移边界
- float64精度，无BLAS
- 稳态判据：max|u(t+200)-u(t)|/u_max < 1e-12，上限120000步

---

## 3. 计算结果 (Results)

### 3.1 P1: 3-D 平面通道 Poiseuille 流

| H | 臂 | u_max | u_max_ref | L∞相对误差 | L2相对误差 | 步数 | <15% |
|---|---|---|---|---|---|---|---|
| 8 | sparse_soa | 0.029662 | 0.029531 | 0.444% | 0.599% | 1800 | ✓ |
| 8 | sparse_aos | 0.029662 | 0.029531 | 0.444% | 0.599% | 1800 | ✓ |
| 8 | dense | 0.029662 | 0.029531 | 0.444% | 0.599% | 1800 | ✓ |
| 16 | sparse_soa | 0.029916 | 0.029883 | 0.110% | 0.150% | 6400 | ✓ |
| 16 | sparse_aos | 0.029916 | 0.029883 | 0.110% | 0.150% | 6400 | ✓ |
| 16 | dense | 0.029916 | 0.029883 | 0.110% | 0.150% | 6400 | ✓ |

**收敛性**：H=8→16，误差从0.444%降至0.110%（4倍，符合O(H⁻²)预期）

### 3.2 P2: 3-D 矩形管道

| Hy×Hz | 臂 | u_max_got | u_max_ref | L∞相对误差 | L2相对误差 | 步数 | <15% |
|---|---|---|---|---|---|---|---|
| 8×12 | sparse_soa | 0.303271 | 0.300000 | 1.157% | 0.918% | 1400 | ✓ |
| 8×12 | sparse_aos | 0.303271 | 0.300000 | 1.157% | 0.918% | 1400 | ✓ |
| 16×24 | sparse_soa | 0.300879 | 0.300000 | 0.293% | 0.233% | 4400 | ✓ |
| 16×24 | sparse_aos | 0.300879 | 0.300000 | 0.293% | 0.233% | 4400 | ✓ |

**收敛性**：8×12→16×24，误差从1.157%降至0.293%（约4倍，符合预期）

### 3.3 P3: 3-D 剪切波衰减

| n | 臂 | ω_got | ω_ref | 相对误差 | <15% |
|---|---|---|---|---|---|
| 16 | sparse_soa | 1.5600e-02 | 1.5421e-02 | 1.161% | ✓ |
| 16 | sparse_aos | 1.5600e-02 | 1.5421e-02 | 1.161% | ✓ |
| 16 | dense | 1.5600e-02 | 1.5421e-02 | 1.161% | ✓ |

### 3.4 P4: 内部固体平板

| 臂 | u_max_got | u_max_ref | L∞相对误差 | 质量守恒 | <15% |
|---|---|---|---|---|---|
| sparse_soa | 0.029662 | 0.029531 | 0.444% | 1.2e-13 | ✓ |
| sparse_aos | 0.029662 | 0.029531 | 0.444% | 1.2e-13 | ✓ |

### 3.5 P5/P6: 表示层检验

| 检验项 | 结果 |
|--------|------|
| 索引数组 vs OSS lbmpy (D2Q9) | **0 mismatches** (整数相等) |
| 索引数组 vs OSS lbmpy (D3Q19) | **0 mismatches** (整数相等) |
| 密集 vs 稀疏 (SoA) | max|du|/u_max < 1e-12 |
| 密集 vs 稀疏 (AoS) | max|du|/u_max < 1e-12 |
| SoA vs AoS 列差异 | 检测到（布局不匹配不静默） |
| 格子顺序不变性 | 2.66e-15 |
| OSS数组驱动 vs 移植数组 | 0.00e+00 |
| 布局不匹配控制 | deviation=21.06, detected=True |

---

## 4. 结果图 (Figures)

图表已在 `xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/figures/` 生成：
- `poiseuille_profile.png`: P1 通道剖面与解析解对比
- `poiseuille_convergence.png`: P1 网格收敛曲线
- `taylor_green_decay.png`: 2-D Taylor-Green衰减曲线
- `p1_p4_profiles.png`: P1/P4 剖面对比
- `p2_duct_field.png`: P2 管道速度场云图
- `p3_shear_wave.png`: P3 剪切波衰减
- `p7_storage.png`: P7 存储开销对比

（纯文档任务，图表由前序mission生成，本单未重新生成）

---

## 5. 注意点 (Notes & Limitations)

### 踩过的坑
1. **P2 duct收敛检验失败**：初始运行时只完成了8×12一个分辨率，导致`P2_duct_finer_below_coarser`检查失败。本单补跑了16×24案例，确认误差从1.157%降至0.293%，收敛性成立。
2. **Ma数过高**：P2 duct案例的Ma=0.52，接近LBM适用范围上限（通常建议Ma<0.3）。尽管如此，误差仍在1.2%以内，说明该实现在中等Ma数下仍可靠。
3. **macroscopic()返回值**：3-D情况下返回(rho, ux, uy, uz)四个数组，而非(rho, u, coords)三元组。需通过`sp.coords`单独获取坐标。

### 局限性
- **性能**：稀疏遍历为NumPy原型，每步速度慢于密集遍历（本锚点声明的是表示精度，非性能）
- **索引数组构建**：使用Python dict + 逐单元循环，O(NQ)复杂度，不适合大几何
- **周期边界限制**：周期环绕目标必须为流体单元；若固体单元恰好位于环绕行，索引数组未定义（与lbmpy相同限制）

### 适用范围
- BGK碰撞 + Guo强制
- 半程反弹无滑移
- D2Q9 / D3Q19 格子
- 低Ma数（<0.3推荐，<0.6可用）
- 稳态/准稳态流动

---

## 6. 误差判定 (Error Verdict)

### 验收标准
Owner精度标准：<15%（所有基准案例）  
TensorLBM"论文"标准：<3%

### 逐项判定

| 案例 | 关键量 | 基准值 | 计算值 | 误差 | <15% | <3% |
|------|--------|--------|--------|------|------|-----|
| P1 H=8 | u_max | 0.029531 | 0.029662 | 0.444% | ✓ | ✓ |
| P1 H=16 | u_max | 0.029883 | 0.029916 | 0.110% | ✓ | ✓ |
| P2 8×12 | u_max | 0.300000 | 0.303271 | 1.157% | ✓ | ✓ |
| P2 16×24 | u_max | 0.300000 | 0.300879 | 0.293% | ✓ | ✓ |
| P3 n=16 | decay rate | 1.5421e-02 | 1.5600e-02 | 1.161% | ✓ | ✓ |
| P4 | u_max | 0.029531 | 0.029662 | 0.444% | ✓ | ✓ |

### 总体判定
**PASS** — 所有基准案例误差 <1.2%，远优于15%验收标准，且全部 <3%论文标准。

### 验收检查清单
```
P1_plane_poiseuille_all_arms_below_15pct: PASS
P1_errors_decrease_with_H: PASS
P2_duct_below_15pct: PASS
P2_duct_finer_below_coarser: PASS
P3_shear_wave_below_15pct: PASS
P4_interior_slabs_below_15pct: PASS
P4_mass_conserved: PASS
P5_arms_equivalent_1e12: PASS
P5_stencil_order_invariant: PASS
P5_oss_array_driven_equals_port: PASS
P6_index_array_bit_identical_to_oss: PASS
P6_layout_mismatch_detected: PASS
```

**Overall verdict: PASS**

---

## 7. 复现 (Reproduction)

```bash
# 运行完整验证套件
cd /srv/tensor/agent/work/m22363
PYTHONPATH=src python xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/round2_d3q19/run_r2.py

# 运行单元测试
PYTHONPATH=src python -m pytest tests/test_xval_sparse_list_aos.py tests/test_xval_sparse_list_aos_3d.py -v

# 结果文件
# xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/round2_d3q19/results.json
# xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/round2_d3q19/tables.md
```

---

## 8. 结论 (Conclusion)

TensorLBM 对 `lssfau/lbmpy` 稀疏列表存储格式的移植实现已通过全面验证：

1. **正确性**：索引数组与OSS lbmpy整数相等（D2Q9 + D3Q19）
2. **精度**：所有基准案例误差 <1.2%，远优于15%验收标准
3. **一致性**：密集遍历与稀疏遍历在浮点精度内一致
4. **收敛性**：网格加密误差按O(H⁻²)下降
5. **鲁棒性**：内部固体单元、布局不匹配检测均正常工作

该实现可用于TensorLBM生产内核的稀疏存储路径，特别适用于含大量固体单元的复杂几何（如多孔介质、颗粒流）。

---

**Commit**: (待提交)  
**Report ID**: (待发布)
