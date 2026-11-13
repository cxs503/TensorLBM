# XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA

## 交叉对标：三元双阱自由能 + 调和 lambda 耦合

**对标对象**: AminZar69/CudaPhaseField (CUDA Fortran, GPL-3.0)
**来源**: https://github.com/AminZar69/CudaPhaseField
**论文**: Zarareh et al. 2022, *Computers & Mathematics with Applications* 109, 100–124
**DOI**: 10.1016/j.camwa.2022.01.005

## 物理模型

三元相场模型（three-component phase-field），三个序参量 `phi1, phi2, phi3` 满足约束 `phi1 + phi2 + phi3 = 1`。

### 双阱势

每个组分的体自由能使用双阱势：
```
f_dw(phi) = phi^2 * (1 - phi)^2
```
其导数为 `phi * (1 - phi) * (1 - 2 * phi)`。

### Lambda 系数

从界面张力矩阵 `sigma(i,j)` 导出：
```
lambda(i) = sigma(j,i) + sigma(k,i) - sigma(j,k)    (循环 i,j,k)
```

### 调和平均 Lambda

```
lambda_t = 3 / (1/lambda(1) + 1/lambda(2) + 1/lambda(3))
```

### 化学势

与 CudaPhaseField 的 `chemical_potential` kernel 完全一致：
```
mu_i = (12/w) * [ lambda(i) * phi_i * (1-phi_i) * (1-2*phi_i)
                 - 2 * lambda_t * phi_1 * phi_2 * (1 - phi_1 - phi_2) ]
       - (3/4) * w * lambda(i) * laplacian(phi_i)
```

### 有限差分算子

- **Laplacian**: D2Q9 各向同性 9 点模板
- **Gradient**: 各向同性 8 点模板

## 对标结果

| 测试项 | 状态 | 误差 |
|---|---|---|
| Lambda 一致性 | ✅ PASS | 0 |
| Laplacian 算子 | ✅ PASS | 0.16% |
| Gradient 算子 | ✅ PASS | 0.32% |
| 平面界面化学势 | ✅ PASS | 有限值，合理范围 |
| 逐点化学势对比 | ✅ PASS | 0 (机器精度) |
| 液滴平衡 | ✅ PASS | std = 1.04e-4 |
| 非均匀 sigma | ✅ PASS | 机器精度 |
| **全场对比（主测试）** | ✅ PASS | **~1e-16 (机器精度)** |

**结论**: TensorLBM 实现与 OSS 公式在机器精度下完全一致，对标误差 << 15% 判据。

## 运行

```bash
python xval/XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA/run_xval.py --out runs/xval_ternary_double_well
```

## 文件

- `src/tensorlbm/phasefield/ternary_free_energy.py` — 三元双阱自由能模块
- `xval/XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA/run_xval.py` — 对标驱动脚本
- `runs/xval_ternary_double_well/results.json` — 对标结果
