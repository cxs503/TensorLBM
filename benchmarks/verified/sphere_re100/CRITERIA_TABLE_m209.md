# sphere_re100 判据表 - Mission 209 (Round 8)

**案例**: 球体绕流 Re=100  
**参考值**: Cd_ref = 1.0917311 (Schiller-Naumann)  
**判据门限**: |err| ≤ 3%  
**验证状态**: ✅ PASS

---

## 1. 网格收敛判据

| 网格 | D | Cd | err vs ref | 门限 | 稳态漂移 | 稳态门限 | 判定 |
|------|---|----|-----------|------|---------|---------|------|
| T1 | 40 | 1.121980 | +2.771% | ≤3% ✅ | -0.145% | <0.3% ✅ | PASS |
| T2 | 60 | 1.119462 | +2.540% | ≤3% ✅ | -0.042% | <0.3% ✅ | PASS |

**收敛跨度**: |2.771% - 2.540%| = 0.231% ≤ 3% ✅  
**单调性**: |err(D40)| > |err(D60)| ✅ (误差随分辨率单调下降)

---

## 2. 物理与数值判据

| 判据 | 要求 | 实际 | 判定 |
|------|------|------|------|
| Re 锁定 | 100.000 | 100.000 (τ校验) | ✅ |
| 堵塞率 | <2% | 1.603% | ✅ |
| 质量漂移 | <1 ppm | ≤0.08 ppm | ✅ |
| 升力对称 | Cl ≈ 0 | ~1e-4 | ✅ |
| BFL 边界链 | 亚格插值 | Bouzidi-Firdaouss-Lallemand | ✅ |
| 力法口径 | 逐链动量账本 | bfl_force_ledger_sparse | ✅ |

---

## 3. 位一致与独立复核

| 检查项 | 方法 | 结果 | 判定 |
|--------|------|------|------|
| T1 稀疏 vs W7 密集 | 240 采样 × 4 量 | 逐位相同 | ✅ |
| W7 vs W8A 驱动链 | 6 步 × 4 量 | 逐位相同 | ✅ |
| 独立 CV 动量平衡 | 窗均闭合 | 0.015% / 0.008% | ✅ |

---

## 4. 域形收敛签名 (W7 诊断)

| up/D | Cd | err | 说明 |
|------|----|----|------|
| 1.25 | 1.214927 | +11.28% | 入口钳制主导 |
| 2.00 | 1.155335 | +5.83% | |
| 2.75 | 1.137950 | +4.23% | |
| 3.00 | 1.134710 | +3.94% | |
| 4.00 | 1.127859 | +3.31% | |
| **大域 lat3/up3/down4** | **1.121980** | **+2.77%** | **本案** |

**拟合**: err ≈ E₀ + C·(a/x)³, R² = 0.9998  
**结论**: 入口平面 Dirichlet 钳制是主因；大域把误差压进 3% 门

---

## 5. 替代口径负结果 (阶梯 surface-MEM)

| 口径 | D=12 Cd | err vs SN | 说明 |
|------|---------|-----------|------|
| cd_mem_surface | 1.1916 | +9.15% | 网格收敛到错的值 |
| cd_mem_all | 1.2301 | +12.68% | |
| cd_mem_interior | 0.0385 | — | |

**原因**: 球是双曲率体，阶梯体素化畸变界面法向；只有光滑/插值边界 (BFL) 能消掉几何偏差

---

## 6. 流程诊断 (Mission 209)

| Mission | 状态 | 原因 | 说明 |
|---------|------|------|------|
| m205 | failed | 模型未调用 finish | 流程断点 |
| m191 | failed | 模型未调用 finish | 流程断点 |
| m186 | failed | 模型未调用 finish | 流程断点 |
| m179 | failed | 模型未调用 finish | 流程断点 |
| **m198** | **succeeded** | — | **历史唯一稳定 PASS** |
| m75 | succeeded | — | 诊断轮 |

**结论**: 7 轮中多轮 FAIL 都是流程断点（未 finish、无落盘），**不是算法/物理失败**

---

## 7. SDAA 设备证据 (Round 8 冒烟测试)

```
torch: 2.7.1
CUDA available: False
SDAA available: True
SDAA device count: 1
SDAA device name: /dev/tcaicard0
```

**冒烟测试**: D=16, 200 步, sdaa:0  
**结果**: ✅ 完成，设备可用

---

## 8. 最终判定

| 判据 | 结果 | 判定 |
|------|------|------|
| D40 |err| ≤ 3% | 2.771% | ✅ |
| D60 |err| ≤ 3% | 2.540% | ✅ |
| 跨度 ≤ 3% | 0.231% | ✅ |
| 单调下降 | True | ✅ |
| 稳态 | True | ✅ |
| 位一致 | True | ✅ |
| 独立 CV | True | ✅ |
| SDAA 可用 | True | ✅ |

**总体判定**: ✅ **PASS** - 所有判据满足，案例已验证

---

## 9. 复现命令

```bash
cd benchmarks/verified/sphere_re100
export PYTHONPATH=/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks
export W8A_DEV=sdaa:0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T1: D=40, 12000 步
python run.py t1_d40big_12k.json --D 40 --steps 12000 --lat 3.0 --up 3.0 --down 4.0 --tau 0.56 --sample 50 --cv_tail 2500

# T2: D=60, 24000 步
python run.py t2_d60big_24k.json --D 60 --steps 24000 --lat 3.0 --up 3.0 --down 4.0 --tau 0.59 --sample 50 --cv_tail 2500

# 独立验证
python verify.py
```

---

**文档生成**: Mission 209, Round 8  
**时间**: 2026-10-29  
**产物**: 
- `verify_m209.py` - 验证脚本
- `verify_m209_summary.json` - 验证摘要
- `sdaa_smoke_test_m209.py` - SDAA 冒烟测试
- `artifacts/sdaa_sphere_re100_v8.json` - 完整证据链
- `CRITERIA_TABLE_m209.md` - 本文档
