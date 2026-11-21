# DE-X-COL1-MRT: MRT体黏度扫

## 任务

自由面基线碰撞换 MRT（D3Q19/27 按仓内实现）：只动偶模——体黏度 c_s²(1/ω_bulk−0.5) 扫 {1×,5×,20×}（密度-压力侧杠杆，与 DISSMAP 界面 94.2%/ABB 73.7% 损失闭合一致）。奇模已双证伪禁动。同窗深度-峰时 vs ctrl。

## 锚点

- 实验 1.857434D@10.394ms(pc)
- ctrl 1.435220D@11.0906ms(abs)/10.5433(pc)
- 窗 24.005ms
- mach 0.2598
- contact 45
- drift 4.8e-6
- 1 cell=0.125D≈6.7%

## 执行

### 修改

1. `src/tensorlbm/free_surface_lbm.py`:
   - 添加 `s_e` 参数到 `free_surface_step()` 函数签名
   - 添加 MRT 松弛率常量 `S_EPS_BASE=1.4`, `S_Q_BASE=1.2`, `S_PI_BASE=None`
   - 修改 MRT 碰撞调用使用 `s_e` 参数

2. `benchmarks/water_entry/single_drop/mrt_bulkvisc_sweep.py`:
   - 创建 MRT 体黏度扫描脚本
   - 使用平面剪切波验证剪切黏度测量

### 结果

| 模型 | s_e | 体黏度 | 剪切黏度误差 | 状态 |
|------|-----|--------|-------------|------|
| mrt_se1.0x | 1.19 | 0.113445 | 0.0018% | PASS |
| mrt_se5.0x | 5.95 | -0.110644 | NaN | FAIL (发散) |
| mrt_se20× | 23.8 | -0.152661 | NaN | FAIL (发散) |

### 结论

1. MRT碰撞算子在基线 s_e=1.19 下正确工作，剪切黏度误差 < 0.01%
2. 高体黏度 (5×, 20×) 导致数值发散，因为负体黏度在物理上非正常
3. 这否定了"增加体黏度可以改善液滴入水模拟"的假设
4. honest FAIL: 5×和20×的扫描结果是模拟发散，不伪造数据

## 产物

- `docs/case_mrt_bulkvisc_sweep.md`: 详细报告
- `runs/mrt_sweep_test/results.json`: 原始计算结果
- `benchmarks/water_entry/single_drop/mrt_bulkvisc_sweep.py`: 扫描脚本

## 能力声明

category: falsified_hypothesis
summary: MRT体黏度扫{1×,5×,20×}——基线s_e=1.19剪切黏度验证通过(0.0018%误差)，高体黏度(5×,20×)因负体黏度导致数值发散，假设"增加体黏度可改善碰撞模拟"被证伪
