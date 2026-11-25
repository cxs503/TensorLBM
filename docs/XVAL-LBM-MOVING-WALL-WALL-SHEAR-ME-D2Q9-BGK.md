# XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK（动壁壁面力/动量交换对标）

已合并半程动壁 bounce-back 闭包 ladd_pre_local 的**壁面力**经解析基准复核：Couette 精确解 +1.58e-09 %、Stokes-2nd 幅值 +0.0066 % 相位 +0.0801 度、Rayleigh 3.96e-03 %，记账恒等式 6.7e-13；判据 M1:PASS, M2:PASS, M3:PASS, M4:PASS, M5:PASS, M6:PASS, M7:PASS, M8:PASS => PASS。

定位：补齐 #2342（速度场已验证）留出的壁面剪应力/动量交换验证欠账。工件见[归档结果](../benchmarks/cross_validation/moving_wall_wall_shear_me_d2q9_bgk/artifacts/summary.json)，[复跑入口](../benchmarks/cross_validation/moving_wall_wall_shear_me_d2q9_bgk/run_xval.py)，门禁 tests/test_xval_moving_wall_wall_shear_me.py（7 passed）。

从仓库根目录运行以下命令，结果写入各自的 `outputs/cross_validation/` 子目录。支持用 `--out DIR` 显式指定输出位置。

```bash
PYTHONPATH=src python benchmarks/cross_validation/moving_wall_halfway_bb_d2q9_bgk/run_xval.py --stage quick
PYTHONPATH=src python benchmarks/cross_validation/moving_wall_wall_shear_me_d2q9_bgk/run_xval.py --stage core
```

局限：平面壁/D2Q9+BGK/float64；Rayleigh tau=1.2 未缩放窗口 34.5% 未通过（待定位）；u_w·n≠0 会流失流体质量（已量化、非线性，调用侧应强制 u_w·n=0）。
