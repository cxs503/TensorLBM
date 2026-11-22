# DE-X-K1-CAVCLOS Draft

## 已读
- WAVE_1.md (not found, relying on task desc)
- ABBS.md, CONV12.md, CAVSWEEP.md (not found, relying on task desc)
- src/tensorlbm/free_surface_lbm.py (L1490-1530)

## 已得结论
- 物理出处：空穴闭合速率由回流水控制，回流水由界面动力学驱动。
- 最小改动：free_surface_lbm.py L1508 已有 `cav_clos_alpha` 参数支持 `threshold = kappa * (1 + alpha)`。
- 处方：设置 `cav_clos_alpha=0.5`，延缓空穴壁面转液，抑制过闭合。
- 预登记预测：peak_depth ∈ [1.50, 1.70]D。

## 下一步
- 递交集群作业：`python3 benchmarks/water_entry/single_drop/free_surface_probe.py --cells 8 --gap 0.18 --speed 0.06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --cav-clos-alpha 0.5 --steps 10000 --out runs/K1-CAVCLOS/result.json`
- 等待结果，分析三元组。
