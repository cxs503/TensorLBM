# 独立复核记录 — m18635（adversarial-review 席）

锚点 `XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN`（sorush-khajepor/listLBM 的 example-per-main 反模式）。
本单对 m17895 已合并功能做**独立对抗复核**，不信任原报告、重跑证据。

## 复核动作
1. **测试绿**：`tests/test_example_harness.py` + `tests/test_example_antipattern_audit.py`
   + `tests/test_example_cli.py` = **18 passed**；`tests/test_xval_example_per_main.py` = **14 passed**。
2. **重跑 referee**：`run_xval.py --arms a1_16,a1_32,a1_48` 复现
   u_max rel err = 0.7755% / 0.1691% / 0.1660%（ny=16/32/48），与已存 run_artifacts 一致。
3. **逐项误差判定**（owner 判据 <3%）：
   - 槽道 u_max（最粗 ny=16）：0.7755% ✅
   - 槽道 u_max（最细 ny=48）：0.1660% ✅
   - 槽道 profile L2（最细）：0.1224% ✅
   - Taylor–Green 能量衰减率：0.1128% ✅
   - 质量漂移：9.55e-13（<1e-10）✅
   - 与独立控制臂相对差：6.124e-13（<1e-8）✅
4. **反模式实测存在**：examples/ 97.5% 自带 main、31.4% 结构重复行、232 个重复驱动块；
   检测器判别 gap=0.6212（>0.2）证实非空仪器。

## 结论
未发现材料缺陷。Guo 全源项（含 `(c_i·u)c_i` 项）使 `ΣS_i=0` 无伪质量源（质量漂移 9.55e-13 证实）；
半程反弹壁面 H=ny-2 与闭式一致；共享运行时与独立控制臂逐位一致（6.124e-13）——合并是**忠实**的，非重定义。

**验收通过**。平台报告已发布（id=17392）；feature_ledger 已置 `verified`。
