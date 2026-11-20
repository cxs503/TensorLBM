# DE-X-DIGEST-T2 · T2 黑板队·集成工程官·跨队蒸馏（≤40行）
自检：5/5（文件在且非空/数字可溯/双写commit/预测区间/失败如实）
CROSS已读：无（CROSS 目录本单首建，无 *_DIGEST.md 可读）
上游验收：P1 PASS · P2 PASS · P3 PASS（REPORT.md 首行；P1 wide8 断言错误在裁决节②更正，未达退回单标准）

## 1. 核心结论
- **任务**：自由面水入「存活修复处方」，cells=8 验证（REPORT.md §1）。
- **根因裁决**：A3 全部 5 个 window_complete 幸存者（cal12/camera8/dt8/motion8/wide8）共享 capacity+simple-exchange+release-speed；camera8/dt8/motion8 在默认 span=4 满窗存活 → P2「边界为根因」被证伪；公共因子=质量交换配方（REPORT.md §1①）。
- **本单新发现**：域×速度交互。s0.04×窄域=1.92%(dt8)、s0.06×宽域=0.68%(wide8)、s0.04×宽域=18.2%(本单 cj5664) → 融合处方必须取 s0.06（REPORT.md §1⑥）。
- **最终处方**（REPORT.md §2）= wide8 精确配方：`--cells 8 --gap 0.18 --speed 0.06 --span 6.0 --pool-depth 4.5 --capacity --simple-exchange --release-speed 1.53 --steps 4000`。弃 P1 之 speed 0.02 与 --nearest-surface；弃 P2 之 8.0/6.0（未证且贵）。

## 2. 关键数字（全部带原卡出处）
| 量 | 值 | 出处 |
| 全域臂 postcontact_healthy_ms | 24.005ms（window_complete rc=1） | REPORT.md §4 集群 cj5664 |
| 下界臂（pool3.0） | 22.741ms（health_limit(mach)） | REPORT.md §4 集群 cj5670 |
| 对照臂（旧域 span4/pool3） | 0.4216ms（topology_step_124） | REPORT.md §4 本机 runs/de_t2_e1_ctrl.json |
| A3 基线 case3_d8 | 1.046ms（health_limit） | REPORT.md §4 |
| 峰值深度@10.394ms | 计算 1.5192D vs 实验 1.8574D（D=2.27mm）→ 18.21% | REPORT.md §6 |
| 触水速度 | 1.4282 vs 1.4 m/s → 2.02% | REPORT.md §6 |
| 体积漂移(全域臂 max) | 2.59e-4（超 1e-4 门 2.6×） | REPORT.md §6 |
| wide8 基线 | 24.005ms / 0.68%（A3 值，非本单复测） | REPORT.md §4 |
| dt8 基线 | 24.005ms / 1.92% | REPORT.md §4 |
| 实验槽深 | ~17.6D（A1#1） | REPORT.md §1⑤ |
| 格点数 | 270k(span8) vs 124k(span6)，2.2× | REPORT.md §1④ |
| 补丁 | commit 0c2bd3493，4 行 | REPORT.md §3 |
| 步速 | ~1.2s/步@124k 格点，60min 作业上限压线 | REPORT.md §7 |

## 3. 可复用方法/仪器
- **三开关组合**（capacity+simple-exchange+release-speed）为存活修复公共因子（REPORT.md §1①）。
- **预测区间法**：预声明 [10.394, 24.005]；P1 预测 [24.0,24.01]、P2/P3 预测 [20.0,24.005]；≥24.0 且 peak_error≤3% 为全过；<10.394 即证伪（REPORT.md §5）。
- **P3 guard 核查**：TL_FS_INPUT_GUARD 默认 True 无需动（src/tensorlbm/free_surface_lbm.py L951/1120/1162）（REPORT.md §2）。
- **score.py 口径**：postcontact_healthy_ms = 末健康行−接触行（REPORT.md §4）。

## 4. 诚实披露的失败与局限
- **honest-PARTIAL**：全域臂 24.005ms 存活判据满足（P1/P2/P3 区间均覆盖），但 peak_error=18.21%>3% → 全过判据不满足（REPORT.md §5）。
- **单网格**（cells=8）无收敛对；converged 判据不满足；domain_and_material_match_confirmed 仍 false；score 状态=pending（REPORT.md §7）。
- **对照臂死因** topology_step_124 与 A3 case3_d8 的 health_limit 触发器不同但同处 ~1ms 早死区（REPORT.md §7）。
- **体积漂移瞬态**：全域臂 rc=1 因 1.216ms 处 2.59e-4 超门（末行 9.9e-6 已回落）；下界臂同有 2.29e-4 瞬态且 23.29ms 处 mach 0.309 触门（REPORT.md §7）。
- **wide8 0.68% 为 A3 基线值**（非本单复测）；本单未复跑 s0.06 臂（60min/次成本），列为遗留验证项（REPORT.md §7）。
- **集群节点** tensorlbm 旧版缺 cac_lbm，须 PYTHONPATH 指向暂存树 src/（REPORT.md §7）。
