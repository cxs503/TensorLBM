# m21594 交差审计 · XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM (m21558)

被判交付提交 `1b685d33c1016089ccaaba225c39215e692c2360`（master 合并 `481ac88da` Merge #2475）。级别 **L2**，判定 **honest-FAIL**。

| 检查项 | 证据 | 结论 |
|---|---|---|
| ① run_artifacts 存在 | quick_validation_results.json + png 在交付提交内（sha256 见 evidence.json） | PASS |
| ①b 全量 campaign 工件 | run_two_component_sc_validation.py 声明 4 例（20k/30k 步）但 artifacts/ 只有 quick_validation_* | MISSING |
| ② benchmark 裁判行 | tensor_benchmark_cases anchor_slug='XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM' -> 0 行 | MISSING |
| ②b 数值 vs 解析（15% 门） | u_peak_err=177.78% / eps_inf=253.30% / eps_l2=264.77% | FAIL |
| ③ repro_cmd 可执行 | python3 run_quick_validation.py 复现 177.78%/253.30%/264.77% | PASS(有缺陷) |
| ③b repro 可移植性 | 脚本 sys.path 硬编码 /srv/tensor/agent/work/m21558/TensorLBM/src；去路径后 import 落到别的工作树（m21384 editable 安装） | DEFECT |
| ③c 比特可复现 | init 用 torch.randn_like 无 seed -> eps_l2 交付 264.42% vs 复跑 264.77% | DEFECT |
| ④ 单测 | 12 passed，但无一条断言物理精度（test_velocity_develops 仅断言 ux_max 增长） | L1 only |
| ⑤ 积分 caller | 无 | MISSING |
| ⑥ 层序口径 | 解析 A 贴 y=0，数值 A 在 y>15 -> 镜像角 22.36% | DEFECT |
| ⑦ 稳态（报告根因假设） | 2000/20000 步 u_max/g·t = 1.0000218451023102/n/a（=1 表示仍在自由加速） | DEFECT |
| ⑧ 单组分对照 | G=0 单组分 20k 步 eps_inf=1.3671018698059019 | DIAG |
| ⑨ 壁格密度 | y=0.5 处 rho_A=584（名义 2.0）×280 | DEFECT |

## 交付数值（程序化复算，源自 quick_validation_results.json 的 profile 数组）

| 量 | 交付存储 | 复算 | 判 15% 门 |
|---|---|---|---|
| u_peak_err | 177.78% | 177.78% | FAIL |
| eps_inf | 253.30% | 253.30% | FAIL |
| eps_l2 | 264.77% | 264.77% | FAIL |
| u_peak (num / ana) | 0.200004 / 0.072000 | 独立 solve_layered: 0.072000 | — |

## 图
![audit](fig_m21558_audit.png)
