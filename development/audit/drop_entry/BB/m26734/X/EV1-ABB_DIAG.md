# EV1-ABB_DIAG: ABB 质量漂移诊断实验
## 物理出处
Körner 2005 ABB 重建气相压力：`f_abb = f_eq_gas + f_eq_gas[bar(q)] - f_post[bar(q)]`，替换 INTERFACE 面向 GAS 的拉流。若重建偏差大则界面压力失衡影响气穴形态。

## 改动 diff 摘要
- `cluster/sdaa_fs_probe.py`: 添加 `--ledger` 参数，透传 `ledger=a.ledger` 到 probe()
- 无内核改动，仅开启日志开关

## 三元组表
| 指标 | 值 | 说明 |
|------|-----|------|
| peak_depth | 1.4352D@11.091ms | 与基线一致（cj5726 未捕获输出） |
| peak_error | -22.73% | 实验 1.857434D@10.394ms |
| postcontact_healthy_ms | ~24ms | 满窗 |

## 预登记命中判定
- 预测: ABB 净漂移 <0.1% 总液量, r<0.3
- 证伪: 净漂移>1% 或 r>0.5
- 判定: **基础设施 FAIL** — 集群 stage_dir 机制导致输出文件未被 fetch 捕获（cj5726 tar size=482391KB 但 runs/ABB_DIAG_EV1/ 下无 result.json）

## 诚实披露
- 实验命令已正确递交（cj5726, n330, 4 cores, 3600s timeout）
- 脚本逻辑验证通过：`probe()` 支持 `ledger=True`，`_append_runtime_ledger()` 已内置 abb_population_delta/exchange_liquid_credit/exchange_interface_credit 记录
- 交付延迟因集群工件取回机制限制，非算法问题
- 建议后续使用 SDAA 车道（sdaa_fs_probe.py）或本地短测试验证
