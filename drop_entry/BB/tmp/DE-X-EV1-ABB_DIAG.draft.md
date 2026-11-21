# DE-X-EV1-ABB_DIAG Draft
## 已读
- TRACK_EV_W1.md: ARM ABB_DIAG 处方卡
- free_surface_probe.py: probe() 函数已支持 ledger=True 参数
- free_surface_lbm.py: _append_runtime_ledger() 已内置 abb_population_delta, exchange_liquid_credit, exchange_interface_credit 记录
- sdaa_fs_probe.py: 已添加 --ledger 支持

## 已得结论
- ABB 日志机制已存在于内核 (_append_runtime_ledger)，只需传递 ledger=True
- SDAA 脚本已修改支持 --ledger 参数
- 集群运行 cj5726 成功但工件取回机制导致输出未捕获

## 下一步
- 交付 EV1-ABB_DIAG.md（基础设施 FAIL 如实报告）
- 双写 docs/drop_entry_bb/X_EV1-ABB_DIAG.md
