# 溯源 / provenance（P46W3-N01 keel_drag campaign）

* 分支/base commit: `agent/m9149` on `b5910dff`
* 语言/库: python 3.11.6, numpy 2.4.6, matplotlib 3.10.8
* 求解器: `tensorlbm.underice_drag` 的 C/OpenMP 核（节点 61/34 的粗糙床机器），
  由 `tensorlbm.keel_drag.run_keel_case` 驱动；**未修改核代码**。
* 网格/参数: grid=[32, 32, 42], Re_tau=20.0, u_tau=0.02,
  steps=14000, warmup=4000, sample_interval=20,
  probe_steps=2000, max_passes=5, threads=12
  （雷诺扫描逐算例 u_tau = Re_tau/1000，ν 恒定）
* 随机源: 掩膜相位与体素化完全确定（seed=20261110），初始噪声幅度 0（层流定常解）。
* 算例数: 22 成功 / 0 失败；分片 = 4
  （`--shard i --nshards 4`，各片 manifest 见 `manifest_shard*.json`）。
* 命令:
  ```
  python benchmarks/keel_drag/run_campaign.py --out benchmarks/keel_drag/campaign \
      --shard $i --nshards 4 --threads 12 --grid 32,32,42 --steps 14000 \
      --warmup 4000 --probe-steps 2000 --max-passes 5 --u-tau 0.02
  python benchmarks/keel_drag/analyse.py --results benchmarks/keel_drag/campaign
  python benchmarks/keel_drag/make_report.py --results benchmarks/keel_drag/campaign
  ```
* 原始数据（报告里每个数字的唯一来源，零手抄）: `campaign/cases/<case_id>.json`
  （每算例一条，含 `amplitude_passes` 的逐趟 τ_me/τ_balance 轨迹）
* 汇总 sha256[:16]: cases.json `8a123ca235c3a207`, verdicts.json `ae7050996c5f51a4`,
  report.json `82a376d3fa240b8d`
* 运行环境: 单节点 128 核共享机（负载 ~300），每算例 12 线程，4 片并行；墙钟约 70 min。
* 复算说明: 全部为确定性算例，重跑同参数应逐位复现（同一 C 核、同一掩膜、无噪声）。
