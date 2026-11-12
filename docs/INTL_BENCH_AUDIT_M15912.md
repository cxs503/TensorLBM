# TensorLBM 国际对标专项审计 · m15912（GAP-LBM-INTL-BENCH，返修轮 1）

> 机器生成报告：`artifacts/intl_bench_v1/run1/module_maturity.json`（数字全部由 `python -m tensorlbm.audit` 抽取写盘，零手抄）。
> 审计纪律：审计 Agent 不得自宣布 verified——每一条 `verified` 必须回链 `benchmarks/verified/<case>/result.json`
> 的真值裁决字段 + 收敛字段，缺一降档（本版由 4 条隔离规则强制）。

## ① 案例介绍（审计对象与判据）

- 对象：TensorLBM 仓 `src/tensorlbm/` 全部能力面，按 owner 十维表归并为 45 个模块、10 个维度。
- 四参照：OpenLB / waLBerla / Palabos / Lettuce，源码级引用 51/51 处活体校验通过
  （OpenLB 12/12, waLBerla 16/16, Palabos 13/13, Lettuce 10/10），失败 0 处。
- 判据（证据层级，非"模块存在"）：1 模块存在 < 2 API 测试 < 3 守恒/门算例 < 4 标准算例 < 5 网格收敛 < 6 实验对比。
- 四级分档：verified（层级 5 + 真值裁决 + 收敛 + 模块专属证据 + 无未闭合缺陷）/ partial / implemented / failed。

## ② 怎么做的（方法/工具链）

1. `src/tensorlbm/audit/intl_bench.py`（可安装包内实现，非脚本化石）逐模块 glob 实现/测试/算例路径，
   读 `benchmarks/STATUS.md` 与 `benchmarks/{verified,pending}/<case>/result.json`，抽取误差/裁决/收敛字段；
2. `check_citations()` 对四上游的每条引用做**活体校验**（文件存在 + 符号串命中），上游缺失不静默通过，
   记 `citation_failures` 并令 `external_anchor_ok=false`，CLI 退出码 3；
3. 4 条诚实隔离规则：已知结构性缺陷不得 verified；verified 算例须带真值裁决；须带收敛字段；
   同一算例被多模块引用时视为"非模块专属证据"，相关模块降 partial（本版因此从 7 条 verified 降到 4 条）；
4. `provenance()` 对每个输入文件记 sha256（STATUS.md + 各 result.json）、仓 HEAD、上游 HEAD、模块清单摘要，
   任何数字可追溯到被哈希的输入；
5. 入口：`python -m tensorlbm.audit --repo-root . --out DIR`（`tools/audit/run_intl_bench_audit.sh` 为可复现 runner）；
   平台侧调用：`tensorlbm.api.intl_bench_scorecard()`（真调用，非文档声明）；
6. 落库：生成的 SQL 直接对齐**线上 owner 表结构**（`tensorlbm_module_maturity`(module_path,area,tier,evidence,intl_ref,action,gap_task)
   与 `tensorlbm_compete_score`(week,dimension,maturity,gap_level,score,notes)），已 `psql -f` 装载成功：
   45 模块行 + 10 维度行（week=2026-W46）。

## ③ 计算结果（机器生成，零手抄）

四级分布：verified 4 / partial 24 / implemented 15 / failed 2；
保留 4 · 重构 26 · 删除 2；差距→开发任务 11 条（挂 T1-T5）。

| 维度 | 模块数 | 平均证据层级 | 成熟度指数(0-10) | 上游最高层级 | 差距等级 | verified/partial/implemented/failed |
|---|---|---|---|---|---|---|
| 基础LBM内核 | 4 | 4.5 | 7.0 | 4 | A(并跑/领先) | 1/3/0/0 |
| 高阶碰撞 | 5 | 2.2 | 2.4 | 4 | C(差2级) | 0/1/4/0 |
| 边界复杂几何 | 5 | 4.2 | 6.4 | 4 | A(并跑/领先) | 1/4/0/0 |
| 多相流 | 5 | 3.6 | 5.2 | 4 | B(差1级) | 1/3/0/1 |
| 自由表面 | 3 | 3.33 | 4.7 | 4 | B(差1级) | 0/2/0/1 |
| 湍流 | 4 | 3.0 | 4.0 | 4 | B(差1级) | 0/2/2/0 |
| GPU单卡 | 4 | 2.25 | 2.5 | 4 | C(差2级) | 0/1/3/0 |
| 多GPU-HPC | 2 | 2.0 | 2.0 | 4 | C(差2级) | 0/0/2/0 |
| 工程验证 | 10 | 4.0 | 6.0 | 4 | A(并跑/领先) | 1/7/2/0 |
| AI融合 | 3 | 2.0 | 2.0 | 1 | A(并跑/领先) | 0/1/2/0 |

verified 名单（回链算例）：`d3q27_kernel`→`taylor_green_3d`; `stl_cad_voxelization`→`wigley_hydrostatics`; `laplace_law_surface_tension`→`laplace_droplet`; `compressible_shock`→`compressible_shock_family`。

**降档记录（诚实性证据）**：

| 模块 | 降档理由 |
|---|---|
| `solver3d_stream_collide` | verified/ 在盘但 result.json 无真值裁决字段 |
| `wall_function_les` | 证据复用：verified 算例 suboff_re1000 同时支撑 3 个模块（wall_function_les,rans_ke,ship_resistance_hull），非模块专属证据 |
| `rans_ke` | 证据复用：verified 算例 suboff_re1000 同时支撑 3 个模块（wall_function_les,rans_ke,ship_resistance_hull），非模块专属证据 |
| `ship_resistance_hull` | 已登记结构性缺陷未闭合，verified 不予保留：STATUS：DTMB5415 阶梯湿面积 16x 光滑→Cd_tot 1.18；KCS 收敛跑未完成 |
| `non_newtonian_yield` | verified/ 在盘但 result.json 无真值裁决字段 |
| `thermal_convection` | verified/ 在盘但 result.json 无真值裁决字段 |

## ④ 结果图

![maturity](../artifacts/intl_bench_v1/run1/level_distribution.png)
![radar](../artifacts/intl_bench_v1/run1/compete_radar.png)
![sphere](../artifacts/intl_bench_v1/run1/fig_sphere_cd_ladder.png)

图 1 四级分档分布（n=45）；图 2 十维竞争度雷达（TensorLBM 成熟度 vs 上游最高层级折算）；
图 3 唯一带完整数值链路的物理锚点复现：sphere_re200 网格阶梯 Cd 误差与 Schiller-Naumann 参考线、
±3% 验收带（数据源 `benchmarks/verified/sphere_re200/result.json`，本图未重算物理，仅重绘既有工件）。

## ⑤ 注意点（坑与局限）

- **自记性能不算第三方复现**：`triton_fused.py` 的 8.6 GLUPS、`triton_fused_distributed` 的 69 GLUPS/8GPU
  均为项目自记（docstring/文档），无第三方复现、无统一口径（格子/碰撞/精度/边界/规模/设备/计时），
  故 GPU 单卡 2.5、多GPU-HPC 2.0 分档，禁跨口径比数；
- **上一版 SQL 建表与实际库结构冲突**（`audit_id` 列不存在）导致落库静默失败——本版改为对齐线上 schema 并实测装载；
- **校准基准揪出真 bug**：单模块评分路径原先不会因"已登记缺陷"降档（只有整仓隔离 pass 才降），
  控制组 `c08` 因此 FAIL（got=verified vs ref=partial），修复后 10/10 通过；
- 证据复用是本仓最容易被忽视的过档来源：`suboff_re1000` 一个算例曾被 3 个模块同时引用；
- 层级 6（实验对比）当前无模块达到；多相/自由表面的 `hadamard_rybczynski_droplet` 五模型全败仍是 pending。

## ⑥ 误差判定（关键量 vs 基准，是否 <3%）

| 指标 | 基准（ref） | 计算（got） | 误差 | 是否 <3% |
|---|---|---|---|---|
| sphere_re200 Cd（D=28 最细） | 0.805615（Schiller-Naumann, Re=200） | 0.798671 | -0.862% | 是 |
| sphere_re200 网格跨度 | 收敛判据 span≤3% | 2.728% | — | 是 |
| 评分器校准一致率（10 控制组） | 100% | 100.0% | 0 | 是 |
| 上游源码引用活体校验 | 51 处 | 51 处 | 0 处缺失 | 是 |
| hadamard 多相液滴 | 期望不得 verified（五模型全败） | got=partial | — | 是（拒绝授予） |

**honest 说明**：本单不改任何物理求解器，故无新场量云图；物理数值均取既有 verified 工件，
不构成对多相/自由表面/GPU 性能的验证结论。CM 级缺陷（DTMB5415 阶梯湿面积 16×、KCS 未收敛）
仍在台账内未闭合，相关模块一律 partial。

## 复现命令

```bash
PYTHONPATH=src python -m tensorlbm.audit --repo-root . --out artifacts/intl_bench_v1/run1
PYTHONPATH=src python benchmarks/intl_bench_calibration/run_benchmark.py --repo-root . \
    --out benchmarks/intl_bench_calibration/RESULT.json
runuser -u postgres -- psql -d tensorplatform -f artifacts/intl_bench_v1/run1/maturity.sql
PYTHONPATH=src python -m pytest tests/test_intl_bench_audit.py -q          # 9 passed
```
