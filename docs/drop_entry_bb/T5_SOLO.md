# T5 单打对照 · 单滴入深水池（Santini case-3）
> DE 五队赛制 · 单打对照组（无队友）· 2026-11-20 · TensorLBM `agent/m26274`
> 全流程：读事实卡→诊断→处方→实现→跑验证。诚实优先于成功。

## 1. 处方卡（五要素）
**假设**：早死（0.7~1.8 ms `health_limit`）因果链 = L/I 质量交换单向记账（interface 记贷、liquid 不记借）+ LIQUID 胞 LBM 密度偏离 1 → 液体库存以 ~1e-4/ms 单调漂移 → 首触后 1~2 ms 越过 1e-4 门。
**处方**：
1) `free_surface_probe.py` 增 `--ledger`：把 `runtime_ledger` 传入 `free_surface_step`，输出算子归因摘要（诊断，默认关，零行为改变）。
2) `solver3d.py` D3Q19 MRT 秩检查 `np.linalg.matrix_rank`→`torch.linalg.matrix_rank`：loong64+numpy2.0.1 的 LAPACK SVD 坏，导入断言炸，集群跑不了任何自由面算例。
**预测**：baseline(cells=8,无旗标) postcontact_healthy_ms∈[0.8,1.1]（复现归档 d8=1.046）；组合配置(simple-exchange+capacity+release1.53) ∈[13,14.5]（归档 cal8=13.851, Ma 0.3006）。可证伪：组合配置若在 10.4 ms 前 drift>1e-4，或集群导入仍失败，则处方被否。
**成本**：`python3 benchmarks/water_entry/single_drop/free_surface_probe.py --cells 8 --speed 0.06 --gap 0.18 --release-speed 1.53 --simple-exchange --capacity --out runs/t5_solo/t5_combined8.json`（集群 cj5662，8 核，~13 min）。
**风险**：最可能死在晚期 Ma>0.3（归档 cal8 死于 13.851 ms）；ledger 有开销（只用于短跑）；torch 秩检查依赖节点 torch LAPACK 健康（n330 已验证 `TORCH_RANK 19`）。

## 2. 补丁摘要
- `benchmarks/water_entry/single_drop/free_surface_probe.py`：+`--ledger`（诊断旗标，默认关）。
- `src/tensorlbm/solver3d.py`：秩检查 np→torch（1 行+注释），与 `linalg_safe` 同后端。
- 验证：本地导入 OK（‖MM⁻¹−I‖=3.6e-15）；集群 n330 `IMPORT_OK`；`--ledger` 120 步 unexplained 残差 2.5e-11（账本闭合）。

## 3. 实测 postcontact_healthy_ms（cells=8）
| 配置 | contact_step | 终止 | postcontact_healthy_ms |
|---|---|---|---|
| baseline（无旗标，能量释放 1.397 m/s） | 49 | health_limit@1.520 ms | **0.9121** |
| 组合（simple-exchange+capacity+1.53 m/s） | 45 | health_limit@14.410 ms（Ma 0.3006） | **13.851** |

组合配置与归档 cal8 逐步一致（step1185 死亡、drift 1.059e-5、impact 1.40918 m/s 全同，深度值 15 位一致）。深度对照（组合配置，np.interp）：
| t_ms | 实验 | 模拟 | 相对误差 |
|---|---|---|---|
| 1.396 | 0.512 | 0.250 | 51.1% |
| 6.400 | 1.647 | 1.217 | 26.1% |
| 10.394 | 1.857 | 1.434 | 22.8% |
| 13.391 | 1.304 | 1.260 | 3.4% |
历史峰值 1.435 D@11.091 ms（对 1.8574 误差 22.7%）；run 死于 13.851 ms，未覆盖末两点（15.4/17.4 ms）。

## 4. 诚实节：卡在哪
- **3% 精度门未过**：13 档归档 run 全 `numeric_pass=False`，最佳峰值误差 13.2%（matched8）；气穴深度系统性偏低 13~23%，我未解决。
- **晚期 Ma 死未修**：step1183 `rho_min` 掉到 0.724（局部低密度袋）→速度尖峰 Ma 0.3006；我没动求解器守恒结构。
- **库存漂移根因**（单向 L/I 交换）是参考模型（waLBerla 不碰 liquid fill）的设计后果，单打席位不敢改。
- **环境坑**：本机工具 120 s 超时 + 集群首次导入失败（MRT 秩断言）→ 全窗只能修复后上集群。
- **首点 0.0389 D** 低于实验 3% 分辨率（0.0132 D），归档保留并计败。
- **结论**：单体全流程可复现「组合配置活过 10.4 ms 峰值时刻」（13.851 ms），但 3% 定量验收仍 FAIL——honest-FAIL。
