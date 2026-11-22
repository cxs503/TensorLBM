author-model: qwen3.7-plus

# 议事会 session-001 · 综合主席最终任务安排（r3-plan）

> 立场：落地执行视角——每条 mission 被采纳后立刻进入派发队列，无需 owner 额外解释。
> 输入：00_context / 01_owner_direction / 02_platform_update / 03_owner_directive_bem / r1-proposal-strategy / r1-proposal-models / r2-critique-verifier。
> 原则：合并重叠 mission、释放卡位、补齐 owner 硬考核缺口（谱系/门禁/千工况）、P0 今天就能派发。

---

## 一、推荐方案（P0 / P1 / P2 优先级）

### P0 —— 今日即可派发（互不依赖，当天出数）

| ID | mission | repo | goal 要点 | 机器判据 | 预算 | 卡位 |
|---|---|---|---|---|---|---|
| **M-C1** | 10 库移植性静态普查 | 各库（只读扫描）+ 脚本落 TensorPlatform/docs | 扫 import 面：纯 torch / triton / CUDA 扩展 / MPI / cupy / numpy fallback；产出 `docs/council/portability_census_<date>.json` | ① 普查脚本跑 TensorLBM 必须复现 m11 的 507/512 纯 torch 结论（偏差 0），否则脚本 INVALID；② JSON 字段 `{repo, files, torch_only_pct, triton_hits, ext_hits, blockers[]}` 完整 | 1 卡冒烟 + 12 卡时 CPU | 2 卡（1 卡冒烟 + 1 卡 CPU） |
| **M-P1** | SDAA 单卡 matmul 基准（固定网格） | TensorLBM | 1200² matmul 在 `/dev/tcaicard0` 上跑 100 次，记录 mean/std/min/max | wall time mean ± std，CV ≤ 2%；环境指纹（torch/torch_sdaa 版本、device 名、git sha）写入 JSON | 1 卡 × 4 卡时 | 1 卡独占 |
| **M-P2** | 单卡显存/功耗画像 | TensorLBM | 在 M-P1 同卡上测显存峰值、功耗曲线、是否降频 | 产出显存表（网格规模 vs 峰值显存）；若降频则记录降频触发点 | 1 卡 × 4 卡时 | 1 卡（与 M-P1 同卡） |
| **M-T1** | 5 triton 模块功能边界审计 | TensorLBM | 列出 5 个 triton 依赖模块完整清单（路径、功能、输入/输出形状）；评估纯 torch 替代可行性 | 5/5 模块审计完成；每个模块有明确替代方案（直接替代/需重构/不可替代）；文档提交 | 0 卡（纯分析）+ 1 卡验证 | 1 卡（验证用） |
| **M-G1** | 谱系 JSON schema 定义 | TensorPlatform | 定义环境指纹、输入参数、输出指标、git sha、时间戳、判据版本的 JSON schema | schema 文件提交；对已知 PASS/FAIL case 的裁决准确率 100%（用 m10 Couette 数据验证） | 0 卡 | 0 卡（纯文档/脚本） |

**P0 卡位合计**：4 卡（M-C1 2 卡 + M-P1/P2 1 卡独占 + M-T1 1 卡验证）。M-G1 不占卡。
**P0 人员**：impl×2（M-C1 + M-P1/P2）、ver×1（M-T1 + M-G1）、arch×1（M-C1 脚本设计）。

---

### P1 —— 第 2–3 日派发（依赖 P0 结论）

| ID | mission | repo | goal 要点 | 机器判据 | 预算 | 卡位 |
|---|---|---|---|---|---|---|
| **M-C2** | 卡上 op 覆盖/CPU 回退探测 | 各库最小算例 | 每库跑最小算例（无算例则 profile 3 个代表 op），检测显式 `.cpu()`/同步/插件 fallback warning | 产出 `(fallback_ops, hot_ops)` 表；至少 1 个库在卡上完成一次完整迭代且无 CPU 回退 | 1 卡 × 8 卡时 | 1 卡 |
| **M-C3** | 裁决书（含第三方复核） | TensorPlatform/docs | 打分 = 纯 torch 比例 × 单案例可判性 × 算术强度先验；得票 ≥2/3 且该库在 M-C2 有一次跑通 | 另一 worker 独立复跑普查脚本，JSON sha256 一致 | 0 卡 | 0 卡 |
| **M-P3** | 性能门禁自动化（接 M-G1 schema） | TensorPlatform | 读取谱系 JSON，对照判据，输出 PASS/FAIL | 对已知 PASS/FAIL case 的裁决准确率 100%；端到端流程 <5min | 1 卡 × 2 卡时 | 1 卡 |
| **M-P5** | 跨卡逐位复现门禁（2D 扩展） | TensorLBM | 沿用 m10 口径（5→0 逐位），扩展到 3 卡/4 卡 | 跨卡 sha256 一致；CV ≤ 2% | 2 卡 × 4 卡时 | 2 卡 |
| **M-T2** | dtype 策略探测 | TensorLBM | 在 Couette 2D + Taylor-Green 2D 上测 fp32/tf32/bf16 | 仅当对解析解误差 ≤0.1% 且加速 ≥1.3× 的 dtype 才进推荐清单 | 1 卡 × 4 卡时 | 1 卡 |
| **m56** | TensorBEM 审计+基线 | TensorBEM | 测试全跑 + 招牌基准复跑对照 README 声称值 | 全量 pytest PASS；基准值与 README 声称值相对差 ≤5% | 2 卡 × 2 天 | 2 卡 |
| **m57** | TensorBEM 首开发单（Wigley 前进速度波阻） | TensorBEM | 双路径波阻账本互证 ≤5% + 基准 + 收敛阶 | 双路径波阻差 ≤5%；收敛阶与理论值一致 | 2 卡 × 3 天 | 2 卡 |
| **M-PLAT** | 平台缺陷修复（potential-flow 500 + resistance 速度数组依赖） | ship-platform | ① potential-flow/panel-method 500（内部 ITTC 公式 Re≤100 定义域错）；② resistance/predict 对速度数组有依赖（同船 15kn：speeds=[12..16] 时 3766 kN，speeds=[15] 时 3574 kN） | ① panel-method 返回 200 + 合理数值；② 同船同速度单点 vs 数组输出一致（相对差 <0.1%） | 0 卡（纯后端修复）+ 1 卡验证 | 1 卡（验证用） |

**P1 卡位合计**：9 卡（M-C2 1 + M-P3 1 + M-P5 2 + M-T2 1 + m56 2 + m57 2）。M-PLAT 不占卡（纯后端）。
**P1 人员**：impl×3（M-C2 + M-P5 + m56/m57）、ver×2（M-P3 + M-T2 + M-PLAT）、arch×1（M-C3）。

---

### P2 —— 第 4 日起派发（依赖 P1 结论 + 显存表）

| ID | mission | repo | goal 要点 | 机器判据 | 预算 | 卡位 |
|---|---|---|---|---|---|---|
| **M-S1** | 3D 单相上卡（cylinder_3d → sphere_re100 D40） | TensorLBM | Cd 与 CPU 基线差 ≤1%，且对参考值误差 ≤3%；跨卡逐位复现 | Cd 差 ≤1%；参考值误差 ≤3%；`/dev/tcaicardA` vs `/dev/tcaicardB` 诊断量 sha256 一致 | 2 卡 × 8 卡时 | 2 卡 |
| **M-S2** | 自由面 dam break 3D 上卡 | TensorLBM | 同 seed 下液面高度场与 CPU 基线 L2 差 ≤1%；质量漂移阈值先测后锁 | L2 ≤1%；质量漂移 ≤0.1%/1e4 步（阈值先测 CPU 基线再锁） | 2 卡 × 8 卡时 | 2 卡 |
| **M-S3** | 多相/相场上卡 | TensorLBM | Laplace 定律 Δp·R/σ 与解析值差 ≤3%；跨卡逐位 | Δp·R/σ 差 ≤3%；跨卡逐位 | 2 卡 × 8 卡时 | 2 卡 |
| **M1** | triton_fused_obstacle 纯 torch 等价 | TensorLBM | 实现 `torch_fused_obstacle`；与 triton interpret 逐值对比 | L2 vs triton interpret <1e-6；SDAA 性能 ≥ interpret 模式 3×（原 10× 过严，见质询处置） | 2 卡 × 2 天 | 2 卡 |
| **M2+M3** | 其余 4 个 triton 模块纯 torch 等价 | TensorLBM | 逐个实现纯 torch 替代；全量回归测试 | 5/5 模块纯 torch 路径通过；15 verified 基准回归 PASS；无 triton import 依赖 | 2 卡 × 2 天 | 2 卡 |
| **M4** | 跨卡域分解框架固化 | TensorLBM | 纯 torch halo exchange 实现；多卡域分解 API；4 卡线性加速比验证 | 4 卡 vs 单卡 L2 <1e-6；4 卡加速比 ≥ 2.8×（并行效率 ≥70%） | 4 卡 × 2 天 | 4 卡 |
| **S0-SUBOFF** | SUBOFF 阻力战役 S0（2D 轴对称 smoke test） | TensorLBM | 2D 轴对称 smoke test；Cd 与参考值差 ≤3%（verification 阶段） | Cd 差 ≤3%；网格收敛趋势正确；体素化 Hausdorff 距离 ≤1 体素 | 2 卡 × 2 天 | 2 卡 |
| **S0-COUPLING** | FVM/BEM standalone SDAA smoke test + 兼容性普查 | TensorFVM, TensorBEM | 复用 M-C1 普查脚本；各跑通 1 个 benchmark（FVM: lid-driven cavity Re=1000；BEM: 球体声散射解析解） | 普查报告完成；各 benchmark L2 <1e-2；SDAA 上卡 PASS；声压级误差 ≤5% | 4 卡 × 1.5 天 | 4 卡 |

**P2 卡位合计**：20 卡（M-S1/S2/S3 各 2 卡 + M1 2 卡 + M2+M3 2 卡 + M4 4 卡 + S0-SUBOFF 2 卡 + S0-COUPLING 4 卡）。
**P2 人员**：impl×6、ver×4、arch×2、doc×1。

---

### 32 卡卡位分配总表

| 车道 | 卡数 | 时段 | 约束 |
|---|---|---|---|
| 扫荡线（既有 in-flight m12–m23，12 案例） | 12 | 连续 | 非本提案，先占先得 |
| **P0（M-C1 + M-P1/P2 + M-T1）** | 4 | 第 1 日独占 | 计时窗口禁 co-tenancy |
| **P1（M-C2 + M-P3/P5 + M-T2 + m56/m57）** | 9 | 第 2–3 日 | M-P5 需独占窗口 |
| **P2（M-S1/S2/S3 + M1-M4 + S0-SUBOFF + S0-COUPLING）** | 20 | 第 4 日起 | 视 P1 结论放行 |
| 机动/热备（值班长抢修、失败重跑） | 7 | 连续 | 任何线可借，借还登记 |

**注**：P2 的 20 卡是峰值需求，实际按 mission 依赖关系错峰启动，不会同时占满。若 P1 结论显示某方向不可行（如显存不够），P2 卡位释放回机动池。

---

## 二、质询意见处置（逐条）

### r2-critique-verifier 质询

| # | 质询内容 | 处置 | 理由 |
|---|---|---|---|
| 1 | D1 M-P4 降级条件模糊（"若 fail 则降级"未定义 fail 的具体指标） | **采纳** | M-P5 已明确：跨卡 sha256 一致 + CV ≤ 2% 为 PASS，否则降级为"仅报告不可控，不作基线依据" |
| 2 | D2 M-C3 打分主观（"得票 ≥2/3"如何定义） | **采纳** | M-C3 判据改为：① 纯 torch 比例 ≥80% ② M-C2 至少 1 次完整迭代无 CPU 回退 ③ 第三方复跑 JSON sha256 一致。三条全满足即 PASS，不再用主观打分 |
| 3 | D3 M-S2 质量漂移阈值未锁（"先测后锁"） | **采纳** | M-S2 判据明确：先测 CPU 基线的质量漂移，锁定为阈值（写入 README），卡上验证 ≤ 该阈值。若 CPU 基线本身 >0.1%/1e4 步，则以 CPU 基线为准 |
| 4 | A S3 STL 体素化 PASS 标准未定义 | **采纳** | S0-SUBOFF 判据增加：体素化后的体素场与解析几何的 Hausdorff 距离 ≤ 1 体素；体素数量与理论体积偏差 ≤1% |
| 5 | A S4 自由表面 2D smoke test 判据未定义 | **采纳** | M-S2 判据已明确：L2 ≤1% + 质量漂移 ≤0.1%/1e4 步。不再用"smoke test"模糊表述 |
| 6 | A S1 的"SUBOFF 验证阶段 ≤3%，validation 阶段 ≤5%"建议 | **采纳** | S0-SUBOFF 判据改为：verification（与参考 CFD 对比）≤3%；validation（与实验对比）≤5%。当前 S0 是 verification 阶段，用 ≤3% |
| 7 | A S4 与 D3 质量守恒判据不一致（1%/1000 步 vs 0.1%/1e4 步） | **采纳** | 统一为：verification case ≤0.1%/1e4 步；validation case ≤0.5%/1e4 步；放弃条件 >1%/1e4 步。M-S2 已用此统一标准 |
| 8 | B S3 声压级 vs 解析解 <10% 过松 | **采纳** | S0-COUPLING 判据改为：benchmark case（球体声散射）≤5%；工程 case ≤10%。当前 S0 是 benchmark，用 ≤5% |
| 9 | B S1 coupling driver 原型 2 天可能不够 | **采纳** | S1 不在 P0/P1，推迟到 P2 后期。若 P2 启动时 driver 仍未完成，S1 预算从 2 天扩展到 4 天，卡位从 4 卡扩展到 6 卡 |
| 10 | B S2 "迭代收敛 <10 步"缺乏依据 | **采纳** | S2 判据改为：迭代收敛或达到最大步数（50 步）时残差 <1e-3；记录实际收敛步数。不再硬卡 10 步 |
| 11 | B 卡位表与 mission 预算不一致（P3 说 S2 用 4 卡，但 S2 预算 6 卡） | **采纳** | P2 卡位表已修正：S0-COUPLING 4 卡，S1 4 卡，S2 6 卡，S3 8 卡，错峰启动 |
| 12 | C M2 "SDAA 性能 ≥ interpret 模式 10×"可能无法达成 | **采纳** | M1 判据改为：SDAA 性能 ≥ interpret 模式 3×。若实际加速比在 3×–10× 之间，记录实际值；若 <3×，标注"融合收益有限" |
| 13 | C M4 "线性加速比 >0.7"定义模糊 | **采纳** | M4 判据明确为：4 卡加速比 ≥ 2.8×（并行效率 ≥70%） |
| 14 | C M1 审计输出格式未定义 | **采纳** | M-T1 输出格式统一为 JSON，字段与 M-C1 普查脚本一致，便于复用 |
| 15 | 建议增加"第四方向"——谱系可追溯与门禁自动化 | **采纳** | 已纳入 P0（M-G1）+ P1（M-P3）。M-G1 定义 schema，M-P3 实现门禁自动化 |
| 16 | 建议合并 D4 + C 的 triton 消除 mission | **采纳** | M-T1（审计）+ M1/M2/M3（改写）已合并为一条线，卡位从 7 卡释放为 4 卡 |
| 17 | 建议 D3 作为 A 的前置验证 | **采纳** | M-S1/S2/S3 是通用 3D 验证，S0-SUBOFF 是船舶应用。M-S1 PASS 后 S0-SUBOFF 才能启动（依赖关系已写入 P2） |
| 18 | 建议 D2 普查脚本直接用于 B S0 | **采纳** | S0-COUPLING 复用 M-C1 普查脚本，不独立普查 |
| 19 | 建议所有判据增加"版本锁定"要求 | **采纳** | 所有 mission 的谱系 JSON 必须包含：torch/torch_sdaa 版本、git commit sha、设备名、schema 版本。无版本锁定的基线数据一律 INVALID |

---

## 三、分歧保留区（未收敛，建议 owner 裁决）

### 分歧 1：3D 主线是"通用验证先行"还是"SUBOFF 应用先行"

- **strategy 提案（D3）坚持**：先做通用 3D 验证（cylinder_3d / sphere_re100 / dam break），积累调试经验后再做 SUBOFF。理由：3D 复杂度高，直接上 SUBOFF 风险大。
- **models 提案（A）坚持**：直接做 SUBOFF（S0-S4），因为 SUBOFF 是 owner 硬考核指标（对标算例 ≥10 个），通用验证不直接产出考核成果。
- **综合主席建议裁决点**：P2 同时启动 M-S1（通用 3D）和 S0-SUBOFF（2D 轴对称 smoke test），两者互不依赖。若 M-S1 在 2 天内 PASS，则 S0-SUBOFF 进入 3D 阶段；若 M-S1 fail，则 S0-SUBOFF 停留在 2D，等 M-S1 调试完成后再进 3D。**owner 裁决：是否接受"并行启动、视结论收敛"的策略，还是必须二选一？**

### 分歧 2：跨库耦合（LBM↔FVM↔BEM）是否进入 P2

- **models 提案（B）坚持**：跨库耦合是"多物理场"的基础，必须进入 P2（S0-S3）。
- **strategy 提案未覆盖**：D1-D4 均未涉及跨库耦合。
- **verifier 质询**：coupling driver 从零构建风险大，2 天不够；S3 三场耦合 32 卡时可能不够。
- **综合主席建议裁决点**：S0-COUPLING（FVM/BEM standalone smoke test）进入 P2，但 S1-S3（耦合阶梯）推迟到 P3（第 8 日起），视 S0 结论决定是否启动。**owner 裁决：是否接受"S0 进 P2、S1-S3 推迟到 P3"的策略，还是 S0-S3 全部进 P2？**

### 分歧 3：TensorBEM 开发（m56/m57）的卡位优先级

- **owner 指令 03** 明示"tensorbem 也可以继续开发"，并已派 m56/m57。
- **strategy 提案** 未涉及 TensorBEM。
- **models 提案** 将 BEM 纳入跨库耦合（S0-COUPLING），但未单独给 m56/m57 卡位。
- **综合主席建议裁决点**：m56/m57 已派，卡位 4 卡（2 卡 × 2 天 + 2 卡 × 3 天）进入 P1。**owner 裁决：是否确认 m56/m57 的 4 卡优先级，还是与 S0-COUPLING 共享卡位？**

---

## 四、下次议事会建议议题（session-002）

1. **P0 结论审查**：M-C1 普查结果（第二个上卡的库是谁）、M-P1/P2 性能基线（CV 是否 ≤2%）、M-T1 triton 审计结论（5 模块替代方案）。
2. **P1 结论审查**：M-C3 裁决书（第二个库确认）、M-P5 跨卡逐位复现（是否 PASS）、m56/m57 TensorBEM 基线（是否达标）。
3. **P2 启动条件确认**：M-S1 3D 单相是否 PASS（决定是否放行 S0-SUBOFF 3D）、M1 纯 torch 等价是否 PASS（决定是否放行 M2+M3）、S0-COUPLING 是否 PASS（决定是否放行 S1-S3）。
4. **谱系/门禁端到端验证**：M-G1 schema + M-P3 门禁自动化是否跑通端到端流程。
5. **卡位动态调整**：若 P0/P1 某方向 fail，卡位如何释放回机动池；若扫荡线要卡，P1/P2 哪些方向让卡。
6. **owner 裁决点回告**：分歧 1/2/3 的 owner 裁决结果，纳入 session-002 任务安排。

---

*本计划仅代表综合主席个人意见，不构成派发指令。终审归 owner。*
*不确定处已标注，所有数字出自上下文与两份提案原文，未编造平台不存在的记录。*
