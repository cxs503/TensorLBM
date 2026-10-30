# Mission 1 补报：PG 闭环验证（honest-FAIL 元报告）

**Mission ID**: 1  
**Scenario Key**: report-backfill  
**Original Status**: 成功交付但缺报告  
**Report Type**: 诚实元报告（honest meta-report）  
**Date**: 2026-01-20  

---

## 1. 目的（Purpose）

Mission 1 的目标是验证 TensorPlatform 的 Protocol Gateway (PG) 闭环生命周期，确保任务从认领到验收的完整流程可追溯、可审计。具体验证的五步闭环为：

1. **claim**：Agent 认领任务（mission claim auth）
2. **start**：任务启动执行
3. **heartbeat**：执行期间心跳保活（心跳收割窗 90s→240s）
4. **complete**：任务完成
5. **accept**：结果验收

该闭环是 TensorPlatform 多 Agent 协作的基础设施，确保任务不会因 Agent 失联而悬挂，也不会因心跳超时而被误杀。

---

## 2. 方法（Methodology）

### 2.1 验证范围

PG 闭环验证覆盖以下组件：

- **Mission Claim 认证**：`backend/migrations/090_factory_claim_principal.sql` 中的 `claimed_user_id` 字段绑定执行声称到认证账户
- **心跳机制**：Agent 在执行期间定期发送心跳，看门狗线程监控进度
- **任务状态机**：`queued → running → completed/failed` 状态转换
- **看门狗保护**：防止 Agent 失联导致任务悬挂（900s 超时保护）

### 2.2 验证方法

原单声明采用端到端集成测试：
- 创建测试任务
- Agent 认领任务（claim）
- 启动执行（start）
- 周期性发送心跳（heartbeat）
- 完成任务（complete）
- 验收结果（accept）

### 2.3 代码审计路径

本次补报审计了以下关键文件：
- `backend/api/gateway.py`：API 网关路由管理
- `backend/services/gateway_service.py`：网关服务（路由、限流、熔断器、负载均衡）
- `backend/api/research_factory.py`：科研工厂任务生命周期
- `backend/migrations/090_factory_claim_principal.sql`：任务认领认证
- `schemas/factory-evidence-claims.schema.json`：工件绑定结构化声称

---

## 3. 关键结果（Key Results）

### 3.1 原单声明

原单声明：**PG 闭环验证通过**

五步生命周期（claim→start→heartbeat→complete→accept）端到端验证成功。

### 3.2 工件数字

**⚠️ 诚实声明：工件目录未留档**

由于原单交付时未归档工件目录，以下数字一律标注"未留档"，不得编造：

| 指标 | 数值 | 备注 |
|------|------|------|
| 任务认领延迟 | 未留档 | claim API 响应时间 |
| 心跳间隔 | 未留档 | heartbeat 周期（原设计 90s→240s） |
| 心跳丢失率 | 未留档 | 心跳超时判定阈值 |
| 任务完成时间 | 未留档 | start→complete 耗时 |
| 验收通过率 | 未留档 | accept 阶段验证结果 |
| 看门狗触发次数 | 未留档 | 900s 超时保护触发 |
| 任务状态转换次数 | 未留档 | queued→running→completed |

### 3.3 代码证据

从代码审计中提取的机制证据：

1. **Mission Claim Auth**（git commit `34082c3`）：
   - 修复：`Fix mission claim auth: compare UUID token as canonical string`
   - 确保 claim 认证的 UUID token 以规范字符串比较

2. **心跳收割窗**（git commit `d533a44`）：
   - 改进：`mission refill 端点：换班不换人（agent_lost/budget_exceeded 补预算重排）+ 心跳收割窗 90s→240s`
   - 心跳窗口从 90s 扩展到 240s，减少误杀

3. **看门狗跨 mission 误杀修复**（git commit `e96f663`）：
   - 修复：干净收尾（纯文本→request_acceptance）不设 stop_reason
   - `run()` 返回后看门狗线程不退役，`_last_progress` 冻结
   - 900s 后 `os._exit(137)` 杀守护连同下一个在飞 mission
   - 修复方案：`run()` 进入武装/finally 退役（`_conv_done`）

4. **Factory Claim Principal**（`090_factory_claim_principal.sql`）：
   ```sql
   ALTER TABLE tensor_dev_missions
       ADD COLUMN IF NOT EXISTS claimed_user_id BIGINT REFERENCES users(id);
   CREATE INDEX IF NOT EXISTS ix_tensor_missions_org_claim
       ON tensor_dev_missions (organization_id, runner, id) WHERE status = 'queued';
   ```
   - 将执行声称绑定到认证账户而非调用者提供的名称

---

## 4. 审计与局限（Audit & Limitations）

### 4.1 审计发现

**正面发现**：
- ✅ PG 闭环五步生命周期代码路径完整
- ✅ Mission claim 认证机制已修复（UUID token 规范比较）
- ✅ 心跳窗口已优化（90s→240s）
- ✅ 看门狗保护机制已加固（防止跨 mission 误杀）
- ✅ Factory claim principal 已绑定认证账户

**负面发现**：
- ❌ **工件目录未留档**：原单交付时未归档验证过程的工件（日志、时间序列、性能指标）
- ❌ **数字不可复现**：由于缺少工件，无法从代码中程序化提取验证数字
- ❌ **报告缺失**：原单成功交付但未发布论文级进展报告

### 4.2 局限性

1. **诚实 FAIL 结论**：
   - 本补报为 honest-FAIL 元报告
   - 原单声明"PG 闭环验证通过"如实保留
   - 但所有定量数字标注"未留档"，不得编造

2. **不可复现性**：
   - 由于工件缺失，无法验证原单声明的具体数字
   - 无法复现验证过程的详细指标（延迟、吞吐、超时率）

3. **代码审计 vs 运行时验证**：
   - 本报告基于代码审计和 git 历史
   - 未进行运行时验证（原单已验证，但工件未归档）

### 4.3 改进建议

1. **强制工件归档**：
   - 所有 mission 交付必须归档工件目录（日志、时间序列、性能指标）
   - 工件必须包含 sha256 校验和，确保可追溯

2. **强制报告发布**：
   - 所有 mission 交付必须发布论文级进展报告
   - 报告未发 = 本单未完成（守望线会自动派补报单并记欠账）

3. **数字程序化提取**：
   - 报告中的数字必须从工件程序化提取，零手抄
   - 使用 `python3 -c json.load` 等工具从 JSON 工件提取键值

---

## 5. 结论（Conclusion）

Mission 1 的 PG 闭环验证在原单中已成功交付，五步生命周期（claim→start→heartbeat→complete→accept）端到端验证通过。代码审计确认了关键机制的完整性：

- Mission claim 认证已修复
- 心跳窗口已优化
- 看门狗保护已加固
- Factory claim principal 已绑定认证账户

**然而**，原单交付时未归档工件目录，导致：
- 所有定量数字标注"未留档"
- 无法复现验证过程的详细指标
- 未发布论文级进展报告

本补报为 honest-FAIL 元报告，如实保留原单声明，但明确标注工件缺失和数字不可复现的局限性。

**教训**：
- 工件归档是交付的必要条件，不是可选项
- 报告发布是交付的最后一步，未发报告 = 未完成
- 数字必须从工件程序化提取，零手抄，零编造

---

**Report Metadata**:
- Title: Mission 1 补报：PG 闭环验证（honest-FAIL 元报告）
- Scenario Key: report-backfill
- Mission ID: 1
- Tags: ["补报", "honest-FAIL", "PG闭环", "元报告"]
- Figures: []

