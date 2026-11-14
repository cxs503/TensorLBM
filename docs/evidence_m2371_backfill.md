# Mission 2371 补报单 - 验证证据

## 1. 报告文件

- **路径**: docs/report_m2371.md
- **大小**: 13812 字节 (14K)
- **行数**: 254 行
- **MD5**: 413563ce7d579c6b22788810560421f3
- **创建时间**: 2026-11-13 22:54

## 2. API POST 结果

- **命令**: `curl -s -X POST $TENSOR_API_BASE/tensor/reports -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' -d @/tmp/rep.json`
- **退出码**: 0
- **返回 report_id**: 17733
- **readback.ok**: true
- **ledger_ok**: true

## 3. Digest 校验

| 字段 | Expected | Observed | 结果 |
|---|---|---|---|
| markdown | 43f2e655e954ad06784066bc585f979acd30d6e3182e3670ec74fe90faa8c903 | 43f2e655e954ad06784066bc585f979acd30d6e3182e3670ec74fe90faa8c903 | PASS |
| figures | 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945 | 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945 | PASS |
| mission_id | 8833ec27c2d477f05ce8764b47ae89a38c0356f70f3b9b85651d52038d54c1b5 | 8833ec27c2d477f05ce8764b47ae89a38c0356f70f3b9b85651d52038d54c1b5 | PASS |

## 4. 报告内容摘要

- **标题**: Mission 2371 补报：MMG 操纵性模型验证（honest-FAIL meta-report）
- **原单任务**: MMG 操纵性模型验证（验证欠账类任务）
- **证据等级**: 内部验证（两条独立路线互裁 + 收敛阶正确），非实船/实验基准
- **关键成果**:
  - 稳态回转（开环）：8/8 PASS，R_err max 0.010%，β_err max 0.012%
  - Zig-zag（闭环）：两条独立路线互裁达 1e-12 一致性
  - IMO MSC.137(76) 判据：对名义船 PASS，对降级船 FAIL，证明判据有鉴别力
- **局限声明**:
  - 绝对精度 WITHELD（无实船/实验基准）
  - 端到端流体-操纵耦合为结构阻塞（待后续任务）

## 5. 结论

本补报单成功完成，报告已写入 docs/report_m2371.md 并通过 API POST 到平台，report_id=17733，所有校验通过。
