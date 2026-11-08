# 平台缺陷留证：`POST /tensor/reports` 报告落库错投（mission 6456 / 2026-11-08）

按报告纪律，本单（m6456）在完成测试与 commit 后尝试发布论文级进展报告，`POST $TENSOR_API_BASE/tensor/reports`
三次失败，且失败形态表明**服务端没有使用请求体里的 title/summary/markdown**。TensorPlatform 属 2026-11-02
owner 控仓令范围（agent 不改码），故只留证不改。

## 复现（逐次记录）

| # | 动作 | 结果 |
|---|---|---|
| 1 | `POST /tensor/reports`，body 含 title/summary/markdown/figures(3×png_b64)/mission_id=6456 | **HTTP 500** `UniqueViolation: (mission_id, md5(markdown))=(6459, 851d759f...)` —— 我的请求被算到 **mission 6459** 上 |
| 2 | 同 payload，仅 markdown 追加一段（md5 变为 `c04e54fe...`） | **HTTP 200**，返回 `{"id":7683, title:"verification-debt 复核：resistance-benchmark-canon …"}` —— **标题/摘要是别的 mission 的内容**，不是我的 |
| 3 | `GET /tensor/reports?mission_id=6456&limit=5` | id 7683 存在、`mission_id=6456` 正确，但 **`markdown` 长度 = 0**、标题为 canon 复核稿 —— 正文丢失 |
| 4 | 重发同一 payload（验幂等） | **HTTP 500** `(mission_id, md5(markdown))=(6456, c04e54fe...) already exists` —— 服务端确实用**我的** markdown 算了 md5 并落库，但存进去的 title/summary/markdown 不是我的 |

## 判读

1. 幂等键 `(mission_id, md5(markdown))` 生效且 md5 来自我的请求体 ⇒ 请求体**被解析过**；
2. 落库的 title/summary 属于另一篇报告（canon 复核稿，同批次 7676/7682 内容），markdown 为空 ⇒
   写入路径读了**错误的字段来源**（疑似服务端复用了全局/队列缓冲的另一份 payload，或 `markdown` 字段
   名与 ORM 字段映射错位后 fallback 到最近一条报告）；
3. 第 1 次把 mission_id 写成 6459 ⇒ 同一错投路径下，`mission_id` 也可能取错（我的 body 里是 6456）。

## 影响与我自己能做的替代

- 本单报告正文未丢失：完整六要素稿在 `docs/roughness_laws_literature_verification.md`，
  提交用的完整 payload（含 summary/tags/figures 清单）在
  `artifacts/roughness_laws_audit/report_payload.json`，3 张图在 `artifacts/roughness_laws_audit/fig_*.png`。
- 建议（留给人/下一位）：给 `POST /tensor/reports` 加一条落库回读断言（写入后按 id 回读 title/markdown
  与请求体逐字比对），或在写库前把 `payload = await request.json()` 只取一次并存局部变量。