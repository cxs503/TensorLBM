#!/usr/bin/env python3
"""Publish DE_A3_DIAG_COMPACT compression report to platform."""
import json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")

markdown = """# DE_A3_DIAG_COMPACT — 滴入水与相场入水基准诊断文言式紧缩

## 1 案例（物理背景 / 几何 / 参数）
源文档 `DE_A3_DIAG.md`（46 行）为 droplet_impact_3d_pool 与 phasefield water entry 两套基准的
诊断报告：13 个滴入水 run（D3Q19，8/12 cells/D，postcontact_healthy_ms 0.713~24.005，
impact_speed_relative_error 0.65%~23.40%）与 1 个相场入水 run（s2_zeta4_dw100，FAIL）。
本任务将其文言式紧缩为 `DE_A3_DIAG_COMPACT.md`（21 行），数字/单位/路径/命令/预测区间原样，
事实零删除只换措辞。

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）
- 读源文件（1 次）→ 统计行数（46）→ 定目标 35-50%（16-23 行）。
- 保留 13 行 run 表（核心数据，不可删）；终止原因表压缩为单行内联（×5/×5/×2/×1）；
  分析节（最远 Run/数据规律/Phasefield）各并成单行。
- 删除节间空行（原 5 处）以达 21 行；首行注明「原 46 行→压缩 21 行」。
- 双写：`docs/drop_entry_bb/DE_A3_DIAG_COMPACT.md`（工作树）+ 源目录同文件；git add+commit。

## 3 计算结果（数字表）
| 指标 | 原 | 压缩 | 变化 |
|---|---|---|---|
| 总行数 | 46 | 21 | -54.3%（达 45.7%，在 35-50% 目标内） |
| run 表行数 | 15 | 15 | 0（全保留） |
| 终止原因表 | 6 行 | 1 行内联 | 压缩 |
| 事实/数字 | 全 | 全 | 零删除（逐项核对） |

关键数字原样保留：24.005 ms、10.394 ms、2.31 倍、0.68%/1.78%、contact_step~21~51、
first_nonfinite_step=141、300-1100 steps、NaN/Inf、fy。

## 4 结果图
无图。理由：本单为纯文档紧缩任务（无计算/无场量/无解析对比），产物是 markdown 压缩文件，
不产生可绘制的数值曲线；行数-内容对照表（§3）即量化证据。

## 5 注意点（坑 / 局限）
- 行数约束与「事实零删除」冲突：run 表 13 行是硬下限，故只能压缩分析节与空行；
  若再压需合并 run 表（会破坏表格可读性），故取 21 行（45.7%）为平衡点。
- 首行「原 N 行→压缩 M 行」按 wc -l 计（含空行），与内容行数口径不同，已注明。

## 6 误差判定
本单为文档压缩任务，非数值基准——无「关键量 vs 基准 <3%」可判对象。
压缩保真度以「事实零删除」为判据：逐项核对 13 run 数据、终止计数、路径/命令/数字全部原样，
无任何数值改写或丢失（声明原因：无数值基准，故不适用 <3% 判据）。
"""

payload = {
    "title": "DE_A3_DIAG_COMPACT：滴入水与相场入水基准诊断文言式紧缩（46→21 行）",
    "summary": "将 DE_A3_DIAG.md（46 行）文言式紧缩为 DE_A3_DIAG_COMPACT.md（21 行，45.7%），"
               "数字/单位/路径/命令/预测区间原样、事实零删除只换措辞；13 个 run 表全保留，"
               "终止原因表与分析节压缩为内联单行；双写至 docs/drop_entry_bb/ 并 git commit。",
    "scenario_key": "de_a3_compact",
    "mission_id": 26275,
    "tags": ["TensorLBM", "DE-A3X-COMPACT", "droplet-impact", "phasefield", "文言紧缩", "document-compression"],
    "markdown": markdown,
    "figures": [],
}

res = post("/tensor/reports", payload, token)
print("report id:", res.get("id"), "| title:", (res.get("title") or "")[:60])
