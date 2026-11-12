# nscbc_outlet_reflection —— NSCBC 出口反射系数验证面

**状态：✅ 仪器标定通过（两个解析锚 |R|=1 ±0.7%）；被测能力 `nscbc_outlet_2d`
修复后 PASS（默认 σ=0.25 时 |R|=0.31%，旧实现 11.2%）**

- 全文（六要素报告）：[`DOC.md`](DOC.md)
- 复算：`python run.py batch runs` → `python run.py assemble runs result.json`
  （`batch` 可分批，`single` 单档；回归门 `tests/test_nscbc_outlet_reflection.py`）
- 归档：`result.json`（判据 + 逐档读数）、`runs/*.json`（逐档原始工件）
- 被测/对照档：`nscbc` / `zerograd` / `neeq` + 两个解析锚 `pin`、`mirror`
