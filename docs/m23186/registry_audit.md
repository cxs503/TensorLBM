# TensorLBM capability registry audit

- capabilities: 25 (mature 4 / developing 21 / planned 0)
- wired 14, verified 4
- findings: 4 {'DOC_CLAIM_CONFLICT': 4}

| code | capability | detail | doc | disposition |
|---|---|---|---|---|
| DOC_CLAIM_CONFLICT | lbm.collision.mrt.d3q27 | 文档断言生产可用/已验证，registry maturity=developing（wired=True verified=False） | d3q27_composition_evidence_r1.md:17 | 在案（已裁定：该行说明的是 D3Q27 composition 的 *范围*（R1 已验证 D3Q19/MRT 组合），按行触发扫描器；registry 的 d3q27 MRT 标 developing 是有意的（无 D3Q27 解析误差带），不改文） |
| DOC_CLAIM_CONFLICT | lbm.collision.mrt.d2q9 | 文档断言生产可用/已验证，registry maturity=developing（wired=True verified=False） | d3q27_composition_evidence_r1.md:17 | 在案（已裁定：同上：跨句上下文误判，不构成对 D2Q9 MRT 的成熟度断言） |
| DOC_CLAIM_CONFLICT | lbm.amr.reflux_restrict | 文档断言生产可用/已验证，registry maturity=developing（wired=True verified=False） | d3q27_composition_evidence_r1.md:17 | 在案（已裁定：同一 composition 范围表述的按行触发；README 的生产级措辞已改文） |
| DOC_CLAIM_CONFLICT | lbm.amr.reflux_restrict | 文档断言生产可用/已验证，registry maturity=developing（wired=True verified=False） | report_m2463.md:164 | 在案（已裁定：历史报告归档件按纪律不改写；README 的生产级措辞已改文（见本次改动）） |
