# TensorKnowledge 检索报告 — m8802 MCMP 两相流量返修

**任务**: m8802 旗舰闭环试验 — TensorKnowledge 赋能 MCMP 两相 Poiseuille 流量误差修复
**日期**: 2026-11-09
**检索范围**: TensorKnowledge knowledge/oss-lbm/claims/ (m8634 落地)

---

## 1. 检索目标

定位与 MCMP 两相流量守恒相关的既有 claim + evidence，为流量误差 >20% 的机理分析提供知识基础。

## 2. 检索到的关键 Claims

### 2.1 速度耦合约定

| Claim ID | 来源 | 核心内容 | 与本任务关联 |
|----------|------|----------|--------------|
| `tkc:oss-lbm:lbmpy-mcmp-barycentric` | lbmpy 教程 | 两组分 SC 使用重均速度 `u_bary = (ρ_a u_a + ρ_b u_b)/(ρ_a + ρ_b)` | 对照：TensorLBM "mixture" 模式 |
| `tkc:oss-lbm:sailfish-shan-chen-binary` | sailfish 实现 | 双组分 SC 使用松弛时间加权合成 `u = Σ_σ(ρ_σ/τ_σ)u_σ / Σ_σ(ρ_σ/τ_σ)` | **关键**：黏性加权速度 = 本次修复方案 |
| `tkc:oss-lbm:kruger-guo-force-term` | Krüger 教科书 | Guo 二阶力项 + 半力修正速度 `u += 0.5F/ρ_total` | 力项离散格式参考 |

### 2.2 验证方法

| Claim ID | 来源 | 核心内容 |
|----------|------|----------|
| `tkc:oss-lbm:kruger-poiseuille-validation` | Krüger ch.6 | Poiseuille 验证管线：relL2 误差 + 稳态判据 1e-12 |
| `tkc:oss-lbm:kruger-shan-chen-twocomponent` | Krüger ch.9 | 两组分 SC 骨架：指数势 + 8 最近邻 |

### 2.3 力项离散

| Claim ID | 来源 | 核心内容 |
|----------|------|----------|
| `tkc:oss-lbm:openlbmpm-porter-explicit-force` | OpenLBMpm | 显式力项处理 |
| `tkc:oss-lbm:lbmpy-force-models` | lbmpy | 多种力模型对比 |

## 3. 关键发现

### 3.1 速度耦合的三种约定

从知识库提取到三种不同的速度耦合方式：

1. **重均速度 (barycentric)**: `u_mix = (ρ₁u₁ + ρ₂u₂)/(ρ₁ + ρ₂)`
   - 来源: lbmpy, Shan & Doolen (1995)
   - TensorLBM 实现: `u_eq="mixture"`
   - 问题: 不等 τ 时不稳定（m8280 已验证发散）

2. **黏性加权速度 (viscous-weighted)**: `u_visc = (ρ₁u₁/τ₁ + ρ₂u₂/τ₂)/(ρ₁/τ₁ + ρ₂/τ₂)`
   - 来源: sailfish, Martys et al. (1996)
   - TensorLBM 实现: **本次新增** `u_eq="viscous"`
   - 优势: 强制界面速度连续 + 动量守恒 + 稳定

3. **自身速度 (self)**: 各组分使用自己的速度
   - 来源: 原始 Shan-Chen
   - TensorLBM 实现: `u_eq="self"` (默认)
   - 问题: 界面速度不连续 → 等效滑移 → 流量高估

### 3.2 误差机理推断

基于知识库证据，20-33% 流量误差的机理链：

```
"self" 模式 → 界面处两相速度不连续
           → 等效于界面滑移边界
           → 流动阻力降低
           → 流量高估 20-33%
           → 误差随网格加密不收敛（系统性偏差）
```

### 3.3 修复方案预测

**假说**: 采用黏性加权速度耦合 (`u_eq="viscous"`) 可强制界面速度连续，消除等效滑移，将流量误差降至 <10%。

**可证伪预测**:
- P1: H64 网格 max_rel_err < 10%
- P2: H128 网格 max_rel_err < 5%（网格收敛）
- P3: 质量漂移 < 1e-6/step
- P4: 数值稳定（不发散）

## 4. 引用路径

所有 claim 文件位于:
```
/root/tp-agent/work/m8634/knowledge/oss-lbm/claims/
├── tkc-oss-lbm-lbmpy-mcmp-barycentric.json
├── tkc-oss-lbm-sailfish-shan-chen-binary.json
├── tkc-oss-lbm-kruger-guo-force-term.json
├── tkc-oss-lbm-kruger-poiseuille-validation.json
└── tkc-oss-lbm-kruger-shan-chen-twocomponent.json
```

## 5. 结论

知识库明确记录了 sailfish 使用黏性加权速度耦合的事实（`tkc:oss-lbm:sailfish-shan-chen-binary`），这为本次修复提供了直接的理论依据。修复方案 = 在 `collide_sc_two_component` 中新增 `u_eq="viscous"` 模式，实现 sailfish 风格的速度耦合。
