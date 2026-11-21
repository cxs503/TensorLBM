# TRACK GAS WAVE 2 - DRAFT

## Self-Check
N/5 (Format OK, Arms=2, Pre-reg OK, No forbidden routes, Sources traceable)

## Rationale
定压闭包已验证稳定，需引入动态物理效应。本波聚焦气相质量源（蒸发）与气相粘性修正，二者均为最小改动内核的增量实验，旨在区分热力学与输运效应对空化稳定性的影响。

## ARM EVAP-MODE
task_key: DE-X-EVAP1-MODE
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 在气相连续性方程中添加基于压力差的蒸发/凝结质量源项 $S_m = \alpha \frac{p_v - p}{\sqrt{T}}$。参考 Hertz-Knudsen 公式简化形式。最小改动：仅在 `cells` 中启用 `gas_evap` 标志，不改内核结构，仅加源项。
prereg: 预测：空化云图出现非对称脱落，质量守恒误差增加但可控（<1%）。证伪：若质量误差 >5% 或计算发散，则当前简化源项不可用。

## ARM GAS-VISC
task_key: DE-X-GASV1-CORR
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 将气相粘度从常数改为 Sutherland 定律或压力相关函数 $\mu_g(p) = \mu_0 (p/p_0)^n$。参考 NASA CEA 数据。最小改动：修改 `gas_viscosity` 函数。
prereg: 预测：高压区气相粘度增加，界面阻尼增强，压力波衰减加快。证伪：若结果与常数粘度无显著差异（<1%），则气相粘性非主导因素。
