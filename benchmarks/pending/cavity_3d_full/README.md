# 全 3D 方腔流 cavity_3d_full（5 壁 + 移动顶盖，Re=1000）—— SUPERSEDED

**状态：⏸️ superseded — 本案达标数据已归档至 `benchmarks/verified/cavity_3d_full/`。本目录为历史诊断稿（MRT/RLBM 发散记录），不再作为独立判决使用。请转至 [../verified/cavity_3d_full](../verified/cavity_3d_full/) 查看完整验证结果。**

## 归档说明

本案例已于 2026-08-20 通过严格四门标准验证，归档至 `benchmarks/verified/cavity_3d_full/`：
- 64³：u_min = −0.27142，误差 1.338% vs Ku (1987)
- 96³：u_min = −0.27581，误差 0.259% vs Ku (1987)
- 严格单调收敛，verified = true

本目录（pending）仅保留 MRT/RLBM 碰撞格式选型诊断的历史记录，已无独立达标数据。
PR 归档后本目录可安全删除，诊断价值已并入 verified 侧 result.json 描述字段。

<!-- PROVENANCE
[{"v": 1.338, "f": "../verified/cavity_3d_full/result.json", "k": "grids.64.err_pct_vs_ku"},
 {"v": 0.259, "f": "../verified/cavity_3d_full/result.json", "k": "grids.96.err_pct_vs_ku"},
 {"v": true, "f": "../verified/cavity_3d_full/result.json", "k": "verified"}]
-->
