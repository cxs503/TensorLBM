# sphere_re100_mem — 球体绕流 Re=100 直接力法（BFL 稀疏核大域） ✅ VERIFIED

**状态：✅ 已入库（agent/m61 归档）** — 两档网格（D=40 / D=60）|err| ≤ 3% 且
随分辨率**严格单调下降**（+2.7708% → +2.5401%）。

判定口径 = **BFL 逐链动量账本（interpolated bouzidi 边界，实验室系），末 20%
采样窗均值 Cd**；参考预注册锁定后跑。

- 参考：Schiller–Naumann Cd_ref = **1.0917310910948732**（`24/Re·(1+0.15·Re^0.687)`）
- D=40（T1）：Cd = **1.1219804015117896** → **+2.7708%**（漂移 −0.145%）
- D=60（T2）：Cd = **1.119462001144486** → **+2.5401%**（漂移 −0.042%）
- 严格单调：|err_T1| > |err_T2| ✅
- 质量漂移：−0.080 ppm / 0.0 ppm；CV 窗闭合 0.015% / 0.008%
- 位一致链：T1 稀疏核对 W7 密集核 240 采样逐位相同；W7/W8-A D20 驱动逐位相同

---

## 1. 物理问题

均匀来流绕球（直径 D，Re = U·D/ν = 100），低于涡脱落下临界；定常轴对称尾流。
观测量 = 阻力系数 Cd = Fx / (½ρU²·πR²)，BFL 逐链动量账本（实验室系），末 20% 采样窗均值。

## 2. 参考口径

主锚：`Cd_ref = 24/100·(1 + 0.15·100^0.687) = 1.0917310910948732`（Schiller–Naumann）
交叉：S-N(0.681) = 1.0685190542670633，Clift–Gauvin = 1.1092345787735252
门：两档 |err| ≤ 3% 且 |err| 单调下降。

## 3. 四门标准核验

| 门 | 要求 | 实际 | 结果 |
|---|---|---|---|
| 共性模块入口 | `src/tensorlbm` 库路径驱动 | D3Q19 + `collide_mrt3d_low_memory` + BFL 插值边界（`bouzidi_bounce_back_d3q19`） | ✅ |
| 直接观测量 ≤3% | |err| ≤ 3% | T1 +2.7708%，T2 +2.5401% | ✅ |
| ≥2 档单调收敛 | |err| 随 D 严格单调降 | 2.7708% → 2.5401% | ✅ |
| 参考先锁定后跑 | 预注册冻结 | 参考 `result.json` 先锁，W8-A 后跑 | ✅ |

## 4. 两档结果

| 档 | D | 域 (nz×ny×nx) | BFL 链数 | 步数 | Cd（末 20% 窗） | err | 稳态漂移 | CV 闭合 | 质量漂移 (ppm) |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 40 | 280×280×320 | 28 722 | 12 000 | 1.1219804015117896 | +2.7708% | −0.1448% | 0.015% | −0.0797 |
| T2 | 60 | 420×420×480 | 64 794 | 24 000 | 1.119462001144486 | +2.5401% | −0.0420% | 0.008% | 0.0 |

## 5. 数据文件索引

W8-A 大域孪生结果文件位于 `../sphere_re100/w8a/`：
- `w8a/result.json` — 完整结果 + judgment（pass=true）
- `w8a/t1_d40big_12k.json` / `w8a/t2_d60big_24k.json` — 两档原始数据
- `w8a/verify.py` — 独立验证脚本
- `w8a/cv_instrument.py` — 控制体积动量平衡仪表
- `w8a/bit_check.json` — 位一致链验证
- `w8a/mem_sparse_D60big.json` / `mem_dense_D60big.json.oom.json` — 稀疏核解锁显存证据

三代历史档案（W5-B / W7-B）保留在 `pending/sphere_re100_mem/` 归档目录下。

## 6. 晋级路径说明

本案与 `sphere_re100`（pending）为同一物理问题的不同力法路径。`sphere_re100` 所有
测力路径（压力积分 / MEM 变体）均因阶梯球几何偏差 >3%，而本案采用 BFL 插值边界
+ 大域 + 稀疏核解锁路线已达标。`sphere_re100` 同步标记为 superseded，指向本案。

## 7. 局限

- 仅两档网格（受 GPU 32G 显存约束；D80 需 64G 卡或多卡分域）
- S-N 主锁位于交叉族低端（T1 对 Clift-Gauvin 差约 +1.1pp），锁定参考不重开
- W5-B / W7-B 世代结果记录为诊断档案，非判定用

<!-- PROVENANCE
[{"v": 1.0917310910948732, "f": "verified/sphere_re100/w8a/result.json", "k": "reference.cd_ref"},
 {"v": 3.0, "f": "verified/sphere_re100/w8a/result.json", "k": "reference.gate_pct"},
 {"v": 1.1219804015117896, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[0].cd_window"},
 {"v": 2.7707656824703957, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[0].err_pct"},
 {"v": -0.14475665095754808, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[0].drift_pct"},
 {"v": 1.119462001144486, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[1].cd_window"},
 {"v": 2.5400861325477218, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[1].err_pct"},
 {"v": -0.0420075943755402, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[1].drift_pct"},
 {"v": true, "f": "verified/sphere_re100/w8a/result.json", "k": "judgment.pass"},
 {"v": true, "f": "verified/sphere_re100/w8a/result.json", "k": "judgment.all_within_gate_and_steady"},
 {"v": true, "f": "verified/sphere_re100/w8a/result.json", "k": "judgment.monotone_error_decrease"}
]
-->
