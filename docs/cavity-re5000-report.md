# CASE-CAVITY-RE5000 · 顶盖方腔 Re=5000（TensorLBM）— 第2轮状态报告

mission 17377 · 分支 agent/m17377 · repo TensorLBM

## ① 问题与参考出处
算例：二维顶盖驱动方腔，L=1、顶盖 U_lid 拖动，Re = U L / ν = 5000，直至定常。
参考：U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* **48** (1982) 387–411。
- Table III = u/U 沿竖直中线 x/L = 0.5（Re=3200/5000/7500/10000 列）
- Table IV  = v/U 沿水平中线 y/L = 0.5（同上）
- Re = 100/400/1000 的列在 Table I/II（生态内已有数字化副本，用于本单的"同码路径自检"）

**参考数据准入状态：PASSED**
GHIA_RE5000 已嵌入 `tensorlbm.verification.lid_driven_cavity.reference_data` 模块，
经机器验证来自 Ghia (1982) Table III/IV（审计见 commit 0dbf26ae）。
与第1轮不同，本环境现已具备 Re=5000 参考数据，可执行容差判据。

## ② 方法与实现
`benchmarks/verified/cavity/re5000/run.py`，复用 TensorLBM 已验证的腔流求解器：
- 格子 D2Q9；BGK 碰撞 + 半程反弹边界（静止壁）+ Zou-He 速度边界（顶盖）；
- 等 Re 配方 τ = 3 U_lid L / Re + 0.5，L = n-1（格子单位），U_lid = 0.1；
- 输出中线剖面（全分辨率）、涡心、残差历史写入 `result.json`；
- 参考表从库模块 `GHIA_RE5000` 读取，缺数据判 WITHHELD；
- 同码路径自检：Re=1000 对 Ghia Table I/II 必须先过 8% 容差。

## ③ 参数表
| 量 | 值 |
|---|---|
| Re = U L/ν | 5000 |
| U_lid（格子单位） | 0.1 |
| L（格子） | n-1（127 / 255） |
| τ | τ(128)=0.50766；τ(256)=0.51532 |
| 声速马赫 Ma = U/cs | 0.1732（压缩性误差 O(Ma²)≈3%） |
| 网格 | 128²、256²（float64） |
| 步数上限 | 50000 |
| 判据（预登记） | 逐点 |ref−got| ≤ 8% U_lid（Table III/IV 行）；128→256 的 u 增量须小于最细网格对参考的误差 |

## ④ 对照表（ref vs got + 误差）
**待运行** — 本报告提交时算例尚未在集群上执行（Re=5000 需较大计算量）。
run.py 已就绪，执行命令：
```
python benchmarks/verified/cavity/re5000/run.py --n 128 256 --steps 50000 --selfcheck
```
输出将写入 `benchmarks/verified/cavity/re5000/result.json`，含逐行误差表。

## ⑤ 代码验证
- 参考数据测试：`pytest TensorLBM/tests/test_lid_driven_cavity.py -k ReferenceData` → 7 passed
- GHIA_RE5000 数据完整性：15 个数据点，u/v 对称性正确，极值符合物理预期
- 求解器链路：CavityConfig + CavitySolver 可正常初始化并执行 step()
- 导入守卫：run.py 使用 `sys.path` 前置确保加载本 worktree 的 tensorlbm

## ⑥ 工件
- `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py` — GHIA_RE5000 参考数据
- `TensorLBM/tests/test_lid_driven_cavity.py` — 回归测试（含 Re=5000 数据验证）
- `benchmarks/verified/cavity/re5000/run.py` — 算例运行脚本（重写以适配当前求解器接口）
- 本报告

## 与第1轮的差异
1. **参考数据已准入**：GHIA_RE5000 经审计嵌入库模块（commit 0dbf26ae），不再 WITHHELD
2. **run.py 重写**：原版本导入不存在的 `collide_rlbm`/`zou_he_moving_lid`，现改用 `CavitySolver` 类
3. **网格规模调整**：从 384²/512² 降至 128²/256²（适配本地/集群资源限制）
4. **报告更新**：从 honest-FAIL 更新为待运行状态（基础设施就绪，待计算执行）
