# GAP-DIFFLBM-ADJOINT-VERDICT — 冻结场代理论文级判决：**判据修复 + 收敛尺度复核 + 论文分辨率运行实录**

* 锚点：`GAP-DIFFLBM-ADJOINT-VERDICT`（`adjoint.py` 冻结场代理判决）　任务：m21600
* 本轮增量：①判决门 `abs(cos)` 缺陷修复（**核心**：反平行梯级曾被判"通过"）；②分片campaign 合并器（同一 verdict 逻辑，不平移）；③收敛尺度（6000 步定常）实测复核；④论文分辨率 R2 运行 honest-FAIL 实录。
* 证据分级：**REAL（收敛尺度实测）** 用于 ③；**NOT_SUPPORTED（运行崩溃）** 用于 ④。

---

## ① 本轮改了什么、为什么

既有判决书（`docs/GAP_DIFFLBM_ADJOINT_VERDICT.md`，quick 700 步降规模冒烟）与
交叉核验（`docs/AUDIT_M20231_GAP-DIFFLBM-ADJOINT-VERDICT.md`，R1 36×64 收敛）都断言
「冻结场代理 cos ≈ −1 ⇒ 不得进优化路径」。但**判据实现与判据措辞不一致**：

| 位置 | 原文 | 后果 |
|---|---|---|
| `benchmarks/adjoint_verdict_diff_lbm.py` `run_config()` | `"ship_passes_gate": bool(abs(rec["ship"]["cos_vs_ad"]) >= gate)` | cos = −1（**完全反向**，最坏失效）时 `|cos| = 1 ≥ 0.99` ⇒ 判"通过" |
| 同上 `summarise()` | `passed = all(abs(...) < 0.99 ...)` ⇒ `frozen_field_surrogate_banned` | 反平行时代理**不被**判禁 ⇒ 总判决字段与总账验收（"cos < 0.99 ⇒ 禁止"）相反 |

总账验收原文是**有符号**的 `cos(展开式AD, 冻结场代理) 若 < 0.99，代理不得进优化路径`。
一处 `abs()` 把"完全反平行"与"完全同向"归为同一结论，这是本锚点最核心的正确性缺陷：
它让一个把优化推向反方向的梯度源看起来合格。修复：两处改为有符号比较
（`cos >= gate` 才通过；`cos < 0.99` 才判禁），并在代码注释中写明理由。
回归测试 `tests/test_adjoint_verdict_merge.py::test_merge_applies_the_benchmark_verdict`
用 cos = −1 与 cos = +1 两个合成分片**双向**断言，避免把预期答案写死。

同时补两处工程缺陷（都在本锚点的复现链路上）：

* `benchmarks/adjoint_verdict_diff_lbm.py --figures` 原本 `from postprocess_adjoint_verdict import make_figures`，**该模块不存在** ⇒ 该 flag 必崩。改为复用 `benchmarks/adjoint_verdict_report.py` 的单一实现。
* `benchmarks/adjoint_verdict_report.py::make_figures` 只在 `dirname(json)` 找 `fields_<cfg>.npz`，而 benchmark 实际写到 `dirname(json)/<basename>/fields_<cfg>.npz` ⇒ 按文档复现命令跑时**一张流场图都不会出**。改为两处候选路径搜索。
* 新增 `benchmarks/adjoint_verdict_merge.py`：论文分辨率 campaign 被拆成多个集群作业（`--only`）后，用**benchmark 自己的** `summarise()`（已从 `main()` 提取为函数）合一，杜绝"第二套判决逻辑"；并对 schema/mode/predeclared 不一致、config 重复直接拒绝合流（合成运行混一起比不合更糟）。

## ② 收敛尺度实测复核（REAL）

`cj3558`（node n315，8 核，rc=0，wall 149.8 s）：`--mode quick --only Q_yc050`，
**默认 6000 步**（前次入库的 700 步 < 对流时标≈640 步，是本次复核要排除的伪像）。
48×32，D=6 lu，τ=0.56，y_c=0.50H（对称点 ⇒ 基态 C_L = 0，符合对称性）。

| 量 | 值 | 说明 |
|---|---|---|
| FD 参考 `dC_L/dy_c` | **−2.9864e−03** | 中心差分 h=1.0/0.5 收敛解 |
| 展开 AD n=25 / 50 / 100 | +2.9847e−03 / +2.6368e−03 / +8.7286e−04 | 相对 FD 误差 2.00 / 1.88 / 1.29 |
| 冻结场上限（仅几何项） | +5.2516e−03 | cos(冻结, AD) = **+1.0000**，幅值比 **6.02** |
| **被判对象 `adjoint.py`（ship）** | **−6.3795e−04** | **cos(ship, AD) = −1.0000**，幅值比 0.731 |
| 优化一步（\|Δy_c\|=2 lu） | AD/冻结: 15.5→17.5，C_L 0 → **−6.31e−03**；ship: 15.5→**13.5**，C_L 0 → **+6.31e−03** | ship 把 y_c 推到**反方向**，目标变化号相反 |

判定（有符号判据，修复后）：`cos_ship = −1.0000 < 0.99` ⇒ **代理不得进优化路径**；
`adjoint.py` 已声明 `shape_derivative_only=True`、`optimization_path_eligible()` 恒定 False。
轨迹探针给出与余弦独立的**第三重证据**：同一目标、同一步长，ship 侧 C_L 变化号与 AD/冻结相反。

**必须同时记录的负面证据（honest）**：本次收敛尺度下 **AD ≠ FD**（n=100 仍有 129% 相对误差，
且**符号相反**）。故 AD 在此窗口尚未成为"总导数"参考，`ad_matches_fd_all_configs=false`；
判决中"冻结场 vs AD 余弦"一项的**方向**结论稳健（AD 与冻结都指向 +y_c、ship 指向 −y_c，两种参考下
ship 都被否），但**幅值**结论不可引用。有限时窗（n ≤ 100 步）对 Re≈15–40 的槽道–圆盘对
流时标不足，这是展开 AD 的适用边界，不是本锚点结论的反证。

## ③ 论文分辨率运行实录（NOT_SUPPORTED / honest-FAIL）

| 作业 | 配置 | 结果 |
|---|---|---|
| `cj3567` | R1_yc050,R1_yc055（128×96，D=16） | 递交后 56 min 仍在队列（`queued`），**未取回**，无结论 |
| `cj3568` | R2_yc050,R2_yc055（192×144，D=24） | 进程 **`malloc(): unsorted double linked list corrupted`**（rc=0 但输出为空壳），**崩溃，无结论** |

按纪律：**honest-FAIL 不得粉饰**。上表意味着论文分辨率（两分辨率 × 两参数点）的正式 campaign
**仍未交付**；`artifacts/adjoint_verdict_diff_lbm.json` 的论文级版本依旧缺位，本轮不伪造。
可复现的修复与合并器已就位，下一轮重跑的第一步应是：把 R2 的图内存问题定位
（unroll 图 × 192×144 的峰值内存）或把 `--threads`/`n_unroll` 降到稳定区，再重递交。

## ④ 注意点 / 局限

* `abs(cos)` 类错误是"判据实现漂移"的典型样本：**措辞有符号、实现取了模**。凡判决器都应有
  "反平行必须失败"的双向测试（本单已加）。
* quick 尺度（48×32、D=6）只服务 CI 与符号级判决；幅值结论必须等论文分辨率。
* 抢跑一次 700 步的降规模运行会把瞬态当成物理（cos=+1.000 的旧冒烟结果即如此），
  故本文件只引用 6000 步收敛运行。
* 合并器拒绝混流（schema/mode/predeclared 不同即退出），避免把不同 `max_steps` 的运行平均成一个判决。

## ⑤ 复现命令

```bash
# 收敛尺度复核（本次 cj3558，149.8 s / 8 核）
PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode quick \
    --only Q_yc050 --threads 8 --out artifacts/adjoint_verdict_diff_lbm_quick.json

# 论文分辨率 campaign（分片跑，再合并；cj3567/cj3568 实录见 ③）
PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode paper \
    --only R1_yc050,R1_yc055 --threads 12 --out artifacts/adjoint_verdict_diff_lbm_R1.json
PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode paper \
    --only R2_yc050,R2_yc055 --threads 12 --out artifacts/adjoint_verdict_diff_lbm_R2.json
PYTHONPATH=src python benchmarks/adjoint_verdict_merge.py \
    --out artifacts/adjoint_verdict_diff_lbm.json \
    artifacts/adjoint_verdict_diff_lbm_R1.json artifacts/adjoint_verdict_diff_lbm_R2.json
PYTHONPATH=src python benchmarks/adjoint_verdict_report.py \
    --json artifacts/adjoint_verdict_diff_lbm.json \
    --doc docs/GAP_DIFFLBM_ADJOINT_VERDICT.md --figdir figures

# 判据/合并器回归
python -m pytest tests/test_adjoint_verdict_merge.py tests/test_shape_gradient.py -q
```