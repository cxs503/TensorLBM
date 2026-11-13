# 交差审计 m18798 — XVAL-LBM-R2-SELECTIVE-FUSED-GHOST-EXCHANGE-PLUS-3（TensorLBM）

审计单：m18817（board-audit）。对象：mission m18798，状态 succeeded，commit `630c7eab`
（"XVAL R2: selective (lbmpy PackInfo) ghost exchange in D3Q19GlooTransport + corrected 3D TGV anchor"）。
本单**只核验、不重做**；所有 got 数值均由本审计独立重跑或从交付工件程序化读取，不引用作者叙述。

## 0. 结论（一句）

**L3 确认（真实实现 + 产品路径集成 + 17 个测试独立复跑通过），L4 不予确认**：
① 该锚点在本平台基准裁判表 `tensor_benchmark_cases` 中**0 行**（无可机器复核的 ref/tolerance）；
② 交付的 repro driver `run_xval.py` 在预声明矩阵（8 配置里 7 个是 xval 类）上**聚合阶段崩溃** `KeyError: 'kind'`，
`table.csv` / `gates.json` 从未由它产生，仓库里的数字由另一个脚本 `make_artifacts.py` 生成。
无虚标判级行为（报告主动披露了 n=16 23.35% 超 15% 带、U0=0.04 8.83%、cluster 未交付），故不是降级惩罚，而是 L4 的两项硬要件缺失。

## 1. 取证：三件套逐项核验

| 要件 | 结论 | 指针 / 证据 |
|---|---|---|
| run_artifacts 文件 | **真存在** | `xval/XVAL-LBM-R2-SELECTIVE-GHOST-3DTGV/artifacts/`：`local_arms.json`(9753B)、`local_ladder.json`(5683B)、`fields_local-n32-r2-lin.pt`(24.9MB)、`decomposed_local-n32-r2-lin.pt`(19.9MB)、`serial_field_local-product-n32-r4.pt`(4.98MB)、`shipped_local-product-n32-r4.pt`(4.98MB)、`fig1..fig4 *.png` |
| benchmark ref vs got | **基准行缺失（WITHHELD）** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-R2-SELECTIVE-FUSED-GHOST-EXCHANGE-PLUS-3'` → **0 rows**（全表 113 行，无 `%SELECTIVE%` 锚点；`tensor_xval_cases` 亦 0 行）。故 ref/tolerance 无法机器比对 |
| repro_cmd 可执行 | **部分可用（aggregation 崩）** | `python run_xval.py --only R2-product-n32-r4` → 正常出 gates/table；`--only R2-n32-r2-f64 --only R2-product-n32-r4` → 跑完 2 个配置后 `KeyError: 'kind'`（`run_xval.py:326 → _arm_rows:151`），`campaign.run_config` 返回体不含 `kind` 键 |
| 报告六段齐全 | **是（六段+§7 差异清单+§8 误差判定）** | `REPORT.md`：§1 问题+ref 出处（lbmpy `_extend_dir`、m18155 审计）、§2 方法、§3 参数表（n/U0/tau/steps/Re_k/Ma/k/nu）、§4 ref vs got 打勾叉、§5 收敛与敏感性（分辨率/U0 阶梯/黏度/rank 数）、§6 失败模式与工件 + 摩擦（cluster 阻塞、knowledge 端点 engine_offline）、§7 vs OSS 取舍、§8 逐项「是否 <3%」 |

## 2. got 数值的独立复核

本审计用交付工件与独立重跑交叉验证（不采信文字叙述）：

| 量 | 交付工件（程序化读取） | 本审计独立重跑 | 一致? |
|---|---|---|---|
| 产品路径 rel L2（n=32, 4 ranks, 100 步, U0=0.01, f64） | `local_arms.json:product.tgv_rel_l2 = 0.02734681058539495` | `run_xval.py --only R2-product-n32-r4` → `2.734681e-02`（`artifacts/m18817/rerun_run_xval_product_n32_r4/table.csv`） | **位级一致** |
| 该臂 identity vs monolithic | `identity_max_abs_vs_serial = 0.0` | 重跑 `0.000e+00` | 一致 |
| 线字节比（selective/full） | 163840/622592 = 0.2631578947368421（n_cross 5 vs 19） | 探针实测发送比 0.2631578947368421 | 一致 |
| n=16 分辨率点（U0=0.01, 200 步） | `local_ladder.json: tgv_rel_l2 = 0.233526393311974` | —（未重跑，引用工件） | 超 15% 带，报告 §4/§8 已标 FAIL |
| n=32 / n=64（U0=0.01） | 0.031224627922259864 / 0.020298028717629726 | —（引用工件） | 3.12% / 2.03%，<15% 通过 |
| 非线性 rung（n=32, U0=0.04, 100 步 f64） | 报告 §4 记 200 步为 +8.830% | 重跑 100 步 → 9.0237% | 趋势一致（步数不同） |

测试独立复跑（在 m18798 检出上，`PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`，未写入该树）：

```
tests/test_xval_tgv3d_anchor.py + tests/test_xval_mpi_ghost_fused.py   -> 14 passed in 12.64s
tests/test_multi_gpu_d3q19_selective_ghosts.py                          ->  2 passed in 14.01s
tests/test_multi_gpu_d3q19_torchrun.py (默认路径回归)                    ->  1 passed in 10.84s
```

## 3. 实现真实性与两个独立探针（adversarial）

实现不是 MOCK/STUB：`src/tensorlbm/multi_gpu.py:941-1010` 由 D3Q19 模板真实推导跨越集
（`crossing_right = cx=+1` 5 个、`crossing_left = cx=-1` 5 个），`:1215-1320` 做 pack→isend/irecv→unpack→仅跨越集校验；
`stream()` 的 pull 移位证据：`cx=+1` 只读 `padded[...,0]`，`cx=-1` 只读 `padded[...,-1]`，`cx=0` 完全不读 ghost。
`src/tensorlbm/xval_mpi_ghost_fused/case.py` 把 3D 单模 TGV 参考指数从 2νk² 改为 ν·|lap eigenvalue|（3νk² 连续、6(1−cos k) 格点），并给出 `re_k` 适用域。

**探针 A（未写满的 ghost 是否真未被读）**：`artifacts/m18817/probes/probe_unused_ghost_entries.py`
—— 子类化后在跨越集之外的 ghost 平面注入 NaN，与干净跑对比：
`max_abs_diff_poisoned_vs_clean = 0.0`，输出无 NaN，`n_cross_per_face = [5,5]`。
→ 「非跨越集 ghost 永不读」为**真**（不是巧合对齐）。

**探针 B（线字节账目）**：`artifacts/m18817/probes/probe_comm_accounting.py`（2 gloo ranks, n=16, slab=8, 1 步）

| 项 | default(all-19) | selective | 比值 |
|---|---|---|---|
| isend / irecv 次数 per step | 4 / 4 | 4 / 4 | 1 |
| 实测发送字节 per step per rank | 155648 | 40960 | **0.2631578947368421 = 5/19** |
| `last_exchange_bytes`（账目口径） | 77824 | 20480 | 0.2631578947368421 |
| 实测 / 账目 | 2.0 | 2.0 | — |

→ 相对结论（5/19）**成立**；绝对字节账目**系统性低记 2 倍**，原因见 F4。

## 4. 发现（按严重度）

**F1（阻断 L4，证据级）基准裁判无此锚点。** `tensor_benchmark_cases` 0 行，`tensor_xval_cases` 0 行 ⇒ 拿不到
case_name/metric/tolerance，本平台无法机器裁定 got 是否在容差内。报告里的 15% 是 owner 通用带，不是注册基准。
补救：注册 3 条 case（product rel L2 / n=32 rel L2 / 拟合衰减率），tolerance 分别为 <15%（或 5%）、<15%、<5%。

**F2（阻断 L4，可复现级）repro driver 在预声明矩阵上崩溃。** `run_xval.py:326` 对 `campaign.run_config`
的返回体调用 `_arm_rows`，而该返回体无 `kind` 键（shipped 路径有）⇒ 只要含 xval 类配置即 `KeyError: 'kind'`，
`table.csv` / `gates.json` 不会落盘。仓库 artifacts 里也确实没有这两个文件，说明交付数字来自 `make_artifacts.py`
（`--stage ladder/arms/report`），而 REPORT §6 「the numbers here come from the same driver executed locally」
**表述不准确**（同 campaign 模块、不同 driver）。复现记录：
```
$ python run_xval.py --only R2-n32-r2-f64 --only R2-product-n32-r4
[R2-n32-r2-f64] kind=xval relL2=9.0237% wall=22.0s
[R2-product-n32-r4] kind=shipped relL2=2.7347% wall=10.9s
KeyError: 'kind'   (run_xval.py:326 -> _arm_rows:151)
```

**F3（中）门槛对空测量集 fail-open。** 单跑 shipped 配置（`--only R2-product-n32-r4`）得到
`gates.json` 全 `pass: true`，但其中 `H2_byte_ratio_5_of_19.measured = []`、
`H4_regime_monotone_in_u0.ladder_u0_rel_l2 = []` —— 零测量仍判 pass。子集跑会给出**空洞全通过**的假象，
若 auditor 只看 gates.json 会被误导。

**F4（中，非阻断，属既有缺陷）校验路径二次交换导致字节账目低记 2 倍。** `step()` → `stream(padded)` →
`validate_ghosts(padded)`（peer=None）→ 选择性模式下再跑一次 `_selective_exchange`（default 模式同样再发一次全 19 群），
实测每步 4 次 isend/4 次 irecv 而非 2/2。因两臂都翻倍，**比值 5/19 与「3.8x 字节削减」的相对结论仍成立**；
但 §3 表里的 622592 / 163840 是账目口径，真实流量为 2 倍。属 `HEAD~1` 即有的既有行为（`git show HEAD~1:src/tensorlbm/multi_gpu.py` 中 `stream()` 已调用 `validate_ghosts(padded)`），本 commit 未引入、也未修正。

**F5（低，诚实披露成立）两条 rung 超出 15% owner 带。** n=16（U0=0.01）23.35% > 15%，n=32 U0=0.04 8.83% > 3%。
报告 §4 直接标 FAIL、§8 明确回答「否」，未粉饰 —— 但因此「15% 带已满足」只对 n≥32 且线性区成立。

**F6（低）预声明矩阵未按预声明交付。** `predeclared.json:predeclared_run_matrix` 8 配置（含 n=64 r2/r4、float32），
实际交付为本地 ladder(n=16/32/64, U0=0.01) + arms(n=32) + product(n=32 r4)，dtype/steps/u0 与预声明标签不一致；
cluster 堵塞已在 §6 披露，但「预声明 vs 交付」的逐项对照缺失。

**F7（低）H1 的 dtype 覆盖未兑现。** H1 声明 float32 与 float64 均位精确；交付工件的 xval 臂只有 float64
（预声明中的 float32 标签未跑），物理解析锚也只在线性 float64 上被复核。

## 5. 判级依据（L0–L5）

- L1 数学/单元：`tests/test_xval_tgv3d_anchor.py`（3νk² 指数、k²/12 特征值差、拟合 ν 5%、非线性可见性）通过。
- L2 独立功能实现：真代码，输入→计算→输出（pack/isend/irecv/unpack/pull-stream，5/19 群），探针 A 独立证实。
- L3 系统集成：**达成**——selective 能力落在已发布产品类 `D3Q19GlooTransport` 上（`selective_ghosts=True`），
  产品臂 4 ranks 由 `run_xval.py:84 transport.step()` 驱动，本审计重跑得 2.7347% 且 identity=0.0；
  注意尚无主求解器（`MultiDeviceSolver3D` / `app/` / `examples/`）调用该开关，集成面为止于运输类。
- L4 基准验证：**不达成**——F1（无注册基准）+ F2（driver 崩，无法一键复现）；且 F5 有一条 rung 越 15% 带。
- L5 科研工程：不适用。

## 6. 处置（disposition）

**approve with notes @ L3**：实现与集成真实、无虚标、测试独立通过；L4 待 F1/F2 修复后复核。
返修建议（按顺序）：① 修 `run_xval.py` 聚合（`campaign.run_config` 补 `kind`，H2/H4 空测量改 fail-closed）；
② 注册 `tensor_benchmark_cases` 三条 case 并在 CI 里跑；③ 把 §3 字节表改成「实测含校验」口径，
或在 `stream()` 中复用 `exchange_ghosts` 已得的 peer 结果以消掉二次交换；④ §6 措辞改为「同 campaign 模块、
make_artifacts.py 驱动，run_xval 矩阵未跑」。

## 7. 本审计未做（unknown / not_run）

- 未重跑完整 8 配置矩阵与 n=64 两臂（时间预算）；n=64 数值仅取自交付工件。
- 未在 cluster（外协方双节点）上复现；未验证 `artifacts/*.pt` 内部浮点是否与重跑逐位相同（仅比对派生标量）。
- 未审计 R1（m17995）交付物本身，仅按 m18155 审计结论引用。
- `make_artifacts.py` 三个阶段未重跑（会在 m18798 树内写文件；本审计不写入对象树）。