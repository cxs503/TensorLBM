# 两相 Poiseuille（SC-MCMP 双组分）— W4-B Part 1 判决记录（Wave-4 多孔介质弧）

日期 2026-09-21 · Agent W4-B（GPU6）· 控制器亲验 2026-09-21
staging: `/nfs/wangxi/runs/bm_widen_w4_20260920/washburn_twophase/`（NOTES.md 预注册 +
修正案 #1/#2 + explore/exp1–exp42 证据链）
库: tensorlbm `multiphase.collide_sc_two_component` + `porous_media` 模块（经
`w4b_lib.py` 编排，零手写物理核）。

## 一句话结论

**FAIL（反收敛）**：H64/H128 中心区最大相对误差 **26.82% / 33.17%**（阈值 ≤3%，
超出 9–11 倍），且**细化恶化**（26.82→33.17，单调判据反向）。误差是 SC-MCMP
在该模型点的**结构性动量耦合偏差**，非 Ma、非互溶度、非驱动强度（归因见下）。

## 判定表（verify.py 独立复算 = result.json）

| 档 | 中心区节点数 | max rel err | mean rel err | u_max 比 | Ma | 判定 |
|---|---|---|---|---|---|---|
| H64 (ny=65) | 55 | 26.82% | 14.21% | 1.26524 | 0.0467 | FAIL |
| H128 (ny=129) | 110 | 33.17% | 15.57% | 1.31458 | 0.1943 | FAIL |
| 单调 | | 26.82 → 33.17 | | | | FAIL（反收敛） |

中心区 = |u| > 0.2·u_max 掩模（两种掩模口径 simmax/anamax 结论相同）。
参考解 = 库自身 `porous_media._two_phase_poiseuille_analytical`（名义黏度比 M=2；
**用实测黏度比重标定参考被预注册明令禁止，未做**）。

## 配置

| 项 | H64 | H128 |
|---|---|---|
| τ (water, gas) | (1.0, 0.75) | (1.0, 0.75) |
| G_12 | −2.5（全交换分层） | −2.5 |
| G_x（体积力） | 5e-6 | 5e-6 |
| 名义 M = ν_w/ν_g | 2.0 | 2.0 |
| 界面 | tanh-3（y=half） | 同 |
| 步数 | 60 000 | 180 000 |
| 实测块互溶度 ρ_g-in-water / ρ_w-in-gas | 0.3001 / 0.1593 | 0.3000 / 0.1515 |
| 伪流 \|uy\|_max | 0.0794 | 0.0797 |

## 归因（诊断链，控制器复核）

- **G_x 不变性**：误差比在两种驱动强度下同为 1.265 → 非 Ma/可压缩性。
- **有效黏度比**：因 ~30% 块互溶度，名义 M=4 实测有效 M≈0.54（库缺陷 #5）；
  用实测比修正解析只把 u_max 比从 1.265 闭合到 1.24 —— 修正不动 26–33% 的
  剖面偏差，方向也不对。
- **根因登记（深物理项，未修）**：`collide_sc_two_component` 各分量用**自身**
  速度（非混合物质心速度）构造平衡态 → 两相动量耦合结构性偏差。这是本 FAIL
  的候选根因，属库核心核改动，需另立项（W4-D 修复范围之外）。

## 预注册链（sha256，服务器时钟 +0800）

- 2026-09-20 23:52:07 prereg 快照 `40e5c803…083fb`
- 2026-09-20 23:59:33 修正案 #1（τ → Design A (1.0, 0.75)；中间哈希行被
  tee-overwrite 丢失，已披露）
- 2026-09-21 00:51:32 修正案 #2 `26e2f9f5…00dca2`（σ 判决 + 缺陷 #9 + Part-2
  协议锁定）
- Part 1 formal 先于修正案 #2 运行，配置由修正案 #1 锁定（时序无害）。

## 控制器独立验证（2026-09-21）

全部判决数字由控制器独立实现复现（`w4b_recheck.py`，非 agent 的 verify.py）：
两层 Poiseuille 解析解 = 控制器自写 3×3 线性解 vs 存档剖面 **逐位一致
（max|diff| 8.1e-16）**；两档 max/mean rel err、u_max 比、掩码节点数全部复现。
铁律 grep（`def collide|stream|equilibrium|bounce|zou_he|far_field`）零命中。

## 库缺陷指针（详见姊妹记录 capillary_invasion_washburn/README.md，共 10 条）

与本记录直接相关：#1 G_12 文档/校验符号错误（分相实需 ≤−2.5，默认 0.9 是
混合模型）；#5 大 τ 对比下有效黏度比反转（互溶度 30%）。

## 文件

- `run.py` — 正式驱动（原 staging `formal/part1_poiseuille.py`，仅改路径头）
- `w4b_lib.py` — 共享驱动库（初始化/驱动/测量，物理核全部来自库函数）
- `result.json` — verify.py 独立复算判决（Part 1 切片 + 预注册链 + 铁律检查）
- `out_part1.json` — 正式运行原始输出（逐点剖面/收敛史/互溶度）
- `log_part1.txt` — 运行日志

复现：`python run.py`（需空闲 CUDA 设备，约 10 分钟 GPU6）。
