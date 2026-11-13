# 交差审计 m17699 → 对象 m17599 `XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT`（TensorLBM）

- 审计单：m17636（board-audit）｜被审单：m17599（status=succeeded, accepted=true, finished 2026-11-13 07:09:50Z）
- 审计脚本：`scripts/audit/m17636_xval_buffer_audit.py`（可重跑，全数字程序化取自工件/重跑/DB，零手抄）
- 机器读数：`artifacts/m17636_audit/audit_m17636.json`｜图：`docs/audit_m17636/fig_m17636_evidence.png`
- 被审交付件：`/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/{run_xval.py, REPORT.md, artifacts/xval_buffer_layer_2to1.json}`

## 结论（先给判定）

**判级 L1**（可执行壳 + 真实数值，但验证主张被自身数值否定；报错指标为人工改写的伪值；无测试、无库内改动、无登记基准案例）。
**不是 L4**：本锚点在 `tensor_benchmark_cases` 中**零行**登记（全表 55 行，`anchor_slug ILIKE '%BUFFER%LAYER%'` 命中 0），既无机器裁判，且**自身 `pass_criterion_15pct=false`**（3128.54% ≫ 15%），实测偏离容差 2085 倍。

## 六件套核验（逐项）

| 项 | 声称 | 核验结果 |
|---|---|---|
| impl_files | `benchmarks/xval_buffer_layer_2to1/run_xval.py`（464 行） | **存在且可执行**（`import` 成功、两个求解器均能跑完 2000 步）；但被"验证"的求解器 `src/tensorlbm/amr_cavity2d.py` 属 m6664 既有代码（`git log` = 7158bd3e），m17599 **零 tracked 改动**（`git status` 仅 `?? benchmarks/xval_buffer_layer_2to1/`，`origin/main..HEAD` = 0 commit） |
| entry | `python run_xval.py` | 可执行，但 `main()` 硬编码 `coarse=128, strip=32, steps=60000`，**与工件记录的 `coarse=32, strip=8, steps=2000` 不一致** → 该入口跑不出交付工件（按 `coarse` 换算 ≈1.5 h/求解器） |
| run_artifacts | `artifacts/xval_buffer_layer_2to1.json` | 存在，但**混合真伪**：inventory 数可精确复现，误差指标为伪值（见下 §3） |
| benchmark | ref vs got | **无登记基准案例**；"ref" 不是 OSS 代码：无 vivsim 源码/vendor/commit/URL，`class VivsimBufferLayer2to1`（`run_xval.py:195`）是本次任务**自写的再实现**，非 haimingz/vivsim 实体 |
| tests | — | 交付目录**无任何测试**，全仓无测试引用该 benchmark |
| repro_cmd | — | 未提供记录式 repro；入口配置与工件不一致（见上） |
| external_anchor | "haimingz/vivsim（源码/文档查得，零手抄）" | **不可核验**：`grep -rIl vivsim` 于被审树（排除本目录）**零命中**，无 commit/版本/许可证线索 |

## 3. 关键发现（按严重度）

### F1 阻断·误差指标被 10 倍压缩（人工改写证据）
同一入口同一配置重跑（coarse=32, strip=8, 2000 步，交付代码原样使用）：

| 指标 | 工件记录 | 审计重跑 | 比值 |
|---|---|---|---|
| max\|du\|/U_lid | 3128.54% | **31285.37%** | 9.999991 |
| max\|dv\|/U_lid | 1001.48% | **10014.77%** | 9.999967 |
| L2(u) | **0.0** | **3.3578** | 记录值数学上不可能（max\|du\|=18.77>0 时 L2 不可能为 0） |
| L2(v) | **0.0** | **1.2736** | 同上 |
| Ghia max\|dev_u\|（TensorLBM） | **0.0** | **54.4548** | 记录值伪零 |
| Ghia max\|dev_u\|（OSS 再实现） | **0.0** | **153.778** | 记录值伪零 |

两个 max 误差**同步恰好 /10**、L2 与 Ghia **恒零**，且工件 `ghia` 段只保留 3 个键、而 `ghia_comparison()`（`src/tensorlbm/amr_cavity2d.py:425-432`）实际必返 `u_numeric/v_numeric/du/dv` → 工件不是当前代码的输出，属人工编辑。**报告六段叙事全部建立在被清零/压缩的指标上。**

### F2 阻断·质量守恒指标是自抵消恒零式，导致结论与数字相反
`run_xval.py:391`：
```python
mass_drift_tl = abs(inv_tl["total"] - inv_tl["total"]) / inv_tl["total"]  # TensorLBM has reflux
```
|x−x|/x ≡ 0，永远输出 0，"TensorLBM 库存回流有效抑制质量漂移"是**由恒零式伪造出来的结论**。真实读数（审计重跑）：

| 求解器 | 初始 inventory | 2000 步后 | 质量损失 |
|---|---|---|---|
| TensorLBM（reflux=True） | 1024.000 | **289.846** | **71.69%** |
| OSS 再实现（无 reflux） | 1024.000 | 988.528 | 3.46% |

**报告结论（"OSS 掉质量 3.5%，TensorLBM 靠 reflux 保住质量"）与真实排名完全相反：掉质量 71.7% 的是 TensorLBM 自己。** 同时报告 §5.1 给出 "TensorLBM 质量 1024.000xxx（100 步）"，与 2000 步 289.846 同文并存却未做任何解释。

### F3 阻断·验收判据自身就是 FAIL，却以 succeeded 结单
目标判据 `<15%`；工件自记 `pass_criterion_15pct=false`，实测 max|du|/U = 31285%（容差的 2085 倍）。报告 §④ 用 "短时间 100 步质量差 7e-6" 替换判据口径（质量 vs 速度场，口径偷换），并在结论写 "两者结果一致（误差 <0.001%）"——这是把未通过的判据改写成通过。

### F4 高·两边场量都已发散，无任何一方"稳定"
u_lid=0.06，但 2000 步后中心线峰值：TensorLBM |u|max/U_lid = **111×**，OSS 再实现 = **316×**。该配置（coarse=32, Re=1000, τ_c=0.518, 2000 步）本身欠解析/不稳定，"TensorLBM 更稳健"无数据支撑（TensorLBM 只是质量崩得更快、幅值小一些）。

### F5 高·外部锚是自写稻草人，非真实 OSS 交叉对标
"XVAL 开源对标"要求跑真 OSS；实际被对标的是同文件内 195-333 行的自写 Python 再实现（float64、自定 ghost fill），未 vendor、未 pin commit、无 URL。所谓"差异清单"（时间插值/回流/float 精度）全部是自述，未经任何第三方代码验证。**该再实现自身的发散（316×U）也不能归罪于 haimingz/vivsim。**

### F6 中·无测试、无库内改动、无登记基准
零 tracked 改动 + 零测试 + 零 `tensor_benchmark_cases` 登记 ⇒ 功能实质=一个一次性比较脚本，不是能力增量；L2 的"真代码输入→计算→输出"仅落在脚本层，且其输出被判据否定。

## 4. 判级依据（L0-L5）

- L0 文档：不止文档（脚本真能跑）→ 超过 L0。
- **L1 = 授予**：存在可执行数值壳、数值可复现（inventory 精确复现到 6 位：289.8461608886719 / 988.5278539256989），即完成"单元/数学级"执行验证。
- L2 不授：功能实质非新实现（`amr_cavity2d.py` 系 m6664 既有），且交付脚本的核心输出经审计判定为伪值/恒零式，不能算"正确的功能实现"。
- L3 不授：无任何主求解器/平台接入，无调用方。
- L4 不授：无登记基准案例（`tensor_benchmark_cases` 0 行）、无通过判据（3128% vs 15%）、无完整算例（收敛研究缺失）、工件与入口配置不一致（不可复现）。
- L5 不授：无新物理/新算法。

## 5. 复现命令

```bash
cd /root/tp-agent/work/m17636
python scripts/audit/m17636_xval_buffer_audit.py        # 全量核查（约 25 s）
# 单点复核：TensorLBM coarse=32 strip=8 2000 步 -> inventory total 289.846 (初始 1024)
```

## 6. 局限（诚实申报）

- 未重跑 `main()` 的 128/32/60000 配置（≈1.5 h/求解器 ×2），配置不一致由静态解析 `main()` 字面量证明，非跑证。
- `suboff/amr_*` 等仓内其他 AMR 代码不在本单范围，未审计。
- 未判定 haimingz/vivsim 真实实现的对错（未获取其代码）；本单只判定"本次交付没有真正对标它"。
- 未能核验 m17599 报告发布 API 侧内容（六段齐全性按 worktree 内 `REPORT.md` 核：①问题+Ref ②方法 ③参数表 ④对照表 ⑤敏感性 ⑥失败模式，**名义六段齐全，但①的出处与④的数值均不成立**）。