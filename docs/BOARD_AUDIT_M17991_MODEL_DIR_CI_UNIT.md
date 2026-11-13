# 交差审计 m18104 — m17991 XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT（TensorLBM）

**裁决**：**L4（基准验证级）**，approve with notes。verdict 已落 `tensor_maturity_verdicts`（id=690, verdict_by=board-audit）；基准 3 行补注册 `tensor_benchmark_cases`（status=passed）。

## ① 对象与判据

- 对象：m17991（CFD-GO/TCLB 的 model-dir-as-build-and-ci-unit 移植，status=succeeded，auto-approve dev_acceptance）。
- 唯一裁判：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT'` + 审计独立复跑。
- 交付时该 anchor **0 行**，mission result 带 `server_quality_audit={gate_valid:false, gate_missing_warning:true}`（同先例 m17968 交付态）。

## ② 验收三件套核验（审计实跑，非读报告）

| 件 | 结果 |
|---|---|
| run_artifacts | ✅ 实存：`artifacts/xval_m17991/{xval.json,sensitivity.json,fig1-3.png}`、`builds/*.build.json`×3、`runs/*/result.json`×3、`runs/model_units/*/result.json`×3、`artifacts/model_units/{ci_matrix.json,Makefile.models,*.build.json}` |
| benchmark ref vs got | ✅ 8/8 行 got 达标：物理 KPI 2.706e-04 / 1.098e-03 / 2.500e-03（阈值 0.15，余量 >99%）；组织度量 C1 0.00%≤15%、C1b 0≤0、C2 0≤0、C3 0.0085≤0.05、C4 100%≥85% |
| repro_cmd | ✅（带条件）`PYTHONPATH=src python tools/xval_m17991_model_dir_ci_unit.py` 实跑 rc=0，**8 行 got 值与交付 xval.json 逐位一致**，all_pass=true；TCLB 侧数字亦一致 |

## ③ 独立复算与测试

- `pytest tests/test_model_units.py -q`（PYTHONPATH=src）→ **10 passed**（25.6 s）。
- `python -m tensorlbm.model_units verify` → ok=true，units=3（flow/poiseuille2d_d2q9_guo, flow/poiseuille3d_d3q19_guo, flow/tgv2d_d2q9_bgk）。
- 敏感性阶梯（sensitivity.json）：ny 16/32/64 误差单调降（1.098e-03→2.706e-04→6.496e-05）；tau 0.6/0.8/1.2 三点；TGV n 16/32/64 衰减率误差 0.6%→0.25%→0.17%——误差分析真实可解释。
- 报告六段齐全（docs/XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT.md ①–⑥，程序化生成，数字零手抄）。

## ④ 判级（L0–L5）

- L1 单元验证=过（10 测试 + 解析锚 KPI）；L2 独立功能=过（model_units 真代码：manifest 解析→隔离 build→case 计算→KPI 输出，零 MOCK）；L3 系统集成=过（Makefile 新增 models/models-build/models-test/models-doctor 目标 + `.github/workflows/model-units.yml` 生成式矩阵 + 主包模块）；**L4 基准验证=过**（3 完整算例 + 逐项误差表 + 4 敏感性阶梯 + 审计独立复跑逐位一致）；L5 不宣称。
- 附百页报告不改变级别；级别=功能实质。

## ⑤ 发现（severity 排序）

1. **[中] 基准未注册（交付时）**：tensor_benchmark_cases 0 行 ⇒ 本审计独立复跑后补注册 3 行（passed）。
2. **[中] repro_cmd 环境脆弱**：裸 `python tools/xval_m17991_model_dir_ci_unit.py` 在当前环境失败（`kernel exposes no equilibrium entry point`）——根因：site-packages 的 tensorlbm 指向另一工作树 m17918（其 d3q19 无 equilibrium3d），而 build.py:37 的 probe 子进程不继承父进程 sys.path、工具未自设 PYTHONPATH。registry.py:105 的 cmd 模板自带 `PYTHONPATH=src`，但 xval 工具未对齐。建议：工具入口自设 `os.environ["PYTHONPATH"]`。
3. **[低] 指纹漂移**：交付 xval.json 的 unit fingerprint 与当前文件哈希不符（case.py/model.toml 于 11:17:15 批量恢复，晚于原 xval.json 写入）；KPI 逐位不变 ⇒ 数值中性，仅 provenance 瑕疵。
4. **[低] 交付物全 untracked**：git status 全 `??`，mission result head_commit=null ⇒ 平台自动推送无从推起，main 无该模块（同 m17968 F3）。
5. **[低] tensor_reports 无 m17991 报告**：本审计补发。

## ⑥ 裁定

L4 成立：基准与能力成立，confirmed 语义=消账。发货资格另需 commit/合并 main（F3 类缺口）与工具 PYTHONPATH 收口（建议另立小节点，不阻塞本单）。
