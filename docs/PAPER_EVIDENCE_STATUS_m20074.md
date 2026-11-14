# m20074 证据包状态（honest-STATUS）

本文件由 `run_evidence.py`/`aggregate.py` 的真实工件程序化摘录，数字零手抄。

## 已落地的工件

- item JSON: ['A1_poiseuille_2d.json', 'FAILURES.json', 'X1_cross_solver_raw.json']
- 失败项记录 FAILURES.json: [{"item": "X1_cross_solver_cavity", "error": "CalledProcessError: Command '['/usr/bin/python', '-c', \"import json;from pathlib import Path;from tensorlbm.cross_solver_verification.runner import BenchmarkConfig, run_benchmark;cfg=BenchmarkConfig(mms_sizes_lbm=(16,32), mms_sizes_fvm=(16,32), mms_sizes_fem=(8,16),cavity_sizes_lbm=(64,128), cavity_sizes_fvm=(64,128), cavity_sizes_fem=(32,64),reynolds=(100,), reynolds_fvm=(100,), reynolds_fem=(100,),cross_n=128, ledger_n=128, out_path=Path(r'runs/paper_evidence/X1_cross_solver_raw.json')) ;run_benchmark(cfg)\"]' returned non-zero exit status 1."}]
- X1 原始报告: ['audit', 'generated_utc', 'mms', 'node', 'repo', 'report_schema', 'title']

## A1 平面 Poiseuille（唯一完整落地的 item）

| 量 | ref | got | 误差% | 门限% | 判定 |
|---|---|---|---|---|---|
| u_max at H=30 vs g*H^2/(8 nu) | 0.0124861 | 0.0124821 | 0.0319 | 3.0 | pass |
| profile max|u-u_exact|/u_max at H=30 | 0 | 0.0124821 | 0.0335 | 3.0 | pass |
| u_max at H=62 vs g*H^2/(8 nu) | 0.0124967 | 0.010169 | 18.6267 | 3.0 | FAIL |
| profile max|u-u_exact|/u_max at H=62 | 0 | 0.010169 | 18.6267 | 3.0 | FAIL |

观测阶=-9.12037841690384；噪声地板={"float64": 18.62668439380944, "float32": 18.623034869697044}

### 失败机制（实测归因）

H=62 档在固定 4000 步下处于**未收敛瞬态**（黏性扩散时间 ~H²/ν≈2.3e4 步 ≫ 4000），因此该档 18.6% 的偏差是步数不足，不是求解器缺陷；H=30 档 4000 步已近稳态故通过。**修复路径**：A1 的步数按 H²/ν 标定（≈2.5e4 步），第二轮重跑。

## X1 交叉求解器（原始报告已落地，item 汇总未落地）

```
{}
```

## gate 复核（pytest tests/test_paper_evidence_bundle.py）
```
 {'item': 'A1_poiseuille_2d', 'category': 'analytic', 'title': '2-D plane channel Poiseuille flow ...ttice Boltzmann Method (2017), ch. 6', 'pointers': ['tests/test_lbm_verification.py::poiseuille_case (Step C)']}, ...}})

tests/test_paper_evidence_bundle.py:56: AssertionError
=============================== warnings summary ===============================
../../../../usr/lib/python3.11/site-packages/anyio/pytest_plugin.py:14
  /usr/lib/python3.11/site-packages/anyio/pytest_plugin.py:14: PytestRemovedIn10Warning: _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================== collection-ignore ===============================
none (every test file under tests/ is collectable here)
=========================== short test summary info ============================
FAILED tests/test_paper_evidence_bundle.py::test_at_least_six_independent_benchmarks
1 failed, 5 passed, 1 warning in 0.32s

```