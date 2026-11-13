INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT 'XVAL-LBM-INSTRUCTION-DSL-TESTS-SH', 'CH-POISEUILLE-ANALYTIC-DSL', '解析解（Stokes，半程反弹壁面 y=1/2 与 ny-3/2）', 'max_rel_error：审计重算 0.68559%(ny=22,H=20) / 0.34831%(ny=32,H=30)；交付 0.686%/0.348%（吻合）', '<15%(owner) 且 <3%(自设)；worst 0.686% 双阈值通过。二阶收敛未证（隐含阶 p=1.670）', 'TensorLBM', 'passed', now()
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INSTRUCTION-DSL-TESTS-SH' AND case_name='CH-POISEUILLE-ANALYTIC-DSL');
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT 'XVAL-LBM-INSTRUCTION-DSL-TESTS-SH', 'TAYLOR-GREEN-DECAY-DSL', '解析衰减率 2 nu k^2（nu=0.018518519 由 tau 声明）', 'nu 相对误差：审计 0.0809%(32^2)/0.1822%(64^2)（sample_every=1）；交付 0.0132%/0.1036%（sample_every=2）', '<3%；两采样口径均通过，但 KPI 依赖拟合窗口，引用须连同采样参数', 'TensorLBM', 'passed', now()
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INSTRUCTION-DSL-TESTS-SH' AND case_name='TAYLOR-GREEN-DECAY-DSL');
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT 'XVAL-LBM-INSTRUCTION-DSL-TESTS-SH', 'PYLBM-SCHEME-OPERATOR-PARITY', 'pylbm 0.12.0 @ /root/oss-cache/pylbm commit f9f5d1647ccc10aa2fc7a93690cc56450432ee24（scheme.py:506/238）', 'M max abs diff=0.0；权重 diff=0.0（=教科书 4/9,1/9x4,1/36x4）；f_eq max abs diff=1.110e-16（8 组随机宏观态）', '<=1e-12 机器精度通过；限定：仅算子层、单进程，非整解场 parity', 'TensorLBM', 'passed', now()
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INSTRUCTION-DSL-TESTS-SH' AND case_name='PYLBM-SCHEME-OPERATOR-PARITY');
SELECT id,case_name,status FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INSTRUCTION-DSL-TESTS-SH' ORDER BY id;
