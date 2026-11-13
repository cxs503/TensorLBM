import pathlib

p = pathlib.Path('/root/tp-agent/work/m17492/docs/board_audit/m17492/insert_verdict.py')
s = p.read_text()
old_l2 = '        "L2": "met: real code, input->computational->output, artifacts on disk, executable repro_cmd (standalone D2Q5 scalar solver + two boundary kernels)",'
new_l2 = ('        "L2": "NOT met: the anchored functional capability (anti-bounce-back Dirichlet) is absent - the '
          'shipped kernel is bit-identical to the equilibrium wet node (max profile diff 0.0e+00 at H=20/40/80), a '
          'form the library already has at src/tensorlbm/convective_mass_transfer.py:441; the D2Q5 solver is a '
          'benchmark-local duplicate with no src/ entry point, so no new functional capability was added to TensorLBM. '
          'What exists is a verification driver (analytic linear-profile comparison) = L1 substance. Aligning with the '
          'm17085 precedent (verification driver whose claimed capability is absent -> L1).",')
assert old_l2 in s
s = s.replace(old_l2, new_l2)
s = s.replace('        "level": 2,', '        "level": 1,')
s = s.replace("VALUES ('TensorLBM', $json$%s$json$, 17492, 2,", "VALUES ('TensorLBM', $json$%s$json$, 17492, 1,")
p.write_text(s)

q = pathlib.Path('/root/tp-agent/work/m17492/docs/board_audit/m17492/AUDIT_m17450.md')
t = q.read_text()
t = t.replace('- 判决：**L2**（`request_changes` / reopen）。', '- 判决：**L1**（`request_changes` / reopen）。')
t = t.replace('**L2** ✅ 达成：真代码、输入→计算→输出、可复现、有工件（判 L2 的全部依据）',
              '**L2** ❌ 未达成：锚点功能（ABB）不存在——交付核与 equilibrium 湿节点逐位相同，而该形式库内已有'
              '（`src/tensorlbm/convective_mass_transfer.py:441`）；D2Q5 求解器是基准脚本内置副本、无 `src/` 入口，'
              '未给 TensorLBM 增加任何新功能能力。存在的是「解析解对照验证驱动」= L1 实质'
              '（对齐 m17085 判例：验证驱动 + 声称能力缺失 → L1）')
t = t.replace('→ **level = 2，disposition = request_changes / reopen**',
              '→ **level = 1，disposition = request_changes / reopen**（gate 语义：gate_level≤1 且原单 succeeded → 自动开返修轮）')
q.write_text(t)
print('patched ok')