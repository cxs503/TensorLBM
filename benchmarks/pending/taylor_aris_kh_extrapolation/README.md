# taylor_aris_kh_extrapolation（pending：kH 口径欠账清偿，待战役机门复判）

**对象**：`src/tensorlbm/taylor_dispersion.py`（m7568）与
`benchmarks/verified/taylor_aris_dispersion` 周期剪切臂的全部读数都取自单一轴向长宽比
`L/H = 8`（`kH = 0.785`），有限波数偏置从未量化。

**结论**：出厂 "≤0.93 %" 是**两个反号偏置的偶然抵消**——有限波数偏置（正，∝Pe²）与
横向离散偏置（负，∝1/H²）。kH→0 外推给出核的真实口径误差
（Pe=2/5/8/12：−0.057/−0.282/−0.525/−0.757 %），横向加密 `1/H²`（`R²=0.999997`）外推
`H→∞` 后残余 **+0.030 %**。Pe=12 处抵消达 **1.685 pp 且符号翻转**。
所有口径均 <3 % ⇒ `verified` 判定维持；需改的是归因文本。

**用法**
```bash
python run_kh.py --list
python run_kh.py --all --out-dir artifacts        # 21 算例，约 6 min 单机 CPU
python run_kh.py --assemble --out-dir artifacts   # -> artifacts/results.json
python run_kh.py --capture --figs --out-dir artifacts
pytest -q tests/test_taylor_aris_kh_extrapolation.py
```

**铁律**：只调用库算子（`taylor_dispersion.run_periodic_shear_dispersion` →
`passive_scalar` D2Q5 三件套），零手写物理核。

文档：`docs/taylor_aris_kh_extrapolation_20261111.md`；结果：`artifacts/results.json`；
图：`figures/{kh_extrapolation,kh_bias_mechanism,shear_field}.png`。
