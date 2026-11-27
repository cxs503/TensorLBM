# m28104 historical DFG root-cause investigation

This is the internal investigation merged by #3086, archived without changing the numerical algorithms, current benchmark verdict, frozen series or submitted result values. The archived table/figure helpers were run against the same actual frozen series before and after relocation; the root-cause solver probe was only imported, not executed. New replay outputs go to ignored `outputs/historical/m28104/` rather than overwriting the original JSON/PNG. The obsolete `--rejudge-only` documentation example was replaced by the existing explicit level argument. This is offline path compatibility, not a scientific rerun.

- [REPORT.md](REPORT.md) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/REPORT.md))
- [cl_cd_table.json](cl_cd_table.json) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/cl_cd_table.json))
- [m28104_dfg_cl_amp.png](m28104_dfg_cl_amp.png) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/m28104_dfg_cl_amp.png))
- [make_figs.py](make_figs.py) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/make_figs.py))
- [reproduce_table.py](reproduce_table.py) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/reproduce_table.py))
- [root_cause_isolate.py](root_cause_isolate.py) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/root_cause_isolate.py))
- [root_cause_verdict.json](root_cause_verdict.json) ([original at immutable main revision](https://github.com/cxs503/TensorLBM/blob/d8fa649255e58a7941fdd5e79c5e1b72eeb76abb/docs/m28104/root_cause_verdict.json))
- [Existing pending DFG matrix case manifest](../../../benchmarks/pending/cylinder_re100_dfg/benchmark_case.json)
- [Original object and concurrent path-correction receipt](../../../docs/architecture/migrations/m28104-dfg-concurrent-path-corrections-2026-11-28.json)
