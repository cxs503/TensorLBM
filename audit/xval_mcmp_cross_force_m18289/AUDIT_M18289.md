# AUDIT m18289 — XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP

Board audit (mission m18348, `verdict_by='board-audit'`). Object: mission **m18289**
("返修轮1 | maturity_gate 自动裁定" of m18126), status `succeeded`, repo TensorLBM.
This is a **cross-check sheet, not a redo**: no new capability is claimed here.

Independence limits: the auditor did not re-run the OSS arm (lbmpy codegen needs the
pinned sdist in a cluster Python) and did not re-run the repo two-phase simulation;
it did execute the *delivered* operator module against the *repo* kernels in this
worktree, and it recomputed the inherited physics-arm error from the pinned `.npz`.

## ① 案例介绍 (what the artifact is supposed to be)

Anchor `XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP`: cross-validate the
TensorLBM two-component Shan–Chen (MCMP) cross-force operator against the live OSS
implementation **lbmpy 2.0**. Physics: D2Q9 two-component lattice with pseudopotentials
ψ, coupling `G_12` (repo, `<0` repulsive) vs `g_ab` (OSS, `>0` repulsive),
bridge `G_12 = -g_ab`, symmetric D2Q9 stencil so gather direction is irrelevant
(forward `Σ w c ψ(x+c) = -Σ w c ψ(x−c)`).

Delivered tree (m18289, 2 files, untracked, never committed):

| file | lines | content |
|---|---|---|
| `xval/mcmp_cross_force/__init__.py` | 28 | docstring layout + `ANCHOR`, `CRITERION_PCT=15.0` |
| `xval/mcmp_cross_force/operators.py` | 138 | `repo_cross_force*`, `oss_cross_force`, `psi_exp/psi_lin`, `stream_periodic` |

Preserved byte-identically for the record: `refs/{__init__,operators}.py`,
sha256 in `MANIFEST.sha256`
(`operators.py` = `bcb03ce105513c1012a19707dac702b23c6ffcb72eff771587857a7030ea245c`).

## ② 怎么做的 (audit method)

1. **Evidence existence** — read the mission record (`/api/v1/tensor/missions/18289`),
   the worktree `/root/tp-agent/work/m18289`, and `git status`: only two files, untracked.
2. **Three-piece check** — impl files: present; entry: absent; run artifacts: absent.
3. **Referee check** — `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='…'` → **0 rows**.
4. **Executed probes** (`probe_delivery.py`, fp64, torch 2.7.1, repo `src/tensorlbm`):
   import the pinned copy; identity of the delivered wrapper with
   `tensorlbm.multiphase.sc_two_component_force`; OSS operator vs repo kernel;
   stencil coefficient vs `cs²=1/3`; uniform-field nil; exp-ψ composition; streaming.
5. **Inherited-number check** (`probe_chain_physics.py`): recomputed the one physics-arm
   number m18289's headline quotes, from `docs/board_audit/m18219/refs/artifacts/*.npz`
   at commit `6ad01d71`, under the delivering code's own metric
   (`||got−ref|| / ||ref − mean(ref)||`, `run_xval.py:177-182`).
6. **External anchor** — re-verified the lbmpy 2.0 sdist sha256 and the vendored
   upstream test file + cited lines.

## ③ 计算结果 (machine-written numbers)

Operator level (delivered code, executed here):

| probe | quantity | value |
|---|---|---|
| P2 | wrapper vs library kernel | bitwise equal (`max_abs_diff = 0.0`) |
| P3 | OSS operator vs repo kernel, linear ψ | worst rel **3.41e-16** (Fx2 2.91e-16, Fy1 3.41e-16) |
| P4 | forward gather coeff / k | 0.33333334 (theory 0.33333333) |
| P4 | backward gather coeff / k | −0.33333334 (antisymmetry residual 2.2e-16) |
| P5 | uniform field force | 0.0 (both operators) |
| P6 | exp-ψ composition vs OSS | worst rel **2.72e-16**, ψ_exp(1)=0.6321205588285577 |
| P7 | streaming | mass rel drift 0.0, equals hand-written pull (0.0) |

Inherited physics arm (m18126 artifacts, recomputed):

| arm | rel-L2 (delivering metric) | NaNs |
|---|---|---|
| N64 g=5 exp | **200.3399 %** (plain-norm 120.3451 %) | 0 |
| N64 g=6 exp | n/a | 64/64 |
| N64 g=6 linear | n/a | 64/64 |
| N128 g=6 exp | n/a | 128/128 |

→ 1 of 4 arms evaluable, and it misses the owner 15 % bar by **13×**; the FAIL is
robust to the normalisation choice.

Referee table: **0 rows** for this anchor (`tensor_benchmark_cases`), so no
ref-vs-got adjudication is possible for this anchor at all.

## ④ 图

* `fig1_operator_identity.png` — per-component identity residual + stencil coefficient.
* `fig2_physics_arm.png` — inherited physics arm vs the 15 % / 3 % bars, NaN arms marked.
* `fig3_evidence_matrix.png` — evidence keys against the maturity-gate scheme.

## ⑤ 注意点 (traps / limits)

* The delivered module is **real** code, not a stub — the missing level comes from the
  missing run chain (entry/artifacts/commit), not from wrong math. Grading it L0 would
  be as wrong as grading it L2.
* `refs/__init__.py:9-18` advertises 7 files; **5 do not exist**, including
  `run.py` ("`python -m xval.mcmp_cross_force.run`") and `tests/`. Docstring overclaim.
* Nothing of m18289 is in any branch: `git status` = `?? xval/mcmp_cross_force/`.
  The gate's `files_exist` roots are `/root/tensor-ecosystem` and
  `/root/tp-agent/work/m18348`, so this audit preserves the files under `refs/` with a
  sha256 manifest — otherwise the evidence would evaporate with the worktree.
* `tensor_reports` has **0** rows for mission 18289; the 6 rows under
  `/api/v1/reports?mission_id=18289` are platform `equipment_status` placeholders
  (`status=generating`, `size=0`, `file_url=null`), not a paper report.
* The upstream lbmpy test also carries `g_aa`/`g_bb`; the delivered comparison sets them
  to 0 (pure cross force). Legitimate for an operator-level XVAL, but it is a scope
  restriction, not a full model equivalence claim.
* Definition sensitivity: quoting "2.00 %" for the physics arm (as the earlier round did)
  is a fraction/percent read error; the same artifact is 200.34 % under its own metric.

## ⑥ 误差判定 (vs baselines, explicit)

| key quantity | baseline | got | error | <3 %? |
|---|---|---|---|---|
| operator identity (linear ψ) | repo kernel `sc_two_component_force` | delivered OSS operator | 3.4e-16 rel | **yes** |
| exp-ψ composition | same | same | 2.7e-16 rel | **yes** |
| stencil coefficient | analytic `cs²=1/3` | 0.33333334 | 3e-9 rel | **yes** |
| physics arm profile rel-L2 | OSS/lbmpy arm | repo arm | **200.34 %** | **no (13× over the 15 % bar)** |
| benchmark tolerance check | `tensor_benchmark_cases` | — | **impossible: 0 referee rows** | **not run** |

**Verdict: level 1** (math/unit validation, auditor-executed). L2 denied (no entry, no
run artifacts, nothing committed). L3 denied (no solver-side caller). L4 denied (no
referee row, no `repro_cmd`). L5 denied (no external benchmark; only the OSS pin and
transcribed lines verified). Disposition: **request changes / reopen** — the rework
scope is: entry emitting `.npz`/`.json`, a pytest suite, referee registration for the
anchor, and a docstring that describes only files that exist.

Reproduce:

```bash
cd audit/xval_mcmp_cross_force_m18289
python3 probe_delivery.py        # -> probe_delivery.json
python3 probe_chain_physics.py   # -> probe_chain_physics.json
python3 make_figs.py             # -> fig1..fig3
```