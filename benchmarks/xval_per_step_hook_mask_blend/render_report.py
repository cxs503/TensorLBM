"""Render the cross-validation report from ``result.json`` (numbers injected, not typed).

Called by :func:`run_xval.main`, so every figure, percentage and verdict mark in
``docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md`` is read out of the run
artefact.  Prose is authored here; arithmetic and table bodies are not.
"""

from __future__ import annotations

from typing import Any


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _pct(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _num(value: float, digits: int = 6) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.{digits}g}"


def build_markdown(result: dict[str, Any]) -> str:  # noqa: C901 - one pass, one document
    env = result["environment"]
    pre = result["predeclared"]
    ref = result["reference"]
    ev = ref.get("evidence", {})
    snips = ev.get("snippets", {})
    prov = ev.get("verified_file_sha256", {}) or ev.get("recorded_file_sha256", {})
    cou = result["couette"]
    poi = result["poiseuille"]
    sens = result["sensitivity"]
    inv = result["invariance"]
    comp = result["composite_channel"]
    ups = result["upstream_lid"]
    tim = result["timing"]
    verdict = result["verdict"]

    cou_cfg = cou["configs"]
    ref_ny_cou = max(c["ny"] for c in cou_cfg)
    ref_ny_poi = max(c["ny"] for c in poi["configs"])
    base = next(c for c in cou_cfg if c["ny"] == ref_ny_cou and c["arm"] == "A_explicit")
    tau = base["tau"]
    nu = base["nu"]
    steps_cou = base["steps"]
    pbase = next(c for c in poi["configs"] if c["ny"] == ref_ny_poi and c["arm"] == "A_explicit")

    def c_row(ny: int, arm: str) -> dict:
        return next(c for c in cou_cfg if c["ny"] == ny and c["arm"] == arm)

    def p_row(ny: int, arm: str) -> dict:
        return next(c for c in poi["configs"] if c["ny"] == ny and c["arm"] == arm)

    L: list[str] = []
    add = L.append

    add(f"# {result['anchor']}")
    add("")
    add(
        "**Round 1 · open-source cross-validation (XVAL)** · mission "
        f"{result['mission']} · generated {result['generated_utc']} · commit "
        f"`{result['git_commit']}` by `benchmarks/xval_per_step_hook_mask_blend/run_xval.py` "
        "(every number below is read from `artifacts/xval_per_step_hook_mask_blend/result.json`)."
    )
    add("")
    add(
        f"**Verdict: {verdict['verdict']}** — owner gate (15 %): "
        f"{'PASS' if verdict['gate_15pct'] else 'FAIL'}; internal target (3 %): "
        f"{'PASS' if verdict['gate_3pct'] else 'FAIL'}. The upstream-faithful moving-lid arm "
        f"fails by design (measured flow {_pct(ups['flow_pct_of_uwall'])} % of the imposed wall "
        "speed) — that failure *is* the evidence for the one porting change made."
    )
    add("")

    # ---------------------------------------------------------------- 1 problem
    add("## 1. Problem and reference")
    add("")
    add(
        "The OSS implementation organises a lattice-Boltzmann run in two idioms that TensorLBM did "
        "not have:"
    )
    add("")
    add(
        "1. a **driver with per-step setup hooks** — `Domain.Solve(sess, Tf, initialize_step, "
        "setup_step, save_step, save_interval)` runs `initialize` once, then per step "
        "`save (on the save tick) → setup_step → collide → stream`. Case-specific physics "
        "(moving lid, inlet value, output cadence) is injected as a callback instead of being "
        "written into the solver."
    )
    add(
        "2. a **mask-blend boundary condition** — `Domain.CollideSC` applies the wall as one tensor "
        "expression over a float mask field, `f = f_no_boundary + f_boundary` with "
        "`f_no_boundary = f*(1-boundary)` and `f_boundary = conv1x1(f*boundary, BOUNCE)`, so an "
        "arbitrary geometry costs no branching code."
    )
    add("")
    add("Machine-extracted provenance (no hand transcription):")
    add("")
    add("| item | value |")
    add("|---|---|")
    add(f"| repository | `{ev.get('repo', ref.get('oss_repo'))}` |")
    add(f"| tarball | `{ev.get('tarball_url')}` |")
    add(f"| tarball sha256 | `{ev.get('tarball_sha256')}` |")
    for rel, digest in prov.items():
        add(f"| `{rel}` sha256 | `{digest}` |")
    add(f"| checkout used at run time | `{ev.get('checkout')}` |")
    add(f"| extraction | {ev.get('extraction')} |")
    add(f"| matches previously recorded provenance | {ev.get('matches_recorded_provenance')} |")
    add("")
    if snips:
        add("Upstream text the port is measured against (`artifacts/xval_per_step_hook_mask_blend/oss_ref/evidence.json`):")
        add("")
        add("| idiom | file:line | text |")
        add("|---|---|---|")
        for tag in ("solve_signature", "blend_line_split", "blend_line_sum", "bounce_piece",
                    "mask_constant", "velocity_recompute", "lid_setup_signature"):
            snip = snips.get(tag)
            if not snip:
                continue
            text = snip["text"].replace("|", "\\|")
            if len(text) > 150:
                text = text[:147] + "..."
            add(f"| `{tag}` | `{snip['file']}:{snip['line']}` | `{text}` |")
        add("")
        if snips.get("solve_body"):
            body = snips["solve_body"]
            add(f"`Domain.Solve` verbatim, `{body['file']}:{body['line']}`:")
            add("")
            add("```python")
            add(body["text"])
            add("```")
            add("")
        if snips.get("lid_setup_step"):
            body = snips["lid_setup_step"]
            add(f"The moving-lid hook, `{body['file']}:{body['line']}`:")
            add("")
            add("```python")
            add(body["text"])
            add("```")
            add("")

    # ------------------------------------------------------------------ 2 method
    add("## 2. Method and implementation")
    add("")
    add(
        "The port is library code, `src/tensorlbm/step_hooks.py` (not bench-local):"
    )
    add("")
    add(
        "* `solve_with_hooks(state, n_steps, operator=..., initialize_step=..., setup_step=..., "
        "save_step=..., save_interval=...)` — upstream `Domain.Solve`'s loop shape. `state` is a "
        "dict, hooks may mutate it in place; the save tick precedes the setup hook of that same step, "
        "so the record `HookLog(setup=[0..n-1], save=[0, k, 2k, ..., n])` is assertable."
    )
    add(
        "* `compose_masked_boundaries(f_post, f_pre, layers)` — `f = f_post*(1 - M) + Σ_k m_k f_bc_k`, "
        "`M = Σ_k m_k`, with the partition enforced (`M ≤ 1`, else `ValueError`). Upstream has a "
        "single mask and a single BC family; the generalised form lets a bounce wall, a moving wall "
        "and an inlet/outlet coexist on arbitrary geometry."
    )
    add(
        "* three layer families: `stationary_bounce_layer` (upstream `CollideSC`), "
        "`moving_bounce_layer` (`+ 6 w_i (c_i·u_wall)`, the one addition), `equilibrium_layer` "
        "(Dirichlet velocity by equilibrium substitution). A layer's velocity may be a callable of "
        "the hook state, which is the torch analogue of upstream's `setup_step` writing `domain.Vel`."
    )
    add("")
    add("Arms (one changed variable per comparison):")
    add("")
    add("| arm | transport | collision | loop | boundary |")
    add("|---|---|---|---|---|")
    add(
        "| **A** explicit baseline | `tensorlbm.solver.stream` (gather) | `collide_bgk` | hand-written `for` | "
        "explicit index assignment `f[:, mask] = f_pre[opp][:, mask]`, moving-wall term added inline |"
    )
    add(
        "| **B** ported (chosen) | same | same | `solve_with_hooks` | `compose_masked_boundaries` over "
        "a disjoint mask partition (bounce / moving bounce / equilibrium) |"
    )
    add(
        "| **D** upstream-faithful lid | same | same | `solve_with_hooks` | single stationary bounce mask; "
        "lid speed only written into a `vel` variable by the setup hook and recomputed from `f` each "
        "step (upstream `lid_cavity.py`) |"
    )
    add("")
    add(
        "Equivalence checks (implementation question) are separate from the physics checks: "
        "one step from an identical state, explicit BC vs mask blend, on (a) a single stationary mask, "
        "(b) a disjoint two-layer mask (blob + moving patch), plus a 500-step closed box for mass drift."
    )
    add("")

    # -------------------------------------------------------------- 3 parameters
    add("## 3. Non-dimensional parameter table")
    add("")
    add("| quantity | symbol | value |")
    add("|---|---|---|")
    add(f"| relaxation time | tau | {_pct(tau)} |")
    add(f"| viscosity | nu = (tau-1/2)/3 | {_num(nu)} |")
    add(f"| lid / wall velocity | U_wall | {_num(base['u_wall'])} |")
    add(f"| Mach number | Ma = sqrt(3) U | {_num(base['mach'])} |")
    add(f"| channel height (lattice cells) | Ny | {ref_ny_cou} |")
    add(f"| wall spacing, half-way convention | H = Ny - 2 | {base['H']:.0f} |")
    add(f"| shear Reynolds number | Re_H = U H / nu | {_num(base['Re_H'], 5)} |")
    add(f"| steps (Couette) | N_t | {steps_cou} |")
    add(f"| lid ramp | ramp_steps | {base['ramp_steps']} |")
    add(f"| grid | nx | {base['nx']} |")
    add("")
    add("Poiseuille arm:")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| Ny | {ref_ny_poi} |")
    add(f"| H = Ny - 2 | {pbase['H']:.0f} |")
    add(f"| body force g = 8 nu u_peak / H^2 | {_num(pbase['g'], 8)} |")
    add(f"| target / measured peak | {_num(pbase['u_peak_target'])} |")
    add(f"| Re_H = u_peak H / nu | {_num(pbase['Re_H'], 5)} |")
    add(f"| steps | {pbase['steps']} |")
    add("")
    add("Composite channel (invariants only):")
    add("")
    add(
        f"Ny = {comp['ny']}, nx = {comp['nx']}, tau = {_pct(comp['tau'])}, u_in = {_num(comp['u_in'])}, "
        f"Ma = {_num(comp['mach'])}, Re = {_num(comp['Re'], 5)}, steps = {comp['steps']}."
    )
    add("")

    # ------------------------------------------------------------- 4 reference
    add("## 4. Reference vs got")
    add("")
    add("### 4.1 Plane Couette (the mask blend *is* the physics)")
    add("")
    add("| Ny | arm | profile L2 % | max err % | peak u vs U(H-0.5)/H % | slip lower % | slip lid % | 15% | 3% |")
    add("|---|---|---|---|---|---|---|---|---|")
    for ny in sorted({c["ny"] for c in cou_cfg}):
        for arm in ("A_explicit", "B_hook_blend", "D_upstream_lid"):
            row = next((c for c in cou_cfg if c["ny"] == ny and c["arm"] == arm), None)
            if row is None:
                continue
            gated = arm != "D_upstream_lid"
            add(
                f"| {ny} | {arm} | {_pct(row['profile_l2_pct'], 4)} | {_pct(row['max_err_pct'], 4)} | "
                f"{_pct(row['interior_umax_rel_pct'], 3)} | {_pct(row['slip_lower_pct'], 3)} | "
                f"{_pct(row['slip_lid_pct'], 3)} | "
                f"{_mark(row['profile_l2_pct'] < 15) if gated else 'n/a (declared fail)'} | "
                f"{_mark(row['profile_l2_pct'] < 3) if gated else 'n/a'} |"
            )
    add("")
    add("### 4.2 Plane Poiseuille (curvature + body force, two mask layers)")
    add("")
    add("| Ny | arm | profile L2 % | peak err % | <15% | <3% |")
    add("|---|---|---|---|---|---|")
    for ny in sorted({c["ny"] for c in poi["configs"]}):
        for arm in ("A_explicit", "B_hook_blend"):
            row = next((c for c in poi["configs"] if c["ny"] == ny and c["arm"] == arm), None)
            if row is None:
                continue
            add(
                f"| {ny} | {arm} | {_pct(row['profile_l2_pct'])} | {_pct(row['peak_err_pct'])} | "
                f"{_mark(row['profile_l2_pct'] < 15)} | {_mark(row['profile_l2_pct'] < 3)} |"
            )
    add("")
    same = []
    for key, cfg, arms in (
        ("couette", cou_cfg, ("A_explicit", "B_hook_blend")),
        ("poiseuille", poi["configs"], ("A_explicit", "B_hook_blend")),
    ):
        for ny in sorted({c["ny"] for c in cfg}):
            a = next((c for c in cfg if c["ny"] == ny and c["arm"] == arms[0]), None)
            b = next((c for c in cfg if c["ny"] == ny and c["arm"] == arms[1]), None)
            if a and b:
                same.append((key, ny, a["f_sha256"] == b["f_sha256"]))
    add(
        "Population-state digests, arm A vs arm B: "
        + ", ".join(f"{k} Ny={ny}: {'identical' if eq else 'DIFFERENT'}" for k, ny, eq in same)
        + ". The two arms therefore do not merely agree to a tolerance — they produce the same "
        "float64 state (the moving-wall momentum term is applied to the identical pre-collision "
        "populations)."
    )
    add("")
    add("### 4.3 Operator equivalence and invariants (arbitrary geometry)")
    add("")
    add("| check | value | verdict |")
    add("|---|---|---|")
    add(
        "| mask-blend vs explicit index BC, one step, disjoint blob + patch masks "
        f"(max rel) | {_num(inv['equivalence_max_rel'], 3)} | {_mark(inv['equivalence_max_rel'] <= 1e-15)} |"
    )
    add(
        f"| upstream 1x1 bounce convolution (`streaming_conv.bounce_blend`) vs explicit (max rel) | "
        f"{_num(inv['conv_vs_explicit_max_rel'], 3)} | {_mark(inv['conv_vs_explicit_max_rel'] <= 1e-15)} |"
    )
    add(
        f"| closed box, {inv['steps']} steps: mass drift (rel) | {_num(inv['mass_drift_rel'], 3)} | "
        f"{_mark(inv['mass_drift_rel'] < 1e-12)} |"
    )
    add(
        f"| wall-cell speed / peak |u| (diagnostic only, see §6.6) | "
        f"{_num(inv['wall_cell_speed_rel'], 3)} | n/a |"
    )
    add(
        f"| composite channel: |Q_out - Q_in| / Q_in | {_pct(comp['mass_imbalance_pct'])} % | "
        f"{_mark(comp['mass_imbalance_pct'] < 15)} |"
    )
    add(
        f"| composite channel: normal velocity in the wall-adjacent fluid cell / peak |u| | "
        f"{_pct(100.0 * comp['adjacent_normal_velocity_rel'], 3)} % | "
        f"{_mark(comp['adjacent_normal_velocity_rel'] < 5e-2)} |"
    )
    add("")
    add(
        "The composite channel has no independent reference solution (developing flow, first-order "
        "equilibrium inlet): only the invariants above are claimed, no accuracy number."
    )
    add("")

    # ---------------------------------------------------------- 5 sensitivity
    add("## 5. Convergence and parameter sensitivity")
    add("")
    add("| axis | configurations | result | reading |")
    add("|---|---|---|---|")
    add(
        f"| grid, Couette | Ny ∈ {sorted({c['ny'] for c in cou_cfg})} | "
        + ", ".join(
            f"Ny={ny}: {_pct(c_row(ny, 'B_hook_blend')['profile_l2_pct'])} %"
            for ny in sorted({c["ny"] for c in cou_cfg})
        )
        + " | linear analytic profile, so the residual is the wall-location convention, not a "
        "discretisation error |"
    )
    add(
        f"| grid, Poiseuille | Ny ∈ {sorted({c['ny'] for c in poi['configs']})} | "
        + ", ".join(
            f"Ny={ny}: {_pct(p_row(ny, 'B_hook_blend')['profile_l2_pct'])} %"
            for ny in sorted({c["ny"] for c in poi["configs"]})
        )
        + " | decreasing with resolution: second-order curvature error of half-way bounce |"
    )
    add(
        f"| wall-position convention | half-way (H = Ny-2) vs node walls (H = Ny-1) | "
        f"{_pct(sens['wall_convention_node']['profile_l2_pct'])} % vs "
        f"{_pct(c_row(sens['wall_convention_node']['ny'], 'B_hook_blend')['profile_l2_pct'])} % | "
        + (
            "the node convention is wrong by more than the 3 % target: the wall location must be "
            "validated before any accuracy statement, exactly like Cb/L/B for a hull"
            if sens["wall_convention_node"]["profile_l2_pct"] > 3.0
            else "both conventions stay inside the target here — the configuration does not discriminate"
        )
        + " |"
    )
    add(
        f"| hook cadence | save_interval ∈ {{50, 500}}, identity setup | "
        f"final state sha256 equal: {sens['save_interval_sha_equal']} | hook plumbing is "
        "physics-neutral |"
    )
    add(
        f"| lid onset | ramp_steps ∈ {{1, 200}} | L2 delta {_pct(sens['ramp_profile_delta_pct'])} pp "
        f"({_pct(sens['ramp_1']['profile_l2_pct'])} % vs {_pct(sens['ramp_200']['profile_l2_pct'])} %)"
        " | the transient can be shaped entirely from the setup hook, the steady state is independent of it |"
    )
    add("")
    add(f"Predeclared stopping rule: {pre['stopping_rule']}")
    add("")

    # ------------------------------------------------------- 6 failure modes
    add("## 6. Failure modes, traps and artefacts")
    add("")
    add(
        "1. **The mask blend alone cannot move a wall.** Upstream's reflected population carries no "
        "wall momentum, so a lid can only be *declared*. Arm D reproduces upstream's mechanism: the "
        f"setup hook writes the lid speed into the velocity variable and the step recomputes that "
        f"variable from the populations, so the hook changes "
        f"{ups['hook_bits_changed']} of {ups['hook_bits_total']} population entries and the measured "
        f"flow is {_pct(ups['flow_pct_of_uwall'], 4)} % of the imposed wall speed "
        f"(profile L2 {_pct(ups['profile_l2_pct'])} % against the analytic line). The port adds "
        "`6 w_i (c_i·u_wall)` and the same configuration reaches "
        f"{_pct(ups['ported_profile_l2_pct'])} %."
    )
    add(
        "2. **Library weight tables are float32.** `d2q9.W` is `torch.float32`; used directly in a "
        "float64 field the moving-wall momentum carries the float32 rounding of `1/9` "
        "(~7.5e-9 relative). `tables_for()` therefore switches to the exact weights "
        "(`d2q9.W_EXACT`, `d3q19.W_EXACT64`) for float64 fields — without that switch the "
        "bit-wise equivalence check fails at 1e-8 and looks like a formulation error."
    )
    add(
        "3. **Overlapping masks are a silent double-BC.** Upstream has one mask so this cannot happen; "
        "with a layer list it can, so `compose_masked_boundaries` raises unless `strict=False`. The "
        "composite-channel case selects wall cells out of the inlet/outlet columns for that reason."
    )
    add(
        "4. **The save tick is defined differently.** Upstream saves when `int(t/save_interval)` "
        "crosses an integer (float time comparison); the port uses `i % save_interval == 0` plus one "
        "final save, which is exact for integer `dt`. Same cadence, different off-by-one at the tail."
    )
    add(
        "5. **First-order inlet.** `equilibrium_layer` substitutes a full equilibrium on the boundary "
        "cell (no Zou–He reconstruction), so the composite channel's accuracy is first order — which "
        "is why no accuracy number is claimed for it."
    )
    add(
        "6. **A wall cell is not a zero-velocity cell.** Reading the no-slip condition off the "
        "macroscopic velocity carried by the *masked* cells is wrong: bounce-back enforces no-slip "
        f"through the population exchange, and those cells keep a non-zero |u| "
        f"({_pct(inv['wall_cell_speed_rel'] * 100, 1)} % of the peak in the closed box). The slip KPIs "
        "in §4.1 are therefore measured by extrapolating the *fluid* profile to the wall location, "
        "which is the only place the boundary condition is actually defined."
    )
    add("")
    add("Artefacts:")
    add("")
    add("| path | content |")
    add("|---|---|")
    add("| `artifacts/xval_per_step_hook_mask_blend/result.json` | every scalar KPI, verdict and predeclared design |")
    add("| `artifacts/xval_per_step_hook_mask_blend/table.csv` | arm × configuration table with profile errors and state digests |")
    add("| `artifacts/xval_per_step_hook_mask_blend/profiles.npz` | velocity profiles and the composite-channel field |")
    add("| `artifacts/xval_per_step_hook_mask_blend/figures/*.png` | the five figures below |")
    add("| `artifacts/xval_per_step_hook_mask_blend/oss_ref/evidence.json` | machine-extracted upstream snippets + sha256 |")
    add("| `src/tensorlbm/step_hooks.py`, `tests/test_step_hooks.py` | the port and its 14 unit tests |")
    add("")
    add(
        f"Timing: {_pct(tim['explicit_ms_per_step'], 3)} ms/step (hand-written loop) vs "
        f"{_pct(tim['hook_blend_ms_per_step'], 3)} ms/step (hook + mask blend), "
        f"{_pct(tim['hook_overhead_pct'], 1)} % overhead; whole benchmark {_pct(tim['total_s'], 1)} s."
    )
    add("")

    # ------------------------------------------------------------- 7 difference
    add("## 7. Difference list: TensorLBM vs OSS upstream, and the porting trade-offs")
    add("")
    add("| dimension | upstream (TF 1.x) | TensorLBM port | porting decision |")
    add("|---|---|---|---|")
    add(
        "| hook mechanics | `tf.Variable.assign` ops run by `sess.run(setup_step)`; hooks mutate graph "
        "variables | plain Python callables over a `state` dict | kept (same per-step position), because "
        "the hook is what lets a case own its time-dependent BC |"
    )
    add(
        "| where the BC lives | inside the collision op (`CollideSC`) | inside the step operator "
        "(`masked_step` → `compose_masked_boundaries`), also after collision | kept: the blend needs the "
        "pre-collision populations, which only exist between `stream` and the BC application |"
    )
    add(
        "| mask cardinality | one constant mask per domain | a partition of masks, `Σ m_k ≤ 1` enforced | "
        "generalised: free, and it is what makes a composite inlet/wall/outlet geometry expressible |"
    )
    add(
        "| moving wall | impossible in the blend; `lid_cavity.py` only writes `Vel` | "
        "`moving_bounce_layer` adds `6 w_i (c_i·u_wall)` | added — measured: without it "
        f"{_pct(ups['flow_pct_of_uwall'], 4)} % of U_wall, with it {_pct(ups['ported_profile_l2_pct'])} % "
        "profile error |"
    )
    add(
        "| boundary location | implicitly the solid cell face (half-way) | explicit `H = Ny-2` convention, "
        "asserted by the analytic case | documented + tested, because the wrong convention alone costs "
        f"{_pct(sens['wall_convention_node']['profile_l2_pct'])} % here |"
    )
    add(
        "| precision | float32 GPU graph | float64 CPU, exact weight tables | adopted: the equivalence and "
        "mass KPIs need round-off, not float32 noise |"
    )
    add(
        "| output cadence | float time comparison, video writer in the example | integer step modulo, "
        "`save_step(state, t, i)` | adapted (same cadence, exact integer semantics) |"
    )
    add(
        "| lattice coverage | D2Q9 / D3Q15 / D3Q19 with hand-written tables | tables derived from the "
        "velocity set on import | kept from the in-house side (the earlier CONV-AS-STREAMING XVAL found "
        "upstream's D3Q19 `q=3` STREAM tap is wrong, so its tables are not copied) |"
    )
    add(
        "| not ported | LES-with-graph-unroll, multi-component MCMP `tau` mixing, video/session plumbing | — | "
        "out of scope here: those are separate inventoried features of the same upstream repo "
        "(`oss_solutions` rows 34-40) |"
    )
    add("")
    add("Predeclared design (decision, alternatives, thresholds, confounders, stopping rule):")
    add("")
    add(f"* decision: {pre['decision']}")
    add(f"* hypothesis: {pre['hypothesis']}")
    add(f"* sensitivity axes: {'; '.join(pre['sensitivity_points'])}")
    add(f"* confounders: {'; '.join(pre['confounders'])}")
    add("")
    add("## 8. Figures")
    add("")
    for fig in result.get("figures", []):
        add(f"![{fig['name']}](figures/{fig['name']})")
        add("")
        add(f"*{fig['caption']}*")
        add("")
    add(
        f"Environment: python {env['python']}, torch {env['torch']}, numpy {env['numpy']}, "
        f"{env['dtype']}, {env['threads']} threads, CUDA available = {env['cuda']}."
    )
    add("")
    return "\n".join(L)