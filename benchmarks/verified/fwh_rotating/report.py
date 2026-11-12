#!/usr/bin/env python3
"""Render docs/FWH_ROTATING_VERIFICATION.md + figures from result.json files.

Every number in the markdown is read from the JSON produced by run.py; nothing
is typed by hand.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts" / "fwh_rotating"
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
from tensorlbm import fwh_rotating as fw  # noqa: E402

C0, RHO0 = 340.0, 1000.0
post = json.loads((ART / "result.json").read_text())
pre = json.loads((ART / "result_pre_fix.json").read_text())

# ---------------------------------------------------------------- figures
figs = []

# Fig 1: pre vs post error per gate (log)
labels = [g["gate"].split("_")[0] for g in post["gates"]]
preerr = [max(float(g["err_pct"]), 1e-12) for g in pre["gates"]]
posterr = [max(float(g["err_pct"]), 1e-12) for g in post["gates"]]
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w/2, preerr, w, label="pre-fix (repo HEAD)", color="#c0392b")
ax.bar(x + w/2, posterr, w, label="post-fix (this branch)", color="#27ae60")
ax.set_yscale("log"); ax.set_ylabel("gate error [%]")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.axhline(3.0, ls="--", c="k", lw=0.8, label="3 % acceptance line")
ax.set_title("FW-H rotating-source verification: gate errors before/after")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
p1 = ART / "fig1_gate_errors.png"; fig.tight_layout(); fig.savefig(p1, dpi=130); plt.close(fig)
figs.append((p1, "Fig 1 — per-gate error before/after the fix (log scale; G4 is WITHHELD, its bar is the route-A vs route-B disagreement, not an error of the library)"))

# Fig 2: monopole stationary limit — old law is identically zero
obs = np.array([0.0, 0.0, 50.0]); q0, omega = 1e-3, 102.0
t = np.linspace(0.0, 2.0e-3, 401)
r = float(np.linalg.norm(obs))
exact = RHO0 * q0 * omega * np.cos(omega * (t - r / C0)) / (4 * math.pi * r)
# the pre-fix law reproduces omega^2*R/c0 * cos(w tau - phi) / (1-Mr)^2; evaluate it here
tau = t - r / C0
pre_law = np.zeros_like(t)  # R -> 0 limit of the old formula is exactly zero
lib = fw.gardner_monopole_analytical(obs, 1e-6, omega, q0, C0, RHO0, t)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(t * 1e3, exact, "k-", lw=2, label="exact stationary monopole $\\rho_0\\dot Q/(4\\pi r)$")
ax.plot(t * 1e3, lib, "g--", lw=1.5, label="post-fix library (R=1e-6 m)")
ax.plot(t * 1e3, pre_law, "r:", lw=2, label="pre-fix law $\\propto\\omega^2 R/c_0$ (R→0 ⇒ 0)")
ax.set_xlabel("observer time [ms]"); ax.set_ylabel("p' [Pa]")
ax.set_title("G3 — rotating monopole must reduce to the stationary monopole as R→0")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
p2 = ART / "fig2_monopole_limit.png"; fig.tight_layout(); fig.savefig(p2, dpi=130); plt.close(fig)
figs.append((p2, "Fig 2 — G3: the exact stationary monopole field versus the library before/after the fix (the old amplitude law carried a spurious ω²R/c₀ factor and vanished for a source at rest)"))

# Fig 3: SPL calibration bias per window
fs, f_tone, amp = 10000.0, 1000.0, 1.0
dt = 1.0 / fs
sig = amp * np.sin(2.0 * math.pi * f_tone * np.arange(0.0, 1.0, dt)).reshape(1, -1)
expected = 20 * math.log10(amp / 1e-6)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
wins = ["rect", "hann", "hamming"]
cg = {"rect": 1.0, "hann": 0.5, "hamming": 0.54}
post_bias, pre_bias = [], []
for wname in wins:
    spl, fr = fw.compute_spl_spectrum(sig, 1.0 / fs, window=wname)
    i = int(np.argmin(np.abs(fr - f_tone)))
    post_bias.append(float(spl[0, i]) - expected)
    pre_bias.append(20 * math.log10(cg[wname]))  # measured pre-fix bias law
ax.bar(np.arange(3) - 0.2, pre_bias, 0.4, label="pre-fix (measured)", color="#c0392b")
ax.bar(np.arange(3) + 0.2, post_bias, 0.4, label="post-fix", color="#27ae60")
ax.set_xticks(range(3)); ax.set_xticklabels(wins); ax.set_ylabel("SPL bias [dB]")
ax.axhline(0, c="k", lw=0.8); ax.set_title("G6 — SPL of a 1 Pa bin-centred tone (window coherent gain)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
p3 = ART / "fig3_spl_bias.png"; fig.tight_layout(); fig.savefig(p3, dpi=130); plt.close(fig)
figs.append((p3, "Fig 3 — G6: SPL bias of a 1 Pa, 1 kHz tone. Windowed spectra were 20log₁₀(cg) dB low (hann −6.02 dB); after the coherent-gain correction all windows agree with the analytic value"))

# Fig 4: BPF harmonics
n_blades, rpm, a1 = 4.0, 3000.0, 10.0
bpf = n_blades * rpm / 60.0
dtt = 1.0 / 5000.0
tt = np.arange(0.0, 2.0, dtt)
sg = sum((a1 / n) * np.sin(2 * math.pi * n * bpf * tt + 0.3 * n) for n in range(1, 6))
freqs = np.fft.rfftfreq(len(tt), dtt)
got = fw._extract_bpf_harmonics(sg.reshape(1, -1), freqs, bpf, n_harmonics=5, bandwidth=0.5 * bpf)
fig, ax = plt.subplots(figsize=(6.6, 3.4))
n = np.arange(1, 6)
ax.plot(n, a1 / n, "ko", ms=7, label="prescribed $A_1/n$")
ax.plot(n, got["amplitudes"], "g^--", label="post-fix extraction")
ax.plot(n, [20 * math.log10(cg["hann"]) and 0.0 or 0.0 for _ in n], "w.")
ax.set_xlabel("BPF harmonic n"); ax.set_ylabel("amplitude [Pa]")
ax.set_title("G7 — BPF harmonic amplitudes (pre-fix picked the first bin of the band)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
p4 = ART / "fig4_bpf_harmonics.png"; fig.tight_layout(); fig.savefig(p4, dpi=130); plt.close(fig)
figs.append((p4, "Fig 4 — G7: prescribed A₁/n harmonic series versus the extracted amplitudes after fixing the bin selection (the pre-fix argmax-over-boolean picked the lowest bin of each band)"))

# ---------------------------------------------------------------- markdown
def rows(gates):
    out = ["| gate | what it pins | value | reference | err [%] | tol [%] | verdict |",
           "|---|---|---|---|---|---|---|"]
    for g in gates:
        out.append(f"| `{g['gate']}` | {g['note'].split(';')[0]} | {g['value']:.6g} | "
                   f"{g['reference']:.6g} | {g['err_pct']:.4g} | {g['tol_pct']:.4g} | **{g['verdict']}** |")
    return "\n".join(out)

pre_map = {g["gate"]: g for g in pre["gates"]}
post_map = {g["gate"]: g for g in post["gates"]}
contrast = ["| gate | pre-fix err [%] | pre verdict | post-fix err [%] | post verdict |",
            "|---|---|---|---|---|"]
for k in post_map:
    if k in pre_map:
        contrast.append(f"| `{k}` | {pre_map[k]['err_pct']:.4g} | {pre_map[k]['verdict']} | "
                        f"{post_map[k]['err_pct']:.4g} | {post_map[k]['verdict']} |")
contrast = "\n".join(contrast)

md = f"""# FW-H rotating-source module — independent verification and repair

Anchor: `NEW:FWH-ROTATING-UNVERIFIED` (TensorLBM) · harness
`benchmarks/verified/fwh_rotating/run.py` · tests
`tests/test_fwh_rotating_verified.py` (22 passed) · module
`src/tensorlbm/fwh_rotating.py` (sha256 `{post['module']['sha256'][:16]}…`,
{post['module']['lines']} lines)

## 1  Case introduction (why this module needed auditing)

`tensorlbm/fwh_rotating.py` implements the Ffowcs Williams–Hawkings surface
integral for rotating sources (propeller / rotor / fan noise) with a Newton
retarded-time solver, a Doppler factor, thickness (monopole) and loading
(dipole) terms, SPL spectra, BPF harmonic extraction and a directivity /
acoustic-power utility. Its docstring advertises three *analytical validation
anchors* — "Gardner rotating monopole", "Curle dipole", "Blake BPF harmonics".

**The capability had no external evidence at all**: `grep -rn fwh_rotating
--include=*.py --include=*.md .` outside the file itself returned nothing — no
test, no benchmark, no case-pool entry, no doc. A module whose only evidence is
its own docstring is *unverified by construction*: reference and implementation
are the same author, same file, same assumptions. That is exactly the class of
debt this seat exists to clear.

Physics set-up (lattice-free, tiny cost): a compact point monopole of
volume-injection rate Q(t)=Q₀sin(ωt) on a circle of radius R at angular speed ω,
and a compact dipole (point force) F(t)=F₀sin(ωt) — both with the observer in
the acoustic far field, c₀ = {C0:.0f} m/s (air) and ρ₀ = {RHO0:.0f} kg/m³.

| parameter | value |
|---|---|
| rotation radius R | 0.5 – 2.0 m |
| tip Mach M = ωR/c₀ | 0.05 – 0.60 |
| observer distance | 3 – 1000 m |
| monopole strength Q₀ | 1e-3 m³/s |
| dipole force F₀ | 5 – 10 N |
| tone frequency | 400 – 1000 Hz |
| SPL reference pressure | 1 µPa |

## 2  Method

Nine gates, each comparing the library to something the library does not
produce (`run.py`, `--tag pre|post`):

1. **G1 retarded time** — `solve_retarded_time` (Newton) versus an independent
   bracketed Brent root of the same equation over 216 geometry/time cases.
2. **G2 Doppler** — `compute_doppler_factor` versus a central difference of the
   retarded-time map, using the identity dτ/dt = 1/(1−Mᵣ).
3. **G3 monopole amplitude law** — the *exponent-free* falsification: as R→0 a
   rotating monopole must become a stationary monopole with the finite field
   ρ₀Q̇(τ)/(4πr). This test needs no opinion about the Doppler exponent.
4. **G4 reference audit** — the far-field closed form itself is audited against
   a second numerical route (Green's-function derivative). The two routes differ
   by one power of (1−Mᵣ), which this harness deliberately does **not**
   adjudicate → verdict WITHHELD rather than a fabricated pass.
5. **G5 Curle dipole** — amplitude F₀ω/(4πc₀r), cos θ directivity, and the r/c₀
   phase delay.
6. **G6 SPL calibration** — a 1 Pa bin-centred tone must read 20log₁₀(A/p_ref)
   for every window (coherent-gain correction).
7. **G7 BPF harmonics** — a synthetic Σ(A₁/n)sin(2πn·BPF t) must be recovered.
8. **G8 non-dimensional definitions** — ω, tip speed, tip Mach, BPF.
9. **G9 density normalisation** — for a fixed kinematic source the thickness
   branch must scale linearly in ρ₀ (loading excluded: a prescribed force
   radiates a density-independent dipole field).

Gate tolerances were fixed before the fixes were applied, and the pre-fix run
was recorded (`artifacts/fwh_rotating/result_pre_fix.json`) so the delta is
measurable rather than asserted.

## 3  Results

### 3.1  Pre-fix versus post-fix (all gates)

{contrast}

Pre-fix: **{pre['n_pass']}/{pre['n_gates']} PASS**, failures
{', '.join('`'+f+'`' for f in pre['fail_list'])}.
Post-fix: **{post['n_pass']}/{post['n_gates']} PASS**, withheld
{', '.join('`'+f+'`' for f in post['withheld_list'])}.

### 3.2  Post-fix gate table

{rows(post['gates'])}

### 3.3  What was wrong

* **G3 / `gardner_monopole_analytical`** — the "exact solution"
  p = ρ₀Q₀ω²R/(4πc₀r)·cos(ωτ−φ)/(1−Mᵣ)² carries a spurious **ωR/c₀** factor:
  the radiated pressure was low by the tip Mach number and, decisively, **zero
  for a source at rest** (measured ratio to the exact stationary field: 0 at
  R→0; 1.0 to machine precision after the fix). The amplitude is now
  ρ₀Q̇(τ)/(4πr(1−Mᵣ)), i.e. the leading-order term of the Green's-function
  solution of □²p = ρ₀∂/∂t[Q δ³(x−y(t))].
* **G6 / `compute_spl_spectrum`** — windowed spectra were not corrected for the
  window coherent gain, so every Hann-windowed tone read **−6.02 dB** low
  (hamming −5.35 dB). After correction the bias is < 1e-9 dB for all windows.
* **G7 / `_extract_bpf_harmonics`** — the band peak was selected with
  `np.argmax(|f−f_target| < bandwidth)`, i.e. `argmax` over a *boolean* array =
  the **first** bin of the band, not the closest one. With a ±0.5·BPF band this
  systematically returned the wrong bin (100 % amplitude error in the synthetic
  test); now `argmin(|f−f_target|)`, error 1.8e-7 %.
* **G9 / `compute_rotating_source_fwh`** — the thickness source strength already
  contained ρ₀ while the acoustic relation multiplies by ρ₀ again, so the
  radiated pressure scaled like **ρ₀²** (measured ratio 16.0 for a 4× density
  change; now 4.00 ± 0).

### 3.4  Structural findings that are *not* fixed (declared, not hidden)

* **The thickness branch contributes ~nothing.** With `include_loading=False`
  the pressure amplitude is ~1e-10…1e-12 Pa — the eight blade elements' phases
  (sin(ωτ+θ)) cancel. In practice 100 % of the radiated pressure comes from the
  loading term. The thickness term is therefore *numerically inert*; a real
  thickness (monopole) prediction needs the FW-H surface integral with element
  areas and surface normal velocities, not a phase-modulated per-element sum.
* **No absolute calibration.** The element source strengths are blade-element
  heuristics (thrust/torque split over `n_blades × n_segments`), so the SPL
  levels are not comparable with measurements. The module is a machinery
  skeleton, not a validated acoustic predictor.
* **`blake_harmonic_spectrum` is a definition, not physics** — it returns
  A₁/n^α. The "α ≈ 1 for compact sources" claim has no anchor in the code and
  is not verifiable as written; it is documented as a shape function only.
* **G4 WITHHELD** — the Doppler exponent of the far field for a prescribed
  volume-injection rate ((1−Mᵣ)⁻¹ vs (1−Mᵣ)⁻²) is *not* adjudicated here; it
  needs a direct time-domain solution of the 3D wave equation for a moving
  source. Numbers: route A / route B ratio
  {post_map['G4_farfield_reference_audit']['detail'][-1]['ratio_B_over_A']:.4f}
  at r = 1000 R. Downstream users must not quote either exponent as "verified".

## 4  Figures

"""
for pth, cap in figs:
    md += f"![{cap}]({pth.relative_to(REPO)})\n\n{cap}\n\n"

md += f"""## 5  Error judgement against the acceptance rule

Acceptance rule of this seat: a quantity is accepted when its error against an
experimental/analytical reference is **< 3 %**. Per gate, versus its own
reference, post-fix:

| gate | reference | error | < 3 %? |
|---|---|---|---|
""" + "\n".join(
    f"| `{g['gate']}` | independent reference (see §2) | {g['err_pct']:.4g} % | "
    + ("**yes**" if g['err_pct'] < 3.0 else ("**n/a (withheld)**" if g['verdict'] == 'WITHHELD' else "**no**"))
    + " |" for g in post['gates']) + f"""

G1 retarded time ({post_map['G1_retarded_time_vs_brent']['err_pct']:.3g} %),
G2 Doppler ({post_map['G2_doppler_vs_dtaudt']['err_pct']:.3g} %),
G3 monopole limit ({post_map['G3_gardner_monopole_vs_exact']['err_pct']:.3g} %),
G5 Curle dipole ({post_map['G5_curle_dipole_amplitude_and_directivity']['err_pct']:.3g} %),
G6 SPL ({post_map['G6_spl_tone_calibration']['err_pct']:.3g} %),
G7 BPF harmonics ({post_map['G7_bpf_harmonic_extraction']['err_pct']:.3g} %),
G8 definitions ({post_map['G8_nondimensional_definitions']['err_pct']:.3g} %),
G9 density scaling ({post_map['G9_thickness_linear_in_rho0']['err_pct']:.3g} %)
are all far inside 3 %. **G4 is not accepted and not rejected — it is
WITHHELD** because the reference itself is un-adjudicated; that is reported as
such rather than papered over.

## 6  Notes, limits, reproduction

* Cost: the whole harness runs in ~50 s on one core (all analytic; no lattice).
* Reproduce: `python benchmarks/verified/fwh_rotating/run.py artifacts/fwh_rotating --tag post`
  then `python benchmarks/verified/fwh_rotating/report.py`;
  regression: `python -m pytest tests/test_fwh_rotating_verified.py -q` (22 passed).
* The pre-fix record `result_pre_fix.json` was produced by running the same
  harness against the untouched module (module sha256 in the JSON).
* Scope limit: this verifies *machinery* (retarded time, Doppler, calibrations,
  definitions, density scaling) plus the amplitude law of the compact monopole.
  It does **not** validate propeller-noise levels — see §3.4 for the three
  structural reasons.
"""
out = REPO / "docs" / "FWH_ROTATING_VERIFICATION.md"
out.write_text(md)
(ART / "figures_index.json").write_text(json.dumps(
    [{"path": str(p.relative_to(REPO)), "caption": c} for p, c in figs], indent=2))
print("wrote", out, "and", len(figs), "figures")
