"""Auditor-side probes on m17744 (bf1927ca): are the protocol's rejections real, and
is the emitted kernel a standalone artifact that matches the independent reference?

Run from a fresh `git archive bf1927ca` extraction:  python runs/m17824_audit/probe_negative.py <extraction>
"""
import json, sys, importlib.util, pathlib
import numpy as np

ext = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(ext / "src"))
from tensorlbm.codegen import (BGKComponent, TRTComponent, D2Q9, ComponentSpec, StencilSpec,
                               ProtocolViolation, StencilViolation, validate_component,
                               generate_kernel, reference_collide, reference_step)
out = {}

# P1: a component that declares a viscosity inconsistent with the relaxation it emits
class LyingBGK(BGKComponent):
    def transport_coefficients(self):
        return {"nu": 0.05, "tau": self.tau, "s_plus": self.s_plus, "s_minus": self.s_minus}
try:
    validate_component(LyingBGK(tau=0.56)); out["P1_lying_nu"] = "ACCEPTED(bad)"
except ProtocolViolation as e:
    out["P1_lying_nu"] = "REJECTED:" + str(e)[:160]

# P2: magic inconsistent with s_minus
class LyingTRT(TRTComponent):
    def transport_coefficients(self):
        d = dict(super().transport_coefficients()); d["magic"] = 0.3; return d
try:
    validate_component(LyingTRT(tau=0.56)); out["P2_lying_magic"] = "ACCEPTED(bad)"
except ProtocolViolation as e:
    out["P2_lying_magic"] = "REJECTED:" + str(e)[:160]
out["P2_note"] = ("only spec.param('magic') is authoritative; a component may report an "
                  "inconsistent 'magic' inside transport_coefficients() and still validate")
class LieSM(TRTComponent):
    def s_minus(self):
        return 1.0
try:
    validate_component(LieSM(tau=0.56)); out["P2b_lying_s_minus"] = "ACCEPTED(bad)"
except ProtocolViolation as e:
    out["P2b_lying_s_minus"] = "REJECTED:" + str(e)[:160]

# P3: stencil whose weights do not sum to 1
try:
    StencilSpec(name="d2q9_bad", ex=D2Q9.ex, ey=D2Q9.ey, opposite=D2Q9.opposite,
                w=tuple(w * 1.01 for w in D2Q9.w)).validate()
    out["P3_bad_stencil"] = "ACCEPTED(bad)"
except (StencilViolation, ValueError) as e:
    out["P3_bad_stencil"] = "REJECTED:" + str(e)[:160]

# P4: emitted kernel is a *standalone* importable module (nothing from the generator at run time)
spec = importlib.util.spec_from_file_location(
    "gen_bgk", ext / "artifacts/xval_component_protocol_codegen/generated_bgk_tau0.56.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
out["P4_standalone_import"] = {"module_imports_src": "tensorlbm" in pathlib.Path(gen.__file__).read_text().split('"""', 2)[-1],
                               "constants_baked": [gen.S_PLUS, gen.S_MINUS], "Q": gen.Q}
rng = np.random.default_rng(0); f = 1.0 + 0.1 * rng.standard_normal((9, 12, 10))
tau, fam = 0.56, "BGK"   # match the kernel's provenance, otherwise the comparison is not a control
out["P4_max_abs_collide"] = float(np.abs(gen.collide(f) - reference_collide(f, family=fam, tau=tau)).max())
out["P4_max_abs_20_steps"] = None
a = f.copy(); b = f.copy()
for _ in range(20):
    a = gen.step(a); b = reference_step(b, family=fam, tau=tau)
out["P4_max_abs_20_steps"] = float(np.abs(a - b).max())
out["P4_mass_drift_collide"] = float(abs(gen.collide(f).sum(axis=0).mean() - f.sum(axis=0).mean()))

# P4b: same control on the near-equilibrium field the harness uses, plus a non-equilibrium one
f2 = 1.0 / 9.0 + 0.01 * np.random.default_rng(0).standard_normal((9, 12, 10))
out["P4b_harness_field_max_abs"] = float(np.abs(gen.collide(f2) - reference_collide(f2, family=fam, tau=tau)).max())

# P5: declared viscosity of the two families at identical tau must be identical
out["P5_nu_bgk_vs_trt"] = {"bgk": BGKComponent(0.56).transport_coefficients()["nu"],
                           "trt": TRTComponent(0.56).transport_coefficients()["nu"],
                           "analytic": (0.56 - 0.5) / 3.0}

# P6: generator determinism -> two separate processes would give the same sha (same process here)
out["P6_sha_repeatable"] = generate_kernel(BGKComponent(0.56)).sha256 == generate_kernel(BGKComponent(0.56)).sha256
print(json.dumps(out, indent=1, default=str))
