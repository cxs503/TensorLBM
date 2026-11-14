import sys
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise')
import stepwise_ref as S

f0 = S.shear_wave_field(64, 8, u0=0.02)
for model, kw in [("srt", dict(tau=0.55)), ("srt", dict(tau=0.7)), ("srt", dict(tau=1.0)),
                  ("trt", dict(tau=0.7)), ("mrt_native", dict(tau=0.7))]:
    r = S.StepwiseReference(model=model, **kw).run(f0, 30, checkpoints=(1, 5, 30))
    got = S.in_repo_run(f0, n_steps=30, model=model, **kw)
    c = S.compare(got, r["final"])
    print(model, kw, "eps=%.3e" % c["eps_scaled_inf"], "rms=%.3e" % c["eps_rel_rms"])

rng = np.random.default_rng(7)
f0b = S.shear_wave_field(7, 11, u0=0.05) + 1e-3 * rng.standard_normal((9, 7, 11))
r = S.StepwiseReference(model="srt", tau=0.7).run(f0b, 30)
got = S.in_repo_run(f0b, n_steps=30, model="srt", tau=0.7)
print("nonsquare eps=%.3e" % S.compare(got, r["final"])["eps_scaled_inf"])