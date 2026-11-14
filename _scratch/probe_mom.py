import sys
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise')
import stepwise_ref as S

f0 = S.shear_wave_field(64, 8, u0=0.02)
vel = S.tensorlbm_velocities("D2Q9")
j0 = np.tensordot(vel, f0, axes=(0, 0))
print("sum j0 =", j0.sum(axis=(1, 2)), " mass0 =", f0.sum())
f = f0.copy()
for t in range(1, 31):
    f = S.in_repo_step(f, model="srt", tau=0.55)
    j = np.tensordot(vel, f, axes=(0, 0))
    if t in (1, 2, 5, 30):
        print(t, "sum j =", j.sum(axis=(1, 2)), " drift =", (j.sum(axis=(1, 2)) - j0.sum(axis=(1, 2))))
print("profile of j_x (y=0..7, x=0):", np.tensordot(vel, f, axes=(0, 0))[0][:8, 0])
print("density profile y=0..7:", f.sum(0)[:8, 0])