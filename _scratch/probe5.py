import sys
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import anisotropic_coefficients, CX, CY
wi,wxx,wyy,wxy=[torch.tensor(a[1:],dtype=torch.float64) for a in ([(0.0,)]*0+[ (0.0,)+tuple(x) for x in anisotropic_coefficients()])]
cx=torch.tensor(CX[1:],dtype=torch.float64); cy=torch.tensor(CY[1:],dtype=torch.float64)
for nm,w,comp in [("wi",wi,cx),("wi",wi,cy),("wxx",wxx,cx),("wxx",wxx,cy),("wyy",wyy,cx),("wyy",wyy,cy),("wxy",wxy,cx),("wxy",wxy,cy)]:
    print(f"sum {nm} * c{'x' if comp is cx else 'y'} = {float((w*comp).sum()):+.6f}")
print("sum wxx cx cx =",float((wxx*cx*cx).sum()), " sum wyy cy cy =",float((wyy*cy*cy).sum()), " sum wxy cx cy =",float((wxy*cx*cy).sum()))
print("sum wxx cy cy =",float((wxx*cy*cy).sum()), " sum wyy cx cx =",float((wyy*cx*cx).sum()))
print("sum wi cx cx =",float((wi*cx*cx).sum()))
