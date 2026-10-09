import json
import unittest
import torch
from tensorlbm.feedback_moving_disk import MovingDiskFeedback

class LiveFeedbackTests(unittest.TestCase):
    def test_actual_feedback_and_restart(self):
        fluid=MovingDiskFeedback(); velocity=[.04,0.]; mass=.3
        initial=torch.tensor(velocity,dtype=torch.float64)*mass
        reservoir=torch.zeros(2,dtype=torch.float64)
        for i in range(18):
            if i==7: restored=MovingDiskFeedback.restore(json.loads(json.dumps(fluid.snapshot())))
            e=fluid.advance(time_s=fluid.time_s,center_m=fluid.center_m,velocity_m_s=velocity)
            if i>=7:
                other=restored.advance(time_s=restored.time_s,center_m=restored.center_m,velocity_m_s=velocity)
                self.assertEqual(e,other); self.assertTrue(torch.equal(fluid.solver.f,restored.solver.f))
            velocity=[v+j/mass for v,j in zip(velocity,e['impulse_on_body_Ns'])]
            reservoir+=torch.tensor(e['reservoir_impulse_on_fluid_Ns'],dtype=torch.float64)
        total=torch.tensor(fluid.fluid_momentum_Ns())+torch.tensor(velocity)*mass
        self.assertLess(float((total-initial-reservoir).abs().max()),1e-9)
        self.assertNotEqual(velocity,[.04,0.])

    def test_stale_rejected(self):
        f=MovingDiskFeedback()
        for kw in [dict(time_s=.1,center_m=f.center_m,velocity_m_s=[0,0]),dict(time_s=0,center_m=[0,0],velocity_m_s=[0,0]),dict(time_s=0,center_m=f.center_m,velocity_m_s=[float('nan'),0])]:
            with self.assertRaises(ValueError): f.advance(**kw)
        self.assertEqual(f.time_s,0)

if __name__=='__main__': unittest.main()
