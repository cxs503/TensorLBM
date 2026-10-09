import copy
import unittest
import torch
from tensorlbm.moving_boundary_2d import MovingDisk, equilibrium, momentum

class MovingBoundaryTests(unittest.TestCase):
    def test_comoving_equilibrium_and_crossing(self):
        sim=MovingDisk(velocity=(.03,0),fluid_velocity=(.03,0))
        for _ in range(100):
            h=sim.step()
            self.assertLess(h['momentum_residual'],1e-10)
            self.assertLess(abs(h['mass_residual']),1e-10)
            self.assertEqual(float(sim.f[:,sim.solid].abs().sum()),0)
        self.assertGreater(sum(h['covered']+h['exposed'] for h in sim.history),0)
        # Geometry changes fluid node count. Uniform speed survives density
        # rescaling; uniform density is NOT claimed.
        rho=sim.f.sum(0)
        u=torch.einsum('iyx,ia->ayx',sim.f,torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=torch.float64))/rho.clamp_min(1e-30)
        self.assertLess(float((u[0][~sim.solid]-.03).abs().max()),1e-12)
        self.assertLess(float(u[1][~sim.solid].abs().max()),1e-12)

    def test_restart_bitwise(self):
        sim=MovingDisk(velocity=(.03,.01))
        for _ in range(20):sim.step()
        restored=MovingDisk.restore(copy.deepcopy(sim.snapshot()))
        for _ in range(30):
            self.assertEqual(sim.step(),restored.step())
        self.assertTrue(torch.equal(sim.f,restored.f))

    def test_stationary_no_reservoir(self):
        sim=MovingDisk(velocity=(0.,0.))
        before=sim.f.clone()
        for _ in range(5):sim.step()
        self.assertLess(float((sim.f-before).abs().max()),1e-15)

    def test_refill_accounted(self):
        sim=MovingDisk(velocity=(.06,0))
        for _ in range(60):sim.step()
        self.assertGreater(sum(abs(h['reservoir_mass']) for h in sim.history),.001)
        self.assertLess(max(h['bounce_momentum_residual'] for h in sim.history),1e-10)

    def test_invalid(self):
        for kwargs in [{'tau':.5},{'radius':-1},{'velocity':(float('nan'),0)},{'velocity':(.2,0)}]:
            with self.assertRaises(ValueError):MovingDisk(**kwargs)
        snap=MovingDisk().snapshot();snap['f'][0][0][0]=-1
        with self.assertRaises(ValueError):MovingDisk.restore(snap)

if __name__=='__main__':unittest.main()
