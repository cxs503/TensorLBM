"""Pytest verification for the 3-D breakable-ice (cohesive bond) DEM leg.

Closes G5-06 of GAP-LBM-T5-ICE-COUPLING: the DEM leg must *fail/break* ice
(R3), not merely repel.  LinearSpringDashpot is normal-only (no tension), so a
bonded ice cover can never fracture; BondedDEM adds a tension-capable cohesive
bond that ruptures at a strain/force threshold and books the released fracture
energy.

Gates
-----
B1  bond formation: contacting pairs form a bond, separated pairs do not.
B2  tension capability: a bonded pair pulled apart carries a *tensile* force
    (the no-tension LinearSpringDashpot returns zero here).
B3  rupture: exceeding breaking_strain breaks the bond irreversibly and the
    pair reverts to no-tension (force drops to ~0 on further separation).
B4  fracture-energy ledger: a ruptured bond releases 0.5*k*(d-rest)^2, and the
    sum over all ruptures matches the analytic value to <1e-12.
B5  fragment pattern: a 3-particle bonded chain under a tip pull breaks the
    far bond first (weakest link), leaving a 2-particle fragment + 1 free.
"""

import math
import pytest
import torch

from tensorlbm.dem.contact import BondedDEM


def _cluster(n=3, spacing=1.0, radius=0.5):
    """A 1-D bonded chain of ``n`` touching spheres along x."""
    pos = torch.tensor([[i * spacing, 0.0, 0.0] for i in range(n)], dtype=torch.float64)
    vel = torch.zeros((n, 3), dtype=torch.float64)
    return pos, vel


class TestB1BondFormation:
    def test_contacting_pairs_form_bonds(self):
        pos, vel = _cluster(n=3, spacing=1.0, radius=0.5)  # all touching
        dem = BondedDEM(pos, vel, radius=0.5)
        assert len(dem.bonds) == 2  # (0,1) and (1,2)
        assert all(not b.broken for b in dem.bonds)

    def test_separated_pairs_do_not_bond(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
        vel = torch.zeros((2, 3), dtype=torch.float64)
        dem = BondedDEM(pos, vel, radius=0.5)
        assert len(dem.bonds) == 0


class TestB2TensionCapability:
    def test_bonded_pair_carries_tension(self):
        """A bonded pair pulled apart must carry a tensile (pulling) force."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        vel = torch.zeros((2, 3), dtype=torch.float64)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3,
                        breaking_strain=0.1)  # carry tension before rupture
        # pull particle 1 away along +x (d -> 1.05, rest = 1.0)
        dem.positions[1, 0] = 1.05
        f = dem.step()
        # particle 0 feels +x (pulled toward 1), particle 1 feels -x
        assert f[0, 0] > 0, f"expected tensile pull on p0, got {f[0,0]}"
        assert f[1, 0] < 0, f"expected tensile pull on p1, got {f[1,0]}"
        # magnitude = k * (d - rest) = 1e3 * 0.05 = 50
        assert abs(f[0, 0] - 50.0) < 1e-6, f"got {f[0,0]}"

    def test_nontension_fallback_is_zero(self):
        """The no-tension LinearSpringDashpot returns ~0 on separation."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        vel = torch.zeros((2, 3), dtype=torch.float64)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3)
        # break the bond first by exceeding the strain threshold
        dem.positions[1, 0] = 1.10  # axial = 0.10 > breaking_strain 0.015
        dem.step()
        assert dem.n_broken == 1
        # now separate further: no-tension -> force ~0
        dem.positions[1, 0] = 1.20
        f = dem.step()
        assert abs(f[0, 0]) < 1e-6, f"no-tension should be ~0, got {f[0,0]}"


class TestB3Rupture:
    def test_bond_breaks_at_strain_threshold(self):
        pos, vel = _cluster(n=2, spacing=1.0, radius=0.5)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3,
                        breaking_strain=0.015)
        assert dem.n_intact == 1
        # axial strain 0.02 > 0.015 -> rupture
        dem.positions[1, 0] = 1.02
        dem.step()
        assert dem.n_broken == 1
        assert dem.n_intact == 0

    def test_rupture_is_irreversible(self):
        pos, vel = _cluster(n=2, spacing=1.0, radius=0.5)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3,
                        breaking_strain=0.015)
        dem.positions[1, 0] = 1.02
        dem.step()
        assert dem.n_broken == 1
        # bring back to rest: bond must NOT reform
        dem.positions[1, 0] = 1.0
        dem.step()
        assert dem.n_broken == 1
        assert dem.n_intact == 0


class TestB4FractureEnergyLedger:
    def test_fracture_energy_matches_analytic(self):
        """A ruptured bond releases 0.5*k*(d-rest)^2, summed to <1e-12."""
        pos, vel = _cluster(n=2, spacing=1.0, radius=0.5)
        k = 1e3
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=k,
                        breaking_strain=0.015)
        # rupture at d = 1.02 (axial 0.02 > 0.015)
        dem.positions[1, 0] = 1.02
        dem.step()
        d = 1.02
        rest = 1.0
        expected = 0.5 * k * (d - rest) ** 2
        assert abs(dem.fracture_energy_released - expected) < 1e-12, (
            f"got {dem.fracture_energy_released}, expected {expected}"
        )

    def test_energy_ledger_is_nonnegative_and_monotone(self):
        """Fracture energy only grows as more bonds rupture."""
        pos, vel = _cluster(n=3, spacing=1.0, radius=0.5)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3,
                        breaking_strain=0.015)
        e0 = dem.fracture_energy_released
        # rupture both bonds
        dem.positions[1, 0] = 1.02
        dem.step()
        e1 = dem.fracture_energy_released
        dem.positions[2, 0] = 2.02
        dem.step()
        e2 = dem.fracture_energy_released
        assert e0 == 0.0
        assert e1 > 0.0
        assert e2 >= e1


class TestB5FragmentPattern:
    def test_weakest_link_breaks_first(self):
        """A 3-particle chain under a tip pull breaks the far bond first."""
        pos, vel = _cluster(n=3, spacing=1.0, radius=0.5)
        dem = BondedDEM(pos, vel, radius=0.5, bond_stiffness=1e3,
                        breaking_strain=0.015)
        # pull the tip (particle 2) away: bond (1,2) sees the largest strain
        dem.positions[2, 0] = 2.02
        dem.step()
        # the far bond (1,2) breaks; the near bond (0,1) stays intact
        broken_pairs = [(b.i, b.j) for b in dem.bonds if b.broken]
        assert (1, 2) in broken_pairs, f"expected (1,2) to break, got {broken_pairs}"
        assert (0, 1) not in broken_pairs
        assert dem.n_intact == 1  # 2-particle fragment remains
