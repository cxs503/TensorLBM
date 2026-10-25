"""Execute moving-mask BGK cases; publish lattice-unit raw states and ledger."""

import hashlib
import json
from pathlib import Path

from tensorlbm.moving_boundary_2d import MovingDisk

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples/moving_boundary/evidence"


def main():
    OUT.mkdir(exist_ok=True)
    reports = []
    # Same displacement but different lattice velocity and duration. At fixed
    # tau this changes Re and diffusion time: motion-rate sensitivity, NOT a
    # same-physical-problem time refinement claim.
    for name, speed, steps, comoving, tau, dt in [
        ("comoving", 0.03, 100, True, 0.8, 0.01),
        ("moving", 0.03, 100, False, 0.8, 0.01),
        ("motion-half", 0.015, 200, False, 0.8, 0.01),
        ("motion-quarter", 0.0075, 400, False, 0.8, 0.01),
        ("time-half", 0.015, 200, False, 0.65, 0.005),
        ("time-quarter", 0.0075, 400, False, 0.575, 0.0025),
        ("time-eighth", 0.00375, 800, False, 0.5375, 0.00125),
        ("time-sixteenth", 0.001875, 1600, False, 0.51875, 0.000625),
    ]:
        sim = MovingDisk(velocity=(speed, 0), fluid_velocity=(speed if comoving else 0, 0), tau=tau)
        for _ in range(steps):
            sim.step()
        import torch

        resumed = MovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
        reference = MovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
        for _ in range(5):
            reference.step()
            resumed.step()
        h = sim.history
        report = dict(
            name=name,
            steps=steps,
            physical_accuracy_qualified=False,
            si=dict(
                dx_m=0.01,
                dt_s=dt,
                rho_kg_m3=1000.0,
                thickness_m=0.01,
                nu_m2_s=(tau - 0.5) / 3 * 0.01**2 / dt,
                velocity_m_s=speed * 0.01 / dt,
                duration_s=steps * dt,
            ),
            restart_bitwise=torch.equal(reference.f, resumed.f),
            max_momentum_residual=max(x["momentum_residual"] for x in h),
            max_mass_residual=max(abs(x["mass_residual"]) for x in h),
            covered=sum(x["covered"] for x in h),
            exposed=sum(x["exposed"] for x in h),
            cumulative_reservoir_mass=sum(x["reservoir_mass"] for x in h),
            total_absolute_reservoir_mass=sum(abs(x["reservoir_mass"]) for x in h),
            solid_impulse=[
                sum(x["wall_impulse_on_solid"][j] + x["conversion_impulse_on_solid"][j] for x in h)
                for j in range(2)
            ],
        )
        report["solid_impulse_si_Ns"] = [x * 1000 * 0.01**4 / dt for x in report["solid_impulse"]]
        path = OUT / (name + ".json")
        path.write_text(json.dumps(dict(report=report, state=sim.snapshot()), indent=2) + "\n")
        report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        reports.append(report)
    summary = dict(
        schema="tensorlbm.moving-disk-study/1",
        units="lattice units: dx=dt=rho0=1",
        algorithm="moving halfway bounce-back; adjacent density wall-speed refill; global mass reservoir",
        physical_accuracy_qualified=False,
        temporal_convergence_qualified=False,
        sensitivity_type="combined time-step/compressibility at fixed grid, viscosity, speed and duration",
        sound_speed_si_m_s={x["name"]: 0.01 / (3**0.5 * x["si"]["dt_s"]) for x in reports},
        rate_sensitivity_changes_reynolds_number=True,
        source_sha256=hashlib.sha256(
            (ROOT / "src/tensorlbm/moving_boundary_2d.py").read_bytes()
        ).hexdigest(),
        cases=reports,
    )
    (OUT / "study.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
