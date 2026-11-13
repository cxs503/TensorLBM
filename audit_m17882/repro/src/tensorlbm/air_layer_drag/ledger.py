"""Mass / gas-inventory ledger for the two-phase air-layer channel.

The solver never creates phase out of nothing: gas enters only through the
explicit volumetric source (``add_gas``) and leaves only through the explicit
vent sink (``vent_gas``).  Both are integrated exactly, so the ledger

    M_gas(t) - M_gas(0) = m_injected - m_vented

must close to round-off.  ``GasLedger`` records the terms and reports the
relative residual; the campaign asserts it below ``1e-10``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GasLedger:
    """Bookkeeping of the gas inventory of a two-phase run."""

    rho_l: float
    rho_g: float
    records: list[dict] = field(default_factory=list)

    @property
    def density_jump(self) -> float:
        return self.rho_l - self.rho_g

    def record(self, *, step: int, gas_volume: float, injected_volume: float,
               vented_volume: float, initial_volume: float) -> dict:
        entry = {
            "step": int(step),
            "gas_volume": float(gas_volume),
            "gas_mass": float(gas_volume) * self.rho_g,
            "volume_gain": float(gas_volume) - float(initial_volume),
            "injected_volume": float(injected_volume),
            "vented_volume": float(vented_volume),
        }
        # volume ledger:  dV_gas = dV_injected - dV_vented
        net = entry["injected_volume"] - entry["vented_volume"]
        entry["ledger_residual"] = entry["volume_gain"] - net
        entry["ledger_rel_residual"] = abs(entry["ledger_residual"]) / max(
            abs(net), abs(entry["volume_gain"]), 1e-30)
        self.records.append(entry)
        return entry

    # -- mixture mass ledger (used for the closed-domain mass gate) ------
    @staticmethod
    def closed_mass_drift(masses: list[float]) -> float:
        """Relative drift of a closed-domain total mass history."""
        if len(masses) < 2:
            raise ValueError("need at least two mass samples")
        m0 = masses[0]
        return max(abs(m - m0) for m in masses) / abs(m0)

    @staticmethod
    def phase_inventory_drift(phi_totals: list[float]) -> float:
        """Relative drift of sum(phi) (conservative Allen-Cahn check)."""
        if len(phi_totals) < 2:
            raise ValueError("need at least two phi samples")
        s0 = phi_totals[0]
        scale = max(abs(s0), 1.0)
        return max(abs(s - s0) for s in phi_totals) / scale

    def summary(self) -> dict:
        if not self.records:
            return {"records": 0}
        last = self.records[-1]
        return {
            "records": len(self.records),
            "final_gas_volume": last["gas_volume"],
            "injected_volume": last["injected_volume"],
            "vented_volume": last["vented_volume"],
            "ledger_rel_residual": last["ledger_rel_residual"],
            "max_ledger_rel_residual": max(r["ledger_rel_residual"] for r in self.records),
        }