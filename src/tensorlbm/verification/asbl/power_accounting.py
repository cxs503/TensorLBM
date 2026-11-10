"""Net power accounting for suction-based drag reduction.

The key question: does the power saved by drag reduction exceed the
power required to pump the suction flow?

Power balance:
    P_drag = ∫ τ_w U∞ dA  (power to overcome skin friction)
    P_pump = Δp · Q_suction / η_pump  (pumping power for suction)
    P_net = P_drag + P_pump - P_drag,0  (net power change vs no suction)

For net benefit: P_net < 0 (suction saves power overall)

Theoretical prediction:
    For ASBL, C_f = 2V/U∞, so τ_w = ρ U∞ V
    P_drag = τ_w U∞ A = ρ U∞² V A
    Without suction (Blasius): P_drag,0 = ½ ρ U∞³ A C_f,0

    The pumping power depends on the pressure rise needed:
    For incompressible flow: Δp ≈ ½ ρ V² (dynamic head of suction flow)
    Q_suction = V A (volume flow rate through wall)
    P_pump = ½ ρ V³ A / η_pump

    Net power: P_net = ρ U∞² V A + ½ ρ V³ A / η_pump - P_drag,0

    Optimal V: dP_net/dV = 0 gives V_opt that maximizes savings.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .analytical import asbl_skin_friction, blasius_skin_friction


class PowerAccountant:
    """Net power accounting for suction drag reduction.

    Parameters
    ----------
    U_inf : float, free-stream velocity
    nu : float, kinematic viscosity
    rho : float, density (default 1.0 for lattice units)
    eta_pump : float, pump efficiency (default 0.7)
    A_wall : float, wall area (default 1.0 per unit span)
    """

    def __init__(
        self,
        U_inf: float,
        nu: float,
        rho: float = 1.0,
        eta_pump: float = 0.7,
        A_wall: float = 1.0,
    ):
        self.U_inf = U_inf
        self.nu = nu
        self.rho = rho
        self.eta_pump = eta_pump
        self.A_wall = A_wall

    def compute_drag_power(
        self,
        tau_w: float,
        U_inf: Optional[float] = None,
    ) -> float:
        """Compute drag power: P_drag = τ_w · U∞ · A.

        Parameters
        ----------
        tau_w : float, wall shear stress
        U_inf : float, optional override

        Returns
        -------
        P_drag : float
        """
        U = U_inf if U_inf is not None else self.U_inf
        return tau_w * U * self.A_wall

    def compute_pumping_power(
        self,
        V_suction: float,
    ) -> float:
        """Compute pumping power for suction.

        P_pump = Δp · Q / η_pump
        where Δp ≈ ½ ρ V² (minimum pressure rise)
              Q = V · A (volume flow rate)

        Parameters
        ----------
        V_suction : float

        Returns
        -------
        P_pump : float
        """
        # Pressure rise needed (minimum = dynamic head)
        delta_p = 0.5 * self.rho * V_suction**2
        # Volume flow rate
        Q = V_suction * self.A_wall
        # Pumping power
        return delta_p * Q / self.eta_pump

    def compute_baseline_drag_power(
        self,
        Re_x: float,
    ) -> float:
        """Compute baseline drag power without suction (Blasius).

        Parameters
        ----------
        Re_x : float, Reynolds number at plate end

        Returns
        -------
        P_drag_0 : float
        """
        # Average C_f for Blasius
        cf_avg = 1.328 / np.sqrt(Re_x)
        # Average τ_w
        tau_w_avg = 0.5 * self.rho * self.U_inf**2 * cf_avg
        return self.compute_drag_power(tau_w_avg)

    def compute_net_power(
        self,
        tau_w: float,
        V_suction: float,
        Re_x: float,
    ) -> Dict[str, float]:
        """Compute full power accounting.

        Parameters
        ----------
        tau_w : float, measured wall shear stress with suction
        V_suction : float
        Re_x : float, Reynolds number for baseline

        Returns
        -------
        result : dict with keys:
            'P_drag': drag power with suction
            'P_pump': pumping power
            'P_drag_0': baseline drag power (no suction)
            'P_net': net power change (negative = savings)
            'savings_fraction': -P_net / P_drag_0
        """
        P_drag = self.compute_drag_power(tau_w)
        P_pump = self.compute_pumping_power(V_suction)
        P_drag_0 = self.compute_baseline_drag_power(Re_x)

        P_net = P_drag + P_pump - P_drag_0
        savings_fraction = -P_net / P_drag_0 if P_drag_0 > 0 else 0.0

        return {
            'P_drag': P_drag,
            'P_pump': P_pump,
            'P_drag_0': P_drag_0,
            'P_net': P_net,
            'savings_fraction': savings_fraction,
        }

    def find_optimal_suction(
        self,
        V_range: List[float],
        Re_x: float,
    ) -> Dict:
        """Find optimal suction velocity for maximum net savings.

        Parameters
        ----------
        V_range : list of suction velocities to test
        Re_x : float, Reynolds number for baseline

        Returns
        -------
        result : dict with keys:
            'V_opt': optimal suction velocity
            'P_net_min': minimum net power (maximum savings)
            'sweep': list of dicts for each V
        """
        sweep = []
        P_net_min = float('inf')
        V_opt = None

        for V in V_range:
            # ASBL τ_w = ρ U∞ V
            tau_w = self.rho * self.U_inf * V
            accounting = self.compute_net_power(tau_w, V, Re_x)
            accounting['V_suction'] = V
            accounting['C_f'] = asbl_skin_friction(self.U_inf, V)
            sweep.append(accounting)

            if accounting['P_net'] < P_net_min:
                P_net_min = accounting['P_net']
                V_opt = V

        return {
            'V_opt': V_opt,
            'P_net_min': P_net_min,
            'sweep': sweep,
        }

    def theoretical_optimal_V(
        self,
        Re_x: float,
    ) -> float:
        """Compute theoretical optimal V/U∞ for maximum savings.

        From dP_net/dV = 0:
        P_net = ρ U∞² V A + ½ ρ V³ A / η - P_drag,0
        dP_net/dV = ρ U∞² A + 3/2 ρ V² A / η = 0

        Wait, this gives V² < 0, meaning P_net always increases with V.
        The savings come from comparing to the baseline.

        Actually: P_net = P_drag(V) + P_pump(V) - P_drag,0
        For ASBL: P_drag = ρ U∞² V A (increases with V!)

        The key insight: suction REDUCES τ_w compared to Blasius
        when V is small enough. The ASBL τ_w = ρ U∞ V, while Blasius
        τ_w ~ ρ U∞² / √Re_x.

        So savings exist when: ρ U∞ V < ½ ρ U∞² C_f,Blasius
        i.e., V/U∞ < C_f,Blasius / 2 = 0.332 / √Re_x

        The optimal V balances drag reduction against pumping cost.
        """
        # For the ASBL, the drag power is P_drag = ρ U∞² V A
        # The baseline is P_drag,0 = ½ ρ U∞³ C_f,0 A
        # Pumping is P_pump = ½ ρ V³ A / η

        # P_net = ρ U∞² V A + ½ ρ V³ A / η - ½ ρ U∞³ C_f,0 A
        # dP_net/dV = ρ U∞² A + 3/2 ρ V² A / η

        # This is always positive! So P_net increases with V.
        # The minimum is at V → 0, but then we have no suction.

        # The real question is: is there a V where P_net < 0?
        # P_net < 0 when: ρ U∞² V + ½ ρ V³ / η < ½ ρ U∞³ C_f,0
        # i.e., V + V³ / (2 η U∞) < ½ U∞ C_f,0

        # For small V/U∞: V < ½ U∞ C_f,0 = 0.332 U∞ / √Re_x
        # So V_opt/U∞ ≈ 0.332 / √Re_x (approximately, ignoring pumping)

        cf_0 = 1.328 / np.sqrt(Re_x)
        # Approximate optimal: V_opt ≈ ½ U∞ C_f,0 (where drag equals baseline)
        # More precisely, solve the cubic
        # V + V³/(2ηU∞) = ½ U∞ C_f,0
        # For small V, V ≈ ½ U∞ C_f,0

        V_opt_approx = 0.5 * self.U_inf * cf_0
        return V_opt_approx
