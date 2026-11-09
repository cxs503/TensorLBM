"""Euler-Bernoulli beam FEM model for DEM-FSI coupling.

Provides a 1-D Euler-Bernoulli beam discretised with Hermite cubic
shape functions (C1-continuous), suitable for computing whipping
response from ice-floe impact loads.

The beam equation:
    ρA ∂²w/∂t² + c ∂w/∂t + EI ∂⁴w/∂t⁴ = f(x,t)

where w(x,t) is the transverse deflection, EI is the flexural rigidity,
ρA is the mass per unit length, and c is the damping coefficient.

References
----------
- Bathe, K.-J. (2006). *Finite Element Procedures*. 2nd ed.
- Hughes, T. J. R. (2000). *The Finite Element Method: Linear Static
  and Dynamic Finite Element Analysis*. Dover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class BeamProperties:
    """Euler-Bernoulli beam material and geometric properties.

    Attributes
    ----------
    youngs_modulus : float
        Young's modulus E in Pa.
    density : float
        Material density ρ in kg/m³.
    width : float
        Beam cross-section width b in m.
    height : float
        Beam cross-section height h in m.
    length : float
        Beam span L in m.
    damping_ratio : float
        Modal damping ratio ζ (dimensionless).
    boundary_condition : str
        "cantilever", "simply_supported", or "clamped_clamped".
    """

    youngs_modulus: float = 2.1e11  # Pa (steel)
    density: float = 7850.0  # kg/m³
    width: float = 0.1  # m
    height: float = 0.01  # m
    length: float = 1.0  # m
    damping_ratio: float = 0.02
    boundary_condition: str = "cantilever"

    @property
    def area(self) -> float:
        """Cross-sectional area A = b × h."""
        return self.width * self.height

    @property
    def moment_of_inertia(self) -> float:
        """Second moment of area I = b h³ / 12."""
        return self.width * self.height**3 / 12.0

    @property
    def flexural_rigidity(self) -> float:
        """EI in N·m²."""
        return self.youngs_modulus * self.moment_of_inertia

    @property
    def mass_per_length(self) -> float:
        """ρA in kg/m."""
        return self.density * self.area

    @property
    def first_natural_frequency(self) -> float:
        """First natural frequency in Hz (analytical for cantilever)."""
        if self.boundary_condition == "cantilever":
            beta_L = 1.875104  # first root of cos(βL)cosh(βL) = -1
        elif self.boundary_condition == "simply_supported":
            beta_L = math.pi
        elif self.boundary_condition == "clamped_clamped":
            beta_L = 4.730041
        else:
            beta_L = 1.875104

        omega_n = (beta_L / self.length) ** 2 * math.sqrt(
            self.flexural_rigidity / self.mass_per_length
        )
        return omega_n / (2.0 * math.pi)


class EulerBernoulliBeamFEM:
    """1-D Euler-Bernoulli beam FEM solver with Hermite cubic elements.

    Each node has 2 DOFs: transverse displacement w and rotation θ = dw/dx.
    Element stiffness and mass matrices are computed analytically.

    Parameters
    ----------
    props : BeamProperties
        Beam material and geometric properties.
    n_elements : int
        Number of beam elements (n_nodes = n_elements + 1).
    """

    def __init__(self, props: BeamProperties, n_elements: int = 20):
        self.props = props
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.n_dof = 2 * self.n_nodes  # 2 DOFs per node (w, θ)

        # Element length
        self.le = props.length / n_elements

        # Assemble global matrices
        self.K_global = self._assemble_stiffness()
        self.M_global = self._assemble_mass()

        # Apply boundary conditions
        self._apply_boundary_conditions()

        # State vectors
        self.displacement = torch.zeros(self.n_dof, dtype=torch.float64)
        self.velocity = torch.zeros(self.n_dof, dtype=torch.float64)
        self.acceleration = torch.zeros(self.n_dof, dtype=torch.float64)

        # Time history
        self.time = 0.0
        self.history: List[dict] = []

        # Modal properties (computed on demand)
        self._eigenvalues: Optional[torch.Tensor] = None
        self._eigenvectors: Optional[torch.Tensor] = None

    def _element_stiffness(self) -> torch.Tensor:
        """Element stiffness matrix for Euler-Bernoulli beam (4×4).

        DOF ordering: [w1, θ1, w2, θ2]
        """
        EI = self.props.flexural_rigidity
        L = self.le

        k = EI / L**3 * torch.tensor(
            [
                [12, 6 * L, -12, 6 * L],
                [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                [-12, -6 * L, 12, -6 * L],
                [6 * L, 2 * L**2, -6 * L, 4 * L**2],
            ],
            dtype=torch.float64,
        )
        return k

    def _element_mass(self) -> torch.Tensor:
        """Consistent element mass matrix for Euler-Bernoulli beam (4×4).

        DOF ordering: [w1, θ1, w2, θ2]
        """
        rhoA = self.props.mass_per_length
        L = self.le

        m = rhoA * L / 420 * torch.tensor(
            [
                [156, 22 * L, 54, -13 * L],
                [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                [54, 13 * L, 156, -22 * L],
                [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
            ],
            dtype=torch.float64,
        )
        return m

    def _assemble_stiffness(self) -> torch.Tensor:
        """Assemble global stiffness matrix."""
        K = torch.zeros((self.n_dof, self.n_dof), dtype=torch.float64)
        ke = self._element_stiffness()

        for e in range(self.n_elements):
            # Global DOF indices for this element
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += ke[i, j]

        return K

    def _assemble_mass(self) -> torch.Tensor:
        """Assemble global consistent mass matrix."""
        M = torch.zeros((self.n_dof, self.n_dof), dtype=torch.float64)
        me = self._element_mass()

        for e in range(self.n_elements):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            for i in range(4):
                for j in range(4):
                    M[dofs[i], dofs[j]] += me[i, j]

        return M

    def _apply_boundary_conditions(self):
        """Apply boundary conditions by zeroing constrained DOFs."""
        bc = self.props.boundary_condition

        if bc == "cantilever":
            # Fixed at x=0: w=0, θ=0 → DOFs 0, 1 constrained
            constrained = [0, 1]
        elif bc == "simply_supported":
            # w=0 at both ends: DOFs 0, 2*n_elements constrained
            constrained = [0, 2 * self.n_elements]
        elif bc == "clamped_clamped":
            # w=0, θ=0 at both ends
            constrained = [0, 1, 2 * self.n_elements, 2 * self.n_elements + 1]
        else:
            constrained = [0, 1]

        # Zero out rows and columns for constrained DOFs
        for dof in constrained:
            self.K_global[dof, :] = 0
            self.K_global[:, dof] = 0
            self.M_global[dof, :] = 0
            self.M_global[:, dof] = 0
            self.K_global[dof, dof] = 1.0  # Identity for numerical stability
            self.M_global[dof, dof] = 1.0

        self._constrained_dofs = constrained

    def compute_eigenmodes(self, n_modes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the first n_modes eigenvalues and eigenvectors.

        Solves the generalized eigenvalue problem: K φ = ω² M φ
        Uses M^{-1} K approach with torch.linalg.eig.

        Returns
        -------
        eigenvalues : torch.Tensor
            ω² values (rad/s)², shape (n_modes,).
        eigenvectors : torch.Tensor
            Mode shapes, shape (n_dof, n_modes).
        """
        # Use only free DOFs
        free_dofs = [i for i in range(self.n_dof) if i not in self._constrained_dofs]
        K_free = self.K_global[free_dofs][:, free_dofs]
        M_free = self.M_global[free_dofs][:, free_dofs]

        # Solve generalized eigenvalue problem: K φ = λ M φ
        # Convert to standard: M^{-1} K φ = λ φ
        M_inv_K = torch.linalg.solve(M_free, K_free)

        # Use eig (not eigh) since M^{-1}K is not symmetric in general
        eigenvalues, eigenvectors = torch.linalg.eig(M_inv_K)

        # Keep only real eigenvalues (physical modes)
        real_mask = eigenvalues.imag.abs() < 1e-8
        eigenvalues = eigenvalues.real[real_mask]
        eigenvectors = eigenvectors.real[:, real_mask]

        # Keep only positive eigenvalues
        pos_mask = eigenvalues > 0
        eigenvalues = eigenvalues[pos_mask]
        eigenvectors = eigenvectors[:, pos_mask]

        # Sort by eigenvalue
        idx = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx[:n_modes]]
        eigenvectors = eigenvectors[:, idx[:n_modes]]

        # Pad back to full size
        full_eigenvalues = torch.zeros(n_modes, dtype=torch.float64)
        full_eigenvectors = torch.zeros((self.n_dof, n_modes), dtype=torch.float64)

        n_avail = min(len(eigenvalues), n_modes)
        full_eigenvalues[:n_avail] = eigenvalues[:n_modes]
        for i in range(n_avail):
            full_eigenvectors[free_dofs, i] = eigenvectors[:, i]

        self._eigenvalues = full_eigenvalues
        self._eigenvectors = full_eigenvectors

        return full_eigenvalues, full_eigenvectors

    def apply_point_load(self, position_m: float, force_N: float) -> torch.Tensor:
        """Apply a point transverse load at a given position.

        Parameters
        ----------
        position_m : float
            Position along the beam in meters (0 ≤ x ≤ L).
        force_N : float
            Transverse force in Newtons.

        Returns
        -------
        force_vector : torch.Tensor
            Global force vector, shape (n_dof,).
        """
        if position_m < 0 or position_m > self.props.length:
            raise ValueError(
                f"Position {position_m} outside beam [0, {self.props.length}]"
            )

        # Find element containing the point
        elem_idx = int(position_m / self.le)
        elem_idx = min(elem_idx, self.n_elements - 1)

        # Local coordinate within element (0 to 1)
        xi = (position_m - elem_idx * self.le) / self.le

        # Hermite shape functions for transverse displacement
        N1 = 1 - 3 * xi**2 + 2 * xi**3
        N2 = self.le * (xi - 2 * xi**2 + xi**3)
        N3 = 3 * xi**2 - 2 * xi**3
        N4 = self.le * (-xi**2 + xi**3)

        # Global DOF indices
        dofs = [2 * elem_idx, 2 * elem_idx + 1, 2 * elem_idx + 2, 2 * elem_idx + 3]

        # Assemble force vector
        F = torch.zeros(self.n_dof, dtype=torch.float64)
        F[dofs[0]] = force_N * N1
        F[dofs[1]] = force_N * N2
        F[dofs[2]] = force_N * N3
        F[dofs[3]] = force_N * N4

        # Zero out constrained DOFs
        for dof in self._constrained_dofs:
            F[dof] = 0.0

        return F

    def apply_distributed_load(
        self, positions_m: torch.Tensor, forces_N: torch.Tensor
    ) -> torch.Tensor:
        """Apply multiple point loads (e.g., from DEM particle impacts).

        Parameters
        ----------
        positions_m : torch.Tensor
            Positions along the beam, shape (N,).
        forces_N : torch.Tensor
            Forces at each position, shape (N,).

        Returns
        -------
        force_vector : torch.Tensor
            Global force vector, shape (n_dof,).
        """
        F = torch.zeros(self.n_dof, dtype=torch.float64)
        for pos, force in zip(positions_m, forces_N):
            F += self.apply_point_load(pos.item(), force.item())
        return F

    def compute_stress(self) -> torch.Tensor:
        """Compute bending stress at each node.

        σ = M y / I, where M = -EI d²w/dx², y = h/2

        Returns
        -------
        stress : torch.Tensor
            Bending stress at each node, shape (n_nodes,).
        """
        stress = torch.zeros(self.n_nodes, dtype=torch.float64)
        c = self.props.height / 2.0  # distance from neutral axis

        for e in range(self.n_elements):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            w_e = self.displacement[dofs]

            # Curvature at element center (ξ = 0.5)
            # d²N/dξ² at ξ=0.5: [-6, -2L, 6, -2L] / L²
            d2N = torch.tensor(
                [-6, -2 * self.le, 6, -2 * self.le], dtype=torch.float64
            ) / (self.le**2)

            curvature = torch.dot(d2N, w_e)
            moment = -self.props.flexural_rigidity * curvature
            sigma = moment * c / self.props.moment_of_inertia

            stress[e] = sigma
            stress[e + 1] = sigma  # assign to both nodes

        return stress

    def compute_midpoint_stress(self) -> float:
        """Compute bending stress at beam midpoint."""
        mid_idx = self.n_elements // 2
        dofs = [2 * mid_idx, 2 * mid_idx + 1, 2 * mid_idx + 2, 2 * mid_idx + 3]
        w_e = self.displacement[dofs]

        # Curvature at element center
        d2N = torch.tensor(
            [-6, -2 * self.le, 6, -2 * self.le], dtype=torch.float64
        ) / (self.le**2)

        curvature = torch.dot(d2N, w_e)
        moment = -self.props.flexural_rigidity * curvature
        c = self.props.height / 2.0
        sigma = moment * c / self.props.moment_of_inertia

        return sigma.item()

    def step_newmark_correct(
        self,
        force_vector: torch.Tensor,
        dt: float,
        beta: float = 0.25,
        gamma: float = 0.5,
    ):
        """Advance one time step using Newmark-β method (corrected version).

        Parameters
        ----------
        force_vector : torch.Tensor
            External force vector, shape (n_dof,).
        dt : float
            Time step in seconds.
        beta : float
            Newmark β parameter (0.25 = constant average acceleration).
        gamma : float
            Newmark γ parameter (0.5 = trapezoidal rule).
        """
        # Store old state
        d_old = self.displacement.clone()
        v_old = self.velocity.clone()
        a_old = self.acceleration.clone()

        # Rayleigh damping
        zeta = self.props.damping_ratio
        if self._eigenvalues is not None and len(self._eigenvalues) > 0:
            omega1 = math.sqrt(max(self._eigenvalues[0].item(), 1e-10))
            omega2 = (
                math.sqrt(max(self._eigenvalues[-1].item(), 1e-10))
                if len(self._eigenvalues) > 1
                else omega1 * 2
            )
            alpha = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
            beta_rayleigh = 2 * zeta / (omega1 + omega2)
        else:
            alpha = 0.0
            beta_rayleigh = 0.0

        C = alpha * self.M_global + beta_rayleigh * self.K_global

        # Effective stiffness matrix
        K_eff = (
            self.K_global
            + gamma / (beta * dt) * C
            + 1.0 / (beta * dt**2) * self.M_global
        )

        # Effective force
        F_eff = (
            force_vector
            + self.M_global
            @ (
                d_old / (beta * dt**2)
                + v_old / (beta * dt)
                + a_old * (1.0 / (2 * beta) - 1.0)
            )
            + C
            @ (
                d_old * gamma / (beta * dt)
                + v_old * (gamma / beta - 1.0)
                + a_old * dt * (gamma / (2 * beta) - 1.0)
            )
        )

        # Zero out constrained DOFs
        for dof in self._constrained_dofs:
            K_eff[dof, :] = 0
            K_eff[:, dof] = 0
            K_eff[dof, dof] = 1.0
            F_eff[dof] = 0.0

        # Solve for new displacement
        self.displacement = torch.linalg.solve(K_eff, F_eff)

        # Update acceleration and velocity
        self.acceleration = (
            (self.displacement - d_old) / (beta * dt**2)
            - v_old / (beta * dt)
            - a_old * (1.0 / (2 * beta) - 1.0)
        )
        self.velocity = v_old + dt * ((1.0 - gamma) * a_old + gamma * self.acceleration)

        self.time += dt

        # Record history
        self.history.append(
            {
                "time": self.time,
                "midpoint_disp": self.displacement[2 * (self.n_elements // 2)].item(),
                "midpoint_stress": self.compute_midpoint_stress(),
            }
        )

    def compute_modal_generalized_force(
        self, force_vector: torch.Tensor, mode_idx: int = 0
    ) -> float:
        """Compute generalized force for a given mode.

        Q_n = φ_n^T F

        Parameters
        ----------
        force_vector : torch.Tensor
            Global force vector, shape (n_dof,).
        mode_idx : int
            Mode index (0-based).

        Returns
        -------
        generalized_force : float
            Modal generalized force.
        """
        if self._eigenvectors is None:
            self.compute_eigenmodes()

        phi_n = self._eigenvectors[:, mode_idx]
        return torch.dot(phi_n, force_vector).item()

    def compute_virtual_work(
        self, force_vector: torch.Tensor, displacement: torch.Tensor
    ) -> float:
        """Compute virtual work: δW = F^T δu.

        Parameters
        ----------
        force_vector : torch.Tensor
            Global force vector, shape (n_dof,).
        displacement : torch.Tensor
            Virtual displacement vector, shape (n_dof,).

        Returns
        -------
        virtual_work : float
            Virtual work value.
        """
        return torch.dot(force_vector, displacement).item()

    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy: KE = 0.5 v^T M v."""
        return 0.5 * torch.dot(self.velocity, self.M_global @ self.velocity).item()

    def compute_strain_energy(self) -> float:
        """Compute total strain energy: SE = 0.5 d^T K d."""
        return 0.5 * torch.dot(
            self.displacement, self.K_global @ self.displacement
        ).item()

    def reset(self):
        """Reset beam state to initial conditions."""
        self.displacement.zero_()
        self.velocity.zero_()
        self.acceleration.zero_()
        self.time = 0.0
        self.history.clear()
