# DE-NXT-MOM-W1 Draft

## Observation
Interface instability persists despite KERN-A patch. "Shallow death" or spurious currents observed.

## Hypotheses
1.  **Force Term Inconsistency:** Standard Guo force does not account for density gradients at the interface, leading to momentum imbalance.
2.  **Velocity Reconstruction Error:** Standard bounce-back rule does not conserve momentum correctly at the interface due to density jump.

## Arms
1.  **GUO_INT:** Density-aware Guo force term.
    *   *Physical Basis:* Guo force term needs density gradient correction at interface.
    *   *Minimal Change:* Modify `guo_force` function.
    *   *Preregistration:* Peak error < 3%, full window for cells=8.

2.  **BOUNCE_H:** Half-step bounce-back rule.
    *   *Physical Basis:* Half-step bounce-back improves momentum conservation at interface.
    *   *Minimal Change:* Modify `bounce_back` function.
    *   *Preregistration:* Peak error < 3%, full window for cells=8.

## Excluded Routes
Grid convergence, parameter tuning, u_limit, ABB removal, rho_g scaling, cavitation.
