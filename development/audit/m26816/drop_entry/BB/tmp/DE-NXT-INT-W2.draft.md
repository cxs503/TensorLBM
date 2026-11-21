# TRACK INT WAVE 2 DRAFT

## ARM COX-VOINOV
- Task: Implement dynamic contact angle using Cox-Voinov law.
- Physical Source: Cox (2001), Voinov (1976).
- Prescription: Add dynamic contact angle calculation in boundary condition.
- Verification: Compare with analytical solution for capillary rise.

## ARM COMPRESS-INT
- Task: Add interface compression term to MCMP advection.
- Physical Source: Popinet (2009), Jaensson et al. (2020).
- Prescription: Modify advection step in MCMP solver.
- Verification: Compare interface thickness and spurious currents.
