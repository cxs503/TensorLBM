# Free-energy two-layer Poiseuille flow

The m18190 experiment compared a standalone D2Q9 Swift–Osborn–Yeomans
implementation with an analytical two-layer channel profile. The
[model comparison](reference/DIFF_OSS_VS_TENSORLBM_PHASEFIELD_MCMP_POISEUILLE.md)
records the method lineage and limitations. The original audit compared
against the analytical solution and did not run the external implementation.

- [Archived velocity and phase profiles, configuration and convergence results](evidence/m18190/xval_poiseuille_results.json)
- [Four original figures as base64-encoded PNGs](evidence/m18190/figures_b64.json)
- [Historical independent audit](../../development/audit/m18190/AUDIT_M18190.md)

With the case's Python dependencies installed, run
`python benchmarks/phasefield_mcmp_poiseuille/reference/round1.py` from the checkout. New
results and figures go to `outputs/phasefield_mcmp_poiseuille/`. This
preserves the archived profiles and the original numerical implementation; the existing Round 2 loader now
resolves the relocated reference. The full Round 2 output payload is not tracked.

The four original PNGs are decoded directly from the tracked base64 archive, not redrawn:

- [fig1_velocity_profile.png](evidence/m18190/figures/fig1_velocity_profile.png)
- [fig2_psi_profile.png](evidence/m18190/figures/fig2_psi_profile.png)
- [fig3_convergence.png](evidence/m18190/figures/fig3_convergence.png)
- [fig4_sensitivity.png](evidence/m18190/figures/fig4_sensitivity.png)
