# p223w3-n02 — turbulent channel verification face (Re_tau = 180 / 395 / 590)

Machine-checkable acceptance harness for the native D3Q19 TRT channel kernel
(`src/tensorlbm/turbulence/les_chan.c`), campaign node **P223W3-N02**.

```
campaign.py         case matrix + per-case reduction (wall units, u_tau routes, Cf, spectrum)
analysis.py         pure reduction functions (log law, Reichardt, stress fit, spectra, ...)
gates.py            gates (a)-(f) -> case_data/gates.json
probe_stability.py  collision-branch / wall-normal-spacing stability envelope
make_figures.py     7 figures, read from case_data/*.json
make_report.py      docs/p223w3_n02_turbulent_channel_dns.md + audit manifest
case_data/            one JSON (+ plane slices) per case, and gates.json
figures/            generated figures (references in the report)
```

Run order (each step is idempotent):

```bash
python3 benchmarks/p223w3_n02_channel_dns/campaign.py --list
python3 benchmarks/p223w3_n02_channel_dns/campaign.py --case R180_dy4 --threads 20
python3 benchmarks/p223w3_n02_channel_dns/gates.py
python3 benchmarks/p223w3_n02_channel_dns/make_figures.py
python3 benchmarks/p223w3_n02_channel_dns/make_report.py
pytest -q tests/test_p223w3_n02_channel_dns.py     # 45 cases, < 1 min, tiny grids
```

Physics anchors used by the gates:

* `u_tau = sqrt(a h)` is exact for a force-driven channel, so the *imposed*
  friction velocity is not a fit; the independent `stress_fit` route is the check.
* DNS reference = MKM-1999 tables in `src/tensorlbm/data/mkm1999/`
  (upstream `chandata.tar.gz`, SHA-256 in `provenance.json`).
  **Do not** use `benchmarks/in99_n03_channel_turb/dns_ref/*.csv` — its
  `uu_plus` column rises to 16.6 at the centreline instead of peaking at 7.07
  (see the report, §5.3).

Two kernel defects were found and fixed here; both are described in
`CHANGELOG.md` and in the report §2.2 (`--reg` default divergence, and
`nan` written into the result JSON on divergence).