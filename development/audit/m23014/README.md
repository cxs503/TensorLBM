# Historical m23014 — DFG 2D-2 hypothesis audit

The historical report and frozen outputs are preserved verbatim. The original assessment did not pass all preregistered gates. This is **not** a fresh solver run or acceptance. Its data inputs are from the confined DFG 2D-2 case, not the free-stream Braza cylinder or a different case-pool runner.

- [Current pending DFG case index](../../../benchmarks/pending/cylinder_re100_dfg/README.md)
- [Original DFG reference](../../../benchmarks/verified/cylinder/re100_dfg/REFERENCE.md)
- [Path and checksum ledger](../../../docs/architecture/migrations/dfg-and-cartesian-mpi-audits-2026-11-27.json)

Original mission records (old paths inside frozen records are historical):
- [DEVIATIONS.json](DEVIATIONS.json)
- [PREREG.json](PREREG.json)
- [REPORT.md](REPORT.md)
- [analyse_m23014.py](analyse_m23014.py)
- [analysis.json](analysis.json)
- [audit_h3_h4.py](audit_h3_h4.py)
- [h3_h4_audit.json](h3_h4_audit.json)
- [make_doc.py](make_doc.py)
- [publish_report.py](publish_report.py)

To reanalyse existing tracked traces in a disposable checkout, use the following current paths from the repository root. These tools overwrite the local copy of the historical outputs; do not overwrite the canonical artifacts.

```sh
python development/audit/m23014/audit_h3_h4.py
python development/audit/m23014/analyse_m23014.py
python development/audit/m23014/make_doc.py
```

The archived publish_report.py remains the original historical publication script, with its original paths. It is not an active entrypoint and was not executed. Published figure paths remain unchanged.
