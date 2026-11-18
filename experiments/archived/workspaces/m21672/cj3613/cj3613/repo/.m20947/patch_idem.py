import pathlib
p = pathlib.Path('benchmarks/xval_job_checkpoint/run_xval.py')
s = p.read_text()
old = """    out = args.out
    out.mkdir(parents=True, exist_ok=True)
"""
new = """    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Idempotent repro (audit m18147 F2): the case directories under ``out/cases``
    # carry ``job_status.txt`` files, and a re-run in a populated directory used to
    # die on the residual ``done`` state.  Drop the harness-owned case dirs first so
    # ``repro.sh`` can be executed any number of times.
    if (out / "cases").exists():
        import shutil as _shutil

        _shutil.rmtree(out / "cases")
"""
assert old in s, "anchor not found"
p.write_text(s.replace(old, new, 1))
print("patched run_xval.py")