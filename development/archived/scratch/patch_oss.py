"""Fix oss_original.import_oss so a foreign editable install of ``lbm`` cannot
shadow the OSS snapshot (main-CI red on test_port_matches_upstream_snapshot:
``ModuleNotFoundError: No module named 'lbm.util'`` because an editable-install
meta-path finder for another checkout's ``src/lbm`` wins over sys.path order)."""
from pathlib import Path

p = Path("xval/oracle_minimal_d2q9_fs/oss_original.py")
s = p.read_text()
old = '''    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    import lbm.util as U  # noqa: WPS433
    import lbm.init as INIT  # noqa: WPS433
    import lbm.streamcollide as SC  # noqa: WPS433
    from lbm.time_evolve import evolve  # noqa: WPS433

    return {"util": U, "init": INIT, "streamcollide": SC, "evolve": evolve}'''
new = '''    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    # An editable install of *another* checkout's ``lbm`` package registers a
    # meta-path finder that outranks sys.path, which made the oracle import
    # resolve to a foreign tree (or fail with ``No module named 'lbm.util'``).
    # Pin the resolution to plain sys.path lookup for the duration of the import.
    import importlib.machinery as _mach

    saved_meta = list(sys.meta_path)
    try:
        sys.meta_path[:] = [f for f in saved_meta if f is _mach.PathFinder] or saved_meta
        import lbm.util as U  # noqa: WPS433
        import lbm.init as INIT  # noqa: WPS433
        import lbm.streamcollide as SC  # noqa: WPS433
        from lbm.time_evolve import evolve  # noqa: WPS433
    finally:
        sys.meta_path[:] = saved_meta

    return {"util": U, "init": INIT, "streamcollide": SC, "evolve": evolve}'''
assert old in s
p.write_text(s.replace(old, new))
print("patched", p)