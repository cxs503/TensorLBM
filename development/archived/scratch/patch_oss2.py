"""Pin the OSS ``lbm`` package explicitly: the upstream snapshot ships no
``lbm/__init__.py`` (namespace package), so a regular ``lbm`` package from any
other checkout on sys.path (editable install) wins and the oracle import fails
with ``No module named 'lbm.util'``.  Installing a synthetic package module
whose ``__path__`` is exactly the snapshot's ``lbm/`` makes resolution
deterministic."""
from pathlib import Path

p = Path("xval/oracle_minimal_d2q9_fs/oss_original.py")
s = p.read_text()
old = '''    # An editable install of *another* checkout's ``lbm`` package registers a
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
'''
new = '''    # The upstream snapshot has no ``lbm/__init__.py`` (namespace package), so
    # any *regular* ``lbm`` package from another checkout on sys.path (editable
    # install, or the repo's own ``src/lbm``) wins the lookup and the oracle
    # import dies with ``No module named 'lbm.util'``.  Pin ``lbm`` to the
    # snapshot by installing a synthetic package module with that __path__.
    pkg = types.ModuleType("lbm")
    pkg.__path__ = [str(Path(root) / "lbm")]
    pkg.__file__ = str(Path(root) / "lbm" / "__init__.py")
    sys.modules["lbm"] = pkg
    import lbm.util as U  # noqa: WPS433
    import lbm.init as INIT  # noqa: WPS433
    import lbm.streamcollide as SC  # noqa: WPS433
    from lbm.time_evolve import evolve  # noqa: WPS433
'''
assert old in s
p.write_text(s.replace(old, new))
print("patched", p)